# `training/llm/llmfoundry` — YAML-driven training with LLM Foundry

YAML-driven continued pre-training and instruction finetuning with
[LLM Foundry](https://github.com/mosaicml/llm-foundry), the MosaicML/Databricks codebase that
trained MPT and DBRX. It sits on top of [Composer](https://github.com/mosaicml/composer) (see
`../composer/`) and adds a StreamingDataset pipeline that reads shards from object storage, a
finetuning dataloader with sequence packing, in-context-learning eval, and a callback that writes
Hugging Face checkpoints as you go. One YAML, launched with
`composer scripts/train/train.py <yaml>`.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, `flash_attention_2`) · AMD MI355X (gfx950, ROCm 7.2.4,
`sdpa`), 1 and 8 GPUs with FSDP `FULL_SHARD` over RCCL. YAML schemas move between minor releases —
run `--dry-run` first and diff against `scripts/train/yamls/finetune/` in your checkout before a
long job.

## Files

- `train_llm_llmfoundry.py` — launcher: picks a recipe YAML, applies per-run overrides, checks the
  env, execs `composer`.
- `yamls/finetune_chat_sft.yaml` — instruction finetuning from an HF checkpoint on local chat
  JSONL; wired to the shipped sample (`--recipe sft`).
- `yamls/finetune_chat_sft_8gpu.yaml` — the same recipe with an 8-rank batch geometry and `sdpa`.
- `yamls/continued_pretrain.yaml` — continued pre-training on MDS-converted text
  (`--recipe pretrain`).
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample so the SFT recipe is self-contained.
- `requirements_llmfoundry.txt` — install paths (Docker / venv / ROCm) and what needs a git source.

## Setup

Foundry must be used from a **git checkout** — the launcher, data-prep and conversion scripts live
in `scripts/` and are not in the wheel. Point the launcher at it with `--foundry-dir` (default
`./llm-foundry`). One venv per recipe folder.

```bash
cd training/llm/llmfoundry
ln -sf ../../../dev.env dev.env                 # HF_TOKEN
```

`dev.env` holds `HF_TOKEN=hf_...` and is git-ignored at the repo root; never commit a token. Both
shipped YAMLs set `use_auth_token: true`, which is what makes Foundry read `HF_TOKEN` for a gated
checkpoint such as Llama. `composer` passes the parent environment to every rank.

Two Python-3.12 packaging constraints apply to both vendors: install from the git checkout, not
the PyPI wheel (`llm-foundry` <= 0.22.0 drags in `pathtools`, which imports the removed `imp`
module), and add `setuptools<81` so `pkg_resources` still exists.

### NVIDIA (CUDA)

Upstream recommends their Docker image (`mosaicml/llm-foundry:2.7.0_cu128-latest`, plus an `_aws`
EFA variant), which ships the **dependencies only**, not the package — inside it, still
`git clone https://github.com/mosaicml/llm-foundry.git && cd llm-foundry && pip install -e ".[gpu]"`
(or `pip install -e .` with no NVIDIA GPU). The images are NVIDIA-only.

A venv recipe that resolves cleanly — dependency closure by name first, checkout `--no-deps`
after:

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry && source .env_llmfoundry/bin/activate
pip install -U pip 'setuptools<81' wheel
pip install torch numpy

# Foundry runtime closure by name. Drop mosaicml's [wandb] extra (it pulls an ancient
# wandb -> pathtools -> imp) and pin a modern wandb explicitly instead.
pip install \
  'mosaicml[libcloud,oci,gcs,mlflow]>=0.32.1,<0.33' 'wandb>=0.18,<0.19' \
  'mlflow>=2.14.1,<3.0' 'accelerate>=0.25,<1.9' 'transformers>=4.51.0,<4.52' \
  'mosaicml-streaming>=0.12.0,<0.13' 'torch>=2.7.0,<2.7.1' 'datasets>=3.3.2,<3.7' \
  fsspec==2023.6.0 sentencepiece==0.2.0 einops==0.8.1 'omegaconf>=2.2.3,<3' 'slack-sdk<4' \
  'mosaicml-cli>=0.6.10,<1' onnx==1.18.0 onnxruntime==1.22.0 'boto3>=1.21.45,<2' \
  'huggingface-hub[hf_xet]>=0.30.0,<0.34' 'beautifulsoup4>=4.12.2,<5' 'tenacity>=8.2.3,<10' \
  'catalogue>=2,<3' 'typer<1' GitPython==3.1.44 'python-dotenv>=1.0.1'

# The checkout supplies scripts/train/train.py (NOT in the wheel). --no-deps protects torch.
cd .env_llmfoundry && git clone --depth 1 https://github.com/mosaicml/llm-foundry.git
cd llm-foundry && pip install --no-deps -e . && cd ../..

python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# -> 2.7.0+cu126 12.6 True
```

**Expect torch to be downgraded to `2.7.0+cu126`, and leave it.** Foundry's closure carries
`torch>=2.7.0,<2.7.1`; the cu126 build runs on a CUDA 13 host and nothing in Foundry or Composer
supports a newer torch. Do not force a newer wheel back, and do not uninstall the `triton 3.3.0`
that ships with torch 2.7.0.

flash-attn works on H100 via the prebuilt wheel matching this torch, and `pip install -e ".[gpu]"`
pulls `flash-attn 2.7.4.post1` through the extra (use `--no-build-isolation`). Keep `PIP_CACHE_DIR`
off tmpfs — an `Invalid cross-device link` failure is that, not a build problem. To fetch the
wheel directly, `pip install --no-deps` this URL:
`https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`

### AMD (ROCm)

Do not follow upstream's "AMD (BETA support)" recipe. torch 2.7 has **no ROCm 7.x wheel** — the
ROCm 7.2 index publishes torch 2.11/2.12/2.13 — so Foundry's `torch>=2.7.0,<2.7.1` pin cannot be
satisfied and pip silently falls back to the CUDA wheel plus the whole `nvidia-*` runtime. Install
the dependency closure *without* the torch-family pins, then Composer and Foundry with
`--no-deps`.

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry
source .env_llmfoundry/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0       # pick your GPU; never set these to ""
export PIP_CACHE_DIR=/path/to/pip_cache                   # off tmpfs (cross-device link errors)
pip install -U pip setuptools wheel

# 1. ROCm torch FIRST - torchvision/torchaudio too, so nothing drags in a CUDA wheel later
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# 2. A constraints file makes any accidental torch swap a loud failure instead of a silent one
printf 'torch==2.11.0+rocm7.2\ntorchvision==0.26.0+rocm7.2\ntorchaudio==2.11.0+rocm7.2\n' \
  > /tmp/rocm_constraints.txt

# 3. Composer's dependencies MINUS torch/torchvision/torchaudio, then Composer itself
pip install -c /tmp/rocm_constraints.txt \
  apache-libcloud coolname databricks-sdk google-cloud-storage importlib-metadata \
  'mlflow>=2.14.1,<3.0' 'mosaicml-cli>=0.5.25,<0.8' 'numpy<2.3.0,>=1.21.5' oci \
  'packaging<25.1,>=21.3.0' 'pillow<12,>=10.3.0' psutil py-cpuinfo 'pynvml<12,>=11.5.0' \
  pyyaml requests tabulate==0.9.0 'torch_optimizer<0.4,>=0.3.0' \
  'torchmetrics<1.7.5,>=1.0' tqdm 'wandb<0.19,>=0.13.2'
pip install --no-deps mosaicml==0.32.1

# 4. Foundry's dependencies MINUS torch and MINUS mosaicml (installed above)
pip install -c /tmp/rocm_constraints.txt \
  'accelerate>=0.25,<1.9' 'transformers>=4.51.0,<4.52' 'mosaicml-streaming>=0.12.0,<0.13' \
  'datasets>=3.3.2,<3.7' fsspec==2023.6.0 sentencepiece==0.2.0 einops==0.8.1 \
  'omegaconf>=2.2.3,<3' 'slack-sdk<4' onnx==1.18.0 onnxruntime==1.22.0 'boto3>=1.21.45,<2' \
  'huggingface-hub[hf_xet]>=0.30.0,<0.34' 'beautifulsoup4>=4.12.2,<5' 'tenacity>=8.2.3,<10' \
  'catalogue>=2,<3' 'typer<1' GitPython==3.1.44 python-dotenv

# 5. The checkout (inside the git-ignored venv dir so `git status` stays clean)
cd .env_llmfoundry
git clone --depth 1 https://github.com/mosaicml/llm-foundry.git
cd llm-foundry && pip install --no-deps -e .     # --no-deps is what protects the ROCm torch
```

Verify with `python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"` ->
`2.11.0+rocm7.2`. `pip check` reports three cosmetic metadata complaints about the torch and
torchvision bounds; nothing asserts on them at runtime. If pip ever does replace torch:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

**Never `pip install flash-attn` on ROCm.** Override the YAML's `attn_implementation` with `sdpa`
instead. Harmless ROCm log noise: `ProcessGroupNCCL ... Guessing device ID based on global rank`
(RCCL under the NCCL name) and `destroy_process_group() was not called` at exit.

### Model compatibility (both vendors)

Foundry pins `transformers>=4.51.0,<4.52`, so any architecture added after 4.52 cannot load — e.g.
`LiquidAI/LFM2.5-350M` (`model_type: lfm2`) fails with "Transformers does not recognize this
architecture". Use a `qwen3`/`llama`-family checkpoint. Do not use a *reranker* checkpoint for
chat SFT either: its chat template renders a yes/no judge prompt, discards the user/assistant
turns, and the loader then drops every row.

## Data

The two recipes take different data because they are different jobs.

### Instruction finetuning (`--recipe sft`) — chat JSONL

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`yamls/finetune_chat_sft.yaml` points at the shipped 10-row sample, so the smoke run needs no data
prep. Foundry's `finetuning` dataloader accepts `messages` natively, rendering each turn with
`tokenizer.apply_chat_template`. It enforces (`llmfoundry/data/finetuning/tasks.py`): roles are
one of `user`, `assistant`, `system`, `tool`; the **last** message is `assistant`; the same role
never repeats back-to-back; each message has exactly the two keys `role` and `content`.

**A chat example must also have `messages` as its only top-level key.** The shipped sample carries
metadata columns (`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
`source_version`), so the YAML sets
`preprocessing_fn: llmfoundry.data.finetuning.tasks:messages_format_preprocessor` to extract just
`messages`. Keep that line for any JSONL with metadata columns; drop it if your rows are pure
`{"messages": [...]}`.

The alternative schema is `{"prompt": "...", "response": "..."}` (two keys, both strings). Foundry
picks whichever it sees; do not mix them in one file. Accepted extensions: `.jsonl`, `.json`,
`.csv`, `.parquet`. To use your own data, edit `variables.data_local` / `variables.train_file` or
pass `--data-local /path/to/dir --extra variables.train_file=my.jsonl`. The eval loader reuses the
train split because the 10-row sample has no held-out one — point it at a real validation file for
any serious run.

Which tokens generate loss is set by two dataset keys, not by the data:

| Setting | Values | Meaning |
|---|---|---|
| `target_prompts` | `none` (default), `all`, `length>=XX` | Whether user turns are training targets |
| `target_responses` | `last` (default), `all` | Every assistant turn, or only the final one |

### Continued pre-training (`--recipe pretrain`) — MDS shards

The `text` dataloader does not read JSONL; it reads MosaicML StreamingDataset (`.mds`) shards of
pre-concatenated, fixed-length token blocks. Convert first:

```bash
cd llm-foundry/scripts

# --concat_tokens packs to full-length blocks, no padding
python data_prep/convert_dataset_hf.py \
  --dataset allenai/c4 --data_subset en \
  --out_root ./my-mds-data --splits train val \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-3.1-8B \
  --eos_text '<|end_of_text|>'

# sanity-check the shards before training
python ../llmfoundry/data/text_data.py --local_path ./my-mds-data --split val
```

`--out_root` can be an object-store URI (`s3://...`, `gs://...`, `oci://...`). Set
`variables.data_remote` to that URI and `variables.data_local` to a fast local cache directory;
Foundry streams remote to local. If `data_remote` is blank, the shards must already be in
`data_local`.

Chat JSONL can also be converted to MDS once the instruction set is large or lives in object
storage. `--skip-preprocessing` is required when rows are already in `messages` (or
`prompt`/`response`) form. Then replace the SFT YAML's `hf_name`/`hf_kwargs` block with `remote:` /
`local:` / `split:` keys.

```bash
python data_prep/convert_finetuning_dataset.py \
  --dataset json --data_files /path/to/train.jsonl \
  --splits train \
  --skip-preprocessing \
  --out_root s3://my-bucket/my-sft-data
```

## Run

```bash
export OUTPUT_DIR=/path/to/outputs     # checkpoints and run artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export MASTER_PORT=29647               # a distinct port; never 29500 on a shared box
```

Dry-run first — it prints the exact `composer` command and validates that the launcher, the
checkout and the config all exist:

```bash
python3 train_llm_llmfoundry.py --recipe sft --dry-run
```

Instruction finetuning on real data:

```bash
nohup python3 train_llm_llmfoundry.py \
  --recipe sft \
  --foundry-dir ./llm-foundry \
  --gpus 8 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data-local /path/to/data \
  --extra variables.train_file=train.jsonl \
  --max-seq-len 4096 \
  --max-duration 3ep \
  --global-batch-size 64 \
  --lr 1e-5 \
  --save-folder ./llmfoundry_run/checkpoints \
  --run-name llama31-8b-sft \
  > train_llm_llmfoundry.log 2>&1 &

tail -f train_llm_llmfoundry.log
```

Continued pre-training is the same launcher with `--recipe pretrain --model
meta-llama/Llama-3.1-8B --data-local ./my-mds-data --data-remote s3://my-bucket/my-mds-data
--max-duration 2000ba --global-batch-size 512 --lr 5e-6`.

Anything without a flag goes through `--extra` as raw OmegaConf `key=value` overrides, which must
come last — e.g. `--extra train_loader.dataset.packing_ratio=auto
train_loader.dataset.target_responses=all eval_interval=100ba`. To skip the wrapper entirely,
`composer llm-foundry/scripts/train/train.py yamls/finetune_chat_sft.yaml
variables.data_local=/path/to/data max_duration=3ep`.

Foundry logs the fully resolved config (confirm your overrides landed), then per-batch blocks with
`LanguageCrossEntropy` / `Perplexity`, throughput from `speed_monitor` and an ETA from
`runtime_estimator`. With `device_train_microbatch_size: auto` it may catch an OOM, halve the
microbatch and carry on — that is intended. Read the loader's `Dropped N examples where the prompt
was longer than ...` warning; it is the only place Foundry reports how much data it actually kept.

### Single-GPU

A one-GPU run of the SFT recipe needs four overrides:

| Override | Why |
|---|---|
| `model.init_device=cpu` | Composer reverts FSDP->DDP on one GPU, and `init_device: mixed` then raises `NotImplementedError: ... only supported with FSDP`. Keep `mixed` for multi-GPU |
| `--global-batch-size 2` | The shipped `global_train_batch_size: 64` with `drop_last: true` yields **zero** batches from a 10-row sample |
| `save_folder=null` | `CheckpointSaver` writes at *fit end* regardless of `save_interval` — 6.7 GB for a 0.6B model |
| `model.attn_implementation=sdpa` (**ROCm only**) | `flash_attention_2` aborts at model build with `ValueError: use_flash_attention_2 is set to True, but flash-attention 2 is not installed.` |

```bash
export HF_DATASETS_CACHE=/dev/shm/dscache_llmfoundry     # datasets .arrow on tmpfs

CUDA_VISIBLE_DEVICES=0 python3 train_llm_llmfoundry.py --recipe sft \
  --foundry-dir .env_llmfoundry/llm-foundry \
  --gpus 1 --model Qwen/Qwen3-0.6B \
  --max-seq-len 2048 --max-duration 40ba \
  --global-batch-size 2 --device-microbatch-size 1 \
  --run-name sft-smoke \
  --extra model.attn_implementation=flash_attention_2 model.init_device=cpu \
          save_folder=null eval_interval=1000ba console_log_interval=2ba \
          callbacks.hf_checkpointer.save_folder=$OUTPUT_DIR/llmfoundry/hf_checkpoints
```

On AMD set `HIP_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES` and swap the attention override
to `sdpa`. Cross-entropy collapsing to ~0 on the 10-row sample is memorisation — the expected
smoke-test signal, not a learning result.

### Multi-GPU (FSDP)

The same recipe scales to 8 GPUs with real FSDP `FULL_SHARD` — no code patch, no new package, no
new pin. `yamls/finetune_chat_sft_8gpu.yaml` is the worked example.

```bash
cd training/llm/llmfoundry
source .env_llmfoundry/bin/activate
# If bin/activate carries a single-GPU device pin from an earlier session, overriding it
# AFTER sourcing is mandatory or you silently train on one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -c "import torch; assert torch.cuda.device_count()==8"

# The 10-row sample cannot feed 8 ranks at global batch 32 with drop_last. Replicate it:
mkdir -p outputs/llmfoundry_gpu8/data
for i in $(seq 128); do cat data/OTel_LLM_sample_10.jsonl; done \
  > outputs/llmfoundry_gpu8/data/OTel_LLM_sample_1280.jsonl

composer -n 8 --master_port 29750 \
  .env_llmfoundry/llm-foundry/scripts/train/train.py \
  yamls/finetune_chat_sft_8gpu.yaml
```

Pass `--master_port` on a shared box: Composer defaults to 29500 and a collision shows up as a
rendezvous hang, not a clear error. The launcher has no `--master_port` flag — use the
`MASTER_PORT` env var with `train_llm_llmfoundry.py --gpus 8 --config yamls/finetune_chat_sft_8gpu.yaml`,
which produces the same command line.

The 8-GPU YAML differs from the single-GPU one in `global_train_batch_size: 32` (must divide by
the rank count or Foundry aborts in config validation; 32 = 8 x 4 with
`device_train_microbatch_size: 1` giving grad accum 4), `init_device: mixed` restored (FSDP is
real at 8 ranks, and `mixed` is what stops 8 CPU copies of the weights at startup), the larger
data slice, and no `save_folder` or `hf_checkpointer`.

`data_parallel_shard` in the log proves sharding is real — with `NO_SHARD` or DDP, Composer
reports `data_parallel_replicate` instead.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--recipe` | `sft` | `sft` (instruction finetuning) or `pretrain` (continued pre-training) |
| `--config` | none | Explicit YAML path, bypassing `--recipe` |
| `--foundry-dir` | `./llm-foundry` | Path to the llm-foundry checkout (needs `scripts/train/train.py`) |
| `--gpus` | autodetect | Processes to launch; omit and `composer` autodetects |
| `--model` | YAML value | HF repo id or local path (sets `variables.model_name`) |
| `--data-local` | YAML value | Data dir (sft; YAML default `./data`) or MDS cache root (pretrain; `./my-mds-data`) |
| `--data-remote` | YAML value | MDS object-store URI (pretrain) |
| `--max-seq-len` | YAML value | Sets `variables.max_seq_len` |
| `--max-duration` | YAML value | Composer Time string: `3ep`, `2000ba`, `10000000tok` |
| `--global-batch-size` | YAML value | Sets `global_train_batch_size` |
| `--device-microbatch-size` | YAML value | int or `auto`; sets `device_train_microbatch_size` |
| `--lr` | YAML value | Sets `optimizer.lr` |
| `--save-folder` | YAML value | Composer checkpoint directory only — the `hf_checkpointer` callback has its own path |
| `--run-name` | YAML value | Sets `run_name` |
| `--extra` | none | Everything after this is forwarded verbatim as `key=value` overrides |
| `--dry-run` | off | Print the command and exit |

Key YAML fields:

| Field | Meaning |
|---|---|
| `model.name` | `hf_causal_lm` to load HF weights; `mpt_causal_lm` for the MPT architecture |
| `model.pretrained_model_name_or_path` | The checkpoint you are starting from |
| `model.attn_implementation` | `flash_attention_2` on CUDA; `sdpa` on ROCm |
| `model.init_device` | `mixed` = rank 0 loads real weights, others on meta, FSDP syncs |
| `train_loader.name` | `finetuning` (prompt/response or chat) vs `text` (streaming pre-training blocks) |
| `train_loader.dataset.preprocessing_fn` | Row transform; here the upstream `messages` extractor |
| `train_loader.dataset.packing_ratio` | `auto` profiles and packs sequences; big win on short rows |
| `target_prompts` / `target_responses` | Which spans generate loss |
| `global_train_batch_size` | The optimization math — fixed regardless of GPU count |
| `device_train_microbatch_size` | The execution — `auto` finds the largest that fits |
| `max_duration`, `eval_interval`, `save_interval` | Composer Time strings |
| `precision` | `amp_bf16` (`amp_fp8` only with TransformerEngine layers on H100) |
| `fsdp_config.*` | Sharding, `mixed_precision: PURE`, activation checkpointing, `state_dict_type` |
| `callbacks.hf_checkpointer` | Writes a Hugging Face folder during training |
| `save_num_checkpoints_to_keep` | Set it, or you will fill the disk |

## Output

```
llmfoundry_run/
  checkpoints/
    ep1-ba500-rank0.pt              # Composer checkpoints (model+optimizer+schedule)
    latest-rank0.pt
  hf_checkpoints/
    ba500/                          # from the hf_checkpointer callback
      config.json, model-*.safetensors, tokenizer files
```

The callback inserts an extra `huggingface/` level under the path you give it
(`<save_folder>/huggingface/ba<N>/`) and writes **at fit end regardless of `save_interval`**.
`save_folder=null` suppresses only the Composer `.pt` checkpoints, so a smoke run still lands a
servable HF folder unless you drop the callback too. Set `hf_checkpointer.precision: bfloat16`
unless you want fp32 files — the default is `float32`.

With `fsdp_config.state_dict_type: sharded` (the continued-pretrain recipe) each save is a
*directory* of `.metadata` plus one `.distcp` per rank, and `load_path` must point at the
directory, not a file. Sharded is what gives elastic resumption — save on 8 GPUs, resume on 16.

Without `hf_checkpointer`, convert a Composer checkpoint by hand and smoke-test it:

```bash
python llm-foundry/scripts/inference/convert_composer_to_hf.py \
  --composer_path llmfoundry_run/checkpoints/latest-rank0.pt \
  --hf_output_path ./final_model \
  --output_precision bf16
  # --hf_repo_for_upload user-org/repo-name   # needs a write-enabled HF_TOKEN

python llm-foundry/scripts/inference/hf_generate.py \
  --name_or_path ./final_model \
  --max_new_tokens 256 \
  --prompts "Summarise the following incident report:"
```

Resume by setting `load_path` in the YAML (or `--extra load_path=...`). Foundry **also
auto-resumes** when `run_name`, `save_folder` and `save_latest_filename` are all set and
`save_overwrite` is false — pass `save_overwrite=true` when you deliberately want a fresh run in
the same folder.

## Notes

- **Unknown YAML keys are a hard error.** `TrainConfig` defines every legal top-level key;
  anything custom must go under `variables:`.
- **`fsdp_config` stays at the top level of the YAML.** Foundry folds it into Composer's
  `parallelism_config` itself — do not "modernise" the YAML to `parallelism_config:`.
- **Override `variables.*`, not the leaf keys.** `variables.max_seq_len` feeds the top-level
  `max_seq_len`, the tokenizer's `model_max_length`, and both dataloaders.
- **`packing_ratio: auto` on an MPT model also needs `attn_uses_sequence_id: true`**, or attention
  bleeds across packed examples. HF models with FlashAttention-2 handle the boundaries themselves.
- **The two dataloaders are not interchangeable.** `finetuning` builds prompt/response or chat
  turns and applies loss masking; `text` yields fixed-length token blocks with no masking.
  Continued pre-training wants `text`.
- **Continued pre-training LRs are small** — `5e-6` with a 100-batch warmup in the shipped config
  versus `1e-5` for SFT. A from-scratch pre-training LR on a converged checkpoint destroys its
  instruction-following ability.
