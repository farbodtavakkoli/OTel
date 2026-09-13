# `training/llm/llmfoundry` — YAML-driven training with LLM Foundry

## 1. Overview & when to use

YAML-driven continued pre-training and instruction finetuning with
[LLM Foundry](https://github.com/mosaicml/llm-foundry), the MosaicML/Databricks codebase
that trained MPT and DBRX. Foundry sits on top of [Composer](https://github.com/mosaicml/composer)
(see `training/llm/composer` in this repo) and adds the pieces a pre-training run actually
needs: a StreamingDataset pipeline that reads shards straight from object storage, a
finetuning dataloader with sequence packing, in-context-learning eval, and a callback that
writes Hugging Face checkpoints as you go. You configure all of it in one YAML and launch
with `composer scripts/train/train.py <yaml>`.

Files in this folder:
- `train_llm_llmfoundry.py` — thin launcher: picks a recipe YAML, applies per-run overrides, checks the env, execs `composer`.
- `yamls/finetune_chat_sft.yaml` — instruction finetuning from an HF checkpoint on local chat JSONL; wired to the shipped sample.
- `yamls/continued_pretrain.yaml` — continued pre-training / domain adaptation on MDS-converted text.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample so the SFT recipe is self-contained.
- `requirements_llmfoundry.txt` — install paths (Docker / venv / ROCm), and what needs a git source.
- `readme_llmfoundry.md` — this file.

> **Hardware coverage.** The SFT recipe runs on **AMD MI355X (gfx950, ROCm 7.2)** at 1 and
> 8 GPUs (FSDP `FULL_SHARD` over RCCL) and on **NVIDIA H100 80GB (CUDA 13.0)** with
> `flash_attention_2` — see [Hardware support](#8-hardware-support). Both need the staged
> installs in section 2: Foundry's `torch>=2.7.0,<2.7.1` pin has no ROCm 7.x wheel at all,
> and on CUDA it downgrades whatever torch you had. The `pretrain` recipe and the
> MDS/StreamingDataset conversion path are **not covered**. YAML schemas move between minor
> releases — always run `--dry-run` first and diff against `scripts/train/yamls/finetune/`
> in your checkout before a long job.

## 2. Install

LLM Foundry is used from a **git checkout**, because the launcher, data-prep and conversion
scripts live in `scripts/` and are not part of the wheel. Point the launcher at the
checkout with `--foundry-dir` (default `./llm-foundry`).

### NVIDIA (CUDA)

Upstream strongly recommends their Docker image:

```bash
docker pull mosaicml/llm-foundry:2.7.0_cu128-latest
```

Note: the `llm-foundry` images ship the **dependencies only**, not the package.
Inside the container (or in a venv, if you are going bare):

```bash
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e ".[gpu]"        # or `pip install -e .` with no NVIDIA GPU
```

Without Docker, install `cmake packaging torch` first — `setup.py` requires them to be
present before it runs. See `requirements_llmfoundry.txt` for both paths plus the
`flash-attn --no-build-isolation` and TransformerEngine (FP8) notes.

Two Python-3.12 packaging requirements: install Foundry from the **git checkout**, not the PyPI
wheel (`llm-foundry` <= 0.22.0 drags in `pathtools`, which imports the removed `imp`
module), and add **`setuptools<81`** so `pkg_resources` still exists. A venv recipe that
resolves cleanly, installing the dependency closure by name first and the checkout
`--no-deps` after:

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry && source .env_llmfoundry/bin/activate
ln -sf ../../../dev.env dev.env                 # HF_TOKEN
pip install -U pip 'setuptools<81' wheel        # setuptools<81 => pkg_resources (Foundry imports it)
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

**Expect torch to be downgraded, and leave it.** The Foundry closure carries
`torch>=2.7.0,<2.7.1` and pulls a current build back to `2.7.0+cu126`. That is correct —
the cu126 build runs on a CUDA 13 host and nothing in Foundry or Composer supports a newer
torch. Do not force the newer wheel back.

**flash-attn** works on H100 via the prebuilt wheel that matches this torch. If
`pip install flash-attn` fails with `Invalid cross-device link`, that is `PIP_CACHE_DIR`
sitting on tmpfs — either move the cache off tmpfs, or fetch the wheel directly:

```bash
curl -sSL -o /tmp/fa.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps /tmp/fa.whl
```

`pip install -e ".[gpu]"` on the checkout resolves to the same torch 2.7.0 and pulls
flash-attn 2.7.4.post1 through the extra; use `--no-build-isolation` and keep
`PIP_CACHE_DIR` off tmpfs if you go that way. Do not uninstall the `triton 3.3.0` that
ships with torch 2.7.0.

**Model compatibility.** Foundry pins `transformers>=4.51.0,<4.52`, so any architecture
added to transformers after 4.52 cannot load — e.g. `LiquidAI/LFM2.5-350M` (`model_type:
lfm2`) fails with *"Transformers does not recognize this architecture"*. Use a
`qwen3`/`llama`-family checkpoint. Do not use a *reranker* checkpoint for chat SFT either:
its chat template renders a yes/no judge prompt, discards the user/assistant turns, and the
loader then drops every row.

### AMD / ROCm (beta upstream)

> **Do not follow upstream's "AMD (BETA support)" recipe — it does not work on a modern
> ROCm box.** Its step order runs `pip install -e .` before the ROCm torch, which
> re-resolves `torch>=2.7.0,<2.7.1` from PyPI and installs the **CUDA** build plus the whole
> `nvidia-*` runtime; its `rocm5.4.2` index no longer exists. More fundamentally, torch 2.7
> has **no ROCm 7.x wheel at all** — the ROCm 7.2 index publishes torch 2.11/2.12/2.13 — so
> the pin cannot be satisfied and pip silently falls back to the CUDA wheel. Install the
> dependency closure *without* the torch-family pins, then Composer and Foundry with
> `--no-deps`, as below.

#### Staged ROCm install

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry
source .env_llmfoundry/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0       # pick your GPU; never set these to ""
export PIP_CACHE_DIR=/path/to/pip_cache                   # off tmpfs (cross-device link errors)
pip install -U pip setuptools wheel

# 1. ROCm torch FIRST — torchvision/torchaudio too, so nothing drags in a CUDA wheel later
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

# 5. The checkout (kept inside the git-ignored venv dir so `git status` stays clean)
cd .env_llmfoundry
git clone --depth 1 https://github.com/mosaicml/llm-foundry.git
cd llm-foundry && pip install --no-deps -e .     # --no-deps is what protects the ROCm torch
```

`pip check` afterwards reports three metadata complaints (`llm-foundry`/`mosaicml` want
torch `<2.7.1`, `mosaicml` wants torchvision `<0.22.1`) — cosmetic, nothing at runtime
asserts on the torch version. Verify with
`python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"` →
`2.11.0+rocm7.2`. **Never `pip install flash-attn` here.** If pip ever does replace torch,
put it back with
`pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2`.

Harmless ROCm log noise: `ProcessGroupNCCL ... Guessing device ID based on global rank`
(RCCL under the NCCL name) and `destroy_process_group() was not called` at exit.

## 3. Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_llmfoundry.py` loads it with `load_dotenv("dev.env")` and the `composer`
launcher passes the parent environment to every rank. Both shipped YAMLs set
`use_auth_token: true` on the model block, which is what makes Foundry read `HF_TOKEN` when
pulling a gated checkpoint such as Llama. The same token is used by
`convert_composer_to_hf.py` if you pass `--hf_repo_for_upload`.

`dev.env` is git-ignored at the repo root. **Never commit a token**; rotate it on the Hub if
one ever lands in a commit.

## 4. Data

The two recipes take different data, because they are different jobs.

### 4a. Instruction finetuning (`--recipe sft`) — chat JSONL

Same format as the rest of this repo, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The folder ships a 10-row sample at `data/OTel_LLM_sample_10.jsonl` and
`yamls/finetune_chat_sft.yaml` points at it, so the smoke run works with no data prep.

Foundry's `finetuning` dataloader accepts the `messages` schema **natively** — it detects
the example type, renders each turn with `tokenizer.apply_chat_template`, and builds labels
from it. Rules it enforces (`llmfoundry/data/finetuning/tasks.py`):

- Roles must be one of `user`, `assistant`, `system`, `tool`.
- The **last** message must be `assistant`.
- The same role may not repeat back-to-back.
- Each message has exactly two keys: `role` and `content`.
- **A chat example must have `messages` as its only top-level key.** `_get_example_type`
  rejects rows with extra columns. The shipped sample carries metadata columns besides
  `messages` (`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
  `source_version`), so the YAML sets
  `preprocessing_fn: llmfoundry.data.finetuning.tasks:messages_format_preprocessor` —
  an upstream-provided function that extracts just the `messages` key. Keep that line for
  any JSONL with metadata columns; drop it if your rows are pure `{"messages": [...]}`.

The alternative schema, if you prefer flat pairs, is `{"prompt": "...", "response": "..."}`
(two keys, both strings). Foundry picks whichever it sees; do not mix them in one file.
Accepted file extensions: `.jsonl`, `.json`, `.csv`, `.parquet`.

To swap in your own data: put your JSONL somewhere, then either edit
`variables.data_local` / `variables.train_file` in the YAML or pass
`--data-local /path/to/dir --extra variables.train_file=my.jsonl`. The eval loader reuses
the train split as a documented smoke choice (the 10-row sample has no held-out split);
point it at a real validation file for any serious run.

Which tokens generate loss is controlled by two dataset keys, not by the data:

| Setting | Values | Meaning |
|---|---|---|
| `target_prompts` | `none` (default), `all`, `length>=XX` | Whether user turns are training targets |
| `target_responses` | `last` (default), `all` | Whether every assistant turn is a target, or only the final one |

For multi-turn conversations where you want to learn from every assistant reply, set
`target_responses: all`. The defaults (`none` / `last`) supervise only the final answer.

### 4b. Continued pre-training (`--recipe pretrain`) — MDS shards

The `text` dataloader does not read JSONL; it reads MosaicML StreamingDataset (`.mds`)
shards of pre-concatenated, fixed-length token blocks. Convert first:

```bash
cd llm-foundry/scripts

# From a HF dataset (C4 shown; --concat_tokens packs to full-length blocks, no padding)
python data_prep/convert_dataset_hf.py \
  --dataset allenai/c4 --data_subset en \
  --out_root ./my-mds-data --splits train val \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-3.1-8B \
  --eos_text '<|end_of_text|>'
```

`--out_root` can be an object-store URI (`s3://...`, `gs://...`, `oci://...`). Then set
`variables.data_remote` to that URI and `variables.data_local` to a fast local cache
directory; Foundry streams remote to local automatically. If `data_remote` is blank, the
shards must already exist in `data_local`.

Sanity-check the shards before training:

```bash
python ../llmfoundry/data/text_data.py --local_path ./my-mds-data --split val
```

### 4c. Optional: MDS for finetuning data too

Chat JSONL can also be converted to MDS, which is worth doing once your instruction set is
large or lives in object storage:

```bash
python data_prep/convert_finetuning_dataset.py \
  --dataset json --data_files /path/to/train.jsonl \
  --splits train \
  --skip-preprocessing \
  --out_root s3://my-bucket/my-sft-data
```

`--skip-preprocessing` is the flag to use when your rows are already in `messages` (or
`prompt`/`response`) form — without it the script demands a registered or explicit
`--preprocessor` and errors out. Chat rows are written to a `messages` column. Then swap the
`hf_name`/`hf_kwargs` block in the SFT YAML for:

```yaml
    remote: s3://my-bucket/my-sft-data
    local: /tmp/mds-cache/
    split: train
```

## 5. Run

Some commands below use two placeholders — set them once to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints and run artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Always dry-run first — it prints the exact `composer` command and validates that the
launcher, the checkout and the config all exist:

```bash
python3 train_llm_llmfoundry.py --recipe sft --dry-run
```

Smoke run against the shipped sample (the YAML already points at `./data`):

```bash
python3 train_llm_llmfoundry.py --recipe sft --gpus 8 \
  --max-duration 20ba --run-name sft-smoke
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

Continued pre-training:

```bash
nohup python3 train_llm_llmfoundry.py \
  --recipe pretrain \
  --model meta-llama/Llama-3.1-8B \
  --data-local ./my-mds-data \
  --data-remote s3://my-bucket/my-mds-data \
  --max-duration 2000ba \
  --global-batch-size 512 \
  --lr 5e-6 \
  > train_llm_llmfoundry.log 2>&1 &
```

Anything not exposed as a flag goes through `--extra` as raw `key=value` overrides (this is
OmegaConf dotted-path syntax, and it must come last):

```bash
python3 train_llm_llmfoundry.py --recipe sft --extra \
  train_loader.dataset.packing_ratio=auto \
  train_loader.dataset.target_responses=all \
  fsdp_config.activation_checkpointing=false \
  eval_interval=100ba
```

Equivalent raw command, if you would rather skip the wrapper:

```bash
composer llm-foundry/scripts/train/train.py \
  yamls/finetune_chat_sft.yaml \
  variables.data_local=/path/to/data \
  max_duration=3ep
```

**What "working" looks like:** the launcher echoes the `composer ...` line; Foundry logs the
fully resolved config (so you can confirm your overrides landed), builds the tokenizer,
loader and model, warns if FSDP was requested on a single GPU, and then prints per-batch
blocks like:

```
[batch=12/500]:
    Train LanguageCrossEntropy: 1.8421
    Train Perplexity: 6.3095
    Train loss/train/total: 1.8421
```

plus throughput from `speed_monitor` and an ETA from `runtime_estimator`. With
`device_train_microbatch_size: auto` you may see it catch an OOM, halve the microbatch and
carry on — that is intended behavior, not a failure. For SFT, loss should drop noticeably in the
first few hundred batches. For continued pre-training it moves much more slowly; watch for
*spikes*, which is why `kill_loss_spike` is enabled in that recipe.

Read the loader's drop warning — it is the only place Foundry tells you how much data it
actually kept:

```
tasks.py:1032: UserWarning: Dropped 1 examples where the prompt was longer than 2048,
the prompt or response was empty, or the response was all padding tokens
```

### Single-GPU overrides

A one-GPU run of the SFT recipe needs four overrides, none of them hardware-specific except
the attention backend:

| Override | Why |
|---|---|
| `model.init_device=cpu` | Composer reverts FSDP→DDP on one GPU (`FSDP is not applicable for single-GPU training`), and the YAML's `init_device: mixed` then raises `NotImplementedError: Using init_device 'mixed' is only supported with FSDP`. Keep `mixed` for real multi-GPU runs |
| `--global-batch-size 2` | The shipped `global_train_batch_size: 64` with `drop_last: true` yields **zero** batches from a 10-row sample |
| `save_folder=null` | Composer's `CheckpointSaver` writes a checkpoint at *fit end* regardless of `save_interval` — for a 0.6B model that is a 6.7 GB `.pt` (fp32 weights + Adam state) |
| `model.attn_implementation=sdpa` (**ROCm only**) | The YAML default `flash_attention_2` aborts at model build with `ValueError: use_flash_attention_2 is set to True, but flash-attention 2 is not installed.` FA2 has no ROCm build and must not be pip-installed there; `sdpa` is torch's own kernel and works on gfx950. On CUDA keep `flash_attention_2` |

```bash
export HF_DATASETS_CACHE=/dev/shm/dscache_llmfoundry     # datasets .arrow on tmpfs
export MASTER_PORT=29647                                 # a distinct port (never 29500)
set -a; . ./dev.env; set +a                              # HF_TOKEN

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

On AMD set `HIP_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES` and swap the attention
override to `sdpa`. **Expected output** — falling cross-entropy and a clean exit:

```
[batch=1/40]  Train metrics/train/LanguageCrossEntropy: 2.0920
[batch=20/40] Train metrics/train/LanguageCrossEntropy: 0.5302
[batch=40/40] Train metrics/train/LanguageCrossEntropy: 0.0039
[Eval batch=3/3] Eval metrics/eval/LanguageCrossEntropy: 0.0154 | LanguagePerplexity: 1.0155
llmfoundry.command_utils.train: Done.          # EXIT=0
```

Loss collapsing to ~0 on a 10-row sample is memorisation — the expected signal for a smoke
test, not a learning result.

### Multi-GPU (FSDP)

The same SFT recipe scales to 8 GPUs with real FSDP `FULL_SHARD` — no code patch, no new
package, no new pin. `yamls/finetune_chat_sft_8gpu.yaml` is the worked example; the 1-GPU
YAML is left with the NVIDIA defaults.

```bash
cd training/llm/llmfoundry
source .env_llmfoundry/bin/activate
# If bin/activate carries a single-GPU device pin from an earlier session, overriding it
# AFTER sourcing is mandatory or you silently train on one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -a; . ./dev.env; set +a            # HF_TOKEN
python3 -c "import torch; assert torch.cuda.device_count()==8"

composer -n 8 --master_port 29750 \
  .env_llmfoundry/llm-foundry/scripts/train/train.py \
  yamls/finetune_chat_sft_8gpu.yaml
```

`--master_port` matters on a shared box: Composer defaults to 29500 and a collision shows
up as a rendezvous hang, not a clear error. `train_llm_llmfoundry.py --gpus 8 --config
yamls/finetune_chat_sft_8gpu.yaml` produces the same command line (it has no
`--master_port` flag; use the `MASTER_PORT` env var with the launcher).

What changes versus the 1-GPU run:

| Change | Why |
|---|---|
| `global_train_batch_size: 32` (was 2) | Must divide by the 8 ranks or Foundry aborts in config validation. 32 = 8 x 4, with `device_train_microbatch_size: 1` giving grad accum 4 |
| `init_device: mixed` **restored** | `mixed` requires FSDP; at 8 ranks FSDP is real, so the single-GPU `cpu` workaround is no longer needed — and `mixed` is what stops 8 CPU copies of the weights at startup |
| A larger data slice, generated not committed | The 10-row sample cannot feed 8 ranks at global batch 32 with `drop_last: true` — zero batches. Write a replica (e.g. 10 rows x128) to `./outputs/llmfoundry_gpu8/data/OTel_LLM_sample_1280.jsonl`, which is what `variables.data_local` in the 8-GPU YAML points at |
| `save_folder` and the `hf_checkpointer` callback omitted | Nothing large should land for a smoke run |

`data_parallel_shard` is the line that proves sharding is real — with `NO_SHARD` or DDP
Composer reports `data_parallel_replicate` instead:

```
composer.core.state: Automatically setting data_parallel_shard to have parallelization degree 8.
n_gpus: 8   global_train_batch_size: 32   device_train_microbatch_size: 1   device_train_grad_accum: 4
config_utils.py:539: UserWarning: Setting `sync_module_states = True` for FSDP.
```

## 6. Arguments

Launcher flags (`python3 train_llm_llmfoundry.py --help`):

| Arg | Default | Meaning |
|---|---|---|
| `--recipe` | `sft` | `sft` (instruction finetuning) or `pretrain` (continued pre-training) |
| `--config` | none | Explicit YAML path, bypassing `--recipe` |
| `--foundry-dir` | `./llm-foundry` | Path to the llm-foundry checkout (needs `scripts/train/train.py`) |
| `--gpus` | autodetect | Processes to launch; omit and the `composer` launcher autodetects |
| `--model` | YAML value | HF repo id or local path (sets `variables.model_name`) |
| `--data-local` | YAML value | Data dir (sft; YAML default `./data`) or MDS cache root (pretrain; YAML default `./my-mds-data`) |
| `--data-remote` | YAML value | MDS object-store URI (pretrain) |
| `--max-seq-len` | YAML value | Sets `variables.max_seq_len` |
| `--max-duration` | YAML value | Composer Time string: `3ep`, `2000ba`, `10000000tok` |
| `--global-batch-size` | YAML value | Sets `global_train_batch_size` |
| `--device-microbatch-size` | YAML value | int or `auto`, sets `device_train_microbatch_size` |
| `--lr` | YAML value | Sets `optimizer.lr` |
| `--save-folder` | YAML value | Composer checkpoint directory |
| `--run-name` | YAML value | Sets `run_name` |
| `--extra` | none | Everything after this is forwarded verbatim as `key=value` overrides |
| `--dry-run` | off | Print the command and exit |

Note that `--save-folder` sets the *Composer* checkpoint directory only. The
`hf_checkpointer` callback has its own path; move it with
`--extra callbacks.hf_checkpointer.save_folder=...` if you relocate the run.

Key YAML fields:

| Field | Meaning |
|---|---|
| `model.name` | `hf_causal_lm` to load HF weights; `mpt_causal_lm` for the MPT architecture |
| `model.pretrained_model_name_or_path` | The checkpoint you are starting from |
| `model.attn_implementation` | `flash_attention_2` for HF models that support it |
| `model.init_device` | `mixed` = rank 0 loads real weights, others on meta, FSDP syncs. Avoids N CPU copies |
| `train_loader.name` | `finetuning` (prompt/response or chat) vs `text` (streaming pre-training blocks) |
| `train_loader.dataset.preprocessing_fn` | Optional row transform; here, the upstream `messages` extractor (see 4a) |
| `train_loader.dataset.packing_ratio` | `auto` profiles and packs sequences; big throughput win on short rows |
| `target_prompts` / `target_responses` | Which spans generate loss (see section 4a) |
| `global_train_batch_size` | The optimization math — fixed regardless of GPU count |
| `device_train_microbatch_size` | The execution — `auto` finds the largest that fits |
| `max_duration`, `eval_interval`, `save_interval` | Composer Time strings: `3ep`, `2000ba`, `10000000tok` |
| `precision` | `amp_bf16` (use `amp_fp8` only with TransformerEngine layers on H100) |
| `fsdp_config.*` | Sharding, `mixed_precision: PURE`, activation checkpointing, `state_dict_type` |
| `callbacks.hf_checkpointer` | Writes a Hugging Face folder during training |
| `save_num_checkpoints_to_keep` | Set it, or you will fill the disk |

## 7. Output

```
llmfoundry_run/
  checkpoints/
    ep1-ba500-rank0.pt              # Composer checkpoints (model+optimizer+schedule)
    latest-rank0.pt
  hf_checkpoints/
    ba500/                          # from the hf_checkpointer callback
      config.json, model-*.safetensors, tokenizer files
```

Note the extra `huggingface/` level the callback inserts under the path you give it
(`<save_folder>/huggingface/ba<N>/`), and that it writes **at fit end regardless of
`save_interval`**. `save_folder=null` suppresses only the *Composer* `.pt` checkpoints, not
this HF export — so a smoke run still lands a servable folder unless you drop the
`hf_checkpointer` callback too.

With `fsdp_config.state_dict_type: sharded` (the continued-pretrain recipe) each save is a
*directory* containing `.metadata` plus one `.distcp` per rank, and `load_path` must point
at that directory rather than a file. Sharded is what gives you elastic resumption — save
on 8 GPUs, resume on 16.

If you did not enable `hf_checkpointer`, convert a Composer checkpoint by hand:

```bash
python llm-foundry/scripts/inference/convert_composer_to_hf.py \
  --composer_path llmfoundry_run/checkpoints/latest-rank0.pt \
  --hf_output_path ./final_model \
  --output_precision bf16
  # --hf_repo_for_upload user-org/repo-name   # needs a write-enabled HF_TOKEN
```

Then the usual smoke test:

```bash
python llm-foundry/scripts/inference/hf_generate.py \
  --name_or_path ./final_model \
  --max_new_tokens 256 \
  --prompts "Summarise the following incident report:"
```

Resume a run by setting `load_path` in the YAML (or via `--extra load_path=...`). Foundry
also auto-resumes when `run_name`, `save_folder` and `save_latest_filename` are all set and
`save_overwrite` is false — convenient on a preemptible cluster, surprising if you did not
expect it, so pass `save_overwrite=true` when you deliberately want a fresh run in the same
folder.

## 8. Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 8 GPUs |
| Recipes | SFT (`--recipe sft`), `flash_attention_2` | SFT (`--recipe sft`), `sdpa`; 8-GPU FSDP `FULL_SHARD` over RCCL |
| torch | `2.7.0+cu126` — Foundry's pin resolves on PyPI and downgrades whatever you had | `2.11.0+rocm7.2` — the pin is unsatisfiable, install `--no-deps` (section 2) |
| Attention | flash-attn 2.7.4.post1 prebuilt wheel builds and runs | no ROCm flash-attn build; use `sdpa` |

Foundry's own code needs no patching on either vendor — the differences are entirely in how
you install it and in a handful of command-line overrides (section 5).

The Docker images (`mosaicml/llm-foundry:2.7.0_cu128-latest`, plus an `_aws` EFA variant)
are NVIDIA-only; on ROCm use the staged install in section 2.

### Not covered

- **Multi-node** FSDP, and FSDP checkpoint *saving* at 8 ranks.
- The `pretrain` recipe (`yamls/continued_pretrain.yaml`) and the MDS/StreamingDataset
  conversion scripts. That recipe carries the same `attn_implementation:
  flash_attention_2` default and needs the same `sdpa` override on ROCm.
- FP8 (`amp_fp8`) — needs TransformerEngine, which is NVIDIA-only.

## 9. Notes

- **Unknown YAML keys are a hard error.** `TrainConfig` defines every legal top-level key;
  anything custom must go under `variables:`. This is why a snippet copied from an old blog
  post fails.
- **`fsdp_config` stays at the top level of the YAML.** Foundry folds it into Composer's
  `parallelism_config` itself — do not "modernise" the YAML to `parallelism_config:`.
- **Override `variables.*`, not the leaf keys.** `variables.max_seq_len` feeds the top-level
  `max_seq_len`, the tokenizer's `model_max_length`, and both dataloaders; setting the
  leaves individually is how configs drift out of sync.
- **`init_device: mixed` requires FSDP.** Rank 0 materialises real weights, the other ranks
  init on meta and FSDP syncs outward. Without it every rank loads a full CPU copy at
  startup and an 8-GPU box can OOM before training begins. On one GPU use
  `init_device=cpu`.
- **`packing_ratio: auto` on an MPT model also needs `attn_uses_sequence_id: true`**, or
  attention bleeds across packed examples. HF models with FlashAttention-2 handle the
  boundaries themselves.
- **The two dataloaders are not interchangeable.** `finetuning` builds prompt/response or
  chat turns and applies loss masking; `text` yields undifferentiated fixed-length token
  blocks with no masking. Continued pre-training wants `text`.
- **Set `hf_checkpointer.precision: bfloat16`** unless you want fp32 files — the default is
  `float32`, which doubles the size on disk.
- **Continued pre-training LRs are small** — `5e-6` with a 100-batch warmup in the shipped
  config, versus `1e-5` for SFT. Reusing a from-scratch pre-training LR on a converged
  checkpoint destroys its instruction-following ability.
