# `training/llm/torchtitan` — PyTorch-native CPT / SFT on torchtitan

PyTorch-native large-scale training on [pytorch/torchtitan](https://github.com/pytorch/torchtitan):
FSDP2 per-parameter sharding, composable Tensor / Pipeline / Context Parallel, DCP
checkpointing that reads and writes Hugging Face safetensors directly, `torch.compile`, and
float8 / MXFP8 / NVFP4 training. torchtitan is a pre-training platform first, so the job here
is **continued pre-training** from an existing HF checkpoint; a chat dataloader gives you SFT
as well. Pick it over the other trainers in this repo when you want raw multi-dimensional
parallelism and plain PyTorch internals — there is no LoRA, DPO, or GRPO path here.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0). Same
code on both; float8/MXFP8/NVFP4 are NVIDIA-hardware features.

**Config is Python, not TOML.** Older torchtitan used `train_configs/*.toml` with
`--job.config_file`; on `main` that directory is gone and a run is a Python function returning
a `Trainer.Config`, selected with `--module <module> --config <function>`. The frozen
`--section.option` CLI flags still work and take precedence. Pin an older torchtitan release
if you need the TOML form.

## Files

- `train_llm_torchtitan.py` — launcher. Validates `--titan-repo`, builds the
  `torchrun ... -m torchtitan.train` (or checkpoint-conversion) command line, and execs it
  from your torchtitan clone.
- `recipe_torchtitan.py` — the run configuration: CPT and SFT recipes plus the local-dataset
  registration. Imported by torchtitan, so it has no CLI of its own; knobs are the environment
  variables below or fields you edit in the recipe functions.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample.
- `requirements_torchtitan.txt` — dependencies.

## Setup

Use a dedicated venv for this folder: torchtitan needs a **PyTorch nightly** on both vendors,
which conflicts with the pinned `torch` in this repo's other trainers. Stable wheels only work
with an older, matching torchtitan release.

torchtitan is a **source install** — the Python-config surface and the conversion scripts only
exist in the git tree. Do not also `pip install torchtitan` from PyPI into the same venv, and
do not `pip install -e .` the clone; the launcher runs with the clone as CWD. Pin commit
`03be241`: a fresh HEAD moved `torchtitan.components.checkpoint`, and
`recipe_torchtitan.py`'s `CheckpointManager` import then fails as
`Cannot import module 'recipe_torchtitan'`.

Do not install `flash-attn` — torchtitan uses SDPA / FlexAttention.

### NVIDIA (CUDA 13)

```bash
cd training/llm/torchtitan
python3.12 -m venv .env_torchtitan && source .env_torchtitan/bin/activate
export CUDA_VISIBLE_DEVICES=0        # no HIP_VISIBLE_DEVICES on NVIDIA

# 1. NIGHTLY torch first. If download.pytorch.org is proxy-blocked, unset the proxy here:
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. torchtitan from source, pinned.
git clone https://github.com/pytorch/torchtitan .env_torchtitan/torchtitan_src
git -C .env_torchtitan/torchtitan_src checkout 03be241
pip install -r .env_torchtitan/torchtitan_src/requirements.txt
pip install torchdata

# 3. This folder's extras.
pip install -r requirements_torchtitan.txt
```

### AMD (ROCm 7.2)

Identical, with the ROCm nightly index at step 1:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # never set either to ""
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

Stock upstream torchtitan works; AMD's [torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd)
fork and the `rocm/primus` images are alternatives, not requirements.

### Verify

Several later pip steps can swap torch. Re-check after each one:

```bash
python -c "import torch, torchtitan; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.device_count(), 'GPUs')"
# CUDA -> 2.15.0.dev20260822+cu130 13.0 None 1 GPUs
# ROCm -> 2.15.0.dev20260818+rocm7.2 None 7.2 2 GPUs
```

If `torch.version.hip` comes back `None` on ROCm (or non-`None` on CUDA), force-reinstall from
the correct nightly index. `--no-build-isolation` is needed only if you later build `torchao`
or another extension from source against the nightly.

Before an 8-GPU run, assert the mask really exposes 8 devices — a stale
`CUDA_VISIBLE_DEVICES=0,1` in `.env_torchtitan/bin/activate` silently caps you at 2:

```bash
python -c "import torch,sys; n=torch.cuda.device_count(); print(n); sys.exit(n!=8)"
```

### Environment and secrets

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded by train_llm_torchtitan.py
export OUTPUT_DIR=/path/to/outputs
export HF_HOME=/path/to/hf_cache
```

`HF_TOKEN` is forwarded to the child process and to `scripts/download_hf_assets.py`. Never
commit tokens.

The recipe reads its paths from environment variables, so you do not have to edit Python.
Relative values resolve against **this folder** (torchtitan runs from its own clone):

| Variable | Default | Meaning |
|---|---|---|
| `TITAN_HF_ASSETS` | `assets/hf/Qwen3-8B` | HF checkpoint dir: safetensors + index + `config.json` + tokenizer files |
| `TITAN_CPT_JSONL` | `data/OTel_LLM_sample_10.jsonl` | Continued-pretraining corpus |
| `TITAN_SFT_JSON` | `data/OTel_LLM_sample_10.jsonl` | Chat data for SFT |
| `TITAN_OUT` | `outputs` | Output folder |

```bash
export TITAN_HF_ASSETS=/data/hf/Qwen3-8B
export TITAN_CPT_JSONL=/data/domain_corpus.jsonl
export TITAN_SFT_JSON=/data/sft_chat.jsonl
export TITAN_OUT=$OUTPUT_DIR/torchtitan
```

Fetch the tokenizer / assets for a gated model:

```bash
python train_llm_torchtitan.py --mode download-assets \
  --titan-repo ../torchtitan --repo-id Qwen/Qwen3-8B --assets tokenizer
```

## Data

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

Both recipe paths take this file as-is; extra columns are ignored.

**Continued pre-training** prefers a plain `text` field (chat rows also work):

```json
{"text": "Full document body, no chat structure, no special tokens."}
```

torchtitan's pre-training dataloader reads from a dataset *registry*, not an arbitrary path,
so `recipe_torchtitan.py` registers your file as the named dataset `domain_corpus`. The file
is streamed, so it never has to fit in RAM; documents are concatenated and packed to `seq_len`.
The loader is `infinite=True`, so a small file re-loops — expect
`Dataset domain_corpus is being re-looped (epoch N)` and treat step count, not epochs, as the
run length.

**SFT** prefers the `messages` column; flat pairs work as a fallback:

```json
{"prompt": "How do I read an OTel span?", "response": "A span carries ..."}
```

**Starting weights** are a normal HF checkpoint directory (`*.safetensors`,
`model.safetensors.index.json`, `config.json`, tokenizer files). Point `TITAN_HF_ASSETS` at
it; torchtitan reads it directly via `checkpoint.initial_load_in_hf`, so no offline conversion
is required. Convert explicitly only if you want the DCP form up front, or want to go back:

```bash
# HF safetensors -> torchtitan DCP
python train_llm_torchtitan.py --mode convert-from-hf --titan-repo ../torchtitan \
  --input-dir /data/hf/Qwen3-8B --output-dir /data/dcp/Qwen3-8B \
  --model-name qwen3 --model-flavor 8B

# torchtitan DCP -> HF safetensors
python train_llm_torchtitan.py --mode convert-to-hf --titan-repo ../torchtitan \
  --input-dir /data/titan_outputs/checkpoint/step-2000 --output-dir /data/hf/Qwen3-8B-domain \
  --hf-assets-path /data/hf/Qwen3-8B --model-name qwen3 --model-flavor 8B
```

Use these scripts rather than renaming tensors by hand: for Llama 3 the adapter permutes
attention matrices to reconcile the HF and native RoPE layouts.

## Run

Run from inside `training/llm/torchtitan/` so `dev.env` and `recipe_torchtitan.py` resolve.
Add `--dry-run` to any command to print the resolved command and environment without executing.

```bash
# bring-up on a new box: ~32M random-init debugmodel, no weight download, no checkpoints
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 2

# 20-step smoke on the shipped sample, foreground
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_qwen3_8b_smoke --ngpu 8

# full continued-pretraining run (set TITAN_CPT_JSONL to your corpus first)
nohup python train_llm_torchtitan.py \
  --titan-repo .env_torchtitan/torchtitan_src \
  --module recipe_torchtitan \
  --config cpt_qwen3_8b \
  --ngpu 8 \
  > train_llm_torchtitan.log 2>&1 &

tail -f train_llm_torchtitan.log
```

SFT instead of CPT: `--config sft_qwen3_8b`. Llama 3.1 8B instead of Qwen3:
`--config cpt_llama3_8b`, with `TITAN_HF_ASSETS` pointed at the Llama checkpoint.

**Before a real CPT run, set `config.checkpoint.enable = True` in the recipe.** It is unset in
`_base_config()` and upstream's default is `False`, which makes the whole checkpoint block
(`initial_load_in_hf`, `initial_load_path`, `interval`, `last_save_in_hf`) inert: the run
trains from random init and saves nothing. Confirm the log does *not* say
`No checkpoint was provided, this is a fresh start`, and that loss does not start near
`ln(vocab_size)` (~12.7 for Qwen3).

### Reshaping the mesh

`--ngpu` must equal the product of the parallelism degrees —
`dp_shard * dp_replicate * tp * pp * cp` — torchtitan does not infer it. Override the degrees
with the frozen `--parallelism.*` flags instead of editing the recipe:

```bash
# composable 2-D: FSDP2(4) x TP(2)
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 4 \
          --parallelism.tensor_parallel_degree 2

# pure FSDP2 across all 8 (the right default at 8B: it fits, and TP adds per-layer collectives)
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 8 \
          --parallelism.tensor_parallel_degree 1
```

The second form is also the shipped default (`data_parallel_shard_degree=-1` = all leftover
ranks). Reach for `tp>1` only when a model genuinely will not fit (32B+ or long context), PP
only across nodes, and CP only when the sequence is the memory problem. On 80 GB H100 cards
the 8B configs require FSDP2 across at least 2 ranks.

The mesh banner must report the degrees you asked for:

```
[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=4, cp=1, tp=2, ep=1
[titan] Total parameter count: dense 8,190,735,360, sparse 0, vision 0, active 8,190,735,360
[titan] step:   1  loss: 12.77955  grad_norm: 14.1134  memory: 18.42GiB(6.40%)  tps: ...
[titan] step: 250  loss:  0.06754  grad_norm:  0.5762  memory: 27.39GiB(9.51%)  tps: ...
[titan] Training completed
```

Confirm every rank reached the end by checking that all N per-rank
`structured_logs/training.global_rank_{0..N-1}.*.jsonl` files were written, not just rank 0's.

## Arguments

### Launcher (`train_llm_torchtitan.py`)

| Argument | Default | What it does |
|---|---|---|
| `--titan-repo` | *(required)* | Path to your torchtitan clone; commands run with this as CWD |
| `--mode` | `train` | `train`, `convert-from-hf`, `convert-to-hf`, `download-assets` |
| `--module` | `recipe_torchtitan` | Module holding the config function; this folder is added to `PYTHONPATH` |
| `--config` | `cpt_qwen3_8b` | Function name inside `--module` |
| `--ngpu` | `8` | `torchrun --nproc_per_node`; must equal the product of the parallelism degrees |
| `--log-rank` | `0` | Ranks whose stdout is shown (`--local-ranks-filter`) |
| `--input-dir` / `--output-dir` | — | Source/destination for `convert-*` |
| `--model-name` / `--model-flavor` | `qwen3` / `8B` | Model package and registered flavor for `convert-*` |
| `--hf-assets-path` | — | Required by `convert-to-hf`: HF `config.json`/tokenizer for the target architecture |
| `--repo-id` / `--assets` | — / `tokenizer` | For `download-assets` |
| `--extra ...` | — | Everything after this is forwarded verbatim, e.g. `--extra --training.steps 200` |
| `--dry-run` | off | Print the command and exit |

### Recipe fields (`recipe_torchtitan.py`)

| Field | Value in the shipped recipe | Notes |
|---|---|---|
| `training.local_batch_size` | `1` | Per data-parallel rank, per gradient-accumulation step |
| `training.seq_len` | `4096` | Packed sequence length; must divide evenly by the context-parallel factor |
| `training.steps` | `2000` (CPT) / `500` (SFT) | Optimizer steps |
| `parallelism.data_parallel_shard_degree` | `-1` | `-1` = all leftover ranks, i.e. FSDP2 across every GPU |
| `parallelism.tensor_parallel_degree` | `1` | Raise for models that will not fit; keep TP inside one node |
| `parallelism.context_parallel_degree` | `1` | Raise for very long sequences |
| `parallelism.pipeline_parallel_degree` | `1` | Multi-node territory; leave at 1 on one node |
| `optimizer` | `default_adamw(lr=1e-5)` | CPT wants a much lower LR than fresh pre-training |
| `lr_scheduler` | warmup 100, cosine decay, `min_lr_factor=0.1` | |
| `activation_checkpoint` | `FullAC.Config()` | Swap to `SelectiveAC.Config()` for speed if you have memory headroom |
| `checkpoint.enable` | *(unset — upstream default `False`)* | Gate for the whole checkpoint block; set `True` for a real run |
| `checkpoint.initial_load_in_hf` | `True` | Cold-start from HF safetensors; needs `initial_load_model_only=True` and `checkpoint.enable=True` |
| `checkpoint.last_save_in_hf` | `True` | Write the final checkpoint back as HF safetensors |
| `checkpoint.interval` / `keep_latest_k` | `500` / `3` | Steps between DCP saves, and retention |
| `metrics.log_freq` | `10` | Steps between loss/throughput lines |

The frozen CLI flags take values (`--parallelism.tensor_parallel_degree 2`) except for
booleans, which are bare switches: use `--checkpoint.enable`, never
`--checkpoint.enable False` (that errors with `Unrecognized options: False`).

## Output

Everything lands under `TITAN_OUT` (default `./outputs`, `Trainer.Config.dump_folder`):

```
$TITAN_OUT/
  checkpoint/
    step-500/        DCP sharded checkpoint (model + optimizer + dataloader state)
    step-1000/
    step-2000/       final step; HF safetensors because last_save_in_hf=True
  tb/                TensorBoard event files
```

- Intermediate checkpoints are full DCP checkpoints, so a killed run resumes by relaunching the
  same command — the checkpointer finds the latest step and ignores the `initial_*` options.
- DCP writes one `.distcp` shard per rank (`__0_0.distcp`, `__1_0.distcp`, ...) — the
  checkpoint is sharded across the mesh, not gathered onto rank 0.
- `keep_latest_k=3` purges older steps in the background.
- The final step is model-only HF safetensors (`last_save_model_only=True`,
  `export_dtype="bfloat16"`), ready for any HF-based trainer or server in this repo. For a
  single `.pt`:
  `python -m torch.distributed.checkpoint.format_utils dcp_to_torch <step-dir> checkpoint.pt`.
- Checkpoints are large — a 5-step 8B run leaves a 31GB `checkpoint/step-10/`. Keep `TITAN_OUT`
  off a small volume and leave `checkpoint.enable` off for smoke tests (~20MB of logs and
  TensorBoard).

## Notes

- Re-check every config key against your checked-out torchtitan commit before a long run; the
  config surface moves fast.
- No `MASTER_ADDR`/`MASTER_PORT` is needed: the launcher uses `--rdzv_endpoint localhost:0`, so
  torchrun picks a free ephemeral port and co-tenant jobs cannot collide.
- `WARNING - CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.` on
  ROCm and `ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3` on both vendors
  are expected and harmless.
- FlexAttention triton-autotunes on first use, printing a wall of `triton_flex_attention_*`
  lines and adding 1-2 minutes to the first run; later runs reuse the cache.
- tf32 is deliberately off (torchtitan sets `allow_tf32 = False` on both matmul and cudnn) —
  the speed path is bf16 mixed precision. Do not "fix" it.
- RCCL/NCCL is picked up as the `nccl` backend with no configuration on either vendor.
- float8 via `Float8LinearConverter`, MXFP8/NVFP4, async checkpointing, and TorchFT exist
  upstream but are not wired into this recipe.
