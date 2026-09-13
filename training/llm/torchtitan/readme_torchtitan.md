# Setup & usage — `train_llm_torchtitan.py`

## Overview & when to use

PyTorch-native large-scale training on [pytorch/torchtitan](https://github.com/pytorch/torchtitan) —
FSDP2 with per-parameter sharding, composable Tensor / Pipeline / Context Parallel,
distributed (DCP) checkpointing that can read and write Hugging Face safetensors directly,
`torch.compile`, and float8 / MXFP8 / NVFP4 training. torchtitan is a **pre-training**
platform first, so the realistic job for it here is **continued pre-training / domain-adaptive
pretraining** from an existing HF checkpoint; it also ships a chat dataloader for **SFT**.
Pick it over the other trainers in this repo when you want raw multi-dimensional parallelism
and clean PyTorch internals rather than a Trainer/PEFT abstraction — there is no LoRA-first
path, no DPO, and no GRPO here.

Files in this folder:
- `train_llm_torchtitan.py` — thin launcher/validator. Builds the upstream
  `torchrun ... -m torchtitan.train --module ... --config ...` command line (or a
  checkpoint-conversion command) and execs it from your torchtitan clone.
- `recipe_torchtitan.py` — the run configuration. torchtitan configs are **Python
  functions returning `Trainer.Config`**, not TOML (see below). Holds the CPT and SFT
  recipes plus the local-dataset registration. **This file is a torchtitan config module
  imported by the framework — it deliberately has no `parse_args()`;** all knobs are either
  environment variables (below) or fields you edit in the recipe functions.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample (see Data below).
- `requirements_torchtitan.txt` — dependencies, with the source-install, PyTorch-nightly,
  and ROCm notes called out.
- `readme_torchtitan.md` — this file.

> **Hardware coverage.** The folder runs end to end on **AMD Instinct MI355X (gfx950,
> ROCm 7.2)** — FSDP2 across 2 GPUs, then the full 8-GPU node with both `dp_shard=8` and a
> 2-D `dp_shard=4 x tp=2` mesh — and on **NVIDIA H100 (CUDA 13.0)** at 1 and 2 GPUs, with a
> real 2-rank FSDP2 mesh and a sharded DCP checkpoint. See
> [§7.1 AMD MI355X](#71-amd-mi355x-rocm-72), [§7.2 8-GPU run](#8-gpu-run-8x-mi355x-rocm-724)
> and [§7.3 NVIDIA H100](#73-nvidia-h100-cuda-130).
>
> torchtitan's config surface changes fast — re-check every config key against your
> checked-out commit before a long run. One such drift is documented in §7.1:
> **`checkpoint.enable` defaults to `False`**, so the checkpoint block in
> `recipe_torchtitan.py` is inert until you set it.

### Important: torchtitan no longer uses TOML

Older torchtitan used `train_configs/*.toml` plus `--job.config_file`. On `main` that
directory is gone. A run is now described by a Python function returning a
`Trainer.Config`, selected with `--module <module> --config <function>`; upstream's
`torchtitan/config/README.md` states the `--section.option` CLI flags are **frozen** and
kept only for backward compatibility. `recipe_torchtitan.py` follows the current Python
form. If you pin an older torchtitan release you will need the TOML form instead.

## 1. Install

### NVIDIA (CUDA)

Use a dedicated venv. torchtitan `main` needs a **PyTorch nightly**, which will conflict
with the pinned `torch` in this repo's other trainers.

```bash
python3.12 -m venv ~/.venv-titan && source ~/.venv-titan/bin/activate

# 1. PyTorch nightly first (match the CUDA build to your driver).
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. torchtitan from source (the conversion scripts and the Python config tree
#    only exist in the git checkout).
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
cd ..

# 3. The extras this launcher and recipe need.
pip install -r requirements_torchtitan.txt
```

### AMD (ROCm)

torchtitan runs on ROCm — the upstream README's own install instructions say the nightly
CUDA index can be swapped for an AMD one (e.g. `rocm6.3`), and AMD builds on torchtitan
directly. Details in [§7.1](#71-amd-mi355x-rocm-72):

```bash
# PyTorch nightly for ROCm instead of CUDA (step 1 above); the rest is identical.
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

Use the **nightly**, not a stable ROCm wheel: torchtitan `main` needs FSDP2 APIs the stable
wheel does not have, and the failure is documented in §7.1. torchtitan is pure PyTorch
(FSDP2 / DTensor / DCP / SDPA), so no CUDA-only extension is required and the stock upstream
tree works — AMD's [torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd) fork and the
`rocm/primus` images are alternatives, not requirements. float8/MXFP8/NVFP4 are
NVIDIA-hardware features.

Sanity check (either vendor):

```bash
python -c "import torch, torchtitan; print('imports OK', torch.__version__, torch.cuda.device_count(), 'GPUs')"
```

`--no-build-isolation` is not needed for anything above. You only need it if you later
build `torchao` (for float8/MXFP8) or another CUDA extension from source against the
nightly torch, since build isolation would otherwise resolve a different torch.

## 2. Environment & secrets

Put a `dev.env` **in this folder**:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_torchtitan.py` loads it with `load_dotenv("dev.env")` and reads `HF_TOKEN` from
the environment; it is forwarded to the child process and to
`scripts/download_hf_assets.py`. `dev.env` is **git-ignored** at the repo root. **Never
commit tokens** — no token is hardcoded anywhere in this folder, and none should be added.

The recipe reads its paths from environment variables so you do not have to edit Python.
Relative values are resolved against **this folder** (torchtitan runs from its own clone):

| Variable | Default | Meaning |
|---|---|---|
| `TITAN_HF_ASSETS` | `assets/hf/Qwen3-8B` | HF checkpoint dir: safetensors + index + `config.json` + tokenizer files. |
| `TITAN_CPT_JSONL` | `data/OTel_LLM_sample_10.jsonl` | Continued-pretraining corpus (the shipped sample by default). |
| `TITAN_SFT_JSON` | `data/OTel_LLM_sample_10.jsonl` | Chat data for SFT (the shipped sample by default). |
| `TITAN_OUT` | `outputs` | Output folder. |

```bash
export TITAN_HF_ASSETS=/data/hf/Qwen3-8B
export TITAN_CPT_JSONL=/data/domain_corpus.jsonl
export TITAN_SFT_JSON=/data/sft_chat.jsonl
export TITAN_OUT=/data/titan_outputs
```

Fetch the tokenizer / HF assets for a gated model:

```bash
python train_llm_torchtitan.py --mode download-assets \
  --titan-repo ../torchtitan --repo-id Qwen/Qwen3-8B --assets tokenizer
```

## 3. Data

### The shipped sample

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

`source_spec_id` / `source_version` are null in most (not all) rows. Both recipe paths
accept this file as-is: the CPT path concatenates every `messages[].content` into one
document (`_text_from_sample`), the SFT path uses the `messages` list directly
(`_chat_pair`); extra columns are ignored in both.

### Continued pre-training

One JSON object per line. A plain `text` field is preferred; chat rows also work as above:

```json
{"text": "Full document body, no chat structure, no special tokens."}
```

torchtitan's pre-training dataloader reads from a **dataset registry**, not from an
arbitrary path, so `recipe_torchtitan.py` registers the file as a named dataset
(`domain_corpus`) with a loader and a sample processor. That is the pattern documented in
torchtitan's `docs/datasets.md`. The recipe streams the file, so it never has to fit in RAM.
Documents are concatenated and packed to `seq_len`.

### SFT

Prefers the `messages` column (so the shipped sample works as-is). Flat `prompt`/`response`
pairs still work as a fallback:

```json
{"prompt": "How do I read an OTel span?", "response": "A span carries ..."}
```

torchtitan's `ChatDataLoader` tokenizes with the model's chat template, masks the prompt
tokens so loss is computed on the assistant response only, and packs short samples into a
sequence.

### Starting weights

A normal Hugging Face checkpoint directory: `*.safetensors`,
`model.safetensors.index.json`, `config.json`, and the tokenizer files. Point
`TITAN_HF_ASSETS` at it. torchtitan reads that directly (`checkpoint.initial_load_in_hf`),
so **no offline conversion is required** for the common case.

Offline conversion is available if you want the DCP form up front (or want to go back to HF):

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

## 4. Run

Run from inside `training/llm/torchtitan/` (so `dev.env` and `recipe_torchtitan.py` resolve).
Start with the smoke recipe against the shipped sample, then the real one.

Some commands below use two placeholders — set them once to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints, TensorBoard, run logs
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

```bash
# 20-step smoke test on the shipped sample, foreground, fails fast if anything is wrong
python train_llm_torchtitan.py --titan-repo ../torchtitan \
  --config cpt_qwen3_8b_smoke --ngpu 8

# full continued-pretraining run (set TITAN_CPT_JSONL to your corpus first)
nohup python train_llm_torchtitan.py \
  --titan-repo ../torchtitan \
  --module recipe_torchtitan \
  --config cpt_qwen3_8b \
  --ngpu 8 \
  > train_llm_torchtitan.log 2>&1 &

tail -f train_llm_torchtitan.log
```

SFT instead of CPT: `--config sft_qwen3_8b`. Llama 3.1 8B instead of Qwen3:
`--config cpt_llama3_8b` (and point `TITAN_HF_ASSETS` at the Llama checkpoint).

Bring-up on a new box (or fewer GPUs): `--config cpt_debugmodel_smoke --ngpu <n>` runs the
same wiring on a ~32M-parameter random-init model with no weight download and no checkpoint
writes. That is the config used for the AMD MI355X verification in §7.1.

Add `--dry-run` to print the exact command and environment without executing.

## 5. Arguments

### Launcher arguments (`train_llm_torchtitan.py`)

| Argument | Default | What it does |
|---|---|---|
| `--titan-repo` | *(required)* | Path to your torchtitan clone. Commands run with this as CWD, which is what upstream's `run_train.sh` and CI do. |
| `--mode` | `train` | `train`, `convert-from-hf`, `convert-to-hf`, `download-assets`. |
| `--module` | `recipe_torchtitan` | Module holding the config function. This folder is added to `PYTHONPATH`. Any importable module works (e.g. `qwen3`, `torchtitan_recipes.llama3`). |
| `--config` | `cpt_qwen3_8b` | Function name inside `--module`. |
| `--ngpu` | `8` | `torchrun --nproc_per_node`. Must equal the product of the parallelism degrees in the recipe — torchtitan does not infer it. |
| `--log-rank` | `0` | Ranks whose stdout is shown (`--local-ranks-filter`). |
| `--input-dir` / `--output-dir` | — | Source/destination for `convert-*`. |
| `--model-name` / `--model-flavor` | `qwen3` / `8B` | Model package and registered flavor for `convert-*`. |
| `--hf-assets-path` | — | Required by `convert-to-hf`: the HF `config.json`/tokenizer describing the target architecture. |
| `--repo-id` / `--assets` | — / `tokenizer` | For `download-assets`. |
| `--extra ...` | — | Everything after this is forwarded verbatim, e.g. `--extra --training.steps 200`. |
| `--dry-run` | off | Print the command and exit. |

### Key recipe fields (`recipe_torchtitan.py`)

The recipe has no CLI of its own (it is imported by torchtitan, not executed); these are
the fields you edit, plus the environment variables in section 2.

| Field | Value in the shipped recipe | Notes |
|---|---|---|
| `training.local_batch_size` | `1` | Per data-parallel rank, per gradient-accumulation step. |
| `training.seq_len` | `4096` | Packed sequence length. Must divide evenly by the sequence/context-parallel factor. |
| `training.steps` | `2000` (CPT) / `500` (SFT) | Optimizer steps. |
| `parallelism.data_parallel_shard_degree` | `-1` | `-1` = all leftover ranks, i.e. FSDP2 across all 8 GPUs. |
| `parallelism.tensor_parallel_degree` | `1` | Raise for models that will not fit; keep TP inside one node. |
| `parallelism.context_parallel_degree` | `1` | Raise for very long sequences. |
| `parallelism.pipeline_parallel_degree` | `1` | Multi-node territory; leave at 1 on one node. |
| `optimizer` | `default_adamw(lr=1e-5)` | CPT wants a much lower LR than fresh pre-training. |
| `lr_scheduler` | warmup 100, cosine decay, `min_lr_factor=0.1` | Warmup-stable-decay scheduler. |
| `activation_checkpoint` | `FullAC.Config()` | Swap to `SelectiveAC.Config()` if you have memory headroom and want speed. |
| `checkpoint.enable` | *(unset — upstream default `False`)* | **Gate for the whole checkpoint block.** While it is false, every row below it does nothing: no HF cold start, no saves. Set it to `True` in the recipe for a real run. Found during the MI355X bring-up, §7.1. |
| `checkpoint.initial_load_in_hf` | `True` | Cold-start from HF safetensors. Requires `initial_load_model_only=True`, **and `checkpoint.enable=True`**. |
| `checkpoint.last_save_in_hf` | `True` | Write the final checkpoint back as HF safetensors. |
| `checkpoint.interval` / `keep_latest_k` | `500` / `3` | Steps between DCP saves, and retention. |
| `metrics.log_freq` | `10` | Steps between loss/throughput lines. |

**What "working" looks like:** the launcher prints the resolved `cwd`, `PYTHONPATH`, and the
full `torchrun` command. torchtitan then logs the model name and flavor, the total parameter
count, and a line confirming the checkpoint folder is active — with `initial_load_in_hf` you
should see it report loading HF safetensors from your assets path, not "No checkpoint was
provided, this is a fresh start" (that message means it is training from random init, which
is almost certainly not what you want for CPT). After that you get periodic step lines with
loss, GPU memory, tokens/sec, TFLOPs, and MFU. A healthy CPT run starts at a *low* loss
(the model already knows the language) and drifts down slowly; a loss that starts around
`ln(vocab_size)` means the HF weights were not loaded.

## 6. Output

Everything lands under `TITAN_OUT` (default `./outputs`, `Trainer.Config.dump_folder`):

```
$TITAN_OUT/
  checkpoint/
    step-500/        DCP sharded checkpoint (resumable: model + optimizer + dataloader state)
    step-1000/
    step-2000/       final step; written as HF safetensors because last_save_in_hf=True
  tb/                TensorBoard event files (metrics.enable_tensorboard)
```

- Intermediate checkpoints are **full** DCP checkpoints, so a killed run resumes by simply
  relaunching the same command — the checkpointer finds the latest step in the folder and
  ignores the `initial_*` options by design.
- `keep_latest_k=3` purges older steps in the background.
- The final step is model-only HF safetensors (`last_save_model_only=True`,
  `export_dtype="bfloat16"`), ready to hand to any HF-based trainer or server in this repo.
  If you need a single `.pt` instead, upstream supports
  `python -m torch.distributed.checkpoint.format_utils dcp_to_torch <step-dir> checkpoint.pt`.
- The launcher's own stdout/stderr go to `train_llm_torchtitan.log` (from the `nohup` line
  above); per-rank torchtitan logs are tee'd through `torchrun --tee 3` with only the ranks
  in `--log-rank` shown.

## 7. Hardware support

| Platform | Status | Detail |
|---|---|---|
| NVIDIA CUDA | First-class; covered here at 1 and 2 GPUs (§7.3) | MXFP8 targets Blackwell. |
| AMD ROCm | **Works at 2 and 8 GPUs** | Instinct MI355X (gfx950) / ROCm 7.2 at 2 GPUs (§7.1) and at 8, both pure FSDP2 and a composable FSDP2(4) x TP(2) mesh, with no code changes (§7.2). Stable ROCm wheels are **not new enough for torchtitan `main`** — use the nightly, see §7.1. |
| Intel XPU / Apple | No supported path documented | |

### 7.1 AMD MI355X (ROCm 7.2)

Unmodified upstream torchtitan, stock PyTorch ROCm nightly, no AMD fork, no source patches.
This folder adds one tiny bring-up config (`cpt_debugmodel_smoke` in `recipe_torchtitan.py`);
both it and the shipped `cpt_qwen3_8b_smoke` recipe train with finite, falling loss on 2 GPUs
with FSDP2.

**Host:** AMD Instinct MI355X (gfx950, 288GB HBM each), ROCm 7.2.4, Python 3.12, two GPUs
selected with `HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1`.

**Versions that work together** — `torch 2.15.0.dev20260818+rocm7.2` (nightly), `triton-rocm
3.8.0+gitdf3f91dd`, `torchdata 0.11.0`, `datasets 4.7.0`, `transformers 5.15.0`,
`tokenizers 0.22.2`, `safetensors 0.8.0`, `tyro 1.0.15`, `einops 0.8.2`,
`spmd_types 0.2.3`, `tensorboard 2.21.0`, torchtitan source at commit `03be241`.

#### Install

```bash
cd training/llm/torchtitan
python3.12 -m venv .env_torchtitan
source .env_torchtitan/bin/activate

# 1. PyTorch ROCm NIGHTLY (see "why the nightly" below — a stable ROCm wheel does not work).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2

# 2. torchtitan from source. Kept INSIDE the git-ignored venv dir so `git status`
#    in this repo stays clean; any path works, it is passed as --titan-repo.
git clone https://github.com/pytorch/torchtitan .env_torchtitan/torchtitan_src
pip install -r .env_torchtitan/torchtitan_src/requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu

# 3. This folder's extras.
pip install -r requirements_torchtitan.txt

# Sanity check (run from the clone, which is where the launcher runs commands):
cd .env_torchtitan/torchtitan_src
python -c "import torch, torchtitan.train; print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
# -> 2.15.0.dev20260818+rocm7.2 2 GPUs
```

No `pip install -e .` of torchtitan is needed — the launcher runs with the clone as CWD, so
the package imports from the tree (it then reports `torchtitan version: 0.0.0+unknown`,
which is harmless). **Do not install `flash-attn`**: torchtitan uses SDPA / FlexAttention,
and both work as shipped. Check `python -c "import torch; print(torch.version.hip)"` after
any later `pip install` and force-reinstall from the ROCm index if it comes back `None`.

#### Why the nightly, and not stable `torch 2.13.0+rocm7.2`

Against the same torchtitan commit, `torch 2.13.0+rocm7.2` **imports torchtitan fine but
cannot train**. The 2-rank run dies while FSDP2 wraps the model:

```
torchtitan/distributed/fsdp.py:188 in apply_fsdp_to_decoder -> torch ... fully_shard
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on the full
SPMD mesh (e.g. via distribute_module). Got plain tensor for parameter 'weight'.
```

torchtitan `main` passes a `dp_mesh_dims` argument whose DTensor contract only exists in the
newer FSDP2 in the nightly. The nightly requirement in this folder's docs is a
torchtitan-vs-torch API constraint, not an AMD one. Pin an older
torchtitan release if you must run a stable wheel.

#### Launch command

```bash
cd training/llm/torchtitan
source .env_torchtitan/bin/activate   # export HIP/CUDA_VISIBLE_DEVICES=0,1 if not already set
export TITAN_OUT=$OUTPUT_DIR/torchtitan

# A. tiny bring-up smoke: random-init qwen3 debugmodel, no weight download, no checkpoints
python train_llm_torchtitan.py \
  --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 2

# B. the folder's own shipped 8B smoke, cut to 5 steps (random init: with no HF weights
#    cached, TITAN_HF_ASSETS points at a tokenizer-only dir)
export TITAN_HF_ASSETS=/path/to/a/qwen3/tokenizer/dir
python train_llm_torchtitan.py \
  --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_qwen3_8b_smoke --ngpu 2 --extra --training.steps 5
```

`--ngpu 2` matches `dp_shard=2`; no `MASTER_ADDR`/`MASTER_PORT` is needed because the
launcher uses `--rdzv_endpoint localhost:0`, so torchrun picks a free port itself (the
usual 29500 collision with other jobs on a shared box cannot happen).

#### Expected output

A healthy run opens with the device-mesh and FSDP lines, then one `step:` line per logged
step with a falling `loss` and finite `grad_norm`:

```
[titan] INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=1, ep=1
[titan] INFO - CUDA capacity: AMD Instinct MI355X with 287.98GiB memory
[titan] INFO - Applied FSDP to the model
[titan] INFO - step:  1  loss:  7.58945  grad_norm:  9.3716  memory:  0.39GiB(0.13%)  tps: 68      tflops: 0.02  mfu: 0.00%
[titan] INFO - step:  8  loss:  6.48329  grad_norm:  3.6826  memory:  0.47GiB(0.16%)  tps: 24,468  tflops: 7.16  mfu: 0.29%
```

The shipped `cpt_qwen3_8b_smoke` recipe (8.19B parameters, seq 4096, FullAC, FSDP2 over
2 ranks) peaks at ~74 GiB per GPU. With no HF weights cached it starts from random init at
loss ≈ 12.7 (`ln(vocab)`) — see the `checkpoint.enable` quirk below.

RCCL is picked up as the `nccl` backend with zero configuration; the mesh line and the
falling loss on both ranks are the proof that collectives worked.

#### Quirks on ROCm

- **`checkpoint.enable` defaults to `False` upstream** (`torchtitan/components/checkpointer/base.py`),
  and `recipe_torchtitan.py` never sets it. Consequence on *any* vendor: the whole checkpoint
  block in the recipes — `initial_load_in_hf`, `initial_load_path`, `interval`,
  `last_save_in_hf` — is **inert**, so a "CPT" run silently trains from **random init** and
  saves nothing. Set `config.checkpoint.enable = True` in the recipe before a real run, and
  confirm the log does *not* say "No checkpoint was provided, this is a fresh start".
  A run starting at loss ≈ `ln(vocab)` is exactly the failure signature §5 warns about.
- **`--checkpoint.enable False` is not valid syntax.** The frozen tyro flags treat booleans
  as switches, so a value errors with `Unrecognized options: False`. Use
  `--checkpoint.enable` to turn it on, or edit the recipe.
- **`WARNING - CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.`**
  — expected and harmless on ROCm; torchtitan degrades to eager and keeps going.
- **FlexAttention triton-autotunes on first use** (the qwen3 debugmodel path), printing a
  wall of `triton_flex_attention_*` benchmarking lines and adding ~1-2 min to the first run.
  It compiles and runs correctly on gfx950; later runs reuse the cache.
- Do **not** set `CUDA_VISIBLE_DEVICES=""` on ROCm — it hides every GPU rather than none.
- torchtitan writes DCP checkpoints that are large by default (a 5-step 8B run leaves a 31GB
  `checkpoint/step-10/` when enabled). Keep `TITAN_OUT` off a small volume, and set
  `checkpoint.enable = False` for smoke tests — with it off a run writes ~20MB total
  (TensorBoard + structured JSONL logs).

#### Not covered here

Cold-starting from real HF safetensors (`initial_load_in_hf`), the `convert-from-hf` /
`convert-to-hf` paths, the SFT recipe (`sft_qwen3_8b`), multi-node, PP/CP degrees > 1,
`torch.compile`, and float8/MXFP8/NVFP4 (NVIDIA-hardware features). With no HF checkpoint
downloaded, both smoke runs above are random-init. TP degree > 1 *is* covered —
`tensor_parallel_degree=2` on 8 GPUs in §7.2 below.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This path works with no changes at all.** The 2-GPU recipe scales straight to all 8 MI355X
with **zero edits to `recipe_torchtitan.py`, `train_llm_torchtitan.py`, or
`requirements_torchtitan.txt`** — only `--ngpu 8` plus the frozen `--parallelism.*` CLI
overrides. Both layouts below — `dp_shard=4 × tp=2` and `dp_shard=8` — train with finite,
falling loss on all 8 ranks, on the qwen3 8B flavor at `seq_len=4096`, `local_batch_size=1`,
FullAC, bf16. Weights are random-init (`checkpoint.enable` is still `False`, see §7.1).

#### Which parallelism to use

**Use pure FSDP2 (`dp_shard=8`) at this model size.** An 8B model needs only ~27 GiB of a
288 GiB HBM stack, so TP buys memory headroom you do not need while adding per-layer
collectives on the activation path. Reach for `tp>1` only when a model genuinely will not fit
(32B+ / long context) — the same advice §8 gives.

#### Launch command

```bash
cd training/llm/torchtitan
source .env_torchtitan/bin/activate
# CRITICAL: if activate ends with a stale 2-GPU pin, override it AFTER
# sourcing, or you silently train on 2 GPUs and think you tested 8.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TITAN_OUT=$OUTPUT_DIR/torchtitan/gpu8
export TITAN_HF_ASSETS=$HF_HOME/hub/models--Qwen--Qwen3-0.6B/snapshots/<snapshot-hash>

# assert all 8 are really visible before burning an hour on a 2-GPU run
python -c "import torch,sys; n=torch.cuda.device_count(); print(n); sys.exit(n!=8)"

# A — composable 2-D: FSDP2(4) x TP(2)
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 4 \
          --parallelism.tensor_parallel_degree 2

# B — pure FSDP2 across all 8 (the recommended layout)
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 8 \
          --parallelism.tensor_parallel_degree 1
```

Stage B's degrees are also just the shipped defaults (`data_parallel_shard_degree=-1` → "all
leftover ranks"), so `--config cpt_qwen3_8b_smoke --ngpu 8` with no `--extra` is equivalent.

**Toolchain (unchanged from §7.1 — the nightly is load-bearing):**
`torch 2.15.0.dev20260818+rocm7.2` / `hip 7.2.53211`, torchtitan source at commit `03be241`,
Python 3.12, ROCm 7.2.4. Stable `torch 2.13.0+rocm7.2` still fails in `fully_shard` exactly as
§7.1 documents — that gap is unrelated to GPU count.

**No fixed master port.** `train_llm_torchtitan.py` hardcodes `--rdzv_endpoint localhost:0`, so
torchrun picks a free ephemeral port and `--master_port` is neither accepted nor needed; this is
what makes concurrent jobs on one box collision-proof.

#### What a healthy 8-GPU run looks like

The mesh banner must show the degrees you asked for:

```
[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=4, cp=1, tp=2, ep=1
[titan] Successfully created meshes with active dimensions: ['batch', 'loss', 'tp', 'dp', 'dp_shard']
[titan] CUDA capacity: AMD Instinct MI355X with 287.98GiB memory
[titan] Total parameter count: dense 8,190,735,360, sparse 0, vision 0, active 8,190,735,360
step:   1  loss: 12.77955  grad_norm: 14.1134  memory: 18.42GiB(6.40%)  ...
step: 250  loss:  0.06754  grad_norm:  0.5762  memory: 27.39GiB(9.51%)  ...
[titan] Training completed
```

Check that **all 8** per-rank structured logs
(`structured_logs/training.global_rank_{0..7}.*.jsonl`) were written — that is the independent
confirmation that every rank reached the end, rather than rank 0 finishing alone.

#### Differences from the 2-GPU setup

- **Nothing in the code, the deps, or the environment.** No new package, no pin change, no
  RCCL/NCCL tuning variables, no batch-size reduction. RCCL is picked up as the `nccl`
  backend with zero configuration across all 8 ranks exactly as it was across 2.
- **A stale `CUDA_VISIBLE_DEVICES=0,1` left in `.env_torchtitan/bin/activate`** is the
  one thing to check. Sourcing the venv silently caps you at 2 GPUs; the `device_count()==8`
  assertion above is what catches it. Consider deleting those two lines from `activate`.
- **`--parallelism.*` frozen CLI flags work** and are the clean way to re-shape the mesh without
  editing the recipe — but note they take *values* (`--parallelism.tensor_parallel_degree 2`),
  unlike the boolean switches §7.1 warns about.
- **`torch.compile` is not enabled** and is not needed; this recipe never turns it on. The
  `CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.` warning from
  §7.1 still appears on all 8 ranks and is still harmless. Async-TP (which needs
  `torch.compile`) is not covered; only plain TP.
- **Benign warning:** `ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based
  on job config` — torchtitan setting its own collective error handling.
- `--ngpu` must still equal the product of the degrees — torchtitan does not infer it.

### 7.3 NVIDIA H100 (CUDA 13.0)

Unmodified upstream torchtitan (same commit `03be241` as the AMD route), stock PyTorch CUDA
nightly, no source patches. The `cpt_debugmodel_smoke` bring-up config from §7.1 runs end to
end on one H100 with finite, monotonically falling loss and a saved DCP checkpoint; the 2-GPU
subsection below adds a real 2-rank FSDP2 mesh and a sharded checkpoint. The AMD §7.1/§7.2
material covers the 8-GPU FSDP2/TP behaviour; nothing in it is device-specific.

**Host:** NVIDIA H100 80GB HBM3, **CUDA 13.0**, Hopper cc(9,0), Python 3.12.

**Versions that work together** — `torch 2.15.0.dev20260822+cu130` (nightly;
`torch.version.cuda==13.0`, `torch.version.hip is None`), `triton 3.8.0`, `torchdata 0.11.0`,
`datasets 4.7.0`, `transformers 5.15.1`, `tokenizers 0.22.2`, `safetensors 0.8.0`,
`tyro 1.0.16`, `grain 0.2.18`, torchtitan source at commit **`03be241`** (same as the AMD
route).

#### The nightly is mandatory on CUDA too

`pip install torch numpy` installs **stable `the current CUDA 13 build`** (real bf16 matmul). Against torchtitan `03be241` that stable wheel gets **further than the AMD
stable wheel did** — it builds the device mesh, applies FSDP2, prepares the dataset, and even
runs **step 1** (`loss: 7.72586`) — but then dies at the end of the first step:

```
File ".../torchtitan/trainer.py", line 1062, in train -> dist_utils.set_pg_timeouts(...)
File ".../torchtitan/distributed/utils.py", line 595, in set_pg_timeouts
    torch.distributed.set_timeout(timeout, group)
AttributeError: module 'torch.distributed' has no attribute 'set_timeout'
```

`torch.distributed.set_timeout` only exists in the newer torch in the nightly. So the "nightly
required" rule in §1 is a **torchtitan-main-vs-torch API gap on CUDA as well** (a *different*
missing symbol than the AMD `fully_shard`/`dp_mesh_dims` one, but the same conclusion). Fix: the
folder's documented NVIDIA path.

#### Install

```bash
cd training/llm/torchtitan
python3.12 -m venv .env_torchtitan            # a tmpfs-backed symlink also works
source .env_torchtitan/bin/activate

# 1. PyTorch CUDA NIGHTLY (stable 2.13.0+cu130 fails at set_pg_timeouts, see above).
#    if download.pytorch.org is proxy-blocked, unset the proxy for THIS install:
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
#    -> torch 2.15.0.dev20260822+cu130

# 2. torchtitan from source, pinned to a known-good commit (a fresh HEAD has drift, below).
git clone https://github.com/pytorch/torchtitan .env_torchtitan/torchtitan_src
git -C .env_torchtitan/torchtitan_src checkout 03be241
pip install -r .env_torchtitan/torchtitan_src/requirements.txt   # torch is NOT pinned here -> safe
pip install torchdata                                            # 0.11.0 from PyPI is fine

# 3. This folder's extras.
pip install -r requirements_torchtitan.txt

# Re-verify torch was not clobbered (several deps can):
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
# -> 2.15.0.dev20260822+cu130 13.0 None
```

`grain`, `datasets`, `torchtitan/requirements.txt`, and the folder extras all install from
**pypi.org** — behind a proxy allowlist, only the torch nightly needs the proxy off. Re-check
`torch.version.hip is None` after each subsequent `pip install`. No `flash-attn`, and no
`pip install -e .` of torchtitan — the launcher runs with the clone as CWD.

**Commit drift:** a fresh `git clone` (e.g. HEAD `304fd88`) **breaks the shipped
recipe** — `recipe_torchtitan.py`'s `from torchtitan.components.checkpoint import CheckpointManager`
raises `ModuleNotFoundError` (that module moved), and torchtitan's `config/manager.py`
swallows the ImportError and reports "Cannot import module 'recipe_torchtitan'". Pinning
`03be241` makes the recipe import cleanly. This is the "re-check every config key
against your checked-out commit" warning at the top of this file.

#### Launch command

```bash
cd training/llm/torchtitan
source .env_torchtitan/bin/activate
export CUDA_VISIBLE_DEVICES=0                       # the assigned free GPU(s)
export HF_DATASETS_CACHE=/dev/shm/dscache_torchtitan
export TITAN_OUT=$OUTPUT_DIR/torchtitan

# A. bring-up smoke: random-init qwen3 debugmodel (~32M), no weight download, 30 steps
python train_llm_torchtitan.py \
  --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 1 \
  --extra --training.steps 30

# B. longer hold (400 steps, seq 2048, local batch 16) to sample sustained GPU utilisation
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 1 \
  --extra --training.steps 400 --training.seq_len 2048 --training.local_batch_size 16

# C. checkpoint-save proof: flip the (default-off) checkpoint block ON via the frozen switch
python train_llm_torchtitan.py --titan-repo .env_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 1 \
  --extra --training.steps 10 --checkpoint.enable --checkpoint.interval 5
```

`--ngpu 1` matches `dp_shard=1` (torchtitan builds a 1-rank FSDP2 mesh — a degenerate but valid
exercise of the wrap/shard/checkpoint path). No `MASTER_PORT` needed: the launcher uses
`--rdzv_endpoint localhost:0`, so torchrun picks a free ephemeral port (collision-proof with any
co-tenant job).

#### Expected output

Run A — `cpt_debugmodel_smoke`, 30 steps, seq 512, local batch 2, loss falling the whole way:

```
[titan] CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] Total parameter count: dense 31,987,968, sparse 0, vision 0, active 31,987,968
[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1
[titan] Applied FSDP to the model
[titan] step:  1  loss:  7.61715  grad_norm:  8.7573  memory: 0.66GiB(0.84%)  tps:     63  tflops:  0.02  mfu: 0.00%
[titan] step: 30  loss:  5.95356  grad_norm:  3.4653  memory: 0.72GiB(0.91%)  tps: 80,422  tflops: 23.53  mfu: 2.38%
[titan] Training completed
[titan] Process group destroyed
INFO - training/llm/torchtitan - command completed successfully
```

Run B holds the GPU long enough to sample residency with
`nvidia-smi -i <gpu> --query-compute-apps=pid,process_name,used_memory --format=csv`. The
`WARNING - Dataset domain_corpus is being re-looped (epoch N)` lines are expected: the loader
is `infinite=True` and the sample is 10 rows.

Run C — **checkpoint save path** (`--checkpoint.enable` overrides the recipe's default-off
block; frozen boolean *switch*, no value — §7.1):

```
[titan] Checkpointing active. Checkpoints will be loaded from and saved to .../ckpt_run/checkpoint
[titan] No checkpoint was provided, this is a fresh start.      <- random init (debugmodel has no HF weights)
[titan] Saving the checkpoint.
[titan] Finished saving the checkpoint in 0.60 seconds.         <- step-5, at interval
[titan] Saving a model only checkpoint in torch.bfloat16 at last step, step 10.   <- step-10, last_save_model_only + export_dtype bf16
[titan] Last step checkpoint completed in 0.12s

$ find .../ckpt_run/checkpoint -type f
.../checkpoint/step-5/.metadata     .../checkpoint/step-5/__0_0.distcp     (DCP, ~370 MB)
.../checkpoint/step-10/.metadata    .../checkpoint/step-10/__0_0.distcp    (model-only bf16, ~63 MB)
```

The DCP dirs are 433 MB total for this tiny model — delete them after verifying if you are on a
shared node. Smoke runs A/B (`checkpoint.enable` left at its default `False`) write only logs +
TensorBoard (~5 MB).

#### Quirks on H100 / CUDA (reversing the ROCm workarounds)

- **tf32 does not auto-engage here, by design.** torchtitan explicitly sets
  `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`;
  its speed path is bf16 mixed precision. Do not "fix" this.
- **`CUDA_VISIBLE_DEVICES` alone is the whole GPU-pinning story on NVIDIA** — no
  `HIP_VISIBLE_DEVICES`, no `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`. Check
  `torch.cuda.device_count()` and `nvidia-smi -i <gpu> --query-compute-apps` (PID-filtered).
- **The `CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.` warning
  from §7.1 does not appear on H100** — that path is available here (it was the
  degrade-to-eager notice for ROCm).
- **FlexAttention triton-autotunes on first use** (same as §7.1) — a wall of
  `triton_flex_attention_*` benchmarking lines on the first step; cached after.
- VRAM here is **80 GB** (vs 288 on MI355X), which is why the 8B `cpt_qwen3_8b_smoke` needs
  FSDP2 across ≥2 GPUs and the debugmodel is the right single-GPU smoke.

#### Scaling to 8 GPUs on NVIDIA

Not covered here beyond 2 GPUs. To run it: exactly the §7.1/§7.2 commands with
`CUDA_VISIBLE_DEVICES=0..7` and `--ngpu 8`, with no code changes expected. For the real 8B
`cpt_qwen3_8b`/`_smoke`, FSDP2 across ≥2 ranks is *required* on 80 GB cards (see the VRAM
note). `--ngpu` must equal the product of the parallelism degrees. Cold-start from real HF
safetensors (`checkpoint.enable=True` + `initial_load_in_hf`) is not covered on either vendor.

### 2-GPU run (2× H100) — real FSDP2 sharding

Real 2-rank FSDP2 (`dp_shard=2`) on H100, unmodified upstream torchtitan `03be241`, no source
patches. The shipped `cpt_debugmodel_smoke` recipe already sets
`data_parallel_shard_degree=-1` (`_base_config`), so `--nproc_per_node 2` builds a 2-way FSDP2
mesh with **zero** code change; the explicit `--parallelism.data_parallel_shard_degree 2`
override below just pins it for the log.

**Host / GPUs:** 2× H100 80GB HBM3, **CUDA 13.0**, Python 3.12, selected with
`CUDA_VISIBLE_DEVICES=6,7` (substitute your free devices); `torch.cuda.device_count()==2` under
that mask. Same venv, torch, and commit as the §7.3 single-GPU route:
`torch 2.15.0.dev20260822+cu130`, torchtitan `03be241`. **`torchao==0.10.0`** must be
pip-installed into the venv (it installs from pypi.org with the proxy left on; re-verify
`torch.version.hip is None` after — it does not clobber torch).

#### Launch command

Run directly with `torchrun` (not the `train_llm_torchtitan.py` wrapper) when you need to pin a
specific `--master_port` on a shared node — the wrapper hard-codes `--rdzv_endpoint localhost:0`
(ephemeral). Set `PYTHONPATH` to this folder + the torchtitan clone (what the wrapper's `build_env`
does) so `--module recipe_torchtitan` resolves, and export `PYTORCH_ALLOC_CONF=expandable_segments:True`.

```bash
cd .env_torchtitan/torchtitan_src                 # torchrun runs with the clone as CWD
source ../../.env_torchtitan/bin/activate
export CUDA_VISIBLE_DEVICES=6,7                    # only the two assigned free GPUs
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="<repo>/training/llm/torchtitan:$PWD"
export TITAN_OUT=$OUTPUT_DIR/torchtitan_2gpu

# A. 2-GPU FSDP2 smoke, 30 steps, + a sharded checkpoint at interval 15 and at the last step
torchrun --nproc_per_node=2 --master_port=29672 --local-ranks-filter 0,1 --role rank --tee 3 \
  -m torchtitan.train --module recipe_torchtitan --config cpt_debugmodel_smoke \
  --training.steps 30 --parallelism.data_parallel_shard_degree 2 \
  --checkpoint.enable --checkpoint.interval 15

# B. 400-step sustained hold (seq 2048, local batch 16) to sample both GPUs busy by PID
torchrun --nproc_per_node=2 --master_port=29672 --local-ranks-filter 0,1 --role rank --tee 3 \
  -m torchtitan.train --module recipe_torchtitan --config cpt_debugmodel_smoke \
  --training.steps 400 --training.seq_len 2048 --training.local_batch_size 16 \
  --parallelism.data_parallel_shard_degree 2
```

`--nproc_per_node 2` = `dp_shard 2` (`dp_shard·dp_replicate·tp·pp·cp = 2 = world size`); torchtitan
does **not** infer `--ngpu`/`nproc`, so it must equal the product of the degrees. `--local-ranks-filter 0,1`
shows **both** ranks' stdout (the single-GPU section only shows rank 0). TP is not covered here —
`dp_shard=2` is real FSDP2 optimizer/gradient sharding; a `tensor_parallel_degree=2` variant is the
same command with that override and `dp_shard` left at 1.

#### Expected output — the 2-rank FSDP2 mesh, on both ranks (Run A)

```
[rank1]:[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=1, ep=1
[rank1]:[titan] Successfully created meshes with active dimensions: ['batch', 'loss', 'dp', 'dp_shard']
[rank0]:[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=1, ep=1
[rank0]:[titan] Successfully created meshes with active dimensions: ['batch', 'loss', 'dp', 'dp_shard']
[rank0]:[titan] Total parameter count: dense 31,987,968, sparse 0, vision 0, active 31,987,968
[rank0]:[titan] Applied FSDP to the model
[rank1]:[titan] Applied FSDP to the model
[rank0]:[titan] step:  1  loss:  7.69873  grad_norm:  8.5730  memory:  0.47GiB(0.60%)  tps:     66  tflops:  0.02  mfu: 0.00%
[rank0]:[titan] step: 15  loss:  6.11777  grad_norm:  3.6688  memory:  0.54GiB(0.68%)  tps: 76,980  tflops: 22.52  mfu: 2.28%
[rank0]:[titan] step: 30  loss:  5.92992  grad_norm:  3.4533  memory:  0.57GiB(0.72%)  tps: 68,007  tflops: 19.90  mfu: 2.01%
[rank0]:[titan] Training completed
[rank0]:[titan] Process group destroyed
```

Both ranks print the identical `dp_shard=2` mesh and `Applied FSDP to the model`, and loss falls
on both. Check that two per-rank structured logs were written —
`structured_logs/training.global_rank_0.*.jsonl` **and** `...global_rank_1.*.jsonl` — the
independent confirmation that *both* ranks reached the end, not rank 0 alone.

#### Checking both GPUs are busy, by PID

While Run B holds, `nvidia-smi -i <gpu-a>,<gpu-b>` filtered by PID should show **two distinct
ranks, one on each physical GPU**:

```
$ nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader -i 6,7
<gpu-uuid-a>, <pid0>, .../.env_torchtitan/bin/python3, 4038 MiB   # rank 0
<gpu-uuid-b>, <pid1>, .../.env_torchtitan/bin/python3, 4036 MiB   # rank 1

$ nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 6,7
```

Two **different** GPU UUIDs and two **different** PIDs mean real 2-process, 2-GPU parallelism,
not one process on two contexts. The final loss is **identical on both ranks** — expected for
FSDP2, which is a single logical model, sharded.

#### The sharded DCP checkpoint (one shard per rank)

`--checkpoint.enable` flips the recipe's default-off block on (Run A). DCP `torch_dist` format
writes **one `.distcp` shard file per rank** — the direct proof the checkpoint is sharded across
the FSDP2 mesh, not gathered onto rank 0:

```
[rank0]:[titan] Saving the checkpoint.
[rank1]:[titan] Saving a model only checkpoint in torch.bfloat16 at last step, step 30.
[rank0]:[titan] Last step checkpoint completed in 0.09s

$ find .../torchtitan_2gpu/checkpoint -type f
.../checkpoint/step-15/.metadata   .../step-15/__0_0.distcp (177 MB)  .../step-15/__1_0.distcp (211 MB)   # full DCP, 2 shards
.../checkpoint/step-30/.metadata   .../step-30/__0_0.distcp ( 25 MB)  .../step-30/__1_0.distcp ( 41 MB)   # model-only bf16, 2 shards
```

`__0_0.distcp` = rank 0's shard, `__1_0.distcp` = rank 1's shard — two files, one per rank, at
both the interval (step-15, full DCP) and the last-step (step-30, `last_save_model_only` bf16)
checkpoints. With `checkpoint.enable` left off, a run writes logs only.

#### 8 GPUs on NVIDIA: expected, not covered here

Pure `dp_shard` FSDP2 scales with no code, dep, or env change — the mesh degree is just
`nproc_per_node`. On NVIDIA use `CUDA_VISIBLE_DEVICES=0..7` with
`--nproc_per_node 8 --parallelism.data_parallel_shard_degree 8`. For the real 8B
`cpt_qwen3_8b`, FSDP2 across ≥2 ranks is *required* on 80 GB cards anyway (see the VRAM note).

#### Notes specific to the 2-GPU run

- **Nothing in the recipe, deps, or environment changes** vs single-GPU besides
  `nproc_per_node 2` and the (redundant) explicit `dp_shard 2` override. NCCL is the collective
  backend with zero configuration across both ranks (RCCL is the AMD drop-in).
- **`torchao==0.10.0` must be present** in the venv. It does not clobber the torch nightly.
- Run `torchrun` **directly** rather than through `train_llm_torchtitan.py` only when you need
  to pin `--master_port` (the wrapper forces an ephemeral rdzv port). The wrapper works fine for
  2 GPUs too (`--ngpu 2`); everything else (PYTHONPATH, `PYTORCH_ALLOC_CONF`) mirrors the
  wrapper's `build_env`.
- The `WARNING - Dataset domain_corpus is being re-looped (epoch N)` lines are expected (the
  loader is `infinite=True` over the 10-row sample) and benign for a smoke.

#### Not covered here

Everything in §7.1's "Not covered" list, plus 8-GPU layouts on NVIDIA (see above), the
8B/Llama/SFT configs on NVIDIA, `torch.compile`, and float8/MXFP8/NVFP4 (NVIDIA-hardware
features — none are wired into this recipe).

## 8. Notes

- **The Python file is a launcher, not a trainer.** torchtitan's entrypoint is
  `python -m torchtitan.train`, driven by `torchrun`. `train_llm_torchtitan.py` validates
  that `--titan-repo` really is a torchtitan checkout, builds the `torchrun` invocation, sets
  `PYTORCH_ALLOC_CONF=expandable_segments:True`, prepends this folder to `PYTHONPATH`, and
  execs.
- **Config is Python.** `--module`/`--config` name a function; the frozen `--section.option`
  flags still work and take precedence, which is why `--extra --training.steps 200` overrides
  without touching the recipe.
- **Continued pre-training vs SFT is a dataloader swap.** Both recipes share the same
  `_base_config()`; CPT uses `HuggingFaceTextDataLoader` over the registered local corpus,
  SFT uses `ChatDataLoader` with a `sample_processor` that emits the row's `messages`. The
  chat loader is what applies the chat template and masks prompt tokens.
- **HF interop happens inside the checkpointer.** `initial_load_in_hf` + `initial_load_path`
  make DCP read safetensors through the model's state-dict adapter at step 0;
  `last_save_in_hf` + `last_save_model_only` write safetensors back at the end. The offline
  `convert_from_hf.py` / `convert_to_hf.py` scripts do the same transformation without a
  training loop. Note that for Llama 3 the adapter permutes several attention matrices to
  reconcile HF and native RoPE layouts — that is why you should convert with the scripts
  rather than renaming tensors by hand.
- **Parallelism is composed, not stacked into one flag.** The degrees multiply:
  `dp_shard * dp_replicate * tp * pp * cp` must equal the world size. On one 8x H100 node,
  FSDP2-only (`dp_shard=-1`) is the right default for an 8B model; reach for TP before PP,
  and CP only when the sequence is the memory problem.

### Where this folder is uncertain

The local-dataset registration in `recipe_torchtitan.py` (`DatasetConfig` + `DATASETS`) works
as written at commit `03be241`. Its loader defaults to `infinite=True`, so the 10-row sample
loops rather than ending the run — fine for a smoke, misleading for an epoch count.

Anything else you want (float8 via `Float8LinearConverter`, MXFP8/NVFP4, async checkpointing,
TorchFT) exists upstream but is not wired up in this recipe.
