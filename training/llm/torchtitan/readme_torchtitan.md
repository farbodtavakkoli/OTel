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

> **Tested topology:** the NVIDIA 8x H100 path this folder was written against is still
> **unrun**. The folder itself **has now been run end-to-end on AMD** — first 2x Instinct
> MI355X (gfx950) under ROCm 7.2, FSDP2 across both GPUs, then the **full 8x MI355X node**
> with both `dp_shard=8` and a 2-D `dp_shard=4 x tp=2` mesh, all on **2026-08-19**: see
> [§7.1 Verified on AMD MI355X](#71-verified-on-amd-mi355x-rocm-72) and
> [§7.2 8-GPU run](#8-gpu-run-8x-mi355x-rocm-724--tested-august-2026). It was written against
> the torchtitan `main` branch documentation and source as of **August 2026** and targets
> the house default of a single node with 8x H100 80GB, launched with `torchrun`.
> torchtitan is explicitly "under extensive development" upstream and its config surface
> changed recently (see the note below); treat every config key as something to re-check
> against your checked-out commit before a long run. One such drift was found during the
> AMD run and is documented in §7.1: **`checkpoint.enable` now defaults to `False`**, so
> the checkpoint block in `recipe_torchtitan.py` is inert until you set it.

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
directly. **This is the path that was actually tested here** (2x MI355X / ROCm 7.2,
2026-08-19 — full transcript in [§7.1](#71-verified-on-amd-mi355x-rocm-72)):

```bash
# PyTorch nightly for ROCm instead of CUDA (step 1 above); the rest is identical.
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

Use the **nightly**, not a stable ROCm wheel, unless you have checked your torchtitan
commit against it — upstream states plainly that running from the source tree "requires
the nightly build of PyTorch", and the tested combination below is a nightly. The stable
`rocm7.2` index does carry `torch-2.13.0+rocm7.2` (also 2.12.1 / 2.12.0 / 2.11.0); §7.1
records how far that stable wheel got against this same torchtitan commit.

- **Upstream:** the [torchtitan README](https://github.com/pytorch/torchtitan#nightly-builds)
  — "You can replace `cu130` with another version of cuda or an AMD GPU (e.g. `rocm6.3`)."
  torchtitan itself is pure PyTorch (FSDP2 / DTensor / DCP / SDPA), so torch-level
  portability is the mechanism: no CUDA-only extension is required.
- **AMD fork:** AMD released an optimized torchtitan fork for AMD GPUs in November 2025:
  [AMD-AGI/torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd).
- **AMD Primus:** [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus) uses **TorchTitan as
  one of its training backends** (added June 2025, upgraded since) and ships it inside the
  `rocm/primus` Docker images — AMD's recommended route on Instinct GPUs.
- **Stable-release wheels:** if you use a stable torch (2.11+) instead of a nightly, ROCm
  wheels live on the `rocm7.2` index: https://download.pytorch.org/whl/rocm7.2 (verified to
  carry `torch-2.11.0+rocm7.2` / `torch-2.12.0+rocm7.2`; the older `rocm6.4` index does not
  have 2.11.0). Note the float8/MXFP8 quantization converters are NVIDIA-hardware features.

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

## 7. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** none claimed beyond NVIDIA/AMD — pure-PyTorch stack; follows torch device support.


| Platform | Status | Evidence (checked August 2026) |
|---|---|---|
| NVIDIA CUDA | First-class upstream | [torchtitan README](https://github.com/pytorch/torchtitan) — CI on 8-GPU NVIDIA runners; MXFP8 targets Blackwell. |
| AMD ROCm | **Works — measured here at 2 and 8 GPUs**, and actively invested in by AMD | **This folder ran on 2x Instinct MI355X (gfx950) / ROCm 7.2.4 on 2026-08-19 (§7.1), then on all 8x MI355X the same day — 8-way FSDP2 at 13.25% MFU and a composable FSDP2(4) x TP(2) mesh, both clean, no code changes (§7.2).** Upstream context: [torchtitan README, Nightly builds](https://github.com/pytorch/torchtitan#nightly-builds) — replace the CUDA index with an AMD one (e.g. `rocm6.3`); upstream CI has a `torchtitan-rocm-ubuntu-22.04-clang12` docker target (`.ci/docker/build.sh`); README news 2025/11 — AMD's optimized fork [AMD-AGI/torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd); [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus) lists TorchTitan as a training backend shipped in `rocm/primus` images. torchtitan is pure-PyTorch (FSDP2/DTensor/DCP/SDPA), so portability comes from torch itself — **the upstream fork was not needed**. Stable ROCm wheels exist on the [`rocm7.2` index](https://download.pytorch.org/whl/rocm7.2) (2.13.0 / 2.12.1 / 2.12.0 / 2.11.0) but are **not new enough for torchtitan `main`** — see §7.1. |
| Intel XPU / Apple | No supported path documented | Not mentioned by upstream. |

### 7.1 Verified on AMD MI355X (ROCm 7.2)

**Verdict: WORKS — unmodified upstream torchtitan, stock PyTorch ROCm nightly, no AMD fork,
no source patches.** Two additions were made in this folder: a tiny bring-up config
(`cpt_debugmodel_smoke` in `recipe_torchtitan.py`) and this section. Both the tiny config
and the folder's own shipped `cpt_qwen3_8b_smoke` recipe trained with finite, falling loss
on 2 GPUs with FSDP2.

**Host:** 8x AMD Instinct MI355X (gfx950, 288GB HBM each), ROCm 7.2.4, Ubuntu, Python 3.12.3.
Only physical GPUs 0 and 1 were used (`HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1`).

**Versions that worked** — `torch 2.15.0.dev20260818+rocm7.2` (nightly), `triton-rocm
3.8.0+gitdf3f91dd`, `torchdata 0.11.0`, `datasets 4.7.0`, `transformers 5.15.0`,
`tokenizers 0.22.2`, `safetensors 0.8.0`, `tyro 1.0.15`, `einops 0.8.2`,
`spmd_types 0.2.3`, `tensorboard 2.21.0`, torchtitan source at commit `03be241`.

#### Install, exactly as run

```bash
cd training/llm/torchtitan
python3.12 -m venv .env_train_llm_torchtitan
source .env_train_llm_torchtitan/bin/activate

# 1. PyTorch ROCm NIGHTLY (see "why the nightly" below — a stable ROCm wheel does not work).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2

# 2. torchtitan from source. Kept INSIDE the git-ignored venv dir so `git status`
#    in this repo stays clean; any path works, it is passed as --titan-repo.
git clone https://github.com/pytorch/torchtitan .env_train_llm_torchtitan/torchtitan_src
pip install -r .env_train_llm_torchtitan/torchtitan_src/requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu

# 3. This folder's extras.
pip install -r requirements_torchtitan.txt

# Sanity check (run from the clone, which is where the launcher runs commands):
cd .env_train_llm_torchtitan/torchtitan_src
python -c "import torch, torchtitan.train; print(torch.__version__, torch.cuda.device_count(), 'GPUs')"
# -> 2.15.0.dev20260818+rocm7.2 2 GPUs
```

No `pip install -e .` of torchtitan is needed — the launcher runs with the clone as CWD, so
the package imports from the tree (it then reports `torchtitan version: 0.0.0+unknown`,
which is harmless). **Do not install `flash-attn`**: torchtitan uses SDPA / FlexAttention,
and both worked as shipped. `torch` was never silently replaced by a CUDA wheel here, but
check `python -c "import torch; print(torch.version.hip)"` after any later `pip install`
and force-reinstall from the ROCm index if it comes back `None`.

#### Why the nightly, and not stable `torch 2.13.0+rocm7.2`

Tested directly, in a throwaway venv against the same torchtitan commit: `torch
2.13.0+rocm7.2` **imports torchtitan fine but cannot train**. The 2-rank run dies while
FSDP2 wraps the model:

```
torchtitan/distributed/fsdp.py:188 in apply_fsdp_to_decoder -> torch ... fully_shard
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on the full
SPMD mesh (e.g. via distribute_module). Got plain tensor for parameter 'weight'.
```

torchtitan `main` passes a `dp_mesh_dims` argument whose DTensor contract only exists in the
newer FSDP2 in the nightly. So the nightly requirement in this folder's docs is **real**,
not caution — it is a torchtitan-vs-torch API constraint, not an AMD one. Pin an older
torchtitan release if you must run a stable wheel.

#### Launch command, exactly as run

```bash
cd training/llm/torchtitan
source .env_train_llm_torchtitan/bin/activate   # exports HIP/CUDA_VISIBLE_DEVICES=0,1
export TITAN_OUT=/mnt/data_1.5t/outputs/train_llm_torchtitan

# A. tiny bring-up smoke: random-init qwen3 debugmodel, no weight download, no checkpoints
python train_llm_torchtitan.py \
  --titan-repo .env_train_llm_torchtitan/torchtitan_src \
  --config cpt_debugmodel_smoke --ngpu 2

# B. the folder's own shipped 8B smoke, cut to 5 steps (random init: no HF weights on this
#    box, so TITAN_HF_ASSETS points at a tokenizer-only dir)
export TITAN_HF_ASSETS=/path/to/a/qwen3/tokenizer/dir
python train_llm_torchtitan.py \
  --titan-repo .env_train_llm_torchtitan/torchtitan_src \
  --config cpt_qwen3_8b_smoke --ngpu 2 --extra --training.steps 5
```

`--ngpu 2` matches `dp_shard=2`; no `MASTER_ADDR`/`MASTER_PORT` is needed because the
launcher uses `--rdzv_endpoint localhost:0`, so torchrun picks a free port itself (the
usual 29500 collision with other jobs on this box cannot happen).

#### Evidence

Run A — `cpt_debugmodel_smoke`, 8 steps, seq 512, local batch 2:

```
[titan] INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=1, ep=1
[titan] INFO - CUDA capacity: AMD Instinct MI355X with 287.98GiB memory
[titan] INFO - Applied FSDP to the model
[titan] INFO - step:  1  loss:  7.58945  grad_norm:  9.3716  memory:  0.39GiB(0.13%)  tps: 68      tflops: 0.02  mfu: 0.00%
[titan] INFO - step:  8  loss:  6.48329  grad_norm:  3.6826  memory:  0.47GiB(0.16%)  tps: 24,468  tflops: 7.16  mfu: 0.29%
```

Longer version of the same config (600 steps, seq 2048, local batch 16, via `--extra`),
used to hold the GPUs busy long enough to sample `rocm-smi`:

```
[titan] INFO - step: 100  loss:  3.20326  memory: 2.40GiB(0.83%)  tps: 180,231  tflops: 107.16  mfu: 4.29%
[titan] INFO - step: 600  loss:  0.33155  memory: 2.40GiB(0.83%)  tps: 182,771  tflops: 108.67  mfu: 4.35%
```

```
$ rocm-smi --showuse    # sampled mid-run
GPU[0] : GPU use (%): 85
GPU[1] : GPU use (%): 90
GPU[2..7] : GPU use (%): 0      <- only the two assigned GPUs are working
```

Run B — the folder's shipped `cpt_qwen3_8b_smoke` at 8.19B parameters, 5 steps, seq 4096,
FullAC, FSDP2 over 2 ranks (weights random-init, see caveat below):

```
[titan] INFO - Total parameter count: dense 8,190,735,360, sparse 0, vision 0, active 8,190,735,360
[titan] INFO - CUDA memory usage for model: 15.65GiB(5.43%)
[titan] INFO - step:  1  loss: 12.74905  grad_norm: 13.0331  memory: 63.93GiB(22.20%)  tps:   456  tflops:  24.00  mfu: 0.96%
[titan] INFO - step:  5  loss:  7.67035  grad_norm: 14.7446  memory: 74.23GiB(25.77%)  tps: 4,679  tflops: 246.36  mfu: 9.85%
```

RCCL is picked up as the `nccl` backend with zero configuration; the mesh line above and the
falling loss on both ranks are the proof that collectives worked.

#### Quirks found on ROCm

- **`checkpoint.enable` defaults to `False` upstream** (`torchtitan/components/checkpointer/base.py`),
  and `recipe_torchtitan.py` never sets it. Consequence on *any* vendor: the whole checkpoint
  block in the recipes — `initial_load_in_hf`, `initial_load_path`, `interval`,
  `last_save_in_hf` — is **inert**, so a "CPT" run silently trains from **random init** and
  saves nothing. Set `config.checkpoint.enable = True` in the recipe before a real run, and
  confirm the log does *not* say "No checkpoint was provided, this is a fresh start".
  This is why Run B above is random-init and starts at loss ~12.7 (≈ `ln(vocab)`), exactly
  the failure signature §5 warns about.
- **`--checkpoint.enable False` is not valid syntax.** The frozen tyro flags treat booleans
  as switches, so a value errors with `Unrecognized options: False`. Use
  `--checkpoint.enable` to turn it on, or edit the recipe.
- **`WARNING - CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.`**
  — expected and harmless on ROCm; torchtitan degrades to eager and keeps going.
- **FlexAttention triton-autotunes on first use** (the qwen3 debugmodel path), printing a
  wall of `triton_flex_attention_*` benchmarking lines and adding ~1-2 min to the first run.
  It compiles and runs correctly on gfx950; later runs reuse the cache.
- Do **not** set `CUDA_VISIBLE_DEVICES=""` on ROCm — it hides every GPU rather than none.
- torchtitan writes DCP checkpoints that are large by default (a 5-step 8B run left a 31GB
  `checkpoint/step-10/` here when enabled). Keep `TITAN_OUT` off the small volume, and set
  `checkpoint.enable = False` for smoke tests — the runs above wrote ~20MB total
  (TensorBoard + structured JSONL logs).

#### Not tested here

Cold-starting from real HF safetensors (`initial_load_in_hf`), the `convert-from-hf` /
`convert-to-hf` paths, the SFT recipe (`sft_qwen3_8b`), multi-node, PP/CP degrees > 1,
`torch.compile`, and float8/MXFP8/NVFP4 (NVIDIA-hardware features). No HF checkpoint was
downloaded on this box, so both smoke runs above are random-init.
(**TP degree > 1 is no longer untested** — `tensor_parallel_degree=2` was exercised on 8 GPUs
in §7.2 below.)

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

**Verdict: WORKS — no changes at all.** The 2-GPU recipe scaled straight to all 8 MI355X with
**zero edits to `recipe_torchtitan.py`, `train_llm_torchtitan.py`, or
`requirements_torchtitan.txt`** — only `--ngpu 8` plus the frozen `--parallelism.*` CLI
overrides. Two parallelism layouts were run back to back on 2026-08-19, **both exited 0 with
finite, monotonically falling loss on all 8 ranks**:

| Stage | Parallelism | Loss (step 1 → 250) | tok/s/GPU | TFLOP/s | MFU | Peak mem/GPU | rc |
|---|---|---|---|---|---|---|---|
| **A** | `dp_shard=4` × `tp=2` (2-D, FSDP2+TP) | 12.74128 → **0.10037** | 2,861 | 150.67 | 6.03% | 21.80 GiB (7.57%) | **0** |
| **B** | `dp_shard=8` × `tp=1` (pure FSDP2) | 12.77955 → **0.06754** | **6,292** | **331.31** | **13.25%** | 27.39 GiB (9.51%) | **0** |

Model: qwen3 **8B flavor, 8,190,735,360 params**, `seq_len=4096`, `local_batch_size=1`, FullAC,
bf16 — deliberately identical to §7.1 Run B so the numbers are comparable. Weights are
random-init (`checkpoint.enable` is still `False`, see §7.1), so the loss curve measures the
*training machinery*, not model quality.

#### Which parallelism to use, and why

**Use pure FSDP2 (`dp_shard=8`) on this box.** Stage B is **2.2x faster per GPU** than Stage A
(6,292 vs 2,861 tok/s) — TP is *functional* on gfx950 but not *preferable* here. The reason is
visible in the log rather than guessed: an 8B model needs only ~27 GiB of a **288 GiB** HBM
stack, so TP buys memory headroom that is not needed, while adding per-layer collectives on the
activation path. torch's own DTensor layer flags the specific cost:

```
torch/distributed/tensor/_redistribute.py:394] While redistributing from
(_NormPartial(2.0), _NormPartial(2.0)) to (Replicate(), Replicate()), 2 sequential all_reduce
operations will be performed. This is suboptimal ... To optimize, flatten mesh dimensions
["dp_shard", "tp"] so DTensor can use a single operation instead.
```

Stage A was included because torchtitan's whole selling point is composable N-D parallelism, and
it is worth recording that **the 2-D mesh really did build and train on ROCm** — TP is not a
paper feature here. Reach for `tp>1` only when a model genuinely will not fit (32B+ / long
context), which is the same advice §8 already gives.

#### Launch command, exactly as run

```bash
cd training/llm/torchtitan
source .env_train_llm_torchtitan/bin/activate
# CRITICAL: activate ends with a stale 2-GPU pin from the §7.1 session. Override it AFTER
# sourcing, or you silently train on 2 GPUs and think you tested 8.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/data_1.5t/hf_cache
export TITAN_OUT=/mnt/data_1.5t/outputs/train_llm_torchtitan/gpu8
export TITAN_HF_ASSETS=/mnt/data_1.5t/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

# assert all 8 are really visible before burning an hour on a 2-GPU run
python -c "import torch,sys; n=torch.cuda.device_count(); print(n); sys.exit(n!=8)"

# A — composable 2-D: FSDP2(4) x TP(2)
python train_llm_torchtitan.py --titan-repo .env_train_llm_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 4 \
          --parallelism.tensor_parallel_degree 2

# B — pure FSDP2 across all 8 (the recommended layout)
python train_llm_torchtitan.py --titan-repo .env_train_llm_torchtitan/torchtitan_src \
  --module recipe_torchtitan --config cpt_qwen3_8b_smoke --ngpu 8 --log-rank 0 \
  --extra --training.steps 250 \
          --parallelism.data_parallel_shard_degree 8 \
          --parallelism.tensor_parallel_degree 1
```

Stage B's degrees are also just the shipped defaults (`data_parallel_shard_degree=-1` → "all
leftover ranks"), so `--config cpt_qwen3_8b_smoke --ngpu 8` with no `--extra` is equivalent.

**Exact toolchain (unchanged from §7.1 — the nightly is load-bearing):**
`torch 2.15.0.dev20260818+rocm7.2` / `hip 7.2.53211`, torchtitan source at commit `03be241`,
Python 3.12.3, ROCm 7.2.4. Stable `torch 2.13.0+rocm7.2` still fails in `fully_shard` exactly as
§7.1 documents — that gap is unrelated to GPU count and was not retested.

**No fixed master port.** `train_llm_torchtitan.py` hardcodes `--rdzv_endpoint localhost:0`, so
torchrun picks a free ephemeral port and `--master_port` is neither accepted nor needed; this is
what makes concurrent jobs on one box collision-proof.

#### Evidence — all 8 GPUs, captured mid-run

`rocm-smi` sampled **from inside the running job** (full capture:
`$TITAN_OUT/rocm_smi_8gpu.txt`, driving log `run.log`):

```
[assert] torch=2.15.0.dev20260818+rocm7.2 hip=7.2.53211 device_count=8
[assert]   GPU0..GPU7: AMD Instinct MI355X            <- all 8, not the stale 0,1 pin

[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=4, cp=1, tp=2, ep=1
[titan] Successfully created meshes with active dimensions: ['batch', 'loss', 'tp', 'dp', 'dp_shard']
[titan] CUDA capacity: AMD Instinct MI355X with 287.98GiB memory
[titan] Total parameter count: dense 8,190,735,360, sparse 0, vision 0, active 8,190,735,360

$ rocm-smi --showuse          # sampled during training
GPU[0..7] : GPU use (%): 100                 <- ALL EIGHT saturated
$ rocm-smi --showmemuse
GPU[0..7] : GPU Memory Allocated (VRAM%): 12

$ rocm-smi --showpids         # 8 ranks, ~40 GB VRAM each, PIDs owned by this run
3231958 python3  1  39531040768      3231962 python3  1  39919009792
3231959 python3  1  39977734144      3231963 python3  1  39826735104
3231960 python3  1  39984021504      3231964 python3  1  39845613568
3231961 python3  1  39971438592      3231965 python3  1  39835127808
```

Step lines (Stage B, pure FSDP2 over 8 ranks):

```
step:   1  loss: 12.77955  grad_norm: 14.1134  memory: 18.42GiB(6.40%)  tps:   411  tflops:  21.67  mfu: 0.87%
step:  10  loss:  2.11147  grad_norm:  4.1048  memory: 27.39GiB(9.51%)  tps: 6,484  tflops: 341.46  mfu: 13.66%
step: 100  loss:  0.21247  grad_norm:  1.0501  memory: 27.39GiB(9.51%)  tps: 6,301  tflops: 331.82  mfu: 13.27%
step: 250  loss:  0.06754  grad_norm:  0.5762  memory: 27.39GiB(9.51%)  tps: 6,292  tflops: 331.31  mfu: 13.25%
[titan] Training completed
[titan] Process group destroyed
```

All **8** per-rank structured logs (`structured_logs/training.global_rank_{0..7}.*.jsonl`) were
written for **both** stages — 16 files — which is the independent check that every rank reached
the end rather than rank 0 finishing alone.

#### Scaling vs the 2-GPU run

| | 2 GPU (§7.1 Run B) | 8 GPU (pure FSDP2) |
|---|---|---|
| tok/s **per GPU** | 4,679 | **6,292** (1.34x) |
| tok/s **aggregate** | ~9,358 | **~50,336** (~5.4x) |
| MFU | 9.85% | **13.25%** |
| Peak memory per GPU | 74.23 GiB (25.77%) | **27.39 GiB (9.51%)** |

Per-GPU throughput went **up**, not down — the usual scaling story is the opposite, and the
reason is memory: FSDP2 shards optimizer + gradients 8 ways instead of 2, dropping per-GPU
resident memory from 74 GiB to 27 GiB and giving the allocator far more room. Caveat, stated
plainly: the 2-GPU figure is from **step 5 of a 5-step run** and was still ramping, so treat
1.34x as indicative; the 8-GPU figures are steady-state over 250 steps.

#### What differed from the 2-GPU run

- **Nothing in the code, the deps, or the environment.** No new package, no pin change, no
  RCCL/NCCL tuning variables, no `HSA_*`/`NCCL_*` overrides, no batch-size reduction, no OOM,
  no hang, no deadlock. RCCL was picked up as the `nccl` backend with zero configuration across
  all 8 ranks exactly as it was across 2.
- **The stale `CUDA_VISIBLE_DEVICES=0,1` in `.env_train_llm_torchtitan/bin/activate`** is the
  one real trap. Sourcing the venv silently caps you at 2 GPUs; the `device_count()==8`
  assertion above is what catches it. Consider deleting those two lines from `activate`.
- **`--parallelism.*` frozen CLI flags work** and are the clean way to re-shape the mesh without
  editing the recipe — but note they take *values* (`--parallelism.tensor_parallel_degree 2`),
  unlike the boolean switches §7.1 warns about.
- **`torch.compile` was not enabled** and was not needed; this recipe never turns it on. The
  `CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.` warning from
  §7.1 still appears on all 8 ranks and is still harmless.
- **Benign warning, worth knowing:** `ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be
  overridden to 3 based on job config` — torchtitan setting its own collective error handling.
- **Async-TP was not exercised** (it requires `torch.compile`); only plain TP. Note that
  `--ngpu` must still equal the product of the degrees — torchtitan does not infer it.
- **Disk:** with `checkpoint.enable=False` the whole 8-GPU campaign wrote **66 MB**, all logs and
  TensorBoard — **zero weight files**, nothing to clean up.

### 7.3 Verified on NVIDIA H100 (CUDA 13.0) — single GPU, tested August 2026

**Verdict: WORKS — unmodified upstream torchtitan (same commit `03be241` as the AMD run),
stock PyTorch CUDA nightly, no source patches.** The `cpt_debugmodel_smoke` bring-up config
from §7.1 ran end-to-end on one H100 with finite, monotonically falling loss, a saved DCP
checkpoint, and GPU-5 residency confirmed by PID. This is a **single-GPU smoke** — multi-GPU
(FSDP2 across 2/8) is deferred here (the box was shared; GPUs 0–3 were a co-tenant job). The
AMD §7.1/§7.2 evidence already covers the 2- and 8-GPU FSDP2/TP behaviour; nothing in that is
device-specific and it is expected to reproduce on H100 unchanged (see "What multi-GPU needs").

**Host:** 8x NVIDIA H100 80GB HBM3, driver **580.173.02**, **CUDA 13.0**, Hopper cc(9,0),
Python 3.12.3. Only **physical GPU 5** was used (`CUDA_VISIBLE_DEVICES=5`; `torch.cuda.device_count()==1`).

**Versions that worked** — `torch 2.15.0.dev20260822+cu130` (nightly; `torch.version.cuda==13.0`,
`torch.version.hip is None`), `triton 3.8.0`, `torchdata 0.11.0`, `datasets 4.7.0`,
`transformers 5.15.1`, `tokenizers 0.22.2`, `safetensors 0.8.0`, `tyro 1.0.16`, `grain 0.2.18`,
torchtitan source at commit **`03be241`** (same as the AMD run). This is the direct CUDA analog
of §7.1's `2.15.0.dev20260818+rocm7.2` — **same `2.15.0.dev` nightly line, cu130 instead of
rocm7.2.**

#### The nightly is mandatory on CUDA too — verified, not assumed

`pip install torch numpy` on this box installs **stable `torch 2.13.0+cu130`** (native CUDA 13,
real bf16 matmul). Against torchtitan `03be241` that stable wheel gets **further than the AMD
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

#### Install, exactly as run

```bash
cd training/llm/torchtitan
python3.12 -m venv .env_torchtitan            # here: venv on tmpfs, .env_torchtitan -> /dev/shm/...
source .env_torchtitan/bin/activate

# 1. PyTorch CUDA NIGHTLY (stable 2.13.0+cu130 fails at set_pg_timeouts, see above).
#    download.pytorch.org is proxy-blocked on this box -> unset the proxy for THIS install:
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
#    -> torch 2.15.0.dev20260822+cu130

# 2. torchtitan from source, pinned to the tested commit (fresh HEAD 304fd88 has drift, below).
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
**pypi.org (proxy-allowlisted)** — only the torch nightly needs the proxy off. torch survived
every subsequent `pip install` here (verified `torch.version.hip is None` after each). No
`flash-attn`, no `pip install -e .` of torchtitan — the launcher runs with the clone as CWD.

**Commit drift, real:** a fresh `git clone` (HEAD `304fd88` at test time) **breaks the shipped
recipe** — `recipe_torchtitan.py`'s `from torchtitan.components.checkpoint import CheckpointManager`
raises `ModuleNotFoundError` (that module moved), and torchtitan's new `config/manager.py`
swallows the ImportError and reports "Cannot import module 'recipe_torchtitan'". Pinning
`03be241` (the AMD-verified commit) makes the recipe import cleanly. This is exactly the
"re-check every config key against your checked-out commit" warning at the top of this file,
now with a concrete failure.

#### Launch command, exactly as run

```bash
cd training/llm/torchtitan
source .env_torchtitan/bin/activate
export CUDA_VISIBLE_DEVICES=5                       # the ONE assigned free GPU on a shared node
export HF_HOME=/mnt/gsma/gsma/gsma/models
export HF_DATASETS_CACHE=/dev/shm/h100/dscache_torchtitan
export TITAN_OUT=/dev/shm/h100/out/torchtitan

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
`--rdzv_endpoint localhost:0`, so torchrun picks a free ephemeral port (collision-proof with the
co-tenant job — the assigned port 29645 was reserved but never contended).

#### Evidence

Run A — `cpt_debugmodel_smoke`, 30 steps, seq 512, local batch 2, **loss falls the whole way**:

```
[titan] CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] Total parameter count: dense 31,987,968, sparse 0, vision 0, active 31,987,968
[titan] Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1
[titan] Applied FSDP to the model
[titan] step:  1  loss:  7.61715  grad_norm:  8.7573  memory: 0.66GiB(0.84%)  tps:     63  tflops:  0.02  mfu: 0.00%
[titan] step: 30  loss:  5.95356  grad_norm:  3.4653  memory: 0.72GiB(0.91%)  tps: 80,422  tflops: 23.53  mfu: 2.38%
[titan] Training completed
[titan] Process group destroyed
2026-... - INFO - training/llm/torchtitan - command completed successfully
```

Run B — 400 steps, **GPU-5 residency + sustained utilisation sampled from inside the run**
(`nvidia-smi -i 5`, i.e. the assigned GPU only, filtered by PID):

```
$ nvidia-smi -i 5 --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
1545153, /dev/shm/h100/venv_torchtitan/bin/python3, 3940 MiB      <- our PID, sole process on GPU 5

$ nvidia-smi -i 5 --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
5, 63 %, 3949 MiB        # three samples over 6 s: 63% / 60% / 70% busy
5, 60 %, 3949 MiB
5, 70 %, 3949 MiB

[titan] step:   1  loss:  7.6...  ...
[titan] step: 100  loss:  3.25099  grad_norm: 1.5728  tps: 176,425  tflops: 104.90  mfu: 10.61%
[titan] step: 200  loss:  2.26661  grad_norm: 6.2903  tps: 185,313  tflops: 110.18  mfu: 11.14%
[titan] step: 300  loss:  1.62513  grad_norm: 8.6927  tps: 182,144  tflops: 108.30  mfu: 10.95%
[titan] step: 400  loss:  1.38723  grad_norm: 7.7393  tps: 182,449  tflops: 108.48  mfu: 10.97%
[titan] Training completed
```

~180k tok/s and ~108 TFLOP/s (MFU ~11%) on a single H100 for the 32M debugmodel — in the same
ballpark as the MI355X debugmodel run in §7.1 (~108 TFLOP/s). (The `WARNING - Dataset
domain_corpus is being re-looped (epoch N)` lines are expected: the loader is `infinite=True`
and the sample is 9 rows — §"Where this folder is uncertain" #1.)

Run C — **checkpoint save path confirmed** (`--checkpoint.enable` overrides the recipe's
default-off block; frozen boolean *switch*, no value — §7.1):

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

The DCP dirs (433 MB total) were **deleted after capturing this evidence** per the shared-node
disk rule; the file listing above is retained at
`/dev/shm/h100/out/torchtitan/checkpoint_tree_listing.txt`. Smoke runs A/B (`checkpoint.enable`
left at its default `False`) wrote only logs + TensorBoard (~5 MB).

#### Quirks found on H100 / CUDA (reversing the ROCm workarounds)

- **tf32 does NOT auto-engage here, and that is torchtitan's choice, not a miss.**
  `torchtitan/distributed/utils.py:383-384` explicitly sets
  `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`
  — torchtitan's speed path is **bf16 mixed precision**, not tf32, so it deliberately disables
  tf32. The FlexAttention autotuned kernels run `FLOAT32_PRECISION='ieee'`. So the usual "verify
  tf32 auto-enables on CUDA" guidance is moot for *this* trainer.
- **`CUDA_VISIBLE_DEVICES=5` alone is the whole GPU-pinning story on NVIDIA** — no
  `HIP_VISIBLE_DEVICES`, no `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`. `device_count()==1`
  and `nvidia-smi -i 5 --query-compute-apps` (PID-filtered) is the residency proof.
- **The `CUDA graph capture is only supported on NVIDIA CUDA; using eager execution.` warning
  from §7.1 does NOT appear on H100** — that path is available here, as expected (it was the
  degrade-to-eager notice for ROCm).
- **FlexAttention triton-autotunes on first use** (same as §7.1) — a wall of
  `triton_flex_attention_*` benchmarking lines adds ~10–15 s to the first step; cached after.
  This is why step 1 shows near-zero tps and step 2+ jump to ~180k.
- VRAM here is **80 GB** (vs 288 on MI355X). Irrelevant for the 32M debugmodel (peaked 2.63 GiB),
  but it is the reason the 8B `cpt_qwen3_8b_smoke` would need FSDP2 across ≥2 GPUs (§7.1 Run B
  peaked 74 GiB on *2* MI355X ranks; a single 80 GB H100 has no headroom for optimizer+grads at
  seq 4096) — hence the debugmodel was the right single-GPU smoke.

#### What multi-GPU (2, then 8) would need on this box — DEFERRED

Not run here (GPUs 0–3 were a co-tenant production job; only GPU 5 was assigned). To run it once
GPUs free up: exactly the §7.1/§7.2 commands with `CUDA_VISIBLE_DEVICES=0,1` (then `0..7`) and
`--ngpu 2` (then `8`), no code changes expected — torchtitan is pure PyTorch and NCCL is the
native backend on H100 (RCCL was the drop-in on AMD). For the real 8B `cpt_qwen3_8b`/`_smoke`,
FSDP2 across ≥2 ranks is *required* on 80 GB cards (see VRAM note). `--ngpu` must equal the
product of the parallelism degrees. Cold-start from real HF safetensors (`checkpoint.enable=True`
+ `initial_load_in_hf`) was not exercised on either vendor.

### 2-GPU run (2× H100) — real FSDP2 sharding, tested August 2026

**Verdict: WORKS — real 2-rank FSDP2 (`dp_shard=2`) proven on H100, unmodified upstream
torchtitan `03be241`, no source patches.** This supersedes the *2-GPU* half of the "DEFERRED"
note above (the 8-GPU half stays projected — see the closing note). torchtitan is *built* for
this: the shipped `cpt_debugmodel_smoke` recipe already sets `data_parallel_shard_degree=-1`
(§`_base_config`), so `--nproc_per_node 2` builds a 2-way FSDP2 mesh with **zero** code change;
the explicit `--parallelism.data_parallel_shard_degree 2` override below just pins it for the log.
Two ranks launched, a `dp_shard=2` device mesh applied on both, both **physical GPUs 6 and 7**
busy by PID, finite decreasing loss over 30 and 400 steps, and a **sharded DCP checkpoint (one
`.distcp` shard file per rank)**.

**Host / GPUs:** same 8× H100 80GB HBM3 box, driver **580.173.02**, **CUDA 13.0**, Python 3.12.3.
Only **physical GPUs 6 AND 7** used (`CUDA_VISIBLE_DEVICES=6,7`; GPUs 0–3 were the co-tenant
production job, 4/5 other agents). `torch.cuda.device_count()==2` under that mask. Same venv,
torch, and commit as §7.3 single-GPU: `torch 2.15.0.dev20260822+cu130`, torchtitan `03be241`.
**`torchao==0.10.0`** must be pip-installed into the venv (it was missing on reuse — installs from
pypi.org with the proxy left on; re-verify `torch.version.hip is None` after, it does not clobber torch).

#### Launch command, exactly as run

Run directly with `torchrun` (not the `train_llm_torchtitan.py` wrapper) to honour the assigned
`--master_port 29672` on this shared node — the wrapper hard-codes `--rdzv_endpoint localhost:0`
(ephemeral). Set `PYTHONPATH` to this folder + the torchtitan clone (what the wrapper's `build_env`
does) so `--module recipe_torchtitan` resolves, and export `PYTORCH_ALLOC_CONF=expandable_segments:True`.

```bash
cd .env_torchtitan/torchtitan_src                 # torchrun runs with the clone as CWD
source ../../.env_torchtitan/bin/activate         # here: venv on /dev/shm
export CUDA_VISIBLE_DEVICES=6,7                    # ONLY the two assigned free GPUs
export HF_HOME=/mnt/gsma/gsma/gsma/models
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="<repo>/training/llm/torchtitan:$PWD"
export TITAN_OUT=/dev/shm/h100/out/torchtitan_2gpu

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
shows **both** ranks' stdout (the single-GPU section only had rank 0). TP was not exercised — dp_shard=2
(real FSDP2 optimizer/gradient sharding) was the primary goal and it succeeded; a `tensor_parallel_degree=2`
variant would be the same command with that override and `dp_shard` left at 1.

#### Evidence — the 2-rank FSDP2 mesh, on both ranks (Run A)

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

Both ranks print the identical `dp_shard=2` mesh and `Applied FSDP to the model`; loss falls
**7.69873 → 5.92993** across the 30 steps on both. Two per-rank structured logs were written —
`structured_logs/training.global_rank_0.*.jsonl` **and** `...global_rank_1.*.jsonl` — the
independent check that *both* ranks reached the end, not rank 0 alone.

#### Evidence — both physical GPUs 6 AND 7 busy, by PID, sampled from inside Run B

Run B (400 steps) held ~2 min; `nvidia-smi -i 6,7` (the two assigned GPUs only, filtered by PID)
sampled live shows **two distinct ranks, one on each physical GPU**:

```
$ nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader -i 6,7
GPU-e4fe48bc-...-759f964bf823, 1740693, /dev/shm/h100/venv_torchtitan/bin/python3, 4038 MiB   # GPU 6, rank 0
GPU-9eb34eec-...-d1e995e95239, 1740694, /dev/shm/h100/venv_torchtitan/bin/python3, 4036 MiB   # GPU 7, rank 1

$ nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 6,7
6,  93 %, 4047 MiB      # three samples over ~8 s, both GPUs sustained busy:
7, 100 %, 4045 MiB      #   GPU 6: 93/79/85 %   GPU 7: 100/76/83 %
```

Two **different** GPU UUIDs (index 6 and 7), two **different** PIDs (1740693 / 1740694) — real
2-process, 2-GPU parallelism, not one process on two contexts. Sustained ~188k tok/s aggregate,
~112 TFLOP/s and MFU ~11.3% **per rank** at seq 2048 / local batch 16 (steady-state, in the same
ballpark as the §7.3 single-GPU hold), loss falling the whole way:

```
[rank0]:[titan] step: 100  loss:  3.21492  ...  tps: 186,147  tflops: 110.68  mfu: 11.19%
[rank0]:[titan] step: 200  loss:  2.24993  ...  tps: 186,780  tflops: 111.06  mfu: 11.23%
[rank0]:[titan] step: 300  loss:  1.55714  ...  tps: 189,301  tflops: 112.55  mfu: 11.38%
[rank1]:[titan] step: 400  loss:  1.30121  ...  tps: 188,273  tflops: 111.94  mfu: 11.32%   # rank1 == rank0 loss (synchronized FSDP2)
```

Step 400 loss is **identical on both ranks** (`1.30121`) — expected for FSDP2 (a single logical
model, sharded). Per-GPU resident memory 2.41 GiB at this shape (vs 0.5 GiB in the tiny Run A) —
real activation memory, still trivial for the 32M debugmodel on 80 GB cards.

**Shared-node safety, verified:** throughout both runs `nvidia-smi -i 0,1,2,3` showed **only** the
co-tenant PIDs 1273508–11 — nothing of this run leaked onto the production GPUs. `CUDA_VISIBLE_DEVICES=6,7`
is the entire pinning story on NVIDIA; torchrun only ever saw the two assigned GPUs.

#### Evidence — the sharded DCP checkpoint (one shard per rank)

`--checkpoint.enable` flipped the recipe's default-off block on (Run A). DCP `torch_dist` format
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
checkpoints. Total 433 MB, **deleted after capturing this evidence** per the shared-node disk rule
(Run B ran with checkpointing off — logs only).

#### 8-GPU on H100: projected, NOT measured

Not run: the co-tenant production job holds GPUs 0–3, so a full-node 8-way pass was impossible in
this window. Projection rests on two solid legs: (1) torchtitan's single-GPU H100 result (§7.3) and
this 2-GPU result both show **pure `dp_shard` FSDP2 scales cleanly** with no code, dep, or env
change — the mesh degree is just `nproc_per_node`; and (2) the AMD §7.2 8-GPU FSDP2 run measured
per-GPU throughput going *up* (1.34×) at 8-way because FSDP2 shards optimizer+grads 8 ways, freeing
per-GPU memory. Nothing in that is device-specific; on H100 replace `CUDA_VISIBLE_DEVICES=0..7` with
`--nproc_per_node 8 --parallelism.data_parallel_shard_degree 8` once the node frees. For the real 8B
`cpt_qwen3_8b`, FSDP2 across ≥2 ranks is *required* on 80 GB cards anyway (§ VRAM note).

#### Deviations / notes specific to the 2-GPU run

- **Nothing in the recipe, deps, or environment changed** vs single-GPU besides `nproc_per_node 2`
  and the (redundant) explicit `dp_shard 2` override. NCCL was the collective backend with zero
  configuration across both ranks (RCCL was the AMD drop-in). No batch reduction, no OOM, no hang.
- **`torchao==0.10.0` had to be reinstalled** into the reused venv (was absent). It does not clobber
  the torch nightly.
- Ran `torchrun` **directly** rather than through `train_llm_torchtitan.py`, solely to pin
  `--master_port 29672` (the wrapper forces an ephemeral rdzv port). The wrapper works fine for
  2-GPU too (`--ngpu 2`); the direct form is only for shared-node port discipline. Everything else
  (PYTHONPATH, `PYTORCH_ALLOC_CONF`) mirrors the wrapper's `build_env`.
- The `WARNING - Dataset domain_corpus is being re-looped (epoch N)` lines are expected (the loader
  is `infinite=True` over the 10-row sample) and benign for a smoke.

#### Not tested here

Everything in §7.1's "Not tested" list, plus: any multi-GPU layout on H100 (deferred, above), the
8B/Llama/SFT configs on H100, `torch.compile`, and float8/MXFP8/NVFP4 (NVIDIA-hardware features —
these are the *one* area where H100 could eventually show something AMD cannot, but none are
wired into this recipe).

## 8. Notes

- **The Python file is a launcher, not a trainer.** torchtitan's entrypoint is
  `python -m torchtitan.train`, driven by `torchrun`. `train_llm_torchtitan.py` validates
  that `--titan-repo` really is a torchtitan checkout (it looks for `torchtitan/train.py`),
  builds the same `torchrun` invocation upstream's `run_train.sh` builds, sets
  `PYTORCH_ALLOC_CONF=expandable_segments:True` the way `run_train.sh` does, prepends this
  folder to `PYTHONPATH`, and execs. Failures surface as the child's exit code.
- **Config is Python, and that is deliberate upstream.** `--module`/`--config` name a
  function; the frozen `--section.option` flags still work and still take precedence, which
  is why `--extra --training.steps 200` is a usable override without touching the recipe.
  This is also why `recipe_torchtitan.py` has no `parse_args()` — it is a config module the
  framework imports, not a script.
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

Everything in `train_llm_torchtitan.py` and the flag table above was read directly from
upstream `main` (`run_train.sh`, `torchtitan/config/README.md`, `torchtitan/trainer.py`,
`torchtitan/config/configs.py`, `torchtitan/components/checkpointer/dcp.py`, the per-model
`config_registry.py` files, `docs/checkpoint.md`, `docs/datasets.md`, and the
`scripts/checkpoint_conversion/` scripts). Two things are less certain:

1. ~~**The local-dataset registration**~~ **RESOLVED (2026-08-19, commit `03be241`).** The
   `DatasetConfig` + `DATASETS` registration in `recipe_torchtitan.py` imports and works as
   written: `from torchtitan.hf_datasets.text_datasets import DatasetConfig, DATASETS` is
   valid (`text_datasets` re-exports `DatasetConfig` from `torchtitan.hf_datasets`), and the
   MI355X runs in §7.1 logged
   `Preparing domain_corpus dataset from .../data/OTel_LLM_sample_10.jsonl` and trained off
   it. Note the loader defaults to `infinite=True`, so the 10-row sample loops rather than
   ending the run — fine for a smoke, misleading for an epoch count.
2. **`torchtitan.experiments`** currently holds `forge` (the `ForgeEngine` that torchforge
   builds its RL/post-training stack on), `rl` (TitanRL), `graph_trainer`, `torchft`, and
   `transformers_modeling_backend`. SFT itself is **not** in experiments — it is a core
   feature, listed in the upstream README as "Supervised Fine-Tuning (SFT) with chat-formatted
   datasets" and implemented by `ChatDataLoader` in `torchtitan/hf_datasets/text_datasets.py`,
   with a working `sft_debugmodel()` config in `torchtitan/models/llama3/config_registry.py`.
   That is what this folder's SFT recipe is modelled on. The forge/torchforge post-training
   story is real but lives in a separate repo and is out of scope here.

Anything else you want (float8 via `Float8LinearConverter`, MXFP8/NVFP4, async checkpointing,
TorchFT) exists upstream but is not wired up in this recipe.
