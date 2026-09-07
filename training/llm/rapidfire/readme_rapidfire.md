# Setup & usage — `train_llm_rapidfire.py`

## Overview & when to use

[RapidFire AI](https://github.com/RapidFireAI/rapidfireai) is a hyperparallel *experimentation*
layer over Hugging Face TRL: instead of training one config at a time, it runs many
fine-tuning/post-training configs **concurrently on the same GPU(s)** using adaptive
chunk-based scheduling, so you can compare them side by side from the first chunk. It
supports **SFT**, **DPO** and **GRPO** through drop-in replacements for TRL's config
classes (`RFSFTConfig`, `RFDPOConfig`, `RFGRPOConfig`), plus a live dashboard with
Interactive Control Ops (stop / resume / clone-modify, optionally warm-started). Pick this
folder over `../deepspeed/` or `../unsloth/` when the question is *"which of
these N configs is best?"* rather than *"train this one config to convergence"* — the
other trainers answer the second question better. RapidFire AI is an officially documented
TRL integration ([TRL docs: RapidFire AI Integration](https://huggingface.co/docs/trl/main/en/rapidfire_integration)).

Files in this folder:
- `train_llm_rapidfire.py` — entrypoint; expands a sweep into N configs and runs them concurrently.
- `requirements_rapidfire.txt` — dependencies, with the `rapidfireai init --train` caveat.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.
- `readme_rapidfire.md` — this document.
- `dev.env` — **you create this**; holds `HF_TOKEN`. Git-ignored, never committed.

> **Tested topology:** **verified on 1× NVIDIA H100 80GB (CUDA 13.0)** (single-GPU
> SFT sweep; multi-GPU deferred) **and on 1× and 8× AMD MI355X (ROCm 7.2.4).** On
> H100 a 2-config SFT grid trained to 40 total steps concurrently on one GPU with decreasing
> loss and a loadable PEFT adapter — see [NVIDIA H100 (CUDA 13) — attempted](#nvidia-h100-cuda-13--attempted),
> which also documents the CUDA `CUDA_VISIBLE_DEVICES` pinning rule (the **opposite** of ROCm's
> "leave unset"). On AMD, an 8-config SFT grid ran concurrently across all 8 GPUs (one config per
> GPU, 85–88% use and ~15.1GB VRAM on every GPU, 4.8× wall-clock vs the same sweep on 1 GPU); see
> [AMD MI355X (ROCm 7.2) — attempted](#amd-mi355x-rocm-72--attempted) and
> [8-GPU run](#8-gpu-run-8x-mi355x-rocm-724) — note `HIP_VISIBLE_DEVICES` must be left **unset**
> there. The original note follows:
> This folder was written against upstream RapidFire AI
> and TRL documentation for the pinned versions below (rapidfireai 0.16.1) and had not been
> executed when that note was written.
> It targets the repo default of a single node with 8xH100 80GB, but RapidFire AI's whole
> premise is that it also works on one GPU — the scheduler time-slices configs through GPU
> memory, so the same command runs on 1 GPU or 8. Treat every flag below as needing a smoke
> test before a real run.

## Install

### NVIDIA (CUDA)

RapidFire AI requires **Python 3.12.x** specifically (not 3.13), CUDA Toolkit 11.8+, PyTorch
2.8+, and an NVIDIA GPU with compute capability 7.x or 8.x.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_rapidfire.txt

# Pull the fine-tuning/post-training dependency set. This step is NOT optional.
rapidfireai init --train

# Sanity check the environment (Python, GPU/CUDA, ports, packages).
rapidfireai doctor
rapidfireai --version
```

The single most common setup failure is running the wrong `init` variant: `rapidfireai init`
installs the **RAG/evals** dependency set, `rapidfireai init --train` installs the
**fine-tuning** set, and they are not interchangeable. If you ran the wrong one you may need
to recreate the venv.

Upstream currently also instructs `pip uninstall -y hf-xet` after install to work around a
Hugging Face download hang ([xet-core#527](https://github.com/huggingface/xet-core/issues/527)).
Check whether that note is still in the upstream README before applying it.

Then start the service stack (dispatcher, MLflow, dashboard) in a separate terminal:

```bash
rapidfireai start
```

### AMD / ROCm

**Upstream says not supported. It works anyway, with changes.**
See the [AMD MI355X (ROCm 7.2) — attempted](#amd-mi355x-rocm-72--attempted) section below for
the full evidence. Short version: upstream's prerequisites (verified against the
[RapidFire AI README](https://github.com/RapidFireAI/rapidfireai)) do explicitly
require an *NVIDIA GPU using the 7.x or 8.x Compute Capability* and *NVIDIA CUDA Toolkit
11.8+*, and there is no ROCm install path or documentation upstream — but that requirement is
a **documented prerequisite and an install-time NVIDIA assumption, not a hard CUDA dependency
in the training code**. RapidFire AI is a scheduler over TRL, and the whole stack under it
(TRL / PEFT / transformers / torch) runs fine on ROCm.

The one thing that actually breaks is `rapidfireai init --train`, which probes for GPUs using
`nvidia-smi`/`nvcc` only, concludes "no GPU", and force-installs a **CPU/CUDA PyTorch wheel
over your ROCm one**. Reinstall ROCm torch after `init` and the folder runs.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hugging Face token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The script calls `load_dotenv('dev.env')` at import time and reads `HF_TOKEN` from the
environment; it logs a warning (rather than failing) if the token is absent, since ungated
models do not need one. `dev.env` is **git-ignored** — never commit tokens, and never paste
one into a script, a notebook cell, or this readme. RapidFire's own tooling expects
`hf auth login --token ...` as an alternative; either path works, but keep the token in
`dev.env` for consistency with the rest of this repo.

**Dashboard port:** the RapidFire frontend serves on **`8853`** (`http://localhost:8853`).
The other services are `8850` (jupyter), `8851` (dispatcher), `8852` (MLflow) and `8855`
(Ray dashboard). On a remote box, forward at minimum the dashboard port:

```bash
ssh -L 8853:localhost:8853 user@remote-host
```

All ports are overridable via `RF_FRONTEND_PORT`, `RF_MLFLOW_PORT`, `RF_API_PORT`, etc.

## Data

This folder ships a 10-row sample, `data/OTel_LLM_sample_10.jsonl` — telecom-spec chat
records — and `--train_file` defaults to it, so SFT and GRPO smoke tests need no data prep.
Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version`; the last two null in most rows — a trap for strict
schema loaders). The script drops every column the selected trainer does not read before
training, so those extras are harmless here.

All three modes read **JSONL, one object per line**, via `--train_file` (and optional
`--eval_file`). The script converts each schema into what the corresponding TRL trainer wants.

**SFT** — chat messages; the final message is the assistant target (the shipped sample's format):
```json
{"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Define entropy."}, {"role": "assistant", "content": "A measure of disorder."}]}
```
`sft_formatting_func` splits this into TRL's `{"prompt": messages[:-1], "completion": messages[-1:]}`.

**DPO** — preference triples, passed through unchanged (TRL's `DPOTrainer` consumes these keys directly; not derivable from the shipped sample):
```json
{"prompt": "Define entropy.", "chosen": "A measure of disorder.", "rejected": "I don't know."}
```

**GRPO** — either chat `messages` (the shipped sample works directly — the last assistant
turn becomes the gold answer) or a flat prompt/answer pair:
```json
{"prompt": "Natalia sold 48 clips in April and half as many in May. How many total?", "answer": "72"}
```
`grpo_formatting_func` accepts both, wraps the prompt in a system prompt asking for a
`<reasoning>...</reasoning><answer>...</answer>` envelope, and the two reward functions in
the script (`correctness_reward_func`, `format_reward_func`) score the extracted answer and
the format. **Replace `correctness_reward_func` before any real GRPO run** — exact string
match against `answer` is a placeholder, not a real verifier.

## Run

Start the RapidFire services first (`rapidfireai start`), then launch. Each experiment
name must be unique.

Smoke test against the shipped sample (2x2 grid, capped at 5 steps per config):

```bash
python train_llm_rapidfire.py --trainer_type sft --experiment_name sft-smoke-001 --max_steps 5
```

Full SFT sweep example:

```bash
nohup python train_llm_rapidfire.py \
  --trainer_type sft \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/train_chat.jsonl \
  --eval_file data/eval_chat.jsonl \
  --experiment_name sft-sweep-001 \
  --learning_rates 2e-4,5e-5 \
  --lora_r 8,32 \
  --num_chunks 4 \
  > train_llm_rapidfire.log 2>&1 &

tail -f train_llm_rapidfire.log
```

That is a 2x2 grid = **4 configs trained concurrently**. GRPO instead:

```bash
nohup python train_llm_rapidfire.py \
  --trainer_type grpo \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --experiment_name grpo-sweep-001 \
  --learning_rates 5e-6,1e-6 --lora_r 16 \
  --num_generations 8 --beta 0.0 \
  > train_llm_rapidfire.log 2>&1 &

tail -f train_llm_rapidfire.log
```

**What "working" looks like:** the log prints the row count, the number of expanded configs
(`Grid search: 4 configs`), then RapidFire starts its workers and begins cycling configs
through chunks. The real signal is the **dashboard at `http://localhost:8853`**: you should
see one row per config, all advancing together chunk by chunk rather than one finishing
before the next starts, with loss curves diverging early enough to tell configs apart. From
there you can stop a losing config or clone-modify a winning one mid-flight. The run ends
with `Experiment <name> complete.` in the log.

## Arguments

Every flag on `train_llm_rapidfire.py`:

| Argument | Default | What it does |
|---|---|---|
| `--trainer_type` | `sft` | `sft` / `dpo` / `grpo`; selects `RFSFTConfig` / `RFDPOConfig` / `RFGRPOConfig` and the expected data schema. |
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal-LM repo id or local path. |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | JSONL training data; schema depends on `--trainer_type`. |
| `--eval_file` | `None` | Optional JSONL eval split; passed as `eval_dataset`. |
| `--experiment_name` | `rf-posttrain` | Must be unique; also the MLflow experiment name. |
| `--learning_rates` | `2e-4,5e-5` | **Swept.** Comma-separated; one config per value. |
| `--lora_r` | `8,32` | **Swept.** Comma-separated LoRA ranks (`lora_alpha` = 2r). |
| `--lora_dropout` | `0.05` | LoRA dropout, shared by all configs. |
| `--batch_size` / `--grad_acc_steps` | `4` / `2` | Per-device batch and accumulation, shared by all configs. |
| `--num_train_epochs` | `1` | Epochs per config (unless `--max_steps` overrides). |
| `--max_steps` | `-1` | `>0` overrides epochs — use it for a smoke test. |
| `--max_length` | `1024` | Token budget; prompt/completion caps are derived as half this for DPO/GRPO. |
| `--logging_steps` | `2` | Steps between metric logs. |
| `--num_generations` | `8` | GRPO only: completions sampled per prompt. |
| `--beta` | `0.1` | DPO/GRPO KL coefficient. `0.0` in GRPO disables the reference model. |
| `--num_chunks` | `4` | Swap granularity. Higher = more frequent side-by-side comparison, more swap overhead. |
| `--search` | `grid` | `grid` = full cross-product (`RFGridSearch`); `random` = sample `--num_samples` (`RFRandomSearch`). |
| `--num_samples` | `4` | Configs sampled when `--search random`. |
| `--seed` | `42` | Passed to `run_fit` for reproducible chunking/sampling. |

## Output

- **Dashboard / MLflow** (`http://localhost:8853`) is the primary artifact: one tracked run
  per config, with live loss/eval curves and the IC Ops panel. A RapidFire experiment maps
  onto an MLflow experiment of the same name.
- **Checkpoints and run state** are written under RapidFire's home directory, `~/rapidfireai`
  by default: `RF_EXPERIMENT_PATH` (default `~/rapidfireai/rapidfire_experiments`) for
  per-run artifacts and adapters, `RF_LOG_PATH` (default `~/rapidfireai/logs`) for service
  logs, and `RF_DB_PATH` for the SQLite state DB. Override with those environment variables.
- **This script's own stdout** goes to `train_llm_rapidfire.log` via the `nohup` lines above.
- Trained adapters are ordinary PEFT adapters and can be loaded with `peft` or merged for
  serving exactly like the ones produced by the other trainers in this repo.

## Hardware support & evidence

| Hardware | Status | Evidence |
|---|---|---|
| NVIDIA GPU, compute capability 7.x/8.x | Required | Upstream README "Prerequisites": NVIDIA GPU (CC 7.x/8.x), CUDA Toolkit 11.8+, Python 3.12.x, PyTorch 2.8+ — <https://github.com/RapidFireAI/rapidfireai> |
| NVIDIA H100 80GB (Hopper cc9.0), CUDA 13.0 | **Works with changes** | Verified locally on **1 GPU** (multi-GPU deferred): 2-config SFT grid trained to 40 total steps concurrently on one GPU, per-config `train_loss` 1.338→0.733, loadable PEFT adapter saved, `torch 2.13.0+cu130`. Needs a transformers 4.57.6→**5.5.0** + trl→1.10.0 upgrade after `rapidfireai init --train`, and `kernels` **uninstalled** (unpicklable-closure crash). Pin ONE GPU via `CUDA_VISIBLE_DEVICES` (**opposite** of ROCm). See [NVIDIA H100 (CUDA 13) — attempted](#nvidia-h100-cuda-13--attempted) |
| Single GPU up to multi-GPU node | Supported | Upstream: the scheduler time-slices configs through one GPU and auto-distributes across all visible GPUs |
| AMD Instinct MI355X (gfx950), ROCm 7.2 | **Works with changes** | Verified locally on **1 GPU and on all 8**: 8-config SFT grid trained to 200/200 steps concurrently across 8× MI355X (85–88% use, ~15.1GB VRAM each, 4.8× wall-clock vs 1 GPU), `torch 2.11.0+rocm7.2`. Needs the ROCm-torch reinstall after `rapidfireai init --train`, and `HIP_VISIBLE_DEVICES` must be left **unset** for multi-GPU. See [AMD MI355X (ROCm 7.2) — attempted](#amd-mi355x-rocm-72--attempted) and [8-GPU run](#8-gpu-run-8x-mi355x-rocm-724) |
| AMD / ROCm, per upstream | Undocumented | No ROCm path anywhere upstream; prerequisites are NVIDIA-explicit — but the gate is an `nvidia-smi`-only probe in the installer, not a CUDA code dependency |
| CPU-only | RAG/evals mode only (closed-model APIs), not this folder's fit mode | Upstream README overview |

**Other hardware (upstream claims — not verified here):** none beyond NVIDIA CUDA.
Upstream's prerequisites are NVIDIA-explicit (CC 7.x/8.x, CUDA 11.8+); the only
non-NVIDIA mode claimed is CPU-only *evals/RAG against closed-model APIs*, which is not
a training path. No AMD, Intel Gaudi/XPU, Apple MPS, Ascend, TPU, or Trainium claims.

## AMD MI355X (ROCm 7.2) — attempted

**Status: works with changes.** Validated on 1× AMD Instinct MI355X (gfx950, 288GB,
`HIP_VISIBLE_DEVICES=6`), ROCm 7.2.4, Ubuntu, Python 3.12.3, `torch 2.11.0+rocm7.2`,
`rapidfireai 0.16.1`. Four SFT sweeps ran to completion on the GPU, including the full
documented 2×2 = 4-config grid.

> The same stack was subsequently verified across **all 8 MI355X concurrently** — 8 configs,
> one per GPU. That result, its evidence, and the extra ROCm environment rule multi-GPU needs
> are in [8-GPU run (8x MI355X, ROCm 7.2.4)](#8-gpu-run-8x-mi355x-rocm-724) below.

### Summary

| Question | Answer |
|---|---|
| Does it run on AMD? | **Yes** — SFT grid sweeps train end-to-end on the MI355X. |
| Is the NVIDIA requirement a hard CUDA dependency? | **No.** No custom CUDA kernels, no `nvcc` build step, no compute-capability assertion anywhere in the fit path. |
| What kind of gate is it, then? | A **soft NVIDIA-assumption check at install time**: the CLI's only GPU probe is `nvidia-smi`. |
| Changes needed? | **One extra command** — reinstall ROCm torch *after* `rapidfireai init --train`. No upstream source was patched. |

### The exact gate

`rapidfireai init --train` — the step this readme calls "NOT optional" — is where AMD dies.
The chain, all in the installed wheel:

1. `rapidfireai/utils/gpu_info.py:8` — `get_compute_capability()` shells out to
   `nvidia-smi --query-gpu=compute_cap`. On ROCm that raises `FileNotFoundError` → returns `None`.
2. `rapidfireai/cli.py:124` — `get_cuda_version()` tries `nvcc --version`, then `nvidia-smi`.
   Neither exists → returns `(0, 0)`.
3. `rapidfireai/cli.py:302-315` — because the capability came back `None`, it prints
   *"Did not detect GPU compute capability (nvidia-smi unavailable)."* and hard-sets
   `compute_cap = 0.0; cuda_major = 0; cuda_minor = 0`.
4. `rapidfireai/cli.py:~437` — `if not ColabConfig.ON_COLAB and cuda_major == 0:` prints
   **`🎯 Using CPU`** and appends `torch==2.8.0`, `torchvision==0.23.0`, `torchaudio==2.8.0`
   with `--upgrade` from **plain PyPI** (i.e. the CUDA build).

What `rapidfireai init --train` prints on a ROCm host:

```
   Did not detect GPU compute capability (nvidia-smi unavailable).
   Disabling CUDA usage for evaluation dependencies.
🎯 Using CPU
   Installing torch==2.8.0...
✅ Successfully installed torch==2.8.0
```

Net effect: `torch 2.11.0+rocm7.2` → `torch 2.8.0+cu128`, `torch.version.hip = None`,
`torch.cuda.is_available() = False`, `device_count() = 0`. The MI355X becomes invisible.

**First runtime failure in that state** (not a RapidFire error at all — it never reaches the
scheduler, because this script sets `bf16=True`):

```
  File ".../rapidfireai/automl/model_config.py", line 116, in __init__
    base_class.__init__(self, **parent_kwargs)
  File ".../trl/trainer/sft_config.py", line 247, in __post_init__
  File ".../transformers/training_args.py", line 1747, in __post_init__
    raise ValueError(error_message)
ValueError: Your setup doesn't support bf16/gpu.
```

**Nothing else is NVIDIA-gated.** `fit/backend/controller.py:64` is
`self.num_workers = torch.cuda.device_count()`, raising `NoGPUsFoundException` only at zero —
and ROCm torch reports HIP devices through the `torch.cuda` namespace, so it sees the MI355X
normally. `experiment.py:143` uses the same pattern. There is no `get_device_capability()`
check and no CC 7.x/8.x range assertion in the fit path at all. (For the record, ROCm torch
reports `torch.cuda.get_device_capability(0) == (9, 5)` for gfx950, so even a naive numeric
check would pass; a literal "7.x or 8.x" range check would not.)

`rapidfireai doctor` is not useful on AMD — it reports `nvidia-smi: not found`,
`nvcc: not found`, `CUDA Installation: not present`, and even
`⚠️ Torch version not found` for a perfectly good ROCm torch, because
`gpu_info.py:141 get_torch_version()` parses the local version by splitting on `"cu"`, which
throws on `2.11.0+rocm7.2`. Ignore it; it does not block anything.

### Working install (exact commands)

Order matters: ROCm torch must go on **after** `init`, because `init` always clobbers it.

Set these to suit your machine first — the commands below use them throughout:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # RapidFire experiments and logs
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

```bash
cd training/llm/rapidfire
python3 -m venv .env_rapidfire
source .env_rapidfire/bin/activate
export HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6   # your GPU(s)

pip install -U pip setuptools wheel
pip install rapidfireai==0.16.1 python-dotenv==1.2.2

# Pulls the fine-tuning dependency set (trl/peft/transformers/accelerate/ray/...).
# It WILL replace torch with a CPU/CUDA wheel and print "🎯 Using CPU" — expected on AMD.
rapidfireai init --train

# >>> THE AMD REPAIR STEP <<< put ROCm torch back over what init just installed.
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# init also leaves a CUDA `triton` that shadows ROCm's, plus ~15 nvidia-cu12 wheels (~2GB dead weight).
pip uninstall -y triton $(pip list | grep -iE '^nvidia-' | grep -v nvidia-ml-py | awk '{print $1}')
pip install --force-reinstall --no-deps "triton-rocm==3.6.0" \
  --index-url https://download.pytorch.org/whl/rocm7.2

# Re-satisfy constraints that the torch reinstall widened.
pip install "numpy<2.1" "setuptools<80" "fsspec[http]<=2025.3.0" "pillow<12.0.0"

pip check                      # expect: No broken requirements found.
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.11.0+rocm7.2 AMD Instinct MI355X
```

Do **not** `pip install flash-attn` — SDPA is the attention path on ROCm, and this script
never asks for flash-attn anyway.

### Smoke test that passed

`rapidfireai start` is **not** required to train; without it MLflow logging just disables
itself with a warning (`MLflow server not available at http://127.0.0.1:8852`) and the run
proceeds. Start it only if you want the dashboard.

```bash
export HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6   # HF_HOME set above
export RF_EXPERIMENT_PATH=$OUTPUT_DIR/rapidfire/experiments
export RF_LOG_PATH=$OUTPUT_DIR/rapidfire/logs

python train_llm_rapidfire.py --trainer_type sft --model_name Qwen/Qwen3-0.6B \
  --experiment_name rf-amd-final-004 --max_steps 4 \
  --learning_rates 2e-4,5e-5 --lora_r 8 --num_chunks 2 \
  --batch_size 1 --logging_steps 1 --max_length 1024
```

Result: `Grid search: 2 configs`, exit 0, `Experiment rf-amd-final-004 complete.` Losses from
`$RF_LOG_PATH/rf-amd-final-004/training.log`: `1.2115, 2.0322` (run 1) and `4.7497, 1.8312`
(run 2), `grad_norm` 1.6–15.1, `mean_token_accuracy` 0.28–0.34.

The chunk scheduler — the actual point of this folder — interleaves correctly on ROCm.
From `rapidfire.log`:

```
controller.py:1086 | Scheduled run 2 on workers (0,) for chunk 0
controller.py:841  | Run 2 completed steps - 2/4
controller.py:1086 | Scheduled run 1 on workers (0,) for chunk 0
controller.py:841  | Run 1 completed steps - 2/4
controller.py:1086 | Scheduled run 2 on workers (0,) for chunk 1
controller.py:841  | Run 2 completed steps - 4/4
controller.py:1086 | Scheduled run 1 on workers (0,) for chunk 1
controller.py:841  | Run 1 completed steps - 4/4
```

The full documented 2×2 grid also passed (`--learning_rates 2e-4,5e-5 --lora_r 8,32
--max_steps 40 --batch_size 2`): all four runs reached `40/40` steps, exit 0, ~10 min wall
clock, `train_loss` 1.17–1.84, `mean_token_accuracy` up to 0.67.

`rocm-smi` sampled during that sweep (GPU 6):

```
GPU[6] : GPU use (%): 4  | VRAM Total Used Memory (B): 5440901120
GPU[6] : GPU use (%): 59 | VRAM Total Used Memory (B): 14252527616
GPU[6] : GPU use (%): 0  | VRAM Total Used Memory (B): 5549420544
GPU[6] : GPU use (%): 11 | VRAM Total Used Memory (B): 5478850560
```

The VRAM oscillation between ~5.4GB and ~14.3GB *is* RapidFire's chunk-boundary model swap
working on AMD — configs being paged in and out of GPU memory, exactly as designed.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**Status: works.** Validated on all 8× AMD Instinct MI355X (gfx950, 288GB each),
ROCm 7.2.4, Python 3.12.3, `torch 2.11.0+rocm7.2`, `rapidfireai 0.16.1`. An 8-config SFT grid
ran to completion with **all 8 GPUs held simultaneously** — 8/8 at 85–88% utilisation and
~15.1GB VRAM each. No upstream source was patched; the only AMD-specific requirement is an
environment-variable rule (below).

#### Working command

```bash
cd training/llm/rapidfire
source .env_rapidfire/bin/activate
# HF_HOME / OUTPUT_DIR: see the "Set these to suit your machine" block above
export RF_EXPERIMENT_PATH=$OUTPUT_DIR/rapidfire/gpu8/experiments
export RF_LOG_PATH=$OUTPUT_DIR/rapidfire/gpu8/logs

# >>> THE 8-GPU ROCm RULE <<< both must be UNSET (see "GPU visibility on ROCm" below).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
python -c "import torch; assert torch.cuda.device_count()==8, torch.cuda.device_count()"

python train_llm_rapidfire.py \
  --trainer_type sft --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file /path/to/OTel_LLM_sample_800.jsonl \
  --experiment_name rf-gpu8-$(date +%Y%m%d-%H%M%S) \
  --learning_rates 2e-4,1e-4,5e-5,2e-5 --lora_r 8,32 \
  --search grid --num_chunks 4 --max_steps 200 \
  --batch_size 2 --grad_acc_steps 2 --logging_steps 1 --max_length 1024
```

#### How RapidFire distributed the configs — this is not DDP

There is no `torchrun`, no `accelerate`, no process group, and no model sharding. The
controller reads `torch.cuda.device_count()` (`fit/backend/controller.py:64`) and **spawns one
worker process per GPU** — 8 workers here — then a Monte-Carlo makespan scheduler
(`fit/backend/scheduler.py:364 schedule()`) hands each `(run, chunk)` task to whichever worker
is free, so the 8 configs train **concurrently, one per GPU**, and migrate between GPUs at
chunk boundaries as the scheduler rebalances. This is *config-level* parallelism: 8 independent
0.5B LoRA trainings at once, not one job split 8 ways.

Migration is visible in the scheduler log — run 5 moved GPU3→GPU1, run 8 GPU1→GPU3,
run 3 GPU6→GPU5 between chunks:

```
Scheduled run 5 on workers (3,) for chunk 0     Scheduled run 8 on workers (1,) for chunk 0
Scheduled run 5 on workers (1,) for chunk 1     Scheduled run 8 on workers (3,) for chunk 1
Scheduled run 5 on workers (1,) for chunk 2     Scheduled run 8 on workers (3,) for chunk 2
Scheduled run 5 on workers (1,) for chunk 3     Scheduled run 8 on workers (3,) for chunk 3
```

Across the run the 40 scheduling events were distributed **exactly evenly — 5 per worker for
workers 0 through 7**.

#### Sweep size, and why 8 configs

`--learning_rates 2e-4,1e-4,5e-5,2e-5` × `--lora_r 8,32` = **8 configs**. This matters: the
scheduler can only occupy a GPU if it has a config to put on it, so **the documented 2×2 = 4
config default physically cannot fill 8 GPUs** — it would leave 4 idle and look like a
half-broken multi-GPU story. Config count is the unit of parallelism here; size the sweep to
the GPU count.

The shipped `data/OTel_LLM_sample_10.jsonl` (10 rows) is also too small to feed 8 configs, so
this run used that sample **tiled 80× to 800 rows**, regenerated with:

```bash
# NB: the shipped sample's last line has no trailing newline, so a naive
# `rows*80` fuses row 10 onto row 1 and yields 720 corrupt lines, not 800.
python -c "
rows=[l if l.endswith('\n') else l+'\n'
      for l in open('data/OTel_LLM_sample_10.jsonl') if l.strip()]
open('/tmp/OTel_LLM_sample_800.jsonl','w').writelines(rows*80)"
```
 `--max_steps` must exceed the steps
available in one chunk or every run finishes inside chunk 0 and never migrates: 800 rows ÷ 4
chunks ÷ effective batch 4 = **50 steps per chunk**, so `--max_steps 200` forces all four
chunks. (A first attempt at `--max_steps 40` completed correctly on all 8 GPUs but did the
whole run in chunk 0, exercising placement but not the swap machinery.)

#### GPU visibility on ROCm — the one change AMD needs

RapidFire pins each worker to its GPU by setting `os.environ["CUDA_VISIBLE_DEVICES"] =
str(self.worker_id)` inside the worker process (`fit/backend/worker.py:267`). **On ROCm,
`HIP_VISIBLE_DEVICES` outranks `CUDA_VISIBLE_DEVICES`**, measured on this box:

| Environment | `torch.cuda.device_count()` |
|---|---|
| `HIP_VISIBLE_DEVICES=0,1` + `CUDA_VISIBLE_DEVICES=3` | 2 — CUDA var **ignored** |
| `HIP_VISIBLE_DEVICES` unset, `CUDA_VISIBLE_DEVICES=3` | 1 — CUDA var honoured |
| both unset | 8 |

So exporting `HIP_VISIBLE_DEVICES=0,...,7` to "enable all 8 GPUs" **silently breaks the
per-worker pin**: every worker would see all 8 devices and `device_map="auto"` would land every
config on GPU 0 — an 8-worker run quietly serialised onto one GPU. **Leave both variables
unset.** This is RapidFire's equivalent of the `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`
workaround other frameworks need; RapidFire's fit path spawns plain `multiprocessing` workers
rather than Ray actors, so no Ray variable is involved. The pin works because workers are
**spawned**, not forked (`controller.py:49` sets `mp.set_start_method("spawn")`), so each one
sets its `CUDA_VISIBLE_DEVICES` before HIP initialises in that process.

#### Measured evidence

`rocm-smi` was sampled every 5s from *inside* the job for its whole life, alongside
`rocm-smi --showpids` so samples are attributable to this run's own PIDs.

Per-GPU peaks (one 8-config × 200-step experiment, wall **87s**):

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| peak use % | 87 | 85 | 88 | 86 | 86 | 88 | 87 | 85 |
| peak VRAM (GB) | 15.08 | 15.08 | 15.15 | 15.15 | 15.15 | 15.15 | 15.15 | 15.08 |
| scheduling events | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

- **Peak concurrency: 8/8.** Every sample once training was under way showed all 8 GPUs
  holding >1GB VRAM, and `--showpids` reported 9 KFD processes (8 workers + the parent).
- All 8 runs reached **200/200 steps**, advancing in lockstep through the chunk barriers
  (50/200 → 100/200 → 150/200 → 200/200 for all 8 within a few seconds of each other).
- Per chunk-trainer throughput: **6.52 steps/s mean** (min 3.67, max 8.76) across the 32
  chunk-trainers; **~18.4 steps/s aggregate** wall-clock (1600 steps / 87s) including all model
  swap, checkpoint and startup overhead.
- Instantaneous *utilisation* fluctuates between 3 and 8 GPUs ≥20% even though VRAM residency
  is a constant 8/8 — that is the chunk-boundary swap (checkpoint save/load) showing up as
  brief per-GPU idle gaps, not lost GPUs.

#### Comparison against the 1-GPU sweep

The identical sweep pinned to one GPU (`HIP_VISIBLE_DEVICES=0` → `device_count()==1` → the
controller creates exactly **one** worker, and all 40 scheduling events go to worker 0):

| | 1× MI355X | 8× MI355X | ratio |
|---|---|---|---|
| workers created | 1 | 8 | — |
| configs resident at once | 1 | 8 | — |
| scheduling events, by worker | 32, all on worker 0 | 40, evenly 5 per worker 0–7 | — |
| **wall clock, 8 configs × 200 steps** | **421s** | **87s** | **4.8×** |
| training window (first→last chunk completion) | 381s | 47s | **8.1×** |
| aggregate throughput | 3.8 steps/s | 18.4 steps/s | 4.8× |
| per-chunk-trainer throughput | 7.26 steps/s | 6.52 steps/s | 0.90× |
| GPUs ≥1GB VRAM | 1 (GPU0 15.13GB @ 90%; GPU1–7 idle at 0.28GB / 1–2%) | 8 (~15.1GB @ 85–88% each) | — |

Both runs completed all 8 configs to 200/200 steps with `rc=0`. The scaling is **8.1× across
the training window and 4.8× wall-clock** — the gap between the two is fixed overhead
(interpreter start, dataset load, 8 worker spawns, first model materialisation) which is
roughly constant at ~40s and therefore dominates an 87s run far more than a 421s one. Longer
sweeps amortise it; expect real-world scaling to sit between those two figures.

Note that **per-config throughput is essentially unchanged** (7.26 vs 6.52 steps/s, i.e. 8
GPUs are each doing the same work at the same rate, ~10% slower from host-side contention
between 8 concurrent data-loading workers). That is the signature of true config-level
parallelism rather than a sharded job: nothing gets faster, eight things happen at once.

#### Limitations and honest caveats

- **Loss values here are meaningless as a quality signal.** The 800-row file is the 10-row
  sample tiled 80×, so the model memorises it almost immediately (final `train_loss` down to
  ~0.001 with `mean_token_accuracy` 1.0 on several configs). This run measures scheduling,
  placement and throughput — not learning quality. Use real data for a real sweep.
- **SFT only.** DPO and GRPO were not exercised on 8 GPUs.
- **Config-level parallelism only.** No config was larger than one GPU, so RapidFire's FSDP
  path (`req_workers > 1`, multi-worker runs) is still untested on ROCm.
- **`num_chunks` is the swap-frequency knob, and swapping is not free** — the per-chunk model
  save/load is what produces the utilisation gaps above. On 8 GPUs with 8 configs there is no
  memory pressure forcing swaps, so a larger `--num_chunks` buys comparison granularity at a
  throughput cost.
- The dashboard/MLflow stack was not started for this run (training does not need it); ports
  were moved off the defaults (`RF_API_PORT=8951`, `RF_MLFLOW_PORT=8952`,
  `RF_FRONTEND_PORT=8953`, `RF_RAY_PORT=8955`) because the box is shared.
- **Do not relocate `RF_HOME`.** `$RF_HOME/rf_mode.txt` is the install-mode marker written by
  `rapidfireai init --train` (it must read `fit`). Pointing `RF_HOME` at a fresh directory makes
  `get_installed_mode()` return `None`, which defaults to `evals`, and `Experiment(mode="fit")`
  aborts before training with a mode-mismatch `ValueError`. Override `RF_EXPERIMENT_PATH` and
  `RF_LOG_PATH` instead — those are where the bulk goes.

### Gotchas found on AMD

- **`--max_length` too small silently zeroes the loss.** At `--max_length 256` the shipped
  telecom sample's long rows get truncated past the assistant turn, so every label is masked
  and you get `{'loss': 0.0, 'grad_norm': 0.0, 'mean_token_accuracy': 0.0}`. This is a
  data/truncation artifact, **not** a ROCm numeric bug — it reproduces identically on any
  backend. Keep `--max_length` at 1024+ for this dataset.
- **`init` is destructive and re-running it re-breaks the venv.** Any later
  `rapidfireai init --train` will clobber ROCm torch again. Re-run the repair step after.
- **Version drift is expected.** `init --train` installs `trl 0.21.0`, `peft 0.18.1`,
  `transformers 4.57.6`, `datasets 3.6.0`, `accelerate 1.14.0`, `ray 2.49.0` — an older stack
  than the rest of this repo uses (transformers 5.x). That is upstream's pin, and it works on
  ROCm as-is.
- **Only SFT was exercised.** DPO and GRPO were not run on AMD. Nothing in their code paths is
  CUDA-specific, but treat them as unverified here.
- **Multi-GPU is now verified too** — see [8-GPU run](#8-gpu-run-8x-mi355x-rocm-724). The one
  extra rule it adds: leave `HIP_VISIBLE_DEVICES` **unset**, because it outranks the
  `CUDA_VISIBLE_DEVICES` pin RapidFire sets per worker and would collapse every config onto
  GPU 0. FSDP-style multi-worker runs (`req_workers > 1`) remain untested on ROCm.

### If you would rather not fight the installer

`../verl/` has first-party ROCm support and is the safer choice for AMD **RL**
post-training. But for the specific job this folder does — *comparing N post-training configs
side by side* — RapidFire now demonstrably works here, and the closest alternatives on AMD are
`../ray/` (Ray Tune sweeps) or simply running `../peft/` / `../deepspeed/`
once per config sequentially.

## NVIDIA H100 (CUDA 13) — attempted

**Status: works with changes.** Validated on 1× NVIDIA H100 80GB HBM3 (Hopper cc9.0,
`CUDA_VISIBLE_DEVICES=7`), driver 580.173.02, **CUDA 13.0**, Ubuntu, Python 3.12.3,
`torch 2.13.0+cu130`, on an **offline** node (HF Hub 403-blocked, `HF_HUB_OFFLINE=1`).
Single-GPU SFT sweep only; multi-GPU (2, then 8) is **deferred** (production co-tenant on GPUs
0–3). A 2-config LoRA grid on `LiquidAI/LFM2.5-350M` trained to 40 total optimizer steps
concurrently on one GPU with decreasing loss and a loadable adapter.

### Summary

| Question | Answer |
|---|---|
| Does it run on NVIDIA H100? | **Yes** — the SFT config sweep scheduler trains end-to-end on one H100. |
| Cleaner than ROCm? | **Partly.** The init GPU probe SUCCEEDS on CUDA (the ROCm "Using CPU" blocker does **not** occur), and there is no CUDA-only-vLLM-wheel gate. But CUDA needs its **own** two fixes (transformers/trl upgrade + remove `kernels`) that ROCm did not, so it is *different* work, not obviously less. |
| Out-of-the-box (just the pinned reqs)? | **No** — three changes below. |

### What upstream `rapidfireai init --train` does on CUDA (vs ROCm)

Unlike ROCm — where init misdetects the GPU and installs a **CPU** torch — on CUDA 13 init
**correctly detects the GPU**:

```
🎯 Detected CUDA 13.0, using cu129
   Installing torch==2.8.0...
❌ Failed to install torch==2.8.0
   ... Failed to fetch: `https://download.pytorch.org/whl/cu129/torch/` ... tunnel error
```

Two consequences:
1. init maps CUDA 13.0 → the **cu129** index and tries to force **torch==2.8.0**. On this
   **offline** box that download **fails**, so our good **torch 2.13.0+cu130 SURVIVES** untouched.
   On a *net-connected* CUDA box init WOULD downgrade torch to 2.8.0+cu129 — re-run
   `pip install torch` afterwards (the CUDA analogue of ROCm's torch-reinstall step). Always
   re-verify: `python -c "import torch;print(torch.__version__, torch.version.cuda)"`.
2. init still installs the rest of the **fit** stack — `trl 0.21.0, peft 0.18.1,
   transformers 4.57.6, datasets 3.6.0, accelerate 1.14.0`, `faiss-gpu-cu12` — and writes
   `rf_mode.txt = fit`. (The "❌ Failed to copy notebooks … Operation not permitted" at the end
   is a harmless tmpfs-permission gripe about tutorial notebooks; it does not affect training.)

### The two changes CUDA needs (beyond torch)

Both are driven by using a **modern (transformers-5.x-era) cached model** — every small chat
model cached on this offline node (`LiquidAI/LFM2.5-350M` → `Lfm2ForCausalLM` /
`tokenizer_class: TokenizersBackend`; also `gemma4`, `ministral3`) needs transformers 5.x:

1. **Upgrade the init-pinned stack.** transformers 4.57.6 raises
   `ValueError: Tokenizer class TokenizersBackend does not exist` on LFM2.5.
   ```bash
   pip install "transformers==5.5.0" "trl>=0.24.0" "peft>=0.20.0"
   # -> transformers 5.5.0, trl 1.10.0, peft 0.20.0, datasets 5.0.1, huggingface-hub 1.28.0
   ```
   trl 0.21.0 is itself **incompatible** with transformers 5.x (it imports
   `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`, removed in tf5 →
   `RuntimeError: Failed to import trl.trainer.dpo_trainer`); bumping trl to 1.10.0 fixes it.

2. **Uninstall `kernels`.** With `kernels` present, transformers 5.x's kernels integration
   attaches a *local closure* `_create_func_module.<locals>.Func` to the model. RapidFire moves
   the model to worker processes through its **shared-memory checkpoint** path
   (`shm_manager._save_full_model` → deep-copy/pickle), which then dies with
   `AttributeError: Can't pickle local object '_create_func_module.<locals>.Func'`.
   ```bash
   pip uninstall -y kernels   # is_kernels_available() -> False; model pickles; scheduler runs
   ```
   Note: **pinning** `kernels<0.13` is *not* enough — any importable `kernels` triggers it; it
   must be **absent**. (This is the H100 face of the brief's "kernels breaks transformers" trap.)

`pip check` then warns `rapidfireai … requires datasets==3.6.0 / huggingface-hub<1.0.0` — both
**benign**; rapidfireai imports and trains fine under datasets 5.0.1 + hf-hub 1.28.0.
`kernels` gone, `numpy` pinned `<2.1` (→ 2.0.2), the tree is otherwise clean.

### GPU visibility on CUDA — the OPPOSITE of the ROCm rule

RapidFire has **no `--gpus`/`--num-gpus` knob**. The controller grabs **all** visible GPUs and
spawns one worker per GPU, then each worker self-pins:

```python
# rapidfireai/fit/backend/controller.py:64
self.num_workers: int = torch.cuda.device_count()      # one worker per VISIBLE GPU
# rapidfireai/fit/backend/worker.py:267
os.environ["CUDA_VISIBLE_DEVICES"] = str(self.worker_id)   # each worker re-pins itself
```

On CUDA there is **no `HIP_VISIBLE_DEVICES`**, and `CUDA_VISIBLE_DEVICES` **is** honoured
(measured on this box):

| env before launch | `torch.cuda.device_count()` | RapidFire behaviour |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` **unset** | **8** | spawns 8 workers → lands on **every** GPU (incl. co-tenant 0–3) |
| `CUDA_VISIBLE_DEVICES=7` | **1** | spawns **1** worker; `worker_id=0` re-pins to CVD="0" = physical GPU 7 |

So to restrict RapidFire to ONE GPU on CUDA you **must** `export CUDA_VISIBLE_DEVICES=<idx>`
before launch. **This is the reverse of ROCm**, where the rule (see
[GPU visibility on ROCm](#gpu-visibility-on-rocm--the-one-change-amd-needs)) is to leave
`HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` **unset** (because HIP outranks CUDA and unset gives
all 8). On CUDA, unset gives all 8 too — but here the *fix is to set it*, not unset it. On a
shared node this is also the safety mechanism: `CUDA_VISIBLE_DEVICES=7` is what kept every worker
off the production GPUs 0–3 and the other agents' 4–6.

### Working install (exact commands)

```bash
export PIP_CACHE_DIR=/dev/shm/pipcache
python3 -m venv .env_rapidfire && source .env_rapidfire/bin/activate
pip install -U pip setuptools wheel
pip install torch numpy                       # -> torch 2.13.0+cu130 (NO --index-url on CUDA 13)
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#   -> 2.13.0+cu130 13.0 True

pip install rapidfireai==0.16.1 python-dotenv==1.2.2
rapidfireai init --train                      # detects CUDA 13.0; torch download fails offline (torch survives)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # RE-VERIFY -> 2.13.0+cu130

pip install "transformers==5.5.0" "trl>=0.24.0" "peft>=0.20.0"   # (1) modern-model support
pip uninstall -y kernels                                          # (2) fix unpicklable closure
pip install "numpy<2.1"                                           # -> 2.0.2, pip check clean
```

Verified versions: `torch 2.13.0+cu130`, `transformers 5.5.0`, `trl 1.10.0`, `peft 0.20.0`,
`accelerate 1.14.0`, `datasets 5.0.1`, `rapidfireai 0.16.1`, `bitsandbytes 0.50.1`,
`numpy 2.0.2`, driver 580.173.02.

### Smoke test that passed

Default script model `Qwen/Qwen2.5-0.5B-Instruct` is **not** cached on this offline node → swapped
to the fully-cached **`LiquidAI/LFM2.5-350M`** (chat template present).

```bash
export CUDA_VISIBLE_DEVICES=7                  # <-- pin ONE GPU (see visibility table above)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1             # HF_HOME points at the model cache (set above)
export HF_DATASETS_CACHE=/dev/shm/dscache_rapidfire   # .arrow writes can fail on a network model-cache mount
export RF_EXPERIMENT_PATH=/dev/shm/rapidfire/experiments RF_LOG_PATH=/dev/shm/rapidfire/logs

python train_llm_rapidfire.py \
  --trainer_type sft --model_name LiquidAI/LFM2.5-350M \
  --experiment_name rf-h100-smoke --learning_rates 2e-4,5e-5 --lora_r 8 \
  --num_chunks 2 --max_steps 20 \
  --batch_size 1 --grad_acc_steps 1 --logging_steps 1 --max_length 1024
```

**Expected output** — the scheduler interleaving 2 configs across 2 chunks on ONE GPU, both
to 20/20 (`rapidfire.log`):

```
Scheduled run 2 on workers (0,) for chunk 0    Run 2 completed steps - 3/20
Scheduled run 1 on workers (0,) for chunk 0    Run 1 completed steps - 3/20
Scheduled run 2 on workers (0,) for chunk 1    Run 2 completed steps - 5/20
...
Scheduled run 2 on workers (0,) for chunk 0    Run 2 completed steps - 20/20
Scheduled run 1 on workers (0,) for chunk 0    Run 1 completed steps - 20/20
```

Loss decreasing — per-config `train_loss` across successive chunk-trainers, lr=2e-4 config
(`training.log`):

```
{'train_loss': '1.338', 'epoch': '1'}     # chunk 0, first pass
{'train_loss': '0.8961', 'epoch': '1'}    # later chunk
{'loss': '0.2771', ... 'mean_token_accuracy': '0.8333', 'epoch': '0.6667'}   # final chunk
{'train_runtime': '0.3535', 'train_samples_per_second': '14.15', 'train_loss': '0.7334'}
```

GPU-7-only residency (nvidia-smi `--query-compute-apps=pid,used_memory,gpu_uuid` sampled from
inside the run, matching the worker PID against GPU 7's UUID):

```
<worker pid>, 2012 MiB, GPU-<uuid of GPU 7>   # the worker, only ever on GPU 7
# GPUs 4,5,6 max_used = 0–1 MiB throughout;  GPUs 0–3 held only the co-tenant job's PIDs
```

A second 1-chunk run (`--num_chunks 1 --learning_rates 2e-4`) wrote a **loadable PEFT adapter**
to disk (`…/runs/1/checkpoints/final_checkpoint/`):

```
adapter_model.safetensors  (987 KB)   adapter_config.json   trainer_state.json
# PeftConfig.from_pretrained(...) -> base=LiquidAI/LFM2.5-350M  r=8  alpha=16  q/k/v/o_proj
# log: "Saved final checkpoint to disk for run 1 on chunk 0"
```

### Quirks / anything changed (vs the MI355X recipe)

- **Model swap:** default `Qwen/Qwen2.5-0.5B-Instruct` not cached → `LiquidAI/LFM2.5-350M`.
- **transformers/trl/peft upgraded** past init's pins (4.57.6→5.5.0, trl 0.21→1.10, peft 0.18→0.20)
  — the LFM2.5 tokenizer needs tf5. (ROCm kept the init pins; it used older Qwen models.)
- **`kernels` removed** — mandatory on CUDA to survive RapidFire's shared-memory model transfer;
  this was not needed on ROCm.
- **No flash-attn / no ROCm vLLM-wheel gate.** LFM2 (hybrid conv/attn) ran on **SDPA**; the ROCm
  blocker (a CUDA-only vLLM wheel in the RAG path) is irrelevant to fit mode here.
- **VRAM:** trivial — LFM2.5-350M + LoRA used ~2.0 GB of 80 GB. No OOM pressure at these sizes;
  the co-tenant's 66 GB/GPU on 0–3 is a different job. A real 7B+ sweep would still fit several
  configs per H100.
- **GPU pinning reversed** (set `CUDA_VISIBLE_DEVICES` vs ROCm's leave-unset) — see the table above.
- **Step-count / "0 steps" trap:** with `max_steps` small and `num_chunks>1`, a run can hit its
  step cap on a *non-final* chunk, so the on-disk `final_checkpoint` (`last=True`, gated on
  `chunk_id == num_chunks-1 AND steps >= total_steps`) is **not** written — the adapter lives in
  shared memory instead. Use `--num_chunks 1` (or set `save_strategy="chunk"`) to force a disk
  adapter for a smoke test.

### What a multi-GPU pass would need (deferred)

Not run (production co-tenant on GPUs 0–3). To go to 2 then 8 GPUs on CUDA: point
`CUDA_VISIBLE_DEVICES` at the exact free set (e.g. `=4,5,6,7`) so `device_count()` equals the
worker count you want — RapidFire then spawns one worker per visible GPU (config-level
parallelism) and each self-pins via `worker_id`. Multi-worker **FSDP** per run (`req_workers>1`,
`master_port` from the assigned range) is untested here, as on ROCm. The ROCm "leave HIP unset"
rule does **not** apply on CUDA — there set `CUDA_VISIBLE_DEVICES` to the GPU list explicitly.

## Notes

- **Drop-in TRL configs.** `RFSFTConfig`, `RFDPOConfig` and `RFGRPOConfig` wrap TRL's
  `SFTConfig` / `DPOConfig` / `GRPOConfig` and accept the same parameters, so the knobs you
  already use in `../deepspeed/` carry over unchanged. `RFLoraConfig` likewise wraps
  PEFT's `LoraConfig`. Nothing about the training math changes — only the scheduling.
- **How the sweep expands.** `List([...])` marks a knob as multi-valued. This script builds a
  `List` of `RFLoraConfig`s and nests it inside each `RFModelConfig`, then wraps the
  `RFModelConfig`s in another `List`. `RFGridSearch` takes the cross-product, so
  `2 learning rates x 2 LoRA ranks = 4 runs`. `RFRandomSearch(num_samples=N)` samples instead.
- **Chunk-based scheduling.** Training data is split into `--num_chunks` shards. Every config
  trains on chunk 1, then every config trains on chunk 2, and so on, with models swapped in
  and out of GPU memory at chunk boundaries and checkpointed automatically. That is what
  makes early comparison possible on a single GPU, and it is why `--num_chunks` trades
  comparison frequency against swap overhead (upstream's rule of thumb: start at 4).
- **Column dropping.** The script keeps only the columns the selected trainer reads
  (`messages` for SFT; `prompt`/`chosen`/`rejected` for DPO; `prompt`/`answer`/`messages`
  for GRPO) and removes the rest before training — this is what makes the shipped sample's
  extra metadata columns safe.
- **`create_model` runs in the worker.** RapidFire calls it once per config inside a worker
  process, so it — and the formatting/reward functions — are module-level and must not close
  over argparse state. That is why they are written as plain top-level functions here.
- **IC Ops.** Stop, resume, delete, and clone-modify (with or without warm-starting from the
  parent's weights) are driven from the dashboard, not from this script. The script's job is
  to define and launch the config group; steering happens interactively while it runs.
- **Multi-GPU.** The scheduler distributes independent configs across all visible GPUs
  automatically — no launcher, no `torchrun`, no `accelerate`. On 8xH100 with 4 configs you
  get config-level parallelism for free. **Measured on 8× MI355X** (see
  [8-GPU run](#8-gpu-run-8x-mi355x-rocm-724)): one worker process per GPU, 8 configs resident
  at once, 4.8× wall-clock over the same sweep on 1 GPU. Size the sweep to the GPU count —
  4 configs leave 4 of 8 GPUs idle. For a model too large for one GPU, RapidFire
  supports FSDP via the ordinary `fsdp` / `fsdp_config` fields on the training args; this
  script does not set them.
- **`experiment.end()` in a `finally`.** Workers and GPU state are released even when a run
  crashes; skipping it tends to leave the experiment name locked.

### Upstream API uncertainty

Recorded honestly rather than guessed:

- **Class and method names are verified.** `Experiment`, `run_fit`, `end`, `List`,
  `RFGridSearch`, `RFRandomSearch`, `RFModelConfig`, `RFLoraConfig`, `RFSFTConfig`,
  `RFDPOConfig`, `RFGRPOConfig` and the `trainer_type="SFT"|"DPO"|"GRPO"` values all come
  from TRL's integration page and RapidFire's own tutorial notebooks.
- **Running as a plain script is not explicitly documented.** Every upstream example is a
  Jupyter notebook. Upstream documents that notebooks cannot be run via `python nb.ipynb`
  because of a multiprocessing restriction on spawning child processes; a normal
  `python train_llm_rapidfire.py` process should be fine (the controller runs in the user
  process and spawns workers itself), but this specific invocation is **unverified**. If it
  misbehaves, port the `main()` body into a notebook cell-by-cell.
- **`RFModelConfig` field coverage.** `model_name`, `ref_model_name`, `peft_config`,
  `training_args`, `formatting_func`, `reward_funcs`, `compute_metrics`, `model_type`,
  `model_kwargs`, `tokenizer_kwargs` and `generation_config` are all attested upstream. The
  exact set of *required* vs optional fields per trainer type is not documented in one place.
- **DPO without an explicit reference model.** Upstream's DPO tutorial starts from an
  SFT-trained adapter and sets `model_adapter_name` / `ref_adapter_name`. This script leaves
  those unset and relies on TRL's default (PEFT base model as the implicit reference), which
  matches `../deepspeed/`'s behavior but is not what the upstream notebook shows.
- **`torch_dtype` vs `dtype` in `model_kwargs`.** The tutorial notebooks use `torch_dtype`;
  the TRL integration page uses `dtype`. This script uses `torch_dtype`. Recent
  `transformers` versions have been renaming this argument, so if you hit a warning or a
  `TypeError` here, switch it.
