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
these N configs is best?"* rather than *"train this one config to convergence"*.

Files in this folder:
- `train_llm_rapidfire.py` — entrypoint; expands a sweep into N configs and runs them concurrently.
- `requirements_rapidfire.txt` — dependencies, with the `rapidfireai init --train` caveat.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.
- `readme_rapidfire.md` — this document.
- `dev.env` — **you create this**; holds `HF_TOKEN`. Git-ignored, never committed.

> **Coverage:** SFT config sweeps are verified on 1× NVIDIA H100 80GB (CUDA 13.0) and on
> 1× and 8× AMD MI355X (ROCm 7.2.4), with rapidfireai 0.16.1. Both platforms need
> post-`init` repairs (see [Install](#install)), and each has its own GPU-visibility rule —
> they are **opposites**, see [GPU visibility](#gpu-visibility--opposite-rules-per-platform).
> DPO and GRPO are untested here, as is RapidFire's multi-worker FSDP path
> (`req_workers > 1`).

## Install

### NVIDIA (CUDA)

RapidFire AI requires **Python 3.12.x** specifically (not 3.13), CUDA Toolkit 11.8+, PyTorch
2.8+, and an NVIDIA GPU with compute capability 7.x or 8.x.

```bash
python3.12 -m venv .env_rapidfire
source .env_rapidfire/bin/activate
pip install -U pip setuptools wheel
pip install torch numpy                       # -> the current CUDA 13 build (NO --index-url on CUDA 13)
pip install -r requirements_rapidfire.txt

# Pull the fine-tuning/post-training dependency set. This step is NOT optional.
rapidfireai init --train
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # RE-VERIFY

# (1) modern-model support — see below
pip install "transformers==5.5.0" "trl>=0.24.0" "peft>=0.20.0"
# (2) fix RapidFire's unpicklable-closure crash — see below
pip uninstall -y kernels
pip install "numpy<2.1"                       # -> 2.0.2, pip check clean

# Sanity check the environment (Python, GPU/CUDA, ports, packages).
rapidfireai doctor
rapidfireai --version
```

**Two repairs CUDA needs after `init`:**

1. **`init` forces `torch==2.8.0`** from a cu129 index over a good CUDA 13 torch. Re-run
   `pip install torch` afterwards and re-verify with the one-liner above.
2. **Upgrade the init-pinned stack and remove `kernels`.** `init --train` pins
   `transformers 4.57.6` / `trl 0.21.0` / `peft 0.18.1`, which fail on transformers-5.x-era
   models (`ValueError: Tokenizer class TokenizersBackend does not exist`,
   `RuntimeError: Failed to import trl.trainer.dpo_trainer`). With `kernels` installed,
   RapidFire's shared-memory model transfer dies with
   `AttributeError: Can't pickle local object`. **Pinning `kernels<0.13` is not enough — any
   importable `kernels` triggers it; it must be absent.**

`pip check` then warns that `rapidfireai` requires `datasets==3.6.0` /
`huggingface-hub<1.0.0`; both are benign — rapidfireai imports and trains fine under
datasets 5.0.1 + hf-hub 1.28.0.

The single most common setup failure is running the wrong `init` variant: `rapidfireai init`
installs the **RAG/evals** dependency set, `rapidfireai init --train` installs the
**fine-tuning** set, and they are not interchangeable. If you ran the wrong one you may need
to recreate the venv. **`init` is destructive: re-running it re-breaks the venv**, so repeat
the repairs after any later `init --train`.

If Hugging Face downloads hang, run `pip uninstall -y hf-xet`.

Then start the service stack (dispatcher, MLflow, dashboard) in a separate terminal:

```bash
rapidfireai start
```

### AMD / ROCm

**Upstream says not supported. It works anyway, with changes.** No upstream source needs
patching; the only thing that breaks is `rapidfireai init --train`, which probes for GPUs
with `nvidia-smi`/`nvcc` only, prints `🎯 Using CPU`, and force-installs a **CPU/CUDA
PyTorch wheel over your ROCm one** (first failure you see:
`ValueError: Your setup doesn't support bf16/gpu`).

Order matters: ROCm torch must go on **after** `init`, because `init` always clobbers it.

```bash
cd training/llm/rapidfire
python3 -m venv .env_rapidfire
source .env_rapidfire/bin/activate

pip install -U pip setuptools wheel
pip install rapidfireai==0.16.1 python-dotenv==1.2.2

# Pulls the fine-tuning dependency set (trl/peft/transformers/accelerate/ray/...).
# It WILL replace torch with a CPU/CUDA wheel and print "🎯 Using CPU" — expected on AMD.
rapidfireai init --train

# The AMD repair step: put ROCm torch back over what init just installed.
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
never asks for flash-attn anyway. ROCm keeps `init`'s own transformers/trl pins (the CUDA
upgrade above is only needed for transformers-5.x-era models).

`rapidfireai doctor` is **not useful on AMD** — it reports `nvidia-smi: not found`,
`CUDA Installation: not present` and `⚠️ Torch version not found` for a perfectly good
ROCm torch. Ignore it; it blocks nothing.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hugging Face token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The script calls `load_dotenv('dev.env')` at import time; a missing token only warns, since
ungated models do not need one. `dev.env` is **git-ignored** — never commit tokens.
`hf auth login --token ...` works as an alternative.

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
Its extra columns (`unmask`, `flow`, `source_id`, ...) are harmless: the script drops every
column the selected trainer does not read before training.

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
name must be unique. `rapidfireai start` is **not** required to train — without it MLflow
logging disables itself with a warning (`MLflow server not available at
http://127.0.0.1:8852`) and the run proceeds; start it only if you want the dashboard.

Point the output paths somewhere with room, but **do not relocate `RF_HOME`**:
`$RF_HOME/rf_mode.txt` is the install-mode marker written by `rapidfireai init --train`
(it must read `fit`). Pointing `RF_HOME` at a fresh directory makes `get_installed_mode()`
return `None`, which defaults to `evals`, and `Experiment(mode="fit")` aborts with a
mode-mismatch `ValueError`. Override these instead:

```bash
export RF_EXPERIMENT_PATH=/path/to/outputs/rapidfire/experiments
export RF_LOG_PATH=/path/to/outputs/rapidfire/logs
export HF_HOME=/path/to/hf_cache
```

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
(`Grid search: 4 configs`), then starts workers and cycles configs through chunks; the
**dashboard at `http://localhost:8853`** shows one row per config, all advancing together
chunk by chunk. The run ends with `Experiment <name> complete.` in the log.

### GPU visibility — opposite rules per platform

RapidFire has **no `--gpus`/`--num-gpus` knob**: it reads `torch.cuda.device_count()` and
spawns **one worker process per visible GPU**, each self-pinned to its own device. Which
environment variable you must set is therefore the whole story, and it differs by platform:

| Platform | Rule |
|---|---|
| **NVIDIA / CUDA** | **Set** `CUDA_VISIBLE_DEVICES` to exactly the GPUs you want (`=7` for one, `=4,5,6,7` for four). Unset means all GPUs, including a co-tenant's. |
| **AMD / ROCm** | **Leave both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` unset** for multi-GPU. |

On ROCm, `HIP_VISIBLE_DEVICES` **outranks** `CUDA_VISIBLE_DEVICES`:

| Environment | `torch.cuda.device_count()` |
|---|---|
| `HIP_VISIBLE_DEVICES=0,1` + `CUDA_VISIBLE_DEVICES=3` | 2 — CUDA var **ignored** |
| `HIP_VISIBLE_DEVICES` unset, `CUDA_VISIBLE_DEVICES=3` | 1 — CUDA var honoured |
| both unset | all GPUs |

So exporting `HIP_VISIBLE_DEVICES=0,...,7` to "enable all 8 GPUs" **silently breaks the
per-worker pin**: every worker sees all 8 devices and the whole run is serialised onto GPU
0. To pin ROCm to a *single* GPU, set `HIP_VISIBLE_DEVICES=<idx> CUDA_VISIBLE_DEVICES=<idx>`.

Assert the count you expect before launching:

```bash
python -c "import torch; assert torch.cuda.device_count()==8, torch.cuda.device_count()"
```

### Sizing the sweep

**Config count is the unit of parallelism.** The scheduler can only occupy a GPU if it has
a config to put on it, so the documented 2×2 = 4-config default physically cannot fill 8
GPUs — it would leave 4 idle. Size the grid to the GPU count (e.g.
`--learning_rates 2e-4,1e-4,5e-5,2e-5 --lora_r 8,32` = 8 configs for 8 GPUs).

**`--max_steps` must exceed the steps available in one chunk**, or every run finishes
inside chunk 0 and never migrates between GPUs. With 800 rows ÷ 4 chunks ÷ effective batch
4 = 50 steps per chunk, `--max_steps 200` forces all four chunks.

The shipped `data/OTel_LLM_sample_10.jsonl` (10 rows) is too small to feed a wide sweep;
tile it:

```bash
# NB: the shipped sample's last line has no trailing newline, so a naive
# `rows*80` fuses row 10 onto row 1 and yields 720 corrupt lines, not 800.
python -c "
rows=[l if l.endswith('\n') else l+'\n'
      for l in open('data/OTel_LLM_sample_10.jsonl') if l.strip()]
open('/tmp/OTel_LLM_sample_800.jsonl','w').writelines(rows*80)"
```

A tiled file memorises almost immediately — use real data for a real sweep.

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

## Hardware support

| Hardware | Status |
|---|---|
| NVIDIA H100 80GB (Hopper cc9.0), CUDA 13.0 | **Works with changes** — transformers/trl upgrade + `kernels` uninstalled after `init --train`; pin GPUs with `CUDA_VISIBLE_DEVICES` |
| AMD Instinct MI355X (gfx950), ROCm 7.2 | **Works with changes** — ROCm-torch reinstall after `init --train`; leave `HIP_VISIBLE_DEVICES` **unset** for multi-GPU |
| Single GPU up to multi-GPU node | Supported — one config resident per visible GPU |
| CPU-only | RAG/evals mode only, not this folder's fit mode |

## Notes

- **How the sweep expands.** `List([...])` marks a knob as multi-valued; `RFGridSearch`
  takes the cross-product, so `2 learning rates x 2 LoRA ranks = 4 runs`.
  `RFRandomSearch(num_samples=N)` samples instead.
- **Chunk-based scheduling.** Training data is split into `--num_chunks` shards; every
  config trains chunk 1, then every config trains chunk 2, with models swapped in and out of
  GPU memory at chunk boundaries (start at 4). VRAM oscillating during a sweep is that swap,
  not a leak.
- **`create_model` runs in the worker**, so it — and the formatting/reward functions — must
  be module-level and must not close over argparse state.
- **IC Ops.** Stop, resume, delete, and clone-modify are driven from the dashboard, not from
  this script.
- **Multi-GPU.** No launcher, no `torchrun`, no `accelerate`: one worker process per GPU,
  one config resident per GPU. Size the sweep to the GPU count (see
  [Sizing the sweep](#sizing-the-sweep)) — 4 configs leave 4 of 8 GPUs idle. For a model
  too large for one GPU, RapidFire supports FSDP via the ordinary `fsdp` / `fsdp_config`
  fields on the training args; this script does not set them, and that multi-worker path
  (`req_workers > 1`) is untested here.
- **`--max_length` too small silently zeroes the loss.** At `--max_length 256` the shipped
  sample's long rows are truncated past the assistant turn, so every label is masked and you
  get `{'loss': 0.0, 'grad_norm': 0.0, 'mean_token_accuracy': 0.0}`. Keep `--max_length` at
  1024+ for this dataset.
- **No adapter written to disk.** With a small `--max_steps` and `--num_chunks > 1`, a
  run can hit its step cap on a *non-final* chunk, so the on-disk `final_checkpoint`
  (gated on `chunk_id == num_chunks-1 AND steps >= total_steps`) is never written — the
  adapter stays in shared memory. Use `--num_chunks 1` (or set `save_strategy="chunk"`) to
  force a disk adapter for a smoke test.
- **Offline / shared-mount environments.** Export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  to use only cached weights, and point `HF_DATASETS_CACHE` at a writable filesystem — a
  network or read-only-ish model-cache mount can reject the `datasets` `.arrow` writes and
  kill the run. Pick a model that is actually cached; the script's default
  `Qwen/Qwen2.5-0.5B-Instruct` may not be.
- **`experiment.end()` in a `finally`.** Workers and GPU state are released even when a run
  crashes; skipping it tends to leave the experiment name locked.
- **`torch_dtype` vs `dtype` in `model_kwargs`.** This script uses `torch_dtype`; recent
  `transformers` versions have been renaming it, so if you hit a warning or a `TypeError`
  here, switch to `dtype`.
