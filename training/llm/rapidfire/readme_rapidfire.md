# `training/llm/rapidfire` — concurrent config sweeps over TRL with RapidFire AI

[RapidFire AI](https://github.com/RapidFireAI/rapidfireai) is a hyperparallel *experimentation*
layer over Hugging Face TRL: instead of training one config at a time it runs many
fine-tuning/post-training configs **concurrently on the same GPU(s)** using adaptive
chunk-based scheduling. It supports SFT, DPO, and GRPO through drop-in replacements for TRL's
config classes (`RFSFTConfig`, `RFDPOConfig`, `RFGRPOConfig`), plus a live dashboard with
Interactive Control Ops. Pick this folder over `../deepspeed/` or `../unsloth/` when the
question is "which of these N configs is best?" rather than "train this one config to
convergence".

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2.4) at 1 and 8 GPUs · NVIDIA H100 80GB
(CUDA 13.0), with rapidfireai 0.16.1. Both platforms need a repair step after
`rapidfireai init --train`, and their GPU-visibility rules are opposites — see below.

## Files

- `train_llm_rapidfire.py` — entrypoint; expands a sweep into N configs and runs them concurrently.
- `data/OTel_LLM_sample_10.jsonl` — 10-row telecom-spec chat sample.
- `requirements_rapidfire.txt` — dependencies.

## Setup

RapidFire AI requires **Python 3.12.x** specifically (not 3.13) and PyTorch 2.8+.

`rapidfireai init --train` is mandatory — pip alone does not give a working environment — and
it is **destructive**: it clobbers torch and pins an older TRL stack, so repeat the repair
steps after any later `init`. Run `init --train`, not the bare `rapidfireai init`, which
installs the RAG/evals dependency set instead; if you ran the wrong one you may need to
recreate the venv.

Do not install `flash-attn`. If Hugging Face downloads hang, `pip uninstall -y hf-xet`.

### NVIDIA (CUDA 13)

```bash
cd training/llm/rapidfire
python3.12 -m venv .env_rapidfire && source .env_rapidfire/bin/activate
pip install -U pip setuptools wheel
pip install torch numpy                       # the current CUDA 13 build; no --index-url
pip install -r requirements_rapidfire.txt

rapidfireai init --train

# repair 1: init forces torch==2.8.0 from a cu129 index over your CUDA 13 build
pip install torch
python -c "import torch;print(torch.__version__, torch.version.cuda)"

# repair 2: init pins transformers 4.57.6 / trl 0.21.0 / peft 0.18.1, too old for
# transformers-5.x-era models; and any importable `kernels` breaks RapidFire's
# shared-memory model transfer with "AttributeError: Can't pickle local object"
pip install "transformers==5.5.0" "trl>=0.24.0" "peft>=0.20.0"
pip uninstall -y kernels                      # pinning kernels<0.13 is not enough; it must be absent
pip install "numpy<2.1"

rapidfireai doctor
rapidfireai --version
```

`pip check` then warns that `rapidfireai` requires `datasets==3.6.0` / `huggingface-hub<1.0.0`.
Both are benign.

### AMD / ROCm 7.2

Order matters: ROCm torch goes on **after** `init`, because `init` probes for GPUs with
`nvidia-smi`/`nvcc` only, reports CPU use, and installs a CUDA wheel over your ROCm one (first
symptom: `ValueError: Your setup doesn't support bf16/gpu`).

```bash
cd training/llm/rapidfire
python3 -m venv .env_rapidfire && source .env_rapidfire/bin/activate
pip install -U pip setuptools wheel
pip install rapidfireai==0.16.1 python-dotenv==1.2.2

rapidfireai init --train

# put ROCm torch back over what init installed
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# init also leaves a CUDA `triton` shadowing ROCm's, plus ~15 nvidia-cu12 wheels (~2GB)
pip uninstall -y triton $(pip list | grep -iE '^nvidia-' | grep -v nvidia-ml-py | awk '{print $1}')
pip install --force-reinstall --no-deps "triton-rocm==3.6.0" \
  --index-url https://download.pytorch.org/whl/rocm7.2

# re-satisfy constraints the torch reinstall widened
pip install "numpy<2.1" "setuptools<80" "fsspec[http]<=2025.3.0" "pillow<12.0.0"

pip check                      # expect: No broken requirements found.
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.11.0+rocm7.2 AMD Instinct MI355X
```

ROCm keeps `init`'s own transformers/trl pins; the CUDA upgrade above is only needed for
transformers-5.x-era models. Ignore `rapidfireai doctor` on AMD — it reports
`nvidia-smi: not found` and `Torch version not found` for a perfectly good ROCm torch, and
blocks nothing.

### Services, secrets, and ports

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded at import time, warns if missing
rapidfireai start                      # dispatcher, MLflow, dashboard — in a separate terminal
```

`rapidfireai start` is not required to train: without it MLflow logging disables itself with a
warning and the run proceeds. Start it if you want the dashboard.

The frontend serves on **8853** (`http://localhost:8853`); the others are 8850 (jupyter), 8851
(dispatcher), 8852 (MLflow), 8855 (Ray dashboard), all overridable via `RF_FRONTEND_PORT`,
`RF_MLFLOW_PORT`, `RF_API_PORT`. On a remote box forward at least the dashboard:

```bash
ssh -L 8853:localhost:8853 user@remote-host
```

## Data

All three modes read JSONL, one object per line, via `--train_file` (and optional
`--eval_file`). `--train_file` defaults to `data/OTel_LLM_sample_10.jsonl`, so SFT and GRPO
smoke tests need no data prep; extra columns (`unmask`, `flow`, `source_id`, ...) are dropped.

**SFT** — chat messages; the final message is the assistant target (the shipped format):

```json
{"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Define entropy."}, {"role": "assistant", "content": "A measure of disorder."}]}
```

**DPO** — preference triples, passed through unchanged (not derivable from the shipped sample):

```json
{"prompt": "Define entropy.", "chosen": "A measure of disorder.", "rejected": "I don't know."}
```

**GRPO** — either chat `messages` (the shipped sample works directly; the last assistant turn
becomes the gold answer) or a flat pair:

```json
{"prompt": "Natalia sold 48 clips in April and half as many in May. How many total?", "answer": "72"}
```

GRPO prompts are wrapped in a system prompt asking for a
`<reasoning>...</reasoning><answer>...</answer>` envelope, and the script's two reward functions
score the extracted answer and the format. **Replace `correctness_reward_func` before any real
GRPO run** — exact string match is a placeholder, not a verifier.

## Run

Experiment names must be unique. Point the output paths somewhere with room, but **do not
relocate `RF_HOME`**: `$RF_HOME/rf_mode.txt` is the install-mode marker written by
`rapidfireai init --train` and must read `fit`; a fresh `RF_HOME` makes the mode default to
`evals` and `Experiment(mode="fit")` aborts with a mode mismatch. Override these instead:

```bash
export RF_EXPERIMENT_PATH=/path/to/outputs/rapidfire/experiments
export RF_LOG_PATH=/path/to/outputs/rapidfire/logs
export HF_HOME=/path/to/hf_cache
```

Smoke test against the shipped sample (2x2 grid, capped at 5 steps per config):

```bash
python train_llm_rapidfire.py --trainer_type sft --experiment_name sft-smoke-001 --max_steps 5
```

Full SFT sweep — a 2x2 grid, 4 configs trained concurrently:

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

GRPO instead: `--trainer_type grpo --learning_rates 5e-6,1e-6 --lora_r 16 --num_generations 8
--beta 0.0`.

The log prints the row count and `Grid search: N configs`, then cycles configs through chunks;
the dashboard shows one row per config advancing together. The run ends with
`Experiment <name> complete.`

### GPU visibility — opposite rules per platform

RapidFire has no `--gpus` knob: it reads `torch.cuda.device_count()` and spawns **one worker
process per visible GPU**, each self-pinned to its own device. Which variable you set is the
whole story.

| Platform | Rule |
|---|---|
| NVIDIA / CUDA | **Set** `CUDA_VISIBLE_DEVICES` to exactly the GPUs you want (`=7`, `=4,5,6,7`). Unset means all GPUs, including a co-tenant's. |
| AMD / ROCm | **Leave both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` unset** for multi-GPU. To pin one GPU, set both to the same index. |

On ROCm `HIP_VISIBLE_DEVICES` outranks `CUDA_VISIBLE_DEVICES`, so exporting
`HIP_VISIBLE_DEVICES=0,...,7` to "enable all 8 GPUs" silently breaks the per-worker pin: every
worker sees all 8 devices and the whole run serialises onto GPU 0.

Assert the count before launching:

```bash
python -c "import torch; assert torch.cuda.device_count()==8, torch.cuda.device_count()"
```

### Sizing the sweep

**Config count is the unit of parallelism.** The scheduler can only occupy a GPU if it has a
config to put on it, so the default 2x2 = 4-config grid leaves 4 of 8 GPUs idle. Size the grid
to the GPU count (e.g. `--learning_rates 2e-4,1e-4,5e-5,2e-5 --lora_r 8,32` = 8 configs).

**`--max_steps` must exceed the steps available in one chunk**, or every run finishes inside
chunk 0 and never migrates between GPUs. With 800 rows / 4 chunks / effective batch 4 = 50
steps per chunk, `--max_steps 200` forces all four chunks.

The shipped 10-row sample is too small to feed a wide sweep — tile it:

```bash
# the shipped sample's last line has no trailing newline, so a naive rows*80
# fuses row 10 onto row 1 and yields 720 corrupt lines instead of 800
python -c "
rows=[l if l.endswith('\n') else l+'\n'
      for l in open('data/OTel_LLM_sample_10.jsonl') if l.strip()]
open('/tmp/OTel_LLM_sample_800.jsonl','w').writelines(rows*80)"
```

A tiled file memorises almost immediately — use real data for a real sweep.

## Arguments

| Argument | Default | What it does |
|---|---|---|
| `--trainer_type` | `sft` | `sft` / `dpo` / `grpo`; selects the RF config class and expected data schema |
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal-LM repo id or local path |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | JSONL training data |
| `--eval_file` | `None` | Optional JSONL eval split |
| `--experiment_name` | `rf-posttrain` | Must be unique; also the MLflow experiment name |
| `--learning_rates` | `2e-4,5e-5` | **Swept.** Comma-separated; one config per value |
| `--lora_r` | `8,32` | **Swept.** Comma-separated LoRA ranks (`lora_alpha` = 2r) |
| `--lora_dropout` | `0.05` | Shared by all configs |
| `--batch_size` / `--grad_acc_steps` | `4` / `2` | Per-device batch and accumulation, shared |
| `--num_train_epochs` | `1` | Epochs per config unless `--max_steps` overrides |
| `--max_steps` | `-1` | `>0` overrides epochs — use it for a smoke test |
| `--max_length` | `1024` | Token budget; DPO/GRPO prompt/completion caps are half this |
| `--logging_steps` | `2` | Steps between metric logs |
| `--num_generations` | `8` | GRPO only: completions sampled per prompt |
| `--beta` | `0.1` | DPO/GRPO KL coefficient; `0.0` in GRPO disables the reference model |
| `--num_chunks` | `4` | Swap granularity; higher = more frequent comparison, more swap overhead |
| `--search` | `grid` | `grid` = full cross-product; `random` = sample `--num_samples` |
| `--num_samples` | `4` | Configs sampled when `--search random` |
| `--seed` | `42` | Reproducible chunking/sampling |

## Output

- **Dashboard / MLflow** (`http://localhost:8853`) is the primary artifact: one tracked run per
  config with live loss/eval curves and the IC Ops panel (stop, resume, delete, clone-modify).
  A RapidFire experiment maps onto an MLflow experiment of the same name.
- **Checkpoints and run state** go under `~/rapidfireai` by default: `RF_EXPERIMENT_PATH`
  (`~/rapidfireai/rapidfire_experiments`) for per-run artifacts and adapters, `RF_LOG_PATH`
  (`~/rapidfireai/logs`) for service logs, `RF_DB_PATH` for the SQLite state DB. The adapters
  are ordinary PEFT adapters.

## Notes

- **`--max_length` too small silently zeroes the loss.** At `--max_length 256` the shipped
  sample's long rows are truncated past the assistant turn, every label is masked, and you get
  `{'loss': 0.0, 'grad_norm': 0.0, 'mean_token_accuracy': 0.0}`. Keep it at 1024+ here.
- **No adapter on disk after a short run.** The on-disk `final_checkpoint` is gated on
  `chunk_id == num_chunks-1 AND steps >= total_steps`, so a small `--max_steps` with
  `--num_chunks > 1` leaves the adapter in shared memory. Use `--num_chunks 1` (or
  `save_strategy="chunk"`) to force a disk adapter for a smoke test.
- **VRAM oscillating during a sweep is the chunk swap, not a leak** — models are moved in and
  out of GPU memory at chunk boundaries.
- **Offline or shared-mount environments.** Export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to
  use only cached weights, and point `HF_DATASETS_CACHE` at a writable filesystem — a network
  mount can reject the `datasets` `.arrow` writes and kill the run. Make sure the model you ask
  for is actually cached; the default `Qwen/Qwen2.5-0.5B-Instruct` may not be.
- For a model too large for one GPU, RapidFire supports FSDP via the ordinary `fsdp` /
  `fsdp_config` fields on the training args; this script does not set them.
