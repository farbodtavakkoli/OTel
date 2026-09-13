# `training/llm/ray` — Ray Train orchestration of an HF/TRL fine-tune

## 1. Overview & when to use

Distributed **orchestration** of an ordinary Hugging Face Transformers/TRL fine-tune using
[Ray Train](https://github.com/ray-project/ray) (`ray.train.torch.TorchTrainer`). The SFT
algorithm itself is TRL's `SFTTrainer` with optional LoRA — the same loop as
`../deepspeed/`. **Ray is the scheduler, not the trainer:** `TorchTrainer` + `ScalingConfig`
place one worker per GPU across however many nodes the cluster has and set up the torch
process group; `FailureConfig(max_failures=...)` restarts the worker group from the last
reported checkpoint when a worker or node dies. The script targets the Ray Train V2 API
(ray >= 2.43, default-on since ~2.57).

Files in this folder:
- `train_llm_ray.py` — entrypoint. Defines the per-worker training function and submits it via `TorchTrainer`.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample shipped with this folder (the default `--train_file`).
- `requirements_ray.txt` — dependencies, with the version-matching constraint across nodes called out.
- `readme_ray.md` — this document.

> **Hardware coverage:** **AMD Instinct MI355X (ROCm 7.2.4)** and **NVIDIA H100 80GB
> (CUDA 13.0)**, with no code change between platforms. ROCm needs one environment variable
> (`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`, [§8.1](#81-mi355x-rocm-724)); on CUDA that
> escape hatch is **not** needed — plain `CUDA_VISIBLE_DEVICES` is sufficient
> ([§8.2](#82-h100-cuda-130)). **Multi-node** (`--num_workers` + S3/NFS `--storage_path`) and
> the **fault-tolerance/recovery** paths are documented but untested.

## 2. Install

### NVIDIA

```bash
python3.12 -m venv ~/.venv-ray && source ~/.venv-ray/bin/activate
pip install -r requirements_ray.txt
```

**On CUDA 13 the `torch==2.11.0` pin resolves to a CUDA 13 build on PyPI**, so no
`--index-url` is needed. Install torch first, then the rest of the stack *without*
re-resolving torch, and re-verify torch was not clobbered:

```bash
python3.12 -m venv .env_ray && source .env_ray/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 numpy            # CUDA 13 build, straight from PyPI
pip install 'ray[train]==2.57.0' transformers==5.5.0 trl==0.27.0 peft==0.18.1 \
            datasets==4.3.0 accelerate==1.14.0 python-dotenv 'pyarrow>=23.0.1'
# re-verify torch was not replaced by the second install:
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# -> 2.11.0 13.0 NVIDIA H100 80GB HBM3
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

Do not install `flash-attn`: the script hard-codes `attn_implementation="sdpa"` and exposes no
flag to switch it.

**GPU scoping on CUDA:** set `CUDA_VISIBLE_DEVICES` to the GPU(s) you own and stop there. Do
**not** set `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` — on NVIDIA both Ray Core and Ray
Train scope on the same variable, so the ROCm workaround ([§8.1](#81-mi355x-rocm-724)) is
unnecessary. `HIP_VISIBLE_DEVICES` and its `NOSET` partner are ROCm-only; drop them.

**Install rule:** every node in the cluster needs an *identical*
environment — same Ray version, same Python minor version, same torch/CUDA build. Ray refuses
to connect a worker whose Ray version differs from the head node's, and a torch mismatch
surfaces later as a confusing NCCL or pickling error. Build one venv or container image and
ship it everywhere.

Verify:
```bash
python -c "import ray, torch, trl, transformers, peft; print('imports OK', torch.cuda.device_count(), 'GPUs')"
ray status   # only after a cluster is up
```

**Single node** (the default): the script calls `ray.init()` with no address and starts a
local Ray instance itself. Nothing to set up.

**Multi-node:** start Ray first, then point the script at it with `--ray_address auto`.

```bash
# On the head node:
ray start --head --port=6379

# On each worker node:
ray start --address='<head-ip>:6379'
```

On AWS, `ray up cluster.yaml` with the official cluster launcher provisions and autoscales
the same topology; the training command is unchanged.

### AMD / ROCm

**Works on MI355X (ROCm 7.2.4) with one required environment variable and no code changes.**
You need a ROCm build of PyTorch on every node, and the same identical-environment rule
applies.

```bash
python3.12 -m venv .env_ray && source .env_ray/bin/activate
pip install --upgrade pip

# 1. ROCm torch FIRST, from the ROCm index (NOT the CUDA wheel pinned for NVIDIA).
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0

# 2. Everything else, without letting pip pull the CUDA torch back in.
pip install ray[train]==2.57.0 transformers==5.5.0 trl==0.27.0 peft==0.18.1 \
            datasets==4.3.0 accelerate==1.14.0 python-dotenv 'pyarrow>=23.0.1'
```

Do **not** `pip install flash-attn` — it does not build against ROCm here and the script
does not need it (`attn_implementation="sdpa"`).

Verify (`torch.cuda` is the HIP shim on ROCm; the names are unchanged):
```bash
python -c "import ray, torch, trl, transformers, peft; print(ray.__version__, torch.__version__, torch.cuda.device_count(), 'GPUs')"
# 2.57.0 2.11.0+rocm7.2 2 GPUs
```

**Scoping which GPUs to use — Ray reads `HIP_VISIBLE_DEVICES`, not `ROCR_VISIBLE_DEVICES`**
(Ray warns if you set the latter). Set `HIP_VISIBLE_DEVICES` and Ray autodetects the rest —
no `ray.init(num_gpus=...)` or `ray start --num-gpus` override is needed.

**The one required env var on ROCm:**
```bash
export HIP_VISIBLE_DEVICES=0,1                       # which GPUs Ray may use
export CUDA_VISIBLE_DEVICES=0,1                      # keep the two lists identical on ROCm
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # REQUIRED — see §8.1
```

If your venv's `activate` script pins `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`,
export them **after** sourcing it, or the pins win.
Without `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`, rank 0 trains and **every other rank
dies instantly** with `torch.AcceleratorError: CUDA error: invalid device ordinal`
([§8.1](#81-mi355x-rocm-724)).

## 3. Environment & secrets

Put a `dev.env` **in this folder** with your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_ray.py` loads it via `load_dotenv("dev.env")`, so run the script from inside
this folder (`training/llm/ray/`). `dev.env` is **git-ignored** at the repo root. **Never
commit tokens.**

The run commands below refer to two directories by environment variable — set them to suit
your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # Ray Train --storage_path (checkpoints, results)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

The driver's environment is *not* automatically the workers' environment: the token is read
on the driver and passed through `train_loop_config`, then re-exported inside the training
function. To keep it out of the config, put it in the worker environment instead via
`RunConfig(worker_runtime_env={"env_vars": {"HF_TOKEN": ...}})`.

## 4. Data

This folder ships a working sample: `data/OTel_LLM_sample_10.jsonl` — 10 rows of this
repo's canonical chat JSONL, one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Extra columns (`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
`source_version`) are dropped after the chat template renders each row to text. To train on
your own data, pass `--train_file /path/to/your.jsonl` in the same schema.

No conversion is needed — each row is rendered with the model's own
`tokenizer.apply_chat_template(...)` into a `text` column, which TRL's `SFTTrainer`
tokenizes. A tokenizer with **no** chat template hard-fails at startup rather than training
on silently wrong formatting — use an `-Instruct` checkpoint.

The whole file is loaded on **every** worker. For datasets too large to fit per-worker,
switch the ingest to **Ray Data** and `ray.train.get_dataset_shard("train")`, which streams a
distinct shard to each worker.

The script preflights the first 64 rows for the `messages` contract before reserving any
GPUs, so a malformed file fails in seconds rather than after a cluster spin-up.

## 5. Run

**Smoke test** — single node, the shipped sample, one epoch on however many GPUs you have
(the preflight and Ray startup paths run before any model download):

```bash
python3 train_llm_ray.py --num_workers 1 --num_train_epochs 1
```

> **On ROCm/AMD:** export `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` first, or any run
> with `--num_workers > 1` dies with `invalid device ordinal`. See
> [§8.1](#81-mi355x-rocm-724).

**Full run:**

```bash
nohup python3 train_llm_ray.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_file data/train.jsonl \
  --num_workers 8 \
  --storage_path s3://my-bucket/ray-runs \
  --run_name otel_sft_001 \
  --max_failures 3 \
  --use_lora --lora_r 64 --lora_alpha 128 \
  --batch_size 1 --grad_acc_steps 8 --num_train_epochs 3 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_ray.log 2>&1 &

tail -f train_llm_ray.log
```

Multi-node: add `--ray_address auto` after `ray start`, raise `--num_workers` to the total
GPU count across nodes, and keep `--storage_path` on S3 or NFS.

**What "working" looks like:** the driver logs the cluster resources and the run name, then
Ray places the workers — if it hangs here the cluster cannot yet satisfy `num_workers` GPUs,
which is a scheduling problem, not a training one. Each worker then emits TRL
`{'loss': ...}` lines with a finite `grad_norm`, and at each `--save_steps` boundary a
checkpoint appears under `<storage_path>/<run_name>`. On completion the driver prints the
final metrics, the result path, and the flags to resume that run. To check fault tolerance,
kill a worker process mid-run: the run should restart the worker group and resume from the
last checkpoint.

## 6. Arguments

Every flag `train_llm_ray.py` accepts:

| Arg | Default | Meaning |
|---|---|---|
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | HF repo id or local path to fine-tune |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat `messages` JSONL; the shipped sample by default |
| `--storage_path` | `./ray_results` | Persistent checkpoint storage. **`s3://` or shared NFS for multi-node** |
| `--run_name` | `None` (fresh uuid) | Unique run id. **Reuse the same `(storage_path, run_name)` to resume** |
| `--num_workers` | `8` | Training workers — **one per GPU**, summed across all nodes |
| `--use_gpu` | on | Reserve 1 GPU per worker |
| `--max_failures` | `3` | Worker/node failure retries (`-1` unlimited, `0` disables recovery) |
| `--num_to_keep` | `2` | Checkpoints retained in storage |
| `--ray_address` | `None` | `auto` to attach to a running cluster; omit for local single-node |
| `--max_seq_length` | `2048` | Max tokens per example |
| `--batch_size` | `1` | Per-worker (per-GPU) batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps |
| `--num_train_epochs` | `3` | Number of epochs |
| `--learning_rate` | `2e-4` | Peak learning rate |
| `--logging_steps` | `10` | Log every N steps |
| `--save_steps` | `100` | Report a checkpoint every N steps |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Trade compute for activation memory |
| `--use_lora` | off | Train a LoRA adapter (omit for full fine-tuning) |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | PEFT target modules: `all-linear` or a comma-separated list |

Effective global batch = `batch_size x grad_acc_steps x num_workers`.

## 7. Output

Results are written to `<storage_path>/<run_name>` — the same layout locally, on NFS, or in
S3. That directory holds the reported checkpoints (capped by `--num_to_keep`) and the run
state Ray needs to resume the job.

On completion the driver logs `result.metrics`, `result.path`, and the latest
`result.checkpoint`. To load the trained model:

```python
import os
from transformers import AutoModelForCausalLM
from ray.train.huggingface.transformers import RayTrainReportCallback

with result.checkpoint.as_directory() as ckpt_dir:
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(ckpt_dir, RayTrainReportCallback.CHECKPOINT_NAME)
    )
```

With `--use_lora` the checkpoint holds a LoRA **adapter**, so inference needs the base
model plus the adapter (`PeftModel.from_pretrained`) — the same contract as the other LoRA
trainers in this repo. Keep the `--run_name` the driver prints: with `--storage_path` it is
the pair that both resumes an interrupted run and locates its outputs later.

## 8. Hardware support

- **Ray is a scheduler; hardware support is torch-level.** The model math runs in whatever
  PyTorch build is installed on each node.
- **NVIDIA GPUs** — scope with plain `CUDA_VISIBLE_DEVICES`; no `NOSET` env var (§8.2).
- **AMD GPUs** — scheduled under the same `GPU` resource name, with two caveats:
  1. Device visibility is scoped by **`HIP_VISIBLE_DEVICES`**, *not* `ROCR_VISIBLE_DEVICES`.
  2. Multi-worker runs need `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` (§8.1).

  Requires a ROCm PyTorch build on every node (see the ROCm install lines in
  `requirements_ray.txt`).

### 8.1 MI355X (ROCm 7.2.4)

**This path works with changes.** No edits to `train_llm_ray.py` are needed; the changes are
a ROCm torch wheel plus one required environment variable. Scaling 2 → 8 GPUs is just
`--num_workers 8` and a wider `HIP_VISIBLE_DEVICES` — no `ScalingConfig` override.

Use a small ungated model such as `Qwen/Qwen3-0.6B` for a smoke run; the default
`meta-llama/Llama-3.2-3B-Instruct` is gated and needs `HF_TOKEN`.

#### How Ray detects the AMD GPUs

**Ray autodetects MI355X GPUs with no flags at all** — none of `ray.init(num_gpus=N)`,
`ray start --head --num-gpus=N`, `ROCR_VISIBLE_DEVICES`, or a `num_gpus` override in
`ScalingConfig` is required. The node is tagged `accelerator_type:AMD-Instinct-MI355X-OAM`,
so `ScalingConfig(accelerator_type=...)` works on AMD too.

#### `invalid device ordinal` on every rank but 0

Without `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`, the run fails immediately:

```
ray.train.WorkerGroupError: Training failed due to worker errors:
[Rank 1 Error Snippet]:
  File ".../ray/train/torch/config.py", line 35, in __enter__
    torch.cuda.set_device(device)
torch.AcceleratorError: CUDA error: invalid device ordinal
GPU device may be out of range, do you have enough GPUs?
```

**The rule: export `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` for every ROCm run,
at any worker count.** On ROCm, Ray Core narrows each worker with `HIP_VISIBLE_DEVICES`
while Ray Train binds devices by index into `CUDA_VISIBLE_DEVICES`, so every rank but 0
addresses a GPU its process cannot see. The variable tells Ray Core to skip the narrowing,
and the binding then lands on the right physical GPU. On NVIDIA both steps use the same
variable, so the equivalent `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` is not needed.

#### Expected output

```
INFO - root - Ray cluster resources: {'GPU': 8.0, 'CPU': 256.0,
  'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, ...}
(RayTrainWorker pid=...) Setting up process group for: env:// [rank=0, world_size=8]
(RayTrainWorker pid=...) {'loss': '2.679', 'grad_norm': '12.94', 'entropy': '1.967', 'mean_token_accuracy': '0.5004', 'epoch': '0.5'}
(RayTrainWorker pid=...) {'loss': '0.00548', 'grad_norm': '0.08838', 'entropy': '0.01059', 'mean_token_accuracy': '0.998', 'epoch': '15'}
(RayTrainWorker pid=...) Checkpoint successfully created at: Checkpoint(filesystem=local, path=<storage_path>/<run_name>/checkpoint_...)
```

Loss falls steeply because the shipped 10-row sample is being memorized — expected for a
smoke run. To confirm the workers landed on *distinct* GPUs, sample `rocm-smi --showpids`
mid-run: N worker processes on N different GPUs, each holding comparable VRAM.

One benign log line to expect at higher worker counts: `(PlacementGroupCleaner) Failed to
query Ray Train Controller actor state. State API may be temporarily unavailable. Continuing
to monitor.` — the run proceeds and exits 0.

#### ROCm quirks worth knowing

- **`ROCR_VISIBLE_DEVICES` is the wrong knob for Ray.** Use `HIP_VISIBLE_DEVICES`; Ray logs
  a warning if only `ROCR_VISIBLE_DEVICES` is set.
- **Never set `CUDA_VISIBLE_DEVICES` to an empty string** on ROCm — a plain
  `@ray.remote(num_gpus=1)` task calling `get_devices()` then dies with
  `ValueError: '<id>' is not in list`. Set `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`
  to the same list.
- **`torch.cuda` is the HIP shim.** `torch.cuda.is_available()`, `device_count()`, and
  `cuda:N` device strings all work unchanged; ROCm errors are still spelled "CUDA error".
- **Do not install `flash-attn`.** The script's `attn_implementation="sdpa"` needs no extra
  package on ROCm.
- **Port collisions.** If port 29500 is busy from another job on the box, export a different
  `MASTER_PORT` before launching.
- **Storage.** Prefer `--use_lora` for repeated smoke tests — a full fine-tune keeps
  `--num_to_keep` full checkpoints per run.

### 8.2 H100 (CUDA 13.0)

**This path works on H100 with no env-var workaround.** The only deviations from the MI355X
recipe are CUDA-native: install `torch==2.11.0` from [§2](#2-install) (the pin is a CUDA 13
build on PyPI), and **drop** `HIP_VISIBLE_DEVICES` /
`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`.

#### GPU visibility on CUDA — `RAY_EXPERIMENTAL_NOSET_*` is not needed

```bash
export CUDA_VISIBLE_DEVICES=6     # Ray SEES only GPU 6; it assigns its single worker there.
# do NOT set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES — left unset, everything works.
# (HIP_VISIBLE_DEVICES / RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES are ROCm-only; drop them.)
```

For multiple GPUs, list them all in `CUDA_VISIBLE_DEVICES` and pass `--num_workers N`; the
`NOSET` variable stays unset. Multi-node additionally needs an S3/NFS `--storage_path` (a
local path such as `/dev/shm` is single-node only).

#### Smoke command

If the default `meta-llama/Llama-3.2-3B-Instruct` is gated or uncached, any small ungated
model that ships a chat template works — e.g. `LiquidAI/LFM2.5-350M`.

```bash
export CUDA_VISIBLE_DEVICES=6
# HF_HOME points at the model cache (set above); the model must be fully cached there
export HF_DATASETS_CACHE=/dev/shm/dscache_ray
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1      # cache-first
export MASTER_PORT=29646                            # if the default 29500 is busy

python3 train_llm_ray.py \
  --model_name LiquidAI/LFM2.5-350M \
  --num_workers 1 \
  --batch_size 1 --grad_acc_steps 1 \
  --num_train_epochs 10 \
  --max_seq_length 512 \
  --logging_steps 1 --save_steps 1000000 --num_to_keep 1 --max_failures 0 \
  --storage_path /dev/shm/ray/results --run_name h100_smoke
```

#### Expected output

```
INFO - root - Ray cluster resources: {'node:__internal_head__': 1.0, ..., 'CPU': 96.0,
  'GPU': 1.0, 'accelerator_type:H100': 1.0, ...}
(RayTrainWorker pid=...) Setting up process group for: env:// [rank=0, world_size=1]
(RayTrainWorker pid=...) {'loss': '4.106', ..., 'mean_token_accuracy': '0.3973', 'epoch': '0.1'}
(RayTrainWorker pid=...) {'loss': '0.008682', 'grad_norm': '0.3809', ..., 'epoch': '10'}
(RayTrainWorker pid=...) Checkpoint successfully created at:
  Checkpoint(filesystem=local, path=.../h100_smoke/checkpoint_<timestamp>)
```

## 9. Notes

- **Ray Train V2 is what this targets.** V2 is the default in Ray 2.57; on Ray 2.43–2.56 you
  must export `RAY_TRAIN_V2_ENABLED=1` yourself or you silently get the old implementation.
  Check your version:
  ```bash
  python -c "import ray; print(ray.__version__)"
  ```
  `TorchTrainer.restore()` / `can_restore()` / `restore_from_checkpoint` are **deprecated** in
  V2. Resume by constructing a normal `TorchTrainer` with the *same*
  `RunConfig(storage_path, name)` — which is why `--run_name` exists and why the script prints
  it on exit.

- **Recovery.** Worker-process and worker-node failures are retried by Ray up to
  `FailureConfig(max_failures)` (`--max_failures`; Ray's own default is 0, this script uses 3)
  and resume from the latest checkpoint. A *driver* failure is not retried from inside —
  relaunch the same command and Ray finds the prior run state at `{storage_path}/{name}`. Node
  preemption is counted separately (`max_preemption_failures`, unlimited by default).

- **Checkpointing is what makes recovery real.** `RayTrainReportCallback` reports a checkpoint
  on the HF `save_steps` cadence; on restart the script passes
  `ray.train.get_checkpoint()` to `trainer.train(resume_from_checkpoint=...)`.

- **`--save_steps` alone does not disable checkpointing.** HF `Trainer` still saves once at the
  end of training. The script exposes no `save_strategy` flag; to write nothing at all, set
  `save_strategy="no"` in the `SFTConfig` inside `train_func`, or use `--use_lora` and delete
  afterwards.

- **Use a private `RAY_TMPDIR` on a shared box.** `ray.init()` with no address will *attach to
  an already-running local cluster* if it finds one under the default `/tmp/ray`. Where
  another job may have a Ray head up, export `RAY_TMPDIR=/tmp/<yours>` so this run gets its
  own session, and shut down via the driver — never a blanket `ray stop`, which would kill the
  other cluster too.

- **Persistent storage is a hard multi-node requirement.** Head-node local disk is **not
  supported** across nodes and Ray raises an error at checkpoint time. Use `s3://bucket/path`
  (pyarrow's S3 filesystem handles it, no extra dependency) or a shared mount like EFS/NFS.
  The script warns when `--storage_path` looks local.

- **Build models, datasets, and tokenizers *inside* `train_func`**, not on the driver —
  passing loaded datasets in from outside causes serialization errors. Only small JSON-ish
  config travels, via `train_loop_config`.

- **Do not call `ray.train.torch.prepare_model()` / `prepare_data_loader()` here.** HF
  `Trainer` already owns distribution, so wrapping again double-wraps the model;
  `prepare_trainer()` validates the pairing.

- **`ScalingConfig` is the scaling knob.** `num_workers` + `use_gpu` is the whole story for
  this script. V2 also accepts a `(min_workers, max_workers)` tuple for elastic training, plus
  `resources_per_worker`, `accelerator_type`, and `label_selector` for heterogeneous clusters.
  Not exposed as flags here; edit the `ScalingConfig` call if you need them.

- **No ZeRO/FSDP sharding is configured**, so each worker holds a full model replica. For
  sharding, build an `Accelerator` inside `train_func` (or use `RayDeepSpeedStrategy` for
  Lightning).

- **Deprecated in V2:** `CheckpointConfig(checkpoint_frequency)`, `checkpoint_at_end`, and
  `FailureConfig(fail_fast)` **raise** — they are not used here. If a keyword is rejected,
  check [the Ray Train API reference](https://docs.ray.io/en/latest/train/api/api.html) for
  your installed version.
