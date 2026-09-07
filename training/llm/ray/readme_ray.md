# `training/llm/ray` — Ray Train orchestration of an HF/TRL fine-tune

## 1. Overview & when to use

Distributed **orchestration** of an ordinary Hugging Face Transformers/TRL fine-tune using
[Ray Train](https://github.com/ray-project/ray) (`ray.train.torch.TorchTrainer`). The SFT
algorithm itself is TRL's `SFTTrainer` with optional LoRA — the same loop as
`../deepspeed/`. **Ray is the scheduler, not the trainer.** Pick this folder when
the hard part is not the loss function but the *cluster*: multi-node scheduling, elastic
GPU placement, and automatic recovery from a dead worker, a dead node, or an AWS spot
reclaim. On a single stable 8xH100 box, the DeepSpeed or Unsloth trainers are simpler and
have one less moving part.

What Ray adds concretely: `TorchTrainer` + `ScalingConfig` place one worker per GPU across
however many nodes the cluster has and set up the torch process group;
`FailureConfig(max_failures=...)` restarts the whole worker group from the last reported
checkpoint when a worker or node dies. The script targets the Ray Train V2 API
(ray >= 2.43, default-on since ~2.57).

Files in this folder:
- `train_llm_ray.py` — entrypoint. Defines the per-worker training function and submits it via `TorchTrainer`.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample shipped with this folder (the default `--train_file`).
- `requirements_ray.txt` — dependencies, with the version-matching constraint across nodes called out.
- `readme_ray.md` — this document.

> **Tested topology:**
> - **NVIDIA H100 80GB (CUDA 13.0, driver 580.173.02) — works, single GPU, no
>   env-var workaround.** On CUDA the `RAY_EXPERIMENTAL_NOSET_*` escape hatch is **NOT
>   needed**: set `CUDA_VISIBLE_DEVICES` to the GPU(s) you own and Ray's real CUDA device
>   manager handles the rest. Verified at **1 GPU** ([§8.2](#82-h100-cuda-130))
>   — `--num_workers 1` placed one worker on the assigned GPU (`world_size=1`), TRL SFT gave
>   finite decreasing loss (4.11 → 0.009 over 100 steps) and wrote a loadable checkpoint.
>   Multi-GPU on H100 is **deferred** (the box's other GPUs were in use); see §8.2 for what a
>   2/8-GPU pass needs.
> - **AMD Instinct MI355X (ROCm 7.2.4) — works with one required env var**
>   (`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`; without it every run past rank 0 dies
>   with `invalid device ordinal`). Verified at **2 GPUs**
>   ([§8.1](#81-mi355x-rocm-724)) and then at the full **8 GPUs**
>   ([8-GPU run](#8-gpu-run-8x-mi355x-rocm-724)) — `--num_workers 8`
>   placed 8 workers on 8 distinct GPUs at `world_size=8` with no code change. TRL SFT
>   produced finite decreasing loss and checkpointing/resume metadata was written.
>
> Everything **multi-node** (scale-out via `--num_workers` + S3/NFS `--storage_path`) and the
> **fault-tolerance/recovery** paths remain **documented-but-unverified** on both platforms.

## 2. Install

### NVIDIA

```bash
python3.12 -m venv ~/.venv-ray && source ~/.venv-ray/bin/activate
pip install -r requirements_ray.txt
```

> **Tested on H100 / CUDA 13.0:** the `torch==2.11.0` pin has **no cu130 wheel**.
> On a CUDA-13 box install the current stable torch instead — `pip install torch numpy` gives
> **`torch 2.13.0+cu130`** — then install the rest of the stack (ray/transformers/trl/…)
> *without* re-resolving torch, and re-verify torch was not clobbered. Full recipe, versions
> and evidence in [§8.2](#82-h100-cuda-130).

**The one install rule that matters:** every node in the cluster needs an *identical*
environment — same Ray version, same Python minor version, same torch/CUDA build. Ray
refuses to connect a worker whose Ray version differs from the head node's, and a torch
mismatch shows up later as a confusing NCCL or pickling error. Build one venv or container
image and ship it everywhere.

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

**Tested and working on 2x MI355X (ROCm 7.2.4) — with one required environment variable.**
Ray itself is hardware-agnostic scheduling; the training math is torch's, so hardware
support is **torch-level**: you need a ROCm build of PyTorch on every node, and the same
identical-environment rule applies. The script itself needed **no code changes** — it has
no CUDA-specific code (`bf16` + `sdpa` attention, no tf32, no FA2), which is why it ported
cleanly.

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

**Scoping which GPUs to use — Ray reads `HIP_VISIBLE_DEVICES`, not `ROCR_VISIBLE_DEVICES`.**
Ray 2.57's AMD accelerator manager (`ray/_private/accelerators/amd_gpu.py`) declares
`HIP_VISIBLE_DEVICES` as its visible-devices variable and actively warns if you set
`ROCR_VISIBLE_DEVICES` instead. Set `HIP_VISIBLE_DEVICES` and Ray autodetects the rest —
no `ray.init(num_gpus=...)` or `ray start --num-gpus` override is needed.

**The one required env var on ROCm:**
```bash
export HIP_VISIBLE_DEVICES=5,6                       # which physical GPUs Ray may use
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # REQUIRED — see §8.1
```
Without the second variable, rank 0 trains and **every other rank dies instantly** with
`torch.AcceleratorError: CUDA error: invalid device ordinal`. The root cause is a genuine
Ray-on-ROCm bug, explained in [§8.1](#81-mi355x-rocm-724).

For a first-class AMD path with vendor-tuned kernels, `../primus/` is still the
better starting point; this folder is the right choice when you want Ray's *scheduling*
and fault tolerance on AMD hardware.

## 3. Environment & secrets

Put a `dev.env` **in this folder** with your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_ray.py` loads it via `load_dotenv("dev.env")`, so run the script from inside
this folder (`training/llm/ray/`). `dev.env` is **git-ignored** at the repo root. **Never commit tokens** —
none belong in the source. If one was ever committed, rotate it on the Hub.

The run commands below refer to two directories by environment variable — set them to suit
your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # Ray Train --storage_path (checkpoints, results)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

One Ray-specific wrinkle: the driver's environment is *not* automatically the workers'
environment. The token is read on the driver and passed explicitly through
`train_loop_config`, then re-exported inside the training function so each worker can
authenticate to the Hub. If you would rather not route the token through the config, drop
it into the worker environment instead via
`RunConfig(worker_runtime_env={"env_vars": {"HF_TOKEN": ...}})`.

## 4. Data

This folder ships a working sample: `data/OTel_LLM_sample_10.jsonl` — 10 rows of this
repo's canonical chat JSONL, one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version` — the last of these null in most rows); they are
dropped after the chat template renders each row to text. To train on your own data, pass
`--train_file /path/to/your.jsonl` in the same schema — the sample is the default so a
smoke run works out of the box.

No conversion is needed. Inside the training function each row is rendered with the
model's own `tokenizer.apply_chat_template(...)` into a `text` column, which TRL's
`SFTTrainer` tokenizes. Using the model's official template keeps training and inference
formatting identical. A tokenizer with **no** chat template hard-fails at startup rather
than training on silently wrong formatting — use an `-Instruct` checkpoint.

The whole file is loaded on **every** worker, and each worker's HF `Trainer` shuffles with
the same seed. That is fine for the dataset sizes this repo deals with. For datasets too
large to fit per-worker, switch the ingest to **Ray Data** and
`ray.train.get_dataset_shard("train")`, which streams a distinct shard to each worker —
that is Ray's real data story, and it is the natural next step if this script outgrows
`load_dataset`.

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

**What "working" looks like:** the driver logs the cluster resources it sees and the run
name, then Ray places the workers (you will see it wait if the cluster cannot yet satisfy
`num_workers` GPUs — a hang here is a scheduling problem, not a training one). Once
placed, each worker loads the model and TRL emits `{'loss': ...}` lines with a finite
`grad_norm`. At each `--save_steps` boundary a checkpoint is reported and appears under
`<storage_path>/<run_name>`. On completion the driver prints the final metrics, the result
path, and the exact flags to resume that run. To *see* fault tolerance work, kill a worker
process mid-run: the run should pause, restart the worker group, and resume from the last
checkpoint rather than dying.

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

## 8. Hardware support & evidence

- **Ray is a scheduler; hardware support is torch-level.** Ray places workers and sets up
  the process group; the model math runs in whatever PyTorch build is installed on each
  node.
- **NVIDIA GPUs** — listed in Ray's accelerator docs as "Fully tested, supported by the
  Ray team". This is the configuration this folder targets. **Measured here on 1x H100
  (CUDA 13.0): the path works with plain `CUDA_VISIBLE_DEVICES` and no NOSET env var** —
  Ray Core and Ray Train agree on the single `CUDA_VISIBLE_DEVICES` variable, so the ROCm
  `RAY_EXPERIMENTAL_NOSET_*` workaround is unnecessary. Details and evidence in §8.2.
- **AMD GPUs** — listed in the same docs as "Experimental, supported by the community",
  scheduled under the same `GPU` resource name. **Measured here on 2x MI355X: the
  "experimental" label is earned but the path works.** Two corrections to what this
  document previously assumed:
  1. Device visibility is scoped by **`HIP_VISIBLE_DEVICES`**, *not* `ROCR_VISIBLE_DEVICES`
     — Ray 2.57's `amd_gpu.py` explicitly warns you to use the former.
  2. **Scheduling is solid; Ray Train's device *binding* is not.** Ray Core detects and
     places AMD GPUs flawlessly, but Ray Train has no AMD torch device manager, which
     breaks multi-worker runs unless you set
     `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`. Details and evidence in §8.1.

  Requires a ROCm PyTorch build on every node (see the ROCm install lines in
  `requirements_ray.txt`).
- Source: [Ray Accelerator Support documentation](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
  (`doc/source/ray-core/scheduling/accelerators.rst` on `ray-project/ray` `master`).
  Ray's docs also list TPU (fully tested), and Intel GPU/Gaudi,
  AWS Neuron, Huawei Ascend and others as community-experimental.
- **Other hardware (upstream claims — not verified here):** Ray's accelerator docs
  claim Google TPU (fully tested, supported with Google), plus Intel GPU (XPU), Intel
  Gaudi (HPU), AWS Neuron (Trainium/Inferentia), and Huawei Ascend NPU as
  experimental/community-supported resource types. The training math still requires a
  matching torch build per node; none of these were tried here.

### 8.1 MI355X (ROCm 7.2.4)

**This path works with changes.** No edits to `train_llm_ray.py` are needed; the change is
one required environment variable plus a ROCm torch wheel.

**Host:** 8x AMD Instinct MI355X (gfx950, 288GB VRAM each), ROCm 7.2.4, Python 3.12.3, no
NVIDIA GPUs. Two GPUs (physical 5 and 6) were made visible to this test.

**Versions:** `torch 2.11.0+rocm7.2`, `ray 2.57.0`, `trl 0.27.0`, `transformers 5.5.0`,
`peft 0.18.1`, `datasets 4.3.0`, `accelerate 1.14.0`.

**Run command:**
```bash
source .env_ray/bin/activate
export HIP_VISIBLE_DEVICES=5,6
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1   # required, see below
export $(grep -v '^#' dev.env | xargs)                # HF_TOKEN

python3 train_llm_ray.py \
  --model_name Qwen/Qwen3-0.6B \
  --num_workers 2 --batch_size 1 --grad_acc_steps 1 \
  --num_train_epochs 1 --max_seq_length 512 --logging_steps 1 \
  --storage_path $OUTPUT_DIR/ray --run_name mi355x_smoke
```
A small ungated model is used for the smoke so the scheduling path is exercised without a
large download; the default `meta-llama/Llama-3.2-3B-Instruct` is gated and needs `HF_TOKEN`.

#### How Ray detected the AMD GPUs — the important part

**Ray autodetects MI355X GPUs with no flags at all.** `ray.init()` reported:

```
Ray cluster resources: {'GPU': 2.0, 'CPU': 256.0,
  'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, 'memory': 2082611638272.0, ...}
```

Mechanism, confirmed by reading `ray/_private/accelerators/amd_gpu.py` and probing a live
cluster: Ray's AMD accelerator manager shells out to detect ROCm devices (it saw all **8**
on the node), registers them under the ordinary **`GPU`** resource name, and scopes them
using **`HIP_VISIBLE_DEVICES`** — so `HIP_VISIBLE_DEVICES=5,6` yielded exactly `'GPU': 2.0`.
It also tagged the node with `accelerator_type:AMD-Instinct-MI355X-OAM`, which means
`ScalingConfig(accelerator_type=...)` works on AMD too. Ray preserves the **physical** ids:
`ray.get_gpu_ids()` returned `['5']` and `['6']`, not remapped `0`/`1`.

Things that were **not** required, contrary to what you might expect: `ray.init(num_gpus=2)`,
`ray start --head --num-gpus=2`, `ROCR_VISIBLE_DEVICES`, and any `num_gpus` override in
`ScalingConfig`. Autodetection handled all of it.

#### The one real bug: `invalid device ordinal` on every rank but 0

Without `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`, the run fails immediately:

```
ray.train.WorkerGroupError: Training failed due to worker errors:
[Rank 1 Error Snippet]:
  File ".../ray/train/torch/config.py", line 35, in __enter__
    torch.cuda.set_device(device)
torch.AcceleratorError: CUDA error: invalid device ordinal
GPU device may be out of range, do you have enough GPUs?
```

Root cause — **two Ray subsystems scope devices with different environment variables on
ROCm**, and they disagree:

1. Ray **Core** narrows each worker to its assigned GPU by rewriting
   **`HIP_VISIBLE_DEVICES`** (e.g. to `"6"`). That worker's torch now sees exactly **one**
   device, whose only valid ordinal is `0`.
2. Ray **Train** then runs `AcceleratorSetupCallback._share_cuda_visible_devices`
   (`ray/train/v2/_internal/callbacks/accelerators.py`; `TorchBackend.share_cuda_visible_devices`
   is `True`), which overwrites **`CUDA_VISIBLE_DEVICES`** on every worker with the
   node-wide union `"5,6"` so peers can see each other.
3. Device binding goes through `ray/air/_internal/device_manager/nvidia_gpu.py` — **there is
   no AMD device manager in that directory**, only `nvidia_gpu`, `npu`, `hpu`, `cpu`. It
   computes the local ordinal as `CUDA_VISIBLE_DEVICES.split(",").index(my_gpu_id)`, so
   rank 1 gets `"5,6".index("6")` = **1** and calls `set_device(cuda:1)` on a process that
   has only `cuda:0`.

On NVIDIA both steps touch the *same* variable, so it is self-consistent. On ROCm step 1
narrows `HIP_*` while step 2 widens `CUDA_*`, and ROCm's torch honors `HIP_*` — hence the
mismatch. Rank 0 survives only by coincidence (`index("5")` = 0, which is valid).

`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` fixes it by telling Ray Core to skip step 1's
narrowing: every worker keeps `HIP_VISIBLE_DEVICES=5,6`, sees both devices, and Train's
index arithmetic (`0` and `1`) then addresses the right physical GPU. This is the same
escape hatch Ray offers on NVIDIA as `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.

#### Evidence

```
INFO - root - Ray cluster resources: {'GPU': 2.0, 'CPU': 256.0,
  'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, ...}
(RayTrainWorker pid=...) Setting up process group for: env:// [rank=0, world_size=2]
(RayTrainWorker pid=...) {'loss': '2.431', 'grad_norm': '18.88', 'learning_rate': '0', 'entropy': '1.717', 'num_tokens': '887', 'mean_token_accuracy': '0.5209', 'epoch': '0.2'}
(RayTrainWorker pid=...) {'train_runtime': '9.85', 'train_samples_per_second': '1.015', 'train_steps_per_second': '0.508', 'train_loss': '2.843', 'epoch': '1'}
(RayTrainWorker pid=...) Checkpoint successfully created at: Checkpoint(filesystem=local, path=$OUTPUT_DIR/ray/mi355x_smoke2/checkpoint_...)
```

Both GPUs genuinely busy, sampled with `rocm-smi` mid-run (a 150-epoch variant, so the
window was long enough to catch):

```
sample 1  GPU[5]: GPU use (%): 98 | GPU[6]: GPU use (%): 97
          GPU[5]: VRAM Total Used (B): 22076882944 | GPU[6]: VRAM Total Used (B): 18299920384
sample 2  GPU[5]: GPU use (%): 96 | GPU[6]: GPU use (%): 96
```

Loss is finite and decreases as expected when overfitting the shipped 10-row sample
(`train_loss` 2.84 over 1 epoch; 0.0021 after 150 epochs — memorization, which is the
correct behavior for this smoke, not a quality result).

**Paths verified:** Ray GPU autodetection; `TorchTrainer` placing 2 workers, 1 GPU each;
torch process group `world_size=2`; TRL `SFTTrainer` full fine-tune; `--use_lora`
(`lora_r 16`) with `--gradient_checkpointing`; `RayTrainReportCallback` writing checkpoints
to `--storage_path`; the resume flags printed on exit.

**Not verified:** multi-node, real fault-tolerance recovery (no worker was killed mid-run),
S3/NFS `--storage_path`, and NVIDIA (no NVIDIA GPU on this host).

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This scales cleanly from 2 to 8 GPUs with no code change and no new package.**
The single required env var from §8.1 is **still required at 8 workers**; everything else is
just `--num_workers 8`. Ray places 8 workers on 8 **distinct** GPUs, `world_size=8`, finite
decreasing loss, exit code 0, clean teardown.

**Launch command** (source the venv, then override the device pins — see the warning
below — and assert both device counts before training):

```bash
cd training/llm/ray
source .env_ray/bin/activate

# If the venv's activate script pins HIP/CUDA_VISIBLE_DEVICES, override it AFTER sourcing.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1   # STILL REQUIRED at 8 workers
# HF_HOME / OUTPUT_DIR: see the "Set these to suit your machine" block above
export RAY_TMPDIR=/tmp/rayjunk8                       # private Ray session (see below)
set -a; . ./dev.env; set +a                           # HF_TOKEN

python3 train_llm_ray.py \
  --model_name Qwen/Qwen3-0.6B \
  --num_workers 8 \
  --batch_size 1 --grad_acc_steps 1 \
  --num_train_epochs 15 \
  --max_seq_length 512 \
  --logging_steps 1 \
  --save_steps 1000000 --num_to_keep 1 \
  --max_failures 0 \
  --storage_path $OUTPUT_DIR/ray/gpu8 --run_name mi355x_gpu8
```

**Parallelism used.** Pure data parallel — 8 Ray Train workers, 1 GPU each, one full model
replica per worker (no ZeRO/FSDP; see §9). `ScalingConfig(num_workers=8, use_gpu=True)`,
driven entirely by `--num_workers`; there is no torchrun and no `--nproc_per_node` here.
Backend: `TorchBackend` → torch DDP over **RCCL**, torch process group `world_size=8`.
Batch geometry: `batch_size 1 x grad_acc_steps 1 x 8 workers` = **global batch 8** (the
2-GPU run's global batch of 2, scaled by worker count rather than by per-GPU batch). The
shipped 10-row sample over 8 ranks gives 2 optimizer steps/epoch, so 15 epochs = **30 steps**.

**Preflight assertions (both device counts, from the run log):**

```
ASSERT torch.cuda.device_count() = 8
ASSERT ray.cluster_resources()['GPU'] = 8.0
RAY RESOURCES: {'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, 'CPU': 256.0, 'GPU': 8.0}
INFO - root - Launching run 'mi355x_gpu8' | workers=8 | max_failures=0
(RayTrainWorker pid=307688) Setting up process group for: env:// [rank=0, world_size=8]
```

**Training log — finite, decreasing loss, and a clean exit:**

```
(RayTrainWorker pid=307688) {'loss': '2.679', 'grad_norm': '12.94', 'learning_rate': '0', 'entropy': '1.967', 'num_tokens': '3959', 'mean_token_accuracy': '0.5004', 'epoch': '0.5'}
(RayTrainWorker pid=307688) {'loss': '2.43', 'grad_norm': '12.94', 'learning_rate': '0.0002', 'entropy': '1.724', 'num_tokens': '4983', 'mean_token_accuracy': '0.5773', 'epoch': '1'}
(RayTrainWorker pid=307688) {'loss': '0.0322', 'grad_norm': '1.203', 'learning_rate': '0.0001268', 'entropy': '0.0644', 'mean_token_accuracy': '0.9716', 'epoch': '7'}
(RayTrainWorker pid=307688) {'loss': '0.00548', 'grad_norm': '0.08838', 'learning_rate': '5.862e-07', 'entropy': '0.01059', 'mean_token_accuracy': '0.998', 'epoch': '15'}
{'train_runtime': '12.97', 'train_samples_per_second': '11.57', 'train_steps_per_second': '2.314', 'train_loss': '0.4587', 'epoch': '15'}
INFO - root - Training complete. Metrics: {... 'epoch': 15.0, 'step': 30}
=== TRAIN EXIT CODE: 0 ===
```

Step time ≈ **0.43 s/step** (2.314 steps/s) at global batch 8; 11.57 samples/s. Loss falls
2.68 → 0.0055 — memorization of the 10-row sample, which is the correct behavior for this
smoke, not a quality result.

#### Proof that 8 **distinct** GPUs were used

Sampled by a `rocm-smi` loop running *inside* the run script (same shell, same lock window),
not from a separate shell after the fact. All eight GPUs are busy in the **same** sample:

```
mid-run sample:
          GPU use (%):  [0]=53 [1]=51 [2]=52 [3]=49 [4]=47 [5]=47 [6]=45 [7]=45
          VRAM Total Used (B): [0]=13.99e9 [1]=14.29e9 [2]=14.80e9 [3]=14.93e9
                               [4]=14.20e9 [5]=14.11e9 [6]=14.12e9 [7]=14.11e9
```

`rocm-smi --showpids` in the same sample, cross-checked against `pgrep -f 'ray::'` — the
PIDs holding VRAM are exactly this run's own Ray Train workers, **eight processes, one GPU
each**, and nothing else was on the GPUs:

```
KFD process information:
PID       PROCESS NAME      GPU(s)  VRAM USED
307692    ray::RayTrainWo   1       14418214912
307689    ray::RayTrainWo   1       14598569984
307697    ray::RayTrainWo   1       14502100992
307695    ray::RayTrainWo   1       14428700672
307691    ray::RayTrainWo   1       14699233280
307688    ray::RayTrainWo   1       14295961600
307696    ray::RayTrainWo   1       14409826304
307694    ray::RayTrainWo   1       14567112704
```

The same PIDs appear as `(RayTrainWorker pid=...)` in the driver log, and the lowest one
is the rank-0 worker that printed `world_size=8`. **8 processes x ~14.4 GB, and 8 GPUs each
reporting ~14.4 GB** — if two workers had collided on GPU 0 (the classic Ray-on-ROCm
failure) one GPU would show ~29 GB and another ~0. None does.

#### What differed from the 2-GPU run

- `--num_workers 2` → **`--num_workers 8`**; `HIP_VISIBLE_DEVICES=5,6` → **`0,...,7`**. That
  is the entire scaling change. No code edit, no new dependency, no `requirements_ray.txt`
  change, no `ScalingConfig` override (`num_gpus`, `resources_per_worker`) needed.
- **`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` is still required.** The §8.1 root cause
  is worker-count-independent — Ray Train's `nvidia_gpu.py` index arithmetic over the
  node-wide `CUDA_VISIBLE_DEVICES` union gets *worse* with more devices (rank 7 would call
  `set_device(cuda:7)` in a process narrowed to one device). Export it for every run.
- **No placement-group deadlock.** Claiming all 8 GPUs was instant — no wait, no
  `PlacementGroupSchedulingError`. The one oddity in the log is benign chatter from the
  cleanup actor: `(PlacementGroupCleaner) Failed to query Ray Train Controller actor state.
  State API may be temporarily unavailable. Continuing to monitor.` — the run proceeded and
  exited 0.
- **No RCCL init hang** between the 8 workers, and `/dev/shm` (1.2 TB, Ray's object store
  sized at 200 GB) never came close to filling — the object store carries only the small
  `train_loop_config`, not data.
- **Ray preserves physical ids at 8 GPUs too**, so `HIP_VISIBLE_DEVICES` is still the only
  scoping knob; `ROCR_VISIBLE_DEVICES` remains wrong.
- **Throughput** rose from 1.015 to 11.57 samples/s versus the 2-GPU 1-epoch smoke, but the
  two runs use different epoch counts and step geometry — treat it as "8 workers do 4x the
  work per step", not a clean scaling efficiency measurement.

#### Two operational gotchas found at 8 GPUs

1. **`--save_steps` alone does not disable checkpointing.** Even with `--save_steps 1000000`
   over a 30-step run, HF `Trainer` still saved once at the end of training, so
   `RayTrainReportCallback.on_save` reported a **3.4 GB** full-fine-tune checkpoint to
   `--storage_path`. On a disk-constrained box that is a surprise. The script exposes no
   `save_strategy` flag; to truly write nothing you must set `save_strategy="no"` in the
   `SFTConfig` in `train_func`, or use `--use_lora` (~127 MB) and delete after the run.
2. **Use a private `RAY_TMPDIR` on a shared box.** `ray.init()` with no address will *attach
   to an already-running local cluster* if it finds one in the default `/tmp/ray`. On a host
   where another job may have a Ray head up, export `RAY_TMPDIR=/tmp/<yours>` so this run
   gets its own session, and shut down via the driver — never a blanket `ray stop`, which
   would kill the other cluster too. Ray honors `RAY_TMPDIR` via
   `ray._common.utils.get_default_system_temp_dir()`.

**Still not verified at 8 GPUs:** multi-node, real fault-tolerance recovery (`--max_failures`
was set to `0` here so any worker error would surface rather than be retried), S3/NFS
`--storage_path`, and NVIDIA.

#### ROCm quirks worth knowing

- **`ROCR_VISIBLE_DEVICES` is the wrong knob for Ray.** Use `HIP_VISIBLE_DEVICES`; Ray logs
  a warning if only `ROCR_VISIBLE_DEVICES` is set.
- **Never set `CUDA_VISIBLE_DEVICES` to an empty string** on ROCm. Ray Train happens to
  survive an *unset* `CUDA_VISIBLE_DEVICES` (step 2 above rewrites it anyway), but a plain
  `@ray.remote(num_gpus=1)` task calling `get_devices()` dies with
  `ValueError: '6' is not in list` — `nvidia_gpu.py` catches only `IndexError`, not
  `ValueError`. Setting `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` to the same list is
  the safest habit.
- **`torch.cuda` is the HIP shim.** `torch.cuda.is_available()`, `device_count()`, and
  `cuda:N` device strings all work unchanged; ROCm errors are still spelled "CUDA error".
- **Do not install `flash-attn`.** The script's `attn_implementation="sdpa"` is already the
  right choice on ROCm and needs no extra package.
- **Port collisions.** If port 29500 is busy from another job on the box, export a different
  `MASTER_PORT` before launching.
- **Storage.** Full fine-tune checkpoints of even a 0.6B model run ~3.4GB per run at
  `--num_to_keep 2`; prefer `--use_lora` (~127MB) for repeated smoke tests.

### 8.2 H100 (CUDA 13.0)

**Single-GPU smoke, single node, 1x NVIDIA H100 80GB HBM3** (driver 580.173.02, CUDA 13.0,
Hopper cc 9.0, Python 3.12.3). One physical GPU (index 6) was assigned via
`CUDA_VISIBLE_DEVICES=6` on a shared 8-GPU box; the other GPUs were in use, so **multi-GPU
was deferred** (see the end of this section). The `meta-llama/Llama-3.2-3B-Instruct` default
is gated/uncached on this host — the cached, ungated **`LiquidAI/LFM2.5-350M`** (which ships
a chat template) was used instead.

#### Install (CUDA)

The pinned `torch==2.11.0` has **no cu130 wheel**, so the base H100 recipe was used instead:
install the current stable torch (`pip install torch numpy` → **`torch 2.13.0+cu130`**, native
CUDA 13), then the rest of the stack *without* re-resolving torch so it is not clobbered.

```bash
python3.12 -m venv .env_ray && source .env_ray/bin/activate
pip install --upgrade pip
pip install torch numpy                    # -> torch 2.13.0+cu130 (pin 2.11.0 has no cu130 wheel)
pip install 'ray[train]==2.57.0' transformers==5.5.0 trl==0.27.0 peft==0.18.1 \
            datasets==4.3.0 accelerate==1.14.0 python-dotenv 'pyarrow>=23.0.1'
# re-verify torch was NOT downgraded by the second install:
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# -> 2.13.0+cu130 13.0 NVIDIA H100 80GB HBM3
```

Resulting versions: **torch 2.13.0+cu130, ray 2.57.0, transformers 5.5.0, trl 0.27.0,
peft 0.18.1, datasets 4.3.0, accelerate 1.14.0, pyarrow 25.0.1**, driver 580.173.02. The
second `pip install` downgraded `fsspec` 2026.7.0 → 2025.9.0 (a `datasets` constraint);
harmless. `kernels` was **not** pulled in. `flash-attn` was not installed — the script
hard-codes `attn_implementation="sdpa"` with no flag to switch it, so sdpa is the path here
(SFT still runs fine; FA2 would only marginally speed a 350M smoke).

#### GPU visibility on CUDA — `RAY_EXPERIMENTAL_NOSET_*` is NOT needed (the key finding)

**This is the deliberate reversal of the ROCm recipe.** The MI355X runs above require
`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` because two Ray subsystems scope devices with
*different* variables on ROCm (Ray Core narrows `HIP_VISIBLE_DEVICES`, Ray Train widens
`CUDA_VISIBLE_DEVICES` — see the two-variable mismatch in §8.1). **On CUDA there is no such
split:** Ray has a real CUDA device manager (`ray/air/_internal/device_manager/nvidia_gpu.py`),
and both Ray Core's narrowing *and* Ray Train's `share_cuda_visible_devices` step touch the
**same** `CUDA_VISIBLE_DEVICES` variable, so the index arithmetic is self-consistent. The
clean approach is simply:

```bash
export CUDA_VISIBLE_DEVICES=6     # Ray SEES only GPU 6; it assigns its single worker there.
# do NOT set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES — left unset, everything works.
# (HIP_VISIBLE_DEVICES / RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES are ROCm-only; drop them.)
```

Because Ray only *sees* one GPU, `ray.cluster_resources()` reported **`'GPU': 1.0`** (and
`'accelerator_type:H100': 1.0`) — Ray did **not** enumerate the box's other 7 GPUs. Watching
`nvidia-smi` across the whole node during startup, **Ray touched only GPU 6**; the co-tenant
production job on GPUs 0–3 and the idle GPUs 4/5/7 were never claimed. Verified with the NOSET
var **unset** — it is genuinely unnecessary on CUDA.

#### Exact smoke command

```bash
export CUDA_VISIBLE_DEVICES=6
# HF_HOME points at the model cache (set above); the 350M model must be fully cached there
export HF_DATASETS_CACHE=/dev/shm/h100/dscache_ray
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1      # cache-first
export MASTER_PORT=29646                            # assigned port; not the default 29500

python3 train_llm_ray.py \
  --model_name LiquidAI/LFM2.5-350M \
  --num_workers 1 \
  --batch_size 1 --grad_acc_steps 1 \
  --num_train_epochs 10 \
  --max_seq_length 512 \
  --logging_steps 1 --save_steps 1000000 --num_to_keep 1 --max_failures 0 \
  --storage_path /dev/shm/h100/out/ray/results --run_name h100_smoke
```

Step geometry: 10 rows / (batch 1 × world 1 × grad_acc 1) = **10 optimizer steps/epoch × 10
epochs = 100 steps** (`train_steps_per_second ≈ 8.9`) — a non-trivial, clearly-converging run.

#### Expected output

The cluster sees exactly one GPU, and one worker at `world_size=1`:

```
INFO - root - Ray cluster resources: {'node:__internal_head__': 1.0, ..., 'CPU': 96.0,
  'GPU': 1.0, 'accelerator_type:H100': 1.0, ...}
(RayTrainWorker pid=1518126) Setting up process group for: env:// [rank=0, world_size=1]
```

Loss decreased monotonically-in-trend from step 1 to step 100 (finite throughout):

```
(RayTrainWorker pid=1518126) {'loss': '4.106', ..., 'mean_token_accuracy': '0.3973', 'epoch': '0.1'}
(RayTrainWorker pid=1518126) {'loss': '3.077', ..., 'mean_token_accuracy': '0.4859', 'epoch': '0.2'}
(RayTrainWorker pid=1518126) {'loss': '2.621', ..., 'epoch': '0.3'}
(RayTrainWorker pid=1518126) {'loss': '0.008682', 'grad_norm': '0.3809', ..., 'epoch': '10'}
(RayTrainWorker pid=1518126) {'train_runtime': '11.24', 'train_samples_per_second': '8.895', 'train_steps_per_second': '8.895', 'train_loss': '1.096', 'epoch': '10'}
```

A full HF checkpoint was written and reported to Ray Train:

```
(RayTrainWorker pid=...) Checkpoint successfully created at:
  Checkpoint(filesystem=local, path=.../h100_smoke/checkpoint_<timestamp>)
# checkpoint/ contents: model.safetensors (709MB), optimizer.pt (1.4GB), config.json,
#   tokenizer.json, chat_template.jinja, trainer_state.json, scheduler.pt, rng_state.pth
```

**GPU-6 residency, sampled from inside the run** (whole-node `nvidia-smi` + compute-apps,
matching the worker PID against GPU 6's UUID):

```
==== sample 20 ====                       # per-GPU memory.used / util
0, 67124 MiB, 100 %    <- co-tenant prod job (GPUs 0-3), untouched
...
6, 1393 MiB, 0 %       <- the training worker
7, 0 MiB, 0 %          <- idle, untouched
-- compute-apps --  (pid, used_memory, gpu_uuid)
<prod job pids>   ...  <- only on GPUs 0-3 UUIDs
<worker pid>, 1393 MiB, GPU-<uuid of GPU 6>   <- ONLY the training PID, ONLY on GPU 6
```

Peak for the training PID was **4775 MiB at 41% GPU util** on GPU 6 — real on-GPU compute.
Across all in-band samples the PID touched **exactly one** GPU UUID (GPU 6); it never
appeared on any other GPU.

#### Multi-GPU (deferred)

Not run on NVIDIA — the box's other free GPUs were reserved and GPUs 0–3 were a production
job. A 2- or 8-GPU pass on H100 should be **trivial** given the single-GPU
result and the §8.1 mechanism: set `CUDA_VISIBLE_DEVICES` to the owned GPUs (e.g. `4,5` or
`0,...,7`), pass `--num_workers N`, and — unlike ROCm — **leave `RAY_EXPERIMENTAL_NOSET_*`
unset**, because the two Ray subsystems agree on `CUDA_VISIBLE_DEVICES` on NVIDIA. Expect
`'GPU': N` in `cluster_resources()` and `world_size=N`. Multi-node still needs an S3/NFS
`--storage_path` (local `/dev/shm` is single-node only) and remains unverified.

#### Summary

**This path works on H100, single GPU, with no env-var workaround.** The only deviations from the
MI355X recipe were expected and CUDA-native: (1) install stable `torch 2.13.0+cu130` because
the `torch==2.11.0` pin has no cu130 wheel; (2) **drop** `HIP_VISIBLE_DEVICES` and
`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` — plain `CUDA_VISIBLE_DEVICES` is sufficient and
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` is **not** needed. 80GB VRAM was ample for the
350M full fine-tune (peak ~4.8GB); no OOM mitigation required.

## 9. Notes

- **Ray Train V2 is what this targets.** Ray revamped the Train API in 2.43 behind
  `RAY_TRAIN_V2_ENABLED=1`; in current Ray (2.57) the internal `is_v2_enabled()` defaults
  to **true**, so V2 is the default and the flag is no longer needed. Between 2.43 and
  ~2.56 you must export `RAY_TRAIN_V2_ENABLED=1` yourself, or you silently get the old
  implementation. The **public names are the same across both** (`TorchTrainer`,
  `ScalingConfig`, `RunConfig`, `ray.train.report`), which is why this script is portable —
  but the *semantics* of failure handling differ, so check your version:
  ```bash
  python -c "import ray; print(ray.__version__)"
  ```
  What V2 removed matters here: `TorchTrainer.restore()` / `can_restore()` and
  `restore_from_checkpoint` are **deprecated**. Resuming is now expressed by constructing
  a normal `TorchTrainer` with the *same* `RunConfig(storage_path, name)` — which is
  exactly why `--run_name` exists and why the script prints it on exit.

- **Three levels of fault tolerance.** (1) *Worker process* failures — an OOM or runtime
  error inside `train_func`. (2) *Worker node* failures — hardware, network, preemption.
  For both, Ray shuts down all workers, requests replacement nodes if needed, restarts the
  group, and hands them the latest checkpoint; each recovery counts against
  `FailureConfig(max_failures)`, which **defaults to 0 (disabled)** — this script defaults
  it to 3. (3) *Driver* failures — the process that called `fit()`, usually on the head
  node. Ray cannot retry that from inside; you relaunch the same command, and Ray finds the
  prior run state at `{storage_path}/{name}`. Node **preemption** is tracked separately
  from real failures (`max_preemption_failures`, unlimited by default), so spot reclaims do
  not burn your `max_failures` budget — which is the single most useful property of this
  setup on AWS spot.

- **Checkpointing is what makes recovery real.** Fault tolerance without checkpoints just
  means restarting from step 0. The `RayTrainReportCallback` added to the TRL trainer
  reports metrics and a checkpoint to Ray on the HF `save_steps` cadence; on restart
  `ray.train.get_checkpoint()` returns the latest one and the script passes it to
  `trainer.train(resume_from_checkpoint=...)`. Both halves are required — saving *and*
  loading.

- **Persistent storage is a hard multi-node requirement.** Ray expects every worker to
  write checkpoints to the *same* location. On a single node a local path is fine; on
  multiple nodes, head-node local disk is **not supported** and Ray raises an error at
  checkpoint time. Use `s3://bucket/path` (pyarrow's S3 filesystem handles it, no extra
  dependency) or a shared mount like EFS/NFS. The script warns when `--storage_path` looks
  local.

- **Why `train_func` builds everything itself.** Models, datasets, and tokenizers are
  constructed *inside* the training function rather than captured from the driver. Ray
  serializes the function to each worker, and upstream explicitly warns that passing loaded
  datasets/metrics in from outside causes serialization errors. Only small JSON-ish config
  travels, via `train_loop_config`.

- **The Ray/HF-Trainer division of labor.** Ray sets up the torch process group and places
  one worker per GPU; each worker then runs a plain HF `Trainer` that believes it is a
  single process. `prepare_trainer()` validates that pairing. Note the asymmetry with the
  raw-PyTorch path: with a bare loop you would call `ray.train.torch.prepare_model()` and
  `prepare_data_loader()` to wrap in DDP and shard the data, but when a framework already
  owns distribution — HF `Trainer`, or Accelerate's `Accelerator.prepare()` — you must
  **not** call those, or you end up double-wrapping. This script is in the second camp.

- **`ScalingConfig` is the scaling knob.** `num_workers` + `use_gpu` is the whole story
  for this script; Ray schedules workers wherever the cluster has capacity. V2 additionally
  accepts a `(min_workers, max_workers)` tuple for elastic training, plus
  `resources_per_worker`, `accelerator_type`, and `label_selector` for heterogeneous
  clusters — useful on AWS to pin workers to a specific instance type. Not exposed as flags
  here; edit the `ScalingConfig` call if you need them.

- **What this deliberately does not do.** No ZeRO/FSDP sharding is configured, so each
  worker holds a full model replica — fine for LoRA or models that fit in 80GB, wrong for a
  70B full fine-tune. Ray composes with DeepSpeed and FSDP (via an `Accelerator` built
  inside `train_func`, or `RayDeepSpeedStrategy` for Lightning), which is the documented
  route if you need sharding *and* Ray's scheduling.

- **API uncertainty, stated plainly.** The V1/V2 split is the main risk: names are stable,
  behavior is not, and `ray.train.report` / `Checkpoint` / `get_checkpoint` were verified
  against Ray 2.57 sources. `CheckpointConfig(checkpoint_frequency)` and
  `checkpoint_at_end`, and `FailureConfig(fail_fast)`, are **deprecated and raise** in V2 —
  they are not used here. If a keyword is rejected, check
  [the Ray Train API reference](https://docs.ray.io/en/latest/train/api/api.html) for your
  installed version rather than guessing.
