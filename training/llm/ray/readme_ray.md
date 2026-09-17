# `training/llm/ray` — Ray Train orchestration of an HF/TRL fine-tune

Distributed orchestration of an ordinary Hugging Face Transformers/TRL SFT fine-tune with
[Ray Train](https://github.com/ray-project/ray) (`ray.train.torch.TorchTrainer`). The training
loop is TRL's `SFTTrainer` with optional LoRA — the same algorithm as `../deepspeed/`. Pick
this over a sibling recipe when you want Ray to place one worker per GPU across a multi-node
cluster and restart the worker group from the last checkpoint after a failure. Targets the Ray
Train V2 API (ray >= 2.43, default-on from 2.57).

**Hardware:** AMD Instinct MI355X (ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0). Identical code
on both; ROCm needs one extra environment variable.

## Files

- `train_llm_ray.py` — entrypoint: the per-worker training function, submitted via `TorchTrainer`.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample, the default `--train_file`.
- `requirements_ray.txt` — pinned dependencies.

## Setup

One venv for this recipe folder. Every node in the cluster needs an *identical* environment —
same Ray version, same Python minor version, same torch build — or Ray refuses to connect the
worker. Build one venv or container image and ship it everywhere.

Do not install `flash-attn`.

### NVIDIA

`torch==2.11.0` resolves to a CUDA 13 build on PyPI, so no `--index-url` is needed. Install
torch first, then the rest, so pip does not re-resolve it:

```bash
python3.12 -m venv .env_ray && source .env_ray/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 numpy            # CUDA 13 build, straight from PyPI
pip install 'ray[train]==2.57.0' transformers==5.5.0 trl==0.27.0 peft==0.18.1 \
            datasets==4.3.0 accelerate==1.14.0 python-dotenv 'pyarrow>=23.0.1'
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# -> 2.11.0 13.0 NVIDIA H100 80GB HBM3
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

Scope GPUs with `CUDA_VISIBLE_DEVICES` and stop there — leave
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` unset.

### AMD / ROCm

Install the ROCm torch wheel first so pip never resolves the CUDA one; do not
`pip install -r requirements_ray.txt` directly on this platform.

```bash
python3.12 -m venv .env_ray && source .env_ray/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0
pip install ray[train]==2.57.0 transformers==5.5.0 trl==0.27.0 peft==0.18.1 \
            datasets==4.3.0 accelerate==1.14.0 python-dotenv 'pyarrow>=23.0.1'
python -c "import ray, torch, trl, transformers, peft; print(ray.__version__, torch.__version__, torch.cuda.device_count(), 'GPUs')"
# -> 2.57.0 2.11.0+rocm7.2 2 GPUs
```

Export before every ROCm run:

```bash
export HIP_VISIBLE_DEVICES=0,1                       # Ray scopes AMD GPUs here, not ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1                      # keep identical to HIP_VISIBLE_DEVICES; never set it empty
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # required on ROCm
```

If your venv's `activate` pins `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`, export these
*after* sourcing it or the pins win.

### Cluster

Single node is the default — the script calls `ray.init()` and starts a local Ray instance
itself. For multi-node, start Ray first and pass `--ray_address auto`:

```bash
ray start --head --port=6379           # head node
ray start --address='<head-ip>:6379'   # each worker node
```

On AWS, `ray up cluster.yaml` provisions the same topology; the training command is unchanged.

### Secrets and paths

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN, for gated checkpoints
export OUTPUT_DIR=/path/to/outputs     # pass as --storage_path
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

`train_llm_ray.py` calls `load_dotenv("dev.env")`, so run it from inside this folder. Never
commit tokens.

## Data

`data/OTel_LLM_sample_10.jsonl` — this repo's canonical chat JSONL, one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Extra columns (`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
`source_version`) are dropped after the chat template renders each row. Point at your own data
with `--train_file /path/to/your.jsonl` in the same schema — no conversion needed.

Use an `-Instruct` checkpoint: a tokenizer with no chat template hard-fails at startup.

The whole file is loaded on every worker. For datasets too large to fit per-worker, switch the
ingest to Ray Data and `ray.train.get_dataset_shard("train")`.

## Run

Smoke test — single node, shipped sample, one epoch:

```bash
python3 train_llm_ray.py --num_workers 1 --num_train_epochs 1
```

Full run:

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

Multi-node: add `--ray_address auto`, raise `--num_workers` to the total GPU count across
nodes, and keep `--storage_path` on S3 or NFS.

```
INFO - root - Ray cluster resources: {'GPU': 8.0, 'CPU': 256.0,
  'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, ...}
(RayTrainWorker pid=...) Setting up process group for: env:// [rank=0, world_size=8]
(RayTrainWorker pid=...) {'loss': '2.679', 'grad_norm': '12.94', 'mean_token_accuracy': '0.5004', 'epoch': '0.5'}
(RayTrainWorker pid=...) Checkpoint successfully created at: Checkpoint(filesystem=local, path=<storage_path>/<run_name>/checkpoint_...)
```

A hang before the workers start means the cluster cannot yet satisfy `num_workers` GPUs.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | HF repo id or local path |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat `messages` JSONL |
| `--storage_path` | `./ray_results` | Checkpoint storage; `s3://` or NFS for multi-node |
| `--run_name` | `None` (fresh uuid) | Run id; reuse the same `(storage_path, run_name)` to resume |
| `--num_workers` | `8` | Training workers, one per GPU, summed across nodes |
| `--use_gpu` | on | Reserve 1 GPU per worker |
| `--max_failures` | `3` | Worker/node failure retries (`-1` unlimited, `0` disables recovery) |
| `--num_to_keep` | `2` | Checkpoints retained in storage |
| `--ray_address` | `None` | `auto` to attach to a running cluster |
| `--max_seq_length` | `2048` | Max tokens per example |
| `--batch_size` | `1` | Per-worker (per-GPU) batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps |
| `--num_train_epochs` | `3` | Number of epochs |
| `--learning_rate` | `2e-4` | Peak learning rate |
| `--logging_steps` | `10` | Log every N steps |
| `--save_steps` | `100` | Report a checkpoint every N steps |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Trade compute for activation memory |
| `--use_lora` | off | Train a LoRA adapter |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or a comma-separated list |

Effective global batch = `batch_size x grad_acc_steps x num_workers`.

## Output

Results land in `<storage_path>/<run_name>` — same layout locally, on NFS, or in S3: the
reported checkpoints (capped by `--num_to_keep`) plus the state Ray needs to resume. On
completion the driver logs `result.metrics`, `result.path`, `result.checkpoint`, and the flags
to resume. Keep the printed `--run_name`; with `--storage_path` it both resumes an interrupted
run and locates its outputs later.

```python
import os
from transformers import AutoModelForCausalLM
from ray.train.huggingface.transformers import RayTrainReportCallback

with result.checkpoint.as_directory() as ckpt_dir:
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(ckpt_dir, RayTrainReportCallback.CHECKPOINT_NAME)
    )
```

With `--use_lora` the checkpoint holds a LoRA adapter, so inference needs the base model plus
the adapter (`PeftModel.from_pretrained`).

## Notes

- On ray 2.43-2.56, export `RAY_TRAIN_V2_ENABLED=1` or you silently get the V1 implementation.
  Check with `python -c "import ray; print(ray.__version__)"`.
- Multi-node runs require `--storage_path` on `s3://` (handled by pyarrow, no extra dependency)
  or a shared NFS/EFS mount. Head-node local disk errors at checkpoint time.
- On a shared box, export `RAY_TMPDIR=/tmp/<yours>` so `ray.init()` does not attach to another
  job's local cluster under `/tmp/ray`, and shut down via the driver — never a blanket
  `ray stop`.
- Export a different `MASTER_PORT` if port 29500 is already taken on the box.
- Resume by relaunching the same command with the same `--storage_path` and `--run_name`;
  `TorchTrainer.restore()` is deprecated in V2.
