# ScalarLM remote training client

`train.py` and `inference.py` format data locally and submit jobs through the
`scalarlm` SDK (`SupermassiveIntelligence`). Training runs on the remote server;
the client needs no accelerator.

Training modes include `language_model`, `embedding` and `classification`, with
LoRA, tokenformer or full parameter training on FSDP or DDP.

> **The client and the server image must come from the same revision.** ScalarLM ships the
> training code,this folder's `ml/` tree,from the client with every job, and that code
> calls into the server's `cray_infra`. A client checkout newer or older than the running
> image fails on the server (e.g. `ModuleNotFoundError: No module named
> 'cray_infra.huggingface'`), after the job has already been accepted and given a
> `model_name` hash. Check the image's `org.opencontainers.image.revision` label with
> `docker inspect` against the checkout you are submitting from.

**Hardware:** the client is pure Python and runs anywhere. The server image is built for
AMD MI355X (gfx950 / ROCm 7.2.4, RCCL) or NVIDIA H100 (sm_90 / CUDA 12.9, NCCL).
A separate Trainium2 recipe awaits hardware validation; see [Trainium2](#trainium2).

## Files

* `train.py`: format data and submit training.
* `inference.py`: submit generation or embedding requests.
* `ml/`: canonical training recipe uploaded with each job. Keep the tracked `repo/ml/`
  copy synchronized through `docs/build_image.sh` when building an image.
* `repo/`: server source used for image builds and infrastructure changes.
* `docs/`: image runbooks, build helper and the AMD integration patch.
* `data/`: sample datasets.

`ml/adapters/merge_lora_and_push.py` merges LoRA checkpoints for Hub export;
`ml/nara_offline_eval.py` evaluates DiffusionGemma offline.

## Setup

The client is pure Python; no NVIDIA/AMD split applies. Run from the OTel repo root:

```bash
cd training/llm/scalarlm
echo "HF_TOKEN=<your token>" > dev.env
python3.12 -m venv .env_scalarlm && source .env_scalarlm/bin/activate
pip install -r requirements_scalarlm.txt
export SCALARLM_API_URL=http://<your-scalarlm-deployment>
```

`scalarlm` is intentionally unpinned; installing it brings the `masint` package and the
`scalarlm` CLI (`logs plot ls squeue stats clear_queue cancel delete`). The endpoint
resolves as `SCALARLM_API_URL` → `MASINT_API_URL` → `http://localhost:8000`.

`hf_upload_token` is submitted with the job and appears in that job's config on the server.
Keep `dev.env` and tokens out of version control.

## Run

### Train

```bash
python train.py \
  --training_mode language_model \
  --model_type qwen3 \
  --data_path data/llm_training_sample.json \
  --adapter_type lora --r 8 --lora_alpha 16 \
  --max_steps 100 --learning_rate 5e-4 --batch_size 2 \
  --gpus 1 --nodes 2
```

A smoke run is the same command with a small `--max_steps`. `train.py` saves the returned
job status to `scalarlm_training/runs/<timestamp>/train_status.json`; that status carries
the `model_name` (job hash) you need for inference.

Jobs are keyed by their config: resubmitting an unchanged config returns the existing job
instead of retraining. Change at least one setting (e.g. bump `--max_steps`) to force a new
job.

### Inference

Pass the job hash from training as `--model_name`:

```bash
python inference.py \
  --test_data_path data/llm_training_sample.json \
  --model_name <model_id_from_training> \
  --model_type qwen3 \
  --inference_mode language_model \
  --max_tokens 500 --batch_size 2
```

To run the **base** model instead, remove `model_name` from the `llm.generate` call. Failed
batches are logged with a traceback and skipped; results contain only successful indices.

### Choosing a distribution strategy

| `--distribution_strategy` | What it is | Use it when |
|---|---|---|
| `ddp` | Full replica per GPU | Small models (~0.6B); OOMs at 32B |
| `fsdp` | SimpleFSDP, wraps every module owning parameters | Legacy default; not with LoRA |
| `pytorch_fsdp` | PyTorch FSDP2 `fully_shard`, per transformer block | A few billion parameters and up |

* Do **not** pair `fsdp` with LoRA: SimpleFSDP issues one collective per wrapped module.
  Use `pytorch_fsdp`.
* Diffusion models (detected from `model_type: diffusion_gemma`) run on `ddp` and
  `pytorch_fsdp` and fail on `fsdp`.
* `pytorch_fsdp` is the only sharded strategy with HSDP. `hsdp_shard_size` and
  `reshard_after_forward` are server settings. `reshard_after_forward` has the
  `--reshard_after_forward` flag; `hsdp_shard_size` has no CLI flag, so add it to
  `build_train_args()` in `train.py` to send it.

## Output

* Training: `scalarlm_training/runs/<timestamp>/train_status.json`, containing the
  `model_name` job hash. Two submissions inside the same second get a `-2`, `-3`, ... suffix.
* Inference: `scalarlm_inference/runs/<timestamp>/inference_results.json` with
  ground truth, prompt, and generated response per record.

## GRPO

`ml/` adds a `grpo` training mode and an `init_adapter_from` option, so one LoRA adapter is carried across jobs and hot loaded into the running vLLM between cycles. Existing modes are unchanged. `train_llm_grpo_scalarlm.py` (with `utils.py` and `train_job_entrypoint_gpu_pinned.sh`) is the outer loop: each cycle samples completions through the server, scores them with the shared judge in [`../grpo_common`](../grpo_common/README.md), computes group advantages and submits one training job that continues the previous adapter.

GRPO needs the sibling folder `training/llm/grpo_common` from this repo; override its path with `GRPO_COMMON_DIR` (see `utils.py`). Copy `../grpo_common/.env.example` to `.env`, fill it in and `source` it first; it defines `REPO_ROOT`, `GRPO_SHARED_DIR`, `HF_CACHE_DIR`, `GRPO_ENV_DIR` and `HF_TOKEN`.

The image must carry the GRPO mode (`mi355-v1.8`, `h100-v1.8` or newer). Check for `/app/cray/ml/cray_megatron/megatron/grpo_loss.py` inside it before trusting a tag.

```bash
python3 -m venv $GRPO_ENV_DIR/env_scalarlm
$GRPO_ENV_DIR/env_scalarlm/bin/pip install repo/sdk transformers tokenizers jinja2 numpy requests
../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_scalarlm/bin/python train_llm_grpo_scalarlm.py \
  --model $GRPO_SHARED_DIR/models/<org>--<name> --data ../grpo_common/data/grpo_prompts.jsonl \
  --output-dir <run> --steps 2 --warmup-steps 0 --prompt-batch-size 8 --group-size 4 --max-tokens 256 \
  --keep-zero-spread --start-server --render-nodes /dev/dri/renderD<a> /dev/dri/renderD<b> \
  --image farbodatdocker/scalarlm:mi355-v1.8 --verify
```

On H100, give the judge a CUDA vLLM image and its own GPU, and pass GPU indices:

```bash
export GRPO_PLATFORM=cuda
JUDGE_IMAGE=<cuda-vllm-image> JUDGE_GPU=7 JUDGE_MODEL_DIR=<judge-dir> ../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_scalarlm/bin/python train_llm_grpo_scalarlm.py \
  --model $GRPO_SHARED_DIR/models/<org>--<name> --data ../grpo_common/data/grpo_prompts.jsonl \
  --output-dir <run> --steps 2 --warmup-steps 0 --prompt-batch-size 8 --group-size 4 --max-tokens 256 \
  --keep-zero-spread --start-server --render-nodes 0 1 \
  --image farbodatdocker/scalarlm:h100-v2.3 --verify
```

Use `--help` on each launcher for the complete argument reference. `--objective rft`
selects rejection sampling SFT rather than GRPO. `--verify` checks adapter continuation,
PEFT export and generation through the loaded adapter.

Notes:

1. Run the launcher from this folder: the SDK ships the adjacent `ml/` with every job.
2. `--model` must be a local directory named `<org>--<name>`; the launcher builds an offline HF cache entry for it.
3. `train_job_entrypoint_gpu_pinned.sh` is mounted over the image entrypoint so training never lands on the vLLM GPU.
4. Jobs are keyed by `sha256(train_args)`; each cycle adds `sampling_seed = seed + cycle` to stay distinct.
5. An image without the GRPO mode returns no log probs and silently falls back to ratio 1. Treat that as a wrong image.

## Server image

One source tree and one `ml/` recipe. AMD and NVIDIA use `repo/Dockerfile`;
H100 v2.4 uses `repo/Dockerfile.h100-v2.4`, and Trainium2 uses `repo/Dockerfile.trainium`.
Runbooks: [AMD](docs/DOCKER_IMAGE_MI355.md), [NVIDIA](docs/DOCKER_IMAGE_H100.md)
and [Trainium2](docs/DOCKER_IMAGE_TRAINIUM2.md).

| Tag | Hardware | Evidence |
|---|---|---|
| `farbodatdocker/scalarlm:mi355-v1.8` | AMD MI355X, gfx950 / ROCm 7.2.4 | Documented in AMD runbook; not installed locally |
| `farbodatdocker/scalarlm:h100-v2.2` | NVIDIA H100, sm_90 / CUDA 12.9 | Local image; older version |
| `farbodatdocker/scalarlm:h100-v2.3` | NVIDIA H100, sm_90 / CUDA 12.9 | Local image and NVIDIA runbook |
| `farbodatdocker/scalarlm:h100-v2.4` | NVIDIA H100, sm_90 / CUDA 12.9 | Local image and overlay recipe; existence is not runtime validation |
| `farbodatdocker/scalarlm:trainium2-v1.0` | AWS Trainium2, Neuron | Build target only; no local image; awaiting Trainium agent validation |

Registry availability was not checked. Inspect the source revision before submitting jobs.

```bash
docker pull farbodatdocker/scalarlm:mi355-v1.8     # AMD MI355X
docker pull farbodatdocker/scalarlm:h100-v2.3      # NVIDIA H100
```

Build the GPU images from source with an explicit vendor and tag:

```bash
cd docs
TARGET=amd    IMAGE_TAG=mi355-v1.8 ./build_image.sh
TARGET=nvidia IMAGE_TAG=h100-v2.3  ./build_image.sh
```

`h100-v2.3` adds over `h100-v2.2`: `hf_xet` pinned in the trainer requirements, no false `Error loading job status`
line when a new job starts, and a clean `Using device cuda:N` log line.

Read an image's source revision from its `org.opencontainers.image.revision` label with
`docker inspect`; do not infer it from the tag. GPU images start the server by default;
passing a script overrides startup.

**NVIDIA H100:**

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SCALARLM_MAX_GPUS_PER_NODE=8 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v2.3
# curl http://localhost:8000/v1/health -> {"api":"up","vllm":"up","all":"up"}
```

**AMD MI355X:**

```bash
docker run -d --name scalarlm --init \
  --shm-size=64g --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE --cap-add IPC_LOCK \
  -p 8000:8000 -p 8001:8001 \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B \
  -e SCALARLM_MAX_GPUS_PER_NODE=8 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:mi355-v1.8
```

Run flags that matter on both vendors:

* `--shm-size=64g` is required for distributed jobs; the 64 MB Docker default can kill
  a rank with SIGBUS, which Slurm then relaunches in a loop.
* `SCALARLM_MAX_GPUS_PER_NODE` defaults to **1**. Without it every job is silently capped
  to a single GPU no matter what `--gpus` says.
* On AMD, `--device=/dev/kfd --device=/dev/dri --group-add video` replaces `--gpus`.
* Serve fine tuned adapters at tensor parallel size 1. Tensor parallelism works for the
  base model, but vLLM's TP path rejects the dynamically loaded unsharded adapter state dict.


## Trainium2

The [Trainium2 guide](docs/DOCKER_IMAGE_TRAINIUM2.md) preserves the build, launch,
training and inference workflow from this README. It describes the current recipe,
not a validated deployment. Base image digests, device allocation, model compilation,
checkpoint export and serving remain gated on the Trainium agent's hardware results.

## Notes

* Keep `--attn_implementation sdpa` on GPU servers; the FA2 override is mapped to SDPA.
* Embedding requires `--batch_size 2` or more for its pairwise ranking loss.
* Keep `--sample_fraction 1.0` with the shipped samples to avoid an empty dataset.
* Declare new `train_args` keys on `JobConfig` in
  `repo/infra/cray_infra/util/default_job_config.py`; undeclared keys are ignored.
  Numeric CLI arguments need an explicit `type=`.
* `upload_to_hf` runs on rank zero. Upload failures do not change COMPLETED status;
  inspect that rank's log.
* Resume selects the highest numbered `checkpoint_*.pt` and continues at `step + 1`.
  Rotation retains three checkpoints. Relaunch through `sbatch`, not a bare entrypoint.
* For multiple nodes, use a separate local HF cache on each node. Shared NFS locks can
  fail with stale handles. `scontrol show hostnames` must resolve for mpirun, and
  container `--device` lists must match each node.
* Keep stop checks, finite gradient checks and loss reduction on every rank on every
  step; placing them inside a branch executed by one rank can hang the job.
