# `training/llm/scalarlm` — ScalarLM remote-training client

`train.py` and `inference.py` are thin **clients**: they format data locally and submit
jobs to a remote **ScalarLM** deployment through the `scalarlm` SDK
(`SupermassiveIntelligence`). Training runs server-side on the deployment's GPUs, so
nothing in this folder needs a GPU.

Three training modes — `language_model`, `embedding`, `classification` — across many model
families (qwen3, llama3, gemma3, rnj-1, olmo3, mistral, lfm, phi4, gpt-oss), with
LoRA / tokenformer / full-parameter training on FSDP or DDP.

> **The client and the server image must come from the same revision.** ScalarLM ships the
> training code — this folder's `ml/` tree — from the client with every job, and that code
> calls into the server's `cray_infra`. A client checkout newer or older than the running
> image fails server-side (e.g. `ModuleNotFoundError: No module named
> 'cray_infra.huggingface'`), after the job has already been accepted and given a
> `model_name` hash. Check the image's `org.opencontainers.image.revision` label with
> `docker inspect` against the checkout you are submitting from.

**Hardware:** the client is pure Python and runs anywhere. The server image is built for
AMD MI355X (gfx950 / ROCm 7.2.4, RCCL) or NVIDIA H100 (sm_90 / CUDA 13, NCCL).

## Files

- `train.py` — format a dataset and submit a training job.
- `inference.py` — submit a batch generate/embedding job against a (fine-tuned) model.
- `ml/` — **the training recipe**, run server-side (`cray_megatron`, `adapters`,
  `tokenformer`). This is the copy you edit: the SDK uploads whatever `ml/` sits next to
  you, so a change takes effect on the next job with no image rebuild. Vendored in upstream
  style. `ml/adapters/merge_lora_and_push.py` merges a trained LoRA into its base model and
  pushes it to the Hub (`--job-dir`, `--checkpoint`; the rank is read from the saved
  tensors, not the job config); `ml/nara_offline_eval.py` evaluates noise-aware LoRA on
  DiffusionGemma offline.
- `repo/` — the full ScalarLM **server** source with the AMD integration applied. Only
  needed to rebuild the image or change something outside `ml/`. `repo/ml/` is generated at
  build time and is not source.
- `docs/build_image.sh` — stages `ml/` into `repo/` and builds the server image.
- `docs/scalarlm_mi355.patch` — the AMD integration as one patch against public upstream
  `supermassive-intelligence/scalarlm` at commit `4566a84`; applying it to a fresh clone at
  that commit reproduces `repo/`.
- `docs/DOCKER_IMAGE_MI355.md`, `docs/DOCKER_IMAGE_H100.md` — server-image runbooks.
- `data/` — small sample datasets for each mode.

There is exactly one `ml/` tree in version control — this one — so the image always bakes
the recipe you see here.

## Setup

The client is pure Python; no NVIDIA/AMD split applies:

```bash
cd training/llm/scalarlm
ln -sf ../../../dev.env dev.env        # HF_TOKEN, used for --hf_upload_token
python3.12 -m venv .env_scalarlm && source .env_scalarlm/bin/activate
pip install -r requirements_scalarlm.txt
export SCALARLM_API_URL=http://<your-scalarlm-deployment>
```

`scalarlm` is intentionally unpinned; installing it brings the `masint` package and the
`scalarlm` CLI (`logs plot ls squeue stats clear_queue cancel delete`). The endpoint
resolves as `SCALARLM_API_URL` -> `MASINT_API_URL` -> `http://localhost:8000`.

`hf_upload_token` is submitted with the job and appears in that job's config on the server.
Never commit `dev.env`, and rotate any token that has been committed.

## Data

Set `--training_mode` and point `--data_path` at a JSON list (or JSONL) with the matching
schema:

| Mode | Required fields | Sample |
|---|---|---|
| `language_model` | `prompt`, `completion` (optional `reasoning`) | `data/llm_training_sample.json` |
| `classification` | `text` + `label` (or `question` + `answer`) | `data/classification_data.json` |
| `embedding` | `sentence1`, `sentence2`, `score` | `data/embedding_test_data.json` |

For `classification` the client derives the class set from the data and submits
`label2id` / `id2label` / `num_labels` nested under `classification` in `train_args`, so
`--num_labels` only needs setting to override that.

For `language_model`, `--gradient_calculation` controls masking: `output_tokens` (default)
masks the prompt and computes loss on the completion only; `entire_sequence` computes loss
on the whole prompt+completion. `--model_type` selects the chat-template tokens used to
format each example.

`inference.py` expects an `anchor` field per record in `embedding` mode and a `prompt`
field in `language_model` mode.

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

- Do **not** pair `fsdp` with LoRA: SimpleFSDP issues one all-gather per wrapped module and
  LoRA adds roughly a thousand tiny ones. Use `pytorch_fsdp`.
- Diffusion models (auto-detected from `model_type: diffusion_gemma`) run on `ddp` and
  `pytorch_fsdp` and fail on `fsdp`.
- `pytorch_fsdp` is the only sharded strategy with HSDP. `hsdp_shard_size` and
  `reshard_after_forward` are server-side settings with no CLI flag — add them to
  `build_train_args()` in `train.py` to send them.

## Arguments

### `train.py`

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | `data/classification_data.json` | Local training data (JSON list or JSONL) |
| `--custom_data_path` | `None` | Path that already exists on the server (overrides `--data_path`) |
| `--training_mode` | `language_model` | `language_model` / `embedding` / `classification` |
| `--model_type` | `qwen3` | Chat-template family for special-token formatting |
| `--gradient_calculation` | `output_tokens` | Prompt-masked vs full-sequence loss |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--max_steps` | `11` | Training steps |
| `--learning_rate` | `0.0005` | Learning rate |
| `--batch_size` | `2` | Batch size |
| `--max_token_block_size` | `1762` | Max token block size |
| `--r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--target_modules` | `q_proj k_proj v_proj o_proj` | LoRA target modules (space-separated) |
| `--use_rslora` | `False` | Rank-stabilized LoRA (`alpha/sqrt(r)` scaling); the merge step reads the same flag |
| `--sampling_seed` | `42` | Dataset packing / replay RNG seed |
| `--steps_per_checkpoint` | `100` | Checkpoint frequency |
| `--adapter_type` | `none` | `lora` / `tokenformer` / `none` (full-parameter) |
| `--optimizer_type` | `adamw` | `adamw` / `sgd` / `rmsprop`; a new one also needs a server-side training-loop edit |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--gradient_checkpointing` | `True` | Gradient checkpointing |
| `--distribution_strategy` | `fsdp` | `fsdp` / `ddp` / `pytorch_fsdp` |
| `--attn_implementation` | `sdpa` | `sdpa` / `eager` / `flash_attention_2` |
| `--freeze_layer_keywords` | `vision_model, vision_tower, multi_modal_projector, visual` | Comma-separated layer keywords to freeze |
| `--num_labels` | `None` | [classification] Label count (derived from the data when unset) |
| `--classification_dropout` | `0.1` | [classification] Head dropout |
| `--label_smoothing` | `0.1` | [classification] Label smoothing |
| `--upload_to_hf` | `False` | Push the trained model to the Hub |
| `--hf_repo_id` | `farbodtavakkoli/scalarlm-test` | Hub repo for uploads — change to your own |
| `--hf_upload_token` | `$HF_TOKEN` | Hub token (from `dev.env`) |
| `--gpus` | `1` | GPUs per node; on a bare-metal 8-GPU box pass `--gpus 8 --nodes 1` |
| `--nodes` | `2` | Number of nodes for the job |

### `inference.py`

| Flag | Default | Meaning |
|---|---|---|
| `--test_data_path` | `data/classification_data.json` | Test data (JSON list) |
| `--model_name` | *(a sample job hash)* | Fine-tuned model id from training |
| `--max_tokens` | `500` | Input + output token budget per request |
| `--batch_size` | `2` | Prompts per batch |
| `--inference_mode` | `language_model` | `language_model` or `embedding` |
| `--model_type` | `rnj-1` | Chat-template family for prompt formatting |

## Output

- Training: `scalarlm_training/runs/<timestamp>/train_status.json`, containing the
  `model_name` job hash. Two submissions inside the same second get a `-2`, `-3`, ... suffix.
- Inference: `scalarlm_inference/runs/<timestamp>/inference_results.json` with
  ground-truth, prompt, and generated response per record.

## Server image

One source tree, one `ml/` recipe, two images built by the same `repo/Dockerfile`.
Runbooks: `docs/DOCKER_IMAGE_MI355.md` (AMD) and `docs/DOCKER_IMAGE_H100.md` (NVIDIA).

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:mi355-v1.7` | AMD MI355X — gfx950 / ROCm 7.2.4 |
| `farbodatdocker/scalarlm:h100-v1.6` | NVIDIA H100 — sm_90 / CUDA 13 |

```bash
docker pull farbodatdocker/scalarlm:mi355-v1.7     # AMD MI355X
docker pull farbodatdocker/scalarlm:h100-v1.6      # NVIDIA H100
```

Build either from source (the script auto-detects the vendor; set `TARGET` to be explicit):

```bash
cd docs
TARGET=amd    IMAGE_TAG=mi355-v1.7 ./build_image.sh
TARGET=nvidia IMAGE_TAG=h100-v1.6  ./build_image.sh
```

Read an image's source revision from its `org.opencontainers.image.revision` label with
`docker inspect`; do not infer it from the tag. Both images are self-starting — `docker run
<image>` serves, and passing a script still overrides.

**NVIDIA H100:**

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SCALARLM_MAX_GPUS_PER_NODE=8 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.6
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
  -v "$PWD/models:/root/.cache/huggingface" \
  farbodatdocker/scalarlm:mi355-v1.7
```

Run flags that matter on both vendors:

- `--shm-size=64g` is required for multi-rank jobs; the 64 MB Docker default kills a rank
  with SIGBUS, which Slurm then relaunches in a loop.
- `SCALARLM_MAX_GPUS_PER_NODE` defaults to **1**. Without it every job is silently capped
  to a single GPU no matter what `--gpus` says.
- On AMD, `--device=/dev/kfd --device=/dev/dri --group-add video` replaces `--gpus`.
- Serve fine-tuned adapters at tensor-parallel size 1. Tensor parallelism works for the
  base model, but vLLM's TP path rejects the hot-reloaded un-sharded adapter state dict.

## Notes

- `--attn_implementation flash_attention_2` is mapped to `sdpa` by the server on both
  vendors. FA2 loads and then aborts mid-training with a device-side
  `HSA_STATUS_ERROR_EXCEPTION` and no traceback, so leave the default.
- `embedding` mode needs `--batch_size 2` or more. CoSENT is a pairwise ranking loss, so at
  `batch_size 1` every step reports exactly 0.0 loss while the job still finishes
  COMPLETED.
- `--sample_fraction 0.1` on the 5-record `data/llm_training_sample.json` truncates to 0
  rows and the SDK rejects the empty archive (`ValueError: The file ... is empty`) — keep
  `1.0` with the shipped samples.
- Every key you add to `train_args` must be declared on `JobConfig` in
  `repo/infra/cray_infra/util/default_job_config.py`. Pydantic runs with `extra="ignore"`,
  so an undeclared key is dropped silently between `config.yaml` and the training code;
  `get_job_config()` logs what it dropped. Numeric CLI arguments need an explicit `type=`
  or the server fails comparing a string.
- `upload_to_hf` is rank-0 only and best-effort: a failed upload still reports COMPLETED,
  with the reason only in the rank-0 log.
- Resume is automatic — the loop picks the highest-numbered `checkpoint_*.pt` in the job
  directory and continues at `step + 1`. Rotation keeps only the last 3 checkpoints, so a
  crash more than three intervals back cannot be resumed. Relaunch through `sbatch`; the
  entrypoint reads `SLURM_JOB_NODELIST` and dies as a bare shell script.
- Multi-node: keep the HF cache **node-local**. `huggingface_hub` guards it with `filelock`
  and `flock()` on NFS fails with `OSError: [Errno 116] Stale file handle`, killing every
  rank during model load. `scontrol show hostnames` must return names that resolve for
  mpirun, and container `--device` lists must be computed per node.
- When editing `ml/`, keep the stop check, the finite-gradient check and the loss reduction
  on every rank on every step — one behind a rank-local branch hangs the job with no error.
