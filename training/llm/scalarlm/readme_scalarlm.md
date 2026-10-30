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
The same `ml/` also trains and serves on AWS Trainium2; see [Trainium2](#trainium2).

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
resolves as `SCALARLM_API_URL` -> `MASINT_API_URL` -> `http://localhost:8000`.

`hf_upload_token` is submitted with the job and appears in that job's config on the server.
Keep `dev.env` and tokens out of version control.

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
  `reshard_after_forward` are server settings. `reshard_after_forward` has the
  `--reshard_after_forward` flag; `hsdp_shard_size` has no CLI flag, so add it to
  `build_train_args()` in `train.py` to send it.

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
| `--hf_repo_id` | `farbodtavakkoli/scalarlm-test` | Hub repo for uploads; set it to your own repo when using `--upload_to_hf` |
| `--hf_upload_token` | `$HF_TOKEN` | Hub token (from `dev.env`) |
| `--gpus` | `1` | GPUs per node; on a bare-metal 8-GPU box pass `--gpus 8 --nodes 1` |
| `--nodes` | `2` | Number of nodes for the job |

### `inference.py`

| Flag | Default | Meaning |
|---|---|---|
| `--test_data_path` | `data/classification_data.json` | Test data (JSON list) |
| `--model_name` | *(a job hash from the author's deployment)* | Fine tuned model id; replace it with the `model_name` from your own `train_status.json` |
| `--max_tokens` | `500` | Input + output token budget per request |
| `--batch_size` | `2` | Prompts per batch |
| `--inference_mode` | `language_model` | `language_model` or `embedding` |
| `--model_type` | `rnj-1` | Chat-template family for prompt formatting |

## Output

- Training: `scalarlm_training/runs/<timestamp>/train_status.json`, containing the
  `model_name` job hash. Two submissions inside the same second get a `-2`, `-3`, ... suffix.
- Inference: `scalarlm_inference/runs/<timestamp>/inference_results.json` with
  ground-truth, prompt, and generated response per record.

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
  --image farbodatdocker/scalarlm:h100-v1.9 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--objective` | `grpo` | `rft` is rejection sampling SFT, not GRPO |
| `--steps` / `--prompt-batch-size` / `--group-size` | 20 / 32 / 8 | cycles, prompts per cycle, completions per prompt |
| `--render-nodes` | required with `--start-server` | first entry is the vLLM GPU, the rest train; device paths on ROCm, indices on CUDA |
| `--trainer-gpus` / `--gpu-mem-util` | 1 / 0.85 | use about 0.35 with a single render node |
| `--keep-zero-spread` | off | also train groups whose rewards are all equal |
| `--verify` | off | checks adapter continuation, PEFT export and a generate through the hot loaded adapter |

Notes:

1. Run the launcher from this folder: the SDK ships the adjacent `ml/` with every job.
2. `--model` must be a local directory named `<org>--<name>`; the launcher builds an offline HF cache entry for it.
3. `train_job_entrypoint_gpu_pinned.sh` is mounted over the image entrypoint so training never lands on the vLLM GPU.
4. Jobs are keyed by `sha256(train_args)`; each cycle adds `sampling_seed = seed + cycle` to stay distinct.
5. An image without the GRPO mode returns no log probs and silently falls back to ratio 1. Treat that as a wrong image.

## Server image

One source tree, one `ml/` recipe, two images built by the same `repo/Dockerfile`.
Runbooks: `docs/DOCKER_IMAGE_MI355.md` (AMD) and `docs/DOCKER_IMAGE_H100.md` (NVIDIA).

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:mi355-v1.8` | AMD MI355X, gfx950 / ROCm 7.2.4, includes GRPO |
| `farbodatdocker/scalarlm:h100-v1.9` | NVIDIA H100, sm_90 / CUDA 12.9, vLLM v0.32, includes GRPO, Muse Glimmer and EFA |

```bash
docker pull farbodatdocker/scalarlm:mi355-v1.8     # AMD MI355X
docker pull farbodatdocker/scalarlm:h100-v1.9      # NVIDIA H100
```

Build either from source (the script auto-detects the vendor; set `TARGET` to be explicit):

```bash
cd docs
TARGET=amd    IMAGE_TAG=mi355-v1.8 ./build_image.sh
TARGET=nvidia IMAGE_TAG=h100-v1.9  ./build_image.sh
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
  farbodatdocker/scalarlm:h100-v1.9
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

- `--shm-size=64g` is required for multi-rank jobs; the 64 MB Docker default kills a rank
  with SIGBUS, which Slurm then relaunches in a loop.
- `SCALARLM_MAX_GPUS_PER_NODE` defaults to **1**. Without it every job is silently capped
  to a single GPU no matter what `--gpus` says.
- On AMD, `--device=/dev/kfd --device=/dev/dri --group-add video` replaces `--gpus`.
- Serve fine-tuned adapters at tensor-parallel size 1. Tensor parallelism works for the
  base model, but vLLM's TP path rejects the hot-reloaded un-sharded adapter state dict.


## Trainium2

One Trainium2 image runs the API, Slurm, vLLM and training, like the AMD and NVIDIA
images. `train.py` and `inference.py` work unchanged. vLLM uses the Neuron plugin;
training uses Neuron PyTorch in its own Python environment (`/opt/venv`).

Build it from a Neuron vLLM serving image and a Neuron PyTorch training image. Neither
image is published.

```bash
cd docs
TARGET=trainium NEURON_BASE=<neuron-vllm-image> NEURON_TRAIN_BASE=<neuron-pytorch-image> ./build_image.sh
```

Run it on 8 devices: vLLM takes the first 4, training uses the rest.

```bash
docker run -d --name scalarlm \
  --device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3 \
  --device=/dev/neuron4 --device=/dev/neuron5 --device=/dev/neuron6 --device=/dev/neuron7 \
  --shm-size=64g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 \
  -e SCALARLM_MODEL=/root/models/<serving-model> \
  -e SCALARLM_TENSOR_PARALLEL_SIZE=16 -e SCALARLM_MAX_GPUS_PER_NODE=16 \
  -e SCALARLM_VLLM_ARGS="<vLLM Neuron flags matching the compiled graphs>" \
  -v /path/to/models:/root/models:ro -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:trainium2-v1.0
```

Train and run inference from this folder:

```bash
python train.py --training_mode language_model --model_type gemma4 \
  --data_path data/llm_training_sample.json --adapter_type lora --r 32 --lora_alpha 32 \
  --target_modules all-linear --distribution_strategy pytorch_fsdp --reshard_after_forward \
  --llm_name google/gemma-4-31B-it --gpus 16 --nodes 1 --max_steps 50
python inference.py --test_data_path data/llm_training_sample.json \
  --model_name /root/models/<serving-model> --model_type gemma4 --inference_mode language_model
```

Notes:

1. Each logical NeuronCore is one Slurm `gpu`, 4 per device. `SCALARLM_MAX_GPUS_PER_NODE`
   sets how many cores training may use; training takes the last cores so vLLM keeps the
   first devices. Run one training job at a time: every job takes the same last cores.
2. Pass devices under their host names (`--device=/dev/neuron12`); renumbering them breaks
   device discovery.
3. `--llm_name` sets the training checkpoint (BF16) when vLLM serves a different one, such
   as an FP8 copy. Without it the job trains the served model.
4. `--target_modules all-linear` adapts only the language decoder of a multimodal model.
   The Gemma4 vision tower uses a layer type LoRA cannot wrap. Pass `all-linear` alone,
   not mixed with module names.
5. `--reshard_after_forward` is required for Gemma4 31B on 4 devices; without it the FSDP
   all gather runs out of device memory. On 16 training devices also set
   `hsdp_shard_size: 16`.
6. The first vLLM start and step 0 of training compile for minutes; mount the compile
   caches to reuse them. Write `--optimization-level=2` with `=` in `SCALARLM_VLLM_ARGS`;
   the spaced form is not passed to the Neuron plugin, which then compiles at a slower
   level.
7. The Neuron vLLM backend cannot load adapters at run time, so `--model_name <job hash>`
   and GRPO do not work. Merge the adapter from `ml/` with
   `python -m adapters.merge_lora_and_push --job-dir <job> --repo-id <id> --dry-run --output-dir <dir>`,
   then restart the container with `SCALARLM_MODEL` set to the merged model. If vLLM serves
   FP8 weights, convert the merged model to the same format first.
8. Tested with Gemma4 31B LoRA r32, sequence 1024, batch 1 per rank. Other models are
   untested.

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
