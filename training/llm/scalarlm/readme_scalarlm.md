# training/llm/scalarlm — ScalarLM remote-training client

## Overview & when to use

Train and run inference on a remote **ScalarLM** server. `train.py` and `inference.py` are thin **clients** — they format data locally and submit jobs to a ScalarLM deployment via the `scalarlm` SDK (`SupermassiveIntelligence`). The actual training runs server-side on the deployment's GPUs/nodes; nothing in this folder needs a GPU.

Supports three training modes — **language_model**, **embedding**, **classification** — across many model families (qwen3, llama3, gemma3, rnj-1, olmo3, mistral, lfm, phi4, gpt-oss), with LoRA / tokenformer / full-parameter training and FSDP or DDP.

Files:
- `train.py` — format a dataset and submit a training job.
- `inference.py` — submit a batch generate/embedding job against a (fine-tuned) model.
- `ml/` — **the training recipe.** Runs server-side (`cray_megatron`, `adapters`, `tokenformer`). This is the copy you edit: the SDK uploads whatever `ml/` sits next to you, so a change here takes effect on your next job with no image rebuild. Vendored and deliberately left in upstream style.
- `repo/` — the full ScalarLM **server** source with the AMD integration applied. Only needed to rebuild the image or change something outside `ml/` (the collectives layer, launcher, Dockerfile). It carries no `.git`; `repo/ml/` is generated at build time and is not source.
- `build_image.sh` — stages `ml/` into `repo/` and builds the MI355X image.
- `scalarlm_mi355.patch` — the AMD integration as one patch against public upstream `4566a84`.
- `scalarlm_mi355.patch` is the integration delta; the full development history is **not** tracked
  here (see `DOCKER_IMAGE_MI355.md` §"Recovering the development history").
- `DOCKER_IMAGE_MI355.md` — build / run / publish runbook for the MI355X server image.
- `data/` — small sample datasets for each mode.

> **There is exactly one `ml/` tree in version control** — this one. `repo/ml/` is created by
> `build_image.sh` at build time and is gitignored, so the image always bakes the recipe you see
> here and the two cannot drift.

> This folder is a checkout of a ScalarLM repo (fork of `supermassive-intelligence/scalarlm`) and carries its own `.git`. The client talks to whatever server `SCALARLM_API_URL` points at.

## Install

The client needs the `scalarlm` SDK plus two helpers:

```bash
python3.12 -m venv ~/.venv-scalarlm && source ~/.venv-scalarlm/bin/activate
pip install -r requirements_scalarlm.txt
```

`scalarlm` is not pinned here — install it from the ScalarLM project (it also ships the `scalarlm` CLI). See that repo for server deployment; the `ml/` subtree's heavy deps (torch/megatron/flash-attn) are managed by the **server** deployment, not this client. No NVIDIA/AMD split applies client-side — the client is pure Python.

## Environment & secrets

Point the client at a deployment:

```bash
export SCALARLM_API_URL=http://<your-scalarlm-deployment>
```

`train.py` also loads a `dev.env` from this folder (used for `--hf_upload_token` when pushing trained models to the Hub):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

> **Security:** never commit `dev.env` or tokens. `hf_upload_token` is submitted with the job and appears in that job's config on the server, so keep it out of version control and rotate any token that has been committed.

## Data

Set `--training_mode` and point `--data_path` at a JSON list (or JSONL) with the matching schema (samples are in `data/`):

| Mode | Required fields | Sample |
|---|---|---|
| `language_model` | `prompt`, `completion` (optional `reasoning`) | `data/llm_training_sample.json` |
| `classification` | `text` + `label` (or `question` + `answer`) | `data/classification_data.json` |
| `embedding` | `sentence1`, `sentence2`, `score` | `data/embedding_test_data.json` |

For `language_model`, `--gradient_calculation` controls masking:
- `output_tokens` (default) — mask the prompt, compute loss on the completion only.
- `entire_sequence` — loss on the full prompt+completion.

`--model_type` selects the chat-template tokens used to format each example.

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

A smoke run is the same command with a small `--max_steps` and `--sample_fraction 0.1` — the server does the heavy lifting either way.

> **Gotcha — job dedup:** ScalarLM keys jobs by their config. If you resubmit with an unchanged config it recognizes the job as already-run and won't retrain. Change at least one setting (e.g. bump `--max_steps`) to force a new job.

`train.py` submits every setting in `train_args` (no local config file is written on this build) and saves the returned job status to `scalarlm_training/runs/<timestamp>/train_status.json`. The status contains the `model_name` (job hash) used for inference.

### Inference

Use the trained model's id (the job hash from training) as `--model_name`:

```bash
python inference.py \
  --test_data_path data/llm_training_sample.json \
  --model_name <model_id_from_training> \
  --model_type qwen3 \
  --inference_mode language_model \
  --max_tokens 500 --batch_size 2
```

- To run against the **base** model (no adapter), remove `model_name` from the `llm.generate` call (see the `--model_name` help text).
- `--inference_mode embedding` expects records with an `anchor` field and returns embeddings; `language_model` formats each `prompt` with the chat template and generates.

## Arguments

### train.py

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | `data/classification_data.json` | Local training data (JSON list or JSONL). |
| `--custom_data_path` | None | Path that already exists on the server/VM (overrides `data_path`). |
| `--training_mode` | `language_model` | `language_model` / `embedding` / `classification`. |
| `--model_type` | `qwen3` | Chat-template family for special-token formatting. |
| `--gradient_calculation` | `output_tokens` | Prompt-masked vs full-sequence loss. |
| `--sample_fraction` | 1.0 | Fraction of the dataset to use (testing). |
| `--max_steps` | 11 | Training steps. |
| `--learning_rate` | 0.0005 | Learning rate. |
| `--batch_size` | 2 | Batch size. |
| `--max_token_block_size` | 1762 | Max token block size. |
| `--r` | 8 | LoRA rank. |
| `--lora_alpha` | 16 | LoRA alpha. |
| `--lora_dropout` | 0.05 | LoRA dropout. |
| `--target_modules` | `q_proj,k_proj,v_proj,o_proj` | LoRA target modules. |
| `--steps_per_checkpoint` | 100 | Checkpointing frequency. |
| `--adapter_type` | `none` | `lora` / `tokenformer` / `none` (full-parameter). |
| `--optimizer_type` | `adamw` | `adamw` / `sgd` / `rmsprop` (new ones need a server-side training-loop edit). |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation. |
| `--gradient_checkpointing` | True | Gradient checkpointing. |
| `--distribution_strategy` | `fsdp` | `fsdp` (sharded), `ddp` (replicated), or `pytorch_fsdp` (PyTorch FSDP2 — faster than `fsdp`, adds HSDP). |
| `hsdp_shard_size` | 1 | [pytorch_fsdp] >1 builds a 2D replicate×shard mesh — FSDP within each group, DDP across groups. Must divide the world size. Pass via `train_args`. |
| `reshard_after_forward` | False | [pytorch_fsdp] Free unsharded params after forward and re-gather for backward: lower peak memory, one extra collective. Pass via `train_args`. |
| `--attn_implementation` | `flash_attention_2` | `flash_attention_2` / `sdpa` / `eager`. |
| `--freeze_layer_keywords` | vision-related keywords | Comma-separated layer keywords to freeze. |
| `--num_labels` | None | [classification] Label count (derived from data when unset). |
| `--classification_dropout` | 0.1 | [classification] Head dropout. |
| `--label_smoothing` | 0.1 | [classification] Label smoothing. |
| `--upload_to_hf` | False | Push the trained model to the Hub. |
| `--hf_repo_id` | `farbodtavakkoli/scalarlm-test` | Hub repo for uploads — change to your own. |
| `--hf_upload_token` | `$HF_TOKEN` | Hub token (from `dev.env`). |
| `--gpus` | 1 | GPUs per Kubernetes pod — keep 1. |
| `--nodes` | 2 | Number of nodes for the job. |

### inference.py

| Flag | Default | Meaning |
|---|---|---|
| `--test_data_path` | `data/classification_data.json` | Test data (JSON list) with `prompt` fields. |
| `--model_name` | *(a sample job hash)* | Fine-tuned model id from training; remove from `llm.generate` for the base model. |
| `--max_tokens` | 500 | Input + output token budget per request. |
| `--batch_size` | 2 | Prompts per batch. |
| `--inference_mode` | `language_model` | `language_model` or `embedding`. |
| `--model_type` | `rnj-1` | Chat-template family for prompt formatting. |

## Output

- Training: job status to `scalarlm_training/runs/<timestamp>/train_status.json` (contains the `model_name` job hash). If that directory already exists — two submissions inside the same second — a `-2`, `-3`, … suffix is used rather than failing.
- Inference: `scalarlm_inference/runs/<timestamp>/inference_results.json` with ground-truth, prompt, and generated response per record.

## Server image

One source tree, one `ml/` recipe, two server images. Runbooks: **`DOCKER_IMAGE_MI355.md`** (AMD)
and **`DOCKER_IMAGE_H100.md`** (NVIDIA).

| Tag | Hardware | Status on this branch |
|---|---|---|
| `farbodatdocker/scalarlm:mi355-v1.6` | AMD MI355X — gfx950 / ROCm 7.2.4 | built from this branch @ `6bc7d82`; gated from the image itself (server and client, no `ml/` mount) before publish |
| `farbodatdocker/scalarlm:h100-v1.5` | NVIDIA H100 — sm_90 / CUDA 13 | built from this branch @ `6bc7d82`; verified on 8×H100 — `ddp`/`fsdp` bit-identical to the pre-merge `nvidia` branch |

Both images are built from the same source revision by the same `repo/Dockerfile`. Build either
with `build_image.sh`; it auto-detects the vendor, or set it explicitly:

```bash
TARGET=amd    IMAGE_TAG=mi355-v1.6 ./build_image.sh
TARGET=nvidia IMAGE_TAG=h100-v1.5  ./build_image.sh
```

> ### H100 status after the amd/nvidia merge
> This branch unifies the two hardware ports. Two things changed for the H100 side: collectives now
> come from `cray_infra.training.distributed` (torch.distributed/NCCL, reads `RANK` first) instead of the
> `gpu_aware_mpi` shim, and the launcher is `mpirun -> torchrun` per node instead of `mpirun -> python`.
> `h100-v1.5` is the first image built from this tree on an H100 host: `ddp` and `fsdp` losses are
> bit-identical to the pre-merge `nvidia` branch, `pytorch_fsdp` agrees to ~1.7e-5 (it reduces at a
> different granularity), and classification at `batch_size > 1` and LoRA complete. Two build-side
> gaps surfaced and are fixed in the Dockerfile (a venv `torchrun` shim, an explicit `scikit-learn`
> install); neither touches `ml/`. `flash_attention_2` crashes on H100 as it does on MI355X, so the
> unconditional `sdpa` mapping stands on both.

The source revision an image was built from is recorded inside it as
`org.opencontainers.image.revision` — read it with `docker inspect`, don't infer it from the tag.

Both images are self-starting (`docker run <image>` serves; passing a script still overrides).

**NVIDIA H100:**

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SCALARLM_MAX_GPUS_PER_NODE=8 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.5
# curl http://localhost:8000/v1/health -> {"api":"up","vllm":"up","all":"up"}
```

Inference note (H100): tensor-parallel works for the **base** model; **adapters must be served at
tensor-parallel size 1** — vLLM's TP path rejects the hot-reloaded un-sharded adapter state dict.

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
  farbodatdocker/scalarlm:mi355-v1.6
```

AMD notes: `--device=/dev/kfd --device=/dev/dri` replaces `--gpus`; `--shm-size=64g` is required
(RCCL's shared-memory transport deadlocks at the 64 MB default); and
**`SCALARLM_MAX_GPUS_PER_NODE` defaults to 1** — without it every job is silently capped to a single
GPU regardless of what `--gpus` says.

### Config channel

All settings travel in `train_args`, which the server materializes into the job config that
`get_job_config()` reads during training. There is no local sidecar YAML. Two groups must be sent
**nested** — `lora_config` and `classification` — and `train.py` does this for you.

> ### DDP/FSDP semantics change — read before comparing to older results
> Gradients are now synchronised and **averaged** across ranks. Previously DDP reduced 0 of 310
> gradient tensors, so an 8-GPU job was really 8 independent models with only rank 0 checkpointed.
> It is now **one model at 8× the effective batch size**. The same averaging was applied to FSDP.
> Loss curves and final results legitimately differ from any history recorded before this fix.

> ### `flash_attention_2` is downgraded, not supported
> `--attn_implementation flash_attention_2` is silently mapped to `sdpa`. FA2 is not a supported
> training backend here: it loads cleanly and then aborts mid-training with a device-side
> `HSA_STATUS_ERROR_EXCEPTION` (no Python traceback, dead 8-GPU job). The CLI default is `sdpa`.

## Hardware support & evidence

| Layer | AMD MI355X / ROCm | NVIDIA H100 / CUDA |
|---|---|---|
| Client (`train.py` / `inference.py`) | pure Python, hardware-agnostic | same |
| Server image | `mi355-v1.6` — measured on this branch | `h100-v1.5` — measured on this branch |
| Collectives | RCCL via `torch.distributed` | NCCL via `torch.distributed` |

**Other hardware (upstream claims — not verified here):** the client is hardware-agnostic (pure HTTP); server-side, upstream documents NVIDIA (A100/H100) and AMD (MI300X production).


This folder is a **client** — training hardware is the server deployment's concern, and the client runs anywhere Python runs. For the server side, the ScalarLM project (www.scalarlm.com) describes the stack as vLLM for inference plus Megatron-LM for distributed training, dispatched via Slurm inside Kubernetes, and states it is GPU-agnostic: it "runs on NVIDIA and AMD GPUs without code changes", with production deployments at TensorWave on AMD MI300X and the same Helm charts and `ml/` directory working on NVIDIA A100 and H100 clusters. Consult the ScalarLM docs for deploying a server on your hardware.

## MI355X box (ROCm 7.2) — client test

Client-side verification on an 8×MI355X / ROCm 7.2.4 / Python 3.12.3 host (August 2026). No GPUs were used — the client is pure Python and hardware-agnostic; this section tests the **client**, not a training deployment.

**Installed & verified** (venv `.env_train_scalarlm`, `pip install -r requirements_scalarlm.txt`):
- `scalarlm 1.151` (PyPI; the SDK's real package is `masint` — `scalarlm` re-exports it), `python-dotenv 1.2.2`, `PyYAML 6.0.3` — all pins install as-is on Python 3.12, no adjustments needed.
- `import scalarlm` + `SupermassiveIntelligence()` construction: OK. URL resolution confirmed in the SDK: `SCALARLM_API_URL` → `MASINT_API_URL` → default `http://localhost:8000`.
- `scalarlm --help` CLI: OK (`logs plot ls squeue stats clear_queue cancel delete`).
- `train.py --help` / no-server dry run: dataset formatting, config write and archive creation all succeed; the run stops exactly at the network call with `aiohttp ClientConnectorError: Cannot connect to host localhost:8000` — expected without a server. *(Recorded before the config-channel change; the sidecar write it mentions no longer happens.)*
- `inference.py` without a server degrades as documented: each batch logs the traceback, is skipped, and an empty results file is written.

**Local CPU dev server attempt** (`gdiamos/scalarlm-cpu:latest`, ~1.4 GB, no GPUs): came up healthy (`/v1/health` 200; serves `masint/tiny-random-llama`) after pre-seeding the HF cache (container→HF CDN downloads timed out on this host). `train.py` then **submitted successfully end-to-end**: job `QUEUED`, `model_name` job hash returned, `train_status.json` saved. Server-side execution failed with `ModuleNotFoundError: No module named 'cray_infra.huggingface'` — this folder's vendored `ml/` tree (shipped with each job) targets the **fork's** server, while the upstream `latest` container lacks that module. True end-to-end training needs a server built from the matching fork (or a deployment whose `cray_infra` matches this `ml/` tree). Container was removed after the test.

**Gotchas found:**
- `--sample_fraction 0.1` on the 5-example `data/llm_training_sample.json` truncates to 0 rows and the SDK rejects the empty archive (`ValueError: The file … is empty`) — use `--sample_fraction 1.0` with the shipped sample.
- `scalarlm ls` CLI crashes (`UnboundLocalError` in `masint/cli/ls.py`) when the server has no models to list — SDK bug, not a folder issue.

**Verdict: tested on MI355X.** The client installs and runs on this box — imports, CLI, formatting, config write, and job submission all verified, with submission proven live against a local CPU dev server. End-to-end training additionally requires a ScalarLM server deployment matching this folder's `ml/` fork; a full end-to-end re-test against such a server is planned.

### 8-GPU run (8× MI355X, ROCm 7.2.4) — DONE

> Supersedes the earlier "not applicable" note. That was written when no matching server image
> existed on this box. One now does (`farbodatdocker/scalarlm:mi355-v1.0`), and the full end-to-end
> exercise has been run.

The client is still a thin HTTP job-submitter — multi-GPU behaviour is a property of the **server**
deployment. What changed is that the server now exists here, so the end-to-end path was measured
rather than deferred.

Workload: Qwen3-0.6B, 128 records, 30 steps, `adapter_type none` (full-parameter, maximum collective
traffic), `gpus: 8, nodes: 1`, same image as both server and client.

| Strategy | Median step | Final loss (30 steps) |
|---|---|---|
| `fsdp` | 0.241–0.251 s | `0.0038679696153849363` |
| `ddp`  | 0.160–0.182 s | `0.005165183916687965` |
| `pytorch_fsdp` | 0.177–0.181 s | `0.0036232012789696455` |

`pytorch_fsdp` is PyTorch's own FSDP2 (`fully_shard`). It is **1.36× faster than `fsdp`** at this
size because it buckets and overlaps its collectives, where SimpleFSDP issues one ungapped
all-gather / reduce-scatter per wrapped module. It also brings HSDP (`hsdp_shard_size`) for
multi-node.

`ddp` is marginally faster than FSDP2 **at 0.6B only**, and the gap reverses well before any model
size worth training seriously. On Qwen3-4B, 8 steps, same 8 GPUs:

| Strategy | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| `ddp` | **0.159 s** | 0.282 s |
| `pytorch_fsdp` | 0.176 s | **0.221 s** |
| `fsdp` | 0.245 s | 0.312 s |

At 4B FSDP2 is the fastest of the three — 1.28× faster than `ddp` and 1.41× faster than `fsdp`.

Sharding is real, not nominal. Measured on Qwen3-4B across 8 ranks with
`torch.cuda.max_memory_allocated()`: replicated, each rank holds 100% of the parameters and peaks at
**37.8 GB**; under `fully_shard` each rank holds **12.5%** — exactly `1/world_size` — and peaks at
**12.6 GB**. On 288 GB cards that puts the `ddp` ceiling near ~30B parameters, with roughly 8× the
headroom when sharded. (The ceiling is arithmetic from those two measurements; a model past it has
not been run.)

Its loss differs from `fsdp` in the 3rd significant figure. That is expected and not a defect: the
two shard at different granularity (per transformer block vs per module), so the bf16 reductions
happen in a different order. Each strategy is bit-reproducible against itself — all three
`pytorch_fsdp` runs returned exactly `0.0036232012789696455`.

Both reproduce bit-exactly across repeat runs. These values are **current as of the `pack()`
tail-padding fix**, which retains 390 tokens per epoch that were previously discarded and takes the
benchmark dataset from 52 blocks to 53. The earlier references — `0.067634217441082` (fsdp) and
`0.07000554352998734` (ddp) — belong to the pre-fix tree and are not comparable. See
`AWS_ScalarLM_port.md` §8.

All 8 GPUs confirmed engaged (per-device VRAM sampled during a live FSDP run: ~12.8 GB on ranks 1–7;
GPU 0 higher because it also hosts vLLM). Also verified on this image: LoRA/FSDP completes
(0.38 % of parameters trainable), the collective suite passes at 8 ranks, and the serving smoke
tests pass.

**Not verified:** `gradient_accumulation_steps > 1`. Multi-node, embedding mode, diffusion mode,
DDP resume, HF upload and a sharding-sized model (Qwen3-32B) have all since been run — see
"Backend internals" below and `DOCKER_IMAGE_MI355.md` §5.

## Backend internals (vendored `ml/`)

The `ml/` tree is uploaded with each job and executed on the cluster, so its behaviour is what
actually runs. The code there is kept lightly commented on purpose; the reasoning lives here.

### Distribution strategies

| `distribution_strategy` | What it is | Use it when |
|---|---|---|
| `ddp` | Full replica per GPU | Small models. Fastest at 0.6B; **OOMs at 32B** |
| `fsdp` | SimpleFSDP, wraps every module owning parameters | Legacy default; see the LoRA caveat |
| `pytorch_fsdp` | PyTorch FSDP2 `fully_shard`, per transformer block | Large models; fastest at 4B and 32B |

Measured on 8x MI355X: DDP is fastest at 0.6B (0.158 s/step), slower than FSDP2 at 4B
(0.235 vs 0.175), and cannot run 32B at all (`CUDA out of memory`, 254 GiB of 288 GiB). Both
sharded strategies train 32B; FSDP2 is ~2x faster than SimpleFSDP there (0.523 vs 1.100 s/step).

**Do not pair `fsdp` with LoRA.** SimpleFSDP issues one all-gather per wrapped module, and LoRA adds
~1000 tiny modules that add almost no compute: wrapped-module count goes 397 -> 1379 (3.47x more
collectives) while FSDP2 stays at 28 blocks. That is the entire ~2x LoRA slowdown; `pytorch_fsdp`
is unaffected.

**SimpleFSDP and direct `.weight` access.** SimpleFSDP only unshards a module's parameters inside
that module's *own* forward. A parent that reads a child's `.weight` directly — weight tying, a
custom logit head, DiffusionGemma's self-conditioning `matmul(probs, embed_tokens.weight)` — would
otherwise get the raw shard and fail on shape. `FSDPLayer.__getattr__` gathers on demand through the
autograd-aware path so gradients still flow, and the next `free_params()` re-shards.

**`reshard_after_forward`** (FSDP2) defaults to `False`, which is what the shipped fingerprints were
measured against. `True` lowers peak memory at the cost of a re-all-gather in backward; it is
verified standalone but not through the full loop, so re-measure rather than assume the loss is
unchanged.

**Gradient checkpointing** is owned solely by `load_model.py`, which calls HF's
`gradient_checkpointing_enable(use_reentrant=False)` on the bare model before PEFT wraps it (order
matters, or LoRA adapters get zero gradient). The FSDP2 strategy deliberately adds no
`checkpoint_wrapper`: double-wrapping the same blocks makes recomputation disagree with the original
forward and kills backward on every rank.

### Model class routing

`load_model.py` picks the model class in this order, and the order is load-bearing:

1. `training_mode: classification` -> `AutoModelForSequenceClassification`
2. `is_diffusion(config)` -> `DiffusionGemmaForBlockDiffusion`
3. `is_multimodal(config)` -> `AutoModelForImageTextToText`
4. otherwise -> `AutoModelForCausalLM`

Diffusion must be checked **before** multimodal. A real DiffusionGemma checkpoint carries a
`vision_config`, so `is_multimodal` is also true and it would be silently misrouted to
`AutoModelForImageTextToText`; a synthetic/default config has no `vision_config` and instead falls
through to `AutoModelForCausalLM`, which rejects `DiffusionGemmaConfig` outright. The
`DiffusionGemmaForBlockDiffusion` import is function-local so images whose `transformers` lacks
`diffusion_gemma` still run every other training mode. No `trust_remote_code` is involved —
`transformers` 5.12.1 ships `diffusion_gemma` natively.

Weights load straight onto the target GPU (`device_map`, `low_cpu_mem_usage`) to avoid a transient
~2x memory peak from a CPU copy plus `.to(device)`; the CPU path is unchanged.

Classification head dropout is set on the **config**, not passed to `from_pretrained` — transformers
5.x builds these classifiers through a generic wrapper that rejects unknown kwargs — and the
attribute name is architecture-specific, so an unknown name warns rather than failing the job.

### Adapter dtype (tokenformer / LoRA)

FSDP2 requires a uniform dtype across a parameter group's **trainable** parameters (frozen ones are
excluded). Tokenformer is not a pure adapter: it unfreezes base modules (`v_proj`, `o_proj`, the
layernorms) which are bf16, so creating its own parameters from `torch.get_default_dtype()` (float32)
made the trainable set mixed and `fully_shard` refused it — `adapter_type: tokenformer` could not run
on `pytorch_fsdp` at all. Adapter parameters therefore take the **host layer's** dtype, falling back
to the global default and then float32 when the layer exposes no floating-point parameter (the
quantized/NVFP4 case, where packed weights cannot carry gradients).

PEFT LoRA is exempt from this because it freezes the entire base, leaving only its own float32
adapters trainable — mixed-dtype blocks are fine, mixed-dtype *trainable* sets are not.

### Tokenformer and multimodal towers

Adapting vision/audio towers is unsupported: they stay frozen at base weights. Lifting that needs
per-tower `hidden_size` handling, a serving-side key mapping (vLLM has no counterpart for the names
the trainer produces), and a training signal that actually reaches them — a text-only dataset never
activates them. `is_non_language_path()` is shared by the surgeon and the trainer's freeze pass so
both agree on what "language model" means.

### Training modes

- **causal** (default) — packs many documents per block. `position_ids` reset per document so RoPE
  sees the right index, and `document_ids` become a block-diagonal attention mask so packed
  documents cannot attend across each other. The tail is padded to a whole block rather than
  truncated: `map(batched=True)` runs per chunk, so truncating emits short blocks and `torch.stack`
  then fails at `batch_size > 1`. Padding is loss-safe (labels `-100`, mask `0`, own document id).
- **classification** — swaps the LM head; takes precedence over the multimodal wrapper.
- **embedding** — needs `sentence_transformers` **and records shaped
  `{sentence1, sentence2, score}`**. CoSENT is a *pairwise ranking* loss that compares pairs within
  a batch, so **`batch_size: 1` yields exactly 0.0 loss at every step while still reporting
  COMPLETED**. Always use `batch_size >= 2`.
- **diffusion** — auto-detected from the model config (`model_type: diffusion_gemma`), not from a
  flag. Runs on `ddp` and `pytorch_fsdp`; **fails on `fsdp`** for the direct-`.weight` reason above.
  Self-conditioning runs a no-grad pass to predict the clean canvas and feeds it back on the
  gradient-carrying pass for a random `sc_prob` subset, matching the serve-time sampler;
  `sc_prob: 0` disables it.

`ml/nara_offline_eval.py` is a standalone offline evaluator for noise-aware LoRA (NaRA) on
DiffusionGemma. Serving NaRA is deferred, so it runs the model's own block-diffusion `generate()`
in-process with the adapter injected and scores the decode against a golden string, comparing three
decodes from one seed: base, LoRA-only (mapper forced to identity), and NaRA. Run it inside the
container with `--checkpoint`, `--model`, `--prompt`, `--golden`, `--seed`.

### Checkpointing and resume

Resume is automatic: on startup, if any `checkpoint_*.pt` exists in the job directory, the loop
loads the highest-numbered one and continues at `step + 1`. Verified end to end — a job resumed from
`checkpoint_50` and continued at step 51.

Two things to know: relaunch must go through `sbatch` (the entrypoint reads `SLURM_JOB_NODELIST` and
dies as a bare shell script), and **rotation keeps only the last 3 checkpoints**, so a crash more
than three intervals back cannot be resumed from.

Under a sharded strategy each rank holds only its slice of the model and optimizer state, and
`save_checkpoint()` is rank-0 only — so both the optimizer state and the CUDA RNG state are gathered
on **every** rank before the save. One RNG state per *rank*, not per visible device: a rank-0-only
capture would rewind ranks 1..N to their initial seed on resume.

`backward()` is always called, even on NaN/Inf loss, because its job here is freeing the saved
activations; skipping it leaks the whole forward graph. Weights stay safe because `optimizer.step()`
is skipped for that step and `zero_grad()` clears the NaN gradients.

### Multi-node

**Pin `NCCL_IB_HCA` to devices ACTIVE on every node.** The launcher now derives this automatically
as the intersection across the allocated nodes, and an operator-set value always wins.

This matters because RCCL builds its rings from each rank's device list: if two nodes enumerate
*different* RDMA devices, the first collective hangs **forever with no error and no timeout**. Our
nodes are asymmetric out of the box (one has an extra `ionic5`, the other an extra management NIC),
which looked for a long time like broken container RDMA. It was not — with a symmetric device set,
a 2-node x 8-GPU job passes (`ALLREDUCE OK world=16 value=136`), and real training runs 30 steps
across both hosts and writes a checkpoint. Isolation showed `NCCL_IB_HCA` is necessary *and*
sufficient; `NCCL_SOCKET_IFNAME` alone still hangs.

Three environment traps, none of which produce an obvious message:

- `scontrol show hostnames` must return names that resolve for mpirun.
- The HF cache must be **node-local**, never a shared NFS mount: `huggingface_hub` guards it with
  `filelock`, and `flock()` on NFS fails with `OSError: [Errno 116] Stale file handle`, killing every
  rank during model load.
- Container `--device` lists must be computed per node, or `docker run` fails on a device the other
  node does not have.

### Config plumbing

Every key must be declared on `JobConfig` in `repo/infra/cray_infra/util/default_job_config.py`.
Pydantic runs with `extra="ignore"`, so an undeclared key sent in `train_args` is silently dropped
between `config.yaml` and the training code — the failure is invisible, not an error. This has bitten
`trust_remote_code`, `hsdp_shard_size`, and `use_rslora` (which produced a silent numeric divergence:
training scaled by `alpha/r` while merging used `alpha/sqrt(r)`). `get_job_config()` now logs any
config key it had to drop.

Classification and NaRA settings must be sent **nested** (`classification.num_labels`, etc.), and
every numeric CLI argument in `train.py` needs an explicit `type=` — without it a passed flag stays a
string and the server fails comparing it numerically.

### HuggingFace upload

`upload_to_hf` is rank-0 only and deliberately best-effort: a failed upload must not fail a finished
training run. The consequence is that **a failed upload still reports COMPLETED**, with the reason
only in the rank-0 log.

The image sets `HF_HUB_OFFLINE=1`, which blocks the upload outright. A job that explicitly asked to
upload has asked for network, so the upload path clears the offline flags for its duration and
restores them afterwards. Clearing the environment alone is not enough — `huggingface_hub` caches
`HF_HUB_OFFLINE` into `constants` at import time — so both the constant and the env var are
overridden.

### GPU discovery

`gres.conf` generation reads `/sys/class/drm`, which is the **host's** view. A container given only a
subset of GPUs would advertise render nodes it cannot open, and `slurmd` exits 1 at startup.
Discovery now gates on the device node actually existing; on a whole-box deployment this is a no-op.

## Notes & troubleshooting

- **Client/server split:** `SupermassiveIntelligence()` connects to `SCALARLM_API_URL`; `.train(dataset, train_args)` and `.generate(prompts, model_name, max_tokens)` submit jobs.
- **Config handoff:** `train.py` sends the run's adapter/optimizer/distribution settings in `train_args`; the server materializes them into the job config that the training loop reads via `get_job_config()`. There is no local sidecar config file.
- **Job identity:** jobs are keyed by their config; identical configs are treated as the same job (see the dedup gotcha above).
- **`ml/` is upstream code:** do not expect it to follow this repo's code style; changes there ship to the cluster with each job submission, which is ScalarLM's mechanism for customizing the training loop.
- Failed inference batches are skipped with a traceback and the run continues; results only include successfully processed indices.
