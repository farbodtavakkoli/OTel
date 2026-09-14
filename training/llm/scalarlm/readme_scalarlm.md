# training/llm/scalarlm — ScalarLM remote-training client

## Overview & when to use

Train and run inference on a remote **ScalarLM** server. `train.py` and `inference.py` are thin **clients** — they format data locally and submit jobs to a ScalarLM deployment via the `scalarlm` SDK (`SupermassiveIntelligence`). The actual training runs server-side on the deployment's GPUs/nodes; nothing in this folder needs a GPU.

Supports three training modes — **language_model**, **embedding**, **classification** — across many model families (qwen3, llama3, gemma3, rnj-1, olmo3, mistral, lfm, phi4, gpt-oss), with LoRA / tokenformer / full-parameter training and FSDP or DDP.

Files:
- `train.py` — format a dataset and submit a training job.
- `inference.py` — submit a batch generate/embedding job against a (fine-tuned) model.
- `ml/` — **the training recipe.** Runs server-side (`cray_megatron`, `adapters`, `tokenformer`). This is the copy you edit: the SDK uploads whatever `ml/` sits next to you, so a change here takes effect on your next job with no image rebuild. Vendored and deliberately left in upstream style.
- `repo/` — the full ScalarLM **server** source with the AMD integration applied. Only needed to rebuild the image or change something outside `ml/` (the collectives layer, launcher, Dockerfile). It carries no `.git`; `repo/ml/` is generated at build time and is not source.
- `docs/build_image.sh` — stages `ml/` into `repo/` and builds the MI355X image.
- `docs/scalarlm_mi355.patch` — the AMD integration as one patch against public upstream
  `supermassive-intelligence/scalarlm` at commit `4566a84`. Applying it to a fresh upstream clone
  at that commit reproduces `repo/` — see `docs/DOCKER_IMAGE_MI355.md` §7.
- `docs/DOCKER_IMAGE_MI355.md` — build / run / publish runbook for the MI355X server image.
- `data/` — small sample datasets for each mode.

> **There is exactly one `ml/` tree in version control** — this one. `repo/ml/` is created by
> `docs/build_image.sh` at build time and is gitignored, so the image always bakes the recipe you see
> here and the two cannot drift.

> The vendored trees here derive from `supermassive-intelligence/scalarlm`. The client talks to
> whatever server `SCALARLM_API_URL` points at.

## Install

The client needs the `scalarlm` SDK plus two helpers:

```bash
python3.12 -m venv .env_scalarlm && source .env_scalarlm/bin/activate
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

> **Note — job dedup:** ScalarLM keys jobs by their config. If you resubmit with an unchanged config it recognizes the job as already-run and won't retrain. Change at least one setting (e.g. bump `--max_steps`) to force a new job.

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

One source tree, one `ml/` recipe, two server images. Runbooks: **`docs/DOCKER_IMAGE_MI355.md`**
(AMD) and **`docs/DOCKER_IMAGE_H100.md`** (NVIDIA).

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:mi355-v1.7` | AMD MI355X — gfx950 / ROCm 7.2.4 |
| `farbodatdocker/scalarlm:h100-v1.6` | NVIDIA H100 — sm_90 / CUDA 13 |

Both images are built from the same source revision by the same `repo/Dockerfile`. Build either
with `docs/build_image.sh`; it auto-detects the vendor, or set it explicitly:

```bash
cd docs
TARGET=amd    IMAGE_TAG=mi355-v1.7 ./build_image.sh
TARGET=nvidia IMAGE_TAG=h100-v1.6  ./build_image.sh
```

On 8×H100, `h100-v1.6` covers `ddp`, `fsdp` and `pytorch_fsdp` training plus classification at
`batch_size > 1` and LoRA. `flash_attention_2` crashes on H100 as it does on MI355X, so the
unconditional `sdpa` mapping applies on both.

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
  farbodatdocker/scalarlm:h100-v1.6
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
  farbodatdocker/scalarlm:mi355-v1.7
```

AMD notes: `--device=/dev/kfd --device=/dev/dri` replaces `--gpus`; `--shm-size=64g` is required
(RCCL's shared-memory transport deadlocks at the 64 MB default); and
**`SCALARLM_MAX_GPUS_PER_NODE` defaults to 1** — without it every job is silently capped to a single
GPU regardless of what `--gpus` says.

### Config channel

All settings travel in `train_args`, which the server materializes into the job config that
`get_job_config()` reads during training. There is no local sidecar YAML. Two groups must be sent
**nested** — `lora_config` and `classification` — and `train.py` does this for you.

> ### `flash_attention_2` is downgraded, not supported
> `--attn_implementation flash_attention_2` is silently mapped to `sdpa`. FA2 is not a supported
> training backend here: it loads cleanly and then aborts mid-training with a device-side
> `HSA_STATUS_ERROR_EXCEPTION` (no Python traceback, dead 8-GPU job). The CLI default is `sdpa`.

## Hardware support

| Layer | AMD MI355X / ROCm | NVIDIA H100 / CUDA |
|---|---|---|
| Client (`train.py` / `inference.py`) | pure Python, hardware-agnostic | same |
| Server image | `mi355-v1.7`, built from this repo | `h100-v1.6`, built from this repo |
| Collectives | RCCL via `torch.distributed` | NCCL via `torch.distributed` |

This folder is a **client** — training hardware is the server deployment's concern, and the client
runs anywhere Python runs.

## Client-side notes

The client needs no GPU. `scalarlm` resolves its endpoint as `SCALARLM_API_URL` →
`MASINT_API_URL` → default `http://localhost:8000`. The SDK's real package is `masint`; `scalarlm`
re-exports it and ships the CLI (`logs plot ls squeue stats clear_queue cancel delete`).

Without a reachable server, `train.py` still formats the dataset and builds the job archive, then
stops at the network call with `aiohttp ClientConnectorError: Cannot connect to host …`;
`inference.py` logs the traceback per batch, skips it, and writes an empty results file.

**The server must match this folder's `ml/` tree.** `gdiamos/scalarlm-cpu:latest` (~1.4 GB, no
GPUs) is useful for exercising submission — it comes up healthy and serves
`masint/tiny-random-llama`, and a job reaches `QUEUED` with a `model_name` hash returned — but
server-side execution then fails with `ModuleNotFoundError: No module named
'cray_infra.huggingface'`, because the uploaded `ml/` targets this integration's server. Real
training needs a server built from `repo/` (or a deployment whose `cray_infra` matches this `ml/`).
Pre-seed the HF cache before starting a container if outbound CDN downloads are slow or proxied.

**Client notes:**
- `--sample_fraction 0.1` on the 5-example `data/llm_training_sample.json` truncates to 0 rows and
  the SDK rejects the empty archive (`ValueError: The file … is empty`) — use
  `--sample_fraction 1.0` with the shipped sample.
- `scalarlm ls` crashes (`UnboundLocalError` in `masint/cli/ls.py`) when the server has no models
  to list. SDK bug, not a folder issue.

### Multi-GPU behaviour (8× MI355X, ROCm 7.2.4)

The client is a thin HTTP job-submitter — multi-GPU behaviour is a property of the **server**
deployment. Choosing `--distribution_strategy`: `ddp` is fine at ~0.6B and OOMs at 32B; from a few
billion parameters upwards use `pytorch_fsdp` (FSDP2, also the only sharded strategy with HSDP via
`hsdp_shard_size`). Each strategy is bit-reproducible against itself; the reference loss values
are in `docs/DOCKER_IMAGE_MI355.md` §4.

## Backend internals (vendored `ml/`)

The `ml/` tree is uploaded with each job and executed on the cluster, so its behaviour is what
actually runs.

### Distribution strategies

| `distribution_strategy` | What it is | Use it when |
|---|---|---|
| `ddp` | Full replica per GPU | Small models; **OOMs at 32B** |
| `fsdp` | SimpleFSDP, wraps every module owning parameters | Legacy default; see the LoRA caveat |
| `pytorch_fsdp` | PyTorch FSDP2 `fully_shard`, per transformer block | Large models |

**Do not pair `fsdp` with LoRA** — SimpleFSDP issues one all-gather per wrapped module and LoRA
adds ~1000 tiny ones. Use `pytorch_fsdp`, which is unaffected.

**`reshard_after_forward`** (FSDP2) defaults to `False`. `True` lowers peak memory at the cost of a
re-all-gather in backward; it is not verified through the full loop, so re-measure rather than
assume the loss is unchanged.

**Gradient checkpointing** is owned solely by `load_model.py`, which enables it on the bare model
before PEFT wraps it — order matters, or LoRA adapters get zero gradient. The FSDP2 strategy adds
no `checkpoint_wrapper`; double-wrapping the same blocks kills backward on every rank.

### Model class routing

`load_model.py` picks the model class in this order, and the order is load-bearing:

1. `training_mode: classification` -> `AutoModelForSequenceClassification`
2. `is_diffusion(config)` -> `DiffusionGemmaForBlockDiffusion`
3. `is_multimodal(config)` -> `AutoModelForImageTextToText`
4. otherwise -> `AutoModelForCausalLM`

Diffusion must be checked **before** multimodal: a real DiffusionGemma checkpoint also carries a
`vision_config`, so it would otherwise be misrouted to `AutoModelForImageTextToText`. No
`trust_remote_code` is involved.

### Adapter dtype (tokenformer / LoRA)

FSDP2 requires a uniform dtype across a parameter group's **trainable** parameters, so adapter
parameters take the **host layer's** dtype (falling back to the global default, then float32 when
the layer exposes no floating-point parameter). Without that, `adapter_type: tokenformer` cannot
run on `pytorch_fsdp` at all, since it unfreezes bf16 base modules alongside its own parameters.

### Tokenformer and multimodal towers

Adapting vision/audio towers is unsupported: they stay frozen at base weights.

### Training modes

- **causal** (default) — packs many documents per block. `position_ids` reset per document so RoPE
  sees the right index, and `document_ids` become a block-diagonal attention mask so packed
  documents cannot attend across each other. The tail is padded to a whole block rather than
  truncated: `map(batched=True)` runs per chunk, so truncating emits short blocks and `torch.stack`
  then fails at `batch_size > 1`. Padding is loss-safe (labels `-100`, mask `0`, own document id).
- **classification** — swaps the LM head; takes precedence over the multimodal wrapper. The
  tokenizer is forced to **left** padding, so the sequence the classification head pools ends at the
  last real token rather than at padding.
- **embedding** — needs `sentence_transformers` **and records shaped
  `{sentence1, sentence2, score}`**. CoSENT is a *pairwise ranking* loss that compares pairs within
  a batch, so **`batch_size: 1` yields exactly 0.0 loss at every step while still reporting
  COMPLETED**. Always use `batch_size >= 2`.
- **diffusion** — auto-detected from the model config (`model_type: diffusion_gemma`), not from a
  flag. Runs on `ddp` and `pytorch_fsdp`; **fails on `fsdp`** (SimpleFSDP does not unshard on
  direct `.weight` access). `sc_prob: 0` disables self-conditioning.

`ml/nara_offline_eval.py` is a standalone offline evaluator for noise-aware LoRA (NaRA) on
DiffusionGemma — there is no serving path for NaRA. Run it inside the container with
`--checkpoint`, `--model`, `--prompt`, `--golden`, `--seed`. `--mode` selects what it measures —
`decode` (default), `probe`, `sweep`, `decode-greedy`, `tail-probe`, `gen-vs-tf` or
`decode-recompute` — and `--dtype` (`bf16` or `fp32`) sets the compute precision, which is what you
change to tell a bf16 numerical effect apart from a structural one.

### Checkpointing and resume

Resume is automatic: on startup, if any `checkpoint_*.pt` exists in the job directory, the loop
loads the highest-numbered one and continues at `step + 1`.

Two things to know: relaunch must go through `sbatch` (the entrypoint reads `SLURM_JOB_NODELIST` and
dies as a bare shell script), and **rotation keeps only the last 3 checkpoints**, so a crash more
than three intervals back cannot be resumed from.

### Publishing a trained LoRA

`ml/adapters/merge_lora_and_push.py` folds a trained LoRA back into its base model and pushes the
result to the Hub. Point it at a job directory with `--job-dir` and it reads that job's
`config.yaml` and the highest-numbered `checkpoint_<step>.pt`, unless you name one with
`--checkpoint`. Run `--help` for the full flag list.

Two things the flags do not tell you. **The LoRA rank comes from the saved tensors, not from the job
config** — the merge reads it off the checkpoint's `lora_A` weights, so editing `r` after the fact
changes nothing. And when the merged model is missing weights that exist in the base model on the
Hub, those tensors are copied across verbatim to complete it; **the LoRA delta is not applied to
those layers**, and the run warns rather than fails, so read the log before trusting the upload.

### Multi-node

**Pin `NCCL_IB_HCA` to devices ACTIVE on every node.** The launcher derives this automatically as
the intersection across the allocated nodes, and an operator-set value always wins.

If two nodes enumerate *different* RDMA devices, the first collective hangs **forever with no error
and no timeout** — nodes are commonly asymmetric out of the box, which presents exactly like broken
container RDMA. `NCCL_IB_HCA` is necessary *and* sufficient here; `NCCL_SOCKET_IFNAME` alone still
hangs.

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
between `config.yaml` and the training code — the failure is invisible, not an error.
`get_job_config()` logs any config key it had to drop.

Classification and NaRA settings must be sent **nested** (`classification.num_labels`, etc.), and
every numeric CLI argument in `train.py` needs an explicit `type=` — without it a passed flag stays a
string and the server fails comparing it numerically.

### HuggingFace upload

`upload_to_hf` is rank-0 only and best-effort, so **a failed upload still reports COMPLETED**, with
the reason only in the rank-0 log. The image sets `HF_HUB_OFFLINE=1`; the upload path clears the
offline flags for its own duration and restores them afterwards.

## Notes & troubleshooting

- **Client/server split:** `SupermassiveIntelligence()` connects to `SCALARLM_API_URL`; `.train(dataset, train_args)` and `.generate(prompts, model_name, max_tokens)` submit jobs.
- **Config handoff:** `train.py` sends the run's adapter/optimizer/distribution settings in `train_args`; the server materializes them into the job config that the training loop reads via `get_job_config()`. There is no local sidecar config file.
- **Job identity:** jobs are keyed by their config; identical configs are treated as the same job (see the job-dedup note above).
- **`ml/` is upstream code:** do not expect it to follow this repo's code style; changes there ship to the cluster with each job submission, which is ScalarLM's mechanism for customizing the training loop.
- **Editing the training loop:** the stop check, the finite-gradient check and the loss reduction are all collective — every rank must reach them on every step, so never put one behind a rank-local branch or the job hangs with no error. Reduce the loss outside the forward/backward window, never between them.
- Failed inference batches are skipped with a traceback and the run continues; results only include successfully processed indices.
