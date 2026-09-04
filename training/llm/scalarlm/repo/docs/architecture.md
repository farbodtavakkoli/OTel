# ScalarLM Architecture

ScalarLM is a single-deployment, closed-loop LLM platform: one running process exposes both an OpenAI-compatible inference endpoint and a training submission endpoint, backed by a shared checkpoint store. This document describes how the system is put together — the top-level processes, how requests flow, how training jobs are scheduled, and where each responsibility lives in the tree.

- **License:** CC-0
- **Primary entry point:** `./scalarlm up` → Docker container → `scripts/start_one_server.sh`
- **Runtime shape:** three cooperating processes (API, vLLM, Generate Worker) plus on-demand SLURM-launched Megatron training jobs

---

## 1. System Overview

### 1.1 Guiding idea

Most LLM infrastructure splits cleanly into two systems: one serves a frozen model (inference) and another produces new checkpoints (training). Closing the loop — using live outputs to drive the next round of post-training — normally requires stitching those systems together manually.

ScalarLM collapses that seam. A single container bundles:

- **vLLM** for inference (PagedAttention, continuous batching, OpenAI-compatible HTTP)
- **Megatron/PyTorch** for distributed training (FSDP, DDP, AdamW, gradient accumulation)
- **SLURM** inside the container to schedule training jobs onto the same GPUs
- A **FastAPI control plane** that accepts inference requests, submits training jobs, and brokers the shared checkpoint directory between them (includes a session-based [chunked upload protocol](/docs/chunked-upload.md) for large datasets)


Inference loads from a checkpoint directory; training writes to one. Post-training, the control plane auto-registers the new weights with vLLM (via LoRA adapter hot-loading or a model swap), and the next inference request uses the updated model — no pod restart required.

### 1.2 Deployment targets

| Target | How it runs |
|---|---|
| Developer laptop | `docker-compose up cray-nvidia` (or `cray-amd`, `cray`) |
| Single node | Prebuilt container from `gdiamos/scalarlm-*:latest` |
| Kubernetes cluster | Helm chart in `deployment/helm/gemma_270m_tw/scalarlm/` |

All three targets run the **same container** and the **same Python packages**; only storage, networking, and GPU count differ.

### 1.3 Process topology inside one container

```
                  ┌─────────────────────────────────────────────┐
                  │  scripts/start_one_server.sh                │
                  │    1. start_slurm.sh → slurmctld + slurmd   │
                  │    2. python -m cray_infra.one_server.main  │
                  └───────────────────┬─────────────────────────┘
                                      │
                                      │ asyncio.gather
                     ┌────────────────┼─────────────────┐
                     │                │                 │
             ┌───────▼────────┐ ┌─────▼──────┐ ┌────────▼───────────┐
             │ FastAPI / API  │ │  vLLM      │ │ Generate Worker    │
             │ port 8000      │ │  port 8001 │ │ (pulls work queue) │
             │                │ │            │ │                    │
             │ create_api.py  │ │ create_vllm│ │ create_generate_   │
             │                │ │  .py       │ │ worker.py          │
             └────────┬───────┘ └─────▲──────┘ └────────┬───────────┘
                      │                │                 │
                      │  SQLite work queue (persistqueue.SQLiteAckQueue)
                      │  /app/cray/inference_work_queue.sqlite
                      │                                  │
                      └──────────────────────────────────┘
                                     │
                                     │ POST /v1/megatron/train
                                     ▼
                          ┌─────────────────────┐
                          │   SLURM (sbatch)    │
                          │   train_job_entry   │
                          │   point.sh          │
                          └──────────┬──────────┘
                                     │
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │   ml/cray_megatron/main.py               │
                 │     TrainingHarness → MegatronTrainer    │
                 │     → TrainingLoop → FSDP/DDP → AdamW    │
                 │   Writes /app/cray/jobs/{hash}/          │
                 └──────────────────────────────────────────┘
```

The API process is the only externally-reachable entrypoint (port 8000). vLLM (port 8001) is exposed for debugging but in normal operation only the Generate Worker and the API's chat proxy talk to it.

---

## 2. Repository Layout

```
scalarlm/
├── scalarlm                  # Top-level bash entrypoint (delegates to bashly-generated CLI)
├── cmd/                      # Bashly CLI spec + per-command shell scripts
├── scripts/                  # Runtime entrypoints (start_one_server.sh, train_job_entrypoint.sh)
├── Dockerfile                # Multi-stage build: CPU / NVIDIA / AMD / ARM variants
├── docker-compose.yaml       # cray / cray-nvidia / cray-amd services
├── setup.py, requirements.txt
│
├── infra/cray_infra/         # Python control plane (FastAPI, SLURM bridge, config, queue)
│   ├── api/                  #   HTTP surface + work queue
│   ├── one_server/           #   Process orchestration (API, vLLM, Megatron, Worker)
│   ├── training/             #   Training job lifecycle (submit, status, cancel, register)
│   ├── generate/             #   Inference metrics + queue cleanup
│   ├── util/                 #   Config loading (default_config.py, get_config.py)
│   ├── slurm/                #   SLURM cluster discovery
│   ├── huggingface/          #   HF token + downloads
│   └── adapters/             #   LoRA/tokenformer adapter integration for vLLM
│
├── ml/                       # Training stack (runs inside sbatch jobs)
│   ├── cray_megatron/        #   Entry point, training loop, distribution strategies
│   ├── adapters/             #   LoRA / tokenformer model-wrapping helpers
│   └── tokenformer/          #   Tokenformer model + surgery (dynamic layer replacement)
│
├── sdk/masint/               # Python client (pip install scalarlm → import scalarlm)
│   ├── api/                  #   SupermassiveIntelligence (sync + async)
│   ├── engines/              #   HTTP engine (AsyncCray) + job submission helpers
│   ├── util/                 #   URL builder, aiohttp session
│   └── cli/                  #   `scalarlm logs/plot/ls/squeue/...` CLI
│
├── deployment/               # Helm charts and ansible playbooks for k8s deploy
│   └── helm/gemma_270m_tw/scalarlm/   # api/vllm/megatron deployments, PVCs, cloudflare tunnel
│
├── frontend/                 # Static assets (logo, etc.)
├── ui/                       # First-party React SPA (served at /app/* by FastAPI)
├── vllm/                     # Vendored vLLM fork (branch: scalarlm-on-v0.19.0)
├── models/                   # HF model cache (mounted into container)
├── test/                     # Integration + benchmark suites
└── docs/                     # This documentation
```

---

## 3. The Control Plane (`infra/cray_infra/`)

This is the Python package that runs inside the container as the "API" and "Megatron" processes. It owns the HTTP surface, the inference work queue, training-job submission to SLURM, and configuration.

### 3.1 Process orchestration — `one_server/`

Three small modules start and wire the cooperating processes:

| File | Role |
|---|---|
| `one_server/main.py:98` — `run_all_servers_async()` | Top-level coroutine. Uses `asyncio.wait(..., FIRST_COMPLETED)` so if any subsystem dies the whole container exits and Kubernetes restarts it. |
| `one_server/start_cray_server.py:12` — `start_cray_server()` | Reads `server_list` config and schedules tasks for each enabled subsystem. |
| `one_server/create_api.py` — `create_api()` | Spawns the FastAPI app via Uvicorn on port 8000. |
| `one_server/create_vllm.py` — `create_vllm()` | Builds and runs the vLLM OpenAI-compatible server on port 8001. Uses vLLM's own `build_app` / `init_app_state` functions directly in-process when appropriate. |
| `one_server/create_generate_worker.py` — `create_generate_worker()` | Long-running coroutine that pulls from the work queue, calls vLLM, and writes results back. |
| `one_server/create_megatron.py` — `create_megatron()` | Optional FastAPI app for training-side bookkeeping (used when `server_list=megatron`). |
| `one_server/wait_for_vllm.py` | Health poll so the Worker doesn't pull jobs before vLLM is ready. |

The Uvicorn server is started with hot-reload watching `infra/cray_infra/**/*.py,*.yaml` (`main.py:81-85`). That makes the control plane live-editable inside the dev container without rebuilds.

### 3.2 HTTP surface — `api/fastapi/`

All routes are mounted on a single FastAPI app (`api/fastapi/main.py:32`):

```python
app = FastAPI(lifespan=add_megatron_tasks)
app.include_router(openai_v1_router, prefix="/v1")
app.include_router(megatron_router,  prefix="/v1")
app.include_router(health_router,    prefix="/v1")
app.include_router(generate_router,  prefix="/v1")
app.include_router(slurm_router)
add_ui(app)  # mounts the React SPA at /app/* (StaticFiles + SPA fallback)
```

The `lifespan=add_megatron_tasks` hook (`api/fastapi/tasks/add_megatron_tasks.py`) spins up a periodic task that scans the jobs directory for new checkpoints and registers them with vLLM — this is the "closed loop" seam.

#### 3.2.1 OpenAI-compatible routes — `routers/openai_v1_router.py`

Proxies standard chat/completion calls to the in-process vLLM engine. Clients point any OpenAI SDK at `http://host:8000/v1` and get normal `/v1/chat/completions`, `/v1/completions`, `/v1/models` behavior.

#### 3.2.2 Async inference routes — `routers/generate_router.py`

These power ScalarLM's batched, queue-based inference (used by the SDK for large jobs and by agents that want reliable persistence):

| Endpoint | Handler | Purpose |
|---|---|---|
| `POST /v1/generate` | `generate()` | Enqueue prompts; return request IDs. |
| `POST /v1/generate/get_work` | `get_work()` | Worker-side dequeue (pulls a batch). |
| `POST /v1/generate/finish_work` | `finish_work()` | Worker-side ack with outputs. |
| `POST /v1/generate/get_results` | `get_results()` | Client-side result polling. |
| `POST /v1/generate/upload` / `download` | | Large-batch streaming (≥128 prompts use the file-upload path in the SDK). |
| `POST /v1/generate/clear_queue` | `clear_queue()` | Admin purge. |
| `POST /v1/generate/get_adaptors` | `get_adaptors()` | Discover which LoRA/tokenformer adapters are loaded in vLLM. |
| `GET  /v1/generate/metrics` | `metrics()` | Throughput and latency. |

#### 3.2.3 Training routes — `routers/megatron_router.py`

| Endpoint | Handler | Purpose |
|---|---|---|
| `POST /v1/megatron/train` | `launch_training_job()` | Multipart: dataset file + training args. Writes job dir, submits sbatch. |
| `GET  /v1/megatron/train/{job_hash}` | `get_training_job_info()` | Read `status.json` from the job dir. |
| `GET  /v1/megatron/train/logs/{model_name}` | `training_logs_generator()` | SSE streaming of training stdout. |
| `POST /v1/megatron/cancel/{job_hash}` | `cancel()` | `scancel` the slurm job. |
| `POST /v1/megatron/delete/{job_hash}` | `delete()` | Remove job directory. |
| `GET  /v1/megatron/list_models` | `list_models()` | Enumerate trained checkpoints. |
| `GET  /v1/megatron/squeue` | `squeue()` | Parse `squeue` output. |
| `GET  /v1/megatron/gpu_count` / `node_count` | | Cluster capacity. |

#### 3.2.4 Health & SLURM

- `routers/health_router.py` — liveness/readiness, checks vLLM reachability.
- `routers/slurm_router.py` — introspection endpoints for the embedded SLURM cluster.
- `setup_ui.py::add_ui()` — mounts the React SPA at `/app/*` (StaticFiles for hashed assets, dynamic `/app/api-config.json`, SPA history-routing fallback for unknown paths). See `docs/ui-design.md`.

### 3.3 Inference work queue — `api/work_queue/`

The queue is the critical piece of shared state between the API handler and the Generate Worker.

- **Storage:** SQLite file at `/app/cray/inference_work_queue.sqlite` (configurable via `inference_work_queue_path`).
- **Library:** `persistqueue.SQLiteAckQueue` — gives crash-safe enqueue/dequeue with explicit ack.
- **Wrapper class:** `api/work_queue/inference_work_queue.py:13` — `InferenceWorkQueue`.
- **Operations:** `push_into_queue.py`, `get_work_item.py`, `update_and_ack.py`, `clear_acked_requests_from_queue.py`.

Why SQLite and not Redis? The whole stack runs in one pod by default; an in-pod file avoids another moving part. It also survives container restarts under Kubernetes, so in-flight inference requests are not lost when the API pod is rolled.

Ack timeout is 300s (`inference_work_queue_ack_timeout` in `default_config.py:44`): if a worker pulls work and then dies, the item returns to the queue automatically.

### 3.4 Training lifecycle — `training/`

The sequence from `POST /v1/megatron/train` to running GPUs:

1. **Upload** — `upload_training_data.py` streams the multipart dataset into `/app/cray/jobs/{hash}/dataset/` and parses the JSON training config.
2. **Launch** — `launch_training_job.py:18` — `launch_training_job()`:
   - Writes `config.yaml` into the job dir.
   - Allocates GPUs per `max_gpus_per_node` × requested nodes.
   - Submits `sbatch scripts/train_job_entrypoint.sh {job_hash}`.
   - Returns the initial status (`QUEUED`) to the caller.
3. **Status tracking** — `training_job_status.py:4` defines the enum `QUEUED | TRAINING | COMPLETED | FAILED`. The training job itself writes `status.json` into its directory; `get_training_job_info.py` reads it.
4. **Model registration** — after the job finishes, `register_megatron_workers.py` scans for new checkpoints and `register_megatron_models.py` tells vLLM about them. For LoRA/tokenformer adapters, vLLM hot-loads them via the runtime-update API (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`, set in `one_server/main.py:4`).
5. **Cancellation/delete** — `cancel.py` wraps `scancel`; `delete.py` removes the job directory after checks.

Related:
- `training_job_context.py` — loads the per-job config from disk during training.
- `vllm_model_manager.py` — in-memory tracker of which models/adapters vLLM currently has loaded.
- `gpu_aware_mpi/` — helpers that call into the `gpu_aware_mpi` extension for MPI initialization with NCCL/RCCL-aware ranks.

### 3.5 Configuration — `util/`

A three-tier override system:

1. **Defaults** — `util/default_config.py:5` — `class Config(BaseModel)`. Every setting has a sane default (see below).
2. **YAML** — `/app/cray/cray-config.yaml` (mounted from Helm `values.yaml` in Kubernetes, or from the bind-mounted source directory in Docker).
3. **Environment** — any `SCALARLM_*` env var wins.

Loaded through `util/get_config.py` → returns a dict-like accessor. Per-job config is loaded via `util/get_job_config.py` from the job directory.

Key fields (from `default_config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `model` | `tiny-random/gemma-4-dense` | HF model ID to serve at startup |
| `api_url` | `http://localhost:8000` | Advertised API URL |
| `vllm_api_url` | `http://localhost:8001` | Where the Worker talks to vLLM |
| `training_job_directory` | `/app/cray/jobs` | Per-job scratch root |
| `inference_work_queue_path` | `/app/cray/inference_work_queue.sqlite` | Queue DB |
| `max_gpus_per_node` | 1 | Upper bound per sbatch allocation |
| `max_train_time` | 86400 (24 h) | SLURM walltime |
| `extra_training_seconds` | 300 | Graceful-exit buffer before SLURM kills |
| `gpu_memory_utilization` | 0.40 | vLLM KV cache fraction |
| `max_model_length` | 0 (auto) | Fallback admission cap; the live vLLM context limit is selected and queried at runtime |
| `generate_batch_size` | 1024 | Worker-side batch size |
| `response_timeout` | 60 | Per-request cap on synchronous SDK calls |
| `tokenformer_r`, `tokenformer_num_heads` | 32, 4 | Tokenformer adapter shape |
| `tokenformer_cache_capacity` | 2 | Adapters kept hot in memory |
| `hf_token` / `hf_encrypted_token` | | HF Hub auth (supports at-rest encryption via Fernet) |

### 3.6 Adapters — `adapters/`

`cray_infra/adapters/` contains the control-plane side of adapter management (deciding which adapters to load/unload in vLLM, managing the LRU cache keyed by `tokenformer_cache_capacity`). The *model-side* adapter implementation lives in `ml/adapters/` and `ml/tokenformer/`.

---

## 4. The Training Stack (`ml/cray_megatron/`)

The control plane's `launch_training_job.py` submits an sbatch job that runs `scripts/train_job_entrypoint.sh`, which ultimately calls `python -m cray_megatron.main`. Everything from that point lives in `ml/`.

### 4.1 Entry point — `ml/cray_megatron/main.py`

```python
# main.py:29
def main():
    harness = TrainingHarness()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()
    try:
        setup_logging()
        setup_signal_handler(harness)   # SIGCONT → return to QUEUED
        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
    except Exception as e:
        print_exception()
        harness.update_status(status=TrainingJobStatus.FAILED, metadata={"error": str(e)})
        raise e
    finalize_mpi()

main()
```

Three things worth noting:

1. `TrainingHarness` is constructed first so that any exception during setup still gets recorded to `status.json`.
2. The HF token is injected into the environment from the encrypted token in `default_config.py:66` (decrypted via `huggingface/get_hf_token.py`).
3. `signal.SIGCONT` is wired to re-queue the job — SLURM uses this pattern for preemption-friendly re-submission.

### 4.2 `megatron/training_harness.py` — status bridge

The harness (`training_harness.py:16`) is the *only* piece of the training process that talks to the control plane. It doesn't make HTTP calls; it just writes JSON to the job directory, which the control plane polls.

- `update_status(status, metadata)` → writes `/app/cray/jobs/{hash}/status.json`.
- `checkpoint(checkpoint_state, name)` → writes checkpoint tensors.
- `get_status()` → reads back, used for resumption.

This keeps training code free of network dependencies and makes the system robust to API-pod restarts mid-training.

### 4.3 `megatron/megatron_trainer.py` and `training_loop.py`

`MegatronTrainer` (`megatron_trainer.py:14`) is a thin wrapper that constructs the `TrainingLoop` and invokes it. Almost all logic lives in `training_loop.py:28`:

```python
class TrainingLoop:
    def train(self):
        self.model_manager = get_model_manager()
        self.training_state.model_info = self.model_manager.load_model()
        self.training_loop()
        self.checkpoint()

    def training_loop(self):
        self.on_train_begin()
        self.training_state.model_info["model"].train()

        max_steps = get_max_steps()
        gradient_accumulation_steps = get_gradient_accumulation_steps()
        self.training_state.optimizer = get_optimizer(...)         # AdamW
        self.training_state.scheduler = get_scheduler(...)         # warmup + decay

        if does_any_checkpoint_exist():
            self.resume_from_checkpoint()

        data_loader = DataLoader(model=..., tokenizer=...)
        for step in range(starting_step, max_steps):
            # forward → backward → allreduce (gpu_aware_mpi) → optimizer.step
            ...
```

Design points:

- **Callback system** (`get_callbacks(self)`): extension hook for logging, metric reporting, and checkpointing without editing the loop.
- **`gpu_aware_mpi.allreduce`**: explicit gradient synchronization. The loop is written around MPI ranks rather than purely PyTorch DDP primitives so it composes with SLURM's rank assignment.
- **`@main_rank_only`** (`collectives/main_rank_only.py:26`): decorator used everywhere non-idempotent I/O happens (status writes, logging, checkpoint saves). Applied by `TrainingHarness` internally.
- **Checkpoint on completion** (`train()` final line): guaranteed even if `max_steps` was reached.

### 4.4 Distribution strategies — `megatron/distribution/`

| Strategy | File | When used |
|---|---|---|
| FSDP | `distribution/fsdp.py` | Multi-GPU (primary). `size_based_auto_wrap_policy` so the user doesn't hand-annotate transformer blocks. |
| DDP | `distribution/ddp.py` | Data-parallel fallback for smaller models / older GPUs. |
| None | `distribution/no_distribution.py` | Single-GPU development and CI. |

Dispatch happens in `distribution/apply_distribution_strategy.py` based on job config (`gpus`, `tensor_parallel_size`, etc.).

### 4.5 Data loading — `megatron/dataset/`

| File | Purpose |
|---|---|
| `dataset/load_dataset.py` | Task-type dispatcher (language-model vs. embedding). |
| `dataset/load_language_model_dataset.py` | Causal-LM tokenization + chunking. |
| `dataset/load_embedding_dataset.py` | Contrastive/embedding-task batching. |
| `dataset/data_loader.py` | PyTorch `DataLoader` wrapper; tracks `epoch` for the training loop. |

### 4.6 Models — `models/`

- `models/get_model_manager.py` — factory that picks the right `ModelManager`.
- `models/load_model.py` — HF `from_pretrained` with caching; also handles the checkpoint-resume path.
- `models/get_latest_checkpoint_path.py` + `delete_old_checkpoints` — retention policy.
- `models/tokenformer/tokenformer_model_manager.py` — Tokenformer-specific loader.

### 4.7 Adapters and Tokenformer — `ml/adapters/`, `ml/tokenformer/`

Tokenformer is ScalarLM's preferred alternative to LoRA for post-training. It replaces attention projections with key-value memory banks of configurable size (`tokenformer_r`, `tokenformer_num_heads`).

- `ml/adapters/create_lora_model.py` — standard LoRA wrapping (PEFT).
- `ml/adapters/create_tokenformer_model.py` — wraps a base model in tokenformer layers.
- `ml/adapters/add_adapters_to_model.py` — uniform entry point used by the training loop.
- `ml/tokenformer/tokenformer_model.py` — core implementation.
- `ml/tokenformer/tokenformer_surgeon.py` — module-level surgery: swaps `nn.Linear` in attention with tokenformer equivalents in-place on the loaded HF model.

The matching control-plane side (`cray_infra/adapters/`) handles *loading* these adapters into the live vLLM process so post-training checkpoints become available without a restart.

### 4.8 MPI collectives — `collectives/`

- `collectives/main_rank_only.py:26` — `@main_rank_only` decorator.
- `collectives/data_parallelism.py` — higher-level allreduce wrappers.

The external `gpu_aware_mpi` package is a ScalarLM dependency that sits between PyTorch and the MPI runtime; it arranges for NCCL (NVIDIA) or RCCL (AMD) to be used as the transport with rank mapping aware of CUDA/HIP device ordering.

---

## 5. The Python SDK (`sdk/masint/`)

The SDK is published as `pip install scalarlm`. `import scalarlm` is a thin alias that re-exports `masint`.

### 5.1 `SupermassiveIntelligence` — synchronous front door

`sdk/masint/api/supermassive_intelligence.py:6` is what most users touch:

```python
class SupermassiveIntelligence:
    def __init__(self, api_url=None):
        self.async_api = AsyncSupermassiveIntelligence(api_url=api_url)

    def train(self, data, model_name=None, train_args={}):
        return asyncio.run(self.async_api.train(...))

    def generate(self, prompts, model_name=None, max_tokens=None):
        return asyncio.run(self.async_api.generate(...))

    # also: submit_slurm_job, list_models, get_training_job, health, metrics,
    #       get_gpu_count, get_node_count, cancel, delete, clear_queue
    # and stubs for: learn_classes, learn_docs, learn_database, learn_code
```

Every method is a `asyncio.run(...)` around the async equivalent. That keeps the common `scalarlm.SupermassiveIntelligence().generate([...])` notebook flow blocking and simple, while still allowing power users to import `AsyncSupermassiveIntelligence` directly.

### 5.2 `AsyncCray` — HTTP engine — `engines/async_cray.py:17`

All HTTP traffic funnels through this class:

- Uses a shared `aiohttp.ClientSession` (via `util/get_session.py`).
- URL construction via `util/make_api_url.py` + `get_api_base.py` (honors `scalarlm.api_url` global and `SCALARLM_API_URL` env var).
- Large-batch heuristic in `generate()`: if `len(prompts) > 128`, it switches to the upload path (`engines/cray/upload_generate.py`) — writes prompts to a file, `POST /v1/generate/upload`, then polls. Smaller batches go through `/v1/generate` synchronously with timeout `response_timeout`.
- Training submission: `engines/cray/submit_training_job.py` multipart-encodes the dataset + JSON args and calls `/v1/megatron/train`.
- Raw SLURM submission: `engines/cray/submit_slurm_job.py` for when you want to run a custom Python entrypoint under SLURM without going through Megatron.

### 5.3 SDK CLI — `sdk/masint/cli/`

After `pip install scalarlm` + `export SCALARLM_API_URL=...`, the `scalarlm` console command from the SDK exposes:

| Command | Source | Purpose |
|---|---|---|
| `scalarlm logs MODEL` | `cli/logs.py` | Stream training logs. |
| `scalarlm plot MODEL` | `cli/plot.py` | Plot loss/metrics for a run. |
| `scalarlm ls` | `cli/ls.py` | List trained models. |
| `scalarlm squeue` | `cli/squeue.py` | Show SLURM queue. |
| `scalarlm cancel MODEL` | `cli/cancel.py` | Cancel training. |
| `scalarlm delete MODEL` | `cli/delete.py` | Remove a model's job dir. |
| `scalarlm stats` | `cli/stats.py` | Cluster utilization. |

Note the naming: this is the **PyPI CLI** (`cli/main.py` on the Python entrypoint). The repo-local `./scalarlm` at the project root is a *different* tool — the bashly-generated developer CLI for running the server itself (§6).

---

## 6. The Developer CLI (`cmd/`, `scalarlm` script)

The top-level `scalarlm` bash script exists to make `./scalarlm up` / `./scalarlm test` / `./scalarlm build-image` one-liners for contributors.

- `scalarlm` (the shell file, not the SDK) — 20 lines. Runs `cmd/bashly.sh generate` to regenerate `scripts/scalarlm`, then delegates to it.
- `cmd/bashly.yml` — CLI spec. Uses [bashly](https://bashly.dannyb.co/) to generate a ~40 KB bash dispatcher.
- `cmd/*_command.sh` — one file per command, kept small (Docker build, `docker-compose up`, `pytest` wrappers, etc.).

Commands surfaced:

| Command | Backing script | Action |
|---|---|---|
| `./scalarlm build-image {cpu\|nvidia\|arm\|amd} [sm_arch]` | `build_image_command.sh` | `docker build` with target-specific args. |
| `./scalarlm up {cpu\|nvidia\|amd}` | `up_command.sh` | `docker-compose up cray-{target}`. |
| `./scalarlm test PATH` | `test_command.sh` | Run pytest inside the container. |
| `./scalarlm llm plot/logs/ls/squeue MODEL` | `llm_*_command.sh` | Thin wrappers around the SDK CLI. |
| `./scalarlm benchmark {cpu\|nvidia\|amd}` | `benchmark_command.sh` | Run `test/benchmark/` suite. |
| `./scalarlm pypi` | `pypi_command.sh` | Build and upload the SDK. |

---

## 7. Docker and Runtime Images

### 7.1 Multi-target Dockerfile

One `Dockerfile` handles all hardware targets. Key build args:

| Arg | Values | Effect |
|---|---|---|
| `BASE_NAME` | `cpu` / `nvidia` / `amd` / `arm` | Picks the base image (NVIDIA PyTorch, ROCm base, Ubuntu). |
| `TORCH_CUDA_ARCH_LIST` | `7.5`, `8.6`, `12.0`, `auto`, … | SM targets for CUDA kernels. |
| `VLLM_TARGET_DEVICE` | `cuda` / `rocm` / `cpu` | vLLM build flavor. |
| `VLLM_SOURCE` | `local` / `remote` | Build from the vendored fork or clone fresh. |
| `VLLM_BRANCH` | `scalarlm-on-v0.19.0` | The fork branch with ScalarLM changes. |
| `VLLM_REPO` | `https://github.com/supermassive-intelligence/vllm-fork.git` | Upstream remote. |

Stages install Python deps, compile vLLM from source against the selected GPU backend, and bake in the chat-UI and frontend assets.

### 7.2 `docker-compose.yaml`

Three services using YAML anchors:

- `cray` — CPU baseline.
- `cray-nvidia` — adds `runtime: nvidia` and the NVIDIA device reservation.
- `cray-amd` — mounts `/dev/kfd` and `/dev/dri`, drops seccomp restrictions for HIP.

Common ports: **3000** (chat UI), **8000** (API), **8001** (vLLM). Bind mounts keep `infra/cray_infra`, `ml`, `scripts`, `test`, and `models` (HF cache) live-editable.

### 7.3 Container startup — `scripts/start_one_server.sh`

```bash
#!/bin/bash
set -Eeuoxa pipefail
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
$LOCAL_DIRECTORY/start_slurm.sh
python -m cray_infra.one_server.main
```

Just two steps:

1. `start_slurm.sh` — boots `munged`, `slurmctld`, and `slurmd` *inside the container*. This is why ScalarLM can schedule training jobs on the same pod's GPUs without needing a separate cluster scheduler.
2. `python -m cray_infra.one_server.main` — runs the async supervisor from §3.1.

Per-job entrypoint (`train_job_entrypoint.sh`) is called by `sbatch`, sets GPU/MPI environment, and runs `python -m cray_megatron.main`.

---

## 8. Kubernetes Deployment (`deployment/helm/`)

`deployment/helm/gemma_270m_tw/scalarlm/` is a reference Helm chart tuned for TensorWave's Gemma 270M deployment. Fork it per-model.

### 8.1 Templates

| Template | Resource | Notes |
|---|---|---|
| `api_deployment.yaml` | Deployment | Runs API container, port 8000. |
| `vllm_deployment.yaml` | Deployment | Runs vLLM, port 8001. Gets `inference_gpus` GPUs. |
| `megatron_deployment.yaml` | Deployment | Runs SLURM controller + Megatron worker. Gets `training_gpus` GPUs. |
| `api_service.yaml` / `vllm_service.yaml` / `megratron_service.yaml` | Service | ClusterIP for intra-pod traffic. |
| `*_configmap.yaml` | ConfigMap | Materializes `cray-config.yaml` per component. |
| `cache_pvc.yaml` | PVC (400 Gi) | HF model cache (`/root/.cache/huggingface`). |
| `jobs_pvc.yaml` | PVC (100 Gi) | Training job dirs (`/app/cray/jobs`), shared so API can read status. |
| `slurm_config_pvc.yaml` | PVC | SLURM state. |
| `cloudflare_deployment.yaml` | Deployment | Cloudflare tunnel for external HTTPS access. |

### 8.2 `values.yaml` knobs

```yaml
image:       { repository: gdiamos/scalarlm-amd, tag: v0.99 }
service:     { api_port: 8000, vllm_port: 8001 }
cache_pvc:   { size: 400Gi }
jobs_pvc:    { size: 100Gi }
model: google/gemma-3-270m
max_model_length: 32768
gpu_memory_utilization: 0.95
dtype: bfloat16
training_gpus: 2
inference_gpus: 1
max_train_time: 86400
```

The three Deployments split "inference GPUs" from "training GPUs" so long training jobs can't preempt live traffic. In single-container dev mode (`docker-compose`) they share one GPU pool because SLURM handles serialization.

---

## 9. The vLLM Fork (`vllm/`)

Vendored from `https://github.com/supermassive-intelligence/vllm-fork.git` on branch `scalarlm-on-v0.19.0`. It provides:

- Runtime LoRA adapter loading (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`, set in `one_server/main.py:4`) — lets new post-trained adapters be hot-added without a restart.
- Tokenformer adapter compatibility — vLLM serves the same adapter format the training loop produces.
- ROCm/AMD MI300 support matching the AMD Dockerfile target.

The fork is built from source during `docker build` so that the C++/CUDA/HIP kernels match the container's compiler toolchain and `TORCH_CUDA_ARCH_LIST`.

`vllm/AGENTS.md` documents the contribution policy for this tree if you're making fork-side changes.

---

## 10. Tests and Benchmarks (`test/`)

### 10.1 Integration tests — `test/infra/`

These run inside the container against a live server:

| Test | Covers |
|---|---|
| `sanity.py`, `health.py` | Server boot + `/v1/health`. |
| `vllm_health.py` | In-pod vLLM reachability. |
| `generate.py`, `get_results.py` | End-to-end queue inference. |
| `upload_dataset.py` | Dataset multipart upload. |
| `get_latest_model.py` | Post-training model discovery. |
| `slurm.py` | sbatch round-trip. |
| `distribution_strategy/benchmark_mpi_collectives.py` | FSDP/DDP allreduce latency. |
| `distribution_strategy/benchmark_mpi_sendrecv.py` | Point-to-point MPI. |
| `infra/vllm/tokenformer/test_tokenformer.py` | Tokenformer adapter integration. |

Run with `./scalarlm test test/infra/*`.

### 10.2 Benchmark harness — `test/benchmark/`

- `benchmark/main.py` — dispatcher.
- `benchmark/pytorch/{forward,backward,gemm,memcpy,mpi_p2p}.py` — microbenchmarks.
- `benchmark/roofline/{plot_roofline,plot_bandwidth_sweep}.py` — roofline analysis to sanity-check how close the stack runs to hardware peak.

### 10.3 Collective tests

Top-level `test/collectives/` (currently untracked — `test_shm_channel.py`, `test_shm_channel_cpp.cpp`) exercises a shared-memory collective channel used by `gpu_aware_mpi`.

---

## 11. Frontend and Chat UI

- `ui/` — first-party React SPA (Vite + React 18 + TanStack Query). Built at image-build time into `/app/ui-bundle/` and served by FastAPI at `/app/*`. Source of truth for chat, training, metrics, inference-request browsing, and settings. See `docs/ui-design.md`.
- `infra/cray_infra/api/fastapi/setup_ui.py` — `add_ui(app)` mounts the SPA: hashed assets under `/app/assets` (immutable 1y cache), dynamic `/app/api-config.json`, and a catch-all SPA fallback that returns `index.html` for unknown `/app/*` paths.
- `frontend/` — static assets (logo) referenced by the React app.

---

## 12. Request Flows

### 12.1 Synchronous chat (OpenAI-compatible)

```
curl http://host:8000/v1/chat/completions
       │
       ▼
FastAPI  openai_v1_router  ──HTTP──►  vLLM :8001
       ▲                                 │
       └──────── JSON response ◄─────────┘
```

No queue involvement. Used by `openai` / `anthropic` / `langchain` clients that speak the OpenAI format.

### 12.2 Batched async inference (SDK `.generate()` on small batches)

```
SDK ──POST /v1/generate──► API ──► push_into_queue  (SQLite)
                                        │
                        Generate Worker ◄┘
                                │
                                ├──► vLLM :8001
                                │
                                └──► update_and_ack (stores outputs)
SDK ──POST /v1/generate/get_results──► API ──► read queue ──► 200 OK
```

SDK polls `get_results` until ready. Timeout governed by `response_timeout` (60 s default).

### 12.3 Large-batch async inference (SDK with ≥128 prompts)

Same as §12.2 except the SDK uploads a prompt file via `/v1/generate/upload` first and then issues generate with a file reference, avoiding a giant JSON body.

### 12.4 Training run
Schedules a training job via the control plane. For datasets exceeding ~100MB, the SDK uses the [chunked upload protocol](/docs/chunked-upload.md) to bypass proxy limits.

```
SDK.train(data, train_args)
 └─ (Legacy) POST /v1/megatron/train  (multipart)
 └─ (Chunked) POST /v1/megatron/upload/init → /chunk → /finalize
```

Background: add_megatron_tasks (lifespan)
 └─ register_megatron_workers scans jobs dir
    └─ register_megatron_models → vLLM runtime LoRA load
       → model available to next /v1/generate request
```

### 12.5 Post-training inference (the closed loop)

After the registration step above, a subsequent `generate()` call that specifies the new model name, or the `latest` alias, routes through the already-loaded adapter in vLLM. No pod restart. No checkpoint copy. Inference and training never stop serving each other.

---

## 13. Design Patterns and Conventions

- **Async-first, single event loop.** The control plane runs entirely on asyncio. Uvicorn + `asyncio.wait(FIRST_COMPLETED)` is the supervisor.
- **File-based coordination between training and control plane.** `status.json` and checkpoint files in `/app/cray/jobs/{hash}/` — no HTTP callbacks, no message broker. Survives restarts on either side.
- **Persistent SQLite queue for inference.** `persistqueue.SQLiteAckQueue` gives crash-safe delivery with explicit ack and a 5-minute lease. One less service to run than Redis or RabbitMQ.
- **Three-tier config.** Pydantic defaults → YAML → `SCALARLM_*` env vars. Same code reads config in Docker and Kubernetes.
- **Rank-0 isolation.** `@main_rank_only` decorator in `collectives/main_rank_only.py:26` is the single source of truth for "only one rank does this" — used for logging, checkpoint saves, status updates.
- **One container, multiple cooperating processes.** SLURM is embedded, vLLM runs in-process, the API is in-process — all started by one `asyncio.gather`. Kubernetes sees a single pod.
- **Bind mounts for hot dev.** The Uvicorn supervisor's `reload_dirs=["/app/cray/infra/cray_infra"]` plus docker-compose bind mounts mean control-plane edits take effect without a container rebuild. Training code lives in `ml/` which is also bind-mounted; next job submission picks up the changes.
- **GPU-agnostic by construction.** No vendor-specific imports above the level of `gpu_aware_mpi`; Dockerfile `BASE_NAME` + `VLLM_TARGET_DEVICE` select the backend at build time.

---

## 14. Key File Reference

A short directory for navigating the code:

| Concern | File | Symbol |
|---|---|---|
| Container entrypoint | `scripts/start_one_server.sh` | — |
| Async supervisor | `infra/cray_infra/one_server/main.py:98` | `run_all_servers_async` |
| Service dispatch | `infra/cray_infra/one_server/start_cray_server.py:12` | `start_cray_server` |
| FastAPI app | `infra/cray_infra/api/fastapi/main.py:32` | `app` |
| OpenAI routes | `infra/cray_infra/api/fastapi/routers/openai_v1_router.py` | `openai_v1_router` |
| Inference queue routes | `infra/cray_infra/api/fastapi/routers/generate_router.py:32` | `generate_router` |
| Training routes | `infra/cray_infra/api/fastapi/routers/megatron_router.py:33` | `megatron_router` |
| Work queue | `infra/cray_infra/api/work_queue/inference_work_queue.py:13` | `InferenceWorkQueue` |
| Submit training | `infra/cray_infra/training/launch_training_job.py:18` | `launch_training_job` |
| Register models | `infra/cray_infra/training/register_megatron_workers.py` | `register_megatron_workers` |
| Training status enum | `infra/cray_infra/training/training_job_status.py:4` | `TrainingJobStatus` |
| Config defaults | `infra/cray_infra/util/default_config.py:5` | `Config` |
| Training entrypoint | `ml/cray_megatron/main.py:29` | `main` |
| Harness | `ml/cray_megatron/megatron/training_harness.py:16` | `TrainingHarness` |
| Trainer | `ml/cray_megatron/megatron/megatron_trainer.py:14` | `MegatronTrainer` |
| Training loop | `ml/cray_megatron/megatron/training_loop.py:28` | `TrainingLoop` |
| FSDP | `ml/cray_megatron/megatron/distribution/fsdp.py` | — |
| Rank-0 decorator | `ml/cray_megatron/collectives/main_rank_only.py:26` | `main_rank_only` |
| Tokenformer surgery | `ml/tokenformer/tokenformer_surgeon.py` | — |
| SDK sync API | `sdk/masint/api/supermassive_intelligence.py:6` | `SupermassiveIntelligence` |
| SDK async engine | `sdk/masint/engines/async_cray.py:17` | `AsyncCray` |
| Bashly CLI spec | `cmd/bashly.yml` | — |
| Helm chart | `deployment/helm/gemma_270m_tw/scalarlm/` | — |

---

## 15. Extending ScalarLM

The most common extension points, in the order you'd reach for them:

1. **Change the training loop.** Edit `ml/cray_megatron/megatron/training_loop.py`. Bind-mounted in dev; redeploy the `ml/` directory in prod (no container rebuild).
2. **Add a new optimizer or scheduler.** `get_optimizer` / `get_scheduler` in the same file.
3. **New dataset format.** Add a loader under `ml/cray_megatron/megatron/dataset/` and register it in `load_dataset.py`.
4. **New adapter type.** Add to `ml/adapters/` (training side) and `infra/cray_infra/adapters/` (serving side). Mirror the pattern between LoRA and Tokenformer.
5. **New HTTP endpoint.** Add a module under `infra/cray_infra/api/fastapi/routers/`, include it in `api/fastapi/main.py`.
6. **Different hardware target.** Add a `BASE_NAME` branch to the `Dockerfile` and a matching service to `docker-compose.yaml`.
7. **New SDK method.** Add to `sdk/masint/engines/async_cray.py` and expose it on `SupermassiveIntelligence`. Keep the sync wrapper as `asyncio.run(self.async_api.X(...))`.

Every one of these extension points is a single file or small group of files — the architecture is deliberately narrow at the seams to make forking straightforward, which is consistent with the CC-0 license stance.
