# Configuration Reference

ScalarLM has two independent configuration systems:

- **Server config** — the shape and behavior of the running pod. Owned by `infra/cray_infra/util/default_config.py`, loaded by `get_config()`, consumed by the API, vLLM, Generate Worker, and periodic tasks.
- **Job config** — the knobs for a single training run. Owned by `infra/cray_infra/util/default_job_config.py`, loaded by `get_job_config()`, consumed by the training loop.

They're distinct on purpose: server config describes a deployment and is uniform across all requests; job config describes one training submission and lives in the job directory for the life of that run. This document covers both, plus the HF token encryption path, the SDK's client-side URL resolution, and every override tier.

---

## 1. Server Configuration

### 1.1 Three-tier override system

`infra/cray_infra/util/get_config.py:11`:

```python
def get_config():
    loaded_config = {}
    config_path = "/app/cray/cray-config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            loaded_config = yaml.safe_load(stream)

    config = Config(**loaded_config).dict()

    for key, value in config.items():
        corresponding_env_var = f"SCALARLM_{key.upper()}"
        if corresponding_env_var in os.environ:
            env_value = os.environ[corresponding_env_var]
            if isinstance(value, bool):    config[key] = env_value.lower() in ("true","1","yes")
            elif isinstance(value, int):   config[key] = int(env_value)
            elif isinstance(value, float): config[key] = float(env_value)
            else:                           config[key] = env_value
    return config
```

Resolution order, lowest to highest precedence:

```
┌─────────────────────────────────────┐
│  1. Pydantic defaults               │   default_config.py
│     (Config(BaseModel) field values)│
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  2. YAML file                       │   /app/cray/cray-config.yaml
│     (only fields set in YAML        │   (Helm ConfigMap or bind-mount)
│      override defaults)             │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  3. SCALARLM_* env vars             │   `SCALARLM_{KEY_UPPER}`
│     (final override, typed by       │   set per-container in k8s
│      defaults at that field)        │
└─────────────────────────────────────┘
```

Three implementation notes worth knowing:

- **Missing YAML is fine.** If `/app/cray/cray-config.yaml` doesn't exist, the loader starts from `Config()` defaults and applies only env vars. This is how the plain `docker compose up cray` dev workflow works without writing any config.
- **Typing is preserved.** Env vars go through the type of the existing field. `SCALARLM_MAX_MODEL_LENGTH=32768` becomes an `int`; `SCALARLM_GPU_MEMORY_UTILIZATION=0.9` becomes a `float`; `SCALARLM_AUTO_RESUME=true` becomes a `bool`. No custom parsers — `int(env)`, `float(env)`, `env.lower() in {"true","1","yes"}`.
- **Every `get_config()` call re-reads.** The function has no cache. Every consumer reads YAML + env fresh. This means a ConfigMap update propagates to the next consumer on the next call without pod restart — but it also means heavy call sites should bind the dict once.

### 1.2 Where YAML comes from

| Runtime | Source |
|---|---|
| Docker Compose dev | No YAML by default; bind-mount your own if needed at `./cray-config.yaml` → `/app/cray/cray-config.yaml`. |
| Prebuilt container | Same — mount or bake in. |
| Helm / Kubernetes | Three `ConfigMap`s (`api_configmap.yaml`, `vllm_configmap.yaml`, `megatron_configmap.yaml`), each mounted at `/app/cray/cray-config.yaml` with `subPath`. |

Each Kubernetes Deployment gets its **own** ConfigMap — they are not identical. The differences matter:

- `api_configmap`: sets `server_list: api`, points `vllm_api_url` at the vLLM service, also sets `max_train_time`.
- `vllm_configmap`: sets `server_list: vllm`, points `api_url` back at the API service, sets `max_model_length` and `gpu_memory_utilization`.
- `megatron_configmap`: sets `server_list: megatron`, sets `max_train_time`.

This is how a single container image selects its role at runtime — `server_list` controls which subsystems `start_cray_server` spawns (see `one_server/start_cray_server.py:20-50`). A pod with `server_list: api` runs the FastAPI + Generate Worker; a pod with `server_list: vllm` runs only vLLM; a pod with `server_list: megatron` runs only the training orchestrator. In single-container dev mode, `server_list: all` runs everything.

### 1.3 Complete field reference — `Config`

From `infra/cray_infra/util/default_config.py:5`. Every field is a Pydantic model attribute with a default; every one is overridable via YAML or `SCALARLM_{KEY}`.

#### Networking & URLs

| Field | Default | Purpose | Read by |
|---|---|---|---|
| `api_url` | `http://localhost:8000` | Where the Generate Worker and periodic tasks call back into the API. | `create_generate_worker.py:68`, `restart_megatron_jobs.py:122` |
| `vllm_api_url` | `http://localhost:8001` | Not used on the hot path (worker uses in-process vLLM), but some health/introspection code resolves here. | `wait_for_vllm.py:24` |
| `server_list` | `all` | Comma-separated list of subsystems to start. Values: `api`, `vllm`, `megatron`, `all`. | `one_server/main.py:27`, `start_cray_server.py:20` |

#### Model selection

| Field | Default | Purpose |
|---|---|---|
| `model` | `tiny-random/gemma-4-dense` | HF Hub ID of the base model loaded at startup. vLLM serves this; training jobs default to it for `llm_name`. |
| `max_model_length` | 0 (auto) | Fallback admission cap when vLLM runtime introspection is unavailable; `0` means no configured fallback cap. vLLM selects its model limit automatically, and the Generate Worker uses that runtime value for KV-cache batching. |
| `dtype` | `auto` | vLLM dtype. `auto` → bf16 on GPUs with SM ≥ 8, float32 on CPU. Forced to `float32` on sm<8 (flashmla backend). |
| `gpu_memory_utilization` | 0.40 | Fraction of GPU memory vLLM reserves for KV cache + activations. Raise for dedicated inference pods; lower when training and inference share GPUs. |
| `tensor_parallel_size` | 1 | vLLM `--tensor-parallel-size`. Number of GPUs to shard the model across. |
| `limit_mm_per_prompt` | `{"image":2}` | vLLM `--limit-mm-per-prompt` — caps multimodal attachments per prompt. |
| `max_log_length` | 100 | Truncation for vLLM's `--max-log-len`. |

#### Training-job plumbing

| Field | Default | Purpose |
|---|---|---|
| `train_job_entrypoint` | `/app/cray/scripts/train_job_entrypoint.sh` | sbatch-invoked script (copied per-job with `REPLACE_CONFIG_PATH` templated in). |
| `training_job_directory` | `/app/cray/jobs` | Parent directory for per-job dirs. Must be on a shared PVC in Kubernetes. |
| `log_directory` | `/app/cray/nfs/logs` | Where `one_server/main.py` writes `{api,vllm,megatron}.log`. |
| `max_gpus_per_node` | 1 | Used by SLURM cluster discovery; cap on GPUs per node when building sbatch commands. |
| `max_train_time` | 86400 (24 h) | Upper bound on `--time` for any sbatch submission. Per-job `timeout` is clamped to this. |
| `extra_training_seconds` | 300 | Extra wall-clock buffer added on top of `--time` so training can observe SIGCONT/SIGTERM and gracefully checkpoint. |
| `megatron_refresh_period` | 30 (s) | Period of the FastAPI lifespan task loop that registers models, restarts dead jobs, clears acked queue rows. |

#### SLURM plumbing

| Field | Default | Purpose |
|---|---|---|
| `slurm_wait_time` | 30 (s) | Max time `wait_for_slurm()` polls `squeue` during boot. |
| `node_info_time_limit` | 3600 (s) | Cluster-discovery cache TTL. |

#### Inference work queue (§ see `inference-queue.md`)

| Field | Default | Purpose |
|---|---|---|
| `generate_batch_size` | 1024 | Hard ceiling on the Generate Worker's per-pull batch size. |
| `response_timeout` | 60 (s) | Max blocking wait for `/v1/generate` + `/v1/generate/get_results` + `/v1/generate/download`. |
| `inference_work_queue_timeout` | 30 (s) | Blocking dequeue timeout on SQLite. Worker HTTP client doubles this. |
| `inference_work_queue_idle_time` | 5 (s) | Minimum idle-worker window before stuck-ack recycling kicks in. |
| `inference_work_queue_ack_timeout` | 300 (s) | Max `unack` age before eligible for recycling. |
| `inference_work_queue_path` | `/app/cray/inference_work_queue.sqlite` | SQLite queue file. Put on a PVC for durability. |
| `upload_base_path` | `/app/cray/inference_requests` | Staging for `{hash}.json`, `_status.json`, `_response.json`. Also PVC. |
| `max_upload_file_size` | 10 GB | Per-file ceiling on `/v1/megatron/train` and `/v1/generate/upload`. Body cap is 2×. |

Defaults:
- `default_max_output_tokens` | 128 | Fallback `max_tokens` when a client omits it.

#### Tokenformer adapter cache

| Field | Default | Purpose |
|---|---|---|
| `tokenformer_r` | 32 | Rank of the Tokenformer key-value memory banks. |
| `tokenformer_num_heads` | 4 | Number of heads in the tokenformer layers. |
| `tokenformer_cache_capacity` | 2 | Max number of tokenformer adapters kept hot in vLLM at once. LRU eviction beyond. |

#### Hugging Face authentication

| Field | Default | Purpose |
|---|---|---|
| `hf_token` | `""` | Plaintext HF token. If set, overrides everything below. |
| `hf_encrypted_token` | Fernet-encrypted placeholder | Default encrypted token, kept out of plaintext in the repo. |
| `encryption_key` | 32-byte Fernet key | Decrypts `hf_encrypted_token`. |

See §3 for the resolution algorithm and how to rotate these.

### 1.4 Every configurable setting, by env var

Every field above can be overridden by setting `SCALARLM_{FIELD_UPPER}`. Examples:

```bash
export SCALARLM_MODEL=google/gemma-3-27b-it
export SCALARLM_MAX_MODEL_LENGTH=32768
export SCALARLM_GPU_MEMORY_UTILIZATION=0.9
export SCALARLM_DTYPE=bfloat16
export SCALARLM_SERVER_LIST=api
export SCALARLM_API_URL=http://scalarlm-api.gemma.svc.cluster.local:8000
export SCALARLM_VLLM_API_URL=http://scalarlm-vllm.gemma.svc.cluster.local:8001
export SCALARLM_TRAINING_JOB_DIRECTORY=/mnt/pvc/jobs
export SCALARLM_INFERENCE_WORK_QUEUE_PATH=/mnt/pvc/queue.sqlite
export SCALARLM_UPLOAD_BASE_PATH=/mnt/pvc/inference_requests
export SCALARLM_MAX_TRAIN_TIME=172800       # 48 hours
export SCALARLM_HF_TOKEN=hf_xxxxxxxx
```

### 1.5 Special env vars NOT covered by `SCALARLM_*`

A few environment variables are read directly without going through `Config`:

| Env var | Read by | Purpose |
|---|---|---|
| `SCALARLM_VLLM_ARGS` | `create_vllm.py:77` | Free-form extra arguments appended to the vLLM CLI invocation. Duplicates of existing flags are replaced. |
| `HF_TOKEN` | `get_hf_token.py:10` | Short-circuits the entire HF token resolution chain (highest priority of all). |
| `HUGGING_FACE_HUB_TOKEN` | Exported by ScalarLM after resolution. | Consumed by HF `from_pretrained` and vLLM. Set programmatically, not by the user. |
| `CUDA_VISIBLE_DEVICES` | Standard NVIDIA plumbing | Scopes which GPUs the pod sees. |
| `VLLM_TARGET_DEVICE` | vLLM build + runtime | `cuda` / `rocm` / `cpu`. Set by the Dockerfile, rarely at runtime. |
| `VLLM_LOGGING_LEVEL` | vLLM | Set to `DEBUG` by `one_server/main.py:3`. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | vLLM | Set to `true` by `one_server/main.py:4`. Enables runtime adapter hot-loading — required for ScalarLM's closed-loop flow. |
| `VLLM_ATTENTION_BACKEND` | vLLM | Forced to `FLASHMLA` on SM < 8 GPUs (`create_vllm.py:41`). |
| `VLLM_USE_STANDALONE_COMPILE` | vLLM | Set to `0` on SM < 8. |
| `CRAY_TRAINING_JOB_CONFIG_PATH` | `get_job_config.py:21` | Per-job config path, set by the sbatch entrypoint (see §2). |
| `SCALARLM_CONFIG_PATH` | `get_config.py` | Overrides the default `/app/cray/cray-config.yaml` server-config YAML path. Primary use is the unit test suite; operators can also point at a baked-in config. |

`SCALARLM_VLLM_ARGS` is the escape hatch for anything vLLM-specific that isn't surfaced as a `Config` field:

```bash
export SCALARLM_VLLM_ARGS="--enforce-eager --disable-log-requests --swap-space=8"
```

These are appended to the vLLM CLI args after the ones built from `Config`, and any `--flag=...` already in the list is removed first so the extras win.

### 1.6 Example: complete YAML

A production-shaped `/app/cray/cray-config.yaml` for the API pod in Helm:

```yaml
model: google/gemma-3-27b-it
dtype: bfloat16
max_model_length: 32768
gpu_memory_utilization: 0.9
vllm_api_url: "http://scalarlm-vllm.gemma.svc.cluster.local:8001"
server_list: api
max_train_time: 172800

# Storage on shared PVC
training_job_directory: /mnt/pvc/jobs
inference_work_queue_path: /mnt/pvc/queue.sqlite
upload_base_path: /mnt/pvc/inference_requests

# Larger batch queue for busy clusters
generate_batch_size: 2048
response_timeout: 120

# Auth
hf_token: hf_xxxxxxxxxxxxxxx
```

Equivalent via env vars is mechanical: uppercase, prefix `SCALARLM_`. Helm's ConfigMap approach is almost always the right choice in Kubernetes — it's declarative, versioned with the release, and avoids leaking secrets into pod specs.

---

## 2. Job Configuration

### 2.1 Source

`infra/cray_infra/util/default_job_config.py` defines `JobConfig`. Unlike server config, this has **three required fields with no defaults**:

- `job_directory` — filled in by the API server during ingestion (`upload_training_data.py:65`).
- `training_data_path` — filled in by the API server (`upload_training_data.py:73`).
- `dataset_hash` — SHA-256 of the uploaded tar, set by the API server (`upload_training_data.py:61`).

The rest have defaults. When the user's `train_args` dict is merged with these required fields, the result is the authoritative `config.yaml` for the job, written to `{job_directory}/config.yaml` and re-read by every rank of the training process.

### 2.2 Loading at runtime

`get_job_config()` (`infra/cray_infra/util/get_job_config.py`):

```python
def get_job_config():
    path = os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"]  # hard-asserted
    with open(path, "r") as stream:
        job_config = yaml.safe_load(stream)
    return JobConfig(**job_config).dict()
```

The env var is set by the sbatch entrypoint before `python -m cray_megatron.main`:

```bash
# scripts/train_job_entrypoint.sh (after REPLACE_CONFIG_PATH templating)
export CRAY_TRAINING_JOB_CONFIG_PATH=/app/cray/jobs/{job_hash}/config.yaml
export PYTHONPATH=$LOCAL_DIRECTORY/ml:$PYTHONPATH
mpirun --allow-run-as-root python $LOCAL_DIRECTORY/ml/cray_megatron/main.py $*
```

Unlike server config, this function **does not** consult env vars — there's no `SCALARLM_*` override for job config. One-off job tweaks happen via `train_args` at submission time. The YAML on disk is the single source of truth for the run.

### 2.3 Complete field reference — `JobConfig`

#### Required (set by the API server, not the client)

| Field | Type | Set by |
|---|---|---|
| `job_directory` | str | `upload_training_data.py:65` — `training_job_directory/{sha256(train_args)}` |
| `training_data_path` | str | `upload_training_data.py:73` — `{job_directory}/dataset.jsonlines` |
| `dataset_hash` | str | `upload_training_data.py:61` — SHA-256 of the uploaded tar file |

#### Model selection

| Field | Default | Purpose |
|---|---|---|
| `llm_name` | `meta-llama/Llama-3.2-1B-Instruct` | HF Hub ID to start from. Overrides server `config["model"]` for this job. |

#### Optimization

| Field | Default | Purpose |
|---|---|---|
| `max_steps` | 100 | Number of optimizer steps. |
| `learning_rate` | 3e-3 | AdamW learning rate. |
| `batch_size` | 1 | Per-rank micro-batch. |
| `gradient_clip_value` | 1.0 | `torch.nn.utils.clip_grad_norm_` max norm. |
| `gradient_accumulation_steps` | 4 | Micro-batches per optimizer step. Effective batch = `batch_size × gradient_accumulation_steps × world_size`. |
| `max_token_block_size` | 16 × 10⁶ | Dataset chunking limit (16 M tokens). |

#### Training mode

| Field | Default | Options |
|---|---|---|
| `training_mode` | `language_model` | `language_model` (causal LM) or `embedding` (contrastive). Selects the dataset loader. |

#### Distributed training

| Field | Default | Purpose |
|---|---|---|
| `distribution_strategy` | `fsdp` | `fsdp` / `ddp` / `none`. Selected by `apply_distribution_strategy.py`. |
| `gpus` | 1 | `sbatch --ntasks-per-node` and `--gres=gpu:N`. Clamped by `scontrol show nodes`. |
| `nodes` | 1 | `sbatch --nodes`. Clamped by total node count. |

#### Checkpointing

| Field | Default | Purpose |
|---|---|---|
| `steps_per_checkpoint` | 100 | `CheckpointCallback` period (skipped at step 0). |
| `max_checkpoints_to_keep` | 3 | Retention for `delete_old_checkpoints`. |

#### Adapters

| Field | Default | Purpose |
|---|---|---|
| `adapter_type` | `tokenformer` | `tokenformer` or `lora`. |
| `lora_config.r` | 32 | LoRA rank. |
| `lora_config.lora_alpha` | 32 | LoRA alpha. |
| `lora_config.lora_dropout` | 0.1 | LoRA dropout. |
| `lora_config.target_modules` | `"all-linear"` | Layer targets. String (`"all-linear"`) or explicit list. |

#### Runtime limits

| Field | Default | Purpose |
|---|---|---|
| `timeout` | 14400 (4 h) | Wall-clock cap enforced by `TimeoutCallback`. Also clamps sbatch `--time` (via `get_train_time_limit`). |
| `training_history_length` | 1024 | Cap on `history` list in `status.json`. Larger runs use `remove_closest_entry` to keep the list uniformly sampled. |

### 2.4 Example `train_args` at submission

```python
llm.train(
    data="./train.jsonl",
    train_args={
        "llm_name": "google/gemma-3-4b-it",
        "max_steps": 2000,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,          # effective batch 16
        "steps_per_checkpoint": 200,
        "max_checkpoints_to_keep": 5,
        "gpus": 4,
        "nodes": 1,
        "adapter_type": "lora",
        "lora_config": {"r": 64, "lora_alpha": 128, "target_modules": ["q_proj", "v_proj"]},
        "timeout": 43200,                          # 12 h (clamped to max_train_time on server)
    },
)
```

Fields not specified fall back to `JobConfig` defaults. The SDK does no local validation — Pydantic validates on the server side when `JobConfig(**job_config)` is called at every rank.

---

## 3. Hugging Face Token Resolution

There are four sources for the HF token, checked in priority order:

```
1. os.environ["HF_TOKEN"]                   ──► return verbatim
2. config["hf_token"]   (if != "")          ──► return verbatim
3. Fernet.decrypt(config["hf_encrypted_token"], config["encryption_key"])
```

Implementation (`infra/cray_infra/huggingface/get_hf_token.py:9`):

```python
def get_hf_token():
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]

    config = get_config()
    if config["hf_token"] != "":
        return config["hf_token"]

    encrypted_token = config["hf_encrypted_token"]
    key             = config["encryption_key"]
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_token).decode()
```

Callers:

- `one_server/create_vllm.py:33` — before building the vLLM app. `os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()`.
- `ml/cray_megatron/main.py:33` — before training loads the HF model. Same pattern.

Setting `os.environ["HUGGING_FACE_HUB_TOKEN"]` is what actually reaches `transformers.from_pretrained` and vLLM's own model loader — the ScalarLM `Config` field is just where the token *comes from*.

### 3.1 Rotating the encrypted token

A helper script exists at `infra/cray_infra/huggingface/encrypt.py`:

```python
from cryptography.fernet import Fernet
key = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="
cipher = Fernet(key)
plaintext = "<your new hf token here>"
encrypted = cipher.encrypt(plaintext.encode())
print(f"Encrypted: {encrypted}")
# Paste the output into Config.hf_encrypted_token
```

To rotate:

1. Generate the encrypted bytes with the snippet above using the current `encryption_key`.
2. Paste them into `default_config.py:66` (`hf_encrypted_token`).

To rotate the *key* itself, generate a new one with `Fernet.generate_key()`, re-encrypt the token with it, and update both fields together. Both fields are `bytes` — keep the `b"..."` prefix.

### 3.2 Priority, in practice

| Context | Typical setting |
|---|---|
| Production Kubernetes | `hf_token` via ConfigMap (or Secret → env var) |
| Local Docker dev | `export HF_TOKEN=...` outside the container, forwarded via `docker-compose` |
| CI / ephemeral | `HF_TOKEN` env var |
| Public/shared image | `hf_encrypted_token` — token embedded in the image without plaintext leakage |

The encrypted-token fallback is explicitly designed for the last case: ship a container that can pull gated models from HF without needing the operator to provision a secret. Not a high-security posture — the encryption key is in the same config file — but it keeps plaintext out of `git`, image layers, and process listings.

---

## 4. Client-Side Configuration (SDK)

The SDK's `SupermassiveIntelligence` needs exactly one configuration value: the API base URL. Resolution happens in `sdk/masint/util/get_api_base.py`:

```python
def get_api_base():
    if hasattr(scalarlm, "api_url") and scalarlm.api_url is not None:
        return scalarlm.api_url
    if hasattr(masint, "api_url") and masint.api_url is not None:
        return masint.api_url
    if "SCALARLM_API_URL" in os.environ:
        return os.environ["SCALARLM_API_URL"]
    if "MASINT_API_URL" in os.environ:
        return os.environ["MASINT_API_URL"]
    return "http://localhost:8000"
```

Resolution order (highest first):

1. `SupermassiveIntelligence(api_url=...)` constructor arg (passed through to `AsyncCray.api_url` → `make_api_url`).
2. `scalarlm.api_url = "..."` module attribute.
3. `masint.api_url = "..."` module attribute (legacy package name).
4. `SCALARLM_API_URL` env var.
5. `MASINT_API_URL` env var (legacy).
6. `http://localhost:8000` default.

Both package names coexist: `sdk/scalarlm/__init__.py` is just `from masint import *`. New code should use `scalarlm.`; `masint.` is kept for compatibility.

---

## 5. Logging

Logging configuration is hardcoded, but its outputs are under config control.

`infra/cray_infra/one_server/main.py:21-55` builds per-subsystem log files:

```
log_directory/                        # config["log_directory"]
├── api.log                           # FastAPI + control plane
├── vllm.log                          # vLLM engine
└── megatron.log                      # Training orchestrator
```

The subsystems included match `config["server_list"]`: with `server_list: all`, all three files are written; with `server_list: api`, only `api.log`. A `StreamHandler` is always added so logs also reach stdout (captured by `kubectl logs` or `docker logs`).

Root logger is `DEBUG`; individual modules override (e.g., `ml/cray_megatron/main.py:53` pins `filelock` to `WARNING` and FSDP to `INFO`). No external config — edit the code to tune levels.

---

## 6. Per-Component Config Overview

Which components care about which config fields (not exhaustive; spot-check when tuning):

| Component | Key fields |
|---|---|
| `create_api.py` | `api_url`, port 8000 |
| `create_vllm.py` | `model`, `dtype`, `max_model_length`, `gpu_memory_utilization`, `tensor_parallel_size`, `limit_mm_per_prompt`, `max_log_length`, `hf_*`, env `SCALARLM_VLLM_ARGS` |
| `create_generate_worker.py` | `api_url`, `max_model_length`, `generate_batch_size`, `inference_work_queue_timeout`, `training_job_directory` |
| `InferenceWorkQueue` | `inference_work_queue_path`, `inference_work_queue_timeout` |
| `clear_acked_requests_from_queue` | `inference_work_queue_ack_timeout`, `inference_work_queue_idle_time` |
| `generate()` handler | `model`, `response_timeout` |
| `upload()` handler | `max_upload_file_size`, `upload_base_path` |
| `launch_training_job.py` | `slurm_wait_time`, `train_job_entrypoint`, `max_gpus_per_node`, `max_train_time`, `extra_training_seconds`, `training_job_directory` |
| `add_megatron_tasks.py` | `megatron_refresh_period` |
| `discover_clusters.py` | `max_gpus_per_node`, `node_info_time_limit` |
| `get_hf_token.py` | `hf_token`, `hf_encrypted_token`, `encryption_key`, env `HF_TOKEN` |
| Training loop | (all from job config, not server config) |

---

## 7. Common Configuration Patterns

### 7.1 One GPU, dev laptop

No config needed. `./scalarlm up cpu` or `./scalarlm up nvidia` will use defaults.

### 7.2 Swap in a different base model

```yaml
# /app/cray/cray-config.yaml
model: Qwen/Qwen2-7B-Instruct
max_model_length: 32768
gpu_memory_utilization: 0.85
```

Or at `./scalarlm up` time:

```bash
SCALARLM_MODEL=Qwen/Qwen2-7B-Instruct \
SCALARLM_MAX_MODEL_LENGTH=32768 \
SCALARLM_GPU_MEMORY_UTILIZATION=0.85 \
./scalarlm up nvidia
```

### 7.3 Dedicated inference pod

```yaml
server_list: vllm
model: google/gemma-3-27b-it
max_model_length: 32768
gpu_memory_utilization: 0.95      # no training competition
api_url: "http://scalarlm-api:8000"
```

### 7.4 Multi-GPU tensor-parallel inference

```yaml
model: Qwen/Qwen3.5-122B-A10B
tensor_parallel_size: 4
max_model_length: 131072
gpu_memory_utilization: 0.95
dtype: bfloat16
```

### 7.5 Durable storage in Kubernetes

Move every on-disk artifact onto a PVC so pod rolls don't drop queue state or trained checkpoints:

```yaml
training_job_directory: /mnt/jobs
inference_work_queue_path: /mnt/queue/inference_work_queue.sqlite
upload_base_path: /mnt/queue/inference_requests
log_directory: /mnt/logs
```

The reference Helm chart (`deployment/helm/gemma_270m_tw/scalarlm/`) already mounts `scalarlm-jobs` at `/app/cray/jobs` and `scalarlm-cache` at `/root/.cache/huggingface`; add a PVC for the queue paths if you want crash-safe inference across pod rolls.

### 7.6 Extra vLLM flags not in Config

```bash
export SCALARLM_VLLM_ARGS="--enforce-eager --disable-custom-all-reduce --swap-space=16"
```

Useful for debugging and for flags that are too vLLM-specific to promote to `Config`.

### 7.7 Big training job

```python
train_args = {
    "llm_name": "Qwen/Qwen2-32B-Instruct",
    "max_steps": 5000,
    "learning_rate": 2e-5,
    "batch_size": 1,
    "gradient_accumulation_steps": 32,   # effective batch = 32 × world_size
    "steps_per_checkpoint": 250,
    "max_checkpoints_to_keep": 10,
    "gpus": 8,
    "nodes": 2,
    "distribution_strategy": "fsdp",
    "adapter_type": "lora",
    "lora_config": {"r": 16, "lora_alpha": 32, "target_modules": "all-linear"},
    "timeout": 57600,                    # 16h
}
```

Clamping happens on the server: `gpus` and `nodes` are capped by actual SLURM capacity; `timeout` is capped by server `max_train_time` plus `extra_training_seconds`.

---

## 8. Pitfalls and Invariants

**Env-var casts are type-strict.** `SCALARLM_GPU_MEMORY_UTILIZATION=0.9f` will raise `ValueError: could not convert string to float: '0.9f'`. No silent fallback.

**Booleans accept `true`/`1`/`yes` (case-insensitive); everything else is False.** Don't trust `on`/`off` or `y`/`n`.

**Job config ignores env vars.** Only server config has the `SCALARLM_*` tier. Per-job tweaks go in `train_args`.

**Config field names are case-sensitive in YAML.** `max_model_length: 32768` works; `Max_Model_Length: 32768` becomes an extra unrecognized field and Pydantic will reject it (default config uses `class Config`, not `class Config(extra="allow")`).

**`server_list: all` vs. `server_list: api,vllm,megatron`.** The first is the "do everything" alias checked explicitly in `one_server/main.py:34`. The second also works — `start_cray_server` parses the comma-separated list. In multi-pod Kubernetes, always use specific names so you know which pod is which.

**Helm ConfigMaps are per-component.** They are not copies of each other. Editing the API pod's ConfigMap does not change vLLM's. Mismatched `model` between API and vLLM ConfigMaps produces very confusing errors — always edit all three or use a shared Helm value.

**`SCALARLM_VLLM_ARGS` dedups by flag prefix.** `--dtype=bfloat16` passed here will replace the one built from `config["dtype"]`. Good for overrides; bad if you forget and set both.

**Changing `training_job_directory` mid-run orphans active jobs.** The directory path is baked into each `status.json` and into SLURM's `--output` path. Point the config at a new location only between runs.
