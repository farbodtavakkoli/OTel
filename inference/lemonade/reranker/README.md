# `inference/lemonade/reranker` — Lemonade Server GGUF reranking (Qwen3-Reranker-0.6B)

## Overview & when to use

Serves **Qwen3-Reranker-0.6B** (GGUF, Q8_0) through **Lemonade Server**, which manages a
`llama-server` subprocess built against **ROCm** and started with `--reranking`
automatically. The client `inference_reranker_lemonade.py` hits **`POST /v1/reranking`**,
the documented endpoint for this workload.

Chat, embeddings and reranking all live behind one process on one port with one model
registry: register the GGUF with `--label reranking` and the endpoint appears, so a whole RAG
loop (embed, retrieve, rerank, generate) runs against `127.0.0.1:8350`. Use
`inference/vllm` or `inference/sglang` for high-QPS reranking, and raw `llama.cpp` when you
need server flags Lemonade does not expose per model.

## Install — the route that works

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache (weights land here)
export DATA_DIR=/path/to/data          # Lemonade tarball, binaries and backend cache
export OUTPUT_DIR=/path/to/outputs     # inference artifacts
```

### Two different Lemonades share the name

| What | Command it gives you | Has `backends install`? | Notes |
|---|---|---|---|
| pip `lemonade-sdk==9.1.4` | `lemonade`, `lemonade-server-dev` | **No** | Deprecated python server — **its ROCm backend rejects gfx950** (see below) |
| C++ Lemonade Server **11.7.0** (GitHub releases) | `lemonade`, `lemond` | **Yes** | **This is the documented server — use this one** |

> **The pip server cannot serve on this GPU.** `lemonade-server-dev serve
> --llamacpp rocm` starts and answers `/api/v1/health` with 200, and `/api/v1/pull`
> succeeds, but the first inference request dies in
> `lemonade/tools/llamacpp/utils.py:375` with
> `ValueError: ROCm backend selected but no compatible ROCm target architecture found.`
> The pip server's ROCm support does not cover gfx950. Only the C++ server 11.7.0 does.

```bash
# (a) python venv — shared across the three inference_*_lemonade folders
python3 -m venv .env_lemonade
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
.env_lemonade/bin/pip install lemonade-sdk        # 9.1.4

# (b) the C++ server that actually implements `backends install`
cd $DATA_DIR/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
```

> **Do not use the `.deb`.** `lemonade-server_11.7.0-debian13_amd64.deb` is built for
> Debian 13 and will not load on Ubuntu 24.04:
> `error while loading shared libraries: libmbedcrypto.so.16` (also
> `libcpp-httplib.so.0.41`). The `embeddable` tarball has no unresolved deps here.

### Shared venv

One venv serves all three `inference_*_lemonade` folders; the other two are symlinks:

```bash
ln -sfn ../../../inference/lemonade/llm/.env_lemonade .env_lemonade
```

The python side is only the HTTP client; the runtime is the C++ server plus its own managed
ROCm venv inside the Lemonade cache.

### Backend install

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

Lemonade resolves the GPU to `gfx950` itself and installs `llamacpp:rocm` with an
arch-matched **`rocm_sdk_device_gfx950`** wheel; no compiler is involved.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed — `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` is ungated and pulls with no
`HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0       # single-GPU runs
# HF_HOME (exported above) is where model weights land
```

- **`HF_HOME` controls the weights** (`$HF_HOME/hub/`, 610 MB for this model).
- **The `cache_dir` positional controls the binaries** — `$DATA_DIR/lemonade/cache`,
  4.8 GB. Neither should land on the root filesystem.
- Set these on the **`lemond`** process, not on the CLI client: `lemond` forks
  `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0

$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register the reranker

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-Reranker-0.6B \
  --checkpoint main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --recipe llamacpp \
  --label reranking
```

`--label reranking` is what makes Lemonade append `--reranking` to the wrapped
`llama-server` and route `/v1/reranking` to it.

### 3. Load and call it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-Reranker-0.6B

.env_lemonade/bin/python inference_reranker_lemonade.py \
  --port 8350 \
  --out $OUTPUT_DIR/inference_reranker_lemonade/rerank_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8350/v1/reranking \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-Reranker-0.6B","query":"What is an OpenTelemetry span?",
       "documents":["A span represents a single unit of work in a distributed trace.",
                    "To bake sourdough, feed the starter twelve hours before mixing."]}'
```

```json
{"model":"Qwen3-Reranker-0.6B","object":"list","results":[
  {"index":0,"relevance_score":0.9959725737571716},
  {"index":1,"relevance_score":5.141833025845699e-05}],
 "usage":{"prompt_tokens":185,...}}
```

> The registered name is `user.Qwen3-Reranker-0.6B`; the id the **API** answers to is
> `Qwen3-Reranker-0.6B` (the `user.` prefix is stripped in `/v1/models`).

### Route aliases

All four of these return byte-identical bodies:

```text
/v1/reranking      -> HTTP 200   <-- the documented route, and this client's default
/api/v1/reranking  -> HTTP 200
/v1/rerank         -> HTTP 200
/api/v1/rerank     -> HTTP 200
```

## Expected output

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999667  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request through a system
2. score=0.996421  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=0.000036  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000022  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

Both OpenTelemetry documents must rank above both distractors (`top_hit_is_relevant: True`).

## GPU-residency check

A CPU fallback would return the same scores, so verify residency. Lemonade
launches `llama-server` **without an explicit `-ngl`**:

```text
$DATA_DIR/lemonade/cache/bin/llamacpp/rocm-stable/llama-b10469/llama-server \
  -m $HF_HOME/hub/models--ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF/snapshots/<sha>/qwen3-reranker-0.6b-q8_0.gguf \
  --ctx-size 40960 --port 8002 --jinja --metrics --reasoning-format auto --no-ui --reranking
```

Check it with `rocm-smi --showpids` — kernel KFD accounting, which a CPU-only process cannot
appear in — and with the per-GPU byte counts in `/sys/class/kfd/kfd/proc/<pid>/vram_*`.
Expect GB-scale VRAM (weights plus the 40960-token KV cache and HIP context) on a single
visible GPU with several live HSA compute queues.

## Multi-GPU

With two GPUs visible, llama.cpp's default `--split-mode layer` puts some layers on each
card, which a 0.6B model does not need. **Run one Lemonade Server per GPU behind a load
balancer instead**; Lemonade exposes no per-model `--tensor-split` / `-ngl` knob.

## Arguments

### Lemonade Server / CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs |
| `backends install llamacpp:rocm` | the working backend | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family |
| `pull --label` | `reranking` | **Required** — drives `--reranking` on the wrapped server |
| `load <name>` | optional | Start/warm the subprocess before the first request |

### `inference_reranker_lemonade.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Lemonade Server host |
| `--port` | `8350` | Lemonade Server port |
| `--model` | `Qwen3-Reranker-0.6B` | API model id (no `user.` prefix) |
| `--endpoint` | `/v1/reranking` | Also accepts `/api/v1/reranking`, `/v1/rerank`, `/api/v1/rerank` |
| `--query` | OpenTelemetry span question | Query to rank against |
| `--documents` | 2 relevant + 2 irrelevant | Candidate documents |
| `--top_n` | `None` | Return only the top N |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/v1/health` polls before giving up |
| `--load_model` | off | `POST /api/v1/load` first |
| `--out` | `None` | Write the raw JSON response here |

## Output

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_reranker_lemonade/` — never the root filesystem. Weights (610 MB)
live in `$HF_HOME/hub/`; Lemonade binaries + ROCm wheels (4.8 GB) in
`$DATA_DIR/lemonade/cache/`.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4 host):** works via `llamacpp:rocm`, which brings its own
  ROCm 7.14 and an arch-matched gfx950 device wheel. `vllm:rocm` is **refused**
  (`Unsupported GPU: gfx950`).
- **NVIDIA H100 (Hopper sm_90):** works via `llamacpp:cuda` — see the H100 section below.
  There is no `vllm:cuda` backend.
- The model is ungated: it pulls with no `HF_TOKEN` exported.

## Notes & quirks

1. **Two different products share the name.** pip `lemonade-sdk` != the C++ Lemonade
   Server. Only the latter has `backends install`.
2. **The Linux `.deb` is Debian-13-only.** Use the `embeddable` tarball on Ubuntu 24.04.
3. **`--no-discovery` on every CLI call.** Without it the client broadcasts UDP looking
   for servers and hangs well past 60 s on this multi-tenant host.
4. **`--label reranking` is mandatory.** Without it the model is treated as a chat model,
   `--reranking` is never passed to `llama-server`, and `/v1/reranking` fails.
5. **The `user.` prefix is registration-only.** `/v1/models` reports
   `Qwen3-Reranker-0.6B`; use that as `"model"`.
6. **Four route aliases, one implementation.** `/v1/reranking` (the documented route),
   `/api/v1/reranking`, `/v1/rerank`, `/api/v1/rerank` all return the same body.
7. **The first call after `load` is slower** — the `load` endpoint returns before the first
   graph is built. Warm up before timing anything.
8. **No `-ngl` is exposed.** Lemonade builds the `llama-server` command line itself.
   Prove residency via `rocm-smi --showpids`, not via server flags.
9. **`--ctx-size 40960` is chosen for you** and is not overridable per model.
10. **Harmless startup warnings** for the embeddable build: `Could not load
    architecture_defaults.json`, `Web app directory not found`. Only the web UI is
    affected; the API is complete.

## H100 (NVIDIA, Hopper sm_90)

Single-GPU, host CUDA 13.0, Python 3.12, with the **exact same GGUF** as MI355X
(`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0`, 610 MB).

**Install = the AMD route with `llamacpp:cuda` in place of `llamacpp:rocm`.** Same
embeddable tarball, same `backends install`; arch-matched Hopper prebuilt
`llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` with no compiler, bundling its own CUDA 12.9
runtime. Confirm with `Using LlamaCpp Backend: cuda` in the `lemond` log. Registration is
identical, and **`--label reranking` is still what drives the route** — it makes Lemonade append
`--reranking` to the wrapped `llama-server` (see the cmdline below) and route
`/v1/reranking`:

```bash
LEM=/dev/shm/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=0   # HF_HOME as exported above
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-Reranker-0.6B \
  --checkpoint main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --recipe llamacpp --label reranking
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-Reranker-0.6B
```

**Smoke command and expected output** (ranking order, not absolute scores, is what matters):

```bash
.env_lemonade/bin/python inference_reranker_lemonade.py --port 8350 \
  --model Qwen3-Reranker-0.6B --out $OUTPUT_DIR/lemonade/rerank_single_gpu.json
```

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999679  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request through a system
2. score=0.996564  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=0.000047  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000025  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

**Ranking is identical to the MI355X run.** All four route aliases return HTTP 200:
`/v1/reranking`, `/api/v1/reranking`, `/v1/rerank`, `/api/v1/rerank`.

**Wrapped cmdline (`--reranking` auto-appended):**

```text
/dev/shm/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF/…/qwen3-reranker-0.6b-q8_0.gguf \
  --ctx-size 4096 --port 8003 --jinja --metrics --reasoning-format auto --no-ui --reranking
```

**GPU-residency check** (`nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid`):
memory must be attributed to the `llama-server` PID on the selected GPU. A CPU-only reranker
returns the same scores, so this VRAM attribution is the decisive check.

**One server, three workloads.** All three llama-servers (llm 8001, embed 8002, rerank 8003)
are forked by the **single** `lemond` on port 8350 and can be resident on one GPU
simultaneously.

**Deviations from MI355X:** (1) backend `llamacpp:cuda` instead of `llamacpp:rocm`;
(2) **`--ctx-size 4096`** here vs `40960` on MI355X — Lemonade picks it per build and it is
not overridable; (3) build the client venv on local disk or tmpfs (a `venv` on an NFS mount
does not create pip reliably). Model, quant, `--label reranking`, endpoints and ranking are
otherwise identical.

**Multi-GPU:** run one server per GPU behind a load balancer, same as MI355X.
