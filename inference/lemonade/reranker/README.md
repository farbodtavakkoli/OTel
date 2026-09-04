# `inference/lemonade/reranker` — Lemonade Server GGUF reranking (Qwen3-Reranker-0.6B) on ROCm/gfx950

## Overview & when to use

Serves **Qwen3-Reranker-0.6B** (GGUF, Q8_0) through **Lemonade Server**, which manages a
`llama-server` subprocess built against **ROCm** and started with `--reranking`
automatically. The client `inference_reranker_lemonade.py` hits **`POST /v1/reranking`**,
the documented endpoint for this workload.

**This folder is the reason Lemonade gets three folders and Ollama gets two.** Lemonade is
documented as "a stronger all-in-one local option than Ollama for this exact
three-workload benchmark" precisely because reranking is a first-class local API here —
Ollama has no equivalent. Chat, embeddings and reranking all live behind one process on
one port with one model registry.

Use this folder when you want:

- **Local reranking with no bespoke serving code** — register the GGUF with
  `--label reranking` and the endpoint appears.
- **A single service for a whole RAG loop** — embed, retrieve, rerank, then generate,
  all against `127.0.0.1:8350`.
- **Zero-build ROCm** — `lemonade backends install llamacpp:rocm` detected `gfx950` and
  fetched a matching prebuilt in 27 s. No `cmake`, no arch flags.

Prefer **vLLM** or **SGLang** for production/high-QPS reranking, and prefer raw
`llama.cpp` when you need server flags Lemonade does not expose per model.

> **Tested topology:** 2x AMD Instinct MI355X (gfx950, 288 GB each), physical GPUs
> **4 and 5**, ROCm 7.2.4 host, Ubuntu 24.04, Python 3.12.3. Verified **2026-08-20**.

## Install — the exact route that worked

### The package-name trap: two different Lemonades

| What | Command it gives you | Has `backends install`? | Verdict here |
|---|---|---|---|
| pip `lemonade-sdk==9.1.4` | `lemonade`, `lemonade-server-dev` | **No** | Deprecated python server — **its ROCm backend rejects gfx950** (tested, see below) |
| C++ Lemonade Server **11.7.0** (GitHub releases) | `lemonade`, `lemond` | **Yes** | **This is the documented server — used here** |

> **The pip server cannot serve on this GPU.** Tested: `lemonade-server-dev serve
> --llamacpp rocm` starts and answers `/api/v1/health` with 200, and `/api/v1/pull`
> succeeds, but the first inference request dies in
> `lemonade/tools/llamacpp/utils.py:375` with
> `ValueError: ROCm backend selected but no compatible ROCm target architecture found.`
> The pip server's ROCm support does not cover gfx950. Only the C++ server 11.7.0 does.
> Transcript: `/mnt/data_450g/outputs/pip_lemonade_server_rocm_gfx950_unsupported.txt`.

```bash
# (a) python venv — shared across the three inference_*_lemonade folders
python3 -m venv .env_inference_llm_lemonade
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache
.env_inference_llm_lemonade/bin/pip install lemonade-sdk        # 9.1.4, 8.8 s

# (b) the C++ server that actually implements `backends install`
cd /mnt/data_450g/lemonade
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
ln -sfn ../../../inference/lemonade/llm/.env_inference_llm_lemonade .env_inference_reranker_lemonade
```

Deliberate — the python side is only the HTTP client. The runtime is the C++ server plus
its own managed ROCm venv inside the Lemonade cache.

### Backend install — and what it chose

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

```text
Installing backend: llamacpp:rocm
[1/2] llama-b10470-bin-ubuntu-rocm-7.14-x64.tar.gz
[2/2] rocm_sdk_core-7.14.0…whl (414.6 MB)
      rocm_sdk_libraries-7.14.0…whl (557.6 MB)
      rocm_sdk_device_gfx950-7.14.0…whl (1240.1 MB)
Backend installed successfully: llamacpp:rocm      # real 0m27.3s
```

Selected backend: **`llamacpp:rocm`, build `b10470`** (binaries labelled `b10469`), with an
arch-matched **`rocm_sdk_device_gfx950`** wheel.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed — `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` is ungated and was pulled with no
`HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4       # single-GPU runs
export HF_HOME=/mnt/data_1.5t/hf_cache                    # where model weights land
```

- **`HF_HOME` controls the weights** (`$HF_HOME/hub/`, 610 MB for this model).
- **The `cache_dir` positional controls the binaries** — `/mnt/data_450g/lemonade/cache`,
  4.8 GB. Neither may land on `/` (99 GB free).
- Set these on the **`lemond`** process, not on the CLI client: `lemond` forks
  `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4
export HF_HOME=/mnt/data_1.5t/hf_cache

$LEM/lemond /mnt/data_450g/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register the reranker

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-Reranker-0.6B \
  --checkpoint main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --recipe llamacpp \
  --label reranking
```

`--label reranking` is the whole trick — it is what makes Lemonade append `--reranking`
to the wrapped `llama-server` and route `/v1/reranking` to it.

### 3. Load and call it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-Reranker-0.6B

.env_inference_reranker_lemonade/bin/python inference_reranker_lemonade.py \
  --port 8350 \
  --out /mnt/data_450g/outputs/inference_reranker_lemonade/rerank_single_gpu.json
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

All four of these returned byte-identical bodies:

```text
/v1/reranking      -> HTTP 200   <-- the documented route, and this client's default
/api/v1/reranking  -> HTTP 200
/v1/rerank         -> HTTP 200
/api/v1/rerank     -> HTTP 200
```

## Results — single GPU (physical GPU 4)

Real client output:

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
latency_s : 0.877
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999667  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request through a system
2. score=0.996421  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=0.000036  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000022  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

The ordering is not merely correct, it is **decisively** correct: both OpenTelemetry
documents score **>0.996** and both distractors score **<4e-05** — a **~28,000x**
separation between the worst relevant document and the best irrelevant one. The reranker
also prefers the document that *defines nesting* (idx 2, 0.999667) over the one that
merely mentions a span (idx 0, 0.996421), which is the finer-grained judgement a
bi-encoder embedding model would not make.

## GPU-residency proof

A CPU fallback would return the same scores, so residency has to be proven. Lemonade
launches `llama-server` **without an explicit `-ngl`**:

```text
/mnt/data_450g/lemonade/cache/bin/llamacpp/rocm-stable/llama-b10469/llama-server \
  -m /mnt/data_1.5t/hf_cache/hub/models--ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF/snapshots/a02f48bb.../qwen3-reranker-0.6b-q8_0.gguf \
  --ctx-size 40960 --port 8002 --jinja --metrics --reasoning-format auto --no-ui --reranking
```

`rocm-smi --showpids` — kernel KFD accounting, which a CPU-only process cannot appear in:

```text
PID      PROCESS NAME    GPU(s)  VRAM USED     SDMA USED       CU OCCUPANCY
1966617  llama-server    1       7818108928    2681185302118   0
```

Per-GPU breakdown from `/sys/class/kfd/kfd/proc/1966617/`:

```text
vram_51023 = 7818108928 bytes   -> gpu_id 51023 = unique_id 0xf743d583ac01fcfd = card4
(every other vram_* entry for this PID is 0)
7 live HSA compute queues
```

**7.82 GB on card 4 only.** The 610 MB of weights plus a 40960-token KV cache and the HIP
context account for the rest.

## Results — two GPUs (physical GPUs 4 and 5)

With `HIP_VISIBLE_DEVICES=4,5`, llama.cpp's default `--split-mode layer` really does put
some layers on each card:

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
latency_s : 0.077
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999659  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request through a system
2. score=0.996421  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=0.000036  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000022  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

```text
PID      PROCESS NAME    GPU(s)  VRAM USED     CU OCCUPANCY
1901569  llama-server    2       10147651584   0
```

**A 0.6B model does not shard, and this README will not pretend otherwise.** The split is
real but pointless: 600 MB of weights on a card with 288 GB of VRAM is 0.2 % occupancy,
and a layer split only adds a device-to-device hop per forward pass. Scores are identical
to 6 decimal places except idx 2 (0.999659 vs 0.999667, a ~8e-06 reassociation
difference), so the split changes nothing but the plumbing.

**The honest scale-out pattern for a 0.6B reranker is one Lemonade Server per GPU behind a
load balancer** — N independent single-GPU replicas, which scale linearly and share
nothing. Lemonade exposes no per-model `--tensor-split` / `-ngl` knob, so a smarter split
is not available even if you wanted one.

## Arguments

### Lemonade Server / CLI

| Argument | Used | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `/mnt/data_450g/lemonade/cache` | Binaries + backend venv. Keep off `/` |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs |
| `backends install llamacpp:rocm` | used | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family |
| `pull --label` | `reranking` | **Required** — drives `--reranking` on the wrapped server |
| `load <name>` | used | Start/warm the subprocess before timing |

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

Artifacts go to `/mnt/data_450g/outputs/inference_reranker_lemonade/`, never to `/`:

```text
rerank_single_gpu.json         raw /v1/reranking response, GPU 4
client_single_gpu.txt          client stdout, GPU 4
rerank_two_gpu.json            raw response, GPUs 4+5
client_two_gpu.txt             client stdout, GPUs 4+5
server_cmdline_single_gpu.txt  the wrapped llama-server command Lemonade built
rocm_smi_pids_single_gpu.txt   KFD process attribution
kfd_vram_single_gpu.txt        per-GPU VRAM for our PID
```

Weights (610 MB) live in `/mnt/data_1.5t/hf_cache/hub/`; Lemonade binaries + ROCm wheels
(4.8 GB) in `/mnt/data_450g/lemonade/cache/`.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| Lemonade detects gfx950 | `backends --all` prints `Unsupported GPU: gfx950` per-recipe -> it resolved the arch |
| `llamacpp:rocm` supported on gfx950 | Listed `installable`, then `installed b10470` |
| Backend really carries gfx950 code | Install pulled `rocm_sdk_device_gfx950-7.14.0…whl` (1240 MB); `libggml-hip.so` (1.26 GB fat binary) contains a `gfx950` code object alongside gfx900/906/908/942/10xx/11xx/12xx |
| No `hipErrorNoBinaryForGpu` | Model loaded and ran; zero HSA/ISA errors in the server log |
| ROCm backend is the live device | Server log names `device 'ROCm0'` at sampler-setup time |
| Model really on GPU | `rocm-smi --showpids`: PID 1966617, **1 GPU, 7.82 GB**, 7 live KFD queues |
| Reranking endpoint exists | `POST /v1/reranking` -> HTTP 200 with `results[].relevance_score` |
| Scores are *correct* | relevant 0.9997 / 0.9964 vs irrelevant 3.6e-05 / 2.2e-05 |
| Two-GPU run agrees | Same ordering, scores identical to ~1e-05 |
| **`vllm:rocm` NOT supported** | `Error: Cannot install vllm:rocm on this system: Unsupported GPU: gfx950` |
| Ungated download | Pulled with no `HF_TOKEN` exported |

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
7. **First call after `load` is slower.** 0.877 s cold vs 0.077 s warm — the `load`
   endpoint returns before the first graph is built. Warm up before benchmarking.
8. **No `-ngl` is exposed.** Lemonade builds the `llama-server` command line itself.
   Prove residency via `rocm-smi --showpids`, not via server flags.
9. **`--ctx-size 40960` is chosen for you** and is not overridable per model.
10. **Harmless startup warnings** for the embeddable build: `Could not load
    architecture_defaults.json`, `Web app directory not found`. Only the web UI is
    affected; the API is complete.

## H100 (NVIDIA, Hopper sm_90) — verified 2026-08-22

Single-GPU smoke on **physical GPU 6** (`CUDA_VISIBLE_DEVICES=6`), driver 580.173.02,
host CUDA 13.0, Python 3.12.3. **Exact same GGUF** as MI355X
(`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0`, 610 MB).

**Install = the AMD route with `llamacpp:cuda` in place of `llamacpp:rocm`.** Same
embeddable tarball, same `backends install`; arch-matched Hopper prebuilt
`llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` (build **b10397**) in **~41 s, no compiler**,
bundling its own CUDA 12.9 runtime. `Using LlamaCpp Backend: cuda` /
`Respecting existing CUDA_VISIBLE_DEVICES=6`. Registration is identical, and
**`--label reranking` is still the whole trick** — it makes Lemonade append `--reranking`
to the wrapped `llama-server` (confirmed in the cmdline below) and route `/v1/reranking`:

```bash
LEM=/dev/shm/h100/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-Reranker-0.6B \
  --checkpoint main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --recipe llamacpp --label reranking
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-Reranker-0.6B
```

**Exact smoke command + real output** (ranking order, not absolute scores, is what matters):

```bash
.env_lemonade/bin/python inference_reranker_lemonade.py --port 8350 \
  --model Qwen3-Reranker-0.6B --out /dev/shm/h100/out/lemonade/rerank_single_gpu.json
```

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
latency_s : 0.149
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999679  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request through a system
2. score=0.996564  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=0.000047  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000025  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

**Ranking is identical to the MI355X run**, and decisively separated: both OpenTelemetry
docs >0.996, both distractors <5e-05 (**~21,000× separation**), and the finer judgement
(idx 2 "nesting" definition > idx 0 bare mention) is preserved. Scores match ROCm to ~4
decimals (idx2 0.999679 vs 0.999667, idx0 0.996564 vs 0.996421). All four route aliases
returned HTTP 200: `/v1/reranking`, `/api/v1/reranking`, `/v1/rerank`, `/api/v1/rerank`.

**Wrapped cmdline (`--reranking` auto-appended):**

```text
/dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF/…/qwen3-reranker-0.6b-q8_0.gguf \
  --ctx-size 4096 --port 8003 --jinja --metrics --reasoning-format auto --no-ui --reranking
```

**GPU-residency proof** (`nvidia-smi` VRAM-by-PID, GPU-6 UUID `GPU-e4fe48bc` = index 6):

```text
1725397, /dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server, 1952 MiB, GPU-e4fe48bc-…
```

**1.95 GB on GPU 6 for our PID.** A CPU-only reranker returns the same scores, so this VRAM
attribution is the proof — and it passes.

**One-server-three-workloads, proven on H100.** All three llama-servers (llm 8001, embed
8002, rerank 8003) were forked by the **single** `lemond` (port 8350) and were resident on
GPU 6 simultaneously — total 1674 + 982 + 1952 = **4628 MiB on GPU-e4fe48bc**. This is the
claim that justifies Lemonade getting three folders: chat, embeddings and reranking behind
one process on one port, and reranking is a first-class local API with no Ollama equivalent.

**Deviations from MI355X:** (1) backend `llamacpp:cuda` (b10397) vs `llamacpp:rocm`
(b10470); (2) **`--ctx-size 4096`** here vs `40960` on MI355X — Lemonade chose a smaller
default context for this model on this build (still fixed / not per-model overridable);
(3) client venv in tmpfs (NFS `venv` on `/mnt/gsma` didn't create pip reliably). Model,
quant, `--label reranking`, endpoints and ranking are otherwise identical.

**Multi-GPU (deferred):** a 0.6B reranker does not shard — run one server per GPU behind a
load balancer, same conclusion as MI355X. Not launched (production on GPUs 0–3).

**H100 VERDICT: PASS — and, as on MI355X, the strongest of the three folders.** Zero-build
sm_90 CUDA backend in ~41 s; `/v1/reranking` returns correct, decisively-separated scores
(~21,000× apart) in ~149 ms; 1.95 GB resident on GPU 6. **No `vllm:cuda` backend exists in
Lemonade** (`vllm` is rocm-only and reports `Unsupported GPU` here), so the llama.cpp CUDA
path is the only Lemonade path on NVIDIA — symmetric to gfx950, where `vllm:rocm` was
refused. For production/high-QPS reranking, prefer `inference/vllm` or `inference/sglang`.

## VERDICT (MI355X)

**PASS — fully working on MI355X (gfx950), and the strongest of the three folders.**

`lemonade backends install llamacpp:rocm` auto-detected `gfx950`, fetched an arch-matched
`rocm_sdk_device_gfx950` wheel plus a prebuilt ROCm `llama.cpp`, and was ready in **27
seconds** with no compiler. The feared prebuilt-backend failure mode
(`hipErrorNoBinaryForGpu` from a gfx90a/gfx942-only build) **did not occur** — the fat
`libggml-hip.so` carries gfx950.

`POST /v1/reranking` returns **correct and decisively separated** scores (0.9997 / 0.9964
relevant vs 3.6e-05 / 2.2e-05 irrelevant, ~28,000x apart) in **77 ms warm**, with **7.82 GB
of VRAM and 7 live KFD compute queues on GPU 4** proving GPU residency.

This is the workload that justifies the claim: **reranking is a first-class local
API in Lemonade and has no Ollama equivalent.** Multi-GPU is not a win for a 0.6B model —
run **one server per GPU** instead.

The experimental **`vllm:rocm` backend is refused outright on this GPU** —
`Unsupported GPU: gfx950` — so the llama.cpp path is the only Lemonade path here.

## Follow-ups

- Measure sustained QPS with a concurrent client; 77 ms for a 4-document request badly
  understates capacity.
- Compare against `inference/vllm/reranker` / `inference/sglang/reranker` for a
  throughput number at real RAG batch sizes.
- Check behaviour past `--ctx-size 40960` with long documents — Lemonade fixes the context
  and offers no per-model override.
- Re-test `vllm:rocm` on a later Lemonade release; `resources/backend_versions.json`
  already pins a gfx950 vLLM build, so the runtime gate looks like the only blocker.
