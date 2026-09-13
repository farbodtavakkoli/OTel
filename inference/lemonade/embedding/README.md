# `inference/lemonade/embedding` — Lemonade Server GGUF embeddings (EmbeddingGemma-300M)

## Overview & when to use

Serves **EmbeddingGemma-300M** (GGUF, Q8_0) through **Lemonade Server**, which drives a
managed `llama-server` subprocess built against **ROCm** and auto-detects the machine's
GPU architecture. The client `inference_embedding_lemonade.py` hits
`POST /v1/embeddings` on the Lemonade port (**8350** here).

Unlike the raw `llama.cpp` folder, Lemonade needs **no build and no GPU-arch choice**.
`lemonade backends
install llamacpp:rocm` inspects the host, sees `gfx950`, and pulls a
**`rocm_sdk_device_gfx950`** wheel plus a prebuilt `llama.cpp` ROCm binary in well under a
minute, with no compiler involved.

Chat, embeddings and reranking share one process, one port and one model registry, and a
model registered with `--label embeddings` is started with `--embeddings` on the wrapped
`llama-server` automatically — you never pass that flag yourself; omitting it on a raw
`llama.cpp` server returns 404. Use raw `llama.cpp` when you need server flags (`-ngl`, `-b`/`-ub`,
`--pooling`) that Lemonade does not expose per model.

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

`pip install lemonade-sdk` gets you the SDK and a **deprecated** Python server whose
backend choice is a serve-time flag (`lemonade-server-dev serve --llamacpp rocm`). The
documented `lemonade backends install llamacpp:rocm` only exists in the **C++ server**, which
ships as a release artifact, not on PyPI.

> **The pip server cannot serve on this GPU.** `lemonade-server-dev serve
> --llamacpp rocm` starts and answers `/api/v1/health` with 200, and `/api/v1/pull`
> succeeds, but the first inference request dies in
> `lemonade/tools/llamacpp/utils.py:375` with
> `ValueError: ROCm backend selected but no compatible ROCm target architecture found.`
> The pip server's ROCm support does not cover gfx950. Only the C++ server 11.7.0 does.


```bash
# (a) python venv — shared across the three inference_*_lemonade folders (see note)
python3 -m venv .env_lemonade
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
.env_lemonade/bin/pip install lemonade-sdk        # 9.1.4

# (b) the C++ server that actually implements `backends install`
cd $DATA_DIR/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
```

> **Do not use the `.deb`.** `lemonade-server_11.7.0-debian13_amd64.deb` is built for
> **Debian 13** and fails to load on Ubuntu 24.04 (missing `libmbedcrypto.so.16`,
> `libcpp-httplib.so.0.41`). Use the **`lemonade-embeddable-…-ubuntu-x64.tar.gz`** artifact.

### Shared venv

One venv serves all three `inference_*_lemonade` folders; the other two are symlinks:

```bash
ln -sfn ../../../inference/lemonade/llm/.env_lemonade .env_lemonade
```

The python side here is only the HTTP client; the actual runtime is the C++ server plus its
own managed ROCm venv under the Lemonade cache.

### Backend install

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

Lemonade resolves the GPU to `gfx950` itself and fetches the matching
`rocm_sdk_device_gfx950` wheel plus a prebuilt ROCm `llama.cpp`; no compiler is involved.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed for this model — `ggml-org/embeddinggemma-300M-GGUF` is ungated and pulls
with no `HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # this folder's GPUs
# HF_HOME (exported above) is where model weights land
```

- **`HF_HOME` controls the weights.** Lemonade downloads into `$HF_HOME/hub/` in an
  HF-hub-shaped tree (`snapshots/<sha>/<file>` + a `.lemonade_registry.json`).
- **The `cache_dir` positional controls the binaries.** `$DATA_DIR/lemonade/cache`
  holds the llama.cpp build and the ROCm wheels — **4.8 GB**. Neither should land on the
  root filesystem.
- These env vars must be set **on the `lemond` process**, not on the CLI client. `lemond`
  is what forks `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register EmbeddingGemma as an embedding model

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.EmbeddingGemma-300M \
  --checkpoint main ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --recipe llamacpp \
  --label embeddings
```

`--label embeddings` is what makes Lemonade start the wrapped server with `--embeddings`
and route `/v1/embeddings` to it.

### 3. Load and call it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.EmbeddingGemma-300M

.env_lemonade/bin/python inference_embedding_lemonade.py \
  --port 8350 \
  --out $OUTPUT_DIR/inference_embedding_lemonade/embeddings_two_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8350/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"EmbeddingGemma-300M","input":["a span is a unit of work in a trace"]}' \
  | jq '.data[0].embedding | length'
```

> The registered name is `user.EmbeddingGemma-300M`; the id the **API** answers to is
> `EmbeddingGemma-300M` (the `user.` prefix is stripped in `/v1/models`). Sending
> `"model": "user.EmbeddingGemma-300M"` to `/v1/embeddings` is a 404.

## Expected output

```text
endpoint   : http://127.0.0.1:8350/v1/embeddings
model      : EmbeddingGemma-300M
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2931
unrelated (0 vs 1) : 0.1026
```

768-dimensional, L2-normalised (`norm = 1.0000`), and the related pair must score above the
unrelated one.

## Multi-GPU

With two GPUs visible, llama.cpp's default `--split-mode layer` spreads the layers of even a
300M model across both cards, which buys nothing. **Run one Lemonade Server per GPU behind a
load balancer instead**; Lemonade offers no per-model `--tensor-split` / `-ngl` knob.

## GPU-residency check

A CPU fallback would still return correct-looking vectors, and Lemonade launches
`llama-server` **without an explicit `-ngl`**, so verify residency rather than assuming it.
The wrapped process (Lemonade proxies 8350 -> 8001) looks like:

```text
$DATA_DIR/lemonade/cache/bin/llamacpp/rocm-stable/llama-b10469/llama-server \
  -m $HF_HOME/hub/models--ggml-org--embeddinggemma-300M-GGUF/snapshots/<sha>/embeddinggemma-300M-Q8_0.gguf \
  --ctx-size 8192 --port 8001 --jinja --metrics --reasoning-format auto --no-ui --embeddings
```

Check it with `rocm-smi --showpids` — the kernel's own KFD accounting, which a CPU-only
process cannot appear in at all — and with the per-GPU byte counts in
`/sys/class/kfd/kfd/proc/<pid>/vram_*`. Expect GB-scale VRAM attributed to the
`llama-server` PID plus live HSA compute queues under
`/sys/class/kfd/kfd/proc/<pid>/queues/*/gpuid`.

## Cross-check vs the Transformers baseline

Lemonade/llama.cpp does **not** apply EmbeddingGemma's task prompts, while the
sentence-transformers baseline (`encode_query` / `encode_document`) does, so a comparison
must send the prefixed strings (`task: search result | query: …`, `title: none | text: …`).
`--reference` with no `--texts` handles that for you:

```bash
.env_lemonade/bin/python inference_embedding_lemonade.py \
  --reference $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_1gpu.json
```

## Arguments

### Lemonade Server / CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv live here. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs looking for beacons |
| `backends install llamacpp:rocm` | the working backend | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family for the model |
| `pull --label` | `embeddings` | **Required** — drives `--embeddings` on the wrapped server |
| `load <name>` | optional | Start/warm the subprocess before the first request |

### `inference_embedding_lemonade.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Lemonade Server host |
| `--port` | `8350` | Lemonade Server port |
| `--model` | `EmbeddingGemma-300M` | API model id (no `user.` prefix) |
| `--endpoint` | `/v1/embeddings` | Also accepts `/api/v1/embeddings` |
| `--texts` | 3-sentence probe | Texts to embed (2 related + 1 unrelated) |
| `--reference` | `None` | Baseline JSON to cosine-check against; adopts its `queries` when `--texts` is omitted |
| `--reference_prompt` | `task: search result \| query: ` | EmbeddingGemma `encode_query` template applied to baseline queries |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/v1/health` polls before giving up |
| `--load_model` | off | `POST /api/v1/load` first |
| `--out` | `None` | Write the raw JSON response here |

## Output

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_embedding_lemonade/` — never the root filesystem. Weights (319 MB)
live in `$HF_HOME/hub/`; Lemonade binaries + ROCm wheels (4.8 GB) in
`$DATA_DIR/lemonade/cache/`.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4 host):** works via `llamacpp:rocm`, which brings its own
  ROCm 7.14 and an arch-matched gfx950 device wheel.
- **NVIDIA H100 (Hopper sm_90):** works via `llamacpp:cuda` — see the H100 section below.
- The model is ungated: it pulls with no `HF_TOKEN` exported.

## Notes & quirks

1. **Two different products share the name.** pip `lemonade-sdk` != the C++ Lemonade
   Server. Only the latter has `backends install`. See the table in "Install".
2. **The Linux `.deb` is Debian-13-only** and will not load on Ubuntu 24.04
   (`libmbedcrypto.so.16`, `libcpp-httplib.so.0.41` missing). Use the `embeddable`
   tarball.
3. **`--no-discovery` on every CLI call.** Without it the client broadcasts UDP looking
   for servers and can hang well past 60 s on a multi-tenant host.
4. **The `user.` prefix is registration-only.** `/v1/models` reports
   `EmbeddingGemma-300M`; use that as `"model"`.
5. **No `-ngl` is exposed.** Lemonade builds the `llama-server` command line itself and
   does not pass `-ngl`; offload happens because the ROCm backend claims the layers.
   Prove residency via `rocm-smi --showpids`, not via server flags.
6. **You can pre-seed the HF cache to skip a re-download.** Mirroring an existing
   `snapshots/<sha>/<file>` tree into `$HF_HOME/hub/<repo>/` with a matching
   `refs/main` and `.lemonade_registry.json` makes `pull` report `(already downloaded)`.
   Used here to avoid re-fetching the 28 GB LLM GGUF — see the LLM folder.
7. **Embeddings are not reproducible across a live server's request history.** The same
   request, to the same PID, returns different vectors depending on what was asked
   *before* it:

   vectors are stable while the batch shape stays constant, and shift by ~1e-3 per component
   after a request of a different shape. The drift comes from `llama-server`'s slot/KV state
   and batch packing, not from Lemonade, and it is semantically harmless (the related/
   unrelated ordering holds). But **identical text indexed at different times yields slightly
   different vectors**: if you need byte-reproducible embeddings, batch a corpus in
   fixed-shape requests. `--load_model` also perturbs the slot, which is why the reference
   runs above do not use it.
8. **The Lemonade cache is large.** 4.8 GB for one backend (the gfx950 ROCm device
   wheel alone is 1.2 GB). Point `lemond` at a big filesystem.
9. **Missing-resource warnings at startup are harmless** for the embeddable build:
   `Could not load architecture_defaults.json`, `Web app directory not found`. Only the
   web UI is affected; the API is complete.

## H100 (NVIDIA, Hopper sm_90)

Single-GPU, host CUDA 13.0, Python 3.12, with the **exact same GGUF** as the MI355X run
(`ggml-org/embeddinggemma-300M-GGUF:Q8_0`, 318 MB).

**Install = the AMD route with `llamacpp:cuda` in place of `llamacpp:rocm`.** Same
embeddable tarball, same `backends install`; it pulls the arch-matched Hopper prebuilt
`llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` with no compiler, bundling its own CUDA 12.9
runtime. Confirm with `Using LlamaCpp Backend: cuda` in the `lemond` log. Registration is
byte-for-byte the MI355X command — **`--label embeddings` is still what routes
`/v1/embeddings`**:

```bash
LEM=/dev/shm/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=0   # HF_HOME as exported above
$LEM/lemonade --port 8350 --no-discovery pull user.EmbeddingGemma-300M \
  --checkpoint main ggml-org/embeddinggemma-300M-GGUF:Q8_0 --recipe llamacpp --label embeddings
$LEM/lemonade --port 8350 --no-discovery load user.EmbeddingGemma-300M
```

**Smoke command and expected output:**

```bash
.env_lemonade/bin/python inference_embedding_lemonade.py --port 8350 \
  --model EmbeddingGemma-300M --out $OUTPUT_DIR/lemonade/embeddings_single_gpu.json
```

```text
endpoint   : http://127.0.0.1:8350/v1/embeddings
model      : EmbeddingGemma-300M
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11443, 0.04268, 0.01675, -0.00695, -0.0285, 0.01905, 0.04997, 0.06326]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2924
unrelated (0 vs 1) : 0.1038
```

**768-dim, L2-normalised**, related above unrelated. Vectors differ from the MI355X run by
~1e-3 per component (CUDA-vs-ROCm accumulation order on the same Q8_0 GGUF).

**Wrapped cmdline (`--embeddings` auto-appended, `--ctx-size 8192` as on MI355X):**

```text
/dev/shm/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--ggml-org--embeddinggemma-300M-GGUF/…/embeddinggemma-300M-Q8_0.gguf \
  --ctx-size 8192 --port 8002 --jinja --metrics --reasoning-format auto --no-ui --embeddings
```

**GPU-residency check.** `nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid`
must attribute memory to the `llama-server` PID on the selected GPU. A CPU fallback — which
would still return correct-looking vectors — shows 0 MiB, so this is the decisive check.

**Multi-GPU:** run one server per GPU, same as MI355X.

**NVIDIA-side note:** build the client venv on a local disk or tmpfs — a `venv` on an NFS
mount does not create pip/console scripts reliably (use `python -m pip` there). Model, quant,
client and API are otherwise identical to MI355X.
