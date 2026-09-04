# `inference/lemonade/embedding` — Lemonade Server GGUF embeddings (EmbeddingGemma-300M) on ROCm/gfx950

## Overview & when to use

Serves **EmbeddingGemma-300M** (GGUF, Q8_0) through **Lemonade Server**, which drives a
managed `llama-server` subprocess built against **ROCm** and auto-detects the machine's
GPU architecture. The client `inference_embedding_lemonade.py` hits
`POST /v1/embeddings` on the Lemonade port (**8350** here).

The thing that distinguishes Lemonade from the raw `llama.cpp` folder next door is that
**you do not build anything and you do not choose a GPU arch**. `lemonade backends
install llamacpp:rocm` inspected this host, saw `gfx950`, and pulled a
**`rocm_sdk_device_gfx950`** wheel plus a prebuilt `llama.cpp` ROCm binary. Total time:
**27 seconds**, no compiler involved.

Use this folder when you want:

- **One local OpenAI-compatible service for all three workloads** — chat, embeddings and
  reranking share one process, one port, one model registry. This is the headline
  reason to prefer Lemonade over Ollama for this benchmark: Ollama has no reranking API.
- **Zero-build ROCm deployment** — no `cmake`, no `-DGPU_TARGETS`, no `libssl-dev`.
- **Automatic model-type handling** — a model registered with `--label embeddings` is
  started with `--embeddings` on the wrapped `llama-server` automatically. You never pass
  the flag yourself (and never forget it, which is the classic llama.cpp 404).

Prefer **HF TEI** or **vLLM** for maximum embedding throughput, and prefer raw
`llama.cpp` when you need to control server flags (`-ngl`, `-b`/`-ub`, `--pooling`) that
Lemonade does not currently expose per model.

> **Tested topology:** 2x AMD Instinct MI355X (gfx950, 288 GB each), physical GPUs
> **4 and 5**, ROCm 7.2.4 host, Ubuntu 24.04, Python 3.12.3. Verified **2026-08-20**.

## Install — the exact route that worked

### The package-name trap: two different Lemonades

| What | Command it gives you | Has `backends install`? | Verdict here |
|---|---|---|---|
| pip `lemonade-sdk==9.1.4` | `lemonade`, `lemonade-server-dev` | **No** | Deprecated python server — **its ROCm backend rejects gfx950** (tested, see below) |
| C++ Lemonade Server **11.7.0** (GitHub releases) | `lemonade`, `lemond` | **Yes** | **This is the documented server — used here** |

`pip install lemonade-sdk` gets you the SDK and a **deprecated** Python server whose
backend choice is a serve-time flag (`lemonade-server-dev serve --llamacpp rocm`). The
documented `lemonade backends install llamacpp:rocm` only exists in the **C++ server**, which
ships as a release artifact, not on PyPI.

> **The pip server cannot serve on this GPU.** Tested: `lemonade-server-dev serve
> --llamacpp rocm` starts and answers `/api/v1/health` with 200, and `/api/v1/pull`
> succeeds, but the first inference request dies in
> `lemonade/tools/llamacpp/utils.py:375` with
> `ValueError: ROCm backend selected but no compatible ROCm target architecture found.`
> The pip server's ROCm support does not cover gfx950. Only the C++ server 11.7.0 does.
> Transcript: `/mnt/data_450g/outputs/pip_lemonade_server_rocm_gfx950_unsupported.txt`.


```bash
# (a) python venv — shared across the three inference_*_lemonade folders (see note)
python3 -m venv .env_inference_llm_lemonade
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache
.env_inference_llm_lemonade/bin/pip install lemonade-sdk        # 9.1.4, 8.8 s

# (b) the C++ server that actually implements `backends install`
cd /mnt/data_450g/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
```

> **Do not use the `.deb`.** `lemonade-server_11.7.0-debian13_amd64.deb` is the only Linux
> package offered and it is built for **Debian 13**. Extracted on Ubuntu 24.04 it dies at
> load time:
>
> ```text
> ./cxx/usr/bin/lemonade: error while loading shared libraries:
>   libmbedcrypto.so.16: cannot open shared object file: No such file or directory
> # also missing: libcpp-httplib.so.0.41
> ```
>
> The **`lemonade-embeddable-…-ubuntu-x64.tar.gz`** artifact has no unresolved
> dependencies on this host (`ldd` clean) and is what the rest of this README uses.

### Shared venv

One venv serves all three `inference_*_lemonade` folders; the other two are symlinks:

```bash
ln -sfn ../../../inference/lemonade/llm/.env_inference_llm_lemonade .env_inference_embedding_lemonade
```

This is deliberate (a sibling folder set does the same) — the python side here is only the
HTTP client, and the actual runtime is the C++ server plus its own managed ROCm venv under
the Lemonade cache.

### Backend install — and what it chose

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

```text
Installing backend: llamacpp:rocm
[1/2] llama-b10470-bin-ubuntu-rocm-7.14-x64.tar.gz
[2/2] rocm-7.14.0.tar.gz
      rocm_sdk_core-7.14.0-py3-none-linux_x86_64.whl        (414.6 MB)
      rocm_sdk_libraries-7.14.0-py3-none-linux_x86_64.whl   (557.6 MB)
      rocm_sdk_device_gfx950-7.14.0-py3-none-linux_x86_64.whl (1240.1 MB)
Backend installed successfully: llamacpp:rocm      # real 0m27.3s
```

**`rocm_sdk_device_gfx950`** is the headline: Lemonade probed the GPU, resolved it to
`gfx950`, and fetched the matching TheRock device wheel. See "Hardware support" below for
the ISA proof.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed for this model — `ggml-org/embeddinggemma-300M-GGUF` is ungated and was pulled
with no `HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5   # this folder's GPUs
export HF_HOME=/mnt/data_1.5t/hf_cache                    # where model weights land
```

- **`HF_HOME` controls the weights.** Lemonade downloads into `$HF_HOME/hub/` in an
  HF-hub-shaped tree (`snapshots/<sha>/<file>` + a `.lemonade_registry.json`).
- **The `cache_dir` positional controls the binaries.** `/mnt/data_450g/lemonade/cache`
  holds the llama.cpp build and the ROCm wheels — **4.8 GB**. Neither may land on `/`
  (99 GB free).
- These env vars must be set **on the `lemond` process**, not on the CLI client. `lemond`
  is what forks `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=/mnt/data_1.5t/hf_cache

$LEM/lemond /mnt/data_450g/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register EmbeddingGemma as an embedding model

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.EmbeddingGemma-300M \
  --checkpoint main ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --recipe llamacpp \
  --label embeddings
```

`--label embeddings` is the whole trick — it is what makes Lemonade start the wrapped
server with `--embeddings` and route `/v1/embeddings` to it.

### 3. Load and call it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.EmbeddingGemma-300M

.env_inference_embedding_lemonade/bin/python inference_embedding_lemonade.py \
  --port 8350 \
  --out /mnt/data_450g/outputs/inference_embedding_lemonade/embeddings_two_gpu.json
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

## Results — single GPU (physical GPU 4)

Server restarted with `HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4`:

```text
endpoint   : http://127.0.0.1:8350/v1/embeddings
model      : EmbeddingGemma-300M
latency_s  : 0.095
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2931
unrelated (0 vs 1) : 0.1026
```

GPU-residency proof for this run — one GPU, and the KFD accounting names it:

```text
PID      PROCESS NAME    GPU(s)  VRAM USED     SDMA USED       CU OCCUPANCY
1961159  llama-server    1       1801560064    1271208351006   0

/sys/class/kfd/kfd/proc/1961159/vram_51023 = 1801560064   -> gpu_id 51023 = card4
```

**1.80 GB on exactly one card**, and every other `vram_*` entry for that PID is `0`.

## Results — two GPUs visible (physical GPUs 4 and 5)

Real client output:

```text
endpoint   : http://127.0.0.1:8350/v1/embeddings
model      : EmbeddingGemma-300M
latency_s  : 0.103
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2931
unrelated (0 vs 1) : 0.1026
```

768-dimensional, L2-normalised (`norm = 1.0000`), and **semantically correct**: the two
OpenTelemetry sentences score **0.2931** against each other while the sourdough sentence
scores **0.1026** — the related pair is ~2.9x closer.

**Splitting a 300M model over two GPUs is pointless and this README will not pretend
otherwise.** What actually happens is subtler and worth stating precisely: llama.cpp's
default `--split-mode layer` puts *some* layers on each visible device, so with
`HIP_VISIBLE_DEVICES=4,5` the 312 MiB model really is spread across both cards — you can
see it in the per-GPU VRAM below — but that buys nothing except an extra device-to-device
hop per forward pass. **The honest scale-out pattern for a 300M embedder is one Lemonade
Server per GPU behind a load balancer**, which is exactly what the sibling
`inference/llamacpp/embedding` folder demonstrates. Lemonade offers no per-model
`--tensor-split` / `-ngl` knob to do anything smarter.

## GPU-residency proof

This is the load-bearing section: a CPU fallback would still return correct-looking
vectors. Lemonade launches `llama-server` **without an explicit `-ngl`**, so residency has
to be proven, not assumed.

The wrapped process (Lemonade proxies 8350 -> 8001):

```text
/mnt/data_450g/lemonade/cache/bin/llamacpp/rocm-stable/llama-b10469/llama-server \
  -m /mnt/data_1.5t/hf_cache/hub/models--ggml-org--embeddinggemma-300M-GGUF/snapshots/0f741b5a.../embeddinggemma-300M-Q8_0.gguf \
  --ctx-size 8192 --port 8001 --jinja --metrics --reasoning-format auto --no-ui --embeddings
```

`rocm-smi --showpids` — the kernel's own KFD accounting, which a CPU-only process cannot
appear in at all:

```text
PID      PROCESS NAME    GPU(s)  VRAM USED     SDMA USED       CU OCCUPANCY
1877700  llama-server    2       3123228672    2525538537143   0
```

Per-GPU breakdown straight from `/sys/class/kfd/kfd/proc/1877700/`:

```text
vram_51023 = 1361514496 bytes   -> gpu_id 51023 = unique_id 0xf743d583ac01fcfd = card4
vram_17306 = 1577820160 bytes   -> gpu_id 17306 = unique_id 0x38d73d00d7b9cb71 = card5
```

and 12 live HSA compute queues split across those two `gpu_id`s
(`/sys/class/kfd/kfd/proc/1877700/queues/*/gpuid`). `rocm-smi` agrees at the card level:

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card4,309220868096,1668128768
card5,309220868096,2051985408
```

## Cross-check vs the Transformers baseline

Two independent comparisons:

1. **Against the `inference/llamacpp/embedding` sibling** (same GGUF, same texts, but a
   locally compiled `-DGPU_TARGETS=gfx950` binary instead of Lemonade's prebuilt one):
   `head[0]` is **bit-identical** — `[-0.11552, 0.04329, 0.01613, -0.00535, -0.02826,
   0.01995, 0.04892, 0.06414]` — and both cosines match to 4 dp (0.2931 / 0.1026). The
   Lemonade-managed backend is numerically the same engine.
2. **Against `inference/transformers/embedding`** (bf16 `google/embeddinggemma-300m` via
   sentence-transformers), reference vectors at
   `/mnt/data_1.5t/outputs/inference_embedding_transformers/`:

The baseline encodes with `SentenceTransformer.encode_query` / `.encode_document`, which
prepend EmbeddingGemma's task prompts. Lemonade/llama.cpp does **not** apply them, so the
client must send the prefixed strings (`task: search result | query: …` and
`title: none | text: …`) for the comparison to mean anything:

```text
query x document cosine matrix
  q0 lemonade  : [0.5727, 0.5317, 0.097, -0.0126]
  q0 baseline  : [0.5763, 0.5371, 0.0965, -0.0107]
  q0 max |diff|: 0.0053
  q1 lemonade  : [0.2012, 0.1924, 0.502, 0.0827]
  q1 baseline  : [0.202, 0.1898, 0.5008, 0.0836]
  q1 max |diff|: 0.0026
raw query-vector agreement (first 16 dims, baseline stores only 16):
  cosine(lemonade q0[:16], transformers q0[:16]) = 0.999364
  cosine(lemonade q1[:16], transformers q1[:16]) = 0.999780
ranking agreement: [[0, 1, 2, 3], [2, 0, 1, 3]] vs [[0, 1, 2, 3], [2, 0, 1, 3]]
```

The client does this for you — `--reference` with no `--texts` adopts the baseline's own
queries and applies the prompt template automatically:

```bash
.env_inference_embedding_lemonade/bin/python inference_embedding_lemonade.py \
  --reference /mnt/data_1.5t/outputs/inference_embedding_transformers/reference_embedding_1gpu.json
```

```text
n_vectors  : 2
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.06729, -0.04331, -0.00479, -0.02985, 0.05449, -0.00028, -0.01817, 0.01222]
--- agreement vs reference baseline ---
cosine(mine[0][:16], ref[0][:16]) : 0.999629
cosine(mine[1][:16], ref[1][:16]) : 0.999708
```

(baseline `head[0]` for the same query: `[-0.06690, -0.04273, -0.00369, -0.03149,
0.05420, 0.00036, -0.01746, 0.01172]`.)

**Cosine agreement vs the Transformers baseline: 0.9994-0.9997.** Document
rankings are **identical** for both queries, and the whole 2x4 similarity matrix agrees
to within **0.0053** — consistent with Q8_0 quantization drift against a bf16 reference,
not with a different model or wrong pooling.

## Arguments

### Lemonade Server / CLI

| Argument | Used | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `/mnt/data_450g/lemonade/cache` | Binaries + backend venv live here. Keep off `/` |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs looking for beacons |
| `backends install llamacpp:rocm` | used | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family for the model |
| `pull --label` | `embeddings` | **Required** — drives `--embeddings` on the wrapped server |
| `load <name>` | used | Start/warm the subprocess before timing |

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

Artifacts go to `/mnt/data_450g/outputs/inference_embedding_lemonade/`, never to `/`:

```text
embeddings_two_gpu.json        raw /v1/embeddings response (3x768 floats)
client_two_gpu.txt             client stdout, GPUs 4+5
embeddings_single_gpu.json     raw response, GPU 4 only
client_single_gpu.txt          client stdout, GPU 4 only
rocm_smi_two_gpu.csv           per-card VRAM
rocm_smi_pids_two_gpu.txt      KFD process attribution
```

Weights (319 MB) live in `/mnt/data_1.5t/hf_cache/hub/`; Lemonade binaries + ROCm wheels
(4.8 GB) in `/mnt/data_450g/lemonade/cache/`.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| Lemonade detects gfx950 | `backends --all` prints `Unsupported GPU: gfx950` for other recipes -> it resolved the arch |
| `llamacpp:rocm` is supported on gfx950 | Listed `installable`, then `installed b10470` |
| Backend really carries gfx950 code | Install pulled `rocm_sdk_device_gfx950-7.14.0-…whl` (1240 MB); `libggml-hip.so` (1.26 GB fat binary) contains a `gfx950` code object alongside gfx900/906/908/942/10xx/11xx/12xx |
| No `hipErrorNoBinaryForGpu` | Model loaded and ran; zero HSA/ISA errors in the server log |
| Model really on GPU | `rocm-smi --showpids`: PID 1877700, **2 GPUs, 3.12 GB VRAM**, live KFD queues |
| Embeddings are real | 3 vectors x 768 dims, L2 norm 1.0000 |
| Embeddings are *meaningful* | related 0.2931 > unrelated 0.1026 |
| Same numbers as a hand-built gfx950 llama.cpp | `head[0]` bit-identical to `inference/llamacpp/embedding` |
| Ungated download | Pulled with no `HF_TOKEN` exported |

## Notes & quirks

1. **Two different products share the name.** pip `lemonade-sdk` != the C++ Lemonade
   Server. Only the latter has `backends install`. See the table in "Install".
2. **The Linux `.deb` is Debian-13-only** and will not load on Ubuntu 24.04
   (`libmbedcrypto.so.16`, `libcpp-httplib.so.0.41` missing). Use the `embeddable`
   tarball.
3. **`--no-discovery` on every CLI call.** Without it the client broadcasts UDP looking
   for servers and hangs well past 60 s on this multi-tenant host.
4. **The `user.` prefix is registration-only.** `/v1/models` reports
   `EmbeddingGemma-300M`; use that as `"model"`.
5. **No `-ngl` is exposed.** Lemonade builds the `llama-server` command line itself and
   does not pass `-ngl`; offload happens because the ROCm backend claims the layers.
   Prove residency via `rocm-smi --showpids`, not via server flags.
6. **You can pre-seed the HF cache to skip a re-download.** Mirroring an existing
   `snapshots/<sha>/<file>` tree into `$HF_HOME/hub/<repo>/` with a matching
   `refs/main` and `.lemonade_registry.json` makes `pull` report `(already downloaded)`.
   Used here to avoid re-fetching the 28 GB LLM GGUF — see the LLM folder.
7. **Embeddings are NOT reproducible across a live server's request history.** This is the
   most important quirk in this folder and it is easy to miss. The same request, to the
   same PID, returns different vectors depending on what was asked *before* it. Isolated
   and reproduced:

   ```text
   unload + load (clean slot)
   A  3-text request  -> head[0] = [-0.11552, …]   related 0.2931
   B  3-text request  -> head[0] = [-0.11552, …]   stable while batch shape is constant
   C  send an unrelated 2-text request
   D  3-text request  -> head[0] = [-0.11467, …]   <-- changed
   E  send an unrelated 1-text request
   F  3-text request  -> head[0] = [-0.11529, …]   <-- changed again
   ```

   The drift is ~8.5e-4 per component. It comes from `llama-server`'s slot/KV state and
   batch packing, not from Lemonade, and it is **semantically harmless**: `related` stays
   ~2.8x `unrelated` at every step and baseline agreement stays 0.9994-0.9997. But it
   means **identical text indexed at different times yields slightly different vectors**.
   If you need byte-reproducible embeddings, batch a corpus in fixed-shape requests, or
   accept ~1e-3 of noise in your index. `--load_model` also perturbs the slot, which is
   why the reference runs above do not use it.
8. **The Lemonade cache is not small.** 4.8 GB for one backend (the gfx950 ROCm device
   wheel alone is 1.2 GB). Point `lemond` at a big filesystem.
9. **Missing-resource warnings at startup are harmless** for the embeddable build:
   `Could not load architecture_defaults.json`, `Web app directory not found`. Only the
   web UI is affected; the API is complete.

## H100 (NVIDIA, Hopper sm_90) — verified 2026-08-22

Single-GPU smoke on **physical GPU 6** (`CUDA_VISIBLE_DEVICES=6`), driver 580.173.02,
host CUDA 13.0, Python 3.12.3. This is the **exact same GGUF** as the MI355X run
(`ggml-org/embeddinggemma-300M-GGUF:Q8_0`, 318 MB), so it doubles as a cross-hardware
numerical check.

**Install = the AMD route with `llamacpp:cuda` in place of `llamacpp:rocm`.** Same
embeddable tarball, same `backends install`; it pulled the arch-matched Hopper prebuilt
`llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` (build **b10397**) in **~41 s, no compiler**,
bundling its own CUDA 12.9 runtime. Lemonade detected `NVIDIA H100 80GB HBM3 (compute 9.0,
sm_90)` and logged `Using LlamaCpp Backend: cuda`. Registration is byte-for-byte the MI355X
command — **`--label embeddings` still does the whole trick** (see the cmdline below):

```bash
LEM=/dev/shm/h100/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models
$LEM/lemonade --port 8350 --no-discovery pull user.EmbeddingGemma-300M \
  --checkpoint main ggml-org/embeddinggemma-300M-GGUF:Q8_0 --recipe llamacpp --label embeddings
$LEM/lemonade --port 8350 --no-discovery load user.EmbeddingGemma-300M
```

**Exact smoke command + real output:**

```bash
.env_lemonade/bin/python inference_embedding_lemonade.py --port 8350 \
  --model EmbeddingGemma-300M --out /dev/shm/h100/out/lemonade/embeddings_single_gpu.json
```

```text
endpoint   : http://127.0.0.1:8350/v1/embeddings
model      : EmbeddingGemma-300M
latency_s  : 0.135
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11443, 0.04268, 0.01675, -0.00695, -0.0285, 0.01905, 0.04997, 0.06326]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2924
unrelated (0 vs 1) : 0.1038
```

**768-dim, L2-normalised, semantically correct** (related 0.2924 ≈ 2.8× unrelated 0.1038).
Cross-hardware sanity: the MI355X README's `head[0]` for the same request was
`[-0.11552, 0.04329, 0.01613, …]`; H100 gives `[-0.11443, 0.04268, 0.01675, …]` — agreeing
to ~1e-3 per component, consistent with CUDA-vs-ROCm FP accumulation order on the same Q8_0
GGUF (and with the ~8.5e-4 slot/batch drift this folder already documents in "Notes"). Same
engine, same model, different vendor math.

**Wrapped cmdline (`--embeddings` auto-appended, `--ctx-size 8192` as on MI355X):**

```text
/dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--ggml-org--embeddinggemma-300M-GGUF/…/embeddinggemma-300M-Q8_0.gguf \
  --ctx-size 8192 --port 8002 --jinja --metrics --reasoning-format auto --no-ui --embeddings
```

**GPU-residency proof** (`nvidia-smi` VRAM-by-PID, GPU-6 UUID `GPU-e4fe48bc` = index 6):

```text
1725003, /dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server, 982 MiB, GPU-e4fe48bc-…
```

**0.98 GB on GPU 6 for our PID.** A CPU fallback (which would still return correct-looking
vectors) shows 0 MiB here — this is the load-bearing check and it passes.

**Multi-GPU (deferred):** pointless for a 300M embedder, same as MI355X — run one server
per GPU. Not launched (production on GPUs 0–3).

**Deviations from MI355X:** backend `llamacpp:cuda` (b10397) vs `llamacpp:rocm` (b10470);
client venv built in tmpfs (a `venv` on the `/mnt/gsma` NFS mount didn't create pip/console
scripts reliably — use `python -m pip`). Model, quant, client and API are otherwise
identical.

**H100 VERDICT: PASS.** Zero-build sm_90 CUDA backend in ~41 s, 768-dim L2-normalised
semantically-correct embeddings in ~135 ms, 0.98 GB resident on GPU 6, and vectors agree
with the MI355X ROCm run to ~1e-3. On NVIDIA, as on AMD, the accelerated path is
llama.cpp/GGUF (`llamacpp:cuda`); there is no vLLM/TEI-style engine inside Lemonade.

## VERDICT (MI355X)

**PASS — fully working on MI355X (gfx950), and the zero-build install is the real story.**

`lemonade backends install llamacpp:rocm` detected `gfx950` on its own, fetched an
arch-matched `rocm_sdk_device_gfx950` wheel and a prebuilt ROCm `llama.cpp` in **27
seconds** with no compiler, no `cmake`, no `libssl-dev`. The concern that a prebuilt ROCm
backend would ship only gfx90a/gfx942 and die with `hipErrorNoBinaryForGpu` **did not
materialise** — the fat `libggml-hip.so` carries gfx950.

The service returns **768-dim, L2-normalised, semantically correct** embeddings in
**~100 ms** for a 3-text batch, with **3.12 GB of VRAM and live KFD queues on GPUs 4 and
5** proving GPU residency, and vectors **bit-identical** to a hand-compiled
`-DGPU_TARGETS=gfx950` llama.cpp.

Multi-GPU here is incidental, not a win: llama.cpp's default layer split spreads a 312 MiB
model over both visible cards for no benefit. Run **one server per GPU** instead.

## Follow-ups

- Measure sustained throughput; ~100 ms for one small request understates capacity badly.
- Lemonade fixes `--ctx-size 8192` for this model and exposes no `-b`/`-ub`; check whether
  the llama.cpp embedding-mode batch clamp (`n_batch = n_ubatch = 512`) caps large batches.
- Compare with `inference/tei/embedding` on the same GPU for a throughput-per-watt number.
- Watch for a Ubuntu-24.04 `.deb`; the `embeddable` tarball has no systemd unit.
