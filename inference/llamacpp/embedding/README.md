# `inference/llamacpp/embedding` — llama.cpp GGUF embedding serving (EmbeddingGemma-300M) on ROCm/HIP

## Overview & when to use

Serves **EmbeddingGemma-300M** as GGUF through `llama-server` with `--embeddings`,
built against **ROCm 7.2.4 / HIP** for **AMD Instinct MI355X (gfx950)**. The client
`inference_embedding_llamacpp.py` hits `POST /v1/embeddings`, the OpenAI-compatible
route, and verifies the vectors are semantically meaningful rather than merely
well-shaped.

Use this folder when you want:

- **A tiny, always-on embedding service** with no python runtime in the serving path —
  312 MiB of model on the GPU, ~50 ms for a 3-text batch.
- **The same binary as your LLM and reranker** — one llama.cpp build covers all three
  workloads, so there is one toolchain to maintain instead of three.
- **CPU or CPU+GPU deployment** — this model is small enough to run acceptably on CPU,
  so llama.cpp lets the same artifact serve laptops and datacenter GPUs.

Prefer **HF Text Embeddings Inference (TEI)** or **vLLM** if you need maximum embedding
throughput, native `sentence-transformers` pooling semantics, or Matryoshka truncation
handled server-side.

> **Format note:** llama.cpp needs **GGUF**, not the HF safetensors. This folder serves
> `ggml-org/embeddinggemma-300M-GGUF:Q8_0`, the converted artifact. The canonical
> `google/embeddinggemma-300m` repo is gated and requires accepting Google's terms;
> **the ggml-org GGUF mirror is not gated and downloaded here with no `HF_TOKEN`.**

> **Tested topology:** 2xAMD Instinct MI355X (gfx950, 288 GB each), physical GPUs 6
> and 7, ROCm 7.2.4, Ubuntu, Python 3.12.3. Verified **2026-08-20**.

## Build — see [`../README.md`](../README.md)

One llama.cpp HIP build serves all three `inference/llamacpp/*` leaves — the full
ROCm build recipe (prerequisites, cmake flags, the mandatory `-DLLAMA_OPENSSL=ON`,
NVIDIA variant) lives in [`../README.md`](../README.md). The campaign's build tree
was removed with the venvs in the 2026-08 reorg; the documented 40.6 s cold rebuild
applies.

Verified at commit **`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`** (Wed Aug 19 2026),
`llama-server` **0.1.2-dev (build 1)**, ggml **0.20.2**.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                 # already done in this folder
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

Not required for this model — the GGUF mirror is ungated. Never echo or commit
`HF_TOKEN`.

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7      # this folder's test GPU
export HF_HOME=/mnt/data_1.5t/hf_cache
export LLAMA_CACHE=/mnt/data_1.5t/hf_cache/llama_cpp     # llama.cpp's own -hf cache
```

`LLAMA_CACHE` is what actually controls where `-hf` writes; weights must land on
`/mnt`. This model is 319 MB on disk.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Serve

### Single GPU (physical GPU 7)

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7
export HF_HOME=/mnt/data_1.5t/hf_cache LLAMA_CACHE=/mnt/data_1.5t/hf_cache/llama_cpp

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings \
  -ngl 99 \
  -lv 5 \
  --host 127.0.0.1 --port 8201 \
  --alias embeddinggemma-300m
```

`--embeddings` is required — without it `/v1/embeddings` is not served.
First run downloads 319 MB and is ready in ~16 s; cached restarts take under 1 s.

### Two concurrent instances, one per GPU (the production pattern)

```bash
# instance A — physical GPU 6, port 8211
HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 --embeddings -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8211 --alias embeddinggemma-300m-gpu6 &

# instance B — physical GPU 7, port 8201
HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 ./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 --embeddings -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8201 --alias embeddinggemma-300m &
```

## Client / smoke command

One shared venv at the software root serves all three leaves (the per-leaf campaign
venvs were removed in the 2026-08 reorg; rebuild from `../requirements.txt`):

```bash
cd .. && python3 -m venv .env_llamacpp && .env_llamacpp/bin/pip install -r requirements.txt && cd embedding

../.env_llamacpp/bin/python inference_embedding_llamacpp.py \
  --port 8201 \
  --out /mnt/data_1.5t/outputs/inference_embedding_llamacpp/embeddings_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8201/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":["a span is a unit of work in a trace"]}' \
  | jq '.data[0].embedding | length'
```

## Results — single GPU

Real client output:

```text
endpoint   : http://127.0.0.1:8201/v1/embeddings
model      : embeddinggemma-300m
latency_s  : 0.050
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2931
unrelated (0 vs 1) : 0.1026
```

The vectors are **768-dimensional** (matching EmbeddingGemma's `n_embd = 768`) and
**L2-normalised** (`norm = 1.0000`, i.e. the server applies the model's pooling and
normalisation). Crucially the geometry is *correct*, not just well-shaped: the two
OpenTelemetry sentences score **0.2931** against each other while the sourdough
sentence scores **0.1026** — the related pair is ~2.9x closer.

**GPU-residency proof** — `-ngl 99` can silently fall back to CPU, so here is the
offload summary at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:dc:00.0) - 294310 MiB free
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
```

Confirmed by `rocm-smi` while only this server was running on GPU 7:

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card6,309220868096,297955328      <-- 0.30 GB idle baseline
card7,309220868096,1918697472     <-- 1.92 GB, this server
```

`rocm-smi --showpids` attributed 1,620,262,912 B to the `llama-server` PID. All 25
layers are on the GPU; the 204 MiB `CPU_Mapped` buffer is the token-embedding table
(EmbeddingGemma has a 262 k vocabulary), which llama.cpp keeps host-side by design.

## Results — multi-GPU

**Splitting a 300M model across two GPUs is pointless, and this README will not pretend
otherwise.** The model is 312 MiB of GPU buffer on a card with 288 GB of VRAM —
0.1 % occupancy. A layer split would put ~12 layers on each GPU and add a
device-to-device transfer to every single forward pass, making a ~50 ms request
*slower* for no capacity benefit. `--split-mode row` is not available on this backend
either (see the LLM folder's README for the root-cause trace).

The honest production pattern for a tiny model is **one independent server instance per
GPU**, load-balanced — which scales linearly and shares nothing. Demonstrated with two
concurrent instances:

```text
### instance A (GPU 6, port 8211)
endpoint   : http://127.0.0.1:8211/v1/embeddings
latency_s  : 0.053
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]

### instance B (GPU 7, port 8201)
endpoint   : http://127.0.0.1:8201/v1/embeddings
latency_s  : 0.009
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
```

The two GPUs produce **bit-identical embedding vectors**, so a load balancer can route
requests to either without changing results. Both instances offloaded 25/25 layers with
`ROCm0 model buffer size = 311.97 MiB`, and both cards show live model residency:

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card6,309220868096,9601277952     <-- 9.60 GB
card7,309220868096,9601253376     <-- 9.60 GB
```

(Each card is running this folder's embedding server *and* the reranker folder's
server; the per-card total is dominated by the KV-cache and HIP context allocation
rather than the 312 MiB of weights.)

## H100 (NVIDIA, CUDA) — verified 2026-08-22

Mirror of the MI355X run above, on **1x NVIDIA H100 80GB HBM3** (physical GPU 7,
`CUDA_VISIBLE_DEVICES=7`), driver **580.173.02**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12.3. Same binary, same **exact model** (`ggml-org/embeddinggemma-300M-GGUF:Q8_0`
— it is only 319 MB, so no substitution was needed), same client, same `/v1/embeddings`
path. Only the backend build flag changed: `-DGGML_CUDA=ON` instead of `-DGGML_HIP=ON`
(full recipe in [`../README.md`](../README.md); `-DLLAMA_OPENSSL=ON` is kept — it is
vendor-neutral and required for the `-hf` HTTPS pull). Verified `llama-server`
**0.2.0-dev (build 1, commit `70adb1b`)**, ggml **0.21.0**.

```bash
cd /dev/shm/h100/out/llamacpp/llama.cpp          # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=7
export HF_HOME=/mnt/gsma/gsma/gsma/models
export LLAMA_CACHE=/dev/shm/h100/out/llamacpp/model_cache

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings -ngl 999 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias embeddinggemma-300m
```

Client (served here on port 8700):

```bash
../.env_llamacpp/bin/python inference_embedding_llamacpp.py --port 8700 --model embeddinggemma-300m
```

**GPU-residency proof** (`-ngl` can silently fall back to CPU). Startup log at `-lv 5`:

```text
llama_prepare_model_devices: using device CUDA0 (NVIDIA H100 80GB HBM3) (000c:00:00.0) - 80552 MiB free
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        CUDA0 model buffer size =   311.97 MiB
```

All **25/25 layers on the GPU**; the **CUDA0 buffer is 311.97 MiB — bit-identical to the
MI355X ROCm0 buffer** (same GGUF). The 204 MiB `CPU_Mapped` buffer is the 262 k-vocab
embedding table, kept host-side by design. Confirmed by `nvidia-smi` filtered to GPU 7,
attributed by PID:

```text
$ nvidia-smi -i 7 --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
1702586, ./build/bin/llama-server, 912 MiB
```

**Real, meaningful embeddings** — client output:

```text
n_vectors  : 3      dim : 768      norm[0] : 1.0000      latency_s : 0.174
related   (0 vs 2) : 0.2924
unrelated (0 vs 1) : 0.1038
```

**768-dimensional, L2-normalised**, and geometrically correct: the two OpenTelemetry
sentences score **0.2924** vs **0.1038** for the sourdough distractor (~2.8x closer) —
matching MI355X (0.2931 / 0.1026) to two decimals, confirming pooling+normalisation
behave identically on CUDA.

**Quirks:** the `n_batch (2048) > n_ubatch (512)` clamp warning appears exactly as on
MI355X. New for this box: no GGUF was cached (this one is ungated, pulled with no
`HF_TOKEN`), and cmake/ninja were absent — installed into a throwaway venv (see
`../README.md`); weights pulled to a tmpfs `LLAMA_CACHE`.

**Multi-GPU (deferred):** only GPU 7 was used (GPUs 0–3 were a co-tenant production job).
As on MI355X, do **not** split a 312 MiB model — run one instance per GPU behind a load
balancer; the CUDA build supports the same two-instance pattern.

**H100 verdict: PASS.** All 25/25 layers on the H100 (CUDA0 buffer + `nvidia-smi` by
PID), 768-d L2-normalised vectors with correct semantic geometry, ~0.17 s for a 3-text
batch.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Used | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Pull GGUF from the Hub. Needs an SSL-enabled build |
| `--embeddings` | on | **Required** to expose `/v1/embeddings` |
| `-ngl N` | `99` | Layers offloaded to GPU. 99 = "all"; verify via the offload summary |
| `--host` / `--port` | `127.0.0.1` / `8201` | Bind address |
| `--alias` | `embeddinggemma-300m` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity. **Default 3 hides the offload summary** |
| `--pooling` | model default (`u32 = 1`, mean) | Override pooling if you need CLS/last |
| `-ub N` | 512 (auto-clamped) | Micro-batch; see quirk 3 |

### `inference_embedding_llamacpp.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8201` | llama-server port |
| `--model` | `embeddinggemma-300m` | Model name echoed in the request body |
| `--texts` | 3-sentence probe | Texts to embed (2 related + 1 unrelated) |
| `--timeout` | `300` | HTTP timeout (s) |
| `--health_retries` | `60` | `/health` polls before giving up |
| `--out` | `None` | Write the raw JSON response here |

## Output

Artifacts go to `/mnt/data_1.5t/outputs/inference_embedding_llamacpp/`, never to `/`:

```text
server_single_gpu_verbose.log  offload summary + load trace, GPU 7
server_instance_gpu6.log       second concurrent instance, GPU 6
embeddings_single_gpu.json     raw /v1/embeddings response (3x768 floats)
client_single_gpu.txt          client stdout
rocm_smi_two_instances.csv     per-card VRAM with both instances live
```

Model weights (319 MB) live in `/mnt/data_1.5t/hf_cache/llama_cpp/`, never on `/`.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| HIP build works on gfx950 | 681/681 ninja targets, EXIT=0, no patches |
| GPUs detected | `--list-devices` → `ROCm0`/`ROCm1` MI355X, 294896 MiB each |
| Model really on GPU | `offloaded 25/25 layers to GPU`, `ROCm0 model buffer size = 311.97 MiB` |
| VRAM occupied | `rocm-smi`: card7 1.92 GB vs 0.30 GB idle baseline on card6 |
| Embeddings are real | 3 vectors x 768 dims, L2 norm 1.0000 |
| Embeddings are *meaningful* | related cosine 0.2931 > unrelated 0.1026 |
| Two-GPU scale-out works | Both cards loaded, bit-identical vectors from each |
| Ungated download | Pulled with no `HF_TOKEN` exported |

## Notes & quirks

1. **`--embeddings` is mandatory.** Without it the server starts happily and
   `/v1/embeddings` 404s.
2. **The standard build recipe is incomplete for `-hf`.** `cmake -B build -DGGML_HIP=ON`
   compiles but cannot download: this revision replaced libcurl with bundled
   cpp-httplib, which needs TLS. Symptom is
   `get_repo_commit: error: HTTPS is not supported`, followed by the misleading
   `failed to load model ''`. Fix: `apt install libssl-dev` + `-DLLAMA_OPENSSL=ON`.
3. **Batch size is silently clamped in embedding mode:**

   ```text
   srv llama_server: embeddings enabled with n_batch (2048) > n_ubatch (512)
   srv llama_server: setting n_batch = n_ubatch = 512 to avoid assertion failure
   ```

   If you are batching many texts per request, raise **both** with `-b` and `-ub`
   together, or the effective batch stays 512.
4. **Harmless tokenizer warning at load:**
   `control-looking token: 212 '</s>' was not control-type; this is probably a bug in
   the model. its type will be overridden`. It comes from the GGUF metadata and does
   not affect output — embeddings were verified correct.
5. **Default log verbosity hides the GPU-offload proof.** Use `-lv 5` when you need to
   prove residency, then drop it for production.
6. **`LLAMA_CACHE`, not `HF_HOME`, controls where `-hf` writes.**
7. **Do not split this model across GPUs.** Run one instance per GPU instead — see
   "Results — multi-GPU".

## VERDICT

**PASS — fully working on MI355X (gfx950).**

The ROCm/HIP build serves `ggml-org/embeddinggemma-300M-GGUF:Q8_0` with **all 25/25
layers on the GPU** (311.97 MiB ROCm0 buffer, 1.92 GB VRAM in `rocm-smi`), returning
**768-dimensional, L2-normalised** embeddings in **~50 ms** for a 3-text batch, with
**semantically correct** geometry (related 0.2931 vs unrelated 0.1026).

Multi-GPU splitting is **deliberately not used** — a 312 MiB model on a 288 GB card
gains nothing from a layer split and would only add transfer latency. The scale-out
pattern demonstrated instead is **two concurrent single-GPU instances**, which produced
**bit-identical vectors** on both GPUs.

The only real gotcha is the build-time one shared by all three folders:
**`-DLLAMA_OPENSSL=ON` + `libssl-dev`** are required for the `-hf` downloader.

## Follow-ups

- Measure sustained throughput (texts/s) with a concurrent client and larger `-b`/`-ub`;
  the ~50 ms figure here is a single small request and understates capacity.
- Compare against HF TEI on the same GPU — TEI is the native-embedding
  recommendation and should win on pure throughput.
- Validate Matryoshka truncation (768 → 512/256/128) if downstream storage matters;
  llama.cpp returns full-width vectors, so truncation must happen client-side.
- Cross-check a handful of vectors against `sentence-transformers` FP16 output to
  quantify Q8_0 quantization drift.
