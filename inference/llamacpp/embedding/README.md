# `inference/llamacpp/embedding` — llama.cpp GGUF embedding serving (EmbeddingGemma-300M)

## Overview & when to use

Serves **EmbeddingGemma-300M** as GGUF through `llama-server` with `--embeddings`,
built against **ROCm 7.2 / HIP** (AMD Instinct MI355X, gfx950) or **CUDA 13** (NVIDIA
H100). The client
`inference_embedding_llamacpp.py` hits `POST /v1/embeddings`, the OpenAI-compatible
route, and verifies the vectors are semantically meaningful rather than merely
well-shaped.

Use this folder for a tiny always-on embedding service with no python runtime in the serving
path, from the **same binary** as the LLM and reranker leaves (one toolchain for all three).
Prefer HF TEI or vLLM if you need native `sentence-transformers` pooling semantics or
server-side Matryoshka truncation.

> **Format note:** llama.cpp needs **GGUF**, not the HF safetensors. This folder serves
> `ggml-org/embeddinggemma-300M-GGUF:Q8_0`, the converted artifact. The canonical
> `google/embeddinggemma-300m` repo is gated and requires accepting Google's terms;
> **the ggml-org GGUF mirror is not gated and needs no `HF_TOKEN`.**

## Build — see [`../README.md`](../README.md)

One llama.cpp HIP build serves all three `inference/llamacpp/*` leaves — the full
ROCm build recipe (prerequisites, cmake flags, the mandatory `-DLLAMA_OPENSSL=ON`,
NVIDIA variant) lives in [`../README.md`](../README.md).

Reference revision for the ROCm recipe: commit
**`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`**, `llama-server` **0.1.2-dev (build 1)**,
ggml **0.20.2**.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                 # already done in this folder
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

Not required for this model — the GGUF mirror is ungated. Never echo or commit
`HF_TOKEN`.

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache             # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp        # llama.cpp's own -hf cache
export OUTPUT_DIR=/path/to/outputs           # inference artifacts

export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0      # the GPU this server uses
```

`LLAMA_CACHE` is what actually controls where `-hf` writes; point it at a filesystem
with room rather than the root filesystem. This model is 319 MB on disk.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Serve

### Single GPU

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
# HF_HOME / LLAMA_CACHE as exported above

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings \
  -ngl 99 \
  -lv 5 \
  --host 127.0.0.1 --port 8201 \
  --alias embeddinggemma-300m
```

`--embeddings` is required — without it `/v1/embeddings` is not served.
The first run downloads 319 MB; cached restarts take under a second.

### Two concurrent instances, one per GPU (the production pattern)

```bash
# instance A — first GPU, port 8211
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 --embeddings -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8211 --alias embeddinggemma-300m-gpu0 &

# instance B — second GPU, port 8201
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 --embeddings -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8201 --alias embeddinggemma-300m &
```

## Client / smoke command

One shared venv at the software root serves all three leaves — build it from
`../requirements.txt`:

```bash
cd .. && python3 -m venv .env_llamacpp && .env_llamacpp/bin/pip install -r requirements.txt && cd embedding

../.env_llamacpp/bin/python inference_embedding_llamacpp.py \
  --port 8201 \
  --out $OUTPUT_DIR/inference_embedding_llamacpp/embeddings_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8201/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":["a span is a unit of work in a trace"]}' \
  | jq '.data[0].embedding | length'
```

## Expected output

```text
endpoint   : http://127.0.0.1:8201/v1/embeddings
model      : embeddinggemma-300m
n_vectors  : 3
dim        : 768
norm[0]    : 1.0000
head[0]    : [-0.11552, 0.04329, 0.01613, -0.00535, -0.02826, 0.01995, 0.04892, 0.06414]
--- cosine similarity (semantic sanity check) ---
related   (0 vs 2) : 0.2931
unrelated (0 vs 1) : 0.1026
```

The vectors must be **768-dimensional** (EmbeddingGemma's `n_embd`) and **L2-normalised**
(`norm = 1.0000`, i.e. the server applies the model's pooling and normalisation), with the
related pair scoring above the unrelated one.

**GPU-residency check** — `-ngl 99` can silently fall back to CPU, so check the
offload summary at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:dc:00.0) - 294310 MiB free
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
```

Confirm with `rocm-smi` while only this server is running, and with `rocm-smi --showpids`,
which attributes that memory to the `llama-server` PID. All 25 layers are on the GPU; the
`CPU_Mapped` buffer is the token-embedding table (EmbeddingGemma has a 262 k vocabulary),
which llama.cpp keeps host-side by design.

## Multi-GPU

**Do not split a 300M model across two GPUs** — a layer split only adds a device-to-device
transfer per forward pass, and `--split-mode row` is unavailable on this backend anyway (see
the LLM leaf).

The pattern for a tiny model is **one independent server instance per GPU**, load-balanced
(commands above). Concurrent instances produce identical vectors, so a load balancer can
route to either without changing results.

## H100 (NVIDIA, CUDA)

Mirror of the MI355X setup on **NVIDIA H100 80GB HBM3**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12. Same binary, same **exact model** (`ggml-org/embeddinggemma-300M-GGUF:Q8_0`
— it is only 319 MB, so no substitution is needed), same client, same `/v1/embeddings`
path. Only the backend build flag changes: `-DGGML_CUDA=ON` instead of `-DGGML_HIP=ON`
(full recipe in [`../README.md`](../README.md); `-DLLAMA_OPENSSL=ON` is kept — it is
vendor-neutral and required for the `-hf` HTTPS pull). Reference revision: `llama-server`
**0.2.0-dev (build 1, commit `70adb1b`)**, ggml **0.21.0**.

```bash
cd /dev/shm/llamacpp/llama.cpp                   # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=0
# HF_HOME as exported above
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, see quirks below

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings -ngl 999 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias embeddinggemma-300m
```

Client (served here on port 8700):

```bash
../.env_llamacpp/bin/python inference_embedding_llamacpp.py --port 8700 --model embeddinggemma-300m
```

**GPU-residency check** (`-ngl` can silently fall back to CPU). Startup log at `-lv 5`:

```text
llama_prepare_model_devices: using device CUDA0 (NVIDIA H100 80GB HBM3) (000c:00:00.0) - 80552 MiB free
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        CUDA0 model buffer size =   311.97 MiB
```

All **25/25 layers on the GPU**, with the same `CUDA0` buffer size as the MI355X `ROCm0` one
(same GGUF). The `CPU_Mapped` buffer is the 262 k-vocab embedding table, kept host-side by
design. Confirm with
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`, which must
attribute the memory to the `llama-server` PID on the selected card.

**Meaningful embeddings** — client output:

```text
n_vectors  : 3      dim : 768      norm[0] : 1.0000
related   (0 vs 2) : 0.2924
unrelated (0 vs 1) : 0.1038
```

**768-dimensional, L2-normalised**, related above unrelated — the same result as MI355X.

**Quirks:** the `n_batch (2048) > n_ubatch (512)` clamp warning appears exactly as on
MI355X. This GGUF is ungated and pulls with no `HF_TOKEN`. If the host has no
cmake/ninja, install them into a throwaway venv (see `../README.md`) and pull the
weights to a tmpfs `LLAMA_CACHE`.

**Multi-GPU:** the commands above are single-GPU. As on MI355X, do **not** split this model —
run one instance per GPU behind a load balancer; the CUDA build supports the same
two-instance pattern.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Value | Meaning |
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

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_embedding_llamacpp/`; redirect the server log there too. Model
weights (319 MB) live in `$LLAMA_CACHE/`, never on the root filesystem.

## Hardware support

- **AMD Instinct MI355X (gfx950, ROCm 7.2):** HIP build works with no source patches;
  scale out with one instance per GPU.
- **NVIDIA H100 (Hopper cc 9.0, CUDA 13):** CUDA build works with no source patches.
- The GGUF mirror is ungated: it pulls with no `HF_TOKEN` exported.

## Notes & quirks

1. **`--embeddings` is mandatory.** Without it the server starts and
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
   "Multi-GPU".
8. **Matryoshka truncation is client-side.** llama.cpp returns full-width 768-d vectors;
   truncating to 512/256/128 has to happen in your code.
