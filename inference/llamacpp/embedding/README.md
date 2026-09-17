# `inference/llamacpp/embedding` — llama.cpp GGUF embedding serving (EmbeddingGemma-300M)

Serves `ggml-org/embeddinggemma-300M-GGUF:Q8_0` through `llama-server --embeddings` and hits
`POST /v1/embeddings` with `inference_embedding_llamacpp.py`, which checks the vectors are
semantically meaningful rather than merely well-shaped.

Pick this folder for a tiny always-on embedding service with no python runtime in the
serving path, from the same binary as the LLM and reranker leaves. Prefer
[`../../tei/embedding/`](../../tei/embedding/) or [`../../vllm/embedding/`](../../vllm/embedding/)
if you need native `sentence-transformers` pooling semantics or server-side Matryoshka
truncation. The canonical `google/embeddinggemma-300m` repo is gated; the ggml-org GGUF
mirror served here is not, and needs no `HF_TOKEN`.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2) and NVIDIA H100 (Hopper cc 9.0, CUDA
13), Python 3.12.

## Files

- `inference_embedding_llamacpp.py` — embedding smoke client; prints dimension, L2 norm and
  a related-vs-unrelated cosine comparison.

## Setup

Build `llama-server` and create the shared venv per [`../README.md`](../README.md) — one
build and one venv serve all three leaves. Then:

```bash
ln -sf ../../../dev.env dev.env                 # only needed for gated repos
export HF_HOME=/path/to/hf_cache                # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp           # llama.cpp's own -hf cache (319 MB for this model)
export OUTPUT_DIR=/path/to/outputs              # inference artifacts
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
```

## Run

### Serve — single GPU

```bash
cd <your llama.cpp checkout>
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings \
  -ngl 99 \
  -lv 5 \
  --host 127.0.0.1 --port 8201 \
  --alias embeddinggemma-300m
```

`--embeddings` is required or `/v1/embeddings` 404s. The first run downloads 319 MB; cached
restarts take under a second.

### Serve — one instance per GPU (the scale-out pattern)

Do not split a 300M model across GPUs — a layer split only adds a device-to-device transfer
per forward pass. Run independent instances behind a load balancer instead; concurrent
instances produce identical vectors.

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

### Client

```bash
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

### NVIDIA / CUDA

Identical serve and client commands and the identical model — at 319 MB no substitution is
needed. Only the build flag differs (`-DGGML_CUDA=ON`, see [`../README.md`](../README.md)).

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # for the HF pull
export CUDA_VISIBLE_DEVICES=0
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, if your model share rejects renames

./build/bin/llama-server \
  -hf ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --embeddings -ngl 999 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias embeddinggemma-300m

../.env_llamacpp/bin/python inference_embedding_llamacpp.py --port 8700 --model embeddinggemma-300m
```

## Output

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

Vectors must be **768-dimensional** (EmbeddingGemma's `n_embd`) and **L2-normalised**
(`norm = 1.0000`), with the related pair scoring above the unrelated one. The CUDA run gives
the same result (0.2924 vs 0.1038).

`-ngl 99` can fall back to CPU silently, so check the offload summary printed at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:dc:00.0) - 294310 MiB free
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
```

All 25 layers must be on the GPU; the `CPU_Mapped` buffer is the 262 k-vocab token-embedding
table, which llama.cpp keeps host-side by design. Confirm independently with
`rocm-smi --showpids` or
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

The client writes raw JSON wherever `--out` points; weights stay under `$LLAMA_CACHE`.

## Arguments

### `llama-server`

| Argument | Value | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Pull GGUF from the Hub; needs an SSL-enabled build |
| `--embeddings` | on | Required to expose `/v1/embeddings` |
| `-ngl N` | `99` | Layers offloaded to GPU; 99 = all |
| `--host` / `--port` | `127.0.0.1` / `8201` | Bind address |
| `--alias` | `embeddinggemma-300m` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity; default 3 hides the offload summary |
| `--pooling` | model default (`u32 = 1`, mean) | Override pooling if you need CLS/last |
| `-ub N` | 512 (auto-clamped) | Micro-batch; raise with `-b` together |

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

## Notes

- Batch size is clamped in embedding mode (`n_batch (2048) > n_ubatch (512)` →
  both set to 512). If you batch many texts per request, raise `-b` and `-ub` together or
  the effective batch stays 512.
- Matryoshka truncation is client-side: llama.cpp returns full-width 768-d vectors, so
  truncating to 512/256/128 has to happen in your code.
- A `control-looking token: 212 '</s>' was not control-type` warning prints at load on both
  backends and does not affect output.
