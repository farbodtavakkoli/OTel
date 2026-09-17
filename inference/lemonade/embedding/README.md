# `inference/lemonade/embedding` — Lemonade Server GGUF embeddings

Serves `EmbeddingGemma-300M` (GGUF, Q8_0, 319 MB) through Lemonade Server, which manages a
`llama-server` subprocess. `inference_embedding_lemonade.py` calls `POST /v1/embeddings` on
port **8350** — the same process that serves [`../llm`](../llm) and
[`../reranker`](../reranker).

Use raw `llama.cpp` when you need server flags (`-ngl`, `-b`/`-ub`, `--pooling`) that
Lemonade does not expose per model.

**Hardware:** AMD MI355X (gfx950) via `llamacpp:rocm` · NVIDIA H100 (sm_90) via
`llamacpp:cuda`, same GGUF on both. Vectors differ by ~1e-3 per component between vendors
(accumulation order), and the related/unrelated ordering holds on both.

## Files

- `inference_embedding_lemonade.py` — embeds a 3-sentence probe, prints dim, L2 norm and a
  cosine sanity check, optionally cross-checks against a reference artifact.

## Setup

Install the server and the client venv from [`../README.md`](../README.md). Then:

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

Register with `--label embeddings` — that is what starts the wrapped server with
`--embeddings` and creates the `/v1/embeddings` route — then load it:

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.EmbeddingGemma-300M \
  --checkpoint main ggml-org/embeddinggemma-300M-GGUF:Q8_0 \
  --recipe llamacpp \
  --label embeddings
$LEM/lemonade --port 8350 --no-discovery load user.EmbeddingGemma-300M
```

The model is ungated; it pulls with no `HF_TOKEN` exported.

## Run

```bash
../.env_lemonade/bin/python inference_embedding_lemonade.py \
  --port 8350 \
  --out $OUTPUT_DIR/inference_embedding_lemonade/embeddings_single_gpu.json
```

Equivalent raw curl (the API model id drops the `user.` prefix):

```bash
curl -s http://127.0.0.1:8350/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"EmbeddingGemma-300M","input":["a span is a unit of work in a trace"]}' \
  | jq '.data[0].embedding | length'
```

Expected output — 768-dimensional, L2-normalised, related pair above unrelated:

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

## Cross-check against the Transformers baseline

llama.cpp does not apply EmbeddingGemma's task prompts, while the sentence-transformers
baseline (`encode_query` / `encode_document`) does — so a comparison must send the prefixed
strings. `--reference` with no `--texts` applies `--reference_prompt` to the baseline's own
queries for you:

```bash
../.env_lemonade/bin/python inference_embedding_lemonade.py \
  --reference $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_1gpu.json
```

## GPU-residency check

A CPU fallback still returns correct-looking vectors, and Lemonade launches `llama-server`
without an explicit `-ngl`, so this check is the decisive one:

```bash
rocm-smi --showpids                          # AMD: kernel KFD accounting
cat /sys/class/kfd/kfd/proc/<pid>/vram_*     # per-GPU bytes for that PID
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid   # NVIDIA; 0 MiB means CPU
```

Expect GB-scale VRAM attributed to the `llama-server` PID (Lemonade proxies 8350 -> 8001/8002)
on a single GPU.

## Arguments

### Lemonade CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | Required on a shared host, or the CLI hangs |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family |
| `pull --label` | `embeddings` | Required — drives `--embeddings` on the wrapped server |
| `load <name>` | optional | Start/warm the subprocess before the first request |

### `inference_embedding_lemonade.py`

| Flag | Default | Meaning |
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

## Notes

- **Embeddings are not reproducible across a live server's request history.** Vectors are
  stable while the batch shape stays constant and shift by ~1e-3 per component after a
  request of a different shape — `llama-server` slot/KV state and batch packing, not
  Lemonade. Semantically harmless, but if you need byte-reproducible embeddings, batch your
  corpus in fixed-shape requests. `--load_model` also perturbs the slot, so the reference
  runs above omit it.
- **`--ctx-size 8192` is chosen for you** and is not overridable per model.
