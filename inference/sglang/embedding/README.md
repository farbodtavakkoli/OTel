# `inference/sglang/embedding` — SGLang embedding serving (EmbeddingGemma-300m)

Serves `google/embeddinggemma-300m` behind an OpenAI-compatible `POST /v1/embeddings`
endpoint with SGLang (`python -m sglang.launch_server --is-embedding`).
`inference_embedding_sglang.py` embeds a query plus documents and prints the vector
dimension and cosine scores.

Use this folder when you already run SGLang for generation and want embeddings from the
same stack. If you want an embedding endpoint without pulling a ~90 GB ROCm image, the
Transformers or vLLM folder is lighter.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `inference_embedding_sglang.py` — embeds a query and documents, reports dim and cosines.

## Setup

Engine install (NVIDIA pip or the ROCm container) is in [`../README.md`](../README.md).
Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # HF_TOKEN — EmbeddingGemma is a gated Google model
export HF_HOME=/path/to/hf_cache
export OUTPUT_DIR=/path/to/outputs
```

Accept Google's terms on the model page before the first pull, or it will 403.

## Serve — NVIDIA H100

```bash
source ../.env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1
set -a; source dev.env; set +a          # HF_TOKEN for the gated repo
python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m \
  --is-embedding \
  --mem-fraction-static 0.5 \
  --host 127.0.0.1 --port 8600
```

Server log on success:

```
EmbeddingGemma detected: disabling radix cache and chunked prefill; using breakable CUDA graph for CUDA prefill.
The server is fired up and ready to roll!
```

## Serve — AMD / ROCm

Single GPU, from the long-lived container:

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8101 \
     > $OUTPUT_DIR/inference_embedding_sglang/embed_rep_a.log 2>&1"
```

`GET /get_model_info` confirms embedding mode (`"is_generation":false`, `"task":"embed"`).

## Scaling — replicate, do not shard

For a 300M model run one independent server per GPU behind a load balancer rather than
`--tp 2`:

```bash
# replica B -> second GPU, port 8111
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8111 > $OUTPUT_DIR/inference_embedding_sglang/embed_rep_b.log 2>&1"
```

The replicas return identical vectors, so the pool is safe to load-balance across. Each
holds the `--mem-fraction-static 0.5` pool rather than real demand — lower it to pack more
replicas per GPU.

## Client

```bash
python inference_embedding_sglang.py --port 8101 \
  --query "Which inference engines support AMD ROCm?" \
  --texts "vLLM supports AMD ROCm." "SQLite is an embedded database."
```

Use `--port 8600` against the H100 server. From inside the container:

```bash
docker exec -w /work/inference/sglang/embedding sglang_bringup \
  python inference_embedding_sglang.py --port 8101
```

## Arguments

Server flags that matter:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `google/embeddinggemma-300m` | Gated; needs `HF_TOKEN` |
| `--is-embedding` | on | Correct for this model (and forbidden for the Qwen3 reranker) |
| `--tp` | `1` | Correct for a 300M model; scale with replicas, not TP |
| `--mem-fraction-static` | `0.5` | KV/static pool fraction; auto-reduced at TP>1 |
| `--attention-backend` | *unset* | Leave unset — `aiter` on gfx950, CUDA path on Hopper |

Client (`inference_embedding_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8101` | Server port (8101 = SGLang embedding slot in the repo port map) |
| `--model` | `google/embeddinggemma-300m` | Model id as served; must match `--model-path` |
| `--query` | ROCm engines question | Query embedded and scored against each document |
| `--texts` | 2 sample documents | Documents to embed (space-separated list) |
| `--endpoint` | `/v1/embeddings` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

## Output

`dim` must be **768** (EmbeddingGemma's native width) and the relevant document must score
above the irrelevant one. The absolute gap is narrow when the client sends raw text; the
ordering is what matters.

```
[health] server ready after 1.0s
[latency] 0.03s | n_vectors=3 dim=768
[vector0 head] [-0.1543, -0.02832, -0.00574, 0.00757, -0.00922, 0.02148, -0.05151, 0.05591]
[cosine] +0.8720  <- vLLM supports AMD ROCm.
[cosine] +0.8061  <- SQLite is an embedded database.
```

The server writes no artifacts; redirect logs where you want them, e.g.
`$OUTPUT_DIR/inference_embedding_sglang/`. Weights live in `$HF_HOME`.

## Notes

- **KV cache is large by default** even for a 300M model. Set `--mem-fraction-static` or
  `--max-total-tokens` explicitly when sharing the box.
- `chunked_prefill_size` is forced to `-1` for embedding runs — expected.
- Benign startup noise: `Failed to import amdsmi` (install `amdsmi` for AMD telemetry),
  `Ignoring corrupted tree cache file ... Permission denied` on a shared `$HF_HOME`,
  AITER's `[aiter] import [mha_batch_prefill_...]` JIT burst and `not found tuned config in
  /tmp/aiter_configs/bf16_tuned_gemm.csv`, `Ignore import error when loading
  sglang.srt.models.inkling`, and `FastAPIDeprecationWarning: ORJSONResponse is deprecated`.
