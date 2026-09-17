# `inference/vllm/embedding` — EmbeddingGemma served by vLLM

Serves `google/embeddinggemma-300m` through vLLM's OpenAI-compatible `POST /v1/embeddings`
endpoint. `embed_vllm.py` embeds a query plus a few documents and prints the vector
dimensions and the query-to-document cosine similarities.

Pick this when you want one serving stack for all three workloads. If embeddings are all
you serve, a dedicated stack (TEI, sentence-transformers) is lighter — see
`training/embedding/sentence_transformers/`.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `embed_vllm.py` — embeds a query and documents, reports dimensions and cosines.

## Setup

Engine install (ROCm container or NVIDIA pip) and the shared client venv are in
[`../README.md`](../README.md). Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # HF_TOKEN — embeddinggemma-300m is a gated repo
```

```bash
export HF_HOME=/path/to/hf_cache
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides every GPU.

## Serve — AMD / ROCm

Single GPU, inside the container:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0
vllm serve google/embeddinggemma-300m \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name embeddinggemma
```

`--runner pooling` is mandatory; without it vLLM loads the model as a generative decoder
and `/v1/embeddings` is not routed.

## Serve — NVIDIA H100

Identical minus the HIP variable:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=<free-gpu>
vllm serve google/embeddinggemma-300m \
  --runner pooling \
  --host 0.0.0.0 --port 8500 \
  --served-model-name embeddinggemma
```

## Scaling — replicate, do not shard

EmbeddingGemma-300m has **3 attention heads**, so `--tensor-parallel-size 2` is rejected at
config validation on either vendor (heads must be divisible by TP size). Run one
independent single-GPU server per GPU behind a load balancer:

```bash
# replica A — first GPU
HIP_VISIBLE_DEVICES=0 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8001 --served-model-name embeddinggemma &

# replica B — second GPU
HIP_VISIBLE_DEVICES=1 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8011 --served-model-name embeddinggemma &
```

Both replicas answer the same request with cosines agreeing to ~1e-3 — ordinary
reduction-order variation, not a correctness problem.

## Client

```bash
python embed_vllm.py --port 8001 --model embeddinggemma
```

Use `--port 8500` against the H100 server. Raw equivalent:

```bash
curl -s http://localhost:8001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma","input":["vLLM supports AMD ROCm.","SQLite is an embedded database."]}'
```

## Arguments

Serve-side flags:

| Flag | Value | Meaning |
|---|---|---|
| `--runner pooling` | required | Switches vLLM to pooling mode; exposes `/v1/embeddings` |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` (ROCm) / `8500` (H100) | Embedding server port |
| `--served-model-name` | `embeddinggemma` | Alias clients pass as `model` |
| `--tensor-parallel-size` | *(not usable)* | Rejected — 3 attention heads, not divisible by 2 |
| `--gpu-memory-utilization` | default `0.9` | Lower it (e.g. `0.15`) when co-locating servers |
| `--max-model-len` | default `2048` | Raise only if you embed longer documents |

`embed_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8001` | Server port |
| `--model` | `embeddinggemma` | Served model name (must match `--served-model-name`) |
| `--query` | ROCm question | Query text; cosine is reported against each document |
| `--document` | 2 built-ins | Document text; repeat the flag for several documents |
| `--encoding_format` | `float` | `float` or `base64` |
| `--timeout` | `60.0` | HTTP timeout in seconds |
| `--show_dims` | `5` | How many leading vector components to print |

## Output

Vectors are 768-dimensional and the ROCm document must outscore the PostgreSQL one:

```
endpoint    : http://localhost:8001/v1/embeddings
model       : embeddinggemma
vectors     : 3
dimensions  : 768
usage       : {'prompt_tokens': 45, 'total_tokens': 45, 'completion_tokens': 0, 'prompt_tokens_details': None}

query       : Which inference engines support AMD ROCm?
  first 5 dims: [-0.09578, -0.08054, 0.04375, -0.00833, 0.03036]

  cos=+0.7105  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
  cos=+0.2011  PostgreSQL is a relational database management system.
```

Redirect server logs to a data volume, e.g. `$OUTPUT_DIR/inference_embedding_vllm/`.
Nothing large is written into the repo.

## Notes

- **The pooling server also exposes `/score`, `/rerank` and `/v2/rerank`.** That is normal
  for a pooling runner and does not make this model a reranker — use the right endpoint.
- **Default `--gpu-memory-utilization 0.9` inflates the VRAM figure** in `rocm-smi` /
  `nvidia-smi`: vLLM preallocates its pool, so a 300M model can show hundreds of GB "used".
- Benign ROCm startup noise: AITER's first-launch JIT build, the `quark_online_quant`
  import traceback, and the `[ROCm] PyTorch's native GELU with tanh approximation` fallback
  warning.
- On a network-mounted (CIFS/NFS) HF cache, an `Operation not permitted` warning while
  setting blob permissions is benign — the cached weights still load.
