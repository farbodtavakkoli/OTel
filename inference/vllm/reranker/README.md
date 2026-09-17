# `inference/vllm/reranker` — Qwen3-Reranker-0.6B served by vLLM

Serves `Qwen/Qwen3-Reranker-0.6B` through vLLM's OpenAI-compatible `POST /v1/rerank`
endpoint. `rerank_vllm.py` sends one query plus candidate documents and prints them
re-ordered with relevance scores.

Qwen3-Reranker is natively a decoder-only yes/no-token scorer. vLLM converts it into a
sequence-classification path, which is why the serve command needs both the
`--hf_overrides` JSON and the jinja chat template — they are what performs the conversion.
Use this as the precision step after a cheap recall step: retrieve top-50 with
[`../embedding/`](../embedding/), then rerank to top-5 here.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `rerank_vllm.py` — sends a query plus documents to `/v1/rerank`, prints the ranking.
- `qwen3_reranker.jinja` — chat template that renders the Instruct/Query/Document prompt
  the model was trained on. Required by the serve command.

## Setup

Engine install (ROCm container or NVIDIA pip) and the shared client venv are in
[`../README.md`](../README.md). Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # supplies HF_TOKEN for the model pull
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
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 \
  --port 8002
```

Two GPUs — add one flag:

```bash
export HIP_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --tensor-parallel-size 2 \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 \
  --port 8002
```

TP=2 works and scores agree with TP=1 to ~1e-5, but at this model size prefer two
independent single-GPU replicas (see [`../embedding/README.md`](../embedding/README.md)).

## Serve — NVIDIA H100

Identical minus the HIP variable:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=<free-gpu>
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 --port 8500
```

## Client

```bash
python rerank_vllm.py --port 8002 --model qwen3-reranker
```

Use `--port 8500` against the H100 server. Raw equivalent:

```bash
curl -s http://localhost:8002/v1/rerank -H 'Content-Type: application/json' -d '{
 "model":"qwen3-reranker",
 "query":"Which inference engines support AMD ROCm?",
 "documents":[
  "vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.",
  "PostgreSQL is a relational database management system.",
  "SGLang also provides a ROCm build for AMD GPUs.",
  "The Eiffel Tower is located in Paris, France."
 ]}'
```

## Arguments

Serve-side flags:

| Flag | Value | Meaning |
|---|---|---|
| `--runner pooling` | required | Pooling/scoring mode; exposes `/v1/rerank`, `/score`, `/v1/score` |
| `--hf_overrides` | JSON below | **Required** — rewrites the loaded architecture |
| `--chat-template` | `qwen3_reranker.jinja` | **Required** — renders the Instruct/Query/Document prompt |
| `--tensor-parallel-size` | `1` / `2` | Shards across GPUs; prefer replication at this size |
| `--served-model-name` | `qwen3-reranker` | Client-facing alias |
| `--host` / `--port` | `0.0.0.0` / `8002` (ROCm), `8500` (H100) | Bind address and port |
| `--gpu-memory-utilization` | default `0.9` | Lower when co-locating servers on one card |

The `--hf_overrides` JSON, field by field:

| Key | Value | Why |
|---|---|---|
| `architectures` | `["Qwen3ForSequenceClassification"]` | Loads the model as a sequence classifier instead of a causal LM |
| `classifier_from_token` | `["no","yes"]` | Builds the 2-logit head from the existing `no`/`yes` token embeddings, preserving the original scoring semantics |
| `is_original_qwen3_reranker` | `true` | Marks the original Qwen reranker layout so the conversion applies correctly |

`rerank_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8002` | Server port |
| `--model` | `qwen3-reranker` | Served model name (must match `--served-model-name`) |
| `--query` | ROCm question | Query to rank documents against |
| `--document` | 4 built-ins | Candidate document; repeat the flag for several |
| `--top_n` | `None` | Return only the top N results (default: all) |
| `--timeout` | `120.0` | HTTP timeout in seconds |

## Output

Both ROCm documents must rank above both irrelevant ones. Score *scale* is
template-driven, so check the ranking, not the absolute value:

```
endpoint    : http://localhost:8002/v1/rerank
model       : qwen3-reranker
query       : Which inference engines support AMD ROCm?
usage       : {'prompt_tokens': 372, 'total_tokens': 372}

rank  index  score       document
   1      0  0.999505  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
   2      2  0.992223  SGLang also provides a ROCm build for AMD GPUs.
   3      1  0.000128  PostgreSQL is a relational database management system.
   4      3  0.000028  The Eiffel Tower is located in Paris, France.
```

Redirect server logs to a data volume, e.g. `$OUTPUT_DIR/inference_reranker_vllm/`.
Nothing large lands in the repo.

## Notes

- **Both `--hf_overrides` and `--chat-template` are mandatory.** Without the overrides the
  model loads as a causal LM with no scoring head; without the template the prompt no
  longer matches what the model was trained on and scores degrade *silently* while the
  endpoint still answers.
- **The pooling server also exposes `/v1/embeddings`.** Same process, different endpoint —
  it does not make the reranker an embedding model.
- **Default `--gpu-memory-utilization 0.9` makes the card look full** for a 0.6B model.
  That is the preallocated KV pool; read `Model loading took N GiB` from the server log for
  the real weight footprint.
- Benign ROCm startup noise: AITER's first-launch JIT build and the `quark_online_quant`
  import traceback.
