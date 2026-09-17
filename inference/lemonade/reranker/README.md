# `inference/lemonade/reranker` — Lemonade Server GGUF reranking

Serves `Qwen3-Reranker-0.6B` (GGUF, Q8_0, 610 MB) through Lemonade Server, which manages a
`llama-server` subprocess started with `--reranking`. `inference_reranker_lemonade.py` calls
`POST /v1/reranking` on port **8350** — the same process that serves [`../llm`](../llm) and
[`../embedding`](../embedding), so a whole RAG loop (embed, retrieve, rerank, generate) runs
against `127.0.0.1:8350`.

Use [`../../vllm`](../../vllm) or [`../../sglang`](../../sglang) for high-QPS reranking.

**Hardware:** AMD MI355X (gfx950) via `llamacpp:rocm` · NVIDIA H100 (sm_90) via
`llamacpp:cuda`, same GGUF and identical ranking on both.

## Files

- `inference_reranker_lemonade.py` — ranks a query against 2 relevant + 2 irrelevant
  documents, prints the ranking and `top_hit_is_relevant`, optionally writes the raw JSON.

## Setup

Install the server and the client venv from [`../README.md`](../README.md). Then:

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

Register with `--label reranking` — without it the model is treated as a chat model,
`--reranking` is never passed to `llama-server`, and `/v1/reranking` fails — then load it:

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-Reranker-0.6B \
  --checkpoint main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --recipe llamacpp \
  --label reranking
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-Reranker-0.6B
```

The model is ungated; it pulls with no `HF_TOKEN` exported.

## Run

```bash
../.env_lemonade/bin/python inference_reranker_lemonade.py \
  --port 8350 \
  --out $OUTPUT_DIR/inference_reranker_lemonade/rerank_single_gpu.json
```

Equivalent raw curl (the API model id drops the `user.` prefix):

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

Expected client output — both OpenTelemetry documents must rank above both distractors:

```text
endpoint  : http://127.0.0.1:8350/v1/reranking
model     : Qwen3-Reranker-0.6B
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=0.999667  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=0.996421  idx=0  A span represents a single unit of work in a distributed trace and carries
3. score=0.000036  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=0.000022  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

## GPU-residency check

A CPU fallback returns the same scores, and Lemonade launches `llama-server` without an
explicit `-ngl`, so this check is the decisive one:

```bash
rocm-smi --showpids                          # AMD: kernel KFD accounting
cat /sys/class/kfd/kfd/proc/<pid>/vram_*     # per-GPU bytes for that PID
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid   # NVIDIA
```

Expect GB-scale VRAM (weights plus the KV cache and device context) attributed to the
`llama-server` PID on a single GPU.

## Arguments

### Lemonade CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | Required on a shared host, or the CLI hangs |
| `pull --checkpoint TYPE REPO:QUANT` | `main ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family |
| `pull --label` | `reranking` | Required — drives `--reranking` on the wrapped server |
| `load <name>` | optional | Start/warm the subprocess before the first request |

### `inference_reranker_lemonade.py`

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Lemonade Server host |
| `--port` | `8350` | Lemonade Server port |
| `--model` | `Qwen3-Reranker-0.6B` | API model id (no `user.` prefix) |
| `--endpoint` | `/v1/reranking` | Also accepts `/api/v1/reranking`, `/v1/rerank`, `/api/v1/rerank` — all four return the same body |
| `--query` | OpenTelemetry span question | Query to rank against |
| `--documents` | 2 relevant + 2 irrelevant | Candidate documents |
| `--top_n` | `None` | Return only the top N |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/v1/health` polls before giving up |
| `--load_model` | off | `POST /api/v1/load` first |
| `--out` | `None` | Write the raw JSON response here |

## Notes

- **The first call after `load` is slower** — `load` returns before the first graph is
  built. Warm up before timing anything.
- **`--ctx-size` is chosen for you per build** (40960 on the ROCm backend, 4096 on CUDA) and
  is not overridable per model.
- **Compare rankings, not absolute scores**, when diffing this against another engine.
