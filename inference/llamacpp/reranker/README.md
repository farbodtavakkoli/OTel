# `inference/llamacpp/reranker` — llama.cpp GGUF reranker serving (Qwen3-Reranker-0.6B)

Serves `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` through `llama-server --reranking` and
hits `POST /v1/rerank` with `inference_reranker_llamacpp.py`, which verifies that relevant
documents actually outrank irrelevant ones.

Pick this folder for a cheap second-stage reranker in a retrieval pipeline, served by the
same binary as the LLM and embedding leaves. HF TEI does not support this Qwen3 reranker, so
llama.cpp or [`../../vllm/reranker/`](../../vllm/reranker/) is the route. The GGUF repo is
ungated and needs no `HF_TOKEN`.

Qwen3-Reranker is a decoder-style yes/no reranker, not a classic cross-encoder with a
regression head. llama.cpp does the yes/no logit comparison internally and exposes a plain
`relevance_score` in `[0, 1]`, so the client never builds the yes/no prompt itself.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2) and NVIDIA H100 (Hopper cc 9.0, CUDA
13), Python 3.12.

## Files

- `inference_reranker_llamacpp.py` — rerank smoke client; re-sorts the response by score and
  prints the ranked documents.

## Setup

Build `llama-server` and create the shared venv per [`../README.md`](../README.md) — one
build and one venv serve all three leaves. Then:

```bash
ln -sf ../../../dev.env dev.env                 # only needed for gated repos
export HF_HOME=/path/to/hf_cache                # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp           # llama.cpp's own -hf cache (610 MB for this model)
export OUTPUT_DIR=/path/to/outputs              # inference artifacts
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
```

## Run

### Serve — single GPU

```bash
cd <your llama.cpp checkout>
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0

./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --reranking \
  -ngl 99 \
  -lv 5 \
  --host 127.0.0.1 --port 8202 \
  --alias qwen3-reranker-0.6b
```

`--reranking` is required or the rerank routes 404. The first run downloads 610 MB; cached
restarts take under a second.

### Serve — one instance per GPU (the scale-out pattern)

Do not split a 0.6B reranker across GPUs — a layer split only adds a device-to-device
transfer per forward pass. Run independent instances behind a load balancer instead;
concurrent instances produce identical relevance scores.

```bash
# instance A — first GPU, port 8212
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --reranking -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8212 --alias qwen3-reranker-0.6b-gpu0 &

# instance B — second GPU, port 8202
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --reranking -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8202 --alias qwen3-reranker-0.6b &
```

### Client

```bash
../.env_llamacpp/bin/python inference_reranker_llamacpp.py \
  --port 8202 \
  --out $OUTPUT_DIR/inference_reranker_llamacpp/rerank_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s -X POST http://127.0.0.1:8202/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker-0.6b",
       "query":"What is an OpenTelemetry span?",
       "documents":["A span is a unit of work in a trace.","Bake bread at 200C."]}' \
  | jq -c '.results'
```

### NVIDIA / CUDA

Identical serve and client commands and the identical model — at 610 MB no substitution is
needed. Only the build flag differs (`-DGGML_CUDA=ON`, see [`../README.md`](../README.md)).

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # for the HF pull
export CUDA_VISIBLE_DEVICES=0
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, if your model share rejects renames

./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --reranking -ngl 999 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias qwen3-reranker-0.6b

../.env_llamacpp/bin/python inference_reranker_llamacpp.py --port 8700 --model qwen3-reranker-0.6b
```

## Output

The default documents are two relevant and two irrelevant, interleaved, so a correct run is
a real reordering rather than an echo of input order:

```text
endpoint  : http://127.0.0.1:8202/v1/rerank
model     : qwen3-reranker-0.6b
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=+0.99966  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99639  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00004  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00002  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

Both OpenTelemetry documents must outrank both distractors. The `/rerank` and
`/v1/reranking` routes return the same scores, on both backends.

`-ngl 99` can fall back to CPU silently, so check the offload summary printed at `-lv 5`:

```text
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   157.37 MiB
load_tensors:        ROCm0 model buffer size =   603.87 MiB
```

All 29 layers must be on the GPU; the `CPU_Mapped` buffer is the token-embedding table, kept
host-side by design. Confirm independently with `rocm-smi --showpids` or
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

The client writes raw JSON wherever `--out` points; weights stay under `$LLAMA_CACHE`.

## Arguments

### `llama-server`

| Argument | Value | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Pull GGUF from the Hub; needs an SSL-enabled build |
| `--reranking` | on | Required to expose `/v1/rerank`, `/rerank`, `/v1/reranking` |
| `-ngl N` | `99` | Layers offloaded to GPU; 99 = all |
| `--host` / `--port` | `127.0.0.1` / `8202` | Bind address |
| `--alias` | `qwen3-reranker-0.6b` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity; default 3 hides the offload summary |
| `-c N` | model default | Context length; Qwen3-Reranker supports 32K |

### `inference_reranker_llamacpp.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8202` | llama-server port |
| `--model` | `qwen3-reranker-0.6b` | Model name echoed in the request body |
| `--query` | OTel span question | Query to rank documents against |
| `--documents` | 2 relevant + 2 irrelevant | Candidate documents |
| `--top_n` | `None` | Return only the top N documents |
| `--endpoint` | `/v1/rerank` | Route: `/v1/rerank`, `/rerank`, or `/v1/reranking` |
| `--timeout` | `300` | HTTP timeout (s) |
| `--health_retries` | `60` | `/health` polls before giving up |
| `--out` | `None` | Write the raw JSON response here |

## Notes

- Send one throwaway request at startup before putting an instance into rotation — the first
  request pays prompt-cache and graph warm-up.
- Scores are absolute, not relative to the candidate set: they come from a yes/no logit
  comparison, so you can threshold on them (e.g. drop anything below 0.5) across queries.
- A minja `Callee is not a function: got Undefined (hint: 'lstrip')` chat-template error
  prints at load on both backends. Reranking does not go through the chat template, so
  scoring is unaffected.
