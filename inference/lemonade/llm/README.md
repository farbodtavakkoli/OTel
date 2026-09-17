# `inference/lemonade/llm` — Lemonade Server GGUF chat completion

Serves `Qwen3.8-27B` as Q8_0 GGUF (27.1 GB) through Lemonade Server, which manages a
`llama-server` subprocess. `inference_llm_lemonade.py` calls
`POST /v1/chat/completions` on port **8350**. The same process also serves
[`../embedding`](../embedding) and [`../reranker`](../reranker).

Use [`../../vllm/llm`](../../vllm/llm) or [`../../sglang/llm`](../../sglang/llm) for the
native FP8 checkpoint, or raw `llama.cpp` when you need server flags (`-ngl`,
`--tensor-split`, `-c`) that Lemonade does not expose.

**Hardware:** AMD MI355X (gfx950) via `llamacpp:rocm` — MTP speculative decoding is enabled
automatically for this checkpoint · NVIDIA H100 (sm_90) via `llamacpp:cuda`.

## Files

- `inference_llm_lemonade.py` — health-polls the server, posts one chat completion, prints
  the text plus llama.cpp timings, optionally writes the raw JSON response.

## Setup

Install the server and the client venv from [`../README.md`](../README.md). Then:

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

Register and load the model (no `--label` for chat):

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen38-Q8 \
  --checkpoint main unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  --recipe llamacpp
$LEM/lemonade --port 8350 --no-discovery load user.Qwen38-Q8
```

Weights (27.1 GB) land in `$HF_HOME/hub/`; Lemonade binaries and backend wheels (4.8 GB) in
`$DATA_DIR/lemonade/cache`. Keep both off the root filesystem.

### Reusing a GGUF you already have

Lemonade downloads into its own HF-hub-shaped tree, so a GGUF in a `LLAMA_CACHE` directory
is invisible to it. Pre-seed the tree and `pull` reports `(already downloaded)`, skipping a
28 GB fetch. Symlinks are followed:

```bash
SRC=$HF_HOME/llama_cpp/models--unsloth--Qwen3.8-27B-GGUF
DST=$HF_HOME/hub/models--unsloth--Qwen3.8-27B-GGUF
SHA=27af057ecb382ddfea5d12837360a8980560e3ed

mkdir -p $DST/refs $DST/snapshots/$SHA
echo -n $SHA > $DST/refs/main
ln -sf $SRC/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf $DST/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf
cat > $DST/.lemonade_registry.json <<EOF
{"processed_models":{"user.Qwen38-Q8":{"selection":"llamacpp\nmain=unsloth/Qwen3.8-27B-GGUF:Q8_0","snapshot_id":"$SHA"}},
 "repo_id":"unsloth/Qwen3.8-27B-GGUF","revision":"$SHA","snapshot_id":"$SHA","source":"huggingface"}
EOF
```

## Run

```bash
../.env_lemonade/bin/python inference_llm_lemonade.py \
  --port 8350 --max_tokens 900 \
  --out $OUTPUT_DIR/inference_llm_lemonade/chat_single_gpu.json
```

Equivalent raw curl (note the model id has no `user.` prefix):

```bash
curl -s http://127.0.0.1:8350/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen38-Q8","messages":[{"role":"user","content":"Explain an OpenTelemetry span."}],
       "max_tokens":900,"temperature":0.0}' | jq -r '.choices[0].message.content'
```

On NVIDIA, `unsloth/Qwen3-0.6B-GGUF:Q8_0` (610 MB) registers identically and exercises the
backend quickly when no large GGUF is cached; pass `--model Qwen3-06B-Q8`. It has no MTP
head, so no speculative decoding is enabled.

Expected output:

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen38-Q8
fingerprint: b10469-666f8898a
finish     : stop
prompt     : In exactly two sentences, explain what an OpenTelemetry span is.
--- generated text ---
An OpenTelemetry span represents a single unit of work in a distributed trace, such as a
request, function call, or database query. ...
```

## GPU-residency check

Lemonade launches `llama-server` without an explicit `-ngl`, so verify the offload rather
than assuming it:

```bash
rocm-smi --showpids                          # AMD: kernel KFD accounting
cat /sys/class/kfd/kfd/proc/<pid>/vram_*     # per-GPU bytes for that PID
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid   # NVIDIA
```

Weights, KV cache, MTP draft model and the device context should all be attributed to the
`llama-server` PID on exactly one GPU.

## Arguments

### Lemonade CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | Required on a shared host, or the CLI hangs |
| `pull --checkpoint TYPE REPO:QUANT` | `main unsloth/Qwen3.8-27B-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` | Backend family |
| `pull --label` | not needed for chat | Only embeddings/reranking need a label |
| `load <name>` | optional | Start/warm the subprocess before the first request |
| `list` / `backends --all` | — | Registry, and the per-arch backend support matrix |

### `inference_llm_lemonade.py`

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Lemonade Server host |
| `--port` | `8350` | Lemonade Server port |
| `--model` | `Qwen38-Q8` | API model id (no `user.` prefix) |
| `--endpoint` | `/v1/chat/completions` | Also `/api/v1/chat/completions`, `/v1/completions` |
| `--prompt` | OpenTelemetry span question | User message |
| `--system` | `None` | Optional system message |
| `--max_tokens` | `256` | Generation cap — raise it, see Notes |
| `--temperature` | `0.0` | Sampling temperature |
| `--timeout` | `900` | HTTP timeout (s) |
| `--health_retries` | `120` | `/api/v1/health` polls before giving up |
| `--load_model` | off | `POST /api/v1/load` first |
| `--show_reasoning` | off | Also print `reasoning_content` |
| `--out` | `None` | Write the raw JSON response here |

## Notes

- **Qwen3.8 is a reasoning model — budget `--max_tokens` for it.** With
  `--reasoning-format auto` the thinking tokens land in `reasoning_content` and `content`
  stays empty until thinking finishes. A 320-token cap on a code prompt returns
  `finish_reason: length` with an empty `content`; 900 works. `--show_reasoning` prints the
  reasoning stream.
- **`--ctx-size 262144` is chosen for you** on the 27B — the model's full context, and a
  large KV allocation. It is not overridable per model.
- `ROCm0 does not have support for op TOP_K` warnings are harmless: the sampler falls back
  to CPU for top-k while the model stays on GPU.
