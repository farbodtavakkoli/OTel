# `inference/ollama/llm` — Ollama GGUF LLM serving (Qwen3.8-27B)

Serves `unsloth/Qwen3.8-27B-GGUF:Q8_0` through Ollama in the official `ollama/ollama:rocm`
(AMD) or `ollama/ollama` (NVIDIA) container. `inference_llm_ollama.py` hits `POST /api/chat`,
`POST /api/generate`, or the OpenAI-compatible `POST /v1/chat/completions`.

One daemon serves this 27B LLM and the 300M embedder from [`../embedding/`](../embedding/)
concurrently, loading each on demand. For datacenter-class serving use
[`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/). Ollama is a
GGUF path — the FP8 safetensors checkpoint `Qwen/Qwen3.8-27B-FP8` needs vLLM or SGLang.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2), single- and two-GPU, and NVIDIA H100 80GB
(CUDA 13). Only the image tag and the device flag differ.

## Files

- `Modelfile.qwen3-27b` — registers the cached GGUF with Ollama, no second 29 GB pull.
- `inference_llm_ollama.py` — chat/generate smoke client; polls `/api/tags`, prints latency,
  token counts, tokens/s and the generated text.

## Setup

Images, GPU-to-render-node mapping, run-line deviations and the shared `.env_ollama` venv:
[`../README.md`](../README.md). Then:

```bash
ln -sf ../../../dev.env dev.env                  # no token actually needed on this path
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
export GGUF=$LLAMA_CACHE                         # or a fresh dir if you download instead
export OUTPUT_DIR=/path/to/outputs               # inference artifacts
```

## Run

### Serve — AMD / ROCm

```bash
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_llm \
  ollama/ollama:rocm
```

Add a second `--device /dev/dri/renderD152` line for a two-GPU container.

### Serve — NVIDIA / CUDA

Same command with `--gpus '"device=N"'` (or `'"device=N,M"'` for two cards) in place of the
`--device /dev/kfd --device /dev/dri/renderD*` flags, and image `ollama/ollama`. If the GGUF
is not already under `$LLAMA_CACHE`, download it first and point the Modelfile `FROM` at the
container path under the `:ro /ggufs` mount:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir $GGUF/Qwen3-0.6B-GGUF
#   FROM /ggufs/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
```

`unsloth/Qwen3-0.6B-GGUF:Q8_0` is a drop-in for a quick smoke test — same family, same Q8_0
quant. Judge answer quality on the 27B GGUF, not on a 0.6B model.

Verify the daemon sees only your GPUs:

```bash
docker logs ollama_llm 2>&1 | grep "inference compute"
```

```text
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

One line per GPU passed in. `available` reflects other tenants on the host, not card
capacity.

### Register the model

The canonical route `ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0` re-downloads the
weights. Register the cached file directly instead.

`Modelfile.qwen3-27b`:

```text
FROM /ggufs/models--unsloth--Qwen3.8-27B-GGUF/snapshots/27af057ecb382ddfea5d12837360a8980560e3ed/Qwen3.8-27B-Q8_0.gguf
```

```bash
docker cp Modelfile.qwen3-27b ollama_llm:/root/Modelfile.qwen3-27b
docker exec ollama_llm ollama create qwen3.8-27b-q8 -f /root/Modelfile.qwen3-27b
docker exec ollama_llm ollama list
```

Registration needs no network but copies the blob into the store. Mount the whole HF cache
tree, not just the snapshot directory — the snapshot symlinks (`../../blobs/<sha256>`) must
resolve in-container.

### Client

```bash
../.env_ollama/bin/python inference_llm_ollama.py \
  --api chat \
  --out $OUTPUT_DIR/inference_llm_ollama/chat_multi_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model":"qwen3.8-27b-q8","think":false,"stream":false,
  "messages":[{"role":"user","content":"In one sentence, what is an OpenTelemetry span?"}],
  "options":{"num_predict":128,"temperature":0.7,"seed":42}}'
```

### GPU placement

Ollama spreads a model over several GPUs when the auto-fitter cannot satisfy its safety
target on one card, and its default context is derived from *total* VRAM, which reserves a
multi-GB KV cache. To force single-GPU placement, pass only one render node (or one
`--gpus '"device=N"'`), or cap the context with `-e OLLAMA_CONTEXT_LENGTH=8192` (or
per-request `"options":{"num_ctx":8192}`). `-e OLLAMA_SCHED_SPREAD=1` forces the opposite —
a spread across every visible GPU.

## Output

```text
endpoint      : http://127.0.0.1:11434/api/chat
model         : qwen3.8-27b-q8
latency_s     : 0.88
prompt_tokens : 36
output_tokens : 38
load_s        : 0.20
tok_per_s     : 76.3
--- generated text ---
An OpenTelemetry span is a fundamental unit of work within a trace that represents a
specific operation or task, containing metadata such as its duration, parent-child
relationships, and associated attributes.
```

Raw JSON is written under `$OUTPUT_DIR/inference_llm_ollama/`, not into the repo.

On a many-core box a CPU fallback still generates text, so check residency explicitly — look
for `100% GPU`, not `CPU` or a split like `40%/60%`:

```bash
docker exec ollama_llm ollama ps
```

```text
NAME                     ID              SIZE     PROCESSOR    CONTEXT    UNTIL
qwen3.8-27b-q8:latest    b0322c83ce26    44 GB    100% GPU     262144     4 minutes from now
```

The server log carries the runner's own offload summary as a cross-check —
`load_tensors: offloaded 66/66 layers to GPU` plus the `ROCm*` / `CUDA*` model and KV buffer
sizes.

## Arguments

### Docker flags

| Flag | Value used | Why |
|---|---|---|
| `--device /dev/kfd` | required | ROCm compute node; without it there is no GPU at all |
| `--device /dev/dri/renderD<N>` | one per GPU | per-GPU pinning; use instead of exposing the whole `/dev/dri` |
| `--gpus '"device=<N>"'` | NVIDIA equivalent | per-GPU pinning on CUDA; `'"device=N,M"'` for two cards |
| `-v $DATA_DIR/ollama:/root/.ollama` | a disk with room | model store; a named volume lands on the root filesystem |
| `-v $LLAMA_CACHE:/ggufs:ro` | read-only | reuse cached GGUFs, no re-download |
| `-p 11434:11434` | default | Ollama's default port |
| `-d --name ollama_llm` | — | detached, named for `docker exec` |

### Server environment (`-e`)

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_SCHED_SPREAD` | `false` | `1` forces a model across every visible GPU |
| `OLLAMA_CONTEXT_LENGTH` | `0` (derived from total VRAM) | caps context; the single biggest VRAM lever |
| `OLLAMA_NUM_PARALLEL` | `1` | concurrent request slots per model |
| `OLLAMA_MAX_LOADED_MODELS` | `0` (auto) | how many models may be resident together |
| `OLLAMA_KEEP_ALIVE` | `5m0s` | idle time before a model is unloaded |
| `OLLAMA_FLASH_ATTENTION` | `false` | enables flash attention in the runner |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | e.g. `q8_0` to halve KV cache size |

### `inference_llm_ollama.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Ollama server host |
| `--port` | `11434` | Ollama server port |
| `--model` | `qwen3.8-27b-q8` | model name created by `ollama create` |
| `--api` | `chat` | `chat` → `/api/chat`, `generate` → `/api/generate`, `openai` → `/v1/chat/completions` |
| `--prompt` | OTel span question | user prompt |
| `--system` | concise assistant | system prompt (ignored by `generate`) |
| `--think` | off | keep the model's thinking block on |
| `--max_tokens` | `128` | maps to `options.num_predict` / OpenAI `max_tokens` |
| `--temperature` | `0.7` | sampling temperature |
| `--top_p` | `0.95` | nucleus sampling mass |
| `--seed` | `42` | sampling seed |
| `--num_ctx` | unset | override the VRAM-derived context window |
| `--keep_alive` | `5m` | how long Ollama keeps the model resident |
| `--timeout` | `900` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/tags` probes before giving up |
| `--out` | none | write the raw JSON response |

## Notes

- Thinking mode is on by default in the GGUF's chat template and consumes the token budget.
  On `/v1/chat/completions` the whole budget goes into the non-standard `message.reasoning`
  field and `content` comes back `''`. Use the native API with `"think": false` (the client's
  default); there is no `think` flag on the OpenAI route.
- The container needs no GPU userspace on the host — it bundles its own (`rocm_v7_2` /
  `cuda_v13`) and takes only the kernel driver from the host.
