# `inference/ollama/llm` — Ollama GGUF LLM serving (Qwen3.8-27B)

> Stack overview, shared install route (images, GPU→render-node mapping, run-line
> deviations, per-vendor pinning), environment/secrets, shared quirks, and venv conventions:
> [`../README.md`](../README.md). Client deps: [`../requirements.txt`](../requirements.txt).

## Overview & when to use

Serves a **Qwen3.8-27B GGUF** checkpoint through **Ollama**, running in the official
`ollama/ollama:rocm` (AMD) or `ollama/ollama` (NVIDIA) container.
The client `inference_llm_ollama.py` hits `POST /api/chat`, `POST /api/generate`, or the
OpenAI-compatible `POST /v1/chat/completions`.

One daemon serves this 27B LLM and the 300M embedder from
[`../embedding/`](../embedding/) concurrently, loading each on demand. For
datacenter-class serving use [`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/).

> **Critical format note:** Ollama is a **GGUF path**. It does **not** consume the Hugging
> Face FP8 safetensors checkpoint `Qwen/Qwen3.8-27B-FP8`. This folder therefore serves
> `unsloth/Qwen3.8-27B-GGUF:Q8_0`. For the exact FP8 checkpoint use vLLM or SGLang.

## Scope note — why there is no `inference/ollama/reranker`

Ollama's HTTP surface (`/api/generate`, `/api/chat`, `/api/embed` + the OpenAI shim) has
no rerank route. Use [`../../llamacpp/reranker/`](../../llamacpp/reranker/) or
[`../../vllm/reranker/`](../../vllm/reranker/).

## Install — the docker route

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
export GGUF=$LLAMA_CACHE                         # or a fresh dir if you download instead
export OUTPUT_DIR=/path/to/outputs               # inference artifacts
```

The shared route — `docker pull ollama/ollama:rocm` / `ollama/ollama` (gfx950 native, no
`HSA_OVERRIDE_GFX_VERSION`), the GPU→render-node mapping, and the two deliberate run-line
deviations — is documented in [`../README.md`](../README.md). The serve commands for this
leaf:

### Serve — single GPU (AMD / ROCm)

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

### Serve — two GPUs (AMD / ROCm)

```bash
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  --device /dev/dri/renderD152 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_llm \
  ollama/ollama:rocm
```

### Serve — NVIDIA / CUDA

Same command with `--gpus '"device=N"'` (or `'"device=N,M"'` for two cards) in place of the
`--device /dev/kfd --device /dev/dri/renderD*` flags, and image `ollama/ollama`. If the GGUF
is not already under `$LLAMA_CACHE`, download it first (unset the proxy if Hugging Face is
proxy-blocked) and point the Modelfile `FROM` at the container path under the `:ro /ggufs`
mount:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir $GGUF/Qwen3-0.6B-GGUF
#   FROM /ggufs/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
```

`unsloth/Qwen3-0.6B-GGUF:Q8_0` is a drop-in for a quick smoke test — same Qwen3 family,
same Q8_0 quant, identical path.

### Confirm the backend sees only your GPUs

```bash
docker logs ollama_llm 2>&1 | grep "inference compute"
```

```text
... id=1 filter_id=1 library=ROCm compute=gfx950 name=ROCm1 libdirs=ollama,rocm_v7_2 pci_id=0000:dc:00.0 type=discrete total="288.0 GiB" available="75.0 GiB"
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

Environment & secrets: see [`../README.md`](../README.md). (`dev.env` is
symlinked to the repo-root file — `ln -sf ../../../dev.env dev.env` — and the client loads
it via `load_dotenv("dev.env")`; no token is actually needed on this path.)

## Model registration — reuse the cached GGUF

The canonical route is `ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0`, which re-downloads
the weights. If the file is already on disk, register it directly instead.

`Modelfile.qwen3-27b`:

```text
FROM /ggufs/models--unsloth--Qwen3.8-27B-GGUF/snapshots/27af057ecb382ddfea5d12837360a8980560e3ed/Qwen3.8-27B-Q8_0.gguf
```

```bash
docker cp Modelfile.qwen3-27b ollama_llm:/root/Modelfile.qwen3-27b
docker exec ollama_llm ollama create qwen3.8-27b-q8 -f /root/Modelfile.qwen3-27b
```

```text
parsing GGUF
verifying conversion
copying file sha256:a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348 100%
creating new layer sha256:b908bbcedf26477f3922a1c59d15c6d106a06c96f7e08fd8f5070c17f877c8d7
writing manifest
success
```

Registration needs **no network**, but it **copies** the blob into the store. Mount the
whole HF cache tree, not just the snapshot directory — the snapshot symlinks
(`../../blobs/<sha256>`) must resolve inside the container.
`docker exec ollama_llm ollama list` then shows the registered model.

## Client / smoke command

```bash
cd .. && python3 -m venv .env_ollama && .env_ollama/bin/pip install -r requirements.txt && cd llm

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

## Confirm GPU residency

On a many-core box a CPU fallback still generates text, so this is the check that matters.
`ollama ps` reports a `PROCESSOR` column:

```text
NAME                     ID              SIZE     PROCESSOR    CONTEXT    UNTIL
qwen3.8-27b-q8:latest    b0322c83ce26    44 GB    100% GPU     262144     4 minutes from now
```

Look for `100% GPU` — not `CPU`, not a split like `40%/60% CPU/GPU`. The server log carries
the runner's own offload summary (`ROCm*` or `CUDA*` buffers depending on the stack):

```text
load_tensors: offloaded 66/66 layers to GPU
load_tensors:        ROCm0 model buffer size = 25972.29 MiB
llama_kv_cache:      ROCm0 KV buffer size = 16384.00 MiB
```

## GPU placement

Ollama spreads a model over several GPUs when the auto-fitter cannot satisfy its safety
target on one card. To force single-GPU placement when several cards are visible, pass only
one render node (or one `--gpus '"device=N"'`), or cap the context:

```bash
-e OLLAMA_CONTEXT_LENGTH=8192      # or per-request "options":{"num_ctx":8192}
```

To force a spread across every visible GPU:

```bash
-e OLLAMA_SCHED_SPREAD=1
```

## Hardware support

Works on **AMD MI355X (gfx950, ROCm 7.2)**, single-GPU and two-GPU, and on
**NVIDIA H100 80GB (CUDA 13)**. Only the image tag and the device flag differ; every quirk
below applies to both.

> A 0.6B smoke model answers trivia poorly regardless of the serving path — use the 27B
> GGUF to judge answer quality.

## Arguments

### Docker flags

| Flag | Value used | Why |
|---|---|---|
| `--device /dev/kfd` | required | ROCm compute node; without it there is no GPU at all |
| `--device /dev/dri/renderD<N>` | one per GPU | per-GPU pinning; **use this instead of exposing the whole `/dev/dri`** |
| `--gpus '"device=<N>"'` | NVIDIA equivalent | per-GPU pinning on CUDA; `'"device=N,M"'` for two cards |
| `-v $DATA_DIR/ollama:/root/.ollama` | a disk with room | model store; a named volume lands on the root filesystem instead |
| `-v $LLAMA_CACHE:/ggufs:ro` | read-only | reuse cached GGUFs, no re-download |
| `-p 11434:11434` | default | Ollama's default port |
| `-d --name ollama_llm` | — | detached, named for `docker exec ollama ps` |

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

### Client (`inference_llm_ollama.py`)

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Ollama server host |
| `--port` | `11434` | Ollama server port |
| `--model` | `qwen3.8-27b-q8` | model name created by `ollama create` |
| `--api` | `chat` | `chat` → `/api/chat`, `generate` → `/api/generate`, `openai` → `/v1/chat/completions` |
| `--prompt` | OTel span question | user prompt |
| `--system` | concise assistant | system prompt (ignored by `generate`) |
| `--think` | off | keep the model's thinking block on — **see the quirk below** |
| `--max_tokens` | `128` | maps to `options.num_predict` / OpenAI `max_tokens` |
| `--temperature` | `0.7` | sampling temperature |
| `--top_p` | `0.95` | nucleus sampling mass |
| `--seed` | `42` | sampling seed |
| `--num_ctx` | unset | override the VRAM-derived context window |
| `--keep_alive` | `5m` | how long Ollama keeps the model resident |
| `--timeout` | `900` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/tags` probes before giving up |
| `--out` | none | write the raw JSON response |

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
raw response written to $OUTPUT_DIR/inference_llm_ollama/chat_multi_gpu.json
```

Raw JSON is written under `$OUTPUT_DIR/inference_llm_ollama/`, not into the
repo, so `git status` stays clean.

## Notes & quirks

1. **Thinking mode is on by default and consumes the token budget.** The Unsloth GGUF's
   chat template emits a reasoning block; on `/v1/chat/completions` the whole budget goes
   into the non-standard `message.reasoning` field and `content` comes back `''`, so a
   client reading `choices[0].message.content` looks broken.
   **Fix:** use the native API with `"think": false` (the client's default). There is no
   `think` flag on the OpenAI route.

2. **Ollama's default context is derived from TOTAL VRAM, not free VRAM.** On a large card
   that reserves a multi-GB KV cache and is the main reason a model that would fit one GPU
   spreads over two. Set `OLLAMA_CONTEXT_LENGTH` for predictable placement.

3. **`ollama create` copies, it does not reference.** The GGUF is duplicated into
   `/root/.ollama/models` — budget disk accordingly, and keep the store off the root
   filesystem.

4. **`PROCESSOR: 100% GPU` can coexist with a non-zero `CPU_Mapped` buffer.** All *layers*
   are on GPU; the percentage refers to layer offload, not to every byte.

5. **The container needs no GPU userspace on the host.** It bundles its own
   (`rocm_v7_2` / `cuda_v13`); only the kernel driver comes from the host.

6. **Outbound calls to ollama.com may fail on an air-gapped or proxied host, and are
   harmless.** `model show cloud cache hydration failed … context deadline exceeded`.
   Purely the model-recommendation refresh; local serving is unaffected.

7. **`available` VRAM in discovery reflects other tenants on the host**, not card capacity.

8. **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm.** Device selection here is done by which
   `renderD*` nodes are passed into the container.
