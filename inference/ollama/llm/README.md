# `inference/ollama/llm` — Ollama GGUF LLM serving (Qwen3.8-27B) on ROCm

> Stack overview, shared install route (image/digest, GPU→render-node mapping, run-line
> deviations, NVIDIA variant), environment/secrets, shared quirks, and venv conventions:
> [`../README.md`](../README.md). Client deps: [`../requirements.txt`](../requirements.txt).

## Overview & when to use

Serves a **Qwen3.8-27B GGUF** checkpoint through **Ollama**, running in the official
`ollama/ollama:rocm` container on **AMD Instinct MI355X (gfx950)** with **ROCm 7.2.4**.
The client `inference_llm_ollama.py` hits `POST /api/chat`, `POST /api/generate`, or the
OpenAI-compatible `POST /v1/chat/completions`.

Use this folder when you want:

- **The least-friction local LLM service that exists.** One `docker run`, one
  `ollama create`, and you have a managed HTTP API. No build toolchain, no ROCm install
  on the host, no python serving stack, no patches for gfx950.
- **A model appliance, not an engine.** Ollama owns model management, GPU discovery,
  automatic VRAM fitting, request queueing, and model lifetime (auto-unload after
  `keep_alive`). llama.cpp gives you a server; Ollama gives you a service.
- **Multiple models behind one endpoint.** The same daemon serves this 27B LLM and the
  300M embedder from [`../embedding/`](../embedding/) concurrently, loading each on demand.

Do **not** use this folder when you need datacenter throughput or multi-node scale.
Ollama's scheduler is **single-host by design** — it has no native multi-node tensor or
pipeline parallelism. Its multi-GPU mode is layer splitting, which buys **capacity, not
speed** (measured below: 75.9 tok/s on two GPUs vs 76.3 tok/s on one). For MI355X-class
throughput use `../../vllm/llm/` or `../../sglang/llm/` with tensor parallelism.

> **Critical format note (confirmed here):** Ollama is a **GGUF path**.
> It does **not** consume the Hugging Face FP8 safetensors checkpoint
> `Qwen/Qwen3.8-27B-FP8`. This folder therefore serves `unsloth/Qwen3.8-27B-GGUF:Q8_0`,
> a converted artifact from the same base family — not the same file as the official FP8
> repo. For the exact FP8 checkpoint use vLLM or SGLang.

## Scope note — why there is no `inference/ollama/reranker`

Deliberate, evidenced scope decision — see [`../README.md`](../README.md). In short:
Ollama's HTTP surface (`/api/generate`, `/api/chat`, `/api/embed` + the OpenAI shim) has
no rerank route, confirmed on the 0.32.14 API used here. You *could* prompt the reranker
GGUF through `/api/generate` and parse yes/no logits by hand, but that is a
re-implementation of the endpoint, not a supported path. Use
[`../../llamacpp/reranker/`](../../llamacpp/reranker/) (local/GGUF answer) or
[`../../vllm/reranker/`](../../vllm/reranker/) (production GPU-serving answer).

> **Tested topology:** 2x AMD Instinct MI355X (gfx950, 288 GB each), physical GPUs **2
> and 3**, ROCm 7.2.4, Ubuntu, Docker 29.7.2, Python 3.12.3.

## Install — the docker route

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
export OUTPUT_DIR=/path/to/outputs               # inference artifacts
```

The shared route — `docker pull ollama/ollama:rocm` (0.32.14, digest, gfx950 native, no
`HSA_OVERRIDE_GFX_VERSION`), the GPU→render-node mapping (GPU2 = `renderD144`,
GPU3 = `renderD152`), and the two deliberate run-line deviations — is documented in
[`../README.md`](../README.md). The exact serve commands used for this leaf:

### Serve — single GPU (physical GPU 2)

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

### Serve — both GPUs (physical GPUs 2 and 3)

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

### Confirm the backend sees only your GPUs

```bash
docker logs ollama_llm 2>&1 | grep "inference compute"
```

```text
... id=1 filter_id=1 library=ROCm compute=gfx950 name=ROCm1 libdirs=ollama,rocm_v7_2 pci_id=0000:dc:00.0 type=discrete total="288.0 GiB" available="75.0 GiB"
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

Two GPUs, the right two — `0000:a5:00.0` and `0000:dc:00.0`. (`available` reads only
~75 GiB of 288 GiB here because co-tenant SGLang jobs held ~214 GiB on each card.)

NVIDIA variant, environment & secrets: see [`../README.md`](../README.md). (`dev.env` is
symlinked to the repo-root file — `ln -sf ../../../dev.env dev.env` — and the client loads
it via `load_dotenv("dev.env")`; no token is actually needed on this path.)

## Model registration — reuse the cached 29 GB GGUF

The canonical route is `ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0`, which downloads
**29 GB**. The file is already on disk, so register it directly instead.

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

real    1m41.398s
```

**1 m 41 s**, no network. Note this **copies** the blob (hence the 55 GB store). The HF
snapshot symlinks (`../../blobs/<sha256>`) resolve fine inside the container because the
whole cache tree is mounted, not just the snapshot directory.

```bash
docker exec ollama_llm ollama list
```

```text
NAME                     ID              SIZE      MODIFIED
embeddinggemma:latest    b48ed6e89ad7    333 MB    About a minute ago
qwen3.8-27b-q8:latest    b0322c83ce26    29 GB     3 minutes ago
```

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

## Results — single GPU (physical GPU 2, `renderD144`)

Cold `/api/generate`, model not previously resident:

```text
http=200
wall_s=5.186057489
model: qwen3.8-27b-q8
done_reason: stop
total_duration: 5178426921  (5.18s)
load_duration: 3836789654  (3.84s)
prompt_eval_count: 64
prompt_eval_duration: 88311000  (0.09s)
eval_count: 93
eval_duration: 1249794000  (1.25s)
```

**Cold start 5.18 s wall, of which 3.84 s is model load** (29 GB from page cache — the
file had just been copied by `ollama create`; a genuinely cold page cache will be slower).
Decode **74.4 tok/s**.

**Expected generated text** (`/api/chat`, `think:false`):

```text
AMD Instinct MI100, AMD Instinct MI200, AMD Instinct MI300
```

```text
An OpenTelemetry span is a fundamental unit of work within a trace that represents a
specific operation or task, containing metadata such as its duration, parent-child
relationships, and associated attributes.
```

### GPU-residency proof — single GPU

On a 256-core box a CPU fallback still generates text, so this is the check that matters.
`ollama ps` reports a `PROCESSOR` column:

```text
NAME                     ID              SIZE     PROCESSOR    CONTEXT    UNTIL
qwen3.8-27b-q8:latest    b0322c83ce26    44 GB    100% GPU     262144     4 minutes from now
```

**`100% GPU`** — not `CPU`, not a split like `40%/60% CPU/GPU`. Corroborated by the
runner's own offload summary:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Radeon Graphics) (0000:a5:00.0) - 76402 MiB free
load_tensors: offloading output layer to GPU
load_tensors: offloading 64 repeating layers to GPU
load_tensors: offloaded 66/66 layers to GPU
load_tensors:        ROCm0 model buffer size = 25972.29 MiB
llama_kv_cache:      ROCm0 KV buffer size = 16384.00 MiB
sched_reserve:      ROCm0 compute buffer size =   378.02 MiB
```

**66/66 layers on GPU**, and every buffer is a `ROCm0` buffer. And by `rocm-smi`,
before vs after load:

```text
                       before                after              delta
GPU[2] VRAM used   230080622592 B      276607836160 B     +46527213568 B  (+43.3 GiB)
GPU[3] VRAM used   229333782528 B      229333782528 B                  0
```

**GPU[3] did not move at all** — proof that `--device /dev/dri/renderD144` really does
confine Ollama to physical GPU 2 and leaves the sibling card untouched.

## Results — multi-GPU (physical GPUs 2 and 3)

**Finding: Ollama split the model across both GPUs by default — `OLLAMA_SCHED_SPREAD=1`
was not needed.** This is worth stating precisely, because the expected outcome was the
opposite. A 29 GB model trivially fits one 288 GB MI355X, so Ollama would normally place
it on a single card. It split here because of the *actual free* VRAM: sibling SGLang jobs
held ~214 GiB on each card, leaving ~75 GiB free, and Ollama's default context length is
derived from **total** VRAM (`default_num_ctx=262144`). The resulting projected footprint
did not clear its safety target on one card, so the auto-fitter spread it.

```text
common_params_fit_impl: projected memory use with initial parameters [MiB]:
common_params_fit_impl:   - ROCm0 (AMD Radeon Graphics): 294896 total,  25744 used,  50575 free vs. target of   1024
common_params_fit_impl:   - ROCm1 (AMD Radeon Graphics): 294896 total,  26409 used,  50124 free vs. target of   1024
common_params_fit_impl: projected to use 52154 MiB of device memory vs. 152854 MiB of free device memory
common_params_fit_impl: targets for free memory can be met on all devices, no changes needed
```

### GPU-residency proof — both GPUs loaded

```text
NAME                     ID              SIZE     PROCESSOR    CONTEXT    UNTIL
qwen3.8-27b-q8:latest    b0322c83ce26    54 GB    100% GPU     262144     4 minutes from now
```

```text
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 12730.50 MiB
load_tensors:        ROCm1 model buffer size = 13241.79 MiB
llama_kv_cache:      ROCm0 KV buffer size =  8192.00 MiB
llama_kv_cache:      ROCm1 KV buffer size =  8192.00 MiB
sched_reserve:      ROCm0 compute buffer size =  4744.25 MiB
sched_reserve:      ROCm1 compute buffer size =  4744.25 MiB
```

Weights **and** KV cache are split near-evenly across the two cards. `rocm-smi` sampled
with the model loaded, against the same pre-load baseline:

```text
                       before                after              delta
GPU[2] VRAM used   230080622592 B      258356191232 B     +28275568640 B  (+26.3 GiB)
GPU[3] VRAM used   229333782528 B      258322071552 B     +28988289024 B  (+27.0 GiB)
```

**Both GPUs loaded, ~26-27 GiB each.** This is a direct two-GPU measurement, not an
inference from the logs.

**Expected generated text** on the two-GPU placement:

```text
Tensor parallelism partitions large weight matrices and intermediate activations across
multiple GPUs, allowing each device to store and process only a fraction of the total
parameters. Consequently, the memory footprint for both model weights and
forward/backward pass computations is distributed, significantly lowering the storage
requirements for each individual GPU.
```

### Single vs multi-GPU — the honest throughput answer

| Placement | Model buffers | KV cache | `ollama ps` SIZE | Decode |
|---|---|---|---|---|
| 1x MI355X (GPU 2) | ROCm0 25972 MiB | ROCm0 16384 MiB | 44 GB | **76.3 tok/s** |
| 2x MI355X (GPU 2+3) | ROCm0 12730 + ROCm1 13241 MiB | 8192 + 8192 MiB | 54 GB | **75.9 tok/s** |

**Adding a second GPU bought 0% throughput** (75.9 vs 76.3 tok/s, i.e. noise). That is
expected and correct: Ollama/llama.cpp multi-GPU is *layer* (pipeline) splitting, so only
one GPU is active at a time for a single request stream. It buys **capacity** — the
ability to hold a model or a KV cache larger than one card — not speed.

**On this hardware, single-GPU is the correct placement for this model.** A 29 GB Q8_0
fits one 288 GB MI355X with room for a 256K context; splitting it only adds
cross-device transfer. To force single-GPU placement when both cards are visible, pass
only one render node, or cap the context so the fitter is satisfied by one card:

```bash
-e OLLAMA_CONTEXT_LENGTH=8192      # or per-request "options":{"num_ctx":8192}
```

To force a spread in the opposite case (a model that *would* fit on one):

```bash
-e OLLAMA_SCHED_SPREAD=1
```

The genuine multi-GPU win here is **concurrency, not tensor parallelism** — and
Ollama does that natively. Both models resident at once, each `100% GPU`:

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
qwen3.8-27b-q8:latest    b0322c83ce26    54 GB     100% GPU     262144     4 minutes from now
```

## H100 (NVIDIA)

Single-GPU smoke on **NVIDIA H100 80GB HBM3** (Hopper cc 9.0, driver 580.173.02, CUDA 13.0),
physical **GPU 4 only**. Container/run-line deviations and the GGUF-download note are in
[`../README.md`](../README.md).

**Model:** when the 29 GB `unsloth/Qwen3.8-27B-GGUF:Q8_0` used in the MI355X sections above
is **not cached** (the Modelfile's `FROM` path under `$LLAMA_CACHE` is absent), serve
**`unsloth/Qwen3-0.6B-GGUF:Q8_0`** for a fast smoke — same Qwen3
family, same Q8_0 quant, ~639 MB. The 27B also fits an 80 GB H100 (weights ~26 GB + KV) and
would follow the identical path; it is skipped here only to keep the pull short.

### Exact commands

```bash
# GGUF (unset the proxy if HF is proxy-blocked on your host):
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir $GGUF/Qwen3-0.6B-GGUF

# H100 Modelfile variant (points at the CONTAINER path under the :ro /ggufs mount;
# the MI355X Modelfile.qwen3-27b is left untouched):
#   FROM /ggufs/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
sudo docker cp Modelfile.qwen3-0.6b.h100 ollama_h100:/root/Modelfile.qwen3-0.6b
sudo docker exec ollama_h100 ollama create qwen3-0.6b-q8 -f /root/Modelfile.qwen3-0.6b

# client (venv on tmpfs; requests only):
python inference_llm_ollama.py --port 11440 --model qwen3-0.6b-q8 --api chat \
  --prompt "What is the capital of France? Answer in one short sentence." --max_tokens 64
```

### Expected output (coherent — the check that matters)

```text
endpoint      : http://127.0.0.1:11440/api/chat
model         : qwen3-0.6b-q8
--- generated text ---
The capital of France is Paris.
```

Warm `/api/generate` ("Name three primary colors.") → `Three primary colors are **red, blue,
and yellow**.` at **374.5 tok/s** (`load_s: 0.00`). (Cold start is ~26 s load + 4.8 tok/s for
the *first* call — that is model load + first-token latency, not CPU: the warm decode rate and
the offload log below prove GPU execution. Caveat: a 0.6B model is weak on trivia — asked for
Japan's capital it answers "Osaka"; the *serving path* is correct, the tiny model is the
limitation. Use the 27B GGUF for answer quality.)

### GPU-residency proof — `100% GPU` + nvidia-smi on GPU 4

On a many-core box a CPU fallback still generates text, so this is the check that matters.

```text
# docker exec ollama_h100 ollama ps
NAME                    ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3-0.6b-q8:latest    605b58ae76ea    5.6 GB    100% GPU     40960      4 minutes from now
```

`100% GPU` — not `CPU`, not a split. Corroborated by the runner's offload summary (every
buffer a `CUDA0` buffer):

```text
llama_prepare_model_devices: using device CUDA0 (NVIDIA H100 80GB HBM3) (0009:00:00.0) - 80552 MiB free
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =   604.15 MiB
llama_kv_cache:      CUDA0 KV buffer size =  4480.00 MiB
sched_reserve:      CUDA0 compute buffer size =   264.04 MiB
```

And by `nvidia-smi -i 4` — VRAM held by the ollama runner on physical GPU 4 (0 MiB
before load):

```text
# nvidia-smi -i 4 --query-compute-apps=pid,process_name,used_memory --format=csv
<pid>, /usr/lib/ollama/llama-server, 5942 MiB
```

**29/29 layers on GPU, 5942 MiB resident on GPU 4** — only the pinned card is touched.

### Single vs multi-GPU on H100

Single-GPU only here. The MI355X finding carries over unchanged and is if anything *more* true on an 80 GB
card: Ollama/llama.cpp multi-GPU is **layer (pipeline) splitting = capacity, not throughput**,
so a model that fits one card should stay on one card. A multi-GPU pass would just add
`--gpus '"device=4,5"'` (or set `OLLAMA_SCHED_SPREAD=1` to force a spread) and re-measure per
card with `nvidia-smi`. The genuinely useful multi-GPU pattern here is **concurrency** — both
models resident on one card at once, both `100% GPU`, demonstrated in [`../README.md`](../README.md).

### H100 summary

`ollama/ollama` (0.32.15, CUDA-13 userspace) serves a Qwen3 GGUF on H100 out of the
box — no build, no patch, no override. `29/29` layers on GPU, `PROCESSOR: 100% GPU`, 5942 MiB
on GPU 4 by `nvidia-smi`, 374.5 tok/s warm decode, coherent output. The only deviation from
the MI355X recipe is the container image tag + `--gpus` pinning (both expected) and using a
small GGUF for a fast smoke when no large GGUF is cached. For answer quality and for
throughput, use the 27B GGUF here, or `../../vllm/llm` / `../../sglang/llm` with the FP8
checkpoint and tensor parallelism.

## Arguments

### Docker flags

| Flag | Value used | Why |
|---|---|---|
| `--device /dev/kfd` | required | ROCm compute node; without it there is no GPU at all |
| `--device /dev/dri/renderD144` | GPU 2 | per-GPU pinning; **use this instead of exposing the whole `/dev/dri`** |
| `--device /dev/dri/renderD152` | GPU 3 | second GPU for the multi-GPU run |
| `-v $DATA_DIR/ollama:/root/.ollama` | a disk with room | model store; a named volume lands on the root filesystem instead |
| `-v $LLAMA_CACHE:/ggufs:ro` | read-only | reuse cached GGUFs, no re-download |
| `-p 11434:11434` | default | Ollama's default port; a common benchmark layout suggests 8300 — this folder uses **11434** |
| `-d --name ollama_llm` | — | detached, named for `docker exec ollama ps` |

### Server environment (`-e`)

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_SCHED_SPREAD` | `false` | `1` forces a model across every visible GPU |
| `OLLAMA_CONTEXT_LENGTH` | `0` (derive from total VRAM → 262144 here) | caps context; the single biggest VRAM lever |
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

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| ROCm backend selected, gfx950 native | `library=ROCm compute=gfx950 libdirs=ollama,rocm_v7_2` |
| No `HSA_OVERRIDE_GFX_VERSION` needed | server config shows `HSA_OVERRIDE_GFX_VERSION:` empty; GPU still detected |
| Only GPUs 2+3 visible | discovery lists exactly `pci_id=0000:a5:00.0` and `0000:dc:00.0` |
| Model is on the GPU, not the CPU | `ollama ps` → `PROCESSOR = 100% GPU`; `offloaded 66/66 layers to GPU` |
| Single-GPU placement measured | `rocm-smi` GPU[2] +46.5 GB, GPU[3] **unchanged** |
| Multi-GPU placement measured | `rocm-smi` GPU[2] +28.3 GB **and** GPU[3] +29.0 GB |
| Generation is real, not a stub | 74–76 tok/s decode with coherent text, `done_reason: stop` |

## Notes & quirks

1. **Thinking mode is on by default and will eat your token budget.** The Unsloth GGUF
   carries a chat template that emits a reasoning block. Against
   `/v1/chat/completions` with `max_tokens: 400`, the result was
   `finish_reason: length`, `completion_tokens: 400`, and **`content: ''`** — the entire
   budget goes into `message.reasoning`, which the OpenAI shim exposes as a *separate,
   non-standard* field. A client that reads only `choices[0].message.content` sees an
   empty string and looks broken.
   **Fix:** use the native API with `"think": false` (the client's default), which
   returns clean text in 24 tokens. There is no `think` flag on the OpenAI route.

2. **Ollama's default context is derived from TOTAL VRAM, not free VRAM.**
   `msg="vram-based default context" total_vram="288.0 GiB" default_num_ctx=262144`.
   That reserves a 16 GB KV cache and is the main reason the model spreads over two GPUs.
   Set `OLLAMA_CONTEXT_LENGTH` if you want predictable placement.

3. **`ollama create` copies, it does not reference.** The 29 GB GGUF is duplicated into
   `/root/.ollama/models`, so the store reaches 55 GB. Budget disk accordingly — and this
   is exactly why the store must not live on the root filesystem.

4. **`PROCESSOR: 100% GPU` can coexist with a non-zero `CPU_Mapped` buffer.** The
   two-GPU run shows `CPU_Mapped model buffer size = 1288.28 MiB` (token embeddings)
   while still reporting `100% GPU`. All 66 *layers* are on GPU; the percentage refers to
   layer offload, not to every byte.

5. **The container needs no ROCm on the host.** It bundles `rocm_v7_2`. Host ROCm 7.2.4
   is present here but is not used by the container.

6. **Outbound calls to ollama.com may fail on an air-gapped or proxied host, and are
   harmless.** `model show cloud cache hydration failed … context deadline exceeded`.
   Purely the model-recommendation refresh; local serving is unaffected.

7. **`available` VRAM reflects other tenants.** Discovery reported 74.8/75.0 GiB free of
   288 GiB because co-tenant SGLang jobs held ~214 GiB per card. All deltas in this
   README are measured against that live baseline.

8. **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm.** Ollama's own config echo shows
   `CUDA_VISIBLE_DEVICES:` and `HIP_VISIBLE_DEVICES:` empty, which means "unset" — device
   selection here is done by which `renderD*` nodes are passed into the container.

## Summary

**✅ Ollama on ROCm/MI355X is fully working for GGUF LLM serving, single-GPU and
multi-GPU, with proven GPU residency.**

- `ollama/ollama:rocm` (`0.32.14`) runs gfx950 **out of the box** — no build, no patch, no
  `HSA_OVERRIDE_GFX_VERSION`. Cheapest install in the whole repo: a 1.43 GB pull.
- A cached 29 GB GGUF registers via `Modelfile` in **1 m 41 s with zero network**.
- **Single GPU:** `100% GPU`, 66/66 layers, +46.5 GB on GPU[2] and **nothing** on GPU[3],
  74.4 tok/s, 5.18 s cold start.
- **Multi-GPU:** split automatically across both cards (+28.3 GB / +29.0 GB measured),
  but **0% faster** — layer-parallel splitting is capacity, not throughput. **Single-GPU
  is the correct placement for a 29 GB model on a 288 GB card**; the useful multi-GPU
  pattern here is concurrent model instances, demonstrated above.
- **Not the throughput answer.** For MI355X-class serving use `inference/vllm/llm` /
  `inference/sglang/llm` with the native FP8 checkpoint and tensor parallelism. Ollama is
  the "make this run locally with the least friction" answer, and at that it is excellent.
- **No reranker route exists** — see the scope note above.
