# `inference/llamacpp/llm` — llama.cpp GGUF LLM serving (Qwen3.8-27B)

## Overview & when to use

Serves a **Qwen3.8-27B GGUF** checkpoint through `llama-server`, llama.cpp's
OpenAI-compatible HTTP server, built against **ROCm 7.2 / HIP** (AMD Instinct MI355X,
gfx950) or **CUDA 13** (NVIDIA H100). The client `inference_llm_llamacpp.py` hits
`POST /v1/chat/completions`.

Use this folder for GGUF / quantized local inference from a single self-contained binary
(no python runtime, no torch), including partial CPU+GPU offload via `-ngl`. llama.cpp's
multi-GPU mode is *pipeline* (layer) parallelism and `--split-mode row` does not work on
this backend, so reach for vLLM or SGLang when you need tensor-parallel throughput.

> **Format note:** llama.cpp does **not** consume the Hugging Face FP8
> safetensors checkpoint `Qwen/Qwen3.8-27B-FP8`. It needs **GGUF**. This folder therefore
> serves `unsloth/Qwen3.8-27B-GGUF:Q8_0`, a converted artifact from the same base family —
> not the same file as the official FP8 repo.

## Build — see [`../README.md`](../README.md)

llama.cpp is a C++ build, not a pip package. **One HIP build serves all three
`inference/llamacpp/*` leaves** — the full ROCm/HIP recipe (prerequisites, cmake
flags, the mandatory `-DLLAMA_OPENSSL=ON`, arch-flag notes, NVIDIA variant) lives in
[`../README.md`](../README.md). This leaf assumes `llama.cpp/build/bin/llama-server`
exists from that recipe.

Reference revision for the ROCm recipe: commit
**`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`**, `llama-server` **0.1.2-dev (build 1)**,
ggml **0.20.2**.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                 # already done in this folder
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

`unsloth/Qwen3.8-27B-GGUF` is **not** gated, so the LLM pull needs no token. Never echo
or commit `HF_TOKEN`.

Point the model caches at a filesystem with room, not at the root filesystem:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache             # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp        # llama.cpp's own -hf cache
export OUTPUT_DIR=/path/to/outputs           # inference artifacts

export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

`LLAMA_CACHE` is the one that actually matters for `-hf`: llama.cpp keeps its own
download cache and ignores `HF_HOME` for this path. Set both. The Q8_0 GGUF is 28 GB
on disk:

```text
28G   models--unsloth--Qwen3.8-27B-GGUF/
610M  models--ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF/
319M  models--ggml-org--embeddinggemma-300M-GGUF/
```

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Serve

### Single GPU

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
# HF_HOME / LLAMA_CACHE as exported above

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 99 \
  -c 32768 \
  -lv 5 \
  --host 127.0.0.1 --port 8200 \
  --alias qwen3.8-27b-gguf
```

The first run downloads 28 GB; subsequent starts load from `LLAMA_CACHE` in a few seconds.

### Multi-GPU (layer split)

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 99 -c 32768 \
  --split-mode layer --tensor-split 1,1 \
  -lv 5 \
  --host 127.0.0.1 --port 8200 --alias qwen3.8-27b-gguf
```

`--split-mode layer` is the default whenever more than one device is visible;
`--tensor-split 1,1` makes the 50/50 ratio explicit.

## Client / smoke command

One shared venv at the software root serves all three leaves — build it from
`../requirements.txt`:

```bash
cd .. && python3 -m venv .env_llamacpp && .env_llamacpp/bin/pip install -r requirements.txt && cd llm

../.env_llamacpp/bin/python inference_llm_llamacpp.py \
  --port 8200 \
  --out $OUTPUT_DIR/inference_llm_llamacpp/chat_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-gguf",
       "messages":[{"role":"user","content":"In one sentence, what is an OpenTelemetry span?"}],
       "max_tokens":128,"seed":42}' | jq -r '.choices[0].message.content'
```

## Expected output

```text
endpoint      : http://127.0.0.1:8200/v1/chat/completions
model         : qwen3.8-27b-gguf
prompt_tokens : 72
output_tokens : 67
--- generated text ---
An OpenTelemetry span is a single unit of work in a trace, capturing a start time,
end time, attributes, events, and its relationship to other spans.
```

**GPU-residency proof** — `-ngl 99` can fall back to CPU silently, so check the offload
summary llama.cpp prints at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:a5:00.0) - 294164 MiB free
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 25972.29 MiB
```

Confirm independently with `rocm-smi` (`VRAM Total Used Memory`) and with
`rocm-smi --showpids`, which attributes that memory to the `llama-server` PID. All 66 layers
are on the GPU; the `CPU_Mapped` buffer is the token-embedding table, which llama.cpp
deliberately keeps host-side.

## Multi-GPU

With both GPUs visible the same prompt splits across them, and the generated text is
byte-identical to the single-GPU run:

```text
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 12730.50 MiB
load_tensors:        ROCm1 model buffer size = 13241.79 MiB
```

Layer split is **pipeline parallelism**: each GPU owns a contiguous slice of layers, so it
buys **capacity**, not speed. A 28 GB Q8_0 model fits many times over on one MI355X, so
**single-GPU is the correct choice here**; multi-GPU matters for a BF16 checkpoint or a far
larger model.

### `--split-mode row` fails on this backend

```text
llama_model_load: error loading model: device ROCm0 does not support split buffers
srv  llama_server: exiting due to model loading error
```

This is **not** AMD-specific: the CUDA backend (which HIP is compiled from) registers no
`ggml_backend_split_buffer_type` in this revision, so row split is unavailable on CUDA
**and** HIP. Use `--split-mode layer` (the default).

## H100 (NVIDIA, CUDA)

Mirror of the MI355X setup on **NVIDIA H100 80GB HBM3**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12. Same `llama-server`, same client, same `/v1/chat/completions` path — only
the backend flag and the model size change.

**Backend build — the only change is `-DGGML_CUDA=ON`** (full recipe in
[`../README.md`](../README.md) "NVIDIA (H100 / CUDA)"). `-DLLAMA_OPENSSL=ON` is kept —
it is vendor-neutral and required for the `-hf` HTTPS pull. Reference revision:
`llama-server` **0.2.0-dev (build 1, commit `70adb1b`)**, ggml **0.21.0**. The CUDA build
is clean, with no source patches and `CMAKE_CUDA_ARCHITECTURES=90-real` auto-detected for
Hopper.

**Model:** the example below substitutes the small same-family
**`unsloth/Qwen3-1.7B-GGUF:Q8_0`** (~1.8 GB) for the 28 GB
`unsloth/Qwen3.8-27B-GGUF:Q8_0` so the GPU path can be exercised without a long download.
The serve/client commands are identical; swap the `-hf` repo back for the production model.

```bash
cd /dev/shm/llamacpp/llama.cpp                   # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=0
# HF_HOME as exported above
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, see quirks below

./build/bin/llama-server \
  -hf unsloth/Qwen3-1.7B-GGUF:Q8_0 \
  -ngl 999 \
  -c 8192 \
  -lv 5 \
  --host 127.0.0.1 --port 8700 \
  --alias qwen3-1.7b-gguf
```

`-ngl 999` = "offload every layer" (same intent as the MI355X `-ngl 99`; either value
saturates a 29-layer model). Client:

```bash
../.env_llamacpp/bin/python inference_llm_llamacpp.py --port 8700 --model qwen3-1.7b-gguf
```

**GPU-residency check** — `-ngl` can silently fall back to CPU, so verify it.
`llama-server` startup log at `-lv 5`:

```text
llama_prepare_model_devices: using device CUDA0 (NVIDIA H100 80GB HBM3) (000c:00:00.0) - 80552 MiB free
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
load_tensors:        CUDA0 model buffer size =  1743.77 MiB
```

All **29/29 layers on the GPU**, weights in the **CUDA0** buffer; the `CPU_Mapped` buffer is
the token-embedding table, kept host-side by design — exactly as on MI355X. `system_info`
confirms the CUDA backend: `CUDA : ARCHS = 900` (Hopper). Confirm independently with
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`, which should
attribute the server's VRAM to the `llama-server` PID on the selected card.

**Generation.** Qwen3 is a *reasoning* model: the default client run produces 128
tokens of coherent `reasoning_content` and hits `finish_reason:length` before emitting
final content (an empty `content` string — a client/prompt artifact, not a GPU failure).
Appending Qwen3's `/no_think` switch gives clean final content:

```text
finish_reason : stop      output_tokens : 43
An OpenTelemetry span is a lightweight unit of execution that represents a single
operation or event in a trace, capturing the context and metadata of the call stack
during a specific period of time.
```

**Quirks on NVIDIA (all shared with MI355X):** the harmless minja
`Callee is not a function: got Undefined (hint: 'lstrip')` chat-template parse error
prints at load and does not affect generation; `-DLLAMA_OPENSSL=ON` remains mandatory
for `-hf`. If the host has **no cached GGUF** and **no cmake/ninja**, install the
toolchain into a throwaway venv (see `../README.md`) and pull the model to a tmpfs
`LLAMA_CACHE` — some network/NFS model shares reject pip/rename operations.

**Multi-GPU:** the commands above are single-GPU. A 2-GPU layer split uses the same
`--split-mode layer --tensor-split 1,1` shown earlier; `--split-mode row` fails identically
on CUDA.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Value | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | Pull GGUF straight from the Hub. Needs an SSL-enabled build |
| `-ngl N` | `99` | Layers offloaded to GPU. 99 = "all"; verify with the offload summary |
| `-c N` | `32768` | Context length |
| `--split-mode` | `layer` | `layer` (default, works) / `row` (broken on HIP) / `none` |
| `--tensor-split` | `1,1` | Ratio of layers per device |
| `--host` / `--port` | `127.0.0.1` / `8200` | Bind address |
| `--alias` | `qwen3.8-27b-gguf` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity. **Default 3 hides the offload summary** |
| `--list-devices` | — | Print visible backends and free VRAM, then exit |

### `inference_llm_llamacpp.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8200` | llama-server port |
| `--model` | `qwen3.8-27b-gguf` | Model name echoed in the request body |
| `--prompt` | OTel span question | User prompt |
| `--system` | concise assistant | System prompt |
| `--max_tokens` | `128` | Generation cap |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.95` | Nucleus sampling mass |
| `--seed` | `42` | Sampling seed |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/health` polls before giving up |
| `--out` | `None` | Write the raw JSON response here |

## Output

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_llm_llamacpp/`; redirect the server log there too. All 28 GB of
weights land under `$LLAMA_CACHE`, off the root filesystem; only the ~514 MB build tree
sits beside the repo (rebuild per `../README.md`).

## Hardware support

- **AMD Instinct MI355X (gfx950, ROCm 7.2):** HIP build works with no source patches,
  single-GPU and 2-GPU layer split.
- **NVIDIA H100 (Hopper cc 9.0, CUDA 13):** CUDA build works with no source patches.
- `--split-mode row` is unsupported on both backends.

## Notes & quirks

1. **The standard build recipe is incomplete for `-hf`.** `cmake -B build -DGGML_HIP=ON`
   compiles fine but produces a server that cannot download anything: this revision
   replaced libcurl with bundled cpp-httplib, which needs a TLS provider. Symptom:

   ```text
   get_repo_commit: error: HTTPS is not supported. Please rebuild with one of:
     -DLLAMA_BUILD_BORINGSSL=ON / -DLLAMA_BUILD_LIBRESSL=ON / -DLLAMA_OPENSSL=ON
   llama_model_load_from_file_impl: exactly one out metadata, path_model, and file must be defined
   srv  load_model: failed to load model, ''
   ```

   Note the misleading second line — it looks like a bad `-hf` argument, but the real
   cause is the HTTPS failure four lines earlier. Fix: `apt install libssl-dev` and
   rebuild with `-DLLAMA_OPENSSL=ON`. Installing `libcurl4-openssl-dev` does **not**
   help this revision.
2. **Default log verbosity hides the GPU-offload proof.** At the default `verbosity = 3`
   you get "model loaded" and nothing else — no `offloaded N/N layers`, no buffer sizes.
   Always start with `-lv 5` when you need to prove residency, then drop it.
3. **`LLAMA_CACHE`, not `HF_HOME`, controls where `-hf` writes.** Set it explicitly or
   28 GB lands in `~/.cache` on the root filesystem.
4. **`--split-mode row` is broken for CUDA and HIP** in this revision — see the source
   trace above. Only `layer` and `none` are usable.
5. **Multi-GPU does not make a single request faster.** It is pipeline parallelism. If
   you need throughput, run independent server instances (one per GPU) behind a load
   balancer, or use vLLM/SGLang tensor parallelism.
6. **No quant substitution is needed.** `unsloth/Qwen3.8-27B-GGUF:Q8_0` resolves to a
   single 28 GB `Qwen3.8-27B-Q8_0.gguf`. The alternative `UD-Q8_K_XL` is present in the same
   repo if you want the dynamic quant.
7. **`--list-devices` renumbers.** Whichever physical indices you put in
   `HIP_VISIBLE_DEVICES`, they appear as `ROCm0`, `ROCm1`, … `rocm-smi` still reports the
   original `cardN`, and `rocm-smi --showpids` reports a KFD index that matches neither —
   trust the per-card VRAM table.
