# `inference/llamacpp/llm` — llama.cpp GGUF LLM serving (Qwen3.8-27B) on ROCm/HIP

## Overview & when to use

Serves a **Qwen3.8-27B GGUF** checkpoint through `llama-server`, llama.cpp's
OpenAI-compatible HTTP server, built against **ROCm 7.2.4 / HIP** for
**AMD Instinct MI355X (gfx950)**. The client `inference_llm_llamacpp.py` hits
`POST /v1/chat/completions`.

Use this folder when you want:

- **GGUF / quantized local inference** — Q8_0, and the whole Unsloth dynamic-quant
  ladder (`UD-Q8_K_XL` down to `UD-IQ1_S`) from one repository, with no conversion step.
- **A single self-contained binary** — no python runtime, no torch, no vLLM/SGLang
  install surface. The server is one ~500 MB build directory.
- **CPU+GPU offload** — `-ngl` lets you place only part of the model on the GPU, which
  is what makes llama.cpp the right answer on memory-constrained or consumer hardware.

Do **not** use this folder if you want maximum multi-GPU throughput on datacenter
parts. See "Single vs multi-GPU" below: llama.cpp's multi-GPU mode is *pipeline*
(layer) parallelism, which buys you capacity, not speed, and `--split-mode row` does
not work on this backend at all. On 8xMI355X with 288 GB per GPU, vLLM or SGLang with
tensor parallelism is the throughput answer; llama.cpp is the portability/GGUF answer.

> **Critical format note (confirmed here):** llama.cpp does **not**
> consume the Hugging Face FP8 safetensors checkpoint `Qwen/Qwen3.8-27B-FP8`. It needs
> **GGUF**. This folder therefore serves `unsloth/Qwen3.8-27B-GGUF:Q8_0`, a converted
> artifact from the same base family — not the same file as the official FP8 repo.

> **Tested topology:** 2xAMD Instinct MI355X (gfx950, 288 GB each), physical GPUs 6
> and 7, ROCm 7.2.4, Ubuntu, Python 3.12.3.

## Build — see [`../README.md`](../README.md)

llama.cpp is a C++ build, not a pip package. **One HIP build serves all three
`inference/llamacpp/*` leaves** — the full ROCm/HIP recipe (prerequisites, cmake
flags, the mandatory `-DLLAMA_OPENSSL=ON`, arch-flag notes, NVIDIA variant) lives in
[`../README.md`](../README.md); a cold build takes about 40.6 s. This leaf assumes
`llama.cpp/build/bin/llama-server` exists from that recipe.

Verified at commit **`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`** (Wed Aug 19 2026),
`llama-server` version **0.1.2-dev (build 1)**, ggml version **0.20.2**.

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

export HIP_VISIBLE_DEVICES=6,7 CUDA_VISIBLE_DEVICES=6,7
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

### Single GPU (physical GPU 6)

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6
# HF_HOME / LLAMA_CACHE as exported above

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 99 \
  -c 32768 \
  -lv 5 \
  --host 127.0.0.1 --port 8200 \
  --alias qwen3.8-27b-gguf
```

First run downloads 28 GB; **download + load took 5 m 38 s** (~88 MB/s). Subsequent
starts load from `LLAMA_CACHE` in ~3 s.

### Multi-GPU (physical GPUs 6 + 7, layer split)

```bash
export HIP_VISIBLE_DEVICES=6,7 CUDA_VISIBLE_DEVICES=6,7

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

## Results — single GPU

**Expected client output**

```text
endpoint      : http://127.0.0.1:8200/v1/chat/completions
model         : qwen3.8-27b-gguf
latency_s     : 1.00
prompt_tokens : 72
output_tokens : 67
tok_per_s     : 66.7
--- generated text ---
An OpenTelemetry span is a single unit of work in a trace, capturing a start time,
end time, attributes, events, and its relationship to other spans.
```

**GPU-residency proof** — `-ngl 99` silently falling back to CPU is the classic trap,
so here is the offload summary llama.cpp printed at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:a5:00.0) - 294164 MiB free
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 25972.29 MiB
```

Independently confirmed by `rocm-smi` (`VRAM Total Used Memory`, bytes):

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card6,309220868096,33158815744     <-- 33.16 GB, the LLM
card7,309220868096,9599221760      <-- embedding + reranker servers (other folders)
```

`rocm-smi --showpids` attributes 32,815,108,096 B to the `llama-server` PID. All 66
layers are on the GPU; the 1288 MiB `CPU_Mapped` buffer is the token-embedding table,
which llama.cpp deliberately keeps host-side.

## Results — multi-GPU

Same prompt, same seed, split across both GPUs:

```text
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 12730.50 MiB
load_tensors:        ROCm1 model buffer size = 13241.79 MiB
```

`rocm-smi` with the model split (card7 also carries this repo's embedding + reranker
servers, ~9.6 GB, hence its larger total):

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card6,309220868096,17838854144     <-- 17.84 GB
card7,309220868096,26447986688     <-- 26.45 GB (incl. 9.6 GB of other servers)
```

Client output — byte-identical text to the single-GPU run:

```text
latency_s     : 1.02
output_tokens : 67
tok_per_s     : 65.9
An OpenTelemetry span is a single unit of work in a trace, capturing a start time,
end time, attributes, events, and its relationship to other spans.
```

### Single vs multi-GPU — the honest reading

| Mode | Model buffers | tok/s | Result |
|---|---|---|---|
| 1xMI355X, `-ngl 99` | ROCm0 25972 MiB | **66.7** | Best latency |
| 2xMI355X, `--split-mode layer` | ROCm0 12730 + ROCm1 13242 MiB | 65.9 | Works, ~1 % *slower* |
| 2xMI355X, `--split-mode row` | — | — | **FAILS**, see below |

Layer split is **pipeline parallelism**: each GPU owns a contiguous slice of layers and
they run in sequence for one request, so a single-stream generation gains nothing and
pays a small inter-device transfer cost. It buys **capacity**, not speed. On MI355X
(288 GB) a 28 GB Q8_0 model fits ~10x over on one GPU, so **single-GPU is the correct
production choice here**; multi-GPU only becomes interesting for a BF16 checkpoint or a
far larger model.

### `--split-mode row` fails on this backend

```text
llama_model_load: error loading model: device ROCm0 does not support split buffers
srv  llama_server: exiting due to model loading error
```

Root cause traced, and it is **not** a gfx950 or AMD-specific defect. The check in
`src/llama-model.cpp:1001` throws when the backend registers no
`ggml_backend_split_buffer_type` function. In this revision (ggml 0.20.2), grepping
`ggml/src/ggml-cuda/*.cu` finds **no `split_buffer_type` implementation at all** — only
`ggml-sycl` declares one (`ggml/include/ggml-sycl.h:28`). Since the HIP backend is the
CUDA backend compiled through hipcc, row split is unavailable to CUDA **and** HIP in
this build. Use `--split-mode layer` (the default).

## H100 (NVIDIA, CUDA)

Mirror of the MI355X run above, on **1x NVIDIA H100 80GB HBM3** (physical GPU 7,
`CUDA_VISIBLE_DEVICES=7`), driver **580.173.02**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12.3. Same `llama-server`, same client, same `/v1/chat/completions` path — only
the backend flag and the model size changed.

**Backend build — the only change is `-DGGML_CUDA=ON`** (full recipe in
[`../README.md`](../README.md) "NVIDIA (H100 / CUDA)"). `-DLLAMA_OPENSSL=ON` is kept —
it is vendor-neutral and required for the `-hf` HTTPS pull. Verified `llama-server`
**0.2.0-dev (build 1, commit `70adb1b`)**, ggml **0.21.0**. Clean CUDA build, no source
patches, **100 s** wall (`-j 32`), `CMAKE_CUDA_ARCHITECTURES=90-real` auto-detected for
Hopper.

**Model:** the documented 28 GB `unsloth/Qwen3.8-27B-GGUF:Q8_0` is substituted with the
small same-family **`unsloth/Qwen3-1.7B-GGUF:Q8_0`** (~1.8 GB) for a time-boxed
single-GPU smoke test — an 80 GB H100 fits the 27B ~2.5x over, so the 27B is not a
capacity problem here, only a download-time one. The serve/client commands are identical;
swap the `-hf` repo back to `unsloth/Qwen3.8-27B-GGUF:Q8_0` for the production model.

```bash
cd /dev/shm/llamacpp/llama.cpp                   # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=7
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

**GPU-residency proof (the #1 thing to verify — `-ngl` can silently fall back to CPU).**
`llama-server` startup log at `-lv 5`:

```text
llama_prepare_model_devices: using device CUDA0 (NVIDIA H100 80GB HBM3) (000c:00:00.0) - 80552 MiB free
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
load_tensors:        CUDA0 model buffer size =  1743.77 MiB
```

All **29/29 layers on the GPU**, weights in the **CUDA0** buffer (1743.77 MiB); the
315 MiB `CPU_Mapped` buffer is the token-embedding table, kept host-side by design —
exactly as on MI355X. `system_info` confirms the CUDA backend: `CUDA : ARCHS = 900`
(Hopper). Independently confirmed by `nvidia-smi` filtered to GPU 7, attributed by PID:

```text
$ nvidia-smi -i 7 --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
<pid>, ./build/bin/llama-server, 3288 MiB
```

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

Throughput is **370.7 tok/s** on the 1.7B (vs 66.7 tok/s for the 27B on MI355X — a
smaller model, not a hardware comparison).

**Quirks seen on H100 (all shared with MI355X):** the harmless minja
`Callee is not a function: got Undefined (hint: 'lstrip')` chat-template parse error
prints at load and does not affect generation; `-DLLAMA_OPENSSL=ON` remains mandatory
for `-hf`. If the host has **no cached GGUF** and **no cmake/ninja**, install the
toolchain into a throwaway venv (see `../README.md`) and pull the model to a tmpfs
`LLAMA_CACHE` — some network/NFS model shares reject pip/rename operations.

**Multi-GPU:** only GPU 7 is used in this single-GPU smoke. A 2-GPU layer split would
use the same `--split-mode layer --tensor-split 1,1` shown above; `--split-mode row` is
expected to fail identically on CUDA (the missing `split_buffer_type` is in the shared
CUDA/HIP backend, not AMD-specific).

**On H100 this path works.** Builds clean with `-DGGML_CUDA=ON` in 100 s, serves a Qwen3
GGUF with all layers on the H100 (CUDA0 buffer + `nvidia-smi` by PID), coherent output.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Used | Meaning |
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

Logs and artifacts go to `$OUTPUT_DIR/inference_llm_llamacpp/`, never to the root
filesystem:

```text
build.log                     cold HIP build (681 targets, EXIT=0)
build_ssl.log                 incremental rebuild with -DLLAMA_OPENSSL=ON
server_single_gpu_verbose.log offload summary, single GPU
server_multi_gpu.log          offload summary, layer split across 2 GPUs
server_multi_gpu_row.log      the --split-mode row failure
rocm_smi_single_gpu.csv       per-card VRAM, single GPU
rocm_smi_multi_gpu.csv        per-card VRAM, layer split
chat_single_gpu.json          raw /v1/chat/completions response
chat_multi_gpu.json           raw response from the 2-GPU server
client_single_gpu.txt         client stdout
client_multi_gpu.txt          client stdout
df_before.txt / df_after.txt  disk check around the 28 GB pull
```

All 28 GB of weights land under `$LLAMA_CACHE`, off the root filesystem; only the
~514 MB build tree sits under the repo (rebuild per `../README.md`).

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| HIP build works on gfx950 | `build.log` EXIT=0, 681/681 targets, no patches |
| Build is genuinely cold | `ccache -s`: 555/559 misses (99.28 %) |
| GPUs detected | `--list-devices` → `ROCm0`/`ROCm1` MI355X, 294896 MiB each |
| Model really on GPU | `offloaded 66/66 layers to GPU`, `ROCm0 model buffer size = 25972.29 MiB` |
| VRAM occupied | `rocm-smi`: card6 33.16 GB vs 0.3 GB idle baseline |
| Generation is real | 67 tokens, coherent OTel answer, 66.7 tok/s |
| Multi-GPU split real | `ROCm0 12730.50 MiB` + `ROCm1 13241.79 MiB`, both cards in rocm-smi |
| Row split unsupported | `device ROCm0 does not support split buffers` + source trace |

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
6. **Quant substitution was not needed.** `unsloth/Qwen3.8-27B-GGUF:Q8_0` resolved to a
   single 28 GB `Qwen3.8-27B-Q8_0.gguf` and fits one MI355X ~10x over. The
   alternative `UD-Q8_K_XL` is present in the same repo if you want the dynamic quant.
7. **`--list-devices` renumbers.** With `HIP_VISIBLE_DEVICES=6,7`, physical GPUs 6 and 7
   appear as `ROCm0` and `ROCm1`. `rocm-smi` still reports them as `card6`/`card7`, and
   `rocm-smi --showpids` reports a KFD index that matches neither — trust the per-card
   VRAM table.

## Summary

**Fully working on MI355X (gfx950), single and multi-GPU.**

llama.cpp builds clean against ROCm 7.2.4 for gfx950 in **40.6 seconds** with
`-DGGML_HIP=ON -DGPU_TARGETS=gfx950`, no source patches. It serves
`unsloth/Qwen3.8-27B-GGUF:Q8_0` with **all 66/66 layers on the GPU** (25972 MiB of
ROCm0 buffer, 33.16 GB VRAM in `rocm-smi`) at **66.7 tok/s** single-GPU, and splits
across two GPUs with `--split-mode layer` producing identical output at 65.9 tok/s.

Two caveats, both documented above and neither fatal: the build needs
**`-DLLAMA_OPENSSL=ON` + `libssl-dev`** for the `-hf` downloader (easily missed),
and **`--split-mode row` is unavailable on the CUDA/HIP backend** in this revision.

For this hardware, prefer **single-GPU llama.cpp instances** — one per GPU — over the
layer split. Reach for vLLM/SGLang when you need tensor-parallel throughput or the
native FP8 checkpoint.

## Follow-ups

- Benchmark `llama-bench` for prompt-processing vs token-generation throughput on
  gfx950; this README reports only end-to-end server latency on a 72-token prompt.
- Compare `Q8_0` against `UD-Q8_K_XL` for quality/latency now that VRAM is a non-issue.
- Try `--parallel N` with a batched client to measure sustained server throughput; the
  single-request numbers here understate what one MI355X can do.
- Watch upstream for a `split_buffer_type` implementation in the CUDA/HIP backend to
  re-enable `--split-mode row`.
