# `inference/llamacpp/llm` — llama.cpp GGUF LLM serving (Qwen3.8-27B)

Serves `unsloth/Qwen3.8-27B-GGUF:Q8_0` through `llama-server` and hits
`POST /v1/chat/completions` with `inference_llm_llamacpp.py`.

Pick this folder for GGUF/quantized local inference from a single self-contained binary,
including partial CPU+GPU offload via `-ngl`. llama.cpp's multi-GPU mode is pipeline (layer)
parallelism, so use [`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/) when you need tensor-parallel throughput. The
Hugging Face FP8 checkpoint `Qwen/Qwen3.8-27B-FP8` is not usable here — llama.cpp needs the
converted GGUF.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2) and NVIDIA H100 (Hopper cc 9.0, CUDA
13), Python 3.12.

## Files

- `inference_llm_llamacpp.py` — chat-completions smoke client; polls `/health`, prints
  latency, token counts and the generated text.

## Setup

Build `llama-server` and create the shared venv per [`../README.md`](../README.md) — one
build and one venv serve all three leaves. Then:

```bash
ln -sf ../../../dev.env dev.env                 # only needed for gated repos
export HF_HOME=/path/to/hf_cache                # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp           # llama.cpp's own -hf cache (28 GB for this model)
export OUTPUT_DIR=/path/to/outputs              # inference artifacts
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

`unsloth/Qwen3.8-27B-GGUF` is ungated, so no `HF_TOKEN` is required.

## Run

### Serve — single GPU

```bash
cd <your llama.cpp checkout>
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 99 \
  -c 32768 \
  -lv 5 \
  --host 127.0.0.1 --port 8200 \
  --alias qwen3.8-27b-gguf
```

The first run downloads 28 GB; cached restarts take seconds.

### Serve — multi-GPU (layer split)

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 99 -c 32768 \
  --split-mode layer --tensor-split 1,1 \
  -lv 5 \
  --host 127.0.0.1 --port 8200 --alias qwen3.8-27b-gguf
```

`--split-mode layer` is already the default when several devices are visible;
`--tensor-split 1,1` just makes the 50/50 ratio explicit. A 28 GB Q8_0 model fits many times
over on one MI355X, so prefer single-GPU here.

### Client

```bash
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

### NVIDIA / CUDA

Identical serve and client commands; only the build flag differs (`-DGGML_CUDA=ON`, see
[`../README.md`](../README.md)). To exercise the GPU path without the 28 GB download,
substitute the same-family `unsloth/Qwen3-1.7B-GGUF:Q8_0` (~1.8 GB):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # for the HF pull
export CUDA_VISIBLE_DEVICES=0
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, if your model share rejects renames

./build/bin/llama-server \
  -hf unsloth/Qwen3-1.7B-GGUF:Q8_0 \
  -ngl 999 -c 8192 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias qwen3-1.7b-gguf

../.env_llamacpp/bin/python inference_llm_llamacpp.py --port 8700 --model qwen3-1.7b-gguf
```

`-ngl 999` and `-ngl 99` both mean "offload every layer".

## Output

Client output:

```text
endpoint      : http://127.0.0.1:8200/v1/chat/completions
model         : qwen3.8-27b-gguf
prompt_tokens : 72
output_tokens : 67
--- generated text ---
An OpenTelemetry span is a single unit of work in a trace, capturing a start time,
end time, attributes, events, and its relationship to other spans.
```

`-ngl 99` can fall back to CPU silently, so check the offload summary printed at `-lv 5`:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI355X) (0001:a5:00.0) - 294164 MiB free
load_tensors: offloaded 66/66 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:        ROCm0 model buffer size = 25972.29 MiB
```

All 66 layers must be on the GPU; the `CPU_Mapped` buffer is the token-embedding table,
which llama.cpp deliberately keeps host-side. Confirm independently with `rocm-smi` and
`rocm-smi --showpids` (or
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` on CUDA), which
attribute the VRAM to the `llama-server` PID. Note that `--list-devices` renumbers devices
from 0 while `rocm-smi` reports the original `cardN` — trust the per-card VRAM table.

The client writes raw JSON wherever `--out` points; weights stay under `$LLAMA_CACHE`.

## Arguments

### `llama-server`

| Argument | Value | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | Pull GGUF from the Hub; needs an SSL-enabled build |
| `-ngl N` | `99` | Layers offloaded to GPU; 99 = all |
| `-c N` | `32768` | Context length |
| `--split-mode` | `layer` | `layer` (default) or `none`; `row` is unsupported |
| `--tensor-split` | `1,1` | Ratio of layers per device |
| `--host` / `--port` | `127.0.0.1` / `8200` | Bind address |
| `--alias` | `qwen3.8-27b-gguf` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity; default 3 hides the offload summary |
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

## Notes

- Qwen3 is a reasoning model: on the default run it can spend the whole 128-token budget on
  the reasoning block and return empty `content` with `finish_reason:length`. Append
  Qwen3's `/no_think` switch to the prompt, or raise `--max_tokens`, for clean final text.
- `unsloth/Qwen3.8-27B-GGUF:Q8_0` resolves to a single 28 GB `Qwen3.8-27B-Q8_0.gguf`. The
  same repo also carries the dynamic `UD-Q8_K_XL` quant if you want it.
- A minja `Callee is not a function: got Undefined (hint: 'lstrip')` chat-template error
  prints at load on both backends and does not affect generation.
- Multi-GPU does not make a single request faster. For throughput, run one server instance
  per GPU behind a load balancer, or use vLLM/SGLang tensor parallelism.
