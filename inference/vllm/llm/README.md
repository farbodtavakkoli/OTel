# `inference/vllm/llm` — LLM chat/generation served by vLLM

Serves a causal LLM through vLLM's OpenAI-compatible `POST /v1/chat/completions` endpoint.
`infer_llm_vllm.py` sends either a free-form chat prompt or a prompted single-label
classification, and prints the generated text with finish reason, latency, decode rate and
token usage.

The target model is `Qwen/Qwen3.8-27B-FP8`; `Qwen/Qwen3-4B` serves alongside it as a
smaller dense control. Pick this stack when you want one engine for generation,
[embeddings](../embedding/) and [reranking](../reranker/).

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `infer_llm_vllm.py` — chat / prompted-classification client for the served endpoint.

## Setup

Engine install (ROCm container or NVIDIA pip) and the shared client venv are in
[`../README.md`](../README.md). Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # supplies HF_TOKEN for the model pull
```

GPU pinning and the gfx950 tuning knobs:

```bash
export HF_HOME=/path/to/hf_cache
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

export VLLM_ROCM_USE_AITER=0   # REQUIRED for Qwen/Qwen3.8-27B-FP8 on gfx950; otherwise
                               # every completion is corrupt at HTTP 200
export VLLM_ROCM_USE_AITER=1   # gfx950 default; correct for dense non-FP8 models (Qwen3-4B)
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides every GPU.

## Serve — AMD / ROCm

Run these inside the container. `Qwen/Qwen3-4B`, single GPU:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0
export VLLM_ROCM_USE_AITER=1
vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen3-4b
```

`Qwen/Qwen3.8-27B-FP8`, single GPU:

```bash
export HF_HOME=/path/to/hf_cache
export VLLM_ROCM_USE_AITER=0          # REQUIRED — see Notes
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen38-27b-fp8
```

`Qwen/Qwen3.8-27B-FP8`, two GPUs:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0,1
export VLLM_ROCM_USE_AITER=0
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen38-27b-fp8
```

Both models also serve at TP=2 with no flag beyond `--tensor-parallel-size 2` — no RCCL
tuning, no `--distributed-executor-backend` override. For a model that fits on one GPU,
replicate (one server per GPU behind a load balancer) instead of sharding.

## Serve — NVIDIA H100

One 80GB card serves the 27B FP8 checkpoint. `--max-num-seqs 256` is required: the text
backbone is a GDN/Mamba hybrid, each decode sequence needs a Mamba cache block, and the
default `max_num_seqs=1024` does not fit.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=<free-gpu>
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8500 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen38-27b-fp8
```

These commands are single-GPU. For tensor parallelism add `--tensor-parallel-size N`; check
the `qwen3_5` text backbone's head/GDN divisibility by N first.

## Client

```bash
python infer_llm_vllm.py --port 8000 --model qwen38-27b-fp8 --max_tokens 220 --seed 42
```

Use `--port 8500` against the H100 server. Raw equivalent:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"What is vLLM?"}],"max_tokens":128}'
```

## Arguments

Serve-side flags:

| Flag | Value | Meaning |
|---|---|---|
| `--tensor-parallel-size` | `1` / `2` | Shards weights and KV across N GPUs |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` (ROCm) / `8500` (H100) | LLM server port |
| `--served-model-name` | `qwen3-4b` / `qwen38-27b-fp8` | Alias clients pass as `model` |
| `--max-model-len` | `8192` / `32768` | Context window; lower it to cut KV footprint |
| `--max-num-seqs` | `256` | Required on one 80GB H100 (Mamba cache blocks) |
| `--gpu-memory-utilization` | `0.45` / `0.90` | VRAM fraction vLLM preallocates; default `0.9` |
| `--trust-remote-code` | required for 27B | The `qwen3_5` checkpoint ships custom modeling code |
| `--quantization` | *(auto)* | Do not pass — vLLM reads `quantization_config` and selects `fp8` |

`infer_llm_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--model` | `qwen38-27b-fp8` | Served model name (must match `--served-model-name`) |
| `--prompt` | vLLM question | User prompt for free generation |
| `--system_prompt` | `None` | Optional system message |
| `--classify` | `None` | Text to classify into one label; forces `temperature=0` |
| `--labels` | `billing,network_outage,device_setup,other` | Label set used by `--classify` |
| `--max_tokens` | `256` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.8` | Nucleus sampling top-p |
| `--seed` | `None` | Sampling seed for reproducible output |
| `--timeout` | `600.0` | HTTP timeout in seconds |

## Output

The client prints to stdout only:

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : qwen3-4b
finish_reason : length
latency       : 1.69s
decode rate   : 118.3 tok/s (200 completion tokens)
usage         : {'prompt_tokens': 25, 'total_tokens': 225, 'completion_tokens': 200}

--- response ---
<think>
Okay, the user wants me to explain what vLLM is in two sentences ...
```

Redirect server logs and client transcripts to a data volume, e.g.
`$OUTPUT_DIR/inference_llm_vllm/`. Nothing large is written into the repo.

## Notes

- **`VLLM_ROCM_USE_AITER=0` is mandatory for the FP8 27B on gfx950.** With AITER on, the
  server is healthy and fast and the generated text is corrupt. `VLLM_ROCM_USE_AITER_MOE=0`
  does not fix it — the fault is in the FP8 block-scaled GEMM, not the MoE path. AITER is
  correct for dense models such as `Qwen/Qwen3-4B`.
- **A passing `/health` says nothing about numerics.** Read real generated text as part of
  every ROCm bring-up.
- **Do not pass `--quantization fp8`.** The checkpoint is already FP8; forcing the flag
  re-quantizes it.
- **AITER JIT-builds on first launch**, so the first cold start in a container is slow;
  later launches reuse the cache.
- Benign startup noise on ROCm: the `quark_online_quant` plugin import traceback and the
  `Qwen3VLVideoProcessorInitKwargs` `min_frames`/`max_frames` lines.
