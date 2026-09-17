# `inference/sglang/llm` — SGLang OpenAI-compatible LLM serving (Qwen3.8-27B-FP8)

Serves a generative LLM behind an OpenAI-compatible HTTP API with SGLang
(`python -m sglang.launch_server`). `inference_llm_sglang.py` waits for `/health`, posts to
`/v1/chat/completions`, and prints latency, token usage and the generated text.

Target checkpoint is `Qwen/Qwen3.8-27B-FP8` (~29 GB). Pick this stack when you want one
engine for generation, [embeddings](../embedding/) and [reranking](../reranker/).

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `inference_llm_sglang.py` — chat smoke client for the served endpoint.

## Setup

Engine install (NVIDIA pip or the ROCm container) is in [`../README.md`](../README.md).
Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # supplies HF_TOKEN for the model pull
export HF_HOME=/path/to/hf_cache   # the ~29 GB checkpoint must not land on /
export OUTPUT_DIR=/path/to/outputs # server logs
```

`transformers` stays at `5.12.1`. The checkpoint's `config.json` pins
`transformers_version: 5.8.0.dev0`, but SGLang has its own in-tree
`Qwen3_5ForConditionalGeneration` class, so do not chase the dev transformers.

## Serve — NVIDIA H100

```bash
source ../.env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --host 127.0.0.1 --port 8600
```

With `HF_HUB_OFFLINE=1` the checkpoint must already be cached in `$HF_HOME`. Server log on
success:

```
Using hybrid linear attention backend for hybrid GDN models.
The server is fired up and ready to roll!
```

## Serve — AMD / ROCm

Single GPU, from the long-lived container:

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > $OUTPUT_DIR/inference_llm_sglang/llm_fp8_single.log 2>&1"
```

Two GPUs, `--tp 2`:

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --tp 2 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > $OUTPUT_DIR/inference_llm_sglang/llm_fp8_tp2.log 2>&1"
```

Stop any sibling SGLang server before launching the 27B — an embedding or reranker server
left running holds a large share of VRAM (`pkill -f sglang.launch_server` inside the
container).

## Client

```bash
python inference_llm_sglang.py --port 8100 --model Qwen/Qwen3.8-27B-FP8 \
  --prompt "Name the two GPU vendors ROCm and CUDA belong to, in one line." --max_tokens 64
```

Use `--port 8600` against the H100 server. From inside the container:

```bash
docker exec -w /work/inference/sglang/llm sglang_bringup \
  python inference_llm_sglang.py --port 8100 --model Qwen/Qwen3.8-27B-FP8
```

## Arguments

Server flags that matter here:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3.8-27B-FP8` | Target checkpoint (`Qwen/Qwen3-0.6B` is a small stand-in) |
| `--tp` | `1` / `2` | Tensor-parallel size across GPUs |
| `--mem-fraction-static` | `0.85` | KV/static pool fraction; auto-reduced at TP>1 |
| `--max-running-requests` | `32` | Bounds the mamba state cache — see Notes |
| `--trust-remote-code` | on | Required by the Qwen3.8 architecture |
| `--reasoning-parser` | `qwen3` | Splits the `</think>` trace into `reasoning_content` |
| `--tool-call-parser` | `qwen3_coder` | Tool-call parsing for this model family |
| `--mamba-ssm-dtype` | `bfloat16` | Halves mamba state size; raises usable concurrency |
| `--attention-backend` | *unset* | Leave unset — `aiter` on gfx950, `fa3` on Hopper |
| `--quantization` | *(auto)* | Do not pass `fp8` — the checkpoint already is FP8 (`torch.float8_e4m3fn`) |

Client (`inference_llm_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8100` | Server port (8100 = SGLang LLM slot in the repo port map) |
| `--model` | `Qwen/Qwen3.8-27B-FP8` | Model id as served; must match `--model-path` |
| `--prompt` | ROCm/CUDA vendor question | User message sent to the chat endpoint |
| `--system` | `You are a terse assistant.` | System message |
| `--max_tokens` | `64` | Max new tokens |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy, reproducible) |
| `--endpoint` | `/v1/chat/completions` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

## Output

The server writes no artifacts; redirect logs where you want them, e.g.
`$OUTPUT_DIR/inference_llm_sglang/`. Weights live in `$HF_HOME`. The client prints to
stdout only:

```
[health] server ready after 1.0s
[latency] 0.82s | prompt_tokens=24 completion_tokens=64
[response] ...
</think>

ROCm: AMD; CUDA: NVIDIA.
```

## Notes

- **Set `--mem-fraction-static 0.85` together with a bounded `--max-running-requests`.**
  `Qwen3.8-27B-FP8` loads as a hybrid mamba/linear-attention model needing a per-request
  mamba state cache on top of the KV cache; a low static fraction loads the weights and
  then fails in memory planning.
- **On an 80 GB card `--max-running-requests` is auto-capped to 24** by that same cache.
  Raise concurrency with `--mamba-ssm-dtype bfloat16` or a lower `--mem-fraction-static`.
- **Reasoning traces are on by default**, so `completion_tokens` far exceeds the visible
  answer. Pass `--reasoning-parser qwen3` to split the `</think>` block into
  `reasoning_content`.
- **`--mem-fraction-static` is auto-reduced at TP>1** (0.5 → 0.425 at TP=2). Set it
  explicitly for reproducible KV sizing.
- Benign startup noise: `Failed to import amdsmi` (install `amdsmi` for AMD telemetry),
  `Ignoring corrupted tree cache file ... Permission denied` on a shared `$HF_HOME`,
  `torchcodec`/`libavutil` import errors, `Ignore import error when loading
  sglang.srt.models.{inkling,mimo_v2,sarashina2_vision}`, and AITER's
  `-amdgpu-coerce-illegal-types=1 is not supported by hipcc` probe.
