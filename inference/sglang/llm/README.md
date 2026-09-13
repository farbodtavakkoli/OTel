# `inference/sglang/llm` — SGLang OpenAI-compatible LLM serving (Qwen3.8-27B-FP8)

## Overview & when to use

Serve a generative LLM behind an OpenAI-compatible HTTP API with **SGLang**
(`python -m sglang.launch_server`), and hit it with `inference_llm_sglang.py`
(`POST /v1/chat/completions`). Target checkpoint is `Qwen/Qwen3.8-27B-FP8`.

Two routes, by vendor. On **NVIDIA** a plain pip install serves. On **AMD gfx950 a pip
install does not serve; the vendor container does** — see
"Container route — `lmsysorg/sglang-rocm`".

## H100 (NVIDIA) — the pip route

**On NVIDIA the pip route works** — no container needed. The import that fails on the ROCm pip
route, `from sgl_kernel import rotary_embedding`, succeeds because `sglang-kernel` ships
native CUDA wheels. The commands below are single-GPU.

### H100 install (pip route)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and run artifacts
```

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip
pip install torch numpy                 # -> the current CUDA 13 build (plain PyPI)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # 2.13.0+cu130 13.0 True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # sgl_kernel/flashinfer wheel hosts are commonly 403'd through a proxy
pip install "sglang[all]"               # -> sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17, flash-attn-4 4.0.0b19, transformers 5.12.1
python -c "import sgl_kernel; print('sgl_kernel OK', hasattr(sgl_kernel,'rotary_embedding'))"   # sgl_kernel OK True  <- impossible on ROCm
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130 (no clobber; no force-reinstall needed)
python -c "import sglang.srt.entrypoints.http_server; print('server import OK')"   # died on ROCm; OK here
```

**`transformers` stays at 5.12.1** — the checkpoint's `config.json` pins
`transformers_version: 5.8.0.dev0`, but SGLang has its own in-tree
`Qwen3_5ForConditionalGeneration` class, so do not chase the dev transformers.
`--trust-remote-code` is still passed.

### H100 serve — single GPU, FP8

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1   # plain CUDA — no HIP_VISIBLE_DEVICES; HF_HOME set above
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --host 127.0.0.1 --port 8600
```

Pass no `--attention-backend`: on Hopper SGLang auto-selects `fa3` with `flashinfer`
sampling. With `HF_HUB_OFFLINE=1` the ~29 GB checkpoint must already be cached in `$HF_HOME`.

**Expected output** (server log):

```
Using hybrid linear attention backend for hybrid GDN models.
The server is fired up and ready to roll!
```

### H100 client / smoke command

```bash
python inference_llm_sglang.py --port 8600 --model Qwen/Qwen3.8-27B-FP8 \
  --prompt "What is the capital of France? Answer in one word." --max_tokens 64
```

**Expected output** (coherence check):

```
[health] server ready after 1.0s
[response] Thinking: ... Retrieve knowledge: Capital of France = Paris. ...
```

### H100 GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader
```

Only the card named by `CUDA_VISIBLE_DEVICES` is touched.

### H100 quirks

- **On an 80 GB card `--max-running-requests` is auto-capped to 24** by the mamba state
  cache. To raise concurrency, pass `--mamba-ssm-dtype bfloat16` (halves state size) or lower
  `--mem-fraction-static`.
- **`torchcodec`/`libavutil` import errors at startup are benign** — SGLang probes for a video
  codec and continues without system FFmpeg. Install FFmpeg only for multimodal video input.
- **`Ignore import error when loading sglang.srt.models.sarashina2_vision …`** and similar
  (`inkling`, `mimo_v2`) — benign; unrelated model classes failing to register.

## Container route — `lmsysorg/sglang-rocm`

**This is the route that works on gfx950.**

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk** (allow
>150 GB free on the filesystem holding it).

### Start the container

```bash
docker run -d --name sglang_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HF_HOME="$HF_HOME" -e HF_HUB_OFFLINE=1 \
  -v "$HF_HOME":"$HF_HOME" \
  -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
  -v /path/to/OTel:/work \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

The `renderD*` entries are the render nodes of the two target GPUs — match them to your own
cards (`ls /dev/dri`). Only those two are passed in, so inside the container they are
`cuda:0` and `cuda:1`, and `HIP_VISIBLE_DEVICES=0` means the first passed-in GPU. Weights and
logs live on mounted volumes — **nothing large lands on `/`**.

The fullest working flag set, if the minimal one above hits a permission or memory error, is:
`--device /dev/kfd --device /dev/dri/renderD<a> --device /dev/dri/renderD<b> --group-add
video --group-add render --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
--shm-size 64G`.

```bash
# sanity check: sgl_kernel imports and both GPUs are visible
docker exec sglang_bringup python3 -c \
  "import torch, sgl_kernel; from importlib.metadata import version; \
   print(version('sglang'), torch.__version__, torch.version.hip, torch.cuda.device_count())"
```

### Serve — single GPU, FP8

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > $OUTPUT_DIR/inference_llm_sglang/llm_fp8_single.log 2>&1"
```

**Expected output** (`llm_fp8_single.log`):

```
Detected fp8 checkpoint.
INFO:     Uvicorn running on http://0.0.0.0:8100 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/llm sglang_bringup \
  python inference_llm_sglang.py --port 8100 --model Qwen/Qwen3.8-27B-FP8
```

**Expected output (single GPU):**

```
[health] server ready after 1.0s
[response] We need answer user's request: "Name the two GPU vendors ROCm and CUDA belong to, in one line." ...
</think>

ROCm: AMD; CUDA: NVIDIA.
```

The `</think>` block is Qwen3.8's reasoning trace, emitted by default — see quirks.

### Multi-GPU — `--tp 2` across both GPUs

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --tp 2 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > $OUTPUT_DIR/inference_llm_sglang/llm_fp8_tp2.log 2>&1"
```

The same client command against port 8100 works unchanged; `rocm-smi --showmemuse` shows both
cards holding memory.

The checkpoint is ~29 GB. Point `HF_HOME` at a volume with room for it — **never `/`**.

### Container-route quirks

- **Hybrid-attention memory budgeting.** `Qwen3.8-27B-FP8` loads as
  `Qwen3_5ForConditionalGeneration`, a **hybrid mamba / linear-attention** model that needs a
  per-request *mamba state cache* on top of the KV cache. A low
  `--mem-fraction-static` (e.g. `0.25`) loads the weights fine and then dies in memory planning:
  ```
  RuntimeError: Not enough GPU memory for hybrid (mamba/linear-attention) state cache.
  Computed max_mamba_cache_size=-28 (total_rest_memory=-7.96 GB, mamba_cache_per_req=146.81 MB).
  ```
  Fix with `--mem-fraction-static 0.85` **and** a bounded `--max-running-requests 32`; the
  default `max_running_requests` wants far more memory than a small static fraction leaves.
  It is not an FP8 or kernel failure — do not misdiagnose it as one.
- **Give the model the whole GPU.** An embedding or reranker server left running from a
  sibling folder can hold a large share of VRAM. Stop other SGLang servers
  (`pkill -f sglang.launch_server` inside the container) before launching the 27B.
- **Reasoning traces are on by default.** Qwen3.8 emits a `</think>`-delimited reasoning block
  before the answer, so `completion_tokens` is much larger than the visible one-line
  answer. Pass `--reasoning-parser qwen3` to have SGLang split it into a separate
  `reasoning_content` field, or instruct the model to skip thinking.
- **`HF_HUB_OFFLINE=1`** is baked into the image env, so nothing is downloaded when the models
  are already cached. Unset it for a fresh pull and point `HF_HOME` at a volume with space.
- **AITER JIT** compiles kernels on first use into `/root/.aiter/build/`. Keep one long-lived
  container (`sleep infinity` + `docker exec -d`) rather than one `docker run` per server, so
  the cache is reused.
- Benign startup noise, safe to ignore: `Ignore import error when loading
  sglang.srt.models.inkling: No module named 'cutlass'`, the same for `mimo_v2`/`torchcodec`,
  and AITER's probe `[aiter] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.`

## Install

Python 3.12. The shared [`../requirements.txt`](../requirements.txt) is the pinned set for
all three leaves.

### AMD / ROCm

```bash
python3 -m venv .env_sglang
source .env_sglang/bin/activate
pip install -U pip
# 1) ROCm torch FIRST, from the ROCm index:
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
# 2) sglang WITHOUT its CUDA dependency wall:
pip install --no-deps sglang==0.5.17
# 3) the pure-python dependency set:
pip install -r ../requirements.txt
# 4) verify torch is STILL ROCm:
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
# -> 2.11.0+rocm7.2 7.2.26015 2
```

**Why `--no-deps` is mandatory.** sglang 0.5.17 lists CUDA packages (`cuda-python`,
`flashinfer_python[cu13]`, `flash-attn-4`, `sglang-kernel`, …) and the PyPI CUDA `torch` as
*base* dependencies, and there is no `srt_hip` extra. A plain `uv pip install sglang`
therefore **overwrites ROCm torch with a CUDA wheel**. If it happens:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` here (CUDA-only build). Never `pip install aiter` — the
PyPI project of that name is an unrelated async-iterator library, **not** AMD's AITER.

All three SGLang leaves use the identical stack, so build **one** shared `.env_sglang` at the
software root (`inference/sglang/`) and activate it from `llm/`, `embedding/` and `reranker/`.

This venv cannot serve on ROCm (`aiter` and `sgl_kernel` have no ROCm wheel), so use it only
for the client; serve from the container.

### NVIDIA / CUDA (see "H100 install" above for the full variant)

```bash
pip install --upgrade pip && pip install uv
uv pip install sglang==0.5.17
```

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env` (`ln -sf ../../../dev.env dev.env`) and holds:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The client loads it via `load_dotenv("dev.env")`, so run from inside this folder.
`dev.env` is git-ignored at the repo root; never print or commit the token.

Weights must not land on `/`:

```bash
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache, on a volume with space
export HIP_VISIBLE_DEVICES=0,1          # the GPUs this job may use
export CUDA_VISIBLE_DEVICES=0,1         # never set this empty on ROCm
```

## Run

### Launch — single GPU

```bash
source .env_sglang/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HOME=/path/to/hf_cache

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.85 \
  --max-prefill-tokens 32768 \
  --host 0.0.0.0 --port 8100
```

This is the **conservative AMD** command. Do **not** copy the NVIDIA cookbook's
`--attention-backend flashinfer` / `--kv-cache-dtype fp8_e4m3` flags onto ROCm; on gfx950
SGLang already auto-selects the AMD path and logs
`Attention backend not specified. Use aiter backend by default.`

### Launch — multi-GPU, TP=2

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B-FP8 --trust-remote-code \
  --tp 2 --mem-fraction-static 0.85 --host 0.0.0.0 --port 8100
```

### Client / smoke command

```bash
python inference_llm_sglang.py --port 8100 --model Qwen/Qwen3.8-27B-FP8 \
  --prompt "Name the two GPU vendors ROCm and CUDA belong to, in one line." --max_tokens 64
```

Expected on a working build:

```
[health] server ready after <N>s
[latency] 0.xx s | prompt_tokens=.. completion_tokens=..
[response] ROCm is AMD; CUDA is NVIDIA.
```

## Where the ROCm pip route stops

Under pip on gfx950 bring-up succeeds — weights load, the KV cache allocates, TP=2 forms the
RCCL process group — and then the first forward pass dies in `sgl_kernel.rotary_embedding`,
which has no ROCm build. `--disable-cuda-graph` only moves the same call into the first
prefill, and `--disable-custom-all-reduce` (SGLang's advised fallback when AITER is absent)
does not help either. Use the container.

## FP8 on gfx950 — `Qwen/Qwen3.8-27B-FP8`

Do **not** pass `--quantization fp8` — the checkpoint is already FP8 and SGLang resolves it to
`torch.float8_e4m3fn` (OCP E4M3FN), the format gfx950 supports in hardware.

## Arguments

Client (`inference_llm_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8100` | SGLang server port (8100 = SGLang LLM in the repo port map) |
| `--model` | `Qwen/Qwen3.8-27B-FP8` | Model id as served; must match `--model-path` |
| `--prompt` | ROCm/CUDA vendor question | User message sent to the chat endpoint |
| `--system` | `You are a terse assistant.` | System message |
| `--max_tokens` | `64` | Max new tokens |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy, reproducible) |
| `--endpoint` | `/v1/chat/completions` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

Server flags that matter here:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3.8-27B-FP8` | Target checkpoint (`Qwen/Qwen3-0.6B` is a small stand-in) |
| `--tp` | `1` / `2` | Tensor-parallel size across GPUs |
| `--mem-fraction-static` | `0.85` | KV-cache/static pool fraction; auto-reduced at TP>1 |
| `--max-running-requests` | `32` | Bounds the mamba state cache — see the container quirks |
| `--trust-remote-code` | on | Required by the Qwen3.8 architecture |
| `--reasoning-parser` | `qwen3` | Splits the `</think>` trace into `reasoning_content` |
| `--tool-call-parser` | `qwen3_coder` | Tool-call parsing for this model family |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang picks `aiter`. `flashinfer` is NVIDIA-only |

## Output

The server writes no artifacts; logs land wherever you redirect them, e.g.
`$OUTPUT_DIR/inference_llm_sglang/`. Weights live in `$HF_HOME`, never on `/`. The client
prints health-wait time, latency, token usage, and the generated text to stdout only.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Status | **Works** — H100 80GB, CUDA 13.0, pip route (see the H100 section above) | **pip route blocked** — MI355X (gfx950), ROCm 7.2.4; container route works |
| Install | `uv pip install sglang` / `pip install "sglang[all]"` | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Kernels | `sglang-kernel` CUDA wheel from PyPI | `sgl_kernel` + `aiter` must be built for HIP — **no wheel published** |
| Attention | `flashinfer` / `fa3` (auto-selected on Hopper) | `aiter` (auto-selected on gfx950) |

## Notes / quirks

- **`Failed to import amdsmi`** is logged on every launch. Harmless here (SGLang falls back
  to other memory queries), but installing `amdsmi` from the ROCm tree gives SGLang proper
  AMD GPU telemetry.
- **`Ignoring corrupted tree cache file ... Permission denied`** — appears when the shared
  `$HF_HOME` has snapshot tree-cache files owned by another user. Cosmetic:
  SGLang re-reads the snapshot and logs `Found local HF snapshot ...; skipping download`.
- **`--mem-fraction-static` is auto-reduced at TP>1** (0.5 → 0.425 at TP=2). Set it
  explicitly for reproducible KV sizing.
- **KV cache is large by default on a large-VRAM card.** Lower `--mem-fraction-static` or
  set `--max-total-tokens` when sharing the box.
- **Ports** — 8100 is this repo's SGLang-LLM slot; 8101/8102 belong to the embedding and
  reranker folders. Siblings run concurrently, so do not reuse them.
- **Never set `CUDA_VISIBLE_DEVICES` empty on ROCm** — re-export both
  `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after activating the venv.
- **If containers are not an option on ROCm**, the only alternative is building `sgl-kernel`
  for HIP and AITER from source (needs hipcc + composable_kernel).
