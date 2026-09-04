# `inference/sglang/llm` — SGLang OpenAI-compatible LLM serving (Qwen3.8-27B-FP8)

## Overview & when to use

Serve a generative LLM behind an OpenAI-compatible HTTP API with **SGLang**
(`python -m sglang.launch_server`), and hit it with `inference_llm_sglang.py`
(`POST /v1/chat/completions`). Target checkpoint is `Qwen/Qwen3.8-27B-FP8`, the official
blockwise-FP8 dense Qwen3.8-27B; SGLang is the high-throughput alternative to vLLM for
generation and prompted classification.

Use this folder on **NVIDIA**, where SGLang has an exact validated Qwen3.8-27B cookbook.
On **AMD gfx950 the split is: a pip install does NOT serve, the vendor container DOES** —
both were tested here. The `lmsysorg/sglang-rocm` container generates from the FP8 27B
checkpoint at **TP=1 and TP=2** (`/v1/chat/completions` 200 OK, correct answer); see
"Container route — `lmsysorg/sglang-rocm` (TESTED)". The pip analysis below is retained
because it explains *why* the wheel route cannot work on ROCm.

> **Tested topology:** 2×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
> Python 3.12.3, physical GPUs 2 and 3, single node. No NVIDIA GPU exists on this host,
> so every NVIDIA statement below is upstream documentation, not a local measurement.

## VERDICT

**🟢 WORKS on ROCm/gfx950 via the vendor container — pip route stays blocked.**
`Qwen/Qwen3.8-27B-FP8` loads **natively as FP8 (E4M3)** and generates real text on MI355X
inside `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`, at both TP=1 and TP=2.
The blocker documented below is entirely a **pip packaging** gap — not a model, FP8, or
hardware gap. See "Container route — `lmsysorg/sglang-rocm` (TESTED)".

| Question | Answer (measured here) |
|---|---|
| Does the container route serve on gfx950? | **Yes** — FP8 27B generates at TP=1 **and** TP=2 |
| Does `pip install sglang` work on ROCm? | **No.** CUDA-only packages are *hard* dependencies and would replace ROCm torch |
| Does SGLang import + detect ROCm? | **Yes** — "Attention backend not specified. Use **aiter** backend by default" |
| Does the model architecture load? | **Yes** — `Qwen3_5ForConditionalGeneration` is registered in sglang 0.5.17 |
| Do weights load / KV cache allocate on gfx950? | **Yes** — 1.21 GB weights, 145 GB KV cache, memory pool OK |
| Does TP=2 bring-up work across 2× MI355X? | **Yes** — 2 ranks, RCCL 2.27.7, `AiterCustomAllreduce (AMD default)` |
| Does a forward pass run **from pip**? | **No** — dies at the first native kernel: `sgl_kernel.rotary_embedding` |
| Does a forward pass run **in the container**? | **Yes** — `/v1/chat/completions` 200 OK, 0.79-0.88 s, correct answer |
| Is the FP8 format itself a problem on gfx950? | **No** — confirmed twice: SGLang resolves the checkpoint to `torch.float8_e4m3fn` (OCP E4M3FN, not FNUZ), and the container reports `Detected fp8 checkpoint … quant=fp8, fmt=e4m3` and runs it |

The supported ROCm route is the container `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`
(an exact ROCm 7.2 / MI35x tag exists and is current). **That container has now been pulled and
tested — 89.9 GB on disk — and it works.** The pip analysis below is kept because it is the
reason the container is mandatory rather than optional.

---

# H100 (NVIDIA) — the PIP ROUTE SERVES, and the 27B FP8 loads (verified 2026-08-22)

**Headline: on H100 the pip route works, which flips the MI355X limitation above.** The exact
import that kills the ROCm pip route — `from sgl_kernel import rotary_embedding` — succeeds on
NVIDIA (`sglang-kernel` ships native CUDA wheels). No container needed. **And the documented
`Qwen/Qwen3.8-27B-FP8` — a `Qwen3_5ForConditionalGeneration` hybrid-GDN vision-language arch
that TRT-LLM 1.2.1 could not load — loads and generates real, correct text under SGLang 0.5.18.**

> **Tested topology:** 1×NVIDIA H100 80GB HBM3, **physical GPU 6 only** (single-GPU smoke test),
> Hopper cc(9,0), CUDA 13.0, driver 580.173.02, Python 3.12.3. GPUs 0–3 were a co-tenant
> production job and were never touched. Multi-GPU (TP) deferred until those GPUs free.

## H100 VERDICT

**🟢 WORKS on H100 via the pip route.** `Qwen/Qwen3.8-27B-FP8` loads natively as FP8 (E4M3,
`quant=fp8, fmt=e4m3`, 28.47 GB weights — matching the MI355X footprint), SGLang handles its
hybrid mamba/GDN linear-attention structure, captures CUDA graphs, and answers correctly.

| Question | Answer (measured on H100 GPU 6) |
|---|---|
| Does `pip install "sglang[all]"` serve on H100? | **Yes** — no container. `sgl_kernel 0.4.6.post1` + `flashinfer 0.6.17` install and import |
| Does `import sgl_kernel` succeed? | **Yes** — the exact import impossible on ROCm; `rotary_embedding` symbol present |
| Does SGLang clobber cu130 torch? | **No** — stays `torch 2.13.0+cu130`; only torchvision/torchaudio pulled (also cu130) |
| Does the 27B `Qwen3_5` VL arch load? | **Yes** — `type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3`, 28.47 GB; TRT-LLM 1.2.1 could not |
| Does the hybrid mamba/GDN path work? | **Yes** — "Using hybrid linear attention backend for hybrid GDN models"; Mamba Cache + KV Cache both allocate |
| Does it generate correct text? | **Yes** — capital-of-France=**Paris**; "ROCm belongs to AMD, and CUDA belongs to NVIDIA" |
| GPU-6 residency proof? | **Yes** — `sglang::scheduler` PID held **72088 MiB** on GPU-6 (UUID `GPU-e4fe48bc-…`) |

## H100 install (pip route — VERIFIED)

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate   # (built on tmpfs here; a folder venv works the same)
pip install -U pip
pip install torch numpy                 # -> torch 2.13.0+cu130 (proxy ON: pypi.org is allowlisted)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # 2.13.0+cu130 13.0 True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # sgl_kernel/flashinfer wheels: pypi 403s them behind the proxy
pip install "sglang[all]"               # -> sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17, flash-attn-4 4.0.0b19, transformers 5.12.1
python -c "import sgl_kernel; print('sgl_kernel OK', hasattr(sgl_kernel,'rotary_embedding'))"   # sgl_kernel OK True  <- impossible on ROCm
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130 (no clobber; no force-reinstall needed)
python -c "import sglang.srt.entrypoints.http_server; print('server import OK')"   # died on ROCm; OK here
```

Key versions: `torch 2.13.0+cu130`, `sglang 0.5.18`, `sglang-kernel 0.4.6.post1`,
`flashinfer_python 0.6.17`, `flash-attn-4 4.0.0b19`, `transformers 5.12.1`,
`torchvision 0.28.0+cu130`, `torchaudio 2.11.0+cu130`, `xgrammar 0.2.1`, `numpy 2.3.5`,
driver 580.173.02. **`transformers` stays at 5.12.1** — the checkpoint's `config.json` pins
`transformers_version: 5.8.0.dev0`, but SGLang has its own in-tree `Qwen3_5ForConditionalGeneration`
model class (it is registered in `sglang.srt.models.registry`), so the loader does not need the
dev transformers. `--trust-remote-code` is still passed.

## H100 serve — single GPU (physical GPU 6), FP8

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1   # plain CUDA — no HIP_VISIBLE_DEVICES
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --host 127.0.0.1 --port 8600
```

No `--attention-backend` flag: on Hopper SGLang auto-selects `fa3` (FlashAttention-3) with
`flashinfer` sampling — the NVIDIA analogue of the ROCm `aiter` auto-selection. The 29 GB
checkpoint was already cached in `$HF_HOME`, so nothing downloaded.

Server-side evidence (`/dev/shm/h100/out/sglang/llm_27b_fp8.log`), verbatim:

```
Load weight end. elapsed=429.52 s, type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3, avail mem=50.01 GB, mem usage=28.47 GB.
Mamba Cache is allocated. max_mamba_cache_size: 124, conv_state size: 0.34GB, ssm_state size: 17.58GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 330323, K size: 10.08 GB, V size: 10.08 GB
Using hybrid linear attention backend for hybrid GDN models.
Capture target decode CUDA graph end. elapsed=4.76 s, mem usage=0.10 GB, avail mem=8.90 GB.
The server is fired up and ready to roll!
```

## H100 client / smoke command

```bash
python inference_llm_sglang.py --port 8600 --model Qwen/Qwen3.8-27B-FP8 \
  --prompt "What is the capital of France? Answer in one word." --max_tokens 64
```

**Real output (coherence check — correct):**

```
[health] server ready after 1.0s
[latency] 0.83s | prompt_tokens=71 completion_tokens=64
[response] Thinking: ... Retrieve knowledge: Capital of France = Paris. ...
```

The README's canonical prompt (`Name the two GPU vendors ROCm and CUDA belong to, in one line.`)
returns, verbatim:

```
[latency] 0.73s | prompt_tokens=75 completion_tokens=56
[response] ... </think>
ROCm belongs to AMD, and CUDA belongs to NVIDIA.
```

Both correct — genuine generation, not garbage. The `</think>` block is Qwen3.8's reasoning
trace (on by default; pass `--reasoning-parser qwen3` to split it into `reasoning_content`).

## H100 GPU-6 residency proof (sampled from inside serving)

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader | grep e4fe48bc
1666647, python,             634 MiB,   GPU-e4fe48bc-0c29-f21e-6523-759f964bf823
1667095, sglang::scheduler, 72088 MiB,  GPU-e4fe48bc-0c29-f21e-6523-759f964bf823
```

GPU 6 (UUID `GPU-e4fe48bc-…`) is the only card touched; the `sglang::scheduler` child holds the
model (72 GB = 28.47 GB FP8 weights + 17.58 GB mamba SSM state + 20 GB KV + graphs). GPUs 0–3
(production) were never read as "mine".

## H100 27B-FP8 arch finding (the key model result)

`Qwen/Qwen3.8-27B-FP8` is **not** a plain dense Qwen3 — `config.json` declares
`model_type: qwen3_5`, `architectures: ["Qwen3_5ForConditionalGeneration"]`, and pins
`transformers_version: 5.8.0.dev0`. It is a **hybrid Gated-DeltaNet (GDN) linear-attention
vision-language** model: SGLang loads it with a per-request **mamba state cache** (17.58 GB
`ssm_state`) alongside the KV cache, logs `Using hybrid linear attention backend for hybrid GDN
models`, and initialises the multimodal (`fa3`) attention path. **TRT-LLM 1.2.1 could not load
this arch; SGLang 0.5.18 does** — the `Qwen3_5ForConditionalGeneration` class is registered
in-tree and runs the FP8 weights natively (E4M3, no dequant, 28.47 GB). This is the same
FP8-native result the MI355X container reported, now reproduced from a pip install on Hopper.

**VRAM note (80 GB vs 288 GB on MI355X):** at `--mem-fraction-static 0.85` the model leaves
only ~9–12 GB free during graph capture, because the mamba SSM state cache (17.58 GB) is large.
It fit on one H100, but `--max-running-requests` was auto-capped to 24 by the mamba cache
(logged: "max_running_requests is capped to 24 by the mamba state cache"). To raise concurrency,
either `--mamba-ssm-dtype bfloat16` (halves state size) or lower `--mem-fraction-static`. On the
288 GB MI355X this pressure did not appear; on H100 it is the one real budgeting quirk.

## H100 quirks

- **`torchcodec`/`libavutil` import errors at startup are benign.** SGLang probes for a video
  codec (`libtorchcodec_core[4-8].so` → `libavutil.so.5x`), fails to find system FFmpeg, logs a
  long traceback, and continues. It does not affect text generation. Install system FFmpeg only
  if you need multimodal video input.
- **`Ignore import error when loading sglang.srt.models.sarashina2_vision …`** and similar
  (`inkling`, `mimo_v2`) — benign; unrelated model classes failing to register.
- **Weight load is slow (~430 s / 66 shards)** off the shared `/mnt/gsma` NFS cache. This is I/O,
  not compute; a local NVMe cache would cut it dramatically. Graph capture adds ~220 s (prefill).
- **Multi-GPU (TP) deferred** — GPUs 0–3 held a production job. A TP=2 pass would add `--tp 2`
  with `CUDA_VISIBLE_DEVICES=6,7` (two free cards); the 27B fits on one H100 already, so TP mainly
  buys KV/mamba-cache headroom and concurrency, not single-stream latency.

## Container route — `lmsysorg/sglang-rocm` (TESTED)

**This is the route that works on gfx950.** Everything under this heading was executed on
2×MI355X (physical GPUs 2 and 3); output is copied verbatim. The pip analysis further down is
retained and still accurate — it explains *why* the container is required.

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk** (`docker images`).
Inside it: `sglang 0.5.17.dev20260819+g574274660f`, `torch 2.9.1+rocm7.2.0.git7e1940d4`,
HIP `7.2.26015-fc0010cf6a`, 2 GPUs visible, and — decisively — **`import sgl_kernel` succeeds**.
That single import is what the pip route cannot satisfy on ROCm.

### Start the container

```bash
docker run -d --name sglang_bringup \
  --device /dev/kfd --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HF_HOME=/mnt/data_1.5t/hf_cache -e HF_HUB_OFFLINE=1 \
  -v /mnt/data_1.5t/hf_cache:/mnt/data_1.5t/hf_cache \
  -v /mnt/data_450g:/mnt/data_450g \
  -v /home/planolab/software_test/training_junk:/work \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

`renderD144`/`renderD152` are **physical GPUs 2 and 3**; only those two render nodes are passed
in, so inside the container they are `cuda:0` and `cuda:1`. `HIP_VISIBLE_DEVICES=0` in the
container therefore means physical GPU 2. Weights and logs live on mounted volumes — **nothing
large lands on `/`**, which had only ~109 GB free.

```bash
docker exec sglang_bringup python3 -c \
  "import torch, sgl_kernel; from importlib.metadata import version; \
   print(version('sglang'), torch.__version__, torch.version.hip, torch.cuda.device_count())"
# 0.5.17.dev20260819+g574274660f 2.9.1+rocm7.2.0.git7e1940d4 7.2.26015-fc0010cf6a 2
```

### Serve — single GPU (physical GPU 2), FP8

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > /mnt/data_450g/outputs/inference_llm_sglang/llm_fp8_single.log 2>&1"
```

Server-side evidence (`llm_fp8_single.log`):

```
[2026-08-20 03:06:46] Detected fp8 checkpoint.
[2026-08-20 03:06:54] Load weight end. elapsed=7.79 s, type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3, avail mem=258.78 GB, mem usage=28.49 GB.
[2026-08-20 03:07:01] max_total_num_tokens=1543256, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=32, context_len=262144, available_gpu_mem=77.76 GB
[2026-08-20 03:07:01] INFO:     Uvicorn running on http://0.0.0.0:8100 (Press CTRL+C to quit)
[2026-08-20 03:07:04] The server is fired up and ready to roll!
```

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/llm sglang_bringup \
  python inference_llm_sglang.py --port 8100 --model Qwen/Qwen3.8-27B-FP8
```

**Real output (single GPU):**

```
[health] server ready after 1.0s
[latency] 0.88s | prompt_tokens=75 completion_tokens=57
[response] We need answer user's request: "Name the two GPU vendors ROCm and CUDA belong to, in one line." ...
</think>

ROCm: AMD; CUDA: NVIDIA.
```

Real end-to-end generation with a factually correct answer. The `</think>` block is Qwen3.8's
reasoning trace, emitted by default — see quirks.

### Multi-GPU — `--tp 2` across physical GPUs 2 and 3

Unlike the 300M/0.6B models in the sibling folders, **TP=2 is genuinely useful here**: a 27B FP8
checkpoint is 28.49 GB and sharding halves the per-GPU weight footprint, freeing memory for KV.

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3.8-27B-FP8 --tp 2 --mem-fraction-static 0.85 --max-running-requests 32 \
     --host 0.0.0.0 --port 8100 \
     > /mnt/data_450g/outputs/inference_llm_sglang/llm_fp8_tp2.log 2>&1"
```

**Weights sharded across both ranks — 28.49 GB → 14.49 GB per GPU:**

```
[2026-08-20 03:10:54 TP0] Load weight end. elapsed=4.76 s, type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3, avail mem=264.75 GB, mem usage=14.49 GB.
[2026-08-20 03:10:55 TP1] Load weight end. elapsed=5.00 s, type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3, avail mem=264.75 GB, mem usage=14.49 GB.
[2026-08-20 03:11:02 TP0] max_total_num_tokens=3228418, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=32, context_len=262144, available_gpu_mem=75.82 GB
[2026-08-20 03:11:05] The server is fired up and ready to roll!
```

**Real output (TP=2):**

```
[health] server ready after 1.0s
[latency] 0.79s | prompt_tokens=75 completion_tokens=60
ROCm belongs to AMD, and CUDA belongs to NVIDIA.
```

`rocm-smi --showmemuse` sampled *during* the TP=2 request — **both GPUs loaded**:

```
GPU[2]		: GPU Memory Allocated (VRAM%): 74
GPU[3]		: GPU Memory Allocated (VRAM%): 74
```

#### Single-GPU vs TP=2

| Metric | TP=1 (GPU 2) | TP=2 (GPUs 2+3) |
|---|---|---|
| Weight memory per GPU | 28.49 GB | **14.49 GB** (exactly halved) |
| Weight load time | 7.79 s | **4.76 / 5.00 s** (parallel reads) |
| `max_total_num_tokens` (KV budget) | 1,543,256 | **3,228,418** (2.09×) |
| Client latency, 75-token prompt | 0.88 s | 0.79 s |
| Answer correctness | correct | correct |

TP=2 more than doubles the usable KV budget and halves per-GPU weight memory, so it is the right
topology for concurrency and long context. At **batch size 1** the latency gain is small
(0.88 s → 0.79 s) because a 27B FP8 model already fits comfortably on one MI355X and the
all-reduce per layer eats most of the compute saving. Choose TP=2 for capacity, not for
single-stream speed.

### FP8 outcome — precise

**`Qwen/Qwen3.8-27B-FP8` runs natively as FP8 on gfx950 under SGLang. No fallback, no
dequantisation, no failure.** Evidence: `Detected fp8 checkpoint.` followed by
`quant=fp8, fmt=e4m3` at both TP=1 and TP=2, and a 28.49 GB resident footprint — that is FP8
width for a 27B model (bf16 would be ~54 GB). `fmt=e4m3` is **OCP E4M3FN**, the format gfx950
supports in hardware, not the older FNUZ variant. This independently corroborates the sibling
finding that Transformers loads the same checkpoint natively on gfx950.

No download was needed: the checkpoint was already cached (29 GB) in
`/mnt/data_1.5t/hf_cache`. Had it not been, the pull must be directed at
`HF_HOME=/mnt/data_450g/hf_cache` (394 GB free) — **never `/`**, which has ~109 GB free.

### Container-route quirks

- **Hybrid-attention memory budgeting is the one real trap.** `Qwen3.8-27B-FP8` loads as
  `Qwen3_5ForConditionalGeneration`, a **hybrid mamba / linear-attention** model that needs a
  per-request *mamba state cache* on top of the KV cache. A first attempt at
  `--mem-fraction-static 0.25` loaded the weights fine and then died in memory planning:
  ```
  RuntimeError: Not enough GPU memory for hybrid (mamba/linear-attention) state cache.
  Computed max_mamba_cache_size=-28 (total_rest_memory=-7.96 GB, mamba_cache_per_req=146.81 MB).
  ```
  Note the failure is **after** `Load weight end` — it is not an FP8 or kernel problem. At
  ~147 MB of state per request, the default `max_running_requests` (4067 for this model) wants
  far more memory than a small static fraction leaves. Fix with a realistic
  `--mem-fraction-static 0.85` **and** a bounded `--max-running-requests 32`. Do not
  misdiagnose this as an FP8 failure.
- **Give the model the whole GPU.** The embedding/reranker servers from the sibling folders
  were still holding 66% VRAM on both GPUs during the first attempt. Stop other SGLang servers
  (`pkill -f sglang.launch_server` inside the container) before launching the 27B.
- **Reasoning traces are on by default.** Qwen3.8 emits a `</think>`-delimited reasoning block
  before the answer, so `completion_tokens` (57-60) is much larger than the visible one-line
  answer. Pass `--reasoning-parser qwen3` to have SGLang split it into a separate
  `reasoning_content` field, or instruct the model to skip thinking.
- **`HF_HUB_OFFLINE=1`** is baked into the image env; every model used here was already cached,
  so nothing downloaded. Unset it for a fresh pull and repoint `HF_HOME` to `/mnt/data_450g`.
- **AITER JIT** compiles kernels on first use into `/root/.aiter/build/`, so the first launch in
  a fresh container is ~2 min and later ones are much faster. Keep one long-lived container
  (`sleep infinity` + `docker exec -d`) rather than one `docker run` per server.
- Benign startup noise, safe to ignore: `Ignore import error when loading
  sglang.srt.models.inkling: No module named 'cutlass'`, the same for `mimo_v2`/`torchcodec`,
  and AITER's probe `[aiter] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.`

## Install

Python 3.12. `requirements_sglang_llm.txt` is the tested set.

### AMD / ROCm — exactly what was run

```bash
python3 -m venv .env_inference_llm_sglang
source .env_inference_llm_sglang/bin/activate
pip install -U pip
# 1) ROCm torch FIRST, from the ROCm index:
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
# 2) sglang WITHOUT its CUDA dependency wall:
pip install --no-deps sglang==0.5.17
# 3) the pure-python dependency set:
pip install -r requirements_sglang_llm.txt
# 4) verify torch is STILL ROCm:
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
# -> 2.11.0+rocm7.2 7.2.26015 2
```

Resolved: `sglang 0.5.17`, `torch 2.11.0+rocm7.2`, `torchvision 0.26.0+rocm7.2`,
`transformers 5.12.1`, `xgrammar 0.2.1`, ROCm 7.2.4.

**Why `--no-deps` is mandatory.** sglang 0.5.17 lists these as *base* dependencies, not
extras: `cuda-python>=13.0`, `flashinfer_python[cu13]`, `flash-attn-4`,
`nvidia-cutlass-dsl[cu13]`, `nvidia-mathdx`, `sgl-deep-gemm`, `sglang-kernel==0.4.5`,
`quack-kernels`, `tokenspeed_mla`, and `torch==2.11.0` (the PyPI CUDA build). The old
`srt_hip` extra no longer exists. A plain `uv pip install sglang` therefore both fails to
build several CUDA packages and **overwrites ROCm torch with a CUDA wheel** — the exact
trap that bit sibling folders. If it happens:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` here (CUDA-only build). Never `pip install aiter` — the
PyPI project of that name is an unrelated async-iterator library, **not** AMD's AITER.

All three SGLang folders use the identical stack, so the sibling folders symlink this venv
(`.env_inference_embedding_sglang` / `.env_inference_reranker_sglang` →
`../../../inference/sglang/llm/.env_inference_llm_sglang`) instead of duplicating 16 GB twice.
Create a real venv per folder instead if you want them independently pinned.

**Client verified independently of the engine.** Because no SGLang server can serve on this
build, `inference_llm_sglang.py` was exercised against a minimal OpenAI-compatible stub to
confirm the client itself is correct (health wait → POST → response parsing → output):

```
[health] server ready after 0.0s
[latency] 0.00s | prompt_tokens=21 completion_tokens=9
[response] ROCm is AMD; CUDA is NVIDIA.
```

That output is from the stub, **not** from SGLang, and proves only the client half.

### The two shims (how the blocker was pinpointed, not a workaround)

After step 3, `import sglang` succeeds but `sglang.srt.entrypoints.http_server` does not:
on HIP, SGLang imports two native packages that have **no ROCm wheel anywhere**:

- **`aiter`** — AMD AITER kernels. Built from source in AMD's containers; not on PyPI.
- **`sgl_kernel`** — `sglang-kernel` 0.4.5 publishes only CUDA wheels
  (383 MB `manylinux2014_x86_64`, 37 MB `aarch64`); no ROCm variant.

To find out *where* the ROCm path actually stops, a **fail-loud** import shim was placed in
site-packages: any attribute chain resolves, but the moment a stubbed function is really
called it raises `RuntimeError: ROCm shim: <name> was really called`. This is a diagnostic,
**not** a way to serve traffic — a passing request is impossible, and every result below is
either a genuine success before the kernel boundary or a genuine failure at it.

```bash
# diagnostic only; delete to get the clean upstream ImportError back
rm .env_inference_llm_sglang/lib/python3.12/site-packages/_rocm_missing_kernel_shim*.p*
```

### NVIDIA / CUDA (upstream route — not run here, no NVIDIA GPU on this host)

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
export HF_HOME=/mnt/data_1.5t/hf_cache
export HIP_VISIBLE_DEVICES=2,3          # this agent owns physical GPUs 2 and 3 only
export CUDA_VISIBLE_DEVICES=2,3         # never set this empty on ROCm
```

## Run

### Launch — single GPU (physical GPU 2)

```bash
source .env_inference_llm_sglang/bin/activate
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 HF_HOME=/mnt/data_1.5t/hf_cache

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
`--attention-backend flashinfer` / `--kv-cache-dtype fp8_e4m3` flags onto ROCm; on this
host SGLang already auto-selects the AMD path and logs
`Attention backend not specified. Use aiter backend by default.`

### Launch — multi-GPU, TP=2 (physical GPUs 2+3)

```bash
export HIP_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=2,3
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

## Single-GPU results (measured, gfx950)

Run with the cached fallback `Qwen/Qwen3-0.6B` — the 27B FP8 download was skipped
deliberately (see "FP8 analysis"), because the failure is engine-level and identical.

```
[01:34:37] Attention backend not specified. Use aiter backend by default.
[01:34:44] Init torch distributed ends. elapsed=0.04 s, mem usage=0.15 GB
[01:34:44] Load weight begin. avail mem=287.26 GB
[01:34:45] Load weight end. elapsed=0.42 s, type=Qwen3ForCausalLM, avail mem=286.05 GB, mem usage=1.21 GB.
[01:34:46] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1360288, K size: 72.65 GB, V size: 72.65 GB
[01:34:46] Memory pool end. avail mem=140.05 GB
[01:34:47] Capture target decode CUDA graph begin. backend=full, ... avail mem=134.61 GB
[01:34:50] Exception: Capture cuda graph failed:
           ROCm shim: sgl_kernel.rotary_embedding was really called - needs a native ROCm build
```

| Metric | Value |
|---|---|
| Cold start to failure | **13 s** (server never reaches `/health`) |
| Idle VRAM before load | 287.26 GB free of 288 GB (`avail mem`) |
| Model-load VRAM | **1.21 GB** (Qwen3-0.6B bf16) |
| KV cache allocated | 145.3 GB (1,360,288 tokens @ `mem-fraction-static 0.6`) |
| Endpoint reachable | **No** — scheduler dies before the HTTP server is served |
| Real generated text | **None** — no forward pass completes |

The failing frame is `model_runner.init_cuda_graphs` → `capture_decode_graph` →
`sglang/srt/layers/rotary_embedding/base.py`. `--disable-cuda-graph` only moves the same
call into the first real prefill; rotary embedding is on every forward path.

## Multi-GPU (TP=2) results (measured, gfx950)

```
[01:40:11 TP0] Init torch distributed begin.
[01:40:11 TP1] Init torch distributed begin.
[01:40:11 TP0] sglang is using nccl==2.27.7
[01:40:14 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[01:40:14 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[01:40:16 TP0] Load weight end. elapsed=0.30 s, type=Qwen3ForCausalLM, avail mem=285.65 GB, mem usage=0.61 GB.
[01:40:16 TP1] Load weight end. elapsed=0.30 s, ...                                        mem usage=0.61 GB.
[01:40:22 TP1] Exception: Capture cuda graph failed:
               ROCm shim: sgl_kernel.rotary_embedding was really called - needs a native ROCm build
```

**Tensor parallelism itself works on ROCm up to the kernel boundary.** Per-rank weight
memory is **0.61 GB vs 1.21 GB at TP=1 — exactly half**, so the weights really were sharded
across the two MI355X, both ranks initialised, and RCCL 2.27.7 formed the process group.
`rocm-smi` sampled during a TP=2 launch shows **both owned cards holding memory
simultaneously** (bytes used, card2/card3, 288 GB each):

```
01:40:59 | card2,309220868096,816869376 | card3,309220868096,816869376
01:41:01 | card2,309220868096,1550635008 | card3,309220868096,1550630912   # ~1.44 GiB each
```

SGLang also chose AMD's `AiterCustomAllreduce` by default and, with AITER absent, logged
the correct fallback advice (`--disable-custom-all-reduce`). Cold start to failure: 24 s.

## FP8 analysis for `Qwen/Qwen3.8-27B-FP8` on gfx950

The 30 GB checkpoint was **not** downloaded: the run dies in a kernel every model shares,
so the download would only reproduce the same frame while consuming scarce disk. What
*was* verified statically against the real checkpoint metadata is more informative:

```
HF arch      : ['Qwen3_5ForConditionalGeneration']    registered in sglang 0.5.17: True
quantization : fp8   -> Fp8Config
ckpt fp8 serialized : True     weight_block_size: [128, 128]     activation_scheme: dynamic
num layers   : 64    context_len: 262144    dtype: torch.bfloat16
sglang fp8_utils dtype : torch.float8_e4m3fn
device       : AMD Instinct MI355X | gcnArch: gfx950:sramecc+:xnack-
native e4m3fn cast on gfx950 OK -> torch.float8_e4m3fn
```

**Nothing about the FP8 checkpoint is wrong for this GPU.** SGLang selects
`torch.float8_e4m3fn` — OCP **E4M3FN**, the correct gfx950 format, *not* the FNUZ variant
older CDNA parts needed — and a native e4m3fn cast executes on the device. The
architecture is registered. So the FP8-on-ROCm story here is: **the format and the
architecture are fine; only the compiled kernel package is missing.** Expect this
checkpoint to work in the ROCm container without FP8-specific flags. Do
**not** pass `--quantization fp8`; the checkpoint is already FP8.

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

Server flags that mattered here:

| Flag | Value used | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3.8-27B-FP8` | Target checkpoint (fallback `Qwen/Qwen3-0.6B` for the probes) |
| `--tp` | `1` / `2` | Tensor-parallel size across the two owned GPUs |
| `--mem-fraction-static` | `0.85` (`0.5-0.6` in probes) | KV-cache/static pool fraction; auto-reduced at TP>1 |
| `--trust-remote-code` | on | Required by the Qwen3.8 architecture |
| `--reasoning-parser` | `qwen3` | From the AMD command; auto-detected from the chat template anyway |
| `--tool-call-parser` | `qwen3_coder` | From the AMD command |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang picks `aiter`. `flashinfer` is NVIDIA-only |

## Output

The server writes no artifacts; probe logs from this validation are under
`/mnt/data_1.5t/outputs/inference_llm_sglang/` (`probe_0p6b.log`, `llm_tp2.log`). Weights
live in `$HF_HOME=/mnt/data_1.5t/hf_cache`, never on `/`. The client prints health-wait
time, latency, token usage, and the generated text to stdout only.

## Hardware support & evidence

| | NVIDIA | AMD |
|---|---|---|
| Status here | **Not tested** — no NVIDIA GPU on this host | **Tested, blocked** — 2×MI355X 288GB (gfx950), ROCm 7.2.4 |
| Install | `uv pip install sglang` (works as documented) | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Kernels | `sglang-kernel` CUDA wheel from PyPI | `sgl_kernel` + `aiter` must be built for HIP — **no wheel published** |
| Attention | `flashinfer` (cookbook) | `aiter` (auto-selected on gfx950) |

Evidence gathered 2026-08-20:

- **PyPI metadata** — `sglang` 0.5.17 base dependencies include `cuda-python>=13.0`,
  `flashinfer_python[cu13]==0.6.15.post1`, `flash-attn-4`, `sglang-kernel==0.4.5`; there is
  no `srt_hip`/`all_hip` extra (extras are only `all, checkpoint-engine, dev, diffusion,
  fastokens, http2, ray, runai, test, tracing`).
- **PyPI wheel list** — `sglang-kernel` 0.4.5 ships `manylinux2014_x86_64` (383 MB) and
  `manylinux2014_aarch64` (37 MB) only. No ROCm build.
- **Upstream source** — `sglang/srt/layers/rotary_embedding/base.py:69-75,116-117`:
  `if _is_hip:` → `from sgl_kernel import fused_qk_rope_with_cos_sin_cache_inplace` and
  `elif _is_hip: from sgl_kernel import rotary_embedding`. So on ROCm SGLang imports
  `sgl_kernel` directly — **building AITER alone can never substitute for it.** AITER is
  additionally opt-in: `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip`.
- **Docker Hub** — `lmsysorg/sglang-rocm` publishes exact tags for this host, e.g.
  `v0.5.17-rocm720-mi35x-20260819` (23.4 GB compressed) and a `rocm700-mi35x` variant,
  rebuilt daily. This is the supported AMD route.
- **AMD wheel index** — `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` publishes no
  `aiter` or `sglang` wheel.

## Notes / quirks

- **`Failed to import amdsmi`** is logged on every launch. Harmless here (SGLang falls back
  to other memory queries), but installing `amdsmi` from the ROCm tree gives SGLang proper
  AMD GPU telemetry.
- **`Ignoring corrupted tree cache file ... Permission denied`** — the shared
  `/mnt/data_1.5t/hf_cache` has snapshot tree-cache files owned by another user. Cosmetic:
  SGLang re-reads the snapshot and logs `Found local HF snapshot ...; skipping download`.
- **`--mem-fraction-static` is auto-reduced at TP>1** (0.5 → 0.425 at TP=2). Set it
  explicitly for reproducible KV sizing.
- **KV cache is enormous by default on a 288 GB card** — 145 GB for a 0.6B model at
  `0.6`. Lower `--mem-fraction-static` or set `--max-total-tokens` when sharing the box.
- **Ports** — 8100 is this repo's SGLang-LLM slot; 8101/8102 belong to the embedding and
  reranker folders. Siblings run concurrently, so do not reuse them.
- **Never set `CUDA_VISIBLE_DEVICES` empty on ROCm** — re-export both
  `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after activating the venv.

## Follow-ups

1. Re-run this exact validation inside
   `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` when `/` has >150 GB free, with
   the canonical AMD flags: `--device /dev/kfd --device /dev/dri/renderD144 --device
   /dev/dri/renderD152 --group-add video --group-add render --ipc=host --cap-add=SYS_PTRACE
   --security-opt seccomp=unconfined --shm-size 64G`. Everything up to the kernel boundary
   already passes, so that image is very likely to serve.
2. If containers are off the table, build `sgl-kernel` for HIP and AITER from source
   (hours, needs hipcc + composable_kernel) — not attempted under this task's time cap.
3. Then benchmark the FP8 27B properly: first-token latency, decode tok/s, and output
   agreement versus the Transformers reference.
