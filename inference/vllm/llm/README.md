# `infer_llm_vllm.py` — LLM chat/generation served by vLLM

## Overview & when to use

Serves a causal LLM through vLLM's OpenAI-compatible `POST /v1/chat/completions`
endpoint. `infer_llm_vllm.py` is a small client that sends either a free-form chat prompt
or a prompted single-label classification, and prints the generated text along with
finish reason, latency, decode rate and token usage.

Use vLLM here when you want **one serving stack for all three workloads** — generation
(this folder), embeddings (`inference/vllm/embedding/`) and reranking
(`inference/vllm/reranker/`). The same binary, the same flags, the same metrics endpoint
and the same OpenAI client code cover the whole fleet, and continuous batching plus paged
KV makes it the right default for multi-user serving.

If you only ever run one prompt at a time on one GPU, llama.cpp is lighter. The reason to
pick vLLM is throughput under concurrency and fleet uniformity.

This folder is also where the **FP8 story on gfx950** gets tested, and that is the most
interesting result here. The target model is **`Qwen/Qwen3.8-27B-FP8`** and it works — at
TP=1 and TP=2 — but only with `VLLM_ROCM_USE_AITER=0`; with AITER on you get a perfectly
healthy server producing pure garbage. See *Single-GPU results*, *Multi-GPU (TP=2)
results* and *Notes & quirks*. `Qwen/Qwen3-4B` is kept alongside as the smaller-model
control.

## Install

**There is no ROCm vLLM wheel — this is the single most important fact in this folder.**
Verified against the pinned versions below:

- PyPI `vllm==0.27.1` publishes exactly two binary wheels
  (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`), both **CUDA-only**. Its
  `requires_dist` hard-depends on `flashinfer-python`, `nvidia-cudnn-frontend`,
  `nvidia-cutlass-dsl[cu13]` and `torch==2.13.0` (the CUDA build). Installing it on a ROCm
  host gives you a non-functional engine.
- `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` publishes `torch`, `torchaudio`,
  `apex`, `jaxlib` and `tensorflow_rocm` wheels — **no `vllm` wheel**.
- Building vLLM from source for `gfx950` works but is an hours-long compile.

So the pip/venv route that would normally be preferred is genuinely unavailable, and the
verified route is a **container**. The validated runs used
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`, which ships a
working ROCm vLLM:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and client transcripts

docker run -d --name vllm_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  --network host \
  -v "$PWD":/workspace/repo -v "$HF_HOME":"$HF_HOME" \
  -e HF_HOME="$HF_HOME" -w /workspace/repo \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

Two gotchas in that command, both learned the hard way:

- The image has **no `render` group**, so the canonical `--group-add render` fails with
  `unable to find group render`. Pass the host's **numeric** GIDs instead (typically
  `video`=44, `render`=993) — that is what the `getent` substitutions above do.
- Pinning GPUs by **render node** (`renderD128` = the first GPU, `renderD136` = the second)
  rather than by `HIP_VISIBLE_DEVICES` means `torch.cuda.device_count()` is `2` inside the
  container no matter what any tool does to the environment: the container sees
  exactly two `gfx950` devices and cannot touch any other card on the box.

**A newer image is not needed.** The obvious assumption is that
0.20.2 predates upstream's Qwen3.8 ROCm enablement and that
`vllm/vllm-openai-rocm:nightly` (~11.5 GB compressed, 25–30 GB on disk) would be required.
That assumption is **false** — this build already contains
`Qwen3_5ForConditionalGeneration` and loads the checkpoint without complaint. Anyone
reaching for a newer image to fix garbage output should set `VLLM_ROCM_USE_AITER=0` first;
the bug is in an AITER kernel, and a newer image is not obviously the fix.

Verified versions inside that image:

| Component | Version |
|---|---|
| vLLM | `0.20.2rc1.dev253+g1ff9d3353` |
| torch | `2.9.1.dev20251204+rocm7.0.2.git351ff442` |
| `torch.version.hip` | `7.0.51831-7c9236b16` |
| transformers | `5.14.1` |
| ROCm (container) | 7.0.2 |
| ROCm (host) | 7.2.4 |
| GPU | AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, 288 GiB, 256 CUs |

The client itself needs only `python-dotenv` (everything else it uses is stdlib), so it
runs on the host outside the container:

```bash
pip install -r requirements_llm_vllm.txt
```

## Environment & secrets

`dev.env` is symlinked to the repo-root file and supplies the HF token used to pull model
repos:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded by `load_dotenv("dev.env")` in the client, and exported into the serving shell.
Never echo it. Model weights are kept off `/`'s root partition by pointing `HF_HOME` at a
data volume.

GPU pinning for the two cards in use:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides *every* GPU.

gfx950 tuning knobs (see *Notes & quirks* for when each matters):

```bash
# Serving Qwen/Qwen3.8-27B-FP8 — MANDATORY, or every completion is garbage:
export VLLM_ROCM_USE_AITER=0

# Serving a dense non-FP8 model (Qwen/Qwen3-4B) — AITER on is correct and is the default:
export VLLM_ROCM_USE_AITER=1        # gfx950 default; AITER attention + GEMM + sampler
export VLLM_ROCM_USE_AITER_MOE=0    # the usual corruption workaround — but see below:
                                    # it does NOT fix the FP8 27B case
```

The `VLLM_ROCM_USE_AITER=0` requirement for the FP8 checkpoint is the most important
operational detail in this folder — the failure it prevents is silent. See *Notes &
quirks*.

## Serve

Single GPU — the verified command (run inside the container):

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

Two GPUs, tensor parallel — the verified TP=2 command:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0,1
export VLLM_ROCM_USE_AITER=1
vllm serve Qwen/Qwen3-4B \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen3-4b
```

### `Qwen/Qwen3.8-27B-FP8` — the real target model

**`VLLM_ROCM_USE_AITER=0` is mandatory for this checkpoint.** With AITER on (the gfx950
default) the server is healthy but every completion is token salad — see *The FP8 27B
result* below. Single GPU:

```bash
export HF_HOME=/path/to/hf_cache
export VLLM_ROCM_USE_AITER=0          # REQUIRED — see Notes & quirks
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen38-27b-fp8
```

Two GPUs, tensor parallel — the 27B TP=2 command:

```bash
export HF_HOME=/path/to/hf_cache
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

Do **not** pass `--quantization fp8`. The checkpoint is already FP8 and vLLM reads
`quantization_config` from it (`quantization=fp8` appears in the engine config
automatically). Forcing the flag would re-quantize.

## Client / smoke command

Against the 27B FP8 server (the default `--model` of the client):

```bash
python infer_llm_vllm.py --port 8000 --model qwen38-27b-fp8 --max_tokens 220 --seed 42
```

Against the smaller Qwen3-4B server:

```bash
python infer_llm_vllm.py --port 8000 --model qwen3-4b --max_tokens 200 --seed 42
```

Prompted classification instead of free generation:

```bash
python infer_llm_vllm.py --port 8000 --model qwen3-4b \
  --classify "My internet has been down since this morning and the router light is red."
```

Raw equivalent:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"What is vLLM?"}],"max_tokens":128}'
```

## Single-GPU results

**`Qwen/Qwen3.8-27B-FP8` works on one MI355X — with one mandatory env var
(`VLLM_ROCM_USE_AITER=0`).** That is the headline result of this folder and it is covered
in *The FP8 27B result* immediately below. The `Qwen/Qwen3-4B` numbers that follow are
retained as the smaller-model datapoint and as the control that proves AITER itself is
not broken in general.

### `Qwen3-4B` — the smaller-model control

**Works, unmodified.** vLLM came up on one MI355X and
produced correct, coherent text on the first attempt. No code changes and no ROCm-specific
workarounds were needed for a standard dense model, **with AITER enabled**.

| Measurement | Value (Qwen3-4B, TP=1) |
|---|---|
| Cold start (launch → `Application startup complete`) | **~208 s** first launch on this image / ~35 s once the torch.compile + AITER caches are warm |
| Weights load | 2.13 s |
| **Model-load VRAM** | **7.56 GiB** |
| `init engine` (profile + KV + warmup) | 195.09 s (compilation 186.57 s) |
| Graph capture | 5 s, 0.55 GiB |
| Available KV cache | 119.39 GiB |
| GPU KV cache size | 869,360 tokens |
| Total process VRAM at `--gpu-memory-utilization 0.45` (`rocm-smi`) | ~141.2 GB |
| Decode rate | **123.0 tok/s** (single stream, 200 tokens) |

Real generated text from `/v1/chat/completions` (prompt: *"In two sentences, explain what
vLLM is and which GPU vendors it supports."*, `--seed 42`):

```
endpoint      : http://localhost:8010/v1/chat/completions
model         : qwen3-4b
finish_reason : length
latency       : 1.63s
decode rate   : 123.0 tok/s (200 completion tokens)
usage         : {'prompt_tokens': 25, 'total_tokens': 225, 'completion_tokens': 200}

--- response ---
<think>
Okay, the user wants me to explain what vLLM is in two sentences and mention the GPU
vendors it supports. Let me start by recalling what vLLM is. From what I remember, vLLM is
a library or framework related to large language models, maybe for inference. It's probably
optimized for efficiency. I think it's used for accelerating the deployment of LLMs,
especially in terms of handling multiple requests or parallel processing.
...
```

That is genuine, coherent, on-topic generation — the model reasons about the question in a
`<think>` block, which is Qwen3's native reasoning format.

### The FP8 27B result — works, but only with `VLLM_ROCM_USE_AITER=0`

`Qwen/Qwen3.8-27B-FP8` is the headline finding of this folder, and the answer is more
interesting than a simple pass/fail: **it works on vLLM 0.20.2 / gfx950, but only after
disabling AITER.** Contrary to the expectation that vLLM 0.20.2 would predate Qwen3.8
ROCm enablement, the engine loads and runs the model correctly — the defect is in one
AITER kernel, not in model support.

**Step 1 — it loads cleanly, out of the box.** vLLM 0.20.2 already contains the
`Qwen3_5ForConditionalGeneration` class the checkpoint asks for (`model_type: qwen3_5`),
auto-detects `quantization=fp8`, and reports:

```
Loading weights took 6.97 seconds
Model loading took 29.38 GiB memory and 7.735900 seconds
GPU KV cache size: 3,505,447 tokens
Application startup complete.
```

**Step 2 — with AITER on (the gfx950 default), every completion is token salad.**
`/health` returns 200, `/v1/models` lists the model, the metrics look normal — and the
text is garbage:

```
--- response ---
WePGG G/CO -ofU + of/v*';FVFV F G Y-vP EVIFa +-b */=ACAB<CJK>Y WEOfoci2 ...
```

**Independently re-verified on a clean relaunch** (`VLLM_ROCM_USE_AITER=1`, same serve
command, `--seed 42`) — the server reports `health HTTP 200`, decodes at a healthy
58.1 tok/s, and emits:

```
finish_reason : length
decode rate   : 58.1 tok/s (60 completion tokens)

--- response ---
WeGPG G02 q GCGvG/G/�?vGG /G2 2/ACO)[  G-' D DI-【YFBV V IC=['-wh)KF-
```

Note the failure is *fast* — garbage decodes at 58.1 tok/s versus 38.1 tok/s for correct
output, because AITER's kernel is genuinely quicker; it is just wrong. Throughput metrics
alone would rate this the better configuration.

Not a sampling artifact — it reproduces exactly at `temperature=0`:

```
prompt : "What is the capital of France? Answer in one word."
output : 'UsermGG BF//CG/ACA VCG/ACA /gF/-*/-YG/WE =*//gG*/-[^IG…'
```

**Step 3 — `VLLM_ROCM_USE_AITER_MOE=0` does NOT fix it.** A full reload with AITER MoE
disabled and AITER attention left on still produced garbage. So this is *not* the
documented AITER MoE corruption bug, and the usual workaround is the wrong one here.

**Step 4 — `VLLM_ROCM_USE_AITER=0` fixes it completely.** Disabling AITER wholesale makes
the model correct on the first try:

```
prompt : "What is the capital of France? Answer in one word."
output : 'User asks: ... Capital is Paris. Ensure one word.\n</think>\n\nParis'
```

The server log shows exactly what changed — the FP8 GEMM kernel selection:

| `VLLM_ROCM_USE_AITER` | Kernel selected for `Fp8LinearMethod` | Output |
|---|---|---|
| `1` (gfx950 default) | `AiterFp8BlockScaledMMKernel` | **garbage** |
| `1` + `..._MOE=0` | `AiterFp8BlockScaledMMKernel` | **garbage** |
| `0` | `TritonFp8BlockScaledMMKernel` | **correct** |

**Root cause: the AITER FP8 block-scaled GEMM produces corrupted numerics for this model
on gfx950 in this AITER build.** It is a GEMM bug, not a MoE bug and not a model-support
gap — which is why `VLLM_ROCM_USE_AITER_MOE=0` misses it. The Triton FP8 block-scale
fallback is numerically correct.

Supporting evidence that the hardware and the weights are both fine:

- The checkpoint is OCP **E4M3** (`quantization_config.fmt: "e4m3"`) — exactly the FP8
  format gfx950 implements natively. No FNUZ conversion is involved.
- A sibling folder proved **Transformers loads and runs this same FP8 checkpoint correctly
  on this same gfx950 hardware**.
- The same vLLM build serves dense `Qwen/Qwen3-4B` correctly *with AITER enabled*, so
  AITER is not broken in general — only its FP8 block-scaled GEMM path.

Real generated text from the working FP8 27B configuration (`--seed 42`, 220 tokens):

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : qwen38-27b-fp8
finish_reason : length
latency       : 5.70s
decode rate   : 38.6 tok/s (220 completion tokens)

--- response ---
We need answer user: "In two sentences, explain what vLLM is and which GPU vendors it
supports." Need produce final two sentences. Need be accurate. vLLM is open-source
library/framework for high-throughput LLM inference/serving, uses PagedAttention. GPU
vendors supports? ... vLLM supports NVIDIA GPUs primarily, AMD GPUs (ROCm), Intel GPUs?
```

Fully coherent and on-topic — the model is reasoning about the question in Qwen's
scratchpad style before committing to the two-sentence answer.

| Measurement | Qwen3.8-27B-FP8, TP=1, `AITER=0` |
|---|---|
| Weights load | 7.43 s |
| **Model-load VRAM** | **29.38 GiB** |
| `init engine` (warm caches) | 61.06 s (compilation 47.96 s) |
| Graph capture | 10 s, 0.73 GiB |
| Available KV cache | 95.41 GiB |
| GPU KV cache size | 1,451,258 tokens |
| Total process VRAM at `--gpu-memory-utilization 0.45` | ~145.6 GB |
| Decode rate | **38.6 tok/s** (single stream) |

**Re-verified end-to-end on a fresh relaunch** (`HIP_VISIBLE_DEVICES=0`,
`VLLM_ROCM_USE_AITER=0`): weights load 7.88 s, model load **29.38 GiB**, available KV
95.75 GiB, GPU KV cache 1,457,083 tokens, graph capture 8 s / 0.73 GiB, `init engine`
23.78 s with warm caches, `Application startup complete` ~64 s after launch, decode
**38.1 tok/s**. The generated text is **token-identical** to the earlier run at the same
`--seed 42`, so this result is reproducible rather than a lucky sample.

`rocm-smi` during single-GPU decode — one card working, the second untouched:

```
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,100,309220868096,145960108032     <- 145.96 GB, 100% busy
card1,0,309220868096,298119168          <- 0.30 GB, idle
```

Prompted classification on the same server (`--classify`, `temperature=0`) returns the
correct label and stops cleanly (`finish_reason: stop`, 80 tokens, 40.4 tok/s):

```
--- response ---
We need answer user's request. Need classify text: "My internet has been down since this
morning and the router light is red." Labels: billing, network_outage, device_setup,
other. ... Strict classifier. Output only label.
</think>

network_outage
```

**A silent wrong-output failure is the dangerous kind** — the server is healthy, the
metrics look normal, and only reading the generated text reveals it. Always eyeball real
generated text as part of ROCm bring-up; a green `/health` proves nothing about numerics.

## Multi-GPU (TP=2) results

### `Qwen3.8-27B-FP8` at TP=2 — the headline multi-GPU result

**This works.** `--tensor-parallel-size 2` shards the FP8 27B across both GPUs
with no flag beyond `--tensor-parallel-size 2` and the mandatory
`VLLM_ROCM_USE_AITER=0`. No RCCL tuning, no `--distributed-executor-backend` override.

| Measurement | 27B TP=1 | 27B TP=2 | Effect |
|---|---|---|---|
| Model-load VRAM **per rank** | 29.38 GiB | **14.9 GiB** | halved — genuinely sharded |
| Weights load | 7.43 s | 4.53 s | parallel reads |
| Available KV cache (per rank) | 95.41 GiB | 109.73 GiB | more room once weights shrink |
| **GPU KV cache size** | 1,451,258 tok | **3,334,326 tok** | **2.3×** context capacity |
| Graph capture | 10 s / 0.73 GiB | 9 s / 0.64 GiB | — |
| `init engine` (warm caches) | 61.06 s (compile 47.96 s) | 101.75 s (compile 87.74 s) | 2 ranks compile |
| Cold start (launch → `Application startup complete`) | ~90 s | **~162 s** | two ranks to bring up |
| FP8 GEMM kernel selected | `TritonFp8BlockScaledMMKernel` | `TritonFp8BlockScaledMMKernel` | same |
| Decode rate (single stream) | 38.6 tok/s | **52.4 tok/s** | **+36%** — TP pays off at 27B |

The per-rank weights halving (29.38 → 14.9 GiB) and the KV cache growth (1.45M → 3.33M
tokens) together prove real sharding rather than replication.

**Unlike the 4B case, TP=2 is genuinely faster here** — 38.6 → 52.4 tok/s single-stream.
A 27B FP8 model is memory-bandwidth-bound enough at batch 1 that splitting the weight
reads across two HBM stacks beats the cost of the per-layer all-reduce. This is the
inversion of the 4B result below, and it is the reason TP exists.

`rocm-smi` sampled **while the 27B TP=2 server was decoding** (`--showuse`, sampled from a
background loop during the client call):

```
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,100,309220868096,142181068800     <- TP rank 0, 142.18 GB, 100% busy
card1,100,309220868096,142184337408     <- TP rank 1, 142.18 GB, 100% busy
card2,0,309220868096,230080622592       <- idle, another job's card
```

Both serving GPUs are at **100% utilisation in the same sample** with VRAM
footprints differing by 3.3 MB out of 142 GB, while the unrelated card2 sits at 0%. That
is a real 2-GPU collective, confined to the two selected cards.

Generated text from the 27B TP=2 server (`--max_tokens 220 --seed 42`):

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : qwen38-27b-fp8
finish_reason : length
latency       : 4.20s
decode rate   : 52.4 tok/s (220 completion tokens)
usage         : {'prompt_tokens': 69, 'total_tokens': 289, 'completion_tokens': 220}

--- response ---
We need to answer user: "In two sentences, explain what vLLM is and which GPU vendors it
supports." ... vLLM is high-throughput, memory-efficient inference and serving engine for
LLMs, PagedAttention. GPU vendors supports: NVIDIA GPUs primarily, AMD GPUs? ... Maybe:
"vLLM is an open-source engine for fast, memory-efficient inference and serving of large
language models, using techniques like PagedAttention. It supports NVIDIA GPUs and, via
ROCm, AMD GPUs." Ensure two sentences.
</think>

vLLM is an open-source engine for fast, memory-efficient inference and serving of large
language models
```

Coherent, on-topic, correct about ROCm, and it closes the `</think>` scratchpad and begins
the real two-sentence answer before hitting the 220-token cap. That is genuine 27B-class
generation, not the AITER token salad.

**Reproduced from a clean relaunch** (`--tensor-parallel-size 2`, `VLLM_ROCM_USE_AITER=0`):
`TritonFp8BlockScaledMMKernel` selected, model load **14.9 GiB per rank** again, GPU KV
cache 3,354,715 tokens, `init engine` 24.88 s with warm caches, up in ~72 s. A different
prompt and a different seed, to prove the output tracks the question rather than replaying
a cached answer:

```
prompt : "Name the capital of France, then say one sentence about MI355X GPUs."
decode rate   : 44.5 tok/s (120 completion tokens)

--- response ---
We need to answer user: ... Simple. Need final: Paris. One sentence about MI355X GPUs.
... "The capital of France is Paris. The AMD Instinct MI355X is a high-performance AI
accelerator GPU designed for large-scale data center workloads."
```

Correct on both counts — Paris, and an accurate description of the hardware it is running
on. `rocm-smi` in the same window, both ranks resident at 142.66 GB:

```
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,3,309220868096,142663155712
card1,3,309220868096,142663372800
card2,0,309220868096,231937077248     <- another job's GPU, untouched
```

### `Qwen3-4B` at TP=2 — the smaller-model datapoint

**Works, unmodified.** `--tensor-parallel-size 2` shards `Qwen/Qwen3-4B` across
both GPUs and produces text that matches the single-GPU run. Nothing beyond
the flag was needed — no NCCL/RCCL tuning, no `--distributed-executor-backend` override.

| Measurement | TP=1 | TP=2 | Effect |
|---|---|---|---|
| Model-load VRAM **per rank** | 7.56 GiB | **3.82 GiB** | halved — genuinely sharded |
| Weights load | 2.13 s | 0.88 s | parallel reads |
| Available KV cache | 119.39 GiB | 120.07 GiB | per rank |
| **GPU KV cache size** | 869,360 tok | **1,748,576 tok** | **doubled** — 2× context capacity |
| Graph capture | 5 s / 0.55 GiB | 5 s / 0.49 GiB | — |
| `init engine` (warm caches) | 195.09 s | 29.06 s | — |
| Decode rate (single stream) | 123.0 tok/s | 118.3 tok/s | ~4% slower — see note |

The weights halving (7.56 → 3.82 GiB per rank) and the KV doubling (869K → 1.75M tokens)
together are the proof that TP=2 is really sharding rather than replicating.

`rocm-smi` sampled **while the TP=2 server was serving** — both serving GPUs
loaded, near-identical footprints, other cards untouched:

```
device,GPU Memory Allocated (VRAM%),GPU Memory R/W Activity (%),VRAM Total (B),VRAM Used (B)
card0,46,7,309220868096,142244859904      <- TP rank 0  (142.2 GB)
card1,46,6,309220868096,142244335616      <- TP rank 1  (142.2 GB)
card2,66,0,309220868096,206072328192      <- idle, another job's card
```

The two ranks differ by 524,288 bytes out of 142 GB, and both show live read/write
activity (7% / 6%) during decode — this is a real 2-GPU collective, not one GPU doing the
work.

TP=2 generated text (same prompt and `--seed 42` as the single-GPU run):

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : qwen3-4b
finish_reason : length
latency       : 1.69s
decode rate   : 118.3 tok/s (200 completion tokens)

--- response ---
<think>
Okay, the user wants me to explain what vLLM is in two sentences and mention the GPU
vendors it supports. Let me start by recalling what vLLM is. From what I remember, vLLM is
a library or framework related to large language models, maybe for inference. ...
```

The TP=2 output is **token-identical to TP=1 for the first ~120 tokens** and then diverges
into an equally coherent continuation. That is the expected signature of a correct TP
implementation: tensor parallelism changes the floating-point reduction order, the tiny
numeric difference eventually flips one sampling decision, and generation forks from
there. Identical-then-diverging is correct; garbage would not be.

Prompted classification over TP=2 (`--classify`), which exercises the deterministic
`temperature=0` path:

```
--- response ---
<think>
... Since the user is reporting a disruption in their internet service, that's probably a
network outage. Billing issues would relate to charges or payments, which isn't mentioned
here. ...
</think>

network_outage
```

Correct label, correctly formatted, at **337 tok/s**.

**Note on the ~4% single-stream slowdown.** TP=2 being marginally slower than TP=1 at
batch size 1 is expected, not a defect: a 4B model on a 288 GiB card is not memory-bound
enough to benefit from sharding, so all TP adds at batch 1 is a per-layer all-reduce. TP
pays off for models that do not fit on one GPU, or under heavy concurrency. **For a 7.56
GiB model on MI355X the right multi-GPU pattern is replication** — one independent server
per GPU behind a load balancer — exactly as recommended in the sibling embedding and
reranker folders. TP=2 is documented here as verified working; it is not what you would
deploy at this model size.

## Arguments / flags

Serve-side flags used here:

| Flag | Value used | Meaning |
|---|---|---|
| `--tensor-parallel-size` | `1` / `2` | Shards weights and KV across N GPUs. Verified at 2; halves per-rank weights and doubles KV capacity |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Benchmark-layout port for the LLM server |
| `--served-model-name` | `qwen3-4b` / `qwen38-27b-fp8` | Short alias clients pass as `model`, decoupling the API from the HF repo id |
| `--max-model-len` | `8192` / `32768` | Context window. Lower it to cut KV footprint; the 27B checkpoint supports 32K |
| `--gpu-memory-utilization` | `0.45` | Fraction of VRAM vLLM preallocates. Set to 0.45 here so two servers could share a card; default is `0.9` |
| `--trust-remote-code` | required for 27B | The `qwen3_5` checkpoint ships custom modeling code |
| `--quantization` | *(auto)* | Not passed — vLLM reads `quantization_config` from the checkpoint and selects `fp8` itself |

`infer_llm_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--model` | `qwen38-27b-fp8` | Served model name (must match `--served-model-name`) |
| `--prompt` | vLLM question | User prompt for free generation |
| `--system_prompt` | `None` | Optional system message |
| `--classify` | `None` | Text to classify into one label via prompting; forces `temperature=0` |
| `--labels` | `billing,network_outage,device_setup,other` | Label set used by `--classify` |
| `--max_tokens` | `256` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (overridden to `0.0` by `--classify`) |
| `--top_p` | `0.8` | Nucleus sampling top-p |
| `--seed` | `None` | Sampling seed for reproducible output |
| `--timeout` | `600.0` | HTTP timeout in seconds |

## Output

The client prints the endpoint, resolved model name, finish reason, wall-clock latency,
decode rate, token usage, an optional `reasoning_content` block, and the generated text.

**Expected output** for `python infer_llm_vllm.py --port 8000 --model qwen3-4b --seed 42`:

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : qwen3-4b
finish_reason : length
latency       : 1.69s
decode rate   : 118.3 tok/s (200 completion tokens)
usage         : {'prompt_tokens': 25, 'total_tokens': 225, 'completion_tokens': 200}

--- response ---
<think>
Okay, the user wants me to explain what vLLM is in two sentences and mention the GPU
vendors it supports. ...
```

Redirect server logs and captured client transcripts to a data volume, e.g.
`$OUTPUT_DIR/inference_llm_vllm/`. Nothing large is written into
the repo, and nothing is written to `/`'s root partition.

## Hardware support & evidence

- **AMD: tested and working, at 27B scale.** 1× and 2× AMD Instinct MI355X (`gfx950:sramecc+:xnack-`,
  288 GiB, 256 CUs), host ROCm 7.2.4, container ROCm 7.0.2, vLLM
  `0.20.2rc1.dev253+g1ff9d3353`, torch `2.9.1.dev20251204+rocm7.0.2`,
  `torch.version.hip 7.0.51831`. `Qwen/Qwen3.8-27B-FP8` serves correct text at TP=1
  (38.1 tok/s) and TP=2 (52.4 tok/s), plus `Qwen/Qwen3-4B` at both.
- **The FP8 format matters and it is already right.** The checkpoint is OCP **E4M3**
  (`quantization_config.fmt: "e4m3"`), which is exactly what gfx950 implements natively —
  **not** the FNUZ variant older AMD hardware needs. No conversion, no `--quantization fp8`.
- **The model-support gap that might be expected does not exist.** vLLM 0.20.2 already ships
  `Qwen3_5ForConditionalGeneration` and resolves `model_type: qwen3_5` without complaint,
  so **no newer image is required** — `vllm/vllm-openai-rocm:nightly` is not needed. See
  *Notes & quirks*.
- **The GGUF fallback is not needed.** `unsloth/Qwen3.8-27B-GGUF:Q8_0` exists but the FP8
  safetensors path works and is the better path (vLLM's GGUF support is limited and would
  not exercise the FP8 hardware at all).
- **NVIDIA: verified via the pip wheel** — see the H100 section below. The same `vllm serve`
  command also applies with the `vllm/vllm-openai:latest` container.

## H100 (NVIDIA)

vLLM `0.27.1` (pip / CUDA 13.0) serves the real
`Qwen/Qwen3.8-27B-FP8` correctly on **one** H100 80GB — coherent, correct text on every
prompt. The headline finding is the **architecture-support answer**: this checkpoint is a
`Qwen3_5ForConditionalGeneration` **vision-language + Mamba/GDN-hybrid** model
(`model_type: qwen3_5`), and **vLLM 0.27.1 handles it where TensorRT-LLM 1.2.1 did not**.

### Install (pip route — no container needed)

The ROCm story above is container-only; on **NVIDIA the pip wheel is native and works**.
Unset the proxy first (`unset HTTP_PROXY HTTPS_PROXY ...` — pypi.nvidia.com is proxy-blocked):

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy          # -> torch 2.13.0+cu130 (native CUDA 13, no --index-url)
pip install vllm                 # -> vllm 0.27.1 (pulls flashinfer/cutlass-dsl[cu13]); torch stays 2.13.0+cu130
pip install python-dotenv        # for the client
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA-only) |
| torch | `2.13.0+cu130` (native CUDA 13.0; **not** clobbered by the vLLM install) |
| transformers | `5.15.1` (bundled; **knows `qwen3_5` / `qwen3_5_vision` / `qwen3_5_text`**) |
| driver / CUDA | 580.173.02 / 13.0, H100 80GB HBM3, cc(9,0) |

### The qwen3_5 arch-support finding (the headline)

- `config.json` is `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type:
  qwen3_5`, and `quantization_config.modules_to_not_convert` lists `visual.blocks.*` — it
  is a **vision-language** model. The text backbone is a **Mamba/GDN (gated delta net)
  linear-attention hybrid** (vLLM logs `qwen_gdn_linear_attn`, `qwen3_5_text`).
- vLLM's registry **contains** `Qwen3_5ForConditionalGeneration` (verified:
  `ModelRegistry.get_supported_archs()`), and at serve time vLLM logs
  `Resolved architecture: Qwen3_5ForConditionalGeneration`, auto-detects `quantization=fp8`,
  and loads clean. **This is the opposite of TRT-LLM 1.2.1**, which rejected the same
  checkpoint as an unrecognized `qwen3_5` arch needing a newer transformers. The bundled
  transformers `5.15.1` is new enough.
- The `Qwen3VLVideoProcessorInitKwargs` `min_frames`/`max_frames` lines in the log are
  harmless docstring warnings from the VL processor, not errors.

### One config change required — the Mamba cache (VRAM-driven)

Because the text backbone is a **GDN/Mamba hybrid**, *each decode sequence needs one Mamba
cache block*. On a single 80GB card the default `max_num_seqs=1024` exceeds the available
Mamba blocks and the engine aborts at CUDA-graph capture:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (694). Each decode
sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. Please
lower max_num_seqs to at most 694 or increase gpu_memory_utilization.
```

Fix: pass **`--max-num-seqs 256`** (well under 694). This is a real single-80GB-card
adjustment, not a defect — MI355X's 288GB cards would not hit it. Do **not** pass
`--quantization fp8` (the checkpoint is already E4M3 FP8; vLLM selects it automatically).

### Serve (the H100 command)

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

Client (same script as ROCm):

```bash
python infer_llm_vllm.py --port 8500 --model qwen38-27b-fp8 --max_tokens 200 --seed 42
```

### Expected output — server log (H100)

```
[model.py:645] Resolved architecture: Qwen3_5ForConditionalGeneration
[__init__.py:634] Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
[default_loader.py:430] Loading weights took 399.75 seconds        <- slow CIFS FS, not vLLM
[gpu_model_runner.py:5405] Model loading took 28.54 GiB memory and 405.34 seconds
[kv_cache_utils.py:2235] GPU KV cache size: 485,083 tokens
[kv_cache_utils.py:2236] Maximum concurrency for 8,192 tokens per request: 59.21x
[api_server.py:678] Supported tasks: ['generate']
INFO:     Application startup complete.
```

### Expected generated text (H100 — coherent and correct, no garbage)

```
prompt : "What is the capital of France? Answer in one word."  (temperature 0)
--- response ---
User asks: ... Capital is Paris. Final: Paris.
</think>

Paris
```

```
prompt : "Name the capital of France, then say one sentence about H100 GPUs."  (seed 7)
decode rate : 68.2 tok/s (61 completion tokens)
--- response ---
...</think>

The capital of France is Paris. H100 GPUs are high-performance AI accelerators designed
for large-scale machine learning training and inference.
```

Prompted classification (`--classify`, `temperature=0`) returns the correct label:
`network_outage`. The default vLLM-explanation prompt (`--seed 42`, 200 tok) decodes at
**73.1 tok/s** and correctly states vLLM supports NVIDIA (CUDA), AMD (ROCm) and Intel GPUs.
Every prompt yields **real, on-topic, correct** text — none of the AITER token-salad seen
on gfx950 (that AITER FP8-GEMM path is ROCm-only and does not apply here; the CUDA
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` path is numerically correct).

### GPU residency check (sampled while serving)

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # during decode
```

Expect ~73 GB resident on the single selected GPU (KV cache is the bulk) and ~88% util
during decode. No other card is touched.

### H100 metrics

| Measurement | Qwen3.8-27B-FP8, 1×H100, `--max-num-seqs 256` |
|---|---|
| Weights load | 399.75 s (**CIFS** network FS; ~5 s/shard × 66 shards — not a vLLM cost) |
| Model-load VRAM | **28.54 GiB** (matches the 29.38 GiB seen on gfx950) |
| GPU KV cache size | **485,083 tokens** (@ `--gpu-memory-utilization 0.90`, 8192 ctx) |
| Max concurrency @ 8192 ctx | 59.21× |
| Total process VRAM (single GPU) | **~73.0 GB** |
| FP8 GEMM kernel | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (CUDA) |
| Decode rate (single stream) | **68–73 tok/s** |

### Accuracy vs the transformers baseline

The sibling transformers reference used a **different** model (`LiquidAI/LFM2.5-350M`), so
there is no token-level diff to run here. Its "capital of France → Paris" answer matches
this server's output, and every prompt here returns correct, coherent text — which is the
accuracy bar for a different model.

### Multi-GPU (not covered)

The H100 notes above are a single-GPU smoke test. A TP pass would add
`--tensor-parallel-size N`; the `qwen3_5` text backbone's head/GDN divisibility by N must
be checked before launching.

### H100 summary

vLLM 0.27.1 (pip / cu130) serves `Qwen/Qwen3.8-27B-FP8` correctly on one H100,
with `--max-num-seqs 256` for the Mamba/GDN cache. The pip route is native on NVIDIA (no
container). **Arch-support headline: vLLM recognises and serves the `qwen3_5` VL/hybrid
checkpoint that TRT-LLM 1.2.1 could not.** No fallback to a smaller model is needed.

## Notes & quirks

- **`VLLM_ROCM_USE_AITER=0` is mandatory for the FP8 27B — this is the single most
  important line in this folder.** With AITER on, the server is healthy and *fast* and the
  output is meaningless. Reproduced from a clean launch twice.
- **`VLLM_ROCM_USE_AITER_MOE=0` is the wrong workaround here.** It is the documented fix
  for the AITER MoE corruption bug, and it does *not* help: the fault is in
  `AiterFp8BlockScaledMMKernel`, the FP8 block-scaled **GEMM**, not the MoE path.
- **A green `/health` proves nothing about numerics.** The broken configuration returns
  HTTP 200, lists the model on `/v1/models`, reports normal metrics, and decodes *faster*
  than the correct one (58.1 vs 38.1 tok/s). Only reading generated text catches it. Make
  eyeballing generated output a required step of every ROCm bring-up.
- **AITER is not broken in general.** The same build serves dense `Qwen/Qwen3-4B` correctly
  *with AITER enabled*. Only the FP8 block-scaled GEMM path is affected.
- **Do not pass `--quantization fp8`.** The checkpoint is already FP8; vLLM reads
  `quantization_config` and selects `fp8` on its own.
- **`--trust-remote-code` is required** for the `qwen3_5` checkpoint.
- **TP=2 helps at 27B and hurts at 4B.** +36% single-stream at 27B (memory-bandwidth-bound),
  −4% at 4B (all-reduce overhead with nothing to gain). Model size decides whether to shard
  or replicate.
- **No `render` group in the image.** `--group-add render` fails with `unable to find group
  render`; pass the host's numeric GIDs via `getent` instead.
- **AITER JIT-builds on first launch**, which is why the first cold start on this image is
  ~200 s and later ones are ~64 s (TP=1) / ~162 s (TP=2).
- **The `quark_online_quant` plugin fails to import** with a traceback on every launch. It
  is non-fatal and unrelated. Do not chase it.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs.

## Summary

**vLLM serves the real `Qwen/Qwen3.8-27B-FP8` on MI355X/gfx950, at TP=1 and
TP=2, with no code changes and no image upgrade — but only with `VLLM_ROCM_USE_AITER=0`.**

The interesting finding is not that it works; it is *how* it fails when it fails. The
expected failure mode was a model-support gap in vLLM 0.20.2, and that is wrong — the
engine recognises `Qwen3_5ForConditionalGeneration`, loads the E4M3 FP8
weights in 7.9 s, and reports 29.38 GiB. The actual defect is a **silent numerical one**:
AITER's FP8 block-scaled GEMM produces corrupted output while every health signal stays
green and throughput actually improves. A dashboard-driven bring-up would have shipped
this. Reading the generated text is what caught it.

TP=2 is worth using at this model size — per-rank weights halve (29.38 → 14.9 GiB), KV
capacity grows 2.3× (1.45M → 3.33M tokens), and single-stream decode gains 36%
(38.1 → 52.4 tok/s), with both GPUs confirmed at 100% in the same `rocm-smi` sample. That
is the opposite of the 4B result in this same folder, where TP=2 costs 4% and replication
is the right pattern.


