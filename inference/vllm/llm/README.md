# `infer_llm_vllm.py` — LLM chat/generation served by vLLM

## Overview & when to use

Serves a causal LLM through vLLM's OpenAI-compatible `POST /v1/chat/completions`
endpoint. `infer_llm_vllm.py` is a small client that sends either a free-form chat prompt
or a prompted single-label classification, and prints the generated text along with
finish reason, latency, decode rate and token usage.

Use vLLM here when you want **one serving stack for all three workloads** — generation
(this folder), embeddings (`inference/vllm/embedding/`) and reranking
(`inference/vllm/reranker/`).

The target model is **`Qwen/Qwen3.8-27B-FP8`**. On gfx950 it works at TP=1 and TP=2, but
**only with `VLLM_ROCM_USE_AITER=0`** — with AITER on the server is healthy but every
completion is corrupt. See *The FP8 27B result on gfx950* and *Notes & quirks*.
`Qwen/Qwen3-4B` is used alongside as a smaller dense-model control.

## Install

**There is no ROCm vLLM wheel.** PyPI `vllm` publishes CUDA-only wheels that hard-depend on
CUDA torch, and `repo.radeon.com` publishes none, so on ROCm the pip/venv route is
unavailable (a gfx950 source build works but takes hours) and the route that works is a
**container**. Known-working image:
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

Two things to note in that command:

- The image has **no `render` group**, so the canonical `--group-add render` fails with
  `unable to find group render`. Pass the host's **numeric** GIDs instead (typically
  `video`=44, `render`=993) — that is what the `getent` substitutions above do.
- Pinning GPUs by **render node** (`renderD128`, `renderD136` — match them to your own
  cards with `ls /dev/dri`) rather than by `HIP_VISIBLE_DEVICES` means the container sees
  exactly those two `gfx950` devices and cannot touch any other card on the box.

**A newer image is not needed** — this build already contains
`Qwen3_5ForConditionalGeneration` and loads the checkpoint. If output is garbage, set
`VLLM_ROCM_USE_AITER=0` rather than reaching for `vllm/vllm-openai-rocm:nightly`; the bug is
in an AITER kernel.

Versions inside that image:

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
pip install -r ../requirements.txt
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

## Serve

Single GPU (run inside the container):

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

Two GPUs, tensor parallel:

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
default) the server is healthy but every completion is corrupt — see *The FP8 27B
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

## The FP8 27B result on gfx950 — works, but only with `VLLM_ROCM_USE_AITER=0`

**With AITER on (the gfx950 default), every completion is corrupt.** `/health` returns
200, `/v1/models` lists the model, the metrics look normal — and the text is garbage:

```
--- response ---
WePGG G/CO -ofU + of/v*';FVFV F G Y-vP EVIFa +-b */=ACAB<CJK>Y WEOfoci2 ...
```

It is not a sampling artifact: it reproduces at `temperature=0`.
**`VLLM_ROCM_USE_AITER_MOE=0` does not fix it** — the fault is in the FP8 block-scaled GEMM,
not the MoE path. **`VLLM_ROCM_USE_AITER=0` fixes it completely.** The setting picks the
kernel:

| `VLLM_ROCM_USE_AITER` | Kernel selected for `Fp8LinearMethod` | Output |
|---|---|---|
| `1` (gfx950 default) | `AiterFp8BlockScaledMMKernel` | **garbage** |
| `1` + `..._MOE=0` | `AiterFp8BlockScaledMMKernel` | **garbage** |
| `0` | `TritonFp8BlockScaledMMKernel` | **correct** |

**This failure is silent** — the server is healthy, the metrics look normal, and only
reading the generated text reveals it. Check real generated text as part of ROCm bring-up;
a passing `/health` says nothing about numerics.

## Single-GPU and TP=2 on gfx950

Both models serve at TP=1 and TP=2. `--tensor-parallel-size 2` needs no other flag — no
RCCL tuning, no `--distributed-executor-backend` override — beyond the mandatory
`VLLM_ROCM_USE_AITER=0` for the FP8 checkpoint. For a model that fits comfortably on one GPU,
replicate (one server per GPU behind a load balancer) instead of sharding.

## Arguments / flags

Serve-side flags:

| Flag | Value | Meaning |
|---|---|---|
| `--tensor-parallel-size` | `1` / `2` | Shards weights and KV across N GPUs; halves per-rank weights and roughly doubles KV capacity |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Benchmark-layout port for the LLM server |
| `--served-model-name` | `qwen3-4b` / `qwen38-27b-fp8` | Short alias clients pass as `model`, decoupling the API from the HF repo id |
| `--max-model-len` | `8192` / `32768` | Context window. Lower it to cut KV footprint; the 27B checkpoint supports 32K |
| `--gpu-memory-utilization` | `0.45` | Fraction of VRAM vLLM preallocates. `0.45` lets two servers share a card; the default is `0.9` |
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

## Hardware support

- **AMD, working at 27B scale.** 1× and 2× AMD Instinct MI355X (`gfx950:sramecc+:xnack-`,
  288 GiB, 256 CUs), host ROCm 7.2.4, container ROCm 7.0.2, vLLM
  `0.20.2rc1.dev253+g1ff9d3353`, torch `2.9.1.dev20251204+rocm7.0.2`,
  `torch.version.hip 7.0.51831`. `Qwen/Qwen3.8-27B-FP8` serves correct text at TP=1 and
  TP=2, as does `Qwen/Qwen3-4B`.
- **NVIDIA: the pip wheel** — see the H100 section below. The same `vllm serve`
  command also applies with the `vllm/vllm-openai:latest` container.

## H100 (NVIDIA)

vLLM `0.27.1` (pip / CUDA 13.0) serves `Qwen/Qwen3.8-27B-FP8` on **one** H100 80GB. The
checkpoint is a `Qwen3_5ForConditionalGeneration` vision-language + Mamba/GDN-hybrid model,
which is why `--max-num-seqs 256` is needed below.

### Install (pip route — no container needed)

The ROCm story above is container-only; on **NVIDIA the pip wheel is native and works**.
Unset any proxy first (`unset HTTP_PROXY HTTPS_PROXY ...` — pypi.nvidia.com is commonly
proxy-blocked):

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy          # -> the current CUDA 13 build (no --index-url)
pip install vllm                 # -> vllm 0.27.1 (pulls flashinfer/cutlass-dsl[cu13]); torch stays 2.13.0+cu130
pip install python-dotenv        # for the client
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA-only) |
| torch | `2.13.0+cu130` (native CUDA 13.0; **not** clobbered by the vLLM install) |
| transformers | `5.15.1` (bundled; **knows `qwen3_5` / `qwen3_5_vision` / `qwen3_5_text`**) |
| CUDA | 13.0, H100 80GB HBM3, cc(9,0) |

### One config change required — the Mamba cache (VRAM-driven)

Because the text backbone is a **GDN/Mamba hybrid**, *each decode sequence needs one Mamba
cache block*. On a single 80GB card the default `max_num_seqs=1024` exceeds the available
Mamba blocks and the engine aborts at CUDA-graph capture:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (694). Each decode
sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. Please
lower max_num_seqs to at most 694 or increase gpu_memory_utilization.
```

Fix: pass **`--max-num-seqs 256`** (well under 694) on an 80 GB card. Do **not** pass
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
[api_server.py:678] Supported tasks: ['generate']
INFO:     Application startup complete.
```

### Expected generated text (H100 — coherent and correct)

```
prompt : "What is the capital of France? Answer in one word."  (temperature 0)
--- response ---
User asks: ... Capital is Paris. Final: Paris.
</think>

Paris
```

Prompted classification (`--classify`, `temperature=0`) returns the correct label:
`network_outage`. The AITER corruption is ROCm-only and does not apply here.

### GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # during decode
```

Only the selected GPU is touched.

### Multi-GPU on NVIDIA

The NVIDIA commands here are single-GPU. For tensor parallelism add
`--tensor-parallel-size N`; the `qwen3_5` text backbone's head/GDN divisibility by N must
be checked before launching.

## Notes & quirks

- **`VLLM_ROCM_USE_AITER=0` is mandatory for the FP8 27B.** With AITER on, the server is
  healthy and fast, and the output is corrupt.
- **`VLLM_ROCM_USE_AITER_MOE=0` is the wrong workaround here.** It is the documented fix
  for the AITER MoE corruption bug, and it does *not* help: the fault is in
  `AiterFp8BlockScaledMMKernel`, the FP8 block-scaled **GEMM**, not the MoE path.
- **A passing `/health` says nothing about numerics.** The broken configuration returns
  HTTP 200, lists the model on `/v1/models` and reports normal metrics. Only reading
  generated text catches it — make that a required step of every ROCm bring-up.
- **AITER is not broken in general.** The same build serves dense `Qwen/Qwen3-4B` correctly
  *with AITER enabled*. Only the FP8 block-scaled GEMM path is affected.
- **Do not pass `--quantization fp8`.** The checkpoint is already FP8; vLLM reads
  `quantization_config` and selects `fp8` on its own.
- **`--trust-remote-code` is required** for the `qwen3_5` checkpoint.
- **No `render` group in the image.** `--group-add render` fails with `unable to find group
  render`; pass the host's numeric GIDs via `getent` instead.
- **AITER JIT-builds on first launch**, so the first cold start on this image is slower than
  later ones in the same container.
- **The `quark_online_quant` plugin fails to import** with a traceback on every launch. It
  is non-fatal and unrelated. Do not chase it.
- **`Qwen3VLVideoProcessorInitKwargs` `min_frames`/`max_frames` log lines** are harmless VL
  processor docstring warnings, not errors.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs.


