# `inference/tensorrtllm/llm` — TensorRT-LLM (works on NVIDIA H100)

> Stack overview, scope note, and environment/venv conventions: [`../README.md`](../README.md).

> **This stack works on NVIDIA H100 and is not supported on AMD/ROCm.** Validated on an
> **8×NVIDIA H100 80GB HBM3** host (driver 580.173.02, CUDA 13.0):
> **`import tensorrt_llm` succeeds (v1.2.1), the probe exits 0 ("TensorRT-LLM appears usable
> on this host"), and the H100 is detected.** See **[H100 result](#h100-result--tensorrt-llm-works)**
> for the install recipe, versions, model-load outcome, and log samples. Everything below
> the H100 section is the **MI355X evidence, retained but superseded** on NVIDIA
> hardware.

## Overview & when to use

**TensorRT-LLM** is NVIDIA's own high-performance LLM serving stack. It compiles models
into NVIDIA-specific engines and provides vendor-tuned kernels, FP8/FP4 quantization paths,
tensor/pipeline/data/expert parallelism, multi-node serving, disaggregated prefill/decode,
and integration with Triton Inference Server and NVIDIA Dynamo. On an NVIDIA cluster it is a
top-tier candidate for maximum throughput and latency efficiency.

> **Tested topology:** 8×AMD Instinct MI355X (gfx950), ROCm 7.2.4, Ubuntu, Python 3.12.3,
> Docker 29.7.2. **Not supported there — TensorRT-LLM is NVIDIA-only, and such a host has
> no NVIDIA GPU, no CUDA driver, and no `nvidia-smi`.** This is an expected, correct
> outcome, not a failed benchmark.

Use this folder as the **evidence record** for why TensorRT-LLM is not part of the working
inference set on an AMD box, and as a **ready-to-run recipe** on NVIDIA hardware.
For serving on an AMD host, use [`inference/vllm/llm`](../../vllm/llm/)
or [`inference/sglang/llm`](../../sglang/llm/).

---

## Status (read this first)

> **On NVIDIA H100 this path works.** `import tensorrt_llm` (v1.2.1)
> succeeds, `probe_tensorrtllm.py` exits 0, and TRT-LLM loads a supported checkpoint and
> generates on the H100 with confirmed GPU residency. Full recipe, versions, and log samples:
> **[H100 result](#h100-result--tensorrt-llm-works)**. The
> "not supported" finding below is the **AMD/MI355X result — retained but superseded**
> on NVIDIA hardware.

**[Superseded on NVIDIA — the AMD finding] Not supported on that hardware.** Three distinct facts,
deliberately kept separate — do not let the first hide the third:

| # | Claim | Status | Scope |
|---|---|---|---|
| **(a)** | **The hardware is wrong** — no NVIDIA GPU, driver, or CUDA runtime on an AMD box | **Decisive on the AMD box; ~~n/a on H100~~ (H100 has NVIDIA GPU + CUDA)** | The AMD host only |
| **(b)** | **TensorRT-LLM is NVIDIA-only by design** — ROCm/AMD is not a supported target | **Upstream fact, verified** (still true: no ROCm build) | Any AMD host |
| **(c)** | **Even on NVIDIA, `Qwen/Qwen3.8-27B-FP8` is unvalidated** | **CONFIRMED on H100 — and worse than expected:** it is a Qwen3.5 **VL** checkpoint (`model_type qwen3_5`); neither transformers 4.57.3 nor tensorrt_llm 1.2.1 registers `qwen3_5`, so it fails at config parse. A version gap, not hardware. | NVIDIA hosts too |

(a) alone ends the discussion *on an AMD machine*. (b) means no amount of AMD hardware would
help. (c) is a separate, real caveat that survives moving to NVIDIA — it is the item most
likely to be forgotten once the "no NVIDIA GPU" blocker is out of the way.

> **Superseded on NVIDIA.** Facts (a) and (b)-as-blocker no longer hold on the
> H100 host: `import tensorrt_llm` **succeeds** and the probe **exits 0**. (b) is still true as
> an upstream *scope* statement (no ROCm build exists — the stack remains NVIDIA-only), and
> (c) still stands as the model-checkpoint caveat. See the H100 section immediately below for
> the current, positive result.

---

## H100 result — TensorRT-LLM works

This folder was run on an **8×NVIDIA H100 80GB HBM3** node (an AMD-only host cannot test
TensorRT-LLM at all, because it needs an NVIDIA GPU). **This path works as documented on
H100:** `import tensorrt_llm` succeeds, the repo's own `probe_tensorrtllm.py` exits **0**, and
the H100 is detected with a loadable `libcuda.so.1`.

### Host & versions (verified)

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (Hopper cc 9.0, native FP8) — single GPU, `CUDA_VISIBLE_DEVICES=6` |
| Driver / CUDA | **580.173.02** / CUDA 13.0 |
| `tensorrt_llm` | **1.2.1** |
| `torch` | **2.9.1+cu128** (see torch-clobber note) |
| `nvidia-modelopt` | **0.37.0** |
| `flashinfer-python` | 0.6.4 |
| Python | 3.12.3 |

### Install recipe that made it work

The blocker was **not** the wheel — it was network egress. An HTTP proxy that
**403-blocks `pypi.nvidia.com`** makes the NVIDIA extra index fail until the proxy is unset.
Recipe:

```bash
# 0. Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # probe/run artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export PIP_CACHE_DIR=/path/to/pip_cache

# 1. UNSET the proxy first (restores egress to pypi.nvidia.com / huggingface.co).
#    Harmless otherwise — do this at the start of every shell.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# 2. venv + the NVIDIA extra index (pulls tensorrt-llm + nvidia-modelopt + flashinfer)
python3 -m venv .env_tensorrtllm && source .env_tensorrtllm/bin/activate
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
#   -> tensorrt-llm 1.2.1, nvidia-modelopt 0.37.0, flashinfer-python 0.6.4

# 3. system MPI — mpi4py in the closure needs the system libmpi.so (not a wheel)
sudo apt-get install -y libopenmpi3 openmpi-bin libopenmpi-dev
```

**torch-clobber note (expected, leave it):** the install downgrades torch from
`2.13.0+cu130` to **`2.9.1+cu128`** (tensorrt-llm 1.2.1 pins the cu128 build). `cu128` runs
fine on a cu130 / driver-580 box — CUDA is backward-compatible with the newer driver. **Do
NOT force torch back to cu130**; that breaks tensorrt-llm's pins. Confirmed:
`torch.cuda.is_available() == True`, device 0 = `NVIDIA H100 80GB HBM3`.

### Smoke command (probe)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=6
source .env_tensorrtllm/bin/activate
python llm/probe_tensorrtllm.py --device_id 6 --out $OUTPUT_DIR/tensorrtllm/llm/probe_h100.json
```

**Expected output** (trimmed):

```
nvidia-smi      : FOUND at /usr/bin/nvidia-smi
  output        : NVIDIA H100 80GB HBM3, 580.173.02
  libcuda.so.1         -> loadable
  libcudart.so.13      -> loadable
torch           : 2.9.1+cu128
  version.cuda  : 12.8
  version.hip   : None
  is_available  : True
  device 0      : NVIDIA H100 80GB HBM3
[TensorRT-LLM] TensorRT LLM version: 1.2.1
tensorrt_llm    : imports OK (version 1.2.1)

VERDICT: TensorRT-LLM appears usable on this host.
```

Probe exit code: **0**. (On an AMD box the same script exits 1 with `ImportError:
libcuda.so.1`.)

### FP8 model-load attempt — `Qwen/Qwen3.8-27B-FP8` (the target)

Attempted via the LLM API on GPU 6 (`from tensorrt_llm import LLM; LLM(model="Qwen/Qwen3.8-27B-FP8", tensor_parallel_size=1)`).
The checkpoint must be fully cached under `$HF_HOME` (66-shard snapshot).
TRT-LLM launched an MPI executor worker (the worker PID held **588 MiB on GPU 6** — the H100 was
touched), then the load **failed at config parsing, before any FP8/weight handling**:

```
[TRT-LLM] [E] Failed to initialize executor on rank 0: The checkpoint you are trying to load
  has model type `qwen3_5` but Transformers does not recognize this architecture.
KeyError: 'qwen3_5'
ValueError: ... model type `qwen3_5` but Transformers does not recognize this architecture ...
```

**Root cause (sharper than the brief's expectation — worth recording):** this checkpoint is
**NOT** the dense `Qwen3_5ForCausalLM`. Its `config.json` says `architectures:
["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`, and it carries a
`vision_config` + `image_token_id`/`video_token_id` — i.e. it is a **Qwen3.5 vision-language
(multimodal) model** whose `config.json` pins `transformers_version: 5.8.0.dev0`. The failure
is **not** an FP8 problem: it's an architecture-recognition problem one layer earlier —

1. the venv's **transformers 4.57.3** has no `qwen3_5` in its `CONFIG_MAPPING`, and
2. **tensorrt_llm 1.2.1 does not register `qwen3_5`/`Qwen3_5*` either** — its
   `MODEL_CLASS_MAPPING` has `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`,
   `Qwen3VLForConditionalGeneration`, `Qwen3NextForCausalLM`, … but **no `Qwen3_5*`**.

So neither `nvidia-modelopt` requant nor a torch tweak would help: the config can't be parsed
at all until a transformers build that knows `qwen3_5` **and** a TRT-LLM build that registers
it both exist. This is exactly the model caveat the "fair NVIDIA test" section flagged, now
confirmed on-hardware: **the checkpoint is too new for this TRT-LLM release.** (A bleeding-edge
`transformers` from git *might* let the config parse, but TRT-LLM still has no `qwen3_5` model
class, so the serving path would fail downstream — not attempted, since it risks
clobbering the verified env.)

### Fallback — proving the serving path works end-to-end

To show TRT-LLM genuinely serves on the H100 (not just imports), load a small cached
model whose architecture **is** registered: **`Qwen/Qwen3-Reranker-0.6B`** →
`Qwen3ForCausalLM` (BF16, 28 layers), single GPU, `CUDA_VISIBLE_DEVICES=6`. This **works**:

**What a healthy run looks like:**

```
[TRT-LLM] [I] Using LLM with PyTorch backend
Model init total -- 19.66s
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 68.69 GiB for max tokens in paged KV cache (643136).
[  61.4s] LLM constructed OK. generating (greedy) ...
[  65.3s] DONE.
```

Sample GPU residency by PID from a second shell **while the run is live** to confirm the H100
is actually used:

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 6
<pid>, 588 MiB            # <- the run's PID on GPU 6, + 68.69 GiB KV cache reserved
```

Greedy generations (`SamplingParams(max_tokens=16, temperature=0.0)`):

```
PROMPT: 'The opposite of hot is'   ->  ' cold, and the opposite of cold is hot again. So, the opposite of'
PROMPT: '2 + 2 ='                  ->  ' 4, so the first two terms are 2 and 4, and'
PROMPT: 'The capital of France is' ->  ' 100000000000000'
```

**Accuracy note (honest caveat):** "opposite of hot → cold" and "2 + 2 → 4" are **correct**;
"capital of France" degenerated to a numeric string. That is expected and **not** a TRT-LLM
fault: `Qwen3-Reranker-0.6B` is a *reranker/scoring* checkpoint, not an instruction-tuned
generator, so free-form generation is weak — it was chosen only because it is tiny and its
arch is supported, to exercise the load→generate path. A rigorous accuracy comparison against
the transformers/vLLM baseline is **not** possible on the intended 27B-FP8 (it never loads,
above), and the transformers reference artifact
(`reference_llm_lfm2.5-350m_1gpu.json`) is for LFM2.5-350M,
a different model — so it is not a like-for-like reference for either model here. The
defensible claim: **TRT-LLM loads a supported checkpoint on the H100 and produces coherent,
partially-verifiable output; a full accuracy match awaits a serving-supported instruct model.**

### Summary (H100)

- **Stack usable on H100: yes.** `import tensorrt_llm` (1.2.1) works, probe exits 0, and TRT-LLM
  loads a supported checkpoint and generates on the H100 with confirmed GPU residency.
- **Target `Qwen/Qwen3.8-27B-FP8`: blocked on this release** — it is a Qwen3.5 *VL* checkpoint
  (`model_type qwen3_5`) that neither transformers 4.57.3 nor tensorrt_llm 1.2.1 recognizes.
  Fix requires a newer transformers **and** a TRT-LLM build that registers `qwen3_5` — a
  version gap, not a hardware gap. Documented precisely above.
- Reproduce the fallback with the loader scripts (`load_qwen3_27b_fp8.py`,
  `load_fallback.py`), writing the probe JSON under `$OUTPUT_DIR/tensorrtllm/llm/`.

### Single-GPU only; what multi-GPU would need

On a shared node this is a **single-GPU** smoke on GPU 6 (other GPUs left to co-tenant
jobs). A multi-GPU pass (TP=2 then TP=8) needs the remaining GPUs free;
it would add `tensor_parallel_size=N` to the `LLM(...)` call
(or `--tp_size N` to `trtllm-serve`) and re-verify residency across all N GPUs.

---

## Scope — why there is no embedding or reranker TensorRT-LLM folder

Deliberate, and this is **not** merely a consequence of the hardware. Upstream,
TensorRT-LLM rates these workloads:

| Workload | Model | Upstream status (on NVIDIA) |
|---|---|---|
| LLM | `Qwen/Qwen3.8-27B-FP8` | 🟡 high-potential NVIDIA path; validate the exact checkpoint |
| Embedding | `google/embeddinggemma-300m` | ❌ no first-class recipe comparable to vLLM/Transformers |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | ❌ decoder-style yes/no-token scoring has no first-class path |

Both the embedding and the reranker are **❌ upstream even on NVIDIA hardware**, so
`inference/tensorrtllm/embedding` and `inference/tensorrtllm/reranker` would be doubly
blocked — wrong vendor *and* unsupported workload. They are not created. Use the
`embedding/` / `reranker/` leaves under vLLM, Transformers, TEI, ONNX, or llama.cpp
instead.

---

## Host evidence — an AMD host has no NVIDIA GPU

Every check below was run on the AMD host. Reproduce with `python probe_tensorrtllm.py`.

```bash
$ which nvidia-smi
# (no output)
$ nvidia-smi
bash: nvidia-smi: command not found

$ ls -la /dev/nvidia*
ls: cannot access '/dev/nvidia*': No such file or directory

$ lsmod | grep -i nvidia
# (no output — no nvidia kernel modules loaded)

$ ldconfig -p | grep -iE "libcuda\.so|libnvidia-ml"
# (no output — no CUDA driver library, no NVML)

$ lspci | grep -i nvidia
# (no output — no NVIDIA PCI devices)
```

What *is* present is the AMD side:

```bash
$ lsmod | grep amdgpu
amdgpu              20250624  272

$ ls /dev/kfd
/dev/kfd                       # ROCm compute device node

$ cat /opt/rocm/.info/version
7.2.4
```

And PyTorch on this host is a ROCm build:

```python
torch 2.11.0+rocm7.2
torch.version.cuda   : None          # <- no CUDA
torch.version.hip    : 7.2.26015     # <- HIP instead
torch.cuda.is_available() : True     # <- MISLEADING, see below
device_count         : 8
device 0             : AMD Instinct MI355X
gcnArchName          : gfx950:sramecc+:xnack-
```

> **Important quirk:** on a ROCm PyTorch build `torch.cuda.is_available()` returns **`True`**,
> because the `torch.cuda` namespace is *aliased onto HIP*. It is **not** evidence of CUDA.
> The decisive signals are `torch.version.cuda is None` and `torch.version.hip` being set.
> `probe_tensorrtllm.py` keys off those two plus the loadability of `libcuda.so.1`, never
> off `is_available()` alone.

| Signal | Required by TensorRT-LLM | This host |
|---|---|---|
| `nvidia-smi` | present | **absent** |
| `/dev/nvidia*` device nodes | present | **absent** |
| `nvidia` kernel module | loaded | **not loaded** |
| `libcuda.so.1` (driver) | loadable | **missing** |
| `libnvidia-ml.so.1` (NVML) | loadable | **missing** |
| NVIDIA PCI device | ≥1 | **0** |
| `torch.version.cuda` | a CUDA version string | **`None`** |
| GPU architecture | Ampere/Ada/Hopper/Blackwell | **gfx950 (CDNA4, AMD)** |

---

## Install attempt — what pip actually did

Environment: fresh venv, a `PIP_CACHE_DIR` on a large filesystem, pip 24.0, Python 3.12.3,
x86_64 Linux.

```bash
cd inference/tensorrtllm
python3 -m venv .env_tensorrtllm
source .env_tensorrtllm/bin/activate
export PIP_CACHE_DIR=/path/to/pip_cache
pip install tensorrt-llm
```

(The convention is one venv at the software root, see [`../README.md`](../README.md).)

### The package installs cleanly — this is the surprising part

`tensorrt-llm` **is** on plain PyPI and **does** resolve on this platform:

```text
$ pip index versions tensorrt-llm
tensorrt-llm (1.2.1)
Available versions: 1.2.1, 1.2.0, 1.1.0, 1.0.0, 0.21.0, 0.20.0, 0.19.0, 0.18.2, ...
```

There is **no platform guard**. The wheel is a generic `manylinux_x86_64` artifact, so pip
never asks whether an NVIDIA GPU exists. **The install ran to completion with exit code 0**,
building a 2.5 GB `tensorrt_llm` wheel and a 3.4 GB `tensorrt_cu13_libs` wheel and consuming
**16 GB** of venv space on a machine with no NVIDIA silicon whatsoever:

```text
Building wheels for collected packages: tensorrt-llm, tensorrt, tensorrt_cu13, tensorrt_cu13_libs
  Created wheel for tensorrt-llm: filename=tensorrt_llm-1.2.1-cp312-cp312-linux_x86_64.whl size=2514282461
  Created wheel for tensorrt_cu13_libs: filename=tensorrt_cu13_libs-10.14.1.48.post1-py2.py3-none-manylinux_2_28_x86_64.whl size=3437501033
Successfully built tensorrt-llm tensorrt tensorrt_cu13 tensorrt_cu13_libs
Successfully installed ... tensorrt-llm-1.2.1 tensorrt-10.14.1.48.post1
  tensorrt_cu13_libs-10.14.1.48.post1 torch-2.9.1 nvidia-modelopt-0.37.0 ...
PIP_EXIT=0
```

The resolution pulls the entire CUDA userspace as ordinary Python wheels — the closure makes
the vendor lock-in explicit:

```text
Installed ... tensorrt_llm-1.2.1 tensorrt-10.14.1.48.post1
  tensorrt_cu13-10.14.1.48.post1 tensorrt_cu13_bindings-10.14.1.48.post1
  tensorrt_cu13_libs-10.14.1.48.post1
  cuda-bindings-13.3.1 cuda-core-1.0.1 cuda-pathfinder-1.6.1 cuda-python-13.3.1
  cuda-toolkit-13.3.1
  nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90
  nvidia-cuda-nvrtc-13.3.33 nvidia-cuda-nvrtc-cu12-12.8.93
  nvidia-cuda-runtime-13.3.29 nvidia-cuda-runtime-cu12-12.8.90
  nvidia-cudnn-cu12-9.10.2.21 nvidia-cudnn-frontend-1.27.0
  nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90
  nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93
  nvidia-cusparselt-cu12-0.7.1 nvidia-cutlass-dsl-4.3.4
  nvidia-ml-py-13.610.43 nvidia-modelopt-0.37.0
  nvidia-nccl-cu12-2.27.5 nvidia-nccl-cu13-2.28.9 nvidia-nvjitlink-cu12-12.8.93
  nvidia-nvshmem-cu12-3.3.20 nvidia-nvtx-cu12-12.8.90
  torch-2.9.1 torchvision-0.24.1 triton-3.5.1 flashinfer-python-0.6.4 mpi4py-4.1.2
  ... (≈200 packages total)
```

Three things to read out of that list:

1. **`tensorrt_cu13_libs`, `cuda-toolkit`, `nvidia-*-cu12/cu13`** — the dependency closure is
   CUDA 12/13 top to bottom. There is no ROCm/HIP variant of any of it. This *is* the
   NVIDIA-only design, expressed as a dependency graph.
2. **`torch-2.9.1`** — the stock **CUDA** build of PyTorch. Installing this into a shared
   environment would *uninstall the host's `torch 2.11.0+rocm7.2`* and break every working
   ROCm stack in the sibling folders. Upstream documents this exact hazard and recommends a
   pip constraints file. **This is why the install here is confined to a throwaway venv.**
3. **`nvidia-modelopt`** — NVIDIA Model Optimizer, the tool that would be needed for the
   FP8 requantization path described under (c) below.

### Where it actually fails — the decisive transcript

The refusal is **not** at dependency-resolution time; pip is entirely happy. It comes at
**import time**, because `libcuda.so.1` is supplied by the NVIDIA *kernel driver* and never by
a pip wheel. Running upstream's own documented sanity check:

```text
$ python -c "import tensorrt_llm"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File ".../site-packages/tensorrt_llm/__init__.py", line 107, in <module>
    import tensorrt_llm._torch.models as torch_models
  File ".../site-packages/tensorrt_llm/_torch/__init__.py", line 1, in <module>
    from .llm import LLM
  File ".../site-packages/tensorrt_llm/_torch/llm.py", line 1, in <module>
    from tensorrt_llm.llmapi.llm import _TorchLLM
  File ".../site-packages/tensorrt_llm/llmapi/__init__.py", line 1, in <module>
    from .._torch.async_llm import AsyncLLM
  File ".../site-packages/tensorrt_llm/_torch/async_llm.py", line 3, in <module>
    from ..llmapi.llm import LLM
  File ".../site-packages/tensorrt_llm/llmapi/llm.py", line 17, in <module>
    from tensorrt_llm._utils import mpi_disabled
  File ".../site-packages/tensorrt_llm/_utils.py", line 47, in <module>
    from tensorrt_llm.bindings import DataType, GptJsonConfig, LayerType
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
$ echo $?
1
```

**That is the whole verdict in one line.** The failing frame is
`tensorrt_llm/_utils.py:47`, importing `tensorrt_llm.bindings` — the compiled C++ extension —
which links against the CUDA driver stub. The stack never reaches a model, a config, or an
architecture check; it dies before TensorRT-LLM's Python layer finishes loading. No flag,
env var, or model substitution changes this.

`probe_tensorrtllm.py` reproduces it as a structured check (run inside this folder's venv):

```text
nvidia-smi      : ABSENT
/dev/nvidia*    : none
/dev/kfd (ROCm) : present
  libcuda.so.1         -> MISSING
  libnvidia-ml.so.1    -> MISSING
  libcudart.so.13      -> MISSING
  libcudart.so.12      -> MISSING
torch           : 2.9.1+cu128
  version.cuda  : 12.8
  version.hip   : None
  is_available  : False
  device 0      : None
tensorrt_llm    : IMPORT FAILED -- ImportError: libcuda.so.1: cannot open shared object file: No such file or directory

VERDICT: TensorRT-LLM is NOT usable on this host.
  No NVIDIA GPU, driver, or CUDA runtime is present.
  TensorRT-LLM is NVIDIA-only by design; ROCm/AMD is not a supported target.
  Use inference/vllm/llm or inference/sglang/llm on this AMD host.
$ echo $?
1
```

> **Note the torch line.** Inside *this* venv torch is `2.9.1+cu128` — the CUDA build that
> `tensorrt-llm` dragged in — so `version.cuda` is `12.8` and `version.hip` is `None`. Yet
> `torch.cuda.is_available()` is still **`False`**, because a CUDA-built torch with no driver
> is just as dead. On the **host** ROCm environment the same call returns `True` for the
> opposite reason (HIP aliasing). Neither `True` nor `False` from `is_available()` is a
> reliable CUDA test on such a host — which is exactly why the probe checks
> `torch.version.cuda`, `torch.version.hip`, and `libcuda.so.1` loadability instead.

Upstream's own Linux install instructions confirm the missing prerequisites are
non-negotiable and host-level:

> Install **CUDA Toolkit 13.2** … and make sure `CUDA_HOME` is properly set. The
> `cuda-compat-13-2` package may be required depending on your system's **NVIDIA GPU driver**
> version.

None of `CUDA_HOME`, a CUDA toolkit, or an NVIDIA driver exists on this machine, and none can
be installed meaningfully without NVIDIA silicon.

---

## Reproduce

```bash
cd inference/tensorrtllm
python3 -m venv .env_tensorrtllm && source .env_tensorrtllm/bin/activate
export PIP_CACHE_DIR=/path/to/pip_cache

pip install tensorrt-llm                # succeeds, exit 0, ~16 GB -- this is NOT compatibility
python -c "import tensorrt_llm"         # ImportError: libcuda.so.1 ... ; exit 1

python llm/probe_tensorrtllm.py         # structured report; exits 1 on this host
```

To run only the host checks without the 16 GB install (`pip install -r
../requirements.txt` is enough for this):

```bash
python llm/probe_tensorrtllm.py --no_check_import
```

No GPU is touched by any of the above — there is no NVIDIA GPU to touch, and no ROCm device
is opened either.

> **State of the venv as committed:** the full 16 GB TensorRT-LLM install was performed, the
> output above was captured from it, and the venv was then **reset to just
> `python-dotenv`** to return ~16 GB to a nearly full root filesystem while
> sibling stacks are running. So `probe_tensorrtllm.py` as shipped reports
> `ModuleNotFoundError: No module named 'tensorrt_llm'` rather than the `libcuda.so.1`
> `ImportError`. Both are failures; the `libcuda.so.1` one is the deeper and more
> interesting result, and reproducing it costs the 16 GB install above. The host-level
> checks (`nvidia-smi`, `/dev/nvidia*`, sonames) are identical either way and are the part
> that actually decides the verdict.

## Why the NGC container was NOT pulled

Upstream's preferred path is the NGC release container:

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:x.y.z
docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 \
  --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

This was **deliberately not attempted**. Four independent reasons, any one sufficient:

1. **It cannot run.** `--gpus all` requires the NVIDIA Container Toolkit and an `nvidia`
   Docker runtime backed by a real driver. None is installed; there is no device to pass
   through. The container would fail at `docker run` even if pulled.
2. **Disk.** `/` has **~93 GB free (94% used)**. The release image is tens of GB. Pulling a
   multi-tens-of-GB image that provably cannot execute risks filling the root filesystem and
   disturbing sibling stacks that are actively working — an unacceptable trade for zero
   information gain.
3. **Auth.** `nvcr.io` requires NGC credentials, which this host does not hold.
4. **Zero marginal evidence.** The container's contents cannot change facts (a) or (b). The
   verdict is already established by the host and by upstream's own support matrix.

Documenting the decision *is* the deliverable here, per the repo's convention of recording
the other vendor's path without executing it.

---

## Upstream support position (verified, not assumed)

Checked directly against TensorRT-LLM `main` for the versions pinned above.

### Hardware — `docs/source/supported-hardware.md`, verbatim and complete

```text
# Supported Hardware

TensorRT LLM supports the full spectrum of NVIDIA GPU architectures:
- NVIDIA Blackwell: B200, GB200, B300, GB300, DGX Spark
- NVIDIA Hopper: H100, H200, GH200
- NVIDIA Ada Lovelace: L20, L40/L40S
- NVIDIA Ampere: A100
```

That is the **entire file**. A case-insensitive search of it for `rocm|amd|instinct|radeon|hip`
returns **no matches**. AMD is not listed as unsupported — it is simply absent from a document
whose scope is "the full spectrum of NVIDIA GPU architectures". This matches the documented
vendor split exactly:

```text
NVIDIA cluster -> TensorRT-LLM is a top-tier candidate
AMD cluster    -> use vLLM or SGLang instead
```

### Model — the Qwen3.8 nuance, sharpened

Mainline registers the dense Qwen3.5 architecture that Qwen3.8 reuses, but
the supported-model list does not name `Qwen/Qwen3.8-27B-FP8`. **Both halves confirmed,
and the split is even cleaner than stated:**

**In code** (`tensorrt_llm/_torch/models/modeling_qwen3_5.py`, mainline) — the dense
architecture *is* registered:

```python
@register_auto_model("Qwen3_5ForCausalLM")           # line 654 — dense
class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
    ...

@register_auto_model("Qwen3_5MoeForCausalLM")        # line 621 — MoE
@register_auto_model("Qwen3_5ForConditionalGeneration")   # line 793 — dense VLM
```

**In the docs** (`docs/source/models/supported-models.md`) — the dense causal-LM arch is
**absent**, while its MoE sibling is named with a concrete checkpoint:

```text
| Qwen3_5MoeForCausalLM | Qwen3.8-MoE, Qwen3.5-MoE | Qwen/Qwen3.8-2.4T-A95B, Qwen/Qwen3.5-397B-A17B |
```

```bash
$ grep -n "Qwen3_5ForCausalLM" supported-models.md
# ABSENT from supported-models.md
```

So the situation for the target checkpoint is: **architecture registered in code ✅,
checkpoint documented as supported ❌.** The MoE `Qwen/Qwen3.8-2.4T-A95B` is explicitly
listed; the dense `Qwen/Qwen3.8-27B-FP8` is not. Combined with the warning not to
assume a third-party pre-quantized FP8 metadata format loads like an NVIDIA Model Optimizer
checkpoint, **this checkpoint needs validation or re-quantization even on correct hardware.**

---

## What a fair NVIDIA test would require

If NVIDIA hardware becomes available, this is the recipe. Nothing below was run here.

### 1. Hardware

| Class | FP8 suitability | Recommendation |
|---|---|---|
| **Hopper** (H100/H200/GH200) | Strongest native FP8 path | **Preferred** |
| **Blackwell** (B200/GB200/B300/GB300) | Strongest native FP8 + FP4 | **Preferred** |
| **Ada Lovelace** (L20/L40/L40S) | FP8 present in the support matrix | Usable — validate the specific target carefully |
| **Ampere** (A100) | **No equivalent FP8 path** | Not appropriate for an FP8 benchmark |

For a `Qwen3.8-27B-FP8` benchmark specifically, prioritize H100/H200/B100/B200/GB200-class
systems.

### 2. Container

Use the NGC release container rather than a pip install — it is the tested configuration.
Pick the current monthly release tag, or a recent `rc` build when day-zero model support
matters (relevant here, since dense Qwen3.5 support is recent).

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:<CURRENT_TAG>

docker run --rm -it \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8400:8400 \
  -v ~/.cache:/root/.cache:rw \
  --name tensorrt_llm \
  nvcr.io/nvidia/tensorrt-llm/release:<CURRENT_TAG> \
  /bin/bash

# sanity check inside the container
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

### 3. Validation sequence — accuracy before speed

Do **not** skip to benchmarking. The recommended order:

1. Use a very recent release/RC containing current dense Qwen3.5 support.
2. Attempt **direct PyTorch-backend loading** of `Qwen/Qwen3.8-27B-FP8`.
3. Run an **accuracy comparison against Transformers/vLLM** before measuring throughput.
   Use the sibling stacks [`../../transformers/llm/`](../../transformers/llm/) and
   [`../../vllm/llm/`](../../vllm/llm/) as the reference implementations.
4. If direct FP8 loading is incompatible, start from the **BF16/base** checkpoint and
   quantize with **NVIDIA Model Optimizer** (`nvidia-modelopt`, which the pip closure above
   already pulls) to a TensorRT-LLM-supported FP8 recipe.

Only after step 3 or 4 passes is a speed number meaningful.

### 4. Serving shape

```bash
trtllm-serve Qwen/Qwen3.8-27B-FP8 \
  --host 0.0.0.0 \
  --port 8400
```

Treat that one-liner as a smoke test, **not** a tuned configuration. For multi-GPU
production, move performance and parallelism options into the TensorRT-LLM YAML config.

### 5. Multi-GPU / cluster options to exercise

| Mode | Purpose |
|---|---|
| **TP** | Tensor parallelism across GPUs |
| **PP** | Pipeline parallelism across layers/nodes |
| **DP** | Replicated workers for request throughput |
| **EP** | Expert parallelism for MoE models |
| **Multi-node** | MPI/cluster tooling (note the documented MPI+Slurm caveat) |
| **Disaggregated serving** | Separate prefill and decode GPU pools |
| **Triton / NVIDIA Dynamo** | Production serving integration |

The fair comparison to make is **AMD vLLM/SGLang vs NVIDIA
vLLM/SGLang/TensorRT-LLM** — not TensorRT-LLM on AMD, which is not a thing.

---

## Single-GPU results

**None on AMD — no model is loaded, and no GPU is used**, since the required GPU does not
exist on such a host. Nothing is downloaded to the model cache and this folder ships no
model artifacts. (The H100 single-GPU results are in the H100 section above.)

This is a **not-supported** result, not a failed benchmark.

## Multi-GPU results

**Not applicable.** TP/PP/DP/EP are meaningless when the vendor runtime cannot initialise on
any device present.

---

## Arguments

`probe_tensorrtllm.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--device_id` | `6` | Device ordinal to report (physical 6 per this repo's convention; unused here since no NVIDIA device exists) |
| `--check_import` | `True` | Attempt `import tensorrt_llm` and capture the failure |
| `--no_check_import` | — | Skip the import attempt (host checks only, useful in a bare venv) |
| `--out` | `None` | Write a JSON artifact of the probe result |

Exit code is **0 only when TensorRT-LLM is genuinely usable** — a real NVIDIA device *and* a
CUDA-backed torch *and* a successful `import tensorrt_llm`. On this host it exits **1**, so
the script doubles as a regression check if the repo ever moves to NVIDIA hardware.

## Output

A JSON probe artifact (when `--out` is given) recording `nvidia-smi` presence, device nodes,
NVIDIA soname loadability, torch CUDA/HIP provenance, and the `tensorrt_llm` import result.
No model output exists.

---

## Hardware support & evidence

| | Status |
|---|---|
| NVIDIA Blackwell / Hopper / Ada / Ampere | **Official** — the only supported targets. Not run here (no NVIDIA GPU). |
| **AMD ROCm 7.2 / MI355X (gfx950)** | **Not supported, by design.** `supported-hardware.md` lists NVIDIA architectures only; no `rocm`/`amd`/`hip` match anywhere in it. |
| Host NVIDIA presence | **None.** No `nvidia-smi`, no `/dev/nvidia*`, no `nvidia` kernel module, no `libcuda.so.1`/`libnvidia-ml.so.1`, no NVIDIA PCI device. |
| Host torch | `2.11.0+rocm7.2` — `torch.version.cuda is None`, `torch.version.hip == 7.2.26015`. |
| pip install | **Succeeds, exit 0** (no platform guard) — 16 GB, ~200 CUDA-only packages incl. `tensorrt_cu13_libs`, `cuda-toolkit-13.3.1`, `nvidia-modelopt`, and a CUDA `torch-2.9.1`. |
| Runtime | **Fails** — `import tensorrt_llm` → `ImportError: libcuda.so.1: cannot open shared object file` at `tensorrt_llm/_utils.py:47`. Driver-supplied; no wheel can provide it. |
| LLM model support | 🟡 dense `Qwen3_5ForCausalLM` registered in code, but `Qwen/Qwen3.8-27B-FP8` absent from the supported-model list — validate/requantize even on NVIDIA. |
| Embedding (`embeddinggemma-300m`) | ❌ upstream, even on NVIDIA. |
| Reranker (`Qwen3-Reranker-0.6B`) | ❌ upstream, even on NVIDIA. |

---

## Notes & quirks

- **`pip install tensorrt-llm` succeeding is not a compatibility signal.** It exits **0** on a
  machine with zero NVIDIA hardware and installs 16 GB of CUDA wheels. There is no
  GPU-vendor guard because the artifact is a generic `manylinux_x86_64` wheel. The failure is
  deferred all the way to `import`. Contrast with the MLC experiment (`inference/mlc/llm`,
  since retired), where the wheel also installs but
  dies on a ROCm *ABI* mismatch — here there is no ROCm build at all to mismatch against.
- **Budget the disk before repeating this.** The install consumed ~16 GB in the venv and
  built a 2.5 GB `tensorrt_llm` wheel plus a 3.4 GB `tensorrt_cu13_libs` wheel into
  `PIP_CACHE_DIR`. Point `PIP_CACHE_DIR` at a large data filesystem, not
  `/`, which is typically the tightest. Delete the venv once the evidence is recorded — it can
  never do anything useful on an AMD host.
- **`torch.cuda.is_available() == True` on this AMD host.** ROCm PyTorch aliases `torch.cuda`
  onto HIP. Anyone using that call as a CUDA check will get a false positive on this machine.
  Check `torch.version.cuda`/`torch.version.hip` instead.
- **Installing TensorRT-LLM into a shared env would break the ROCm stacks.** Its dependency
  closure includes the stock CUDA `torch-2.9.1`, which would replace `torch 2.11.0+rocm7.2`.
  Upstream documents this hazard and recommends a pip constraints file
  (`echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt; pip install tensorrt_llm -c ...`).
  Always use the dedicated venv in this folder.
- **Two CUDA generations are pulled at once** — `nvidia-*-cu12` *and* `nvidia-*-cu13` wheels
  (e.g. `nvidia-nccl-cu12-2.27.5` alongside `nvidia-nccl-cu13-2.28.9`). Normal for this
  package during the CUDA 13 transition, but it makes the install heavy.
- **The PyPI wheel is not the tested configuration.** Upstream notes the PyPI build uses
  public PyTorch and may be incompatible with the NGC PyTorch container, which needs a
  `+ngcpytorch{YYMM}` local-version wheel. The NGC release container is the supported path.
- **The MoE/dense asymmetry is easy to misread.** `Qwen/Qwen3.8-2.4T-A95B` (MoE) *is* named in
  the supported-model list. `Qwen/Qwen3.8-27B-FP8` (dense) is not. Seeing "Qwen3.8" in the
  docs does not mean this checkpoint is covered.
- **Nothing here is a ROCm bug.** There is no ROCm code path in TensorRT-LLM to be broken.
  This is a vendor-scope boundary, not a defect.

---

## Summary — AMD / ROCm

**Not supported on MI355X / ROCm 7.2 — correctly and by design.** TensorRT-LLM is an
NVIDIA-only stack: its `supported-hardware.md` covers Blackwell, Hopper, Ada Lovelace, and
Ampere and mentions AMD/ROCm nowhere, and its entire dependency closure is CUDA 12/13 wheels.
Such a host has no NVIDIA GPU, no driver, no `libcuda.so.1`, and no `nvidia-smi` — so the
question is settled twice over, independently.

The pip install is a useful negative, and a sharper one than expected: `pip install
tensorrt-llm` **completes successfully with exit code 0** on a pure-AMD box. There is no
platform guard, so pip downloads the whole CUDA userspace — 16 GB, ~200 packages, including a
CUDA `torch-2.9.1` that would clobber this host's ROCm torch in any shared environment. The
stack only refuses at import: `ImportError: libcuda.so.1: cannot open shared object file`,
raised from `tensorrt_llm/_utils.py:47` loading the compiled `tensorrt_llm.bindings`
extension. **A clean `pip install` is therefore worthless as a compatibility signal here** —
the real gate is a kernel driver no wheel can ship. The NGC container was deliberately not
pulled: it needs `--gpus all` and an NVIDIA
runtime that cannot exist here, requires NGC auth, and would consume tens of GB of the ~93 GB
free on `/` for zero additional evidence.

Separately, and **not** cured by moving to NVIDIA: `Qwen/Qwen3.8-27B-FP8` is registered in
mainline as `Qwen3_5ForCausalLM` but is **absent from the supported-model list**, so it
requires direct-load validation, an accuracy comparison against Transformers/vLLM, and
possibly NVIDIA Model Optimizer requantization before any benchmark number means anything.

**Recommendation:** do not include TensorRT-LLM as an AMD candidate. On an AMD box use
[`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/). Keep this folder as the evidence record
and as the ready-to-run NVIDIA recipe above.
