# `inference/tensorrtllm/llm` — TensorRT-LLM (NVIDIA only)

> Stack overview, scope note, and environment/venv conventions: [`../README.md`](../README.md).

## Overview & when to use

**TensorRT-LLM** is NVIDIA's own high-performance LLM serving stack. It compiles models
into NVIDIA-specific engines and provides vendor-tuned kernels, FP8/FP4 quantization paths,
tensor/pipeline/data/expert parallelism, multi-node serving, disaggregated prefill/decode,
and integration with Triton Inference Server and NVIDIA Dynamo. On an NVIDIA cluster it is a
top-tier candidate for maximum throughput and latency efficiency.

**It is NVIDIA-only, by design.** There is no ROCm build upstream, and the entire pip
dependency closure is CUDA 12/13 wheels. For LLM serving on an AMD host use
[`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/).

## Hardware support

| Target | Status |
|---|---|
| NVIDIA Hopper (H100/H200/GH200) | **Works** — `tensorrt_llm` 1.2.1 imports and the PyTorch backend loads a supported checkpoint |
| NVIDIA Blackwell / Ada Lovelace / Ampere | upstream targets (Ampere has no FP8 path) |
| **AMD ROCm (any GPU)** | **Not supported** — there is no ROCm build upstream |
| Embedding / reranker workloads | ❌ upstream even on NVIDIA — no first-class recipe. See [`../README.md`](../README.md) |

## Install

Versions this was built against: `tensorrt_llm` **1.2.1**, `torch` **2.9.1+cu128**,
`nvidia-modelopt` **0.37.0**, `flashinfer-python` 0.6.4, Python 3.12, CUDA 13.0.

### NVIDIA / CUDA — pip

The likeliest blocker is **not** the wheel — it is network egress. An HTTP proxy that
**403-blocks `pypi.nvidia.com`** makes the NVIDIA extra index fail until the proxy is unset.

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

**torch-clobber note (expected, leave it):** the install downgrades torch to
**`2.9.1+cu128`** — tensorrt-llm 1.2.1 pins the cu128 build. `cu128` runs
fine on a cu130 box, because CUDA is backward-compatible with the newer driver. **Do
not force torch back to cu130**; that breaks tensorrt-llm's pins.

**Never install this into a shared environment.** Its dependency closure includes the stock
CUDA `torch`, which would replace a ROCm torch and break every sibling ROCm stack. Upstream
recommends a pip constraints file if you must:

```bash
echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
pip install tensorrt_llm -c /tmp/torch-constraint.txt
```

Budget disk: the install consumes ~16 GB in the venv and builds a 2.5 GB `tensorrt_llm`
wheel plus a 3.4 GB `tensorrt_cu13_libs` wheel into `PIP_CACHE_DIR`. Point `PIP_CACHE_DIR`
at a large filesystem, not `/`.

### NVIDIA — NGC container

The release container is upstream's supported path (requires NGC credentials and the NVIDIA
Container Toolkit):

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:<TAG>

docker run --rm -it \
  --ipc host \
  --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8400:8400 \
  -v ~/.cache:/root/.cache:rw \
  --name tensorrt_llm \
  nvcr.io/nvidia/tensorrt-llm/release:<TAG> \
  /bin/bash

# sanity check inside the container
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

Pick the current monthly release tag, or a recent `rc` build when day-zero model support
matters.

## Run

### Probe the host

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=0
source .env_tensorrtllm/bin/activate
python llm/probe_tensorrtllm.py --device_id 0 --out $OUTPUT_DIR/tensorrtllm/llm/probe.json
```

**Expected output** on a working NVIDIA host (trimmed):

```
nvidia-smi      : FOUND at /usr/bin/nvidia-smi
  output        : NVIDIA H100 80GB HBM3, <driver>
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

Probe exit code is **0 only when TensorRT-LLM is genuinely usable** — a real NVIDIA device
*and* a CUDA-backed torch *and* a successful `import tensorrt_llm`. It exits 1 otherwise,
so the script doubles as a regression check.

### Serve

```bash
trtllm-serve <model> --host 0.0.0.0 --port 8400
```

Treat that one-liner as a smoke test, **not** a tuned configuration. For multi-GPU
production, move performance and parallelism options into the TensorRT-LLM YAML config.

### Load a model from Python

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="<model>", tensor_parallel_size=1)
llm.generate(["2 + 2 ="], SamplingParams(max_tokens=16, temperature=0.0))
```

A healthy run logs:

```
[TRT-LLM] [I] Using LLM with PyTorch backend
[TensorRT-LLM][INFO] [MemUsageChange] Allocated ... for max tokens in paged KV cache
```

Confirm the GPU is really used with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <N>` from a second
shell while the run is live.

## Model support caveat — `Qwen/Qwen3.8-27B-FP8`

**This checkpoint does not load on tensorrt_llm 1.2.1, on any hardware.** It declares
`model_type: "qwen3_5"`, which neither transformers 4.57.3 nor tensorrt_llm 1.2.1 registers,
so the load fails at config parsing before any FP8 handling:

```
KeyError: 'qwen3_5'
ValueError: ... model type `qwen3_5` but Transformers does not recognize this architecture ...
```

This is a version gap, not a hardware or FP8 problem. To exercise the load→generate path,
use a model whose architecture **is** registered — e.g. any `Qwen3ForCausalLM` checkpoint.

If direct FP8 loading of your own checkpoint is incompatible, start from the **BF16/base**
checkpoint and quantize with **NVIDIA Model Optimizer** (`nvidia-modelopt`, already in the
pip closure) to a TensorRT-LLM-supported FP8 recipe. Validate accuracy against
[`../../transformers/llm/`](../../transformers/llm/) or [`../../vllm/llm/`](../../vllm/llm/)
before measuring throughput.

## Multi-GPU

Add `tensor_parallel_size=N` to the `LLM(...)` call (or `--tp_size N` to `trtllm-serve`) and
re-verify residency across all N GPUs with `nvidia-smi`. Upstream also exposes PP, DP, EP,
multi-node MPI, disaggregated prefill/decode, and Triton / NVIDIA Dynamo integration.

## Checking whether a host qualifies

`probe_tensorrtllm.py` checks `nvidia-smi`, the `/dev/nvidia*` nodes, the `nvidia` kernel
module, `libcuda.so.1` / `libnvidia-ml.so.1` loadability, and `torch.version.cuda`. On a
host without NVIDIA hardware it reports:

```
nvidia-smi      : ABSENT
/dev/nvidia*    : none
  libcuda.so.1         -> MISSING
tensorrt_llm    : IMPORT FAILED -- ImportError: libcuda.so.1: cannot open shared object file

VERDICT: TensorRT-LLM is NOT usable on this host.
```

Exit code **1**. Run the host checks without the 16 GB install (just
`pip install -r ../requirements.txt`) with:

```bash
python llm/probe_tensorrtllm.py --no_check_import
```

Upstream's Linux install instructions also require host-level prerequisites no wheel can
supply: **CUDA Toolkit 13.2** with `CUDA_HOME` set, and possibly `cuda-compat-13-2`
depending on the NVIDIA driver version.

---

## Arguments

`probe_tensorrtllm.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--device_id` | `0` | Device ordinal to report |
| `--check_import` | `True` | Attempt `import tensorrt_llm` and capture any failure |
| `--no_check_import` | — | Skip the import attempt (host checks only, useful in a bare venv) |
| `--out` | `None` | Write a JSON artifact of the probe result |

## Output

A JSON probe artifact (when `--out` is given) recording `nvidia-smi` presence, device nodes,
NVIDIA soname loadability, torch CUDA/HIP provenance, and the `tensorrt_llm` import result.

---

## Notes & quirks

- **`pip install tensorrt-llm` succeeding is not a compatibility signal.** It exits **0** on a
  machine with zero NVIDIA hardware and installs 16 GB of CUDA wheels, because the artifact is
  a generic `manylinux_x86_64` wheel with no GPU-vendor guard. The failure is deferred all the
  way to `import`.
- **Budget the disk.** ~16 GB in the venv plus a 2.5 GB `tensorrt_llm` wheel and a 3.4 GB
  `tensorrt_cu13_libs` wheel in `PIP_CACHE_DIR`. Point `PIP_CACHE_DIR` at a large data
  filesystem, not `/`.
- **`torch.cuda.is_available()` is not a CUDA check.** ROCm PyTorch aliases `torch.cuda` onto
  HIP, so it returns `True` on an AMD box. Check `torch.version.cuda`/`torch.version.hip`.
- **Installing TensorRT-LLM into a shared env will break a ROCm stack.** Its closure includes
  the stock CUDA `torch`, which replaces a ROCm torch. Use the dedicated venv, or a pip
  constraints file (above).
- **Two CUDA generations are pulled at once** — `nvidia-*-cu12` *and* `nvidia-*-cu13` wheels.
  Expected; it makes the install heavy.
- **System MPI is required.** `mpi4py` in the closure links against the system `libmpi.so`,
  which is not a wheel — install `libopenmpi3 openmpi-bin libopenmpi-dev`.
- **Not supported on AMD/ROCm** — there is no ROCm code path at all. Use
  [`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/) instead.
