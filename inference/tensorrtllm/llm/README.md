# `inference/tensorrtllm/llm` — TensorRT-LLM engine build and serving

Compiles a model into an NVIDIA-specific engine and serves it with vendor-tuned kernels,
FP8/FP4 quantization, tensor/pipeline/data/expert parallelism, multi-node serving,
disaggregated prefill/decode, and Triton / NVIDIA Dynamo integration.

**Hardware:** NVIDIA only, by design — Hopper (H100/H200/GH200) verified with
`tensorrt_llm` 1.2.1; Blackwell / Ada / Ampere are upstream targets (Ampere has no FP8 path).
There is no ROCm build; on AMD use [`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/).

## Files

- `probe_tensorrtllm.py` — checks `nvidia-smi`, `/dev/nvidia*`, the `nvidia` kernel module,
  `libcuda.so.1` / `libnvidia-ml.so.1` loadability, torch CUDA/HIP provenance, and
  `import tensorrt_llm`. Exit 0 only when the stack is genuinely usable, so it doubles as a
  regression check.

## Setup

Built against `tensorrt_llm` **1.2.1**, `torch` **2.9.1+cu128**, `nvidia-modelopt` **0.37.0**,
`flashinfer-python` 0.6.4, Python 3.12, CUDA 13.0.

```bash
export OUTPUT_DIR=/path/to/outputs     # probe/run artifacts
export HF_HOME=/path/to/hf_cache       # model cache
export PIP_CACHE_DIR=/path/to/pip_cache

# Unset the proxy first — a proxy that 403s pypi.nvidia.com makes the extra index fail.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

python3 -m venv .env_tensorrtllm && source .env_tensorrtllm/bin/activate
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm

# mpi4py in the closure links the system libmpi.so, which is not a wheel.
sudo apt-get install -y libopenmpi3 openmpi-bin libopenmpi-dev
```

Budget the disk: ~16 GB in the venv plus a 2.5 GB `tensorrt_llm` wheel and a 3.4 GB
`tensorrt_cu13_libs` wheel in `PIP_CACHE_DIR`. Point `PIP_CACHE_DIR` at a large filesystem,
not `/`. The install downgrades torch to `2.9.1+cu128`, which tensorrt-llm 1.2.1 pins; cu128
runs fine on a cu130 host, and forcing torch back to cu130 breaks the pins.

**Never install this into a shared environment** — its closure includes the stock CUDA
`torch`, which replaces a ROCm torch and breaks every sibling ROCm stack. If you must, use a
constraints file:

```bash
echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
pip install tensorrt_llm -c /tmp/torch-constraint.txt
```

Upstream also requires host-level prerequisites no wheel supplies: CUDA Toolkit 13.2 with
`CUDA_HOME` set, and possibly `cuda-compat-13-2` depending on the driver version.

### NGC container (upstream's supported path)

Requires NGC credentials and the NVIDIA Container Toolkit. Pick the current monthly release
tag, or a recent `rc` build when day-zero model support matters:

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

python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

## Run

Probe the host first:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=0
source .env_tensorrtllm/bin/activate
python llm/probe_tensorrtllm.py --device_id 0 --out $OUTPUT_DIR/tensorrtllm/llm/probe.json
```

```
nvidia-smi      : FOUND at /usr/bin/nvidia-smi
  libcuda.so.1         -> loadable
torch           : 2.9.1+cu128
  version.cuda  : 12.8
  device 0      : NVIDIA H100 80GB HBM3
tensorrt_llm    : imports OK (version 1.2.1)

VERDICT: TensorRT-LLM appears usable on this host.
```

To run the host checks in a bare venv without the 16 GB install
(`pip install -r ../requirements.txt`), pass `--no_check_import`.

Serve — a smoke test, not a tuned configuration; move performance and parallelism options
into the TensorRT-LLM YAML config for production:

```bash
trtllm-serve <model> --host 0.0.0.0 --port 8400
```

Or load from Python:

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="<model>", tensor_parallel_size=1)
llm.generate(["2 + 2 ="], SamplingParams(max_tokens=16, temperature=0.0))
```

A healthy run logs `Using LLM with PyTorch backend` and a paged-KV-cache allocation. Confirm
the GPU is really used from a second shell:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <N>
```

For multi-GPU add `tensor_parallel_size=N` to the `LLM(...)` call (or `--tp_size N` to
`trtllm-serve`), then re-verify residency across all N GPUs.

## Arguments

`probe_tensorrtllm.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--device_id` | `0` | Device ordinal to report |
| `--check_import` | `True` | Attempt `import tensorrt_llm` and capture any failure |
| `--no_check_import` | — | Host checks only — useful in a bare venv |
| `--out` | `None` | Write a JSON artifact of the probe result |

With `--out`, the artifact records `nvidia-smi` presence, device nodes, NVIDIA soname
loadability, torch CUDA/HIP provenance, and the `tensorrt_llm` import result.

## Notes

- **Model support is by registered architecture.** `Qwen/Qwen3.8-27B-FP8` declares
  `model_type: "qwen3_5"`, which tensorrt_llm 1.2.1 does not register, so it fails at config
  parsing with `KeyError: 'qwen3_5'` before any FP8 handling. Use a checkpoint whose
  architecture is registered (for example any `Qwen3ForCausalLM`), or start from the BF16
  base checkpoint and quantize with NVIDIA Model Optimizer (`nvidia-modelopt`, already in the
  closure) to a supported FP8 recipe — then validate accuracy against
  [`../../transformers/llm/`](../../transformers/llm/) before measuring throughput.
- **`torch.cuda.is_available()` is not a CUDA check.** ROCm PyTorch aliases `torch.cuda` onto
  HIP, so it returns `True` on an AMD box. Check `torch.version.cuda` / `torch.version.hip`.
- **Two CUDA generations are pulled at once** — `nvidia-*-cu12` and `nvidia-*-cu13` wheels.
  Expected; it makes the install heavy.
