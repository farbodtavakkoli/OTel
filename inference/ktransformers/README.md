# `inference/ktransformers` — CPU-GPU heterogeneous MoE serving

[KTransformers](https://github.com/kvcache-ai/ktransformers) runs giant **Mixture-of-Experts**
models — DeepSeek-V3/R1 671B, Kimi-K2, Qwen3-MoE — on modest GPUs by offloading the MoE
experts to CPU DRAM (AMX / AVX512 / BLIS kernels, NUMA-aware) while attention and the active
weights stay on the GPU. The inference path is two pip packages: `kt-kernel` (the CPU MoE
kernels) and `sglang-kt` (the kvcache-ai SGLang fork, which adds `--kt-*` flags and an
OpenAI-compatible server).

Use it for an MoE model whose weights exceed VRAM on a host with lots of DRAM and a strong
server CPU. For a dense model, or anything that fits in VRAM, use
[`../vllm/llm/`](../vllm/llm/) or [`../sglang/llm/`](../sglang/llm/). KTransformers is
LLM-only — like [`../tensorrtllm/`](../tensorrtllm/), only an [`llm/`](llm/) leaf exists.

**Hardware:** NVIDIA H100 (CUDA 13, compute capability 8.0+) — full serving path. On AMD
MI355X (gfx950, ROCm 7.2) the kernel library builds and runs via the direct Python API, but
serving is blocked: `sglang-kt` 0.7.0 hard-pins CUDA-only dependencies.

## Setup

One venv at this stack root; `dev.env` is symlinked here and in the leaf
(`ln -sf ../../dev.env dev.env`) for `HF_TOKEN`.

NVIDIA — prebuilt wheels, no build:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # proxies 403 pypi.nvidia.com / HF
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch numpy python-dotenv
pip install kt-kernel==0.7.0              # AMX/AVX512 CPU variants + CUDA SM80/86/89/90
pip install sglang-kt==0.7.0              # kvcache-ai fork, NOT the official `sglang`
pip install nvidia-cudnn-cu12==9.16.0.29  # or set SGLANG_DISABLE_CUDNN_CHECK=1
```

Expect `kt-kernel`/`sglang-kt` to pull CUDA `torch 2.9.1+cu128`; that is fine on a cu130
host, and forcing torch back to cu130 breaks their pins.

AMD — the PyPI `kt-kernel` wheel is CUDA-only, so build the kernel library from source:

```bash
export DATA_DIR=/path/to/data          # source checkouts and scratch space
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # torch FIRST
pip install -r requirements.txt
git clone https://github.com/kvcache-ai/ktransformers $DATA_DIR/ktransformers_src
cd $DATA_DIR/ktransformers_src/kt-kernel
export CPUINFER_USE_ROCM=1 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950
export CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_ENABLE_AMX=OFF
pip install . -v --no-build-isolation --no-deps   # --no-deps: the pyproject pins CUDA torch==2.9.1
```

Re-check `torch.version.hip` after any pip operation. Verify the build is HIP-linked:

```bash
readelf -d $(python -c 'import kt_kernel;print(kt_kernel.kt_kernel_ext.__file__)') | grep NEEDED
# libamdhip64.so.7
```

See [`llm/README.md`](llm/README.md) for the serving and generation commands.
