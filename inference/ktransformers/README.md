# `inference/ktransformers` — CPU-GPU heterogeneous MoE serving (H100 ✅ serving · MI355X ⚠️ kernels only)

**KTransformers** (github.com/kvcache-ai/ktransformers) is a research framework for
**CPU-GPU heterogeneous** LLM inference. Its reason to exist is running **giant
Mixture-of-Experts** models — DeepSeek-V3/R1 671B, Kimi-K2, Qwen3-MoE — on modest GPUs by
**offloading the MoE experts to CPU DRAM** (Intel AMX / AVX512 / AMD BLIS kernels,
NUMA-aware) while attention and the active weights stay on the GPU. In v0.6+/0.7 the
inference path is two pip packages: **`kt-kernel`** (the CPU-optimized MoE kernels) and
**`sglang-kt`** (the kvcache-ai SGLang fork exposing `--kt-*` flags and an
OpenAI-compatible server).

**When to use it:** a very large **MoE** model whose weights do not fit in GPU VRAM, plus
lots of CPU RAM and a modern server CPU. For a **dense** model, or anything that fits in
VRAM, use [`../vllm/llm/`](../vllm/llm/) or [`../sglang/llm/`](../sglang/llm/) instead.

## Per-vendor outcome

| | NVIDIA H100 | AMD MI355X |
|---|---|---|
| Route | prebuilt PyPI wheels (`kt-kernel` + `sglang-kt`), no build | **source build** of kt-kernel with `CPUINFER_USE_ROCM=1` (the PyPI wheel is CUDA-only) |
| CPU kernel tier | **AMX** | **AVX512-BF16** (no AMX on Zen 5) |
| Serving | ✅ `sglang-kt` serves an MoE checkpoint with experts in CPU DRAM | ❌ **blocked** — `sglang-kt` 0.7.0 hard-pins `cuda-python`, `flashinfer`, `sgl-kernel` (CUDA-only wheels), so a container does not route around them |
| Direct Python API | not exercised | ✅ hybrid works — MoE layers on CPU, output identical to the all-GPU baseline |

**The kernel library ports across vendors; the serving layer is CUDA-first.** On AMD, use
[`../vllm/llm/`](../vllm/llm/) or [`../sglang/llm/`](../sglang/llm/) for serving.

## Scope — why only an `llm/` leaf

KTransformers is **LLM-only**: it optimizes the MoE/attention path of causal language
models and has no embedding or reranker serving path — like
[`../tensorrtllm/`](../tensorrtllm/), only an `llm/` subfolder exists.

## Install

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data          # source checkouts and scratch space
```

### NVIDIA (the serving route)

```bash
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch numpy python-dotenv
pip install kt-kernel                     # 0.7.0 — prebuilt wheel (AMX/AVX512 variants + CUDA SM80/86/89/90)
pip install sglang-kt                     # 0.7.0 — kvcache-ai SGLang fork (NOT official `sglang`)
pip install nvidia-cudnn-cu12==9.16.0.29  # sglang-kt guards against a torch-2.9.1/cuDNN<9.15 bug
```

Notes for the NVIDIA route: unset any host proxy first (a proxy typically 403s
pypi.nvidia.com / HF), and expect `kt-kernel`/`sglang-kt` to pull CUDA `torch 2.9.1+cu128`
(fine on a cu130 host). Serve on a **distinct port** (e.g. 8380 or 38612), never
SGLang's default.

### AMD (kernel library only; source build mandatory)

```bash
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # torch FIRST
pip install numpy python-dotenv huggingface_hub transformers accelerate
git clone https://github.com/kvcache-ai/ktransformers $DATA_DIR/ktransformers_src
cd $DATA_DIR/ktransformers_src/kt-kernel
export CPUINFER_USE_ROCM=1 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950
export CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_ENABLE_AMX=OFF
pip install . -v --no-build-isolation --no-deps    # --no-deps is MANDATORY (see trap below)
```

⚠️ **The `--no-deps` trap:** kt-kernel's `pyproject.toml` hard-pins `torch==2.9.1` +
`triton`; a plain `pip install .` replaces the ROCm torch with a CUDA wheel. ⚠️ **The PyPI
wheel is CUDA-only** (statically linked cudart, no HIP path) — on AMD it silently degrades
to CPU-only, which is why the source build is mandatory. Build detail, evidence, and the
`sglang-kt` blocking analysis: [`llm/README.md`](llm/README.md).

## Environment & secrets

- Client/probe deps: [`requirements.txt`](requirements.txt) (both vendor routes inside).
- Venv convention: `python3 -m venv .env_ktransformers` at this software root.
- `dev.env` symlinked at this root (`ln -sf ../../dev.env dev.env`) and in the leaf
  (`ln -sf ../../../dev.env dev.env`) for `HF_TOKEN`. Never commit tokens.

## Hardware support

- **NVIDIA H100 (CUDA 13)**: full serving path works. kt-kernel requires compute
  capability 8.0+ (Ampere/Ada/Hopper); Volta/Turing and older are not supported.
- **AMD MI355X (gfx950, ROCm 7.2)**: kt-kernel builds from source and runs the hybrid via
  the direct Python API. Serving is blocked upstream by `sglang-kt`'s CUDA pins.

## Leaves

| Leaf | Status |
|---|---|
| [`llm/`](llm/README.md) | H100: **serving works** on MoE checkpoints (for a dense model the offload is inert — use vLLM/SGLang). MI355X: **kernel library works** via the direct API; serving blocked upstream. |
