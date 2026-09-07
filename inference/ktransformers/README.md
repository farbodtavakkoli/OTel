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

## Verified on both vendors — with opposite outcomes

This stack is the repo's clearest case of an outcome that is *hardware-dependent by
design*. Full evidence for both platforms lives in [`llm/README.md`](llm/README.md).

| | NVIDIA H100 | AMD MI355X |
|---|---|---|
| Route | prebuilt PyPI wheels (`kt-kernel` + `sglang-kt`), no build | **source build** of kt-kernel with `CPUINFER_USE_ROCM=1` (PyPI wheel is CUDA-only — verified) |
| CPU kernel tier | **AMX** (Xeon 8480C Sapphire Rapids) | **AVX512-BF16** (EPYC 9575F — no AMX on Zen 5, by design) |
| Serving | ✅ `sglang-kt` serves `Qwen/Qwen3-30B-A3B` (128 experts): coherent output, ~23–50 tok/s, experts in CPU DRAM (~72 GB RSS) with only **16.41 GB** on GPU | ❌ **blocked** — `sglang-kt` 0.7.0 hard-pins `cuda-python`, `flashinfer`, `sgl-kernel` (CUDA-only wheels); metadata requirements, so a container does not route around them |
| Direct Python API | not exercised | ✅ hybrid proven: 48 MoE layers on CPU → **4.09 GB** VRAM vs 64.62 GB all-GPU (15.8×), 17.44 tok/s, output **character-identical** to the all-GPU baseline |
| Outcome | **works on its designed workload** — use for MoE models too big for 80 GB | kernel library is genuinely ROCm-portable, but no serving layer — and 288 GB/card removes the VRAM-scarcity premise for this repo's targets |

Bottom line: **the kernel library ports across vendors; the serving layer is CUDA-first.**
On NVIDIA this is a working niche tool; on AMD it is a well-evidenced
"runs, but use vLLM/SGLang instead".

## Scope — why only an `llm/` leaf

KTransformers is **LLM-only**: it optimizes the MoE/attention path of causal language
models and has no embedding or reranker serving path — like
[`../tensorrtllm/`](../tensorrtllm/), only an `llm/` subfolder exists.

## Install

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data          # source checkouts and scratch space
```

### NVIDIA (verified on H100 — the serving route)

```bash
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch numpy python-dotenv
pip install kt-kernel                     # 0.7.0 — prebuilt wheel (AMX/AVX512 variants + CUDA SM80/86/89/90)
pip install sglang-kt                     # 0.7.0 — kvcache-ai SGLang fork (NOT official `sglang`)
pip install nvidia-cudnn-cu12==9.16.0.29  # sglang-kt guards against a torch-2.9.1/cuDNN<9.15 bug
```

Notes for the H100 route: unset any host proxy first (a proxy typically 403s
pypi.nvidia.com / HF), and expect `kt-kernel`/`sglang-kt` to pull CUDA `torch 2.9.1+cu128`
(fine on a cu130/driver-580 host). Serve on a **distinct port** (e.g. 8380 or 38612), never
SGLang's default.

### AMD (verified on MI355X — kernel library only; source build mandatory)

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

- **NVIDIA H100 (CUDA 13.0): verified** — full serving path, AMX expert offload
  demonstrated and used. kt-kernel GPU support: compute capability 8.0+ (Ampere/Ada/
  Hopper); Volta/Turing and older are not supported.
- **AMD MI355X (gfx950, ROCm 7.2.4): verified** — kt-kernel builds from source, is
  genuinely HIP-linked, validates numerically, and runs the hybrid via the direct API;
  the ROCm role is host-callback plumbing (no device kernels compiled), so there is
  nothing gfx950-specific to break. Serving is blocked upstream (`sglang-kt` CUDA pins).
- **Other hardware (upstream claims — not verified here):** Intel AMX CPUs (the flagship
  tier), Intel Arc XPU, AMD Zen4+ CPUs via BLIS, universal CPUs via the llamafile/GGUF
  backend, and an Ascend NPU tutorial for the older ktransformers architecture.

## Leaves

| Leaf | Status |
|---|---|
| [`llm/`](llm/README.md) | H100: **serving works** on the designed MoE workload (dense 27B FP8 is a documented caveat — offload inert, use vLLM/SGLang). MI355X: **kernel library works** (hybrid proven via direct API); serving blocked upstream; premise moot on 288 GB cards. |
