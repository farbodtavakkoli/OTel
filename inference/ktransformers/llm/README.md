# `inference/ktransformers/llm` — CPU-GPU heterogeneous MoE inference

One leaf, two vendor routes with different capabilities — both are documented below.

## Overview & when to use

KTransformers exists to serve **large Mixture-of-Experts models on VRAM-limited GPUs** by
holding the MoE experts in CPU DRAM (AMX/AVX512/BLIS kernels, NUMA-aware) while the GPU
runs attention and the active weights. Reach for it when total weights exceed VRAM and the
host has a strong server CPU; for dense models, or models that fit in VRAM, use
[`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/).

## Summary (read this first)

| | NVIDIA H100 (Intel Xeon, AMX) | AMD MI355X (EPYC, AVX512) |
|---|---|---|
| kt-kernel library | PyPI wheel, `__cpu_variant__ == 'amx'` | **source build** (`CPUINFER_USE_ROCM=1`), `avx512_bf16` variant, HIP-linked |
| Full serving (`sglang-kt`) | **works** — Qwen3-30B-A3B (128 experts), experts in CPU DRAM | blocked — `sglang-kt` hard-pins CUDA-only deps (`cuda-python`, `flashinfer`, `sgl-kernel`); not fixable by a container |
| Direct Python API hybrid | not exercised (serving path preferred) | **works** — MoE layers on CPU, output **character-identical** to the all-GPU baseline |
| Dense `Qwen3.8-27B-FP8` | no experts to offload, and output is incoherent on this stack. Use vLLM/SGLang | premise absent |

The script serves four modes: `--mode probe` (import/environment check, both vendors),
`--mode chat` (client for a running `sglang-kt` server — the NVIDIA serving path),
`--mode kernel` (numerical CPU-kernel validation, both vendors), and `--mode generate`
(direct-API hybrid generation — the only GPU path on AMD).

---

## NVIDIA / CUDA — serving works

Built and run on **H100 80GB (CUDA 13)** with an **Intel Xeon (Sapphire Rapids, AMX)** host
CPU. Versions:

| Item | Value |
|---|---|
| `kt-kernel` | **0.7.0** (`__cpu_variant__ == 'amx'`) |
| `sglang-kt` | **0.7.0** (imports as `sglang`; `sglang.__version__` = `0.0.0.dev0`) |
| `transformers` | **5.15.1** (recognizes `qwen3_5`) + `transformers-kt` 5.6.0.post2 |
| `torch` | **2.9.1+cu128** (see torch-clobber note) |
| `flashinfer-python` / `sgl-kernel` | 0.6.3 / 0.3.21 |
| `nvidia-cudnn-cu12` | 9.16.0.29 (bumped, see cuDNN note) |
| Python | 3.12 |

kt-kernel GPU support is compute capability **8.0+** (Ampere/Ada/Hopper); Volta/Turing and
older are not supported.

### Install

On NVIDIA, kt-kernel and sglang-kt ship **prebuilt wheels** — no source build needed. The
wheel bundles 6 CPU variants (AMX / AVX512-BF16/VBMI/VNNI/Base / AVX2), auto-selected at
import, plus a CUDA SM80/86/89/90 build.

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache          # Hugging Face model cache

# 1. UNSET any host proxy (proxies typically 403 pypi.nvidia.com / HF; pypi.org is allowlisted).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# 2. venv + base
python3 -m venv .env_ktransformers && source .env_ktransformers/bin/activate
pip install torch numpy python-dotenv

# 3. KTransformers inference stack (prebuilt wheels)
pip install kt-kernel                     # 0.7.0
pip install sglang-kt                     # 0.7.0  (kvcache-ai fork; NOT official `sglang`)

# 4. cuDNN bump (sglang-kt guards against a torch-2.9.1 + cuDNN<9.15 nn.Conv3d bug)
pip install nvidia-cudnn-cu12==9.16.0.29  # or set SGLANG_DISABLE_CUDNN_CHECK=1
```

**torch-clobber note (expected, leave it):** both wheels pin the stock CUDA `torch 2.9.1+cu128`,
which downgrades a default `torch 2.13.0+cu130`. `cu128` runs fine on a cu130 host
(CUDA is backward-compatible). Do **not** force torch back to cu130 — it breaks
the kt-kernel / sglang-kt pins.

**cuDNN note:** without the bump, `sglang.launch_server` aborts at startup with
`RuntimeError: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected` (a known
`nn.Conv3d` bug on cuDNN < 9.15). Installing `nvidia-cudnn-cu12==9.16.0.29` clears it (the
`torch 2.9.1 requires nvidia-cudnn-cu12==9.10.2.21` pip warning is benign — torch still imports
and runs).

### Probe — kt-kernel imports with the AMX variant

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=0
source .env_ktransformers/bin/activate
python llm/infer_llm_ktransformers.py --mode probe --out $OUTPUT_DIR/ktransformers/llm/probe.json
```

**Expected output**

```
torch 2.9.1+cu128 cuda 12.8 avail True
device0 NVIDIA H100 80GB HBM3 cc (9, 0)
kt_kernel version 0.7.0
kt_kernel __cpu_variant__ amx
CUDA stream support: True
KTMoEWrapper import OK
CPUInfer[0x...]: Hello
WorkerPool[0x...] 2 subpools, [numa:threads][0:2] [1:2]
===========In NumaPool============
In Numa Worker Pool at NUMA 0, 2 threads
===========In NumaPool============
In Numa Worker Pool at NUMA 1, 2 threads
```

The `amx` variant + the 2-NUMA worker pools are the CPU-side proof that KT loaded its best
kernel path for the host.

### Serve an MoE model — `Qwen/Qwen3-30B-A3B`

This is the workload KTransformers exists for: a genuine **Mixture-of-Experts** model
(`Qwen3MoeForCausalLM`, **128 experts, 8 active/token, 48 layers**) whose experts are quantized
to AMX INT8 and **offloaded to CPU DRAM**, while attention/active weights and the hot experts
stay on the GPU.

#### Step 1 — download the MoE weights (bf16) with the proxy unset

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download Qwen/Qwen3-30B-A3B --local-dir <models>/Qwen3-30B-A3B     # Qwen3MoeForCausalLM, 128 experts
```

#### Step 2 — convert experts to AMX INT8 CPU weights

Use INT8, not INT4: INT4 can cause a large accuracy drop on Qwen3-30B-A3B.

```bash
cd kt-kernel
python scripts/convert_cpu_weights.py \
  --input-path <models>/Qwen3-30B-A3B --input-type bf16 \
  --output <models>/Qwen3-30B-A3B-INT8 --quant-method int8 \
  --cpuinfer-threads 48 --threadpool-count 2
```

**Expected output** (the NUMA-aware AMX MoE quant path):

```
TP MOE layer 0, pool: 0x..., expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
  online quant from bf16
Conversion completed successfully!
```

#### Step 3 — serve with the AMX INT8 heterogeneous backend

`--model` points at the **bf16 GPU** weights; `--kt-weight-path` at the **INT8 CPU** weights;
`--kt-num-gpu-experts 32` keeps 32/128 hot experts on GPU, the rest stream from DRAM.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
  --host 127.0.0.1 --port 8380 \      # a distinct port, never SGLang's default
  --model <models>/Qwen3-30B-A3B \
  --kt-method AMXINT8 --kt-weight-path <models>/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 48 --kt-threadpool-count 2 --kt-num-gpu-experts 32 \
  --served-model-name qwen3-30b-a3b --trust-remote-code \
  --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-mixed-chunk \
  --tensor-parallel-size 1
```

**Expected output**

```
TP MOE layer 0, pool: 0x..., expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0        # <- experts placed on CPU (NUMA-aware), not GPU
Creating AMX_MOE_TP 1 at numa 1
Load weight end. ... type=Qwen3MoeForCausalLM, dtype=torch.bfloat16
KV Cache is allocated.
The server is fired up and ready to roll!
```

#### Confirm the offload actually happened

Sample GPU and DRAM residency by PID from a second shell while generating — only a fraction
of the parameter mass should be on the GPU, with the INT8 experts resident in DRAM:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <N>
ps -o rss= $(pstree -p <server-pid>) | awk '{s+=$1} END{printf "%.1f GB\n",s/1048576}'
```

Then check the output with
`infer_llm_ktransformers.py --mode chat --port 8380 --model qwen3-30b-a3b`.
`Qwen3-30B-A3B` is a *thinking* model, so it emits a `<think>…</think>` block unless you
append `/no_think` to the prompt.

### Caveat — a dense checkpoint has nothing to offload

**`Qwen/Qwen3.8-27B-FP8` is a dense model** (`num_experts: null`), so KT's expert offload is
inert and the run is just SGLang-on-GPU. Its output on this stack is also **incoherent**: the
load log shows 128 unmatched FP8 per-block scales (`weight_scale_inv not found in
params_dict`) plus a `qwen3_5` RoPE-compat warning — a model-support/version gap in
`sglang-kt 0.7.0` + `transformers 5.15.1`, not a KT runtime fault.

For this model use [`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/).

If you serve a dense hybrid-GDN model here anyway, the server enters a **DeepGEMM JIT
pre-compile** after weight load, once per CUDA-graph batch size. Pre-run
`python -m sglang.compile_deep_gemm` to avoid it.

### Multi-GPU

Add `--tensor-parallel-size N` (and for MoE, tune `--kt-num-gpu-experts` per GPU), then
re-verify residency across all N GPUs with `nvidia-smi`.

### Notes & quirks (NVIDIA)

- **Prebuilt wheels, no source build.** A slow `kt-kernel` CPU-kernel compile is not required:
  the current PyPI wheel makes it unnecessary on standard x86-64 + NVIDIA. Source build
  (`cd kt-kernel && ./install.sh`) is only for AMD BLIS / ARM KML / custom CUDA.
- **`sglang-kt`, not `sglang`.** Install the kvcache-ai fork. If the official `sglang` is
  present, `pip uninstall sglang -y` first.
- **cuDNN guard** (torch 2.9.1 + cuDNN < 9.15) blocks server startup until you bump cuDNN or set
  `SGLANG_DISABLE_CUDNN_CHECK=1`.
- **DeepGEMM JIT** first-run pre-compile is a one-time cost per CUDA-graph batch size;
  pre-run `python -m sglang.compile_deep_gemm` to amortize it.
- **KT is for MoE checkpoints.** "Qwen3.8"/"Qwen3.5" in KT's docs means the **MoE** variants
  (Qwen3.5-MoE-400B, Qwen3-30B-A3B), not the dense 27B-FP8 checkpoint.


---

## AMD / ROCm — kernel library works, serving blocked

Built and run on **MI355X (gfx950, ROCm 7.2)** with an **EPYC (Zen 5, AVX512-BF16, no AMX)**
host CPU, against the same MoE model **`Qwen/Qwen3-30B-A3B`** (48 layers, 128 experts,
top-8 routing, BF16 safetensors). Its per-expert on-disk layout
(`model.layers.N.mlp.experts.M.gate_proj.weight`) is auto-detected by kt-kernel's
`BF16SafeTensorLoader` as `deepseek` format. `deepseek-ai/DeepSeek-V2-Lite-Chat` is a smaller
fallback.

### Install

See [`../README.md`](../README.md) for the full source build. The short version — **the
PyPI wheel is the NVIDIA build and will not give you a ROCm kt-kernel**:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export DATA_DIR=/path/to/data           # source checkouts and scratch space
export OUTPUT_DIR=/path/to/outputs      # run logs and artifacts

source ../.env_ktransformers/bin/activate
cd $DATA_DIR/ktransformers_src/kt-kernel
export CPUINFER_USE_ROCM=1 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950
export CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_ENABLE_AMX=OFF
export CPUINFER_ENABLE_AVX512_VNNI=ON CPUINFER_ENABLE_AVX512_BF16=ON CPUINFER_ENABLE_AVX512_VBMI=ON
pip install . -v --no-build-isolation --no-deps      # --no-deps is MANDATORY
```

### Environment & secrets

`dev.env` is symlinked to the repo-root file (`ln -sf ../../../dev.env dev.env`) and
supplies `HF_TOKEN`, loaded by `load_dotenv("dev.env")`. Never echo it.

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides every GPU. Send logs to
`$OUTPUT_DIR/inference_ktransformers/` so nothing large lands in the repo.

### Commands

Environment probe — no model download, reports what the build actually links against:

```bash
python infer_llm_ktransformers.py --mode probe
```

Synthetic MoE kernel check against a PyTorch fp32 reference — no model download:

```bash
python infer_llm_ktransformers.py --mode kernel
```

The real hybrid run:

```bash
python infer_llm_ktransformers.py --mode generate --max_new_tokens 160 --sample_hw
```

Tuned for a 128-physical-core, 2-NUMA-node CPU — these are already the defaults:

```bash
python infer_llm_ktransformers.py --mode generate \
  --kt_method BF16 --cpuinfer_threads 128 --threadpool_count 2 \
  --max_new_tokens 160 --seed 42 --sample_hw
```

### Expected output — probe

The point of the probe on AMD is to confirm the build is genuinely ROCm, not a silently
CPU-only PyPI wheel:

```
torch              : 2.11.0+rocm7.2
torch.version.hip  : 7.2.26015
torch.version.cuda : None
device 0           : AMD Instinct MI355X / gfx950:sramecc+:xnack-
kt_kernel version  : 0.7.0
kt_kernel variant  : avx512_bf16
GPU runtime linked : HIP (libamdhip64)          <- the line that matters
stream interop     : True
cpu isa            : avx512f=yes avx512bw=yes avx512vbmi=yes avx512_vnni=yes avx512_bf16=yes amx_tile=NO amx_int8=NO amx_bf16=NO
amx fast path      : UNAVAILABLE (EPYC — AVX512 path used)
```

`avx512_bf16` is the correct selection for a Zen 5 EPYC: kt-kernel picked the best AVX512
variant and did **not** try AMX.

### Expected output — kernel check

```
relative L1 error  : 0.3913% vs PyTorch fp32 reference
verdict            : PASS
```

Sub-0.5% is BF16 rounding, not a broken kernel (upstream's own threshold is 5%). Look for
`backend=AVX512-BF16` and **two NUMA subpools** — kt-kernel splits the expert bank across
both sockets by itself.

### Expected output — hybrid generate

This path works with no code changes to kt-kernel. All 48 MoE layers offloaded:

```
kt-kernel layers   : 48 MoE layers offloaded to CPU
model              : Qwen/Qwen3-30B-A3B
kt_method          : BF16
gpu vram (card0)    : a few GB resident after load
```

Generated text is **character-identical** to the all-GPU `transformers` baseline with the
same prompt and greedy decode: routing the expert FFNs through kt-kernel's CPU AVX512-BF16
kernels reproduces the GPU BF16 output token-for-token. Pre-declare batch sizes with
`KTMoEWrapper.set_capture_batch_sizes([...])` to remove the first-call warm-up.

### Arguments — `generate` / `kernel` modes

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `probe` | `probe` (environment only) / `kernel` (synthetic MoE vs torch) / `generate` (real model) |
| `--model` | `Qwen/Qwen3-30B-A3B` | HF repo id or local path of the MoE model |
| `--kt_method` | `BF16` | CPU backend. Available here: `BF16`, `FP8`, `FP8_PERCHANNEL`, `RAWINT4`, `LLAMAFILE`. `AMXINT4`/`AMXINT8` are **impossible on EPYC** |
| `--cpuinfer_threads` | `128` | CPU inference threads — set to *physical* cores, not hyperthreads |
| `--threadpool_count` | `2` | Thread pools — set to the NUMA node count (2 here) |
| `--num_gpu_experts` | `0` | Experts kept on GPU. **Leave at 0 in this harness** — see *Notes & quirks* |
| `--max_layers` | `0` | Offload only the first N MoE layers (0 = all) — useful for bisecting |
| `--chunked_prefill_size` | `512` | Maximum prefill chunk |
| `--prompt` | MoE question | User prompt |
| `--max_new_tokens` | `64` | Tokens to generate. Use ≥160 for a meaningful tok/s |
| `--temperature` | `0.0` | 0 = greedy (required for the token-identity comparison) |
| `--seed` | `42` | Sampling seed |
| `--device` | `cuda:0` | Torch device for the non-expert half (ROCm exposes itself as `cuda`) |
| `--sample_hw` | off | Sample `rocm-smi` VRAM and CPU load during decode |

### Arguments — `chat` mode (against a running `sglang-kt` server)

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `38612` | Server address; never SGLang's default port |
| `--model` | `qwen38-27b-fp8` | Served model name (matches `--served-model-name`) |
| `--prompt` / `--system_prompt` | see script | Chat inputs |
| `--max_tokens` / `--temperature` / `--top_p` / `--seed` | `128` / `0.0` / `1.0` / none | Sampling |
| `--timeout` | `600.0` | HTTP timeout |
| `--out` | none | Write a JSON artifact of the probe/chat result |

Probe exits 0 only when `kt_kernel` imports **and** a GPU is visible.

### Notes & quirks (AMD)

- **`--no-deps` is mandatory when building.** `kt-kernel/pyproject.toml` hard-pins
  `torch==2.9.1` and requires `triton>=2.0.0`. A plain `pip install .` uninstalls
  `torch 2.11.0+rocm7.2` and installs the **CUDA** torch and triton — ROCm torch ships
  `triton-rocm`, which does not satisfy the `triton` name. Re-check `torch.version.hip`
  after any pip operation.
- **`pip install kt-kernel` does not give you a ROCm build.** The PyPI wheel statically
  links the CUDA runtime and dlopens `libcuda.so.1`; there is no `libamdhip64` in it. On
  an AMD host it silently degrades to CPU-only. Verify with
  `readelf -d $(python -c 'import kt_kernel;print(kt_kernel.kt_kernel_ext.__file__)') | grep NEEDED`
  — you want to see `libamdhip64.so.7`.
- **The AMX backends are unavailable on any AMD CPU** — Zen 5 has no `amx_tile`.
  `--kt-method AMXINT4/AMXINT8` — the configuration in most of upstream's tutorials and
  benchmarks — cannot run there. Build with `CPUINFER_ENABLE_AMX=OFF` and use the
  AVX512 native-precision methods.
- **`--num_gpu_experts > 0` is not usable from this standalone harness.** kt-kernel's
  `gpu_experts_mask` tells the CPU side to *skip* the masked experts on the assumption that
  the serving engine computes them with its own fused GPU MoE kernel. `transformers` has no
  such kernel, so masking experts here would silently drop them from the output. Partial
  expert placement requires the SGLang integration.
- **SGLang integration is blocked on ROCm.** `sglang-kt` 0.7.0 (the kvcache-ai fork you must
  use, *not* upstream `sglang`) declares unconditional dependencies on `cuda-python`,
  `flashinfer_python`, `flashinfer_cubin`, `nvidia-cutlass-dsl`, `nvidia-ml-py`,
  `sgl-kernel==0.3.21` and `torch==2.9.1`, and `sgl-kernel` publishes CUDA-only wheels. The
  CUDA dependency is in the package metadata, so a container does not route around it.
- **`kt version` runs but its diagnostics are CUDA-centric.** It reports
  `kt-kernel 0.7.0` correctly and then `CUDA  Not found` / `sglang-kt  Not installed`,
  followed by install instructions that do not apply on ROCm. It has no notion of HIP, so
  it cannot tell you whether your build is the ROCm one. Use `--mode probe` instead.
- **The extension prints unconditionally to stdout** (`CPUInfer[...]: Hello`,
  `Created BF16_MOE_TP 0 at numa 0`, one pair per layer). There is no quiet flag; filter it
  if you are parsing output.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs.

### Platform notes (AMD)

`kt-kernel` 0.7.0 builds from source with `CPUINFER_USE_ROCM=1` in a couple of minutes at
`-j64`, links `libamdhip64.so.7`, auto-selects the `avx512_bf16` CPU variant, and builds
NUMA-aware expert pools across both sockets. Its ROCm backend compiles **no device kernels** —
the entire GPU dependency is `hipLaunchHostFunc` for enqueuing CPU work on a HIP stream, so
there is nothing gfx-specific to break. The stale `doc/en/ROCm.md` (conda, ROCm 6.2.4) is not
needed: a plain Python 3.12 venv on Ubuntu 24.04 works.

The direct Python API is a single-stream research harness, not a server — serving needs
SGLang, and `sglang-kt` is CUDA-only by package metadata.

---

## Hardware support

| | Status |
|---|---|
| NVIDIA Hopper H100 | **Works** — kt-kernel wheel supports SM 90, full serving path. |
| NVIDIA Ampere/Ada (cc 8.0–8.9) | Upstream-supported; cc < 8.0 is not. |
| Intel AMX CPU (Sapphire Rapids) | **Works** — `__cpu_variant__ == 'amx'`, KT's best CPU tier. |
| **AMD MI355X (gfx950, ROCm 7.2)** | **Kernel library only** — source build with `CPUINFER_USE_ROCM=1`, hybrid generate via the direct Python API. Serving is **blocked upstream** by `sglang-kt`'s CUDA-only pins. |
| AMD Zen4/Zen5 CPU (AVX512/BLIS) | **Works** — `avx512_bf16` variant; AMX methods are impossible. |
| Dense `Qwen/Qwen3.8-27B-FP8` | Loads but the MoE offload is inert and output is incoherent. Use vLLM/SGLang. |
| Embedding / reranker | upstream — KT is LLM-only. Use vLLM/TEI/SGLang/llama.cpp. |

## Recommendation

- **On NVIDIA:** use KTransformers for giant **MoE** models that do not fit in VRAM but fit
  in VRAM+DRAM (DeepSeek-V3/R1, Kimi-K2, Qwen3.5-MoE-400B, Qwen3-30B-A3B).
- **On AMD:** the kernel library works (check with `--mode probe`), but there is no serving
  layer — `sglang-kt`'s CUDA-only dependency chain has no ROCm equivalent.
- **For anything that already fits in VRAM**, on either vendor, use
  [`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/) instead.
