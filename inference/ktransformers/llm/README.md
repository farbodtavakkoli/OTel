# `inference/ktransformers/llm` — CPU-GPU heterogeneous MoE inference (H100 + MI355X)

One leaf, two vendor stories — both campaigns' evidence is preserved below in full.

## Overview & when to use

KTransformers exists to serve **large Mixture-of-Experts models on VRAM-limited GPUs** by
holding the MoE experts in CPU DRAM (AMX/AVX512/BLIS kernels, NUMA-aware) while the GPU
runs attention and the active weights. Reach for it when total weights exceed VRAM and the
host has a strong server CPU; for dense models, or models that fit in VRAM, use
[`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/).

## VERDICT (read this first)

| | NVIDIA H100 (Intel Xeon 8480C, AMX) | AMD MI355X (EPYC 9575F, AVX512) |
|---|---|---|
| kt-kernel library | ✅ PyPI wheel, `__cpu_variant__ == 'amx'` | ✅ **source build** (`CPUINFER_USE_ROCM=1`), `avx512_bf16` variant, HIP-linked |
| Full serving (`sglang-kt`) | ✅ **works** — Qwen3-30B-A3B (128 experts) coherent at ~23–50 tok/s, experts in CPU DRAM (~72 GB RSS, 16.41 GB VRAM) | ❌ blocked — `sglang-kt` hard-pins CUDA-only deps (`cuda-python`, `flashinfer`, `sgl-kernel`); not fixable by a container |
| Direct Python API hybrid | not exercised (serving path preferred) | ✅ **works** — 48 MoE layers on CPU: 4.09 GB VRAM vs 64.62 GB all-GPU (15.8×), 17.44 tok/s, output **character-identical** to the all-GPU baseline |
| Sensible to use here? | ✅ for MoE models too big for 80 GB | ⚠️ no — 288 GB/card removes the premise for this repo's targets |
| Dense `Qwen3.8-27B-FP8` (repo target) | ❌ documented caveat — no experts to offload; output incoherent on this stack (FP8-scale + `qwen3_5` RoPE gap). Use vLLM/SGLang | not attempted (premise absent) |

The script serves four modes: `--mode probe` (import/environment check, both vendors),
`--mode chat` (client for a running `sglang-kt` server — the NVIDIA serving path),
`--mode kernel` (numerical CPU-kernel validation, both vendors), and `--mode generate`
(direct-API hybrid generation — the only GPU path on AMD).

---

# NVIDIA H100 (CUDA 13.0) — serving works · tested 2026-08-23

### Host & versions (verified)

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (Hopper cc 9.0, native FP8) — single GPU, `CUDA_VISIBLE_DEVICES=6` |
| CPU | Intel Xeon Platinum 8480C (Sapphire Rapids) — `amx_bf16`/`amx_int8`/`amx_tile` + `avx512_bf16`, 96 threads, 2 NUMA nodes |
| DRAM | 1870 GB |
| Driver / CUDA | **580.173.02** / CUDA 13.0 |
| `kt-kernel` | **0.7.0** (`__cpu_variant__ == 'amx'`) |
| `sglang-kt` | **0.7.0** (imports as `sglang`; `sglang.__version__` = `0.0.0.dev0`) |
| `transformers` | **5.15.1** (recognizes `qwen3_5`) + `transformers-kt` 5.6.0.post2 |
| `torch` | **2.9.1+cu128** (see torch-clobber note) |
| `flashinfer-python` / `sgl-kernel` | 0.6.3 / 0.3.21 |
| `nvidia-cudnn-cu12` | 9.16.0.29 (bumped, see cuDNN note) |
| Python | 3.12.3 |

### Install recipe that worked

The current kt-kernel and sglang-kt ship **prebuilt wheels** — the slow `kt-kernel` source
build (the `-march=native` CPU-kernel compile) was **not needed**. kt-kernel's wheel bundles 6
CPU variants (AMX / AVX512-BF16/VBMI/VNNI/Base / AVX2) auto-selected at import, plus a fat CUDA
SM80/86/89/90 wheel with a static CUDA runtime.

```bash
# 1. UNSET the box proxy (403s pypi.nvidia.com / HF; pypi.org itself is allowlisted).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models

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
which downgrades this box's default `torch 2.13.0+cu130`. `cu128` runs fine on the cu130 /
driver-580 host (CUDA is backward-compatible). Do **not** force torch back to cu130 — it breaks
the kt-kernel / sglang-kt pins. Confirmed: `torch.cuda.is_available() == True`, device 0 =
`NVIDIA H100 80GB HBM3`, cc `(9, 0)`.

**cuDNN note:** without the bump, `sglang.launch_server` aborts at startup with
`RuntimeError: CRITICAL WARNING: PyTorch 2.9.1 & CuDNN Compatibility Issue Detected` (a known
`nn.Conv3d` bug on cuDNN < 9.15). Installing `nvidia-cudnn-cu12==9.16.0.29` clears it (the
`torch 2.9.1 requires nvidia-cudnn-cu12==9.10.2.21` pip warning is benign — torch still imports
and runs).

### Probe — kt-kernel imports with the AMX variant

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models CUDA_VISIBLE_DEVICES=6
source .env_ktransformers/bin/activate
python llm/infer_llm_ktransformers.py --mode probe --out /dev/shm/h100/out/ktransformers/llm/probe.json
```

Real output:

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
kernel path for this host.

### PRIMARY RESULT — `Qwen/Qwen3-30B-A3B` (MoE, KT's real job)

This is the workload KTransformers exists for: a genuine **Mixture-of-Experts** model
(`Qwen3MoeForCausalLM`, **128 experts, 8 active/token, 48 layers**) whose experts are quantized
to AMX INT8 and **offloaded to CPU DRAM**, while attention/active weights and the hot experts
stay on the H100. On this Xeon-8480C (AMX) + 1870 GB DRAM box KT ran it end-to-end with
**coherent output**.

#### Step 1 — download the MoE weights (bf16, ~57 GB) with the proxy unset

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models
hf download Qwen/Qwen3-30B-A3B --local-dir <models>/Qwen3-30B-A3B     # Qwen3MoeForCausalLM, 128 experts
```

#### Step 2 — convert experts to AMX INT8 CPU weights (~2.6 s/layer, 48 layers)

INT8, not INT4: upstream notes INT4 can cause a large accuracy drop on Qwen3-30B-A3B.

```bash
cd kt-kernel
python scripts/convert_cpu_weights.py \
  --input-path <models>/Qwen3-30B-A3B --input-type bf16 \
  --output <models>/Qwen3-30B-A3B-INT8 --quant-method int8 \
  --cpuinfer-threads 48 --threadpool-count 2
#   -> "Conversion completed successfully!"  (74163 tensors across 49 shards, ~31 GB)
```

Real convert log (the NUMA-aware AMX MoE quant path):

```
TP MOE layer 0, pool: 0x..., expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0
Creating AMX_MOE_TP 1 at numa 1
  online quant from bf16
  Layer 47 quantized and saved in 2.62s
Conversion completed successfully!
```

#### Step 3 — serve with the AMX INT8 heterogeneous backend (single GPU 6, port 8380)

`--model` points at the **bf16 GPU** weights; `--kt-weight-path` at the **INT8 CPU** weights;
`--kt-num-gpu-experts 32` keeps 32/128 hot experts on GPU, the rest stream from DRAM.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=6
python -m sglang.launch_server \
  --host 127.0.0.1 --port 8380 \
  --model <models>/Qwen3-30B-A3B \
  --kt-method AMXINT8 --kt-weight-path <models>/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 48 --kt-threadpool-count 2 --kt-num-gpu-experts 32 \
  --served-model-name qwen3-30b-a3b --trust-remote-code \
  --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-mixed-chunk \
  --tensor-parallel-size 1
```

Real log lines (`/dev/shm/h100/out/ktransformers/llm/serve_30b_a3b.log`):

```
TP MOE layer 0, pool: 0x..., expert num: 128, num_experts_per_tok: 8
Creating AMX_MOE_TP 0 at numa 0        # <- experts placed on CPU (NUMA-aware), not GPU
Creating AMX_MOE_TP 1 at numa 1
Load weight end. elapsed=8.01 s, type=Qwen3MoeForCausalLM, dtype=torch.bfloat16, avail mem=62.08 GB, mem usage=16.41 GB.
KV Cache is allocated. #tokens: 549277, K size: 25.14 GB, V size: 25.14 GB
Capture cuda graph end. Time elapsed: 26.12 s.
The server is fired up and ready to roll!      # <- ready in ~70 s (no DeepGEMM JIT for this arch)
```

#### Evidence that matters for KT — CPU-expert offload, and it was USED

**GPU-6 residency vs DRAM residency, sampled by PID from a second shell while generating:**

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 6
1898288, 70692 MiB          # 16.41 GB MoE+dense weights on GPU + ~50 GB KV cache
$ ps -o rss= $(pstree -p <server-pid>) | awk '{s+=$1} END{printf "%.1f GB\n",s/1048576}'
71.8 GB                     # <- process-tree DRAM RSS: the 128 INT8 experts live in the 1870 GB DRAM
```

The contrast with the dense 27B is the whole story:

| | GPU weights | Expert location | Coherent? |
|---|---|---|---|
| **Qwen3-30B-A3B (MoE, this section)** | **16.41 GB** | **CPU DRAM (~72 GB RSS)** via AMX INT8 | **YES** |
| Qwen3.8-27B-FP8 (dense, next section) | 39.09 GB | all in GPU VRAM (no experts) | no (version gap) |

Only 16.4 GB of a 30B model on the GPU — the rest of the parameter mass is in DRAM. That
reduction *is* CPU-expert offload; on a smaller GPU it is what lets the model fit at all.

#### Real coherent generations (the pass criterion)

`infer_llm_ktransformers.py --mode chat --port 8380 --model qwen3-30b-a3b`:

```
PROMPT: "In one short sentence, what is the capital of France? /no_think"
  finish_reason: stop | 37 tok/s
  -> "The capital of France is Paris."

PROMPT: "What is the capital of France, and what is 2 + 2? ..."   (thinking model, emits <think>)
  50 tok/s
  -> "... the capital is Paris. Then ... 2 plus 2. That's straightforward arithmetic. 2 plus 2 equals 4."
```

Correct and coherent — the decisive contrast with the dense 27B's garbage output below.
`Qwen3-30B-A3B` is a *thinking* model (Qwen3), so it emits a `<think>…</think>` block unless you
append `/no_think`.

### Serving the repo target — `Qwen/Qwen3.8-27B-FP8` (dense — documented caveat)

The checkpoint is fully cached (66-shard FP8 snapshot at `HF_HOME`). Its `config.json`:
`architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: "qwen3_5"`,
**`num_experts: null`**, `vision_config` present, `quant_method: "fp8"` — i.e. a **dense
vision-language** model, not an MoE. Serve command (single GPU 6, distinct port 38612):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=6
snap=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
python -m sglang.launch_server \
  --host 127.0.0.1 --port 38612 \
  --model "$snap" --kt-weight-path "$snap" \
  --kt-method FP8 --kt-cpuinfer 48 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --served-model-name qwen38-27b-fp8 --trust-remote-code \
  --mem-fraction-static 0.85 --tensor-parallel-size 1
```

Real log lines (`/dev/shm/h100/out/ktransformers/llm/serve_27b_fp8.log`):

```
WARNING model_config.py: Transformers version 5.15.1 is used for model type qwen3_5. ...
WARNING server_args.py: KTransformers EP is enabled. --disable-shared-experts-fusion is automatically set ...
Loading safetensors checkpoint shards: 100% Completed | 66/66 [07:29<00:00,  6.80s/it]
Load weight end. elapsed=449.70 s, type=Qwen3_5ForConditionalGeneration, dtype=torch.bfloat16, avail mem=39.42 GB, mem usage=39.09 GB.
Using hybrid linear attention backend for hybrid GDN models.
KV Cache is allocated. #tokens: 238285, K size: 7.27 GB, V size: 7.27 GB
Capture cuda graph begin. ... Entering DeepGEMM JIT Pre-Compile session. It may take a long time (typically 10-20 mins) ...
```

**GPU-6 residency, sampled by PID from a second shell while loaded (proof the H100 was used):**

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 6
1881913, 69834 MiB          # our sglang-kt worker: 39 GiB weights + KV/Mamba cache + CUDA graphs
```

**CPU/DRAM residency (the tell-tale of the caveat):** the same process held only **~5 GB DRAM
RSS** total (parent 2.1 GB + children 2.8 GB). For a real MoE run KT would park tens-to-hundreds
of GB of experts in DRAM; here there is nothing to offload, so DRAM stays near-empty and the
whole 39 GB of weights sits in HBM. This is the design-point mismatch made visible.

**Generation is real but the output is incoherent (honest result).** After ~15 min (weight load
~7.5 min + DeepGEMM JIT warmup over the 8 CUDA-graph batch sizes) the server reached
`"The server is fired up and ready to roll!"` and served real completions on GPU 6:

```
$ python llm/infer_llm_ktransformers.py --mode chat --port 38612 --model qwen38-27b-fp8 \
    --prompt "The opposite of hot is" --max_tokens 32
finish_reason : length   latency : 1.29s   decode rate : 24.9 tok/s (32 completion tokens)
--- response ---
alth和基本等基本等基本等基本等基本等基本等基本等基本等基本等基本等基本等 ...   # <- garbage
```

The HTTP path, latency, and ~25–42 tok/s decode are all genuine, but the text is **not
coherent**. Root cause visible in the load log, and it is **not** a KT-runtime fault:

```
$ grep -c "weight_scale_inv not found in params_dict" serve_27b_fp8.log
128                                    # FP8 per-block scales for gate_up_proj not matched
WARNING model_config.py: Transformers version 5.15.1 is used for model type qwen3_5.
  ... they may be due to incompatibilities between Transformers >=5.0.0 and some models.
  You can try downgrading to transformers==4.57.1 as a workaround.
Detected fp8 checkpoint.
```

So this too-new FP8 **VL** checkpoint (`transformers_version: 5.8.0.dev0` in its config) is not
correctly dequantized / RoPE-wired by `sglang-kt 0.7.0` + `transformers 5.15.1`: the model
loads and the runtime generates, but the weights are effectively mis-loaded → gibberish. This is
a **model-support/version gap** on the specific FP8 VL checkpoint, layered on top of the
dense-vs-MoE mismatch. It is a cleaner outcome than the sibling `tensorrtllm` stack (which could
not even parse `qwen3_5`), but still short of usable output on this exact checkpoint.

**Notes on the serve:**
- The KT/SGLang stack **did register `qwen3_5`** (only a RoPE-compat *warning*) — a strictly
  better outcome than the sibling [`../../tensorrtllm/`](../../tensorrtllm/) stack, whose
  older `transformers 4.57.3` + `tensorrt_llm 1.2.1` could not parse the config at all.
- The model is a **hybrid GDN (Gated Delta Net / Mamba) linear-attention** arch — the log
  allocates a "Mamba Cache" and dispatches Triton GDN kernels. It loads and runs, but this is
  vanilla SGLang GPU serving; the `--kt-*` flags had no experts to act on (`--kt-num-gpu-experts 0`).
- After weight load the server enters a **DeepGEMM JIT pre-compile** (per upstream, "typically
  10-20 mins" on first run, once per CUDA-graph batch size). This is a one-time first-run cost;
  pre-run `python -m sglang.compile_deep_gemm` to avoid it.

### Verdict (H100)

- **Stack builds/imports on H100: YES** — prebuilt wheels, `__cpu_variant__ == 'amx'`, CUDA visible.
- **WORKS on its designed MoE workload: YES.** `Qwen/Qwen3-30B-A3B` (128 experts) converted to
  AMX INT8, served via `--kt-method AMXINT8`, with the **experts offloaded to CPU DRAM** (~72 GB
  process RSS) so only **16.41 GB** sat on GPU 6, producing **coherent** output at ~23–50 tok/s.
  This is the primary result — KT doing exactly what it exists for. See
  [Primary result](#primary-result--qwenqwen3-30b-a3b-moe-kts-real-job).
- **Repo target `Qwen/Qwen3.8-27B-FP8`: documented caveat, two reasons.** (1) It is a **dense**
  model (`num_experts: null`) — nothing to offload, so KT's feature is inert and it is just
  SGLang-on-GPU (39 GB in VRAM). (2) On this stack its output is **incoherent** (128 unmatched
  FP8 `weight_scale_inv` scales + a `qwen3_5` RoPE warning = model-support/version gap). For this
  specific dense model, use [`../../vllm/llm/`](../../vllm/llm/) or [`../../sglang/llm/`](../../sglang/llm/).
- **When to reach for KT:** giant **MoE** models (DeepSeek-V3/R1, Kimi-K2, Qwen3.5-MoE-400B,
  Qwen3-30B-A3B) that do not fit in VRAM but fit in VRAM+DRAM — proven here.

### Single-GPU only; what multi-GPU would need

Per the shared-node rules this was a **single-GPU** run on GPU 6 (GPUs 0–3 = co-tenant
production job, never touched). Multi-GPU is **deferred**. A multi-GPU pass would add
`--tensor-parallel-size N` (and for MoE, tune `--kt-num-gpu-experts` per-GPU) and re-verify
residency across all N GPUs.

### Arguments

`infer_llm_ktransformers.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `probe` | `probe`: import kt-kernel, report `__cpu_variant__` + CUDA; `chat`: hit a running server |
| `--host` / `--port` | `127.0.0.1` / `38612` | Server address (chat mode); never SGLang's default port |
| `--model` | `qwen38-27b-fp8` | Served model name (matches `--served-model-name`) |
| `--prompt` / `--system_prompt` | see script | Chat inputs |
| `--max_tokens` / `--temperature` / `--top_p` / `--seed` | `128` / `0.0` / `1.0` / none | Sampling |
| `--timeout` | `600.0` | HTTP timeout (chat mode) |
| `--out` | none | Write a JSON artifact of the probe/chat result |

Probe exits 0 only when `kt_kernel` imports **and** a CUDA GPU is visible.

### Hardware support & evidence

| | Status |
|---|---|
| NVIDIA Hopper H100 (cc 9.0) | **Verified** — kt-kernel wheel supports SM 90; served on GPU 6. |
| **MoE offload (KT's real job)** | **Verified** — `Qwen/Qwen3-30B-A3B` (128 experts) AMX INT8, experts in CPU DRAM (~72 GB RSS, 16.41 GB on GPU), coherent output at ~23–50 tok/s. |
| NVIDIA Ampere/Ada (cc 8.0–8.9) | Upstream-supported (A100, RTX 3000/4000). Not tested here. |
| Intel AMX CPU (this host) | **Verified** — `__cpu_variant__ == 'amx'`, KT's best CPU tier; AMX MoE kernels created per-NUMA-node. |
| AMD GPU / ROCm | **Upstream-supported (Beta)** — `doc/en/ROCm.md`. **Not tested here.** |
| AMD Zen4 CPU (BLIS) / universal CPU (llamafile/GGUF) / Intel Arc XPU / Ascend NPU | Upstream-supported. Not tested here. |
| Repo dense target `Qwen/Qwen3.8-27B-FP8` | **Loads + runs, but DENSE** (`num_experts: null`) → KT's MoE offload is inert, and output is incoherent (FP8-scale + `qwen3_5` RoPE version gap). Use vLLM/SGLang for this model. |
| Embedding / reranker | ❌ upstream — KT is LLM-only. Use vLLM/TEI/SGLang/ONNX/llama.cpp. |

### Notes & quirks

- **Prebuilt wheels, no source build.** The brief anticipated a slow `kt-kernel` CPU-kernel
  compile; the current PyPI wheel makes it unnecessary on standard x86-64 + NVIDIA. Source build
  (`cd kt-kernel && ./install.sh`) is only for AMD BLIS / ARM KML / custom CUDA.
- **`sglang-kt`, not `sglang`.** Install the kvcache-ai fork. If the official `sglang` is
  present, `pip uninstall sglang -y` first.
- **cuDNN guard** (torch 2.9.1 + cuDNN < 9.15) blocks server startup until you bump cuDNN or set
  `SGLANG_DISABLE_CUDNN_CHECK=1`.
- **DeepGEMM JIT** first-run pre-compile is a 10-20 min one-time cost per CUDA-graph batch size;
  pre-run `python -m sglang.compile_deep_gemm` to amortize it.
- **The dense-vs-MoE point is the whole finding.** Seeing "Qwen3.8" or "Qwen3.5" in KT's docs
  refers to the **MoE** variants (Qwen3.5-MoE-400B, Qwen3-30B-A3B). The repo's dense 27B-FP8 VL
  checkpoint is outside KT's design point even though KT will happily serve it on the GPU.


---

# AMD MI355X (ROCm 7.2.4) — kernel library works, serving blocked · tested 2026-08-22

### Model

**`Qwen/Qwen3-30B-A3B`** — upstream's own worked example in `kt-kernel/README.md`, and a
genuine MoE: 48 layers, **128 experts per layer, top-8 routing**, `hidden_size 2048`,
`moe_intermediate_size 768`, 30.5B total / 3.3B active parameters, BF16 safetensors
(~57 GB). Chosen because:

- It is the smallest model that exercises **every** part of the split — a real router, a
  real 128-way expert bank, and enough expert parameters (~96% of the model) that moving
  them off the GPU is visible in `rocm-smi` rather than noise.
- `Qwen3MoeForCausalLM` is natively supported by `transformers` 5.15.1, so the all-GPU
  baseline is the *same code path* with only the placement changed. That makes the
  comparison honest.
- Its on-disk layout is the per-expert `model.layers.N.mlp.experts.M.gate_proj.weight`
  form, which kt-kernel's `BF16SafeTensorLoader` auto-detects as `deepseek` format.

`deepseek-ai/DeepSeek-V2-Lite-Chat` was the smaller fallback and was not needed.

### Install

See [`../README.md`](../README.md) for the full source build. The short version — **the
PyPI wheel is the NVIDIA build and will not give you a ROCm kt-kernel**:

```bash
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache HF_HOME=/mnt/data_1.5t/hf_cache
source /mnt/data_450g/envs/.env_inference_ktransformers/bin/activate
cd /mnt/data_450g/ktransformers_src/kt-kernel
export CPUINFER_USE_ROCM=1 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950
export CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_ENABLE_AMX=OFF
export CPUINFER_ENABLE_AVX512_VNNI=ON CPUINFER_ENABLE_AVX512_BF16=ON CPUINFER_ENABLE_AVX512_VBMI=ON
pip install . -v --no-build-isolation --no-deps      # --no-deps is MANDATORY
```

### Environment & secrets

`dev.env` is symlinked to the repo-root file (`ln -sf ../../../dev.env dev.env`) and
supplies `HF_TOKEN`, loaded by `load_dotenv("dev.env")`. Never echo it.

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides every GPU. Logs go to
`/mnt/data_450g/outputs/inference_ktransformers/`; nothing large lands in the repo.

### Commands

Environment probe — no model download, answers "is this build real?":

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

Tuned for this CPU (128 physical cores, 2 NUMA nodes) — these are already the defaults:

```bash
python infer_llm_ktransformers.py --mode generate \
  --kt_method BF16 --cpuinfer_threads 128 --threadpool_count 2 \
  --max_new_tokens 160 --seed 42 --sample_hw
```

### Probe results — the build is genuinely ROCm

```
torch              : 2.11.0+rocm7.2
torch.version.hip  : 7.2.26015
torch.version.cuda : None
visible devices    : 2
device 0           : AMD Instinct MI355X / gfx950:sramecc+:xnack-
kt_kernel version  : 0.7.0
kt_kernel variant  : avx512_bf16
GPU runtime linked : HIP (libamdhip64)
stream interop     : True
cpu isa            : avx512f=yes avx512bw=yes avx512vbmi=yes avx512_vnni=yes avx512_bf16=yes amx_tile=NO amx_int8=NO amx_bf16=NO
amx fast path      : UNAVAILABLE (EPYC — AVX512 path used)
```

`avx512_bf16` is the correct selection for an EPYC 9575F (Zen 5) and confirms kt-kernel's
detection is right: it picked the best AVX512 variant and did **not** try AMX.

Kernel check — the CPU MoE kernel is numerically correct on this silicon:

```
kernel forward     : 1.02 ms
relative L1 error  : 0.3913% vs PyTorch fp32 reference
verdict            : PASS
```

Upstream's own `examples/test_bf16_moe.py` agrees, over 5 iterations at 128 experts / top-8:

```
Mean relative L1 diff: 0.4160%
Max relative L1 diff:  0.4425%
PASS: Mean error 0.4160% < 5.0% threshold
Created BF16_MOE_TP 0 at numa 0 (backend=AVX512-BF16)
Created BF16_MOE_TP 1 at numa 1 (backend=AVX512-BF16)
```

Sub-0.5% is BF16 rounding, not a broken kernel. Note `backend=AVX512-BF16` and the
**two NUMA subpools** — kt-kernel splits the expert bank across both sockets by itself.

### The hybrid run — both halves are live

**Verdict: works, first try, no code changes to kt-kernel.** All 48 MoE layers offloaded:

```
host load          : 10.7s
kt-kernel layers   : 48 MoE layers offloaded to CPU
gpu upload         : 1.6s
model              : Qwen/Qwen3-30B-A3B
kt_method          : BF16
latency            : 9.17s
decode rate        : 17.44 tok/s (160 new tokens)
gpu vram (card0)   : 4.09 GB resident after load
cpu utilisation    : peak 56.7% / mean 30.6% over 15 samples
gpu vram during    : peak 5.60 GB
```

Per-layer expert load, from the same run (48 of these, ~0.21 s each, ~10 s total):

```
[BF16SafeTensorLoader] Detected format: deepseek
[NativeMoEWrapper Layer 0]  load_experts: 4.2ms, create_moe: 2.8ms, cpp_load_weights: 86.0ms, cleanup: 151.6ms, total: 244.6ms
[NativeMoEWrapper Layer 47] load_experts: 2.9ms, create_moe: 2.3ms, cpp_load_weights: 72.7ms, cleanup: 123.4ms, total: 201.4ms
```

#### Proof the GPU half is real

`rocm-smi` sampled every 2 s across the run. `card0` is this agent's GPU; it sits at
0.30 GB idle and jumps to ~5.4 GB when the non-expert half uploads:

```
timestamp  device,GPU use (%),VRAM Total (B),VRAM Used (B)
1787485761 card0,0,309220868096,299364352      <- 0.30 GB, before load
1787485786 card0,0,309220868096,299470848      <- still 0.30 GB (experts loading into DRAM)
1787485794 card0,0,309220868096,5406072832     <- 5.41 GB, non-expert half resident
1787485773 card1,100,309220868096,68042088448  <- sibling agent's TRAINING job, untouched
```

`card1`'s 68 GB at 100% belongs to a concurrent `mini_trainer` process, not to this run —
useful negative evidence that the hybrid run stayed on `card0` and used **4.09 GB** of a
288 GB card.

#### Proof the CPU half is real

System-wide CPU utilisation sampled at 2 Hz during decode: **peak 56.7%, mean 30.6%** of
256 logical CPUs — i.e. roughly 78–145 cores busy — while the GPU held only 4 GB and the
model kept emitting tokens. An idle machine reads ~1%. The `Created BF16_MOE_TP 0 at numa 0`
/ `numa 1` lines, printed 96 times (48 layers × 2 NUMA subpools), are kt-kernel building
that expert bank across both sockets.

#### Proof it is numerically correct end-to-end

The decisive test. Same model, same prompt, same greedy decode, **only the expert placement
differs**:

| Configuration | Placement | VRAM (`rocm-smi`, card0) | Decode |
|---|---|---|---|
| **kt-kernel hybrid** | 48 MoE layers on CPU (AVX512-BF16), rest on GPU | **4.09 GB** | **17.44 tok/s** |
| **All-GPU baseline** | whole model on one MI355X, `transformers` | **64.62 GB** (61.11 GB torch-allocated) | **35.13 tok/s** |
| Ratio | — | **15.8× less VRAM** | **2.0× slower** |

The generated text is **character-identical** across the two configurations for the full
length of the baseline output (291 characters / 64 tokens, verified programmatically). That
is the strongest possible correctness signal: routing the expert FFNs through kt-kernel's
CPU AVX512-BF16 kernels reproduces MI355X BF16 output token-for-token.

Real generated text from the hybrid run (`--max_new_tokens 160 --seed 42`):

```
--- response ---
<think>
Okay, the user is asking for a two-sentence explanation of a mixture-of-experts model and
why it saves compute. Let me start by recalling what a mixture-of-experts (MoE) model is.
From what I remember, MoE is a type of neural network architecture where multiple
specialized sub-models, called experts, are combined. Each expert is responsible for
different parts of the input data, and a gating network decides which experts to activate
for each input.

Now, why does this save compute? Well, in traditional models, all parameters are active for
every input, which can be computationally heavy. But with MoE, only the relevant experts
are used for each input, so the model doesn't have to process all parameters each time.
This reduces the amount of computation
```

Coherent, on-topic, correct about MoE, in Qwen3's native `<think>` scratchpad format. Not a
degraded or garbled sample.

**Throughput note.** A 64-token run measured **8.11 tok/s** and the 160-token run **17.44
tok/s** — the short run is dominated by first-call pinned-buffer allocation in
`KExpertsCPUBuffer`. Quote the longer number; pre-declare batch sizes with
`KTMoEWrapper.set_capture_batch_sizes([...])` to remove the warm-up entirely.

### Arguments / flags

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

### Notes & quirks

- **`--no-deps` is mandatory when building.** `kt-kernel/pyproject.toml` hard-pins
  `torch==2.9.1` and requires `triton>=2.0.0`. A plain `pip install .` uninstalls
  `torch 2.11.0+rocm7.2` and installs the **CUDA** torch and triton — ROCm torch ships
  `triton-rocm`, which does not satisfy the `triton` name. This is the single easiest way
  to destroy the environment. Re-check `torch.version.hip` after any pip operation.
- **`pip install kt-kernel` does not give you a ROCm build.** The PyPI wheel statically
  links the CUDA runtime and dlopens `libcuda.so.1`; there is no `libamdhip64` in it. On
  this host it silently degrades to CPU-only. Verify with
  `readelf -d $(python -c 'import kt_kernel;print(kt_kernel.kt_kernel_ext.__file__)') | grep NEEDED`
  — you want to see `libamdhip64.so.7`.
- **The AMX backends are permanently unavailable here.** EPYC 9575F has no `amx_tile`.
  `--kt-method AMXINT4/AMXINT8` — the configuration in most of upstream's tutorials and
  benchmarks — cannot run on this box. Build with `CPUINFER_ENABLE_AMX=OFF` and use the
  AVX512 native-precision methods.
- **`--num_gpu_experts > 0` is not usable from this standalone harness.** kt-kernel's
  `gpu_experts_mask` tells the CPU side to *skip* the masked experts on the assumption that
  the serving engine computes them with its own fused GPU MoE kernel. `transformers` has no
  such kernel, so masking experts here would silently drop them from the output. Partial
  expert placement requires the SGLang integration.
- **SGLang integration is structurally blocked on ROCm — verified, not assumed.**
  `sglang-kt` 0.7.0 (the kvcache-ai fork you must use, *not* upstream `sglang`) declares
  unconditional dependencies on `cuda-python`, `flashinfer_python`, `flashinfer_cubin`,
  `nvidia-cutlass-dsl`, `nvidia-ml-py`, `sgl-kernel==0.3.21` and `torch==2.9.1`. These are
  hard requirements, not extras, and `sgl-kernel` publishes CUDA-only wheels. This matches
  what `inference/sglang/` already found independently. A container would not fix it —
  the CUDA dependency is in the package metadata and the kernels themselves. **This is the
  real ceiling on KTransformers for ROCm: the kernel library ports cleanly, the serving
  layer does not.**
- **Quote the longer run for throughput.** First-call pinned-buffer allocation makes short
  generations look ~2× slower than steady state.
- **`kt version` runs but its diagnostics are CUDA-centric.** It reports
  `kt-kernel 0.7.0` correctly and then `CUDA  Not found` / `sglang-kt  Not installed`,
  followed by install instructions that do not apply on ROCm. It has no notion of HIP, so
  it cannot tell you whether your build is the ROCm one. Use `--mode probe` instead.
- **The extension prints unconditionally to stdout** (`CPUInfer[...]: Hello`,
  `Created BF16_MOE_TP 0 at numa 0`, one pair per layer). There is no quiet flag; filter it
  if you are parsing output.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs.

### Verdict

#### 1. Does it work on MI355X / gfx950 / ROCm 7.2.4? — ✅ **YES, cleanly.**

`kt-kernel` 0.7.0 builds from source with `CPUINFER_USE_ROCM=1` in ~2 minutes at `-j64`
with **zero errors**, links `libamdhip64.so.7`, imports `hipLaunchHostFunc`, auto-selects
the correct `avx512_bf16` CPU variant on the EPYC 9575F, builds NUMA-aware expert pools
across both sockets, and runs a real 30B MoE model with all 48 expert layers on CPU and the
rest on one MI355X — producing text **character-identical to the all-GPU baseline**.

This was expected to be the hard part and it was the easy part. The reason is architectural:
kt-kernel's ROCm backend compiles **no device kernels**. Its entire GPU dependency is
`hipLaunchHostFunc`, used to enqueue CPU work on a HIP stream. There is nothing
gfx950-specific to break. The stale `doc/en/ROCm.md` (Radeon 7900XTX, ROCm 6.2.4, conda)
was not needed — a plain Python 3.12 venv on Ubuntu 24.04 worked, with no GLIBCXX wall.

#### 2. Is it useful on this box? — ⚠️ **No, not for anything this repo runs.**

The measurement that settles it: **4.09 GB vs 64.62 GB of VRAM, for 17.44 vs 35.13 tok/s.**
kt-kernel saved 60 GB of VRAM on a card that has **288 GB**, and charged 2× throughput for
it. It optimised the resource this machine has most of, using the resource it can least
easily scale.

Concretely: `Qwen/Qwen3.8-27B-FP8`, this repo's target model, occupies 29.38 GiB on one
MI355X (measured in `inference/vllm/llm/`) and serves at 38.6 tok/s with 95 GiB of KV cache
left over. There is no VRAM problem to solve. Even DeepSeek-V3/R1 671B FP8 — the flagship
KTransformers use case — fits in 8× 288 GB at full GPU speed.

**Where it would earn its place, and does not today:**

- **1T-class models at BF16** (Kimi-K2, ~2 TB) genuinely strain 2.3 TB of VRAM, and this
  agent only owns 2 of 8 GPUs. There, CPU-expert offload into 2.2 TB of DRAM is a real
  option rather than a downgrade — and the 2× throughput cost measured here is the honest
  estimate of what it would cost. **But you cannot get there on ROCm today**, because
  serving a model that size needs SGLang, and `sglang-kt` is CUDA-only by package
  metadata. The Python API demonstrated here is a single-stream research harness, not a
  server.
- **Keeping GPUs free for training** while a big MoE answers low-QPS requests is a real
  scheduling argument — this host is running concurrent training jobs on other cards. But
  17 tok/s single-stream with no continuous batching is not a serving story.

#### Recommendation

**Keep the folder — as a well-evidenced, precisely-bounded negative.** Its value is that it
converts two open questions into settled facts, both of which are non-obvious:

1. *"Does the ROCm path in kt-kernel actually work on modern AMD hardware?"* — **Yes, and
   better than the beta-quality docs suggest**, because the port is trivially thin. That is
   worth knowing, and worth re-checking cheaply with `--mode probe`.
2. *"Could we use KTransformers to serve something huge here?"* — **Not today.** The blocker
   is not the GPU, the CPU, ROCm, or gfx950; it is that `sglang-kt`'s CUDA-only dependency
   chain has no ROCm equivalent. That is the specific thing to re-check when revisiting,
   and it is upstream's decision, not ours.

Do **not** promote this to a serving stack. For every workload in this repo,
`inference/vllm` and `inference/sglang` are strictly better on this hardware.
