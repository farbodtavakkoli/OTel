# `inference/ktransformers/llm` — CPU-GPU heterogeneous MoE inference

Serves a large Mixture-of-Experts model by holding the MoE experts in CPU DRAM
(AMX/AVX512/BLIS kernels, NUMA-aware) while the GPU runs attention and the active weights.
Reach for it when the weights exceed VRAM and the host has a strong server CPU; for dense
models, or models that fit in VRAM, use [`../../vllm/llm/`](../../vllm/llm/) or
[`../../sglang/llm/`](../../sglang/llm/).

**Hardware:** NVIDIA H100 (Intel Xeon/AMX host) — full `sglang-kt` serving path; kt-kernel
needs compute capability 8.0+ (Ampere/Ada/Hopper). AMD MI355X (gfx950, EPYC Zen 5,
AVX512-BF16 — no AMX) — the kernel library and the direct-API hybrid work, but serving is
blocked: `sglang-kt` 0.7.0 hard-pins CUDA-only dependencies.

## Files

- `infer_llm_ktransformers.py` — four modes: `probe` (environment/import check), `kernel`
  (synthetic MoE kernel vs a PyTorch fp32 reference), `generate` (direct-API hybrid
  generation), `chat` (client for a running `sglang-kt` server).

## Setup

See [`../README.md`](../README.md) for the per-vendor install.

```bash
export HF_HOME=/path/to/hf_cache       # model cache
export DATA_DIR=/path/to/data          # source checkouts and scratch space
export OUTPUT_DIR=/path/to/outputs     # run logs and artifacts
source ../.env_ktransformers/bin/activate
```

`dev.env` (symlinked to the repo-root file) supplies `HF_TOKEN`. Never set
`CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides every GPU; use explicit indices.
Keep the proxy unset for any command that downloads.

## Probe first

```bash
python infer_llm_ktransformers.py --mode probe --out $OUTPUT_DIR/ktransformers/llm/probe.json
```

It exits 0 only when `kt_kernel` imports and a GPU is visible. The lines that matter are the
CPU variant and the linked GPU runtime:

```
kt_kernel version  : 0.7.0
kt_kernel variant  : amx            # avx512_bf16 on EPYC — AMX does not exist on Zen 5
GPU runtime linked : HIP (libamdhip64)   # or CUDA
stream interop     : True
```

Numerical check of the CPU kernels against a PyTorch fp32 reference (no model download):

```bash
python infer_llm_ktransformers.py --mode kernel
# relative L1 error  : 0.3913%   verdict: PASS
```

Sub-0.5% is BF16 rounding; upstream's own threshold is 5%.

## Serve an MoE model (NVIDIA)

`Qwen/Qwen3-30B-A3B` is `Qwen3MoeForCausalLM`: 48 layers, 128 experts, 8 active per token.

Download the bf16 weights:

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir <models>/Qwen3-30B-A3B
```

Convert the experts to AMX INT8 CPU weights. Use INT8, not INT4 — INT4 costs significant
accuracy on this model:

```bash
cd kt-kernel
python scripts/convert_cpu_weights.py \
  --input-path <models>/Qwen3-30B-A3B --input-type bf16 \
  --output <models>/Qwen3-30B-A3B-INT8 --quant-method int8 \
  --cpuinfer-threads 48 --threadpool-count 2
```

Serve. `--model` points at the **bf16 GPU** weights, `--kt-weight-path` at the **INT8 CPU**
weights, and `--kt-num-gpu-experts 32` keeps 32 of 128 hot experts on the GPU while the rest
stream from DRAM. Use a distinct port, never SGLang's default:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
  --host 127.0.0.1 --port 8380 \
  --model <models>/Qwen3-30B-A3B \
  --kt-method AMXINT8 --kt-weight-path <models>/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 48 --kt-threadpool-count 2 --kt-num-gpu-experts 32 \
  --served-model-name qwen3-30b-a3b --trust-remote-code \
  --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-mixed-chunk \
  --tensor-parallel-size 1
```

`Creating AMX_MOE_TP 0 at numa 0` / `1 at numa 1` in the log is the proof the experts landed
on the CPU, NUMA-aware. Wait for `The server is fired up and ready to roll!`, then prompt it:

```bash
python infer_llm_ktransformers.py --mode chat --port 8380 --model qwen3-30b-a3b
```

`Qwen3-30B-A3B` is a thinking model — it emits a `<think>...</think>` block unless you append
`/no_think` to the prompt.

For multi-GPU add `--tensor-parallel-size N` and tune `--kt-num-gpu-experts` per GPU.

### Confirm the offload happened

From a second shell while generating, only a fraction of the parameter mass should be on the
GPU with the INT8 experts resident in DRAM:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <N>
ps -o rss= $(pstree -p <server-pid>) | awk '{s+=$1} END{printf "%.1f GB\n",s/1048576}'
```

## Hybrid generate (direct Python API)

The only GPU path on AMD, and equally usable on NVIDIA. It offloads all 48 MoE layers and
reproduces the all-GPU `transformers` baseline character-for-character under greedy decode:

```bash
python infer_llm_ktransformers.py --mode generate --max_new_tokens 160 --sample_hw
```

The defaults below are already tuned for a 128-physical-core, 2-NUMA-node host:

```bash
python infer_llm_ktransformers.py --mode generate \
  --kt_method BF16 --cpuinfer_threads 128 --threadpool_count 2 \
  --max_new_tokens 160 --seed 42 --sample_hw
```

Pre-declare batch sizes with `KTMoEWrapper.set_capture_batch_sizes([...])` to remove the
first-call warm-up.

## Arguments

### `probe` / `kernel` / `generate`

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `probe` | `probe` / `kernel` / `generate` / `chat` |
| `--model` | `Qwen/Qwen3-30B-A3B` | HF repo id or local path of the MoE model |
| `--kt_method` | `BF16` | CPU backend: `BF16`, `FP8`, `FP8_PERCHANNEL`, `RAWINT4`, `LLAMAFILE`, `MOE_INT8`. `AMXINT4`/`AMXINT8` require an Intel AMX CPU |
| `--cpuinfer_threads` | `128` | CPU inference threads — set to *physical* cores, not hyperthreads |
| `--threadpool_count` | `2` | Thread pools — set to the NUMA node count |
| `--num_gpu_experts` | `0` | Experts kept on GPU. Leave at 0 here — see Notes |
| `--max_layers` | `0` | Offload only the first N MoE layers (0 = all); useful for bisecting |
| `--chunked_prefill_size` | `512` | Maximum prefill chunk |
| `--prompt` | MoE question | User prompt |
| `--max_new_tokens` | `64` | Use 160 or more for a meaningful tok/s |
| `--temperature` | `0.0` | 0 = greedy (required for the token-identity comparison) |
| `--seed` | `None` | `generate` resolves `None` to 42 |
| `--device` | `cuda:0` | Torch device for the non-expert half (ROCm exposes itself as `cuda`) |
| `--sample_hw` | off | Sample `rocm-smi` VRAM and CPU load during decode |
| `--out` | `None` | Write a JSON artifact of the probe/chat result |

### `chat` (against a running `sglang-kt` server)

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `38612` | Server address; never SGLang's default port |
| `--model` | `qwen38-27b-fp8` | Served model name (matches `--served-model-name`) |
| `--prompt` / `--system_prompt` | see script | Chat inputs |
| `--max_tokens` / `--temperature` / `--top_p` | `128` / `0.0` / `1.0` | Sampling |
| `--timeout` | `600.0` | HTTP timeout (s) |

## Notes

- **Install `sglang-kt`, not `sglang`.** If the official package is present,
  `pip uninstall sglang -y` first.
- **`--num_gpu_experts > 0` is not usable from this harness.** kt-kernel's
  `gpu_experts_mask` tells the CPU side to skip masked experts on the assumption that the
  serving engine computes them with its own fused GPU MoE kernel; `transformers` has no such
  kernel, so masking would silently drop them from the output. Partial expert placement
  requires the SGLang integration.
- **KT is for MoE checkpoints.** A dense checkpoint (for example `Qwen/Qwen3.8-27B-FP8`,
  `num_experts: null`) has nothing to offload — use vLLM or SGLang instead. "Qwen3.8" /
  "Qwen3.5" in KT's own docs means the MoE variants.
- **A dense hybrid-GDN model triggers a DeepGEMM JIT pre-compile** after weight load, once
  per CUDA-graph batch size. Pre-run `python -m sglang.compile_deep_gemm` to amortize it.
- **The extension prints to stdout unconditionally** (`CPUInfer[...]: Hello`,
  `Created BF16_MOE_TP 0 at numa 0`, one pair per layer). There is no quiet flag; filter it
  if you parse output.
- **`kt version` cannot tell you whether your build is the ROCm one** — its diagnostics are
  CUDA-centric and it has no notion of HIP. Use `--mode probe`.
