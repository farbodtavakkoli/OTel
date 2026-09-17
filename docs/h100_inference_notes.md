# H100 (NVIDIA / CUDA) inference notes

Lessons from serving the inference stacks on **NVIDIA H100 80GB HBM3** (Hopper cc 9.0,
CUDA 13.0), including what reverses when moving from ROCm to CUDA. Covers `transformers`,
`tensorrtllm`, `llamacpp`, and `sglang`. The per-stack support table is in the top-level
[README](../README.md).

Target models: LLM `Qwen/Qwen3.8-27B-FP8`, embedding `google/embeddinggemma-300m`, reranker
`Qwen/Qwen3-Reranker-0.6B`, plus GGUF equivalents.

The AMD notes in [mi355x_inference_notes.md](mi355x_inference_notes.md) cover the same
stacks; its silent-corruption traps and GPU-residency rule apply here too.

## TensorRT-LLM: NVIDIA-only, no ROCm counterpart

- `pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm` gives
  **tensorrt-llm 1.2.1** + nvidia-modelopt 0.37.0 + flashinfer. It downgrades torch to
  2.9.1+cu128, which runs fine on a cu130 driver.
- It needs a **system MPI**. Without `libmpi.so`, `import tensorrt_llm` fails with
  `cannot load MPI library` — install it with `sudo apt install libopenmpi3 openmpi-bin`.
- **`Qwen/Qwen3.8-27B-FP8` does not load here.** It declares `model_type: qwen3_5`, which
  TRT-LLM 1.2.1 does not register, so it fails at config parse (`KeyError: 'qwen3_5'`)
  before reaching FP8 or weights. This is a version gap, not FP8 or hardware, so a modelopt
  requant does not help. A supported arch such as `Qwen3ForCausalLM` loads and generates on
  the PyTorch backend.

## SGLang: the pip route works on CUDA

Unlike on ROCm, no container is needed. `pip install "sglang[all]"` gives sglang 0.5.18 +
sgl_kernel 0.4.6.post1, and torch stays 2.13.0+cu130. All three workloads (LLM, embedding,
reranker) serve from that one venv.

SGLang 0.5.18 loads `Qwen/Qwen3.8-27B-FP8` where TRT-LLM 1.2.1 cannot, because the `qwen3_5`
arch is registered in-tree. It reads FP8 natively (E4M3) and generates correct text.

One 80 GB quirk: the mamba cache auto-caps `--max-running-requests` to about 24. Raise it
with `--mamba-ssm-dtype bfloat16`.

## llama.cpp: build with `-DGGML_CUDA=ON`

The build is the clean reversal of the ROCm recipe — drop `-DGGML_HIP=ON
-DGPU_TARGETS=gfx950` and use:

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON
```

Keep `-DLLAMA_OPENSSL=ON`; it is vendor-neutral and required for `-hf` downloads. cmake
auto-detects `CMAKE_CUDA_ARCHITECTURES=90-real` for Hopper, so no arch flag is needed. All
three workloads work.

Two host requirements: `cmake` and `ninja` may be missing (pip-install them), and the build
tree plus `LLAMA_CACHE` must sit on local disk or tmpfs — a network share can reject the
rename operations that pip and `-hf` rely on.

## Cross-cutting reversals and traps

- **Always prove GPU residency.** A backend can report itself active and return correct
  numbers while running on CPU. Check a concrete signal sampled from inside the job:
  `nvidia-smi` VRAM by PID, llama.cpp's `offloaded N/N layers`, or Ollama's `100% GPU`. On a
  shared box, sample by PID or UUID so you do not measure a co-tenant's process.
- **Do not assume GGUF files are cached.** If `$HF_HOME` holds no `*-GGUF` repos, the GGUF
  stacks (llamacpp, ollama, lemonade) have to download one. A few-hundred-MB model is enough
  for a smoke test; the production 29 GB `unsloth/Qwen3.8-27B-GGUF` Q8_0 fits one H100.
- **The prebuilt flash-attn 2.8.3 wheel is ABI-broken against torch 2.13/cu130**
  (`undefined symbol _ZN3c10…materialize_cow_storage`). Rebuild from source or use
  `attn_implementation=sdpa`. Only bare imports at module load force the issue.
- **Reverse the ROCm knobs.** Use plain `CUDA_VISIBLE_DEVICES` and drop
  `HIP_VISIBLE_DEVICES`. The AITER GEMM corruption trap is gfx950-specific and does not
  apply here — but still read the output and confirm it is coherent rather than token salad.
- **A clean `pip install` proves nothing.** Test the import and a real device op, never just
  the install.

## FP8 on Hopper

FP8 is native on Hopper, so the constraint is never the datatype — it is whether the engine
registers the `qwen3_5` arch. The 27B FP8 checkpoint (OCP E4M3FN) loads and serves wherever
that arch is supported (SGLang 0.5.18 yes, TRT-LLM 1.2.1 no).
