# MI355X (AMD / ROCm) inference notes

Traps and lessons from serving the inference stacks on AMD Instinct MI355X (gfx950, 288 GB),
ROCm 7.2.4. The per-stack support table is in the top-level [README](../README.md).

Target models: LLM `Qwen/Qwen3.8-27B-FP8`, embedding `google/embeddinggemma-300m`, reranker
`Qwen/Qwen3-Reranker-0.6B`, with GGUF equivalents where a stack needs a converted checkpoint.

## The two silent-corruption traps

Read these first. Both return **HTTP 200 and plausible-looking output**, so a smoke test that
only checks for a response will pass while the deployment is wrong.

1. **vLLM + AITER emits corrupt output on gfx950.** `VLLM_ROCM_USE_AITER=1` is the gfx950
   default, and with the FP8 27B it selects `AiterFp8BlockScaledMMKernel`, which produces
   garbage like `WeGPG G02 q GCGvG/G/…`. It is also *faster* than the correct
   `TritonFp8BlockScaledMMKernel` path, so a throughput benchmark will actively choose the
   broken config.
   **Set `VLLM_ROCM_USE_AITER=0`.** `VLLM_ROCM_USE_AITER_MOE=0` is the wrong fix — this is a
   GEMM bug, not an MoE bug.
2. **EmbeddingGemma GGUF ships no prompt template.** Ollama and any raw GGUF path pass input
   through verbatim, so omitting the `task: search result | query:` and
   `title: none | text:` prefixes still returns well-formed, L2-normalised 768-dim vectors
   while retrieval quality quietly degrades.

## Lessons from the inference runs

- **FP8 works on gfx950.** Transformers loads `Qwen3.8-27B-FP8` natively (OCP E4M3FN, not
  the gfx942-era FNUZ), and SGLang's container generates from it at TP=1 and TP=2. When an
  engine fails on FP8, it is an engine-version problem, not hardware.
- **A blocked stack is usually packaging, not hardware.** SGLang's HIP path imports
  `sgl_kernel`, which ships CUDA-only wheels — its vendor container serves all three
  workloads fine. That is not a gfx950 limitation.
- **Always prove GPU residency.** A backend that reports itself active and returns correct
  answers can still be running on CPU. Check a concrete signal: llama.cpp's
  `offloaded N/N layers`, Ollama's `100% GPU`, TEI's `warmup_rocm`, or a `rocm-smi` VRAM
  delta on the specific card.
- **Small models do not shard.** For the 300M embedder and 0.6B reranker, TP=2 is pointless
  or rejected outright — vLLM fails EmbeddingGemma at config validation. Run N single-GPU
  replicas instead, which is what those folders demonstrate.
- **A clean `pip install` proves nothing about hardware.** `pip install tensorrt-llm`
  succeeds on a pure-AMD host, installs ~16 GB of CUDA userspace, and only fails at import
  with `libcuda.so.1: cannot open shared object file`. Test the import and a real device op.
- **`torch.cuda.is_available()` is not a CUDA check on ROCm.** ROCm aliases `torch.cuda` onto
  HIP, so it returns `True` with no NVIDIA hardware present. Check `torch.version.cuda`
  against `torch.version.hip` instead.
- **Prebuilt ROCm backends may already be arch-matched.** Lemonade downloads a
  `rocm_sdk_device_gfx950` wheel and a fat `libggml-hip.so` carrying gfx950 alongside
  gfx90a and gfx942, so it works with no compiler and no `hipErrorNoBinaryForGpu`.
- **Three `llama.cpp` details that are easy to miss:** the `-DGGML_HIP=ON` build cannot
  download models unless you add `-DLLAMA_OPENSSL=ON` (this revision swapped libcurl for its
  own HTTP stack); `LLAMA_CACHE`, not `HF_HOME`, controls `-hf` downloads; and
  `--split-mode row` is unimplemented on the CUDA/HIP backend.

## vLLM and the 27B checkpoint

The full `Qwen/Qwen3.8-27B-FP8` checkpoint serves at TP=1 and TP=2 on the vLLM 0.20.2
container — no newer image is needed. Despite the version number, 0.20.2 already ships
`Qwen3_5ForConditionalGeneration` and auto-detects `quantization=fp8`. `inference/vllm/llm`
keeps a `Qwen3-4B` run as the smaller-model control.

The AITER trap above applies to the default configuration.

## SGLang: use the container, not pip

The pip route does not work on ROCm. SGLang's HIP path does
`from sgl_kernel import rotary_embedding`, and `sglang-kernel` publishes CUDA-only wheels, so
no AITER build can substitute.

Use the vendor image `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-*`, which serves all three
workloads including the FP8 27B at TP=1 and TP=2. See the `inference/sglang/*` folders.
