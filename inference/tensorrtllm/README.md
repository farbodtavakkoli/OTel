# `inference/tensorrtllm` — TensorRT-LLM (NVIDIA only)

NVIDIA's high-performance LLM serving stack: compiled engines, vendor-tuned kernels, FP8/FP4
quantization, TP/PP/DP/EP parallelism, and Triton/Dynamo integration. On an NVIDIA cluster it
is a top-tier candidate for throughput.

**Hardware:** NVIDIA H100 80GB (CUDA 13). There is no ROCm build — on an AMD host use
[`../vllm/llm/`](../vllm/llm/) or [`../sglang/llm/`](../sglang/llm/). Note that
`pip install tensorrt-llm` still exits 0 on an AMD box and only fails later at
`import tensorrt_llm` with `ImportError: libcuda.so.1`, so a clean install proves nothing.

TensorRT-LLM is LLM-only here: the embedding (`google/embeddinggemma-300m`) and reranker
(`Qwen/Qwen3-Reranker-0.6B`) workloads have no first-class recipe in it, so only an
[`llm/`](llm/) leaf exists.

## Setup

The install, the NGC container route, the host probe and the run commands live in
[`llm/README.md`](llm/README.md). The venv convention at this stack root is
`python3 -m venv .env_tensorrtllm`; [`requirements.txt`](requirements.txt) installs only what
the probe needs (`python-dotenv`), so you can check a host before committing to the ~16 GB
engine install. `dev.env` is symlinked from the leaf (`ln -sf ../../../dev.env dev.env`).
