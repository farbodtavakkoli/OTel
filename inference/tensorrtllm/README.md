# TensorRT-LLM — NVIDIA only

**TensorRT-LLM** is NVIDIA's high-performance LLM serving stack: compiled engines,
vendor-tuned kernels, FP8/FP4 quantization, TP/PP/DP/EP parallelism, and Triton/Dynamo
integration. On an NVIDIA cluster it is a top-tier candidate.

**Works on NVIDIA H100 80GB (CUDA 13).** The install recipe, versions, and model-support
caveats are in [`llm/README.md`](llm/README.md).

**Not supported on AMD/ROCm.** There is no ROCm build. `pip install tensorrt-llm` *succeeds*
on an AMD box and only fails later at `import tensorrt_llm` with `ImportError: libcuda.so.1`,
so a clean install is not a compatibility signal.

For LLM serving on an AMD host use [`../vllm/llm/`](../vllm/llm/) or
[`../sglang/llm/`](../sglang/llm/) instead.

## Scope — why only an `llm/` leaf exists

The embedding (`google/embeddinggemma-300m`) and reranker (`Qwen/Qwen3-Reranker-0.6B`)
workloads have no first-class recipe in TensorRT-LLM even on NVIDIA hardware, so no leaves
were created for them.

## Leaves

| Leaf | Status |
|---|---|
| [`llm/`](llm/README.md) | **Works on NVIDIA; not supported on AMD** — the NVIDIA install/run recipe, the host probe, and the model-support caveats. |

## Environment & requirements

- Client/probe deps: [`requirements.txt`](requirements.txt) (just `python-dotenv`; the
  commented NVIDIA install route lives there too).
- Venv convention: `python3 -m venv .env_tensorrtllm` at this software root.
- Secrets/ports: `dev.env` at this root; leaves symlink it (`ln -sf ../../../dev.env` from
  a leaf against the repo-root `dev.env`, or use the local copies as committed).
