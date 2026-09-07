# TEI — Hugging Face Text Embeddings Inference (MI355X: source build; H100: prebuilt Hopper image)

**TEI** is Hugging Face's Rust HTTP server purpose-built for embedding,
sequence-classification, and (cross-encoder) reranking models — `POST /embed`, an
OpenAI-compatible `POST /v1/embeddings`, batching, and Prometheus metrics, with a much
smaller footprint than a general LLM engine.

**The MI355X route that worked:** there is no published AMD/ROCm container tag, so the
Rust router (TEI 1.9.3) was **built from source inside a ROCm PyTorch container** (any
ROCm 7.2 torch image for gfx950 works; an SGLang-ROCm image already on disk was used).
Rust + protoc added, `cargo build --release ... -F python -F http --no-default-features`,
3m 44s, **no source patches needed for gfx950**. Upstream documents ROCm as *experimental*
and tested only on MI200/MI300, so gfx950 is beyond its documented matrix.
Decisive ROCm-not-CPU evidence: the `warmup_rocm` log line.

**The H100 route that worked:** NVIDIA *does* publish an image, so
there is **no source build** — pull `ghcr.io/huggingface/text-embeddings-inference:hopper-1.9`
(the **Hopper** tag for cc 9.0; the generic `cuda-*` tag is the wrong arch) and `docker run`
it. Same router (TEI 1.9.3). One deviation: the Hopper (candle CUDA) backend **rejects
`bfloat16`** — it accepts only `float16`/`float32`, so serve `--dtype float32` (the
EmbeddingGemma card warns against fp16). Verified on 1×H100 80GB, driver 580.173.02, CUDA 13.0:
correct 768-dim fp32 embeddings, correct semantic ranking, ~19–35 ms/request. Decisive
CUDA-not-CPU evidence: the candle `Starting Gemma3 model on Cuda(CudaDevice(...))` log line
plus `text-embeddings-router` holding ~1.8 GB on the assigned GPU by PID. Full commands and
evidence in [`embedding/README.md`](embedding/README.md).

## Scope — why only an `embedding/` leaf exists

Deliberate, not an omission:

- **No `llm/`** — TEI is not a generative server; there is no token-generation path.
  Use `../vllm/llm/`, `../sglang/llm/`, `../llamacpp/llm/`, or `../transformers/llm/`.
- **No `reranker/`** — TEI's `/rerank` targets sequence-classification cross-encoders
  (XLM-RoBERTa, GTE, ModernBERT). `Qwen/Qwen3-Reranker-0.6B` is decoder-only yes/no
  scoring, unsupported upstream. Use [`../llamacpp/reranker/`](../llamacpp/reranker/) or
  [`../vllm/reranker/`](../vllm/reranker/).

## Leaves

| Leaf | Status |
|---|---|
| [`embedding/`](embedding/README.md) — MI355X | **Works (build-from-source required)** — EmbeddingGemma-300m, correct 768-dim bf16 embeddings, ~20 ms/request on 1×MI355X; genuine ROCm backend confirmed. |
| [`embedding/`](embedding/README.md) — H100 | **Works (prebuilt Hopper image, no build)** — `hopper-1.9` image, EmbeddingGemma-300m, correct 768-dim **fp32** embeddings, correct ranking, ~19–35 ms/request on 1×H100; candle CUDA backend resident by PID. Deviation: `--dtype float32` (Hopper image rejects bf16). |

## Environment & requirements

- Client deps: [`requirements.txt`](requirements.txt) — just `requests` + `python-dotenv`;
  the server is a Rust binary (build toolchain documented in the same file).
- Venv convention: `python3 -m venv .env_tei` at this software root.
- Secrets: `HF_TOKEN` from `dev.env` (EmbeddingGemma is a gated model). TEI echoes a
  partially-masked token in its startup log — keep server logs out of git.

## Shared quirks

- `backend device: cuda` on AMD is correct, not a misconfiguration — ROCm reuses the
  `torch.cuda` API surface. Confirm ROCm via `warmup_rocm`.
- No tensor-parallel option exists; the production multi-GPU pattern is N independent
  single-GPU replicas behind a load balancer.

## Other hardware (upstream claims — not verified here)

Upstream ships prebuilt CPU images (ONNX/Intel-MKL backends) and CUDA images for NVIDIA
Turing through Blackwell (compute ≥7.5 only; ARM64/aarch64 covered for CPU and CUDA);
Apple Silicon Metal is a supported local build target; Intel CPU/XPU/HPU (Gaudi 2/3 only)
images exist in the main repo (the separate `tei-gaudi` fork is deprecated). Of these, the
**NVIDIA Hopper (H100, cc 9.0) CUDA path is now verified here** (`hopper-1.9`, see above and
`embedding/README.md`); the remaining targets (CPU/ONNX, other CUDA arches, Metal, Gaudi)
are still upstream claims, not verified in this repo.
