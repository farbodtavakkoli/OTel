# TEI — Hugging Face Text Embeddings Inference

**TEI** is Hugging Face's Rust HTTP server purpose-built for embedding,
sequence-classification, and (cross-encoder) reranking models — `POST /embed`, an
OpenAI-compatible `POST /v1/embeddings`, batching, and Prometheus metrics, with a much
smaller footprint than a general LLM engine.

## Hardware support

**AMD / ROCm — source build required.** There is no published AMD/ROCm container tag, so
the Rust router (TEI 1.9.3) must be **built from source inside a ROCm PyTorch container**
(any ROCm 7.2 torch image for gfx950 works). Add Rust + protoc, then
`cargo build --release ... -F python -F http --no-default-features`, with **no source
patches needed for gfx950**. Confirm the ROCm backend (not a CPU fallback) via the
`warmup_rocm` log line.

**NVIDIA / CUDA — prebuilt image, no build.** Pull
`ghcr.io/huggingface/text-embeddings-inference:hopper-1.9` (the **Hopper** tag for cc 9.0;
the generic `cuda-*` tag is the wrong arch) and `docker run` it. Same router (TEI 1.9.3).
One deviation: the Hopper (candle CUDA) backend **rejects `bfloat16`** — it accepts only
`float16`/`float32`, so serve `--dtype float32` (the EmbeddingGemma card warns against fp16).
Confirm the CUDA backend via the candle `Starting Gemma3 model on Cuda(CudaDevice(...))`
log line plus `text-embeddings-router` holding VRAM on the assigned GPU by PID.

Works on **MI355X (gfx950, ROCm 7.2)** and **H100 80GB (Hopper cc 9.0, CUDA 13)**. Full
commands in [`embedding/README.md`](embedding/README.md).

## Scope — why only an `embedding/` leaf exists

- **No `llm/`** — TEI is not a generative server; there is no token-generation path.
  Use `../vllm/llm/`, `../sglang/llm/`, `../llamacpp/llm/`, or `../transformers/llm/`.
- **No `reranker/`** — TEI's `/rerank` targets sequence-classification cross-encoders
  (XLM-RoBERTa, GTE, ModernBERT). `Qwen/Qwen3-Reranker-0.6B` is decoder-only yes/no
  scoring, unsupported upstream. Use [`../llamacpp/reranker/`](../llamacpp/reranker/) or
  [`../vllm/reranker/`](../vllm/reranker/).

## Leaves

| Leaf | Status |
|---|---|
| [`embedding/`](embedding/README.md) — AMD | **Works, build-from-source required** — EmbeddingGemma-300m, 768-dim bf16 embeddings. |
| [`embedding/`](embedding/README.md) — NVIDIA | **Works, prebuilt `hopper-1.9` image, no build** — EmbeddingGemma-300m, 768-dim embeddings. Deviation: `--dtype float32` (the image rejects bf16). |

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
