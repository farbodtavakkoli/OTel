# `inference/tei` — Hugging Face Text Embeddings Inference

TEI is Hugging Face's Rust HTTP server purpose-built for embedding and
sequence-classification models — `POST /embed`, an OpenAI-compatible `POST /v1/embeddings`,
batching and Prometheus metrics, with a much smaller footprint than a general LLM engine.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2) and NVIDIA H100 80GB (Hopper cc 9.0, CUDA 13),
TEI 1.9.3. The two routes differ substantially:

| Vendor | Route | dtype |
|---|---|---|
| NVIDIA / CUDA | Prebuilt `ghcr.io/huggingface/text-embeddings-inference:hopper-1.9` — no build | `float32` (the Hopper backend rejects `bfloat16`) |
| AMD / ROCm | No published image; build the Rust router from source inside a ROCm PyTorch container | `bfloat16` |

Use the **Hopper** tag for cc 9.0 — the generic `cuda-*` tag targets a different
architecture. No source patches are needed for gfx950.

## Leaves

The one leaf is [`embedding/`](embedding/README.md), serving `google/embeddinggemma-300m` as
768-dim mean-pooled vectors.

TEI is not a generative server, so there is no `llm/` leaf — use
[`../vllm/llm/`](../vllm/llm/), [`../sglang/llm/`](../sglang/llm/),
[`../llamacpp/llm/`](../llamacpp/llm/) or [`../transformers/llm/`](../transformers/llm/).
Reranker support is limited: TEI's `/rerank` serves sequence-classification cross-encoders
(XLM-RoBERTa, GTE, ModernBERT), but not the decoder-only yes/no
`Qwen/Qwen3-Reranker-0.6B` this repo uses, so there is no `reranker/` leaf either — use
[`../llamacpp/reranker/`](../llamacpp/reranker/) or [`../vllm/reranker/`](../vllm/reranker/).

## Setup

Full per-vendor commands are in [`embedding/README.md`](embedding/README.md). Client deps are
just `requests` + `python-dotenv`:

```bash
python3 -m venv .env_tei        # at this stack root; use local or tmpfs disk, not CIFS/NFS
.env_tei/bin/pip install -r requirements.txt
```

`HF_TOKEN` from `dev.env` is required — EmbeddingGemma is a gated model.

## Notes

- `backend device: cuda` on AMD is correct, not a misconfiguration: ROCm reuses the
  `torch.cuda` API surface. Confirm the real backend via the `warmup_rocm` log line.
- There is no tensor-parallel option. The multi-GPU pattern is N independent single-GPU
  replicas behind a load balancer.
- TEI echoes a partially-masked `HF_TOKEN` in its startup log — keep server logs out of git.
