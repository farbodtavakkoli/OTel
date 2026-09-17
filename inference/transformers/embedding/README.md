# `inference/transformers/embedding` — reference embedding inference

Reference embeddings for `google/embeddinggemma-300m` via sentence-transformers. The
cosine matrix this leaf writes is the correctness baseline that vLLM, SGLang, llama.cpp and
TEI are diffed against.

The script encodes a query set and a document set with EmbeddingGemma's own
`encode_query` / `encode_document` prompts, prints the ranked cosine-similarity matrix,
checks that relevant documents outrank irrelevant ones, and writes a JSON artifact. It
exits non-zero on a failed check, so it works as a CI gate.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2) — 1 and 2 GPUs · NVIDIA H100 80GB (CUDA 13) —
single-GPU, pass `--attn_impl sdpa`.

## Files

- `embed_transformers.py` — encode, score, sanity-check, write the reference artifact.

## Setup

See [`../README.md`](../README.md) — one venv at `inference/transformers/` serves all three
leaves. Do not install `flash-attn`.

```bash
export HF_HOME=/path/to/hf_cache       # model cache
export OUTPUT_DIR=/path/to/outputs
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # the GPUs you want to use
```

`dev.env` (symlinked to the repo-root file) must supply `HF_TOKEN` — the EmbeddingGemma
repo is gated. Offline runs: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Never set
`CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--devices` indexes *within* the
visible set, so `cuda:0` is the first card listed above.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source ../.env_transformers/bin/activate
python embed_transformers.py --dtype float32 --devices cuda:0 \
  --output $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_1gpu.json
```

Multi-GPU (one full model replica per device, batch split between them — data parallel).
Pool startup dominates on small text sets, so use one GPU unless the corpus is large:

```bash
python embed_transformers.py --dtype float32 --devices cuda:0,cuda:1 \
  --output $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_2gpu.json
```

Your own corpus, Matryoshka-truncated to 256 dims:

```bash
python embed_transformers.py --queries_file my_queries.json --documents_file my_docs.json \
  --truncate_dim 256 --batch_size 128
```

Use `--dtype float32` for golden artifacts. bf16 is not batch-invariant — a different batch
composition moves cosines by ~2e-3 and leaves `‖q‖` slightly off `1.0`. Rankings are
unaffected either way, but a bf16 artifact is not bitwise reproducible.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `google/embeddinggemma-300m` | Embedding model id or path |
| `--queries_file` | `None` | JSON list / JSONL / plain-text queries; omit for the built-in set |
| `--documents_file` | `None` | JSON list / JSONL / plain-text documents; omit for the built-in set |
| `--dtype` | `bfloat16` | `bfloat16`, `float32`, `float16`. Use `float32` for reference artifacts; the model card warns against `float16` |
| `--devices` | `cuda:0` | Comma-separated devices; 2+ switches to multi-process encoding |
| `--batch_size` | `32` | Encode batch size |
| `--max_seq_length` | `None` | Override the model's max sequence length (default 2048) |
| `--truncate_dim` | `None` | Matryoshka output dim (768/512/256/128); omit for full 768 |
| `--normalize` / `--no_normalize` | on | L2-normalize embeddings |
| `--attn_impl` | `None` | Auto-selects `sdpa` on ROCm, `eager` elsewhere; pass `sdpa` on CUDA |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

stdout carries the model/dtype/attention/device line, load time and VRAM, encode time,
embedding shapes (dim **768**), the first 8 components and L2 norm of `query[0]`, the
ranked cosine matrix, and a `PASS`/`FAIL` verdict. Exit code 0 on PASS, 1 on FAIL.

`--output` writes the artifact other stacks diff against:

```
$OUTPUT_DIR/inference_embedding_transformers/
  reference_embedding_fp32_1gpu.json    <- canonical: exactly reproducible
  reference_embedding_fp32_2gpu.json    <- byte-identical to the above
  reference_embedding_1gpu.json         <- bf16 variant
```

This is the file [`../../ollama/embedding`](../../ollama/embedding) and the other serving
stacks diff against. Each JSON records model, dtype, attention implementation, devices,
seed, torch/HIP versions, GPU arch, embedding dim, timings, the queries and documents
verbatim, the full cosine matrix, the first 16 components of each query embedding, the
per-query L2 norms, and the verdict. To compare another engine, encode the same texts and
diff `cosine_similarities` — dimensionless and dtype-robust, unlike raw vectors.
