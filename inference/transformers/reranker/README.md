# `inference/transformers/reranker` — reference reranker inference

Reference reranking for `Qwen/Qwen3-Reranker-0.6B` via the sentence-transformers
`CrossEncoder` API. The scores this leaf writes are the correctness baseline for vLLM's
`/v1/rerank`, SGLang's reranker path and llama.cpp's `--reranking` server.

The script scores every `(query, document)` pair, prints them ranked, checks that the
relevant documents outscore the irrelevant ones and occupy the top-k slots, and writes a
JSON artifact. It exits non-zero on a failed margin check, so it works as a CI gate.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2) — 1 and 2 GPUs · NVIDIA H100 80GB (CUDA 13) —
single-GPU, pass `--attn_impl sdpa`. TEI cannot serve this reranker: its `/rerank` targets
sequence-classification cross-encoders, not decoder-only yes/no scoring.

## Files

- `rerank_transformers.py` — score, rank, sanity-check, write the reference artifact.

## Setup

See [`../README.md`](../README.md) — one venv at `inference/transformers/` serves all three
leaves. Do not install `flash-attn`.

```bash
export HF_HOME=/path/to/hf_cache       # model cache
export OUTPUT_DIR=/path/to/outputs
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # the GPUs you want to use
```

`dev.env` (symlinked to the repo-root file) supplies `HF_TOKEN`. Offline runs:
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Never set `CUDA_VISIBLE_DEVICES=""` on ROCm —
use explicit indices. `--devices` indexes *within* the visible set.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source ../.env_transformers/bin/activate
python rerank_transformers.py --dtype float32 --devices cuda:0 \
  --output $OUTPUT_DIR/inference_reranker_transformers/reference_rerank_fp32_1gpu.json
```

Multi-GPU (one full model replica per device, pairs split between them — data parallel).
Pool startup dominates on a handful of pairs, so use one GPU for a typical top-50 rerank:

```bash
python rerank_transformers.py --dtype float32 --devices cuda:0,cuda:1 \
  --output $OUTPUT_DIR/inference_reranker_transformers/reference_rerank_fp32_2gpu.json
```

Your own query and candidate set:

```bash
python rerank_transformers.py --query "How do I enable ROCm in PyTorch?" \
  --documents_file candidates.json --batch_size 64 --max_len 2048
```

Use `--dtype float32` for reference artifacts: bf16 quantizes scores visibly (~0.2 logits),
and even in fp32 re-chunking pairs across processes moves them ~3e-5. Rankings are
unaffected in every configuration.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder model id or path |
| `--query` | built-in ROCm query | Query to rerank documents against |
| `--documents_file` | `None` | JSON list / JSONL / plain-text documents; omit for the built-in set |
| `--dtype` | `bfloat16` | `bfloat16`, `float32`, `float16`. Use `float32` for reference artifacts |
| `--devices` | `cuda:0` | Comma-separated devices; 2+ switches to multi-process scoring |
| `--batch_size` | `32` | Scoring batch size |
| `--max_len` | `1024` | Max sequence length (query + document); the model supports 32K |
| `--activation` | `default` | `default` keeps the model's LogitScore head; `sigmoid` squashes to (0,1); `none` = identity |
| `--fix_pair_template` / `--no_fix_pair_template` | on | Installs a Query/Document pair chat template when the loaded one cannot carry both roles. Required for a bare checkpoint cache on sentence-transformers 5.7.0, where `predict()` otherwise raises `ValueError` |
| `--attn_impl` | `None` | Auto-selects `sdpa` on ROCm, `eager` elsewhere; pass `sdpa` on CUDA |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

stdout carries the model/dtype/attention/device line, load time and VRAM, scoring time,
the ranked `(score, document)` list, and two `PASS`/`FAIL` checks — margin and top-k
membership. Exit code is 0 on PASS, 1 on FAIL (margin check only).

`--output` writes the artifact other stacks diff against:

```
$OUTPUT_DIR/inference_reranker_transformers/
  reference_rerank_fp32_1gpu.json    <- canonical
  reference_rerank_fp32_2gpu.json    <- two-GPU run
  reference_rerank_bf16_1gpu.json    <- bf16 variant
```

Each JSON records model, dtype, attention implementation, devices, seed, max_len,
activation, torch/HIP versions, GPU arch, timings, the query and documents verbatim, the
per-document scores, the ranking permutation, and the verdict.

## Notes

- **Scores are unbounded logits, not probabilities.** Large positive and large negative
  values are normal; only relative order is meaningful. `--activation sigmoid` gives (0,1)
  values if a downstream consumer needs them.
- **Score scale depends on the prompt template.** A checkpoint that ships the publishing
  template yields raw logits; the yes-no template installed by `--fix_pair_template` yields
  sigmoid-like probabilities through the same head. The ranking is identical either way.
- **When comparing engines, diff `ranking` first — it must match exactly** — then compare
  score *gaps*, never absolute values. Raw reranker scores are not standardised across
  engines or prompts.
- sentence-transformers logs `Default prompt name is set to 'query'`: the repo ships a
  prompt template applied to every pair. Overriding it changes scores.
