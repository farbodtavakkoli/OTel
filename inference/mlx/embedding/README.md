# `embed_mlx.py`

## Overview & when to use

**EmbeddingGemma-300M embedding inference on Apple silicon** through `mlx-embeddings`, running in-process. 

This leaf encodes the same built-in query and document sets as [`inference/transformers/embedding`](../../transformers/embedding/README.md) using EmbeddingGemma's prompt templates, prints the ranked cosine-similarity matrix, verifies relevancy ordering, and writes the JSON reference artifact for diffing against the baseline.

> **Format note:** The default model `mlx-community/embeddinggemma-300m-bf16` is an ungated bf16 conversion of `google/embeddinggemma-300m`. Smaller `-4bit` and `-8bit` footprint variants are also available on the Hub.

## Install

See [`../README.md`](../README.md) — one shared venv at `inference/mlx/` serves all three leaves.

## Environment & secrets

`dev.env` is symlinked to the repo-root and loaded automatically. `HF_TOKEN` is only needed if passing the gated `--model google/embeddinggemma-300m`.

```bash
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # inference artifacts
export HF_HUB_OFFLINE=1                # Optional for offline runs
```

## Run

Run with built-in sets and write the reference artifact:

```bash
source ../.env_mlx/bin/activate
python embed_mlx.py --output $OUTPUT_DIR/inference_embedding_mlx/reference_embedding_bf16.json
```

Encode your own corpus, Matryoshka-truncated to 256 dimensions:

```bash
python embed_mlx.py --queries_file my_queries.json --documents_file my_docs.json \
  --truncate_dim 256 --batch_size 128
```

The script exits non-zero if the relevance sanity check fails, enabling its use as a CI gate.

## Expected output (Apple M4, 24 GB)

```text
model=mlx-community/embeddinggemma-300m-bf16 device=gpu (Metal, unified memory) chip=Apple M4
load: 1.3s | max_seq_length=2048 | peak(MiB)=586.7
encode: 0.050s | query_shape=(2, 768) doc_shape=(4, 768) | dim=768 | peak(MiB)=648.1
query[0] norm=1.000000 first8=[-0.064279, -0.037816, -0.001914, -0.033134, 0.055433, -0.001491, -0.017054, 0.011187]

Q0: What GPU runtimes support ROCm?
  #1 cos=+0.576914  vLLM supports NVIDIA CUDA and AMD ROCm.
  ...
  sanity: min(relevant)=+0.538104 > max(irrelevant)=+0.101806 -> PASS

SANITY: PASS
```

**Compared to the fp32 transformers reference**, cosine values agree within **4e-3** (due to bf16 batch-composition noise), and document rankings are identical.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `mlx-community/embeddinggemma-300m-bf16` | MLX model id or path |
| `--queries_file` | `None` | JSON/JSONL/text queries (omit for built-in set) |
| `--documents_file` | `None` | JSON/JSONL/text documents (omit for built-in set) |
| `--batch_size` | `32` | Encode batch size |
| `--max_seq_length` | `2048` | Tokenizer truncation length |
| `--truncate_dim` | `None` | Matryoshka output dim (768/512/256/128) |
| `--normalize` | on | L2-normalize embeddings |
| `--output` | `None` | Path for the JSON reference artifact |

## Output

Prints system info, timing, dimensions, and the ranked cosine-similarity matrix. 

`--output` writes the JSON artifact mapping the MLX specific keys (detailed in [`../README.md`](../README.md)):

```
$OUTPUT_DIR/inference_embedding_mlx/reference_embedding_bf16.json
```

Diff `cosine_similarities` when comparing across stacks, as it is dimensionless and dtype-robust.

## Hardware support

- **Apple M4, 24 GB (macOS 26.6, Metal)**: Verified. Peak memory is 0.65 GB, supported on any M-series Mac.

## Notes & quirks

- **`mlx-embeddings` kwargs bug**: `mlx_embeddings.generate()` passes `input_ids=` but the Gemma3 text model expects positional `inputs`, causing a TypeError. `embed_mlx.py` works around this by tokenizing and calling the model directly.
- **bf16 cosine shift**: bf16 shifts cosine similarities in the 3rd decimal place relative to the fp32 baseline, but rankings remain unaffected.
- **`OTel-Embedding-300M` loading issue**: The upstream Hub repository is missing `2_Dense/` and `3_Dense/` weights, preventing successful load in both MLX and `sentence-transformers`.

## Recommended reference command

```bash
python embed_mlx.py --output $OUTPUT_DIR/inference_embedding_mlx/reference_embedding_bf16.json
```
