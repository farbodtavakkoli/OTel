# `rerank_mlx.py`

## Overview & when to use

**Qwen3-Reranker-0.6B scoring on Apple silicon** through `mlx-lm`, running in-process.

This leaf scores the same built-in query/document set as [`inference/transformers/reranker`](../../transformers/reranker/README.md), prints the ranking, verifies relevant documents outrank irrelevant ones, and writes the JSON reference artifact.

> **Format note:** The default model `mlx-community/Qwen3-Reranker-0.6B-4bit` is an ungated 4-bit (group size 64) conversion of `Qwen/Qwen3-Reranker-0.6B`.

> **Architecture note:** Qwen3-Reranker is a decoder-style yes/no reranker. The script formats an `(Instruct, Query, Document)` triple into a system/user prompt (matching the `transformers` pair chat template), runs a forward pass, and calculates the score as `logit(yes) − logit(no)` at the last position. `--activation sigmoid` transforms this into a `P(yes)` score between `[0, 1]`.

## Install

See [`../README.md`](../README.md) — one shared venv at `inference/mlx/` serves all three leaves.

## Environment & secrets

`dev.env` is symlinked to the repo-root and loaded automatically. Nothing here is gated.

```bash
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # inference artifacts
```

## Run

Run with the built-in query/document set and write the reference artifact:

```bash
source ../.env_mlx/bin/activate
python rerank_mlx.py --output $OUTPUT_DIR/inference_reranker_mlx/reference_rerank_4bit.json
```

Score a custom dataset and output probabilities:

```bash
python rerank_mlx.py --query "..." --documents_file my_docs.json --activation sigmoid
```

## Expected output (Apple M4, 24 GB)

```text
model=mlx-community/Qwen3-Reranker-0.6B-4bit device=gpu (Metal, unified memory) chip=Apple M4
load: 0.8s | max_length=1024 | peak(MiB)=319.7
score: 0.181s | 4 pairs | peak(MiB)=485.4

Q: Which inference engines support AMD ROCm?
  #1 score=+6.562500  vLLM supports AMD ROCm.
  #2 score=+3.125000  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=-10.546875  PostgreSQL is a relational database.
  #4 score=-11.359375  The Eiffel Tower is located in Paris, France.

sanity: min(relevant)=+3.125000 > max(irrelevant)=-10.546875 -> PASS
sanity: top-2 == relevant set -> PASS

SANITY: PASS
```

**Compared to the fp32 transformers reference** (`+7.98, +5.02, −9.42, −9.72`), the rankings and sign are identical. The 4-bit weights and bf16 logits shift absolute score magnitudes slightly, but the correct ordering is preserved.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `mlx-community/Qwen3-Reranker-0.6B-4bit` | MLX model id or path |
| `--query` | built-in | Query string |
| `--documents_file` | `None` | JSON/JSONL/text documents (omit for built-in set) |
| `--instruct` | built-in | Task instruction rendered into the prompt |
| `--max_len` | `1024` | Max sequence length (document is truncated to fit) |
| `--activation` | `default` | `default`/`none` (raw logit diff); `sigmoid` (`P(yes)`) |
| `--output` | `None` | Path for the JSON reference artifact |

## Output

Prints system info, timing, the ranked documents, and two sanity verdicts. 
Exit code is 0 on PASS, 1 on FAIL.

`--output` writes the reference artifact using the baseline's schema (`scores`, `ranking`, `sanity_check_passed`, plus MLX keys):

```
$OUTPUT_DIR/inference_reranker_mlx/reference_rerank_4bit.json
```

Diff the `ranking` array first, followed by `scores` (using a tolerance threshold due to quantization).

## Hardware support

- **Apple M4, 24 GB (macOS 26.6, Metal)**: Verified. Requires 0.48 GB peak memory and ~45 ms per pair (after warm-up). A top-20 rerank takes under a second.

## Notes & quirks

- **No batching:** Pairs are scored one at a time because each document results in a different prompt length, and `mlx_lm` models require unpadded batches. At 45 ms per pair, batch padding overhead isn't beneficial for typical reranking depths.
- **Quantized logits:** The last-position logits are evaluated in bf16, meaning the `logit(yes) - logit(no)` differences fall on a bf16 grid (e.g. `+6.5625`). Use `--activation sigmoid` for continuous `[0, 1]` scores.
- **Manual templating:** The prompt is rendered directly by the script rather than through a chat template, bypassing the empty-checkpoint template issue found in the transformers baseline.

## Recommended reference command

```bash
python rerank_mlx.py --output $OUTPUT_DIR/inference_reranker_mlx/reference_rerank_4bit.json
```
