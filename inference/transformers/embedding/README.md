# `embed_transformers.py`

## Overview & when to use

**Reference embedding inference** for `google/embeddinggemma-300m` on the
Transformers + PyTorch + sentence-transformers stack. This is the **correctness baseline**:
the numbers this folder produces are what vLLM, SGLang, llama.cpp and TEI get diffed against.

The script encodes a query set and a document set with EmbeddingGemma's own prompt
templates (`encode_query` / `encode_document`, which apply `search_query:` /
`search_document:`), prints the full cosine-similarity matrix ranked per query, asserts
the relevant documents outrank the irrelevant ones, and writes a JSON reference artifact.

`load_model()` picks `sdpa` when `torch.version.hip` is set, because `flash-attn` is a
CUDA-only build; on NVIDIA the default is `eager` unless you pass `--attn_impl`.

## Install

Python 3.12, in its own venv.

One venv at `inference/transformers/` serves all three leaves.

### AMD / ROCm

Install torch from the ROCm wheel index **before** the rest of the requirements:

```bash
cd inference/transformers
python3 -m venv .env_transformers
source .env_transformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

The `torch==2.11.0` pin exists on the ROCm index and installs as `2.11.0+rocm7.2`, which
satisfies the plain `==2.11.0` pin in the requirements file — so the second command does
**not** pull a CPU/CUDA torch over the top. Do **not** install `flash-attn`; it is a CUDA
build and the script selects `sdpa` automatically.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# True 7.2.x
```

### NVIDIA / CUDA 13

The default PyPI torch wheels are CUDA-enabled:

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0        # CUDA 13 build from PyPI; no --index-url needed
pip install -r requirements.txt  # the torch pin is already satisfied
pip install flash-attn==2.8.3    # optional; then pass --attn_impl flash_attention_2
```

The PyPI `torch==2.11.0` wheel is a CUDA 13 build, so the `requirements.txt` pin holds on
CUDA as it does on ROCm; install torch first so the requirements install does not re-resolve
it. Add `--index-url https://download.pytorch.org/whl/cu130` if you want the `+cu130` local
version tag. flash-attn has no prebuilt cu130 wheel; on CUDA 13 **pass `--attn_impl sdpa`
explicitly**, because the script otherwise auto-selects `eager` when `torch.version.hip` is
`None`.

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env` and supplies the Hub token for the
gated EmbeddingGemma repo:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`. Point the model cache at a large volume, choose an
output directory, and pin the two GPUs you own:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # inference artifacts

export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # the GPUs you want to use
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — it does not mean "no GPUs" reliably; use
explicit indices. `--devices` is indexed **within** the visible set, so `cuda:0` is the
first card listed above.

Offline runs (cache already populated): `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source .env_transformers/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

python embed_transformers.py --dtype float32 --devices cuda:0 \
  --output $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_1gpu.json
```

Multi-GPU (sentence-transformers multi-process encoding across both cards):

```bash
python embed_transformers.py --dtype float32 --devices cuda:0,cuda:1 \
  --output $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_2gpu.json
```

Your own corpus, Matryoshka-truncated to 256 dims:

```bash
python embed_transformers.py --queries_file my_queries.json --documents_file my_docs.json \
  --truncate_dim 256 --batch_size 128
```

The script exits non-zero if the relevance sanity check fails, so it works as a CI gate.

## Expected output

`--dtype float32 --devices cuda:0`:

```
model=google/embeddinggemma-300m dtype=float32 attn=sdpa devices=['cuda:0']
load: 3.1s | max_seq_length=2048 | VRAM(MiB)=[1177.7, 0.0]
encode: 1.86s | query_shape=(2, 768) doc_shape=(4, 768) | dim=768 | VRAM(MiB)=[1253.7, 0.0]
query[0] norm=1.000000 first8=[-0.066597, -0.043119, -0.003417, -0.031592, 0.053912, 0.000171, -0.017533, 0.012261]

Q0: What GPU runtimes support ROCm?
  #1 cos=+0.574747  vLLM supports NVIDIA CUDA and AMD ROCm.
  #2 cos=+0.537463  SGLang provides a ROCm build for AMD Instinct accelerators.
  #3 cos=+0.097941  SQLite is an embedded database.
  #4 cos=-0.011063  The Eiffel Tower is located in Paris, France.
  sanity: min(relevant)=+0.537463 > max(irrelevant)=+0.097941 -> PASS

Q1: Which database is embedded and serverless?
  #1 cos=+0.502445  SQLite is an embedded database.
  #2 cos=+0.202286  vLLM supports NVIDIA CUDA and AMD ROCm.
  #3 cos=+0.190229  SGLang provides a ROCm build for AMD Instinct accelerators.
  #4 cos=+0.083020  The Eiffel Tower is located in Paris, France.
  sanity: min(relevant)=+0.502445 > max(irrelevant)=+0.202286 -> PASS

SANITY: PASS
```

Embedding dimensionality is **768** and both queries must rank correctly (`SANITY: PASS`).

**Use `--dtype float32` for the golden artifact.** bf16 shifts values in the 4th–5th decimal
and is **not batch-invariant** — splitting the same inputs across a different batch
composition moves cosines by ~2e-3, because padding changes the matmul reduction order.
Rankings are unaffected, but a bf16 artifact is not bitwise reproducible.

## Multi-GPU

`--devices cuda:0,cuda:1` switches sentence-transformers to multi-process encoding: one
worker per device, each holding a full model replica, with the input batch split between
them (data parallelism — this model is too small to shard). Pool startup dominates on small
text sets, so use one GPU unless you are encoding a large corpus.

With `--dtype float32` the single-GPU and 2-GPU cosines agree exactly; in bf16 expect the
~2e-3 batch-composition noise described above.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `google/embeddinggemma-300m` | Embedding model id or path |
| `--queries_file` | `None` | JSON list / JSONL / plain-text file of queries; omit for the built-in set |
| `--documents_file` | `None` | JSON list / JSONL / plain-text file of documents; omit for the built-in set |
| `--dtype` | `bfloat16` | `bfloat16`, `float32`, or `float16`. Use `float32` for reference artifacts |
| `--devices` | `cuda:0` | Comma-separated devices; 2+ switches to multi-process encoding |
| `--batch_size` | `32` | Encode batch size |
| `--max_seq_length` | `None` | Override the model's max sequence length (default 2048) |
| `--truncate_dim` | `None` | Matryoshka output dim (768/512/256/128); omit for full 768 |
| `--normalize` / `--no_normalize` | on | L2-normalize embeddings |
| `--attn_impl` | `None` | Attention implementation; auto-selects `sdpa` on ROCm |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

Printed to stdout: model/dtype/attention/device line, load time and VRAM, encode time,
embedding shapes and dimensionality, the first 8 components and L2 norm of `query[0]`,
the full ranked cosine-similarity matrix, and a `PASS`/`FAIL` sanity verdict. Exit code
is 0 on PASS, 1 on FAIL.

`--output` writes the **reference artifact** other stacks are diffed against:

```
$OUTPUT_DIR/inference_embedding_transformers/
  reference_embedding_fp32_1gpu.json    <- canonical: exactly reproducible
  reference_embedding_fp32_2gpu.json    <- byte-identical to the above
  reference_embedding_1gpu.json         <- bf16 variant
  reference_embedding_2gpu.json         <- bf16 variant
```

This is the file [`../../ollama/embedding`](../../ollama/embedding) and the other serving
stacks diff against.

Each JSON records the model, dtype, attention implementation, devices, seed, torch/HIP
versions, GPU arch, embedding dim, timings, the queries and documents verbatim, the full
cosine-similarity matrix, the first 16 components of every query embedding, the per-query
L2 norms, and the sanity verdict. To compare another engine, encode the same texts and
diff `cosine_similarities` — that is dimensionless and dtype-robust, unlike raw vectors.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2)**: works, 1 and 2 GPUs. No code changes beyond the
  automatic `flash-attn → sdpa` switch.
- **NVIDIA H100 80GB (CUDA 13)**: works, single-GPU, with `torch==2.13.0+cu130` and
  `--attn_impl sdpa`. You can also opt into `--attn_impl flash_attention_2` where a
  flash-attn wheel exists.

## Notes & quirks

- **`float16` is available but never automatic.** The model card explicitly warns against
  forcing FP16 for EmbeddingGemma; `--dtype` will accept it if you insist.
- **bf16 breaks L2 normalization slightly** — `‖q‖` comes back a little off `1.0`, because
  the normalize happens in bf16. fp32 gives exactly `1.000000`. Cosine similarity in the
  artifact is recomputed in float32, so this does not leak into the comparison numbers.
- **The multi-process pool leaks semaphores at shutdown** — Python prints
  `resource_tracker: There appear to be 4 leaked semaphore objects`. Harmless, comes from
  sentence-transformers' pool teardown, and does not affect results.
- **Shared HF cache permission warnings.** If `$HF_HOME` was populated by
  another user you will see `Ignoring corrupted tree cache file ... Permission denied`.
  It is a cache-metadata write failing, not a model-load failure; output is unaffected.
- `--devices` indexes *within* `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, not physical
  card numbers. Check with
  `python -c "import torch; print(torch.cuda.device_count())"` before a multi-GPU run.

## Recommended reference command

`--dtype float32 --devices cuda:0`
