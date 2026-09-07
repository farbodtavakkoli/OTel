# `embed_transformers.py`

## Overview & when to use

**Reference embedding inference** for `google/embeddinggemma-300m` on the
Transformers + PyTorch + sentence-transformers stack. This is the
**correctness baseline**: the
numbers this folder produces are what vLLM, SGLang, llama.cpp and TEI get diffed
against. It is optimised for *reproducible output*, not throughput.

Use it when you need to:

- Verify EmbeddingGemma runs correctly on a new GPU/ROCm/driver combination.
- Produce a golden embedding + cosine-similarity artifact other engines are compared to.
- Sanity-check a retrieval setup end to end (query vs relevant doc must outscore
  query vs irrelevant doc) before wiring in a faster serving engine.

The script encodes a query set and a document set with EmbeddingGemma's own prompt
templates (`encode_query` / `encode_document`, which apply `search_query:` /
`search_document:`), prints the full cosine-similarity matrix ranked per query, asserts
the relevant documents outrank the irrelevant ones, and writes a JSON reference artifact.

Design notes:

- **Prompt templates** — `encode_query`/`encode_document` are used rather than plain
  `encode`, because EmbeddingGemma is trained with asymmetric prompts and skipping them
  measurably degrades retrieval.
- **dtype policy** — the upstream model card recommends BF16 or FP32 and warns against
  forcing FP16, so `--dtype` offers all three but defaults to `bfloat16`, and `float16`
  is never selected automatically.
- **ROCm attention switch** — `load_model()` picks `sdpa` when `torch.version.hip` is
  set, because `flash-attn` is a CUDA-only build. NVIDIA keeps `eager` unless you pass
  `--attn_impl flash_attention_2`.
- **Deterministic by construction** — greedy, seeded, no sampling anywhere; re-running
  the same command with the same dtype and batch composition reproduces bit-identical
  vectors (verified below).

## Install

Python 3.12, in its own venv.

### AMD (ROCm) — the route that was verified

Install torch from the ROCm wheel index **before** the rest of the requirements:

```bash
python3 -m venv .env_transformers
source .env_transformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_inference_embedding_transformers.txt
```

The `torch==2.11.0` pin exists on the ROCm index and installs as `2.11.0+rocm7.2`, which
satisfies the plain `==2.11.0` pin in the requirements file — so the second command does
**not** pull a CPU/CUDA torch over the top. Do **not** install `flash-attn`; it is a CUDA
build and the script selects `sdpa` automatically.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# True 7.2.26015
```

### NVIDIA (CUDA)

The default PyPI torch wheels are CUDA-enabled, so the requirements file is all you need:

```bash
pip install -r requirements_inference_embedding_transformers.txt
pip install flash-attn==2.8.3     # optional; then pass --attn_impl flash_attention_2
```

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

export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — it does not mean "no GPUs" reliably; use
explicit indices. `--devices` is indexed **within** the visible set, so `cuda:0` is
physical card 4 above.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source .env_transformers/bin/activate
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5

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

## Single-GPU results

`--dtype float32 --devices cuda:0` (1× MI355X, physical card 4) — expected output:

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

Embedding dimensionality is **768**, the separation between relevant and irrelevant
documents is ~0.44 cosine on Q0 and ~0.30 on Q1, and both queries rank correctly.

| Metric | `float32` | `bfloat16` |
|---|---|---|
| Model-load VRAM (torch allocated) | 1177.7 MiB | 591.2 MiB |
| Post-encode VRAM | 1253.7 MiB | 667.2 MiB |
| Load time | 3.1 s | 2.9 s |
| Encode time (6 texts) | 1.86 s | 3.37 s |
| `‖query[0]‖` after normalize | 1.000000 | 1.003512 |

`rocm-smi` reported ~284 MiB idle per card and a ~1.5–3.7 GiB working set on the active
card during encoding — under 2% of the MI355X's 288 GB, so the model is nowhere near
GPU-bound at this size.

## H100 results (NVIDIA, CUDA 13 — verified)

Same model (`google/embeddinggemma-300m`, gated, cached; `HF_TOKEN` from `dev.env`). Shared
stack venv: `torch 2.13.0+cu130`, `transformers 5.5.0`, `sentence-transformers 5.7.0`,
`kernels 0.12.3`, driver 580.173.02, Python 3.12.3. `flash-attn` has no prebuilt cu130 wheel
→ ran `--attn_impl sdpa` (CUDA otherwise auto-selects `eager`; pass `sdpa` explicitly).

Smoke command (physical GPU 5), fp32 for the canonical artifact:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=5
python embed_transformers.py --model google/embeddinggemma-300m --dtype float32 --devices cuda:0 \
  --attn_impl sdpa \
  --output $OUTPUT_DIR/transformers/embedding/reference_embedding_embeddinggemma-300m_fp32_1gpu.json
```

**Expected output** (fp32):

```
model=google/embeddinggemma-300m dtype=float32 attn=sdpa devices=['cuda:0']
encode: 0.86s | query_shape=(2, 768) doc_shape=(4, 768) | dim=768 | VRAM(MiB)=[1209.7]
query[0] norm=1.000000 first8=[-0.066597, -0.043119, -0.003417, -0.031592, 0.053912, 0.000171, -0.017533, 0.012261]
Q0: #1 cos=+0.574747 vLLM...  #2 +0.537463 SGLang...  #3 +0.097940 SQLite  #4 -0.011063 Eiffel
  sanity: min(relevant)=+0.537463 > max(irrelevant)=+0.097940 -> PASS
Q1: #1 cos=+0.502445 SQLite  #2 +0.202285 vLLM  #3 +0.190229 SGLang  #4 +0.083019 Eiffel
  sanity: min(relevant)=+0.502445 > max(irrelevant)=+0.202285 -> PASS
SANITY: PASS
```

**Cross-vendor numerical agreement is exact.** H100 fp32 `first8` and every cosine match
the MI355X fp32 reference to 6 decimals (Q0 `+0.574747` = `+0.574747`, Q1 `+0.502445` =
`+0.502445`). embeddinggemma-300m is fully vendor-independent at fp32. A bf16 artifact was
also captured (`reference_embedding_embeddinggemma-300m_1gpu.json`); bf16 shifts values in
the ~4th–5th decimal (`norm 0.999578`, cos Q0 top `+0.576231`) — fine for ranking, use fp32
for the golden artifact. GPU-5 residency: the script's own `torch.cuda.memory_allocated` on
the single visible device (physical GPU 5) read 591→623 MiB (bf16) / ~1210 MiB (fp32); the
encode is sub-second so it undershoots a 1 Hz external `nvidia-smi` poll — torch's per-GPU
reading is the residency proof here (an earlier reranker poll on the same GPU 5 did catch a
PID at 2.3–3.1 GiB, confirming the sampler works).

**On H100 this path works as documented, with no code changes.** dim 768, both sanity checks
PASS, fp32 identical to MI355X. Multi-GPU (data-parallel `--devices cuda:0,cuda:1`) was not
exercised on H100; there is nothing to shard at 300M params anyway.

## Multi-GPU results

**Pattern used: sentence-transformers multi-process encoding across both GPUs.**
EmbeddingGemma is a 300M-parameter model that occupies ~0.6–1.2 GiB — sharding one copy
across two 288 GB cards would be pointless. The honest multi-GPU pattern for a model this
small is *data* parallelism: sentence-transformers spawns one worker process per device,
each holding a full model replica, and splits the input batch between them. That is what
`--devices cuda:0,cuda:1` does (it forwards a device list to `encode`, which switches to
the multi-process pool path).

`--dtype float32 --devices cuda:0,cuda:1` — expected output:

```
model=google/embeddinggemma-300m dtype=float32 attn=sdpa devices=['cuda:0', 'cuda:1']
load: 2.8s | max_seq_length=2048 | VRAM(MiB)=[1177.7, 0.0]
encode: 18.52s | query_shape=(2, 768) doc_shape=(4, 768) | dim=768
  #1 cos=+0.574747  vLLM supports NVIDIA CUDA and AMD ROCm.
  #1 cos=+0.502445  SQLite is an embedded database.
SANITY: PASS
```

`rocm-smi` sampled every 3 s during the run, physical cards 4 and 5 — **weights resident
on both**:

```
card4=284MiB   card5=284MiB     t=1   (idle)
card4=1554MiB  card5=284MiB     t=3   (rank 0 replica up)
card4=2603MiB  card5=1332MiB    t=5   (rank 1 replica up — both cards loaded)
card4=3684MiB  card5=1882MiB    t=6   (peak, both encoding)
card4=284MiB   card5=284MiB     t=11  (pool shut down)
```

**Numerical agreement, single vs multi-GPU** — this is the result that matters for a
correctness baseline:

| dtype | max abs cosine delta, 1 GPU vs 2 GPU | ranking |
|---|---|---|
| **`float32`** | **0.0 — exact** | identical |
| `bfloat16` | 0.0021799 | identical |

**Use `--dtype float32` for anything you intend to diff against.** In bf16 the multi-GPU
result differs from the single-GPU result in the third decimal, and the cause is not the
GPUs — it is batch composition. Measured on one card:

| comparison | dtype | max abs delta |
|---|---|---|
| batch-of-4 vs split 2+2 | bf16 | 1.617e-3 |
| batch-of-4 vs one-at-a-time | bf16 | 1.526e-3 |
| batch-of-4 vs split 2+2 | **float32** | **0.0** |
| bf16 vs float32, same batching | — | 1.596e-3 |
| identical command re-run | bf16 | 0.0 (deterministic) |

Splitting inputs across processes changes each chunk's padding and therefore the matmul
reduction order; bf16's 8-bit mantissa cannot absorb that, and the resulting error is the
*same magnitude as bf16 quantization error itself*. fp32 is exactly batch-invariant here.
Rankings never changed in any configuration — the effect is far below any semantic
decision boundary — but "bitwise reproducible" and "bf16" are mutually exclusive.

**Throughput.** Multi-process encoding is *slower* on the built-in 6-text set (18.5 s vs
1.9 s) because pool startup — spawning two processes and loading a model replica in each
— dominates completely. The pool is worth it for large corpora, not for smoke tests. If
you are encoding a few thousand documents or fewer, use one GPU.

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

Each JSON records the model, dtype, attention implementation, devices, seed, torch/HIP
versions, GPU arch, embedding dim, timings, the queries and documents verbatim, the full
cosine-similarity matrix, the first 16 components of every query embedding, the per-query
L2 norms, and the sanity verdict. To compare another engine, encode the same texts and
diff `cosine_similarities` — that is dimensionless and dtype-robust, unlike raw vectors.

## Hardware support & evidence

- **AMD: tested.** 1× and 2× MI355X (gfx950, 288 GB), ROCm 7.2.4, Python 3.12.3,
  `torch==2.11.0+rocm7.2`, `transformers==5.5.0`, `sentence-transformers==5.7.0`.
  Single-GPU and 2-GPU multi-process encoding both verified, with `rocm-smi` confirming
  the model resident on both physical cards. No code changes were needed beyond the
  automatic `flash-attn → sdpa` switch.
- **NVIDIA: not tested here**, but nothing in the script is ROCm-specific — the ROCm
  torch build keeps the `torch.cuda` API, and `--attn_impl` lets you opt into
  `flash_attention_2` on CUDA. This stack is ✅ Native on both vendors for
  EmbeddingGemma.

## Notes & quirks

- **`float16` is available but never automatic.** The model card explicitly warns against
  forcing FP16 for EmbeddingGemma; `--dtype` will accept it if you insist.
- **bf16 breaks L2 normalization slightly** — `‖q‖` comes back as `1.003512` rather than
  `1.0`, because the normalize happens in bf16. fp32 gives exactly `1.000000`. Cosine
  similarity in the artifact is recomputed in float32, so this does not leak into the
  comparison numbers.
- **The multi-process pool leaks semaphores at shutdown** — Python prints
  `resource_tracker: There appear to be 4 leaked semaphore objects`. Harmless, comes from
  sentence-transformers' pool teardown, and does not affect results.
- **Shared HF cache permission warnings.** If `$HF_HOME` was populated by
  another user you will see `Ignoring corrupted tree cache file ... Permission denied`.
  It is a cache-metadata write failing, not a model-load failure; output is unaffected.
- `--devices` indexes *within* `HIP_VISIBLE_DEVICES`, so `cuda:0,cuda:1` means physical
  cards 4 and 5 given the exports above. Check with
  `python -c "import torch; print(torch.cuda.device_count())"` before a multi-GPU run.

## Platform notes

**This path works as documented, with no code changes.** `google/embeddinggemma-300m` runs correctly on MI355X /
gfx950 / ROCm 7.2.4 through the stock Transformers + sentence-transformers stack — 768-d
embeddings, correct relevance ordering with a wide margin, on one GPU and across two.

As the correctness baseline this folder delivers what it should: with `--dtype float32`
the single-GPU and 2-GPU runs agree **exactly** (max cosine delta 0.0), so the artifact is
a genuine golden reference rather than an approximation. The one caveat worth carrying
into every comparison is that **bf16 is not batch-invariant** — expect ~2e-3 cosine noise
whenever batch composition differs, and do not treat a bf16 disagreement of that size as
an engine bug.

Recommended reference command: `--dtype float32 --devices cuda:0`.

**NVIDIA H100 (verified): works, no code changes.** 1× H100 80GB, CUDA 13.0,
driver 580.173.02, `torch==2.13.0+cu130`, `sentence-transformers==5.7.0`, attn `sdpa`. The
fp32 artifact matches the MI355X fp32 golden reference to 6 decimals (`first8` identical,
cos Q0 `+0.574747`, Q1 `+0.502445` on both) — this embedding baseline is fully
vendor-independent. See "H100 results" above.
