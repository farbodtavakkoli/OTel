# `rerank_transformers.py`

## Overview & when to use

**Reference reranker inference** for `Qwen/Qwen3-Reranker-0.6B` via the
sentence-transformers `CrossEncoder` API. This is the **correctness baseline** for the
reranker workload: the scores this folder produces are what vLLM's `/v1/rerank`, SGLang's
reranker path and llama.cpp's `--reranking` server get diffed against.

The script scores every `(query, document)` pair, prints them ranked, asserts that the
relevant documents outscore the irrelevant ones *and* that they occupy the top-k slots,
and writes a JSON reference artifact.

- **Scores are raw logits, not probabilities.** They are unbounded and can be strongly
  negative; only *relative order* is meaningful. Pass `--activation sigmoid` if you need
  them squashed to (0,1).
- `load_model()` picks `sdpa` when `torch.version.hip` is set, because `flash-attn` is a
  CUDA-only build.
- **Qwen tokenizers have no pad token**; the script sets `pad_token = eos_token` and syncs
  `config.pad_token_id`. Without it, batched scoring raises `pad_token_id not set`.

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

`dev.env` is symlinked to the repo-root `dev.env` and supplies the Hub token:

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

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--devices` is indexed
**within** the visible set, so `cuda:0` is the first card listed above.

Offline runs (cache already populated): `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source .env_transformers/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

python rerank_transformers.py --dtype float32 --devices cuda:0 \
  --output $OUTPUT_DIR/inference_reranker_transformers/reference_rerank_fp32_1gpu.json
```

Multi-GPU (multi-process scoring across both cards):

```bash
python rerank_transformers.py --dtype float32 --devices cuda:0,cuda:1 \
  --output $OUTPUT_DIR/inference_reranker_transformers/reference_rerank_fp32_2gpu.json
```

Your own query and candidate set:

```bash
python rerank_transformers.py --query "How do I enable ROCm in PyTorch?" \
  --documents_file candidates.json --batch_size 64 --max_len 2048
```

The script exits non-zero if the relevance sanity check fails, so it works as a CI gate.

## Expected output

`--dtype float32 --devices cuda:0`:

```
model=Qwen/Qwen3-Reranker-0.6B dtype=float32 attn=sdpa devices=['cuda:0']
load: 2.5s | max_length=1024 | VRAM(MiB)=[2272.7, 0.0]
score: 1.78s | 4 pairs | VRAM(MiB)=[2348.7, 0.0]

Q: Which inference engines support AMD ROCm?
  #1 score=+7.980374  vLLM supports AMD ROCm.
  #2 score=+5.021234  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=-9.418544  PostgreSQL is a relational database.
  #4 score=-9.723047  The Eiffel Tower is located in Paris, France.

sanity: min(relevant)=+5.021234 > max(irrelevant)=-9.418544 -> PASS
sanity: top-2 == relevant set -> PASS
SANITY: PASS
```

Relevant documents must outscore irrelevant ones and occupy the top-k slots; both sanity
lines must read `PASS`.

## Required fix for a bare checkpoint cache: `--fix_pair_template`

This depends on the cached checkpoint and the sentence-transformers version, not the GPU.
With a bare cache, `model.predict()` raises:

```
ValueError: The chat template of Qwen/Qwen3-Reranker-0.6B cannot carry a 'query'/'document'
pair (... content does not reach the rendered prompt). ... set a chat template that renders
both roles ...
```

**Cause:** a *bare* Qwen3-Reranker checkpoint carries only the generic Qwen3 **generation**
chat template, with no packaged `chat_template.jinja`, and sentence-transformers **5.7.0**
refuses a template that cannot render the `query`/`document` roles. Scoring itself is fine —
only the prompt rendering is broken.

**Fix (in `rerank_transformers.py`, default on):** `install_pair_chat_template()` installs a
Query/Document yes-no instruct template via `model.processor.chat_template` when the loaded
one cannot carry both roles. Opt out with `--no_fix_pair_template` (reproduces the failure).
A checkpoint that already ships a pair template is left untouched.

When the fix engages, the run prints an extra first line and the scores land on a
**different scale**:

```
reranker fix: installed Query/Document pair chat template (bare checkpoint)
...
  #1 score=+0.999731  vLLM supports AMD ROCm.
  #2 score=+0.996374  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=+0.000034  PostgreSQL is a relational database.
  #4 score=+0.000020  The Eiffel Tower is located in Paris, France.
```

The shipped publishing template yields **raw logits**; the yes-no instruct template
installed by the fix yields **sigmoid-like probabilities** through the same `LogitScore`
head. The **ranking is identical** either way. Because raw reranker scores are not
standardised across engines or prompts, downstream stacks must **diff `ranking` first (it
must match exactly)**, then compare score *gaps*, never absolute values.

## Multi-GPU

`--devices cuda:0,cuda:1` switches `CrossEncoder.predict` to multi-process scoring: one
worker per device, each with a full model replica, splitting the candidate pairs between
them (data parallelism — this model is too small to shard). Pool startup dominates on a
handful of pairs, so use one GPU for a typical top-50 rerank.

Use `--dtype float32` for reference artifacts: bf16 shifts scores by ~0.2 logits, and even
in fp32 re-chunking pairs across processes moves scores slightly (~3e-5). Rankings are
unaffected in every configuration.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder model id or path |
| `--query` | built-in ROCm query | Query to rerank documents against |
| `--documents_file` | `None` | JSON list / JSONL / plain-text file of documents; omit for the built-in set |
| `--dtype` | `bfloat16` | `bfloat16`, `float32`, or `float16`. Use `float32` for reference artifacts |
| `--devices` | `cuda:0` | Comma-separated devices; 2+ switches to multi-process scoring |
| `--batch_size` | `32` | Scoring batch size |
| `--max_len` | `1024` | Max sequence length (query + document); model supports 32K |
| `--activation` | `default` | `default` keeps the model's LogitScore head; `sigmoid` squashes to (0,1); `none` = identity |
| `--fix_pair_template` / `--no_fix_pair_template` | on | Install a Query/Document pair chat template when the loaded one cannot carry both roles. **Required for a bare checkpoint cache on sentence-transformers 5.7.0** |
| `--attn_impl` | `None` | Attention implementation; auto-selects `sdpa` on ROCm. Pass `sdpa` explicitly on CUDA |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

Printed to stdout: model/dtype/attention/device line, load time and VRAM, scoring time,
the ranked `(score, document)` list, and two `PASS`/`FAIL` sanity checks — margin and
top-k membership. Exit code is 0 on PASS, 1 on FAIL.

`--output` writes the **reference artifact** other stacks are diffed against:

```
$OUTPUT_DIR/inference_reranker_transformers/
  reference_rerank_fp32_1gpu.json    <- canonical
  reference_rerank_fp32_2gpu.json    <- two-GPU run
  reference_rerank_bf16_1gpu.json    <- bf16 variant
```

Each JSON records the model, dtype, attention implementation, devices, seed, max_len,
activation, torch/HIP versions, GPU arch, timings, the query and documents verbatim, the
per-document scores, the ranking permutation, and the sanity verdict. To compare another
engine, score the same pairs and diff `ranking` first (must match exactly) then `scores`
(expect engine-dependent offsets — raw logits are not standardised across engines, so
compare *gaps* rather than absolute values).

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2)**: works, 1 and 2 GPUs. No code changes beyond the
  automatic `flash-attn → sdpa` switch and the Qwen pad-token fix.
- **NVIDIA H100 80GB (CUDA 13)**: works, single-GPU, with `torch==2.13.0+cu130` and
  `--attn_impl sdpa`.
- **A bare checkpoint cache needs `--fix_pair_template`** (default on) on
  sentence-transformers 5.7.0 — see above. Library requirement, not a vendor one.
- **TEI cannot serve this reranker** — its `/rerank` targets sequence-classification
  cross-encoders, not decoder-only yes/no scoring.

## Notes & quirks

- **Scores are unbounded logits.** Large positive and large negative values are normal. Do
  not interpret them as probabilities or compare them against another engine's scale;
  compare ordering and relative gaps. `--activation sigmoid` gives (0,1) values if a
  downstream consumer needs them.
- **bf16 quantizes the scores visibly** — coarse steps from the 8-bit mantissa. Fine for
  ranking, wrong for a golden artifact.
- **Qwen has no pad token.** Handled in `load_model()`; if you adapt this script,
  keep the `pad_token = eos_token` + `config.pad_token_id` lines or batched scoring fails.
- **A default prompt is applied.** sentence-transformers logs
  `Default prompt name is set to 'query'` — the repo ships a prompt template that is
  applied to every pair. That is correct behaviour for this model; overriding it changes
  scores.
- **The multi-process pool leaks semaphores at shutdown.** Harmless teardown warning.
- **Shared HF cache permission warnings.** If `$HF_HOME` was populated by
  another user you may see `Ignoring corrupted tree cache file ... Permission denied`.
  Cache-metadata only; results unaffected.
- `--devices` indexes *within* `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, not physical
  card numbers.

## Recommended reference command

`--dtype float32 --devices cuda:0`
