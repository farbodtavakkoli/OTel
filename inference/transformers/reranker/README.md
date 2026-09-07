# `rerank_transformers.py`

## Overview & when to use

**Reference reranker inference** for `Qwen/Qwen3-Reranker-0.6B` via the
sentence-transformers `CrossEncoder` API. This is the
**correctness baseline** for the
reranker workload: the scores this folder produces are what vLLM's `/v1/rerank`, SGLang's
reranker path and llama.cpp's `--reranking` server get diffed against. Correctness and
reproducibility come first; throughput is not the point.

Use it when you need to:

- Verify the Qwen3 reranker runs correctly on a new GPU/ROCm/driver combination.
- Produce a golden score artifact other engines are compared to.
- Sanity-check that a retrieval reranking stage actually orders relevant documents above
  irrelevant ones before swapping in a faster serving engine.

The script scores every `(query, document)` pair, prints them ranked, asserts that the
relevant documents outscore the irrelevant ones *and* that they occupy the top-k slots,
and writes a JSON reference artifact.

Design notes:

- **`CrossEncoder`, not raw yes/no logits.** Qwen3-Reranker is natively a decoder-only
  yes/no-token scorer, but the repo ships `modules.json` + `1_LogitScore/` so
  sentence-transformers 5.x loads it as a proper CrossEncoder and exposes a single scalar
  score per pair. That is the recommended high-level path and avoids
  hand-rolling the yes/no logit arithmetic.
- **Scores are raw logits, not probabilities.** They are unbounded and can be strongly
  negative; only *relative order* is meaningful. Pass `--activation sigmoid` if you need
  them squashed to (0,1).
- **ROCm attention switch** — `load_model()` picks `sdpa` when `torch.version.hip` is set,
  because `flash-attn` is a CUDA-only build.
- **Qwen pad-token quirk** — Qwen tokenizers have no dedicated pad token; the script sets
  `pad_token = eos_token` and syncs `config.pad_token_id`, matching what
  `train_reranker_standalone.py` does. Without it, batched scoring raises a
  `pad_token_id not set` error.

## Install

Python 3.12, in its own venv.

### AMD (ROCm) — the route that was verified

Install torch from the ROCm wheel index **before** the rest of the requirements:

```bash
python3 -m venv .env_transformers
source .env_transformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_inference_reranker_transformers.txt
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
pip install -r requirements_inference_reranker_transformers.txt
pip install flash-attn==2.8.3     # optional; then pass --attn_impl flash_attention_2
```

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

export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--devices` is indexed
**within** the visible set, so `cuda:0` is physical card 4 above.

## Run

Single GPU, built-in query/document set, writing the reference artifact:

```bash
source .env_transformers/bin/activate
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5

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

## Single-GPU results

`--dtype float32 --devices cuda:0` (1× MI355X, physical card 4) — expected output:

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

The ordering is correct and the **margin is enormous** — the worst relevant document
scores `+5.02` while the best irrelevant one scores `-9.42`, a gap of **14.4 logits**.
That is exactly the behaviour a reranker baseline should show: no ambiguity to explain
away when a faster engine disagrees.

| Metric | `float32` | `bfloat16` |
|---|---|---|
| Model-load VRAM (torch allocated) | 2272.7 MiB | 1136.4 MiB |
| Post-scoring VRAM | 2348.7 MiB | 1212.4 MiB |
| Load time | 2.5 s | 2.2 s |
| Score time (4 pairs) | 1.78 s | 3.60 s |
| Top score | `+7.980374` | `+8.187500` |

## H100 results (NVIDIA, CUDA 13 — verified)

Same model (`Qwen/Qwen3-Reranker-0.6B`, cached). Shared stack venv: `torch 2.13.0+cu130`,
`transformers 5.5.0`, `sentence-transformers 5.7.0`, `kernels 0.12.3`, driver 580.173.02,
Python 3.12.3. `flash-attn` has no prebuilt cu130 wheel → ran `--attn_impl sdpa`.

### One code fix was required (hardware-neutral): `--fix_pair_template`

Out of the box on H100, `model.predict()` raised:

```
ValueError: The chat template of Qwen/Qwen3-Reranker-0.6B cannot carry a 'query'/'document'
pair (... content does not reach the rendered prompt). ... set a chat template that renders
both roles ...
```

**Root cause (not a GPU issue):** the node's offline cache held a *bare* Qwen3-Reranker
checkpoint — it carries only the generic Qwen3 **generation** chat template (system/user/
assistant), with no packaged `chat_template.jinja`. sentence-transformers **5.7.0** added
`_verify_pair_roles_supported`, which maps a pair to one message per named role
(`query`/`document`) and refuses a template that renders neither. The MI355X run (same ST
version) did not hit this, because its cache carried the model's shipped pair
template; a bare cache would fail there identically. The `1_LogitScore` head IS present
(`modules: ['Transformer', 'LogitScore']`, `num_labels=2`), so scoring itself is fine — only
the prompt rendering was broken.

**Fix (added to `rerank_transformers.py`, default on):** `install_pair_chat_template()`
checks `input_formatter.pair_roles_failure({})` and, if the loaded template can't carry both
roles, installs a Query/Document yes-no instruct template via `model.processor.chat_template`.
Opt out with `--no_fix_pair_template` (reproduces the failure). A checkpoint that already
ships a pair template is left untouched.

Smoke command (physical GPU 5), fp32:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=5
python rerank_transformers.py --model Qwen/Qwen3-Reranker-0.6B --dtype float32 --devices cuda:0 \
  --attn_impl sdpa \
  --output $OUTPUT_DIR/transformers/reranker/reference_reranker_qwen3-reranker-0.6b_fp32_1gpu.json
```

**Expected output:**

```
reranker fix: installed Query/Document pair chat template (bare checkpoint)
model=Qwen/Qwen3-Reranker-0.6B dtype=float32 attn=sdpa devices=['cuda:0']
load: 28.0s | max_length=1024 | VRAM(MiB)=[2272.7]
score: 0.81s | 4 pairs | VRAM(MiB)=[2304.7]

Q: Which inference engines support AMD ROCm?
  #1 score=+0.999731  vLLM supports AMD ROCm.
  #2 score=+0.996374  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=+0.000034  PostgreSQL is a relational database.
  #4 score=+0.000020  The Eiffel Tower is located in Paris, France.
sanity: min(relevant)=+0.996374 > max(irrelevant)=+0.000034 -> PASS
sanity: top-2 == relevant set -> PASS
SANITY: PASS
```

GPU-5 residency, sampled by PID from inside the run:

```
[smi] GPU5 pid=<pid>  2378 MiB
[smi] GPU5 pid=<pid>  3106 MiB
```

### Scores are on a DIFFERENT scale than the MI355X reference — ranking is identical

The MI355X artifact reports **raw logits** (`+7.98 / +5.02 / -9.42 / -9.72`) from the model's
shipped publishing template. H100, using the yes-no instruct template above with the same
`LogitScore` head, reports **sigmoid-like probabilities** (`0.9997 / 0.9963 / 3.4e-5 /
2.0e-5`). The **ranking is identical** (`[0,1,2,3]`) and the relevant/irrelevant separation
is total (~1.0 vs ~0). Because raw reranker logits are not standardised across engines/
prompts anyway, downstream stacks must **diff `ranking` first (must match exactly)**, then
compare score *gaps*, never absolute values — as the "Output" section already prescribes.
The H100 artifact is the correctness reference for this cache; the MI355X logit artifact
remains valid for its own prompt template.

**On H100 this path works with the fix above.** It runs correctly after `--fix_pair_template`
(default on). Ranking perfect, both sanity checks PASS. The single change is a
sentence-transformers-5.7.0 + bare-cache requirement, not a Hopper/CUDA issue. Multi-GPU
data-parallel (`--devices cuda:0,cuda:1`) was not exercised on H100; a 0.6B model has
nothing to shard.

## Multi-GPU results

**Pattern used: sentence-transformers multi-process scoring across both GPUs.**
Qwen3-Reranker-0.6B is a 600M-parameter model occupying ~1.1–2.3 GiB. Sharding one copy
across two 288 GB MI355X cards would be pointless — there is nothing to split. The honest
multi-GPU pattern here is *data* parallelism: `CrossEncoder.predict` accepts a device
list and spawns one worker process per device, each with a full model replica, splitting
the candidate pairs between them. That is what `--devices cuda:0,cuda:1` does.

`--dtype float32 --devices cuda:0,cuda:1` — expected output:

```
model=Qwen/Qwen3-Reranker-0.6B dtype=float32 attn=sdpa devices=['cuda:0', 'cuda:1']
load: 2.4s | max_length=1024
score: 8.07s | 4 pairs
  #1 score=+7.980375  vLLM supports AMD ROCm.
  #2 score=+5.021244  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=-9.418510  PostgreSQL is a relational database.
  #4 score=-9.723064  The Eiffel Tower is located in Paris, France.
sanity: min(relevant)=+5.021244 > max(irrelevant)=-9.418510 -> PASS
sanity: top-2 == relevant set -> PASS
```

`rocm-smi` sampled every 3 s during the run, physical cards 4 and 5 — **weights resident
on both**:

```
card4=284MiB   card5=284MiB    t=2   (idle)
card4=3233MiB  card5=284MiB    t=3   (rank 0 replica up)
card4=6502MiB  card5=3554MiB   t=5   (rank 1 replica up — both cards loaded, peak)
card4=284MiB   card5=284MiB    t=6   (pool shut down)
```

**Numerical agreement, single vs multi-GPU:**

| comparison | max abs score delta | ranking |
|---|---|---|
| fp32, 1 GPU vs 2 GPU | **3.34e-5** | identical |
| fp32 vs bf16, 1 GPU | 0.2071 | identical |

The fp32 multi-GPU delta of `3.3e-5` on scores of magnitude ~8 is a **relative error of
4e-6** — ordinary fp32 reduction-order noise from the pairs being re-chunked across
processes, not a correctness problem. Note this is *not* exactly zero the way the
embedding model's fp32 result is; the reranker is a causal decoder and re-chunking
changes padding, which changes the attention reduction slightly even in fp32.

**bf16 costs ~0.2 logits** — visible in the third significant figure and much larger than
the multi-GPU effect. Rankings were identical in every configuration, but for an artifact
that other engines will be diffed against, use `--dtype float32`.

**Throughput.** Multi-process scoring is *slower* on the built-in 4-pair set (8.07 s vs
1.78 s) because pool startup dominates. The pool pays off when reranking thousands of
candidates, not on a smoke test. For a typical top-50 rerank, use one GPU.

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
| `--attn_impl` | `None` | Attention implementation; auto-selects `sdpa` on ROCm |
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
  reference_rerank_fp32_2gpu.json    <- agrees to 3.3e-5
  reference_rerank_bf16_1gpu.json    <- bf16 variant
```

Each JSON records the model, dtype, attention implementation, devices, seed, max_len,
activation, torch/HIP versions, GPU arch, timings, the query and documents verbatim, the
per-document scores, the ranking permutation, and the sanity verdict. To compare another
engine, score the same pairs and diff `ranking` first (must match exactly) then `scores`
(expect engine-dependent offsets — raw logits are not standardised across engines, so
compare *gaps* rather than absolute values).

## Hardware support & evidence

- **AMD: tested.** 1× and 2× MI355X (gfx950, 288 GB), ROCm 7.2.4, Python 3.12.3,
  `torch==2.11.0+rocm7.2`, `transformers==5.5.0`, `sentence-transformers==5.7.0`.
  Single-GPU and 2-GPU multi-process scoring both verified, with `rocm-smi` confirming the
  model resident on both physical cards. No code changes were needed beyond the automatic
  `flash-attn → sdpa` switch and the Qwen pad-token fix.
- **NVIDIA H100: tested** — 1× H100 80GB (Hopper cc9.0), CUDA 13.0, driver
  580.173.02, Python 3.12.3, `torch==2.13.0+cu130`, `transformers==5.5.0`,
  `sentence-transformers==5.7.0`, attn `sdpa`. Single-GPU verified, `nvidia-smi` confirming
  the model resident (~2.3–3.1 GiB, GPU 5). **Required one hardware-neutral code fix**
  (`--fix_pair_template`, default on) because that node's *bare* cache lacks the packaged
  pair chat template that ST 5.7.0 now demands — see "H100 results" above. Ranking identical
  to AMD; scores land on a different scale (probabilities vs logits) because of the template.
- **TEI cannot serve this reranker** — its `/rerank` targets sequence-classification
  cross-encoders, not decoder-only yes/no scoring — and SGLang's AMD target still needs
  validation. Transformers is the path with the fewest caveats on AMD.

## Notes & quirks

- **Scores are unbounded logits.** `+7.98` and `-9.72` are normal. Do not interpret them
  as probabilities or compare them numerically against another engine's scale; compare
  ordering and relative gaps. `--activation sigmoid` gives (0,1) values if a downstream
  consumer needs them.
- **bf16 quantizes the scores visibly** — bf16 has an 8-bit mantissa, so scores land on
  coarse steps (`+8.187500`, `+5.125000`, `-9.375000`). Fine for ranking, wrong for a
  golden artifact.
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
- `--devices` indexes *within* `HIP_VISIBLE_DEVICES`, so `cuda:0,cuda:1` means physical
  cards 4 and 5 given the exports above.

## Platform notes

**This path works as documented on AMD, with no code changes.** `Qwen/Qwen3-Reranker-0.6B` runs correctly on MI355X / gfx950 /
ROCm 7.2.4 through the stock Transformers + sentence-transformers `CrossEncoder` path, on
one GPU and across two. Relevant documents are ranked above irrelevant ones with a
14.4-logit margin, and the ranking is identical across every dtype and device
configuration tested.

As the correctness baseline this folder is solid: fp32 single-GPU and 2-GPU scores agree
to `3.3e-5` (4e-6 relative), so any engine that disagrees on *ordering*, or on score gaps
by more than ~0.01, is genuinely diverging rather than showing numerical noise. Use
`--dtype float32` for reference artifacts; bf16 shifts scores by ~0.2 logits, which is
harmless for ranking but too coarse to diff against.

Recommended reference command: `--dtype float32 --devices cuda:0`.
