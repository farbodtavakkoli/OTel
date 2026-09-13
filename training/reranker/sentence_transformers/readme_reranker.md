# `train_reranker_standalone.py`

## Overview & when to use

Cross-encoder **reranker** fine-tuner built on sentence-transformers `CrossEncoder`.
Trains a reranker (default `Qwen/Qwen3-Reranker-0.6B`) to score (query, document) pairs,
using one positive and N hard negatives per query with `BinaryCrossEntropyLoss`, and
evaluates with a reranking nDCG@10 evaluator. Single-node multi-GPU via `accelerate`.

Use it as the second stage of a retrieval pipeline — a cross-encoder scores a (query,
document) pair jointly (`num_labels=1`), which is more accurate than embedding
similarity but too slow for first-stage retrieval, so it reranks a candidate set
produced by an embedding model (see `../../embedding/sentence_transformers/`).

Design notes:

- **Pair expansion** — `load_and_split` turns each query into `1 + n_neg` labeled pairs
  (positive = 1.0, negatives = 0.0) for `BinaryCrossEntropyLoss`.
- **Deterministic split** — a fixed `random.seed(42)` shuffle + slice gives every rank
  the same train/eval split.
- **Qwen pad token** — Qwen has no dedicated pad token, so the tokenizer's `eos` is used
  and synced to `model.config.pad_token_id`; otherwise the trainer raises a
  "pad_token_id not set" ValueError.
- **Evaluation** — `CrossEncoderRerankingEvaluator` at nDCG@10; a baseline eval runs on
  rank 0 before training (barriers keep ranks in sync) and `load_best_model_at_end`
  selects on `reranking_ndcg@10`.

## Install

Python 3.12, in its own venv — deps differ from the generative trainers:

```bash
python3.12 -m venv ~/.venv-rerank && source ~/.venv-rerank/bin/activate
pip install -r requirements_reranker.txt
```

Verify:

```bash
python -c "from sentence_transformers.cross_encoder import CrossEncoder; print('imports OK')"
```

### NVIDIA (CUDA)

The default `torch` wheels from PyPI ship with CUDA support, and the `torch==2.11.0` pin in
`requirements_reranker.txt` resolves to a native `+cu130` wheel there, so no `--index-url`
is needed. Install torch first, then the rest of the file:

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_sentence_transformers && source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 numpy               # -> 2.11.0+cu130, no --index-url
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install -r requirements_reranker.txt
python -c "import torch; print(torch.__version__)"   # re-verify: not clobbered
ln -sf ../../../dev.env dev.env                # HF_TOKEN for the Hub
```

`flash-attn` is optional — the script falls back to `attn_implementation="sdpa"` when it
cannot import it. No prebuilt wheel is published for torch 2.11/cu130/py3.12, so building
it needs the CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3 --no-build-isolation
```

### AMD (ROCm)

Install torch from the ROCm index **before** the rest of the requirements:

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_sentence_transformers && source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> 2.11.0+rocm7.2
pip install -r requirements_reranker.txt      # all pins install unchanged
ln -sf ../../../dev.env dev.env               # HF_TOKEN for the Hub
```

The `torch==2.11.0` pin exists on the ROCm 7.2 index (the rocm7.1 index also carries it —
match your installed ROCm). Skip `flash-attn` (CUDA-only build here); the script
auto-detects its absence and falls back to `attn_implementation="sdpa"`.


## Environment & secrets

Optional `dev.env` next to the script, loaded via `load_dotenv("dev.env")` — only needed
for gated models on the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`--hf_home` optionally points the HF model cache somewhere other than the default
`~/.cache/huggingface`.

## Data

`--data` is a JSONL, one row per query — an `anchor` (query), one `positive` (relevant
doc), and hard negatives `negative_1 … negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

Each row is expanded into `1 + n_neg` labeled pairs — the positive with label `1.0` and
each negative with `0.0` — as `{sentence_0=query, sentence_1=doc, label}`. Extra columns
(e.g. an `answer` field) are ignored.

Shipped sample: `OTel_reranker_sample_100.jsonl` (100 rows, includes the `answer`
column) — the default `--data`, there to smoke-test end to end. To train for real, point
`--data` at your own JSONL with the same columns.

## Run

Smoke test against the shipped sample (`accelerate launch` is fine at more than one
process; for a single GPU use `torchrun` — see *Launching* below):

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_reranker_standalone.py --test_mode
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_reranker_standalone.py --data /path/to/your_train.jsonl --out reranker_run1 \
  > train_reranker_standalone.log 2>&1 &

tail -f train_reranker_standalone.log
```

**What "working" looks like:** a baseline eval prints before training, then per-epoch
`{'loss': ...}` and reranking metrics; the best checkpoint (by `reranking_ndcg@10`) is
loaded at the end and saved to `<out>/final`.

### Launching

The script calls `dist.init_process_group` unconditionally, and
`accelerate launch --num_processes=1` does not set `RANK`/`WORLD_SIZE` — it dies with
"environment variable RANK expected, but not set". **Use `torchrun` for every world size**,
so the single- and multi-GPU commands have the same shape:

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
torchrun --nproc_per_node=1 --master_port 29648 train_reranker_standalone.py \
  --test_mode --epochs 3 --batch 16 --eval_frac 0.05 --max_len 512 --out <out>
```

`accelerate launch --num_processes=8` also works for multi-GPU, because it does set the
environment variables.

**Expected output** — the baseline evaluator, then a decaying loss, then a saved model:

```
Baseline Results: {'reranking_map': 1.0, 'reranking_mrr@10': 1.0, 'reranking_ndcg@10': 1.0}
{'loss': '2.279', 'grad_norm': '7.281', 'learning_rate': '9.375e-06', 'epoch': '0.5714'}
{'loss': '0.1893', 'grad_norm': '6.938', 'learning_rate': '5.208e-06', 'epoch': '1.686'}
{'loss': '0.1686', 'grad_norm': '7.25', 'learning_rate': '3.125e-06', 'epoch': '2.229'}
{'train_runtime': '47.91', 'train_samples_per_second': '35.7', 'train_loss': '0.5844', 'epoch': '3'}
Training complete. Best model loaded and saved to <out>/final
```

A post-training `predict()` sanity check should rank the relevant document first.

### Batch geometry

`gradient_accumulation_steps` is hardcoded to 2 and `dataloader_drop_last=True`, so

```
global_batch  = --batch x num_GPUs x 2
steps/epoch   = floor(pairs / (--batch x num_GPUs)) / 2
pairs         = train_rows x (1 + --n_neg)
```

With the shipped `--batch 64` at 8 ranks the global batch is 1024 against a 100-row sample
that expands to only ~594 pairs, so the run performs **a single optimizer step**, does not
crash, and exits 0 — a silent no-op, not a pass. Lower `--batch` (or enlarge the dataset)
until the step count is meaningful before launching multi-GPU. Watch `--eval_frac` too: on a 100-row
dataset `--eval_frac 0.003` leaves **one** eval query, which makes `reranking_ndcg@10` a
coin-flip metric that reports exactly `1.0` and `metric_for_best_model` meaningless.

### Multi-GPU

Plain single-node DDP, **no code change required**. `torchrun` spawns the ranks and the
script binds one rank per GPU (NCCL maps to RCCL on ROCm).

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port 29632 train_reranker_standalone.py \
  --data /path/to/train.jsonl --out /path/to/out --batch 16 --epochs 3
```

Budget disk: with `save_strategy="epoch"` + `load_best_model_at_end=True`, a single epoch
writes ~4.5 GB (checkpoint + optimizer state + final model). A full filesystem is a more
likely failure here than anything GPU-related.

### Quirks

- **Pair-role chat template.** `CrossEncoder(...)` installs ST's Qwen3-Reranker-specific
  `query`/`document` chat_template at load, so the default path needs no intervention. If
  another model raises ST 5.7.0's `_verify_pair_roles_supported` `ValueError`, supply a
  Query/Document template via `model.processor.chat_template = ...`,
  `processor_kwargs={"chat_template": ...}`, or a `chat_template.jinja` beside the model.
- **Score scale.** The saved model uses ST's `LogitScore` module with an `Identity`
  activation, so `predict()` returns **raw logits** (e.g. -0.625 / -9.375), not 0–1
  probabilities. When comparing against a reference implementation (e.g. the generative
  yes/no scorer in `inference/*/reranker/`), **diff the rankings, not the absolute scores** —
  the scale is template- and head-dependent.
- **flash-attn is optional.** The script falls back to `sdpa` automatically when the import
  fails — always the case on ROCm.
- **tf32 is never set by this script**, so the ROCm tf32 crash that hits other trainers in
  this repo does not apply here; matmuls run in bf16 via `bf16=True`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is honored by the CUDA allocator and
  ignored by the HIP one, which logs `expandable_segments not supported on this platform`.
  Harmless. So are the `warmup_ratio` / `logging_dir` v5 deprecations and
  `destroy_process_group() was not called before program exit`.
- Only `--batch` needs tuning across world sizes. `--max_len`, `--n_neg`, `--lr`, the
  hardcoded `gradient_accumulation_steps=2` and `gradient_checkpointing=True` work at their
  shipped values.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder base model |
| `--data` | `OTel_reranker_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--out` | `output` | Output dir for checkpoints, logs, and the final model |
| `--max_len` | `1024` | Max sequence length (query + document) |
| `--batch` | `64` | Per-GPU train/eval batch size — tuned for 80GB GPUs |
| `--epochs` | `2` | Number of training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--n_neg` | `5` | Hard negatives per query (`negative_1..negative_N`) |
| `--eval_frac` | `0.003` | Fraction of rows held out for evaluation |
| `--test_mode` | off | Cap the dataset at 200 rows for a quick test |
| `--hf_home` | `None` | Optional `HF_HOME` model-cache override |

## Output

- Checkpoints and TensorBoard logs under `--out` (`<out>/logs`).
- Baseline evaluation results written to `--out` before training.
- The best model (by `reranking_ndcg@10`) saved to `<out>/final`.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 8 GPUs |
| PyTorch | `torch==2.11.0` resolves to `+cu130` on plain PyPI — no `--index-url` | `torch==2.11.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2` |
| Attention | `flash_attention_2` when `flash_attn` imports, else `sdpa` | `sdpa` |

**No code change is needed on either vendor.** The two things to get right are the launcher
(`torchrun`, not `accelerate launch --num_processes=1`) and `--batch`, which silently
degenerates to a single optimizer step on a small dataset — see *Batch geometry*. The
shipped batch defaults are tuned for 80GB-class GPUs.

## Notes

- `transformers 5.x` development builds can miss `save_safetensors` — the script patches
  it onto the training args if absent.
- Gradient accumulation is fixed at 2 in the training args — effective batch =
  `batch × 2 × num_GPUs`.
- The shipped OTel sample is only for pipeline validation, not for producing a useful
  model.
