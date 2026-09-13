# `train_embedding_pylate.py`

## Overview

**ColBERT late-interaction (multi-vector) retriever fine-tuner** built on
[PyLate](https://github.com/lightonai/pylate) (`lightonai/pylate`, PyPI `pylate`). One
script trains a late-interaction model either from a plain BERT-style encoder (PyLate
appends a fresh linear projection to a small per-token dimension) or from an existing
ColBERT checkpoint.

Unlike the bi-encoder in `../sentence_transformers`, PyLate keeps **one vector per token**
and scores query/document pairs with MaxSim. `.encode()` therefore returns a ragged list of
`[n_tokens, 128]` arrays — one 2-D array per text, not a `[n, dim]` matrix. It can also
build a multi-vector index (PLAID, Voyager, ScaNN, WARP) and serve `retrieve` / `rank`.

Design notes:

- **Registry-driven** — `resolve_cfg` merges `DEFAULT_CFG` ← `MODELS[model_name]` ← CLI
  overrides, matching `../sentence_transformers`.
- **Two model routes** — a base encoder (`BAAI/bge-small-en-v1.5`) gets a *new* random
  `Dense(384→128)` projection; a ColBERT checkpoint (`lightonai/GTE-ModernColBERT-v1`,
  `colbert-ir/colbertv2.0`, `answerdotai/answerai-colbert-small-v1`) brings its own. PyLate
  still appends a `Dense(→128)` on top unless you pass the checkpoint's native
  `--embedding_dim` (e.g. `96` for `answerai-colbert-small-v1`).
- **ROCm switches** are automatic: `--attn_implementation auto` → `sdpa` on HIP builds,
  `--tf32` is ignored unless `torch.version.cuda is not None`, and `--scores_backend auto`
  pins PyLate's MaxSim kernel to `torch` on HIP (see *Notes & quirks*).
- **Built-in late-interaction proof** — after training, the script encodes a query and
  three documents, logs each embedding's shape, ranks them with `colbert_scores` (MaxSim),
  re-ranks with `rank.rerank`, and optionally builds a PLAID index and retrieves.

## Install

Python 3.12, in its own venv (PyLate hard-pins sentence-transformers — see
*Notes & quirks* — so do not share a venv with the other embedding folders). If the root
filesystem is tight, put the venv outside the repo on a larger volume:

```
$DATA_DIR/envs/.env_pylate
```

The commands below refer to a few machine-specific locations through environment
variables — set them to suit your machine:

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

### AMD (ROCm)

Install torch from the ROCm wheel index **first**, then PyLate, then re-check that pip did
not swap in a CUDA torch:

```bash
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
python3 -m venv $DATA_DIR/envs/.env_pylate
source $DATA_DIR/envs/.env_pylate/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding_pylate.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda)"
# expected: 2.11.0+rocm7.2 7.2.26015 None
```

If that last line ever shows a CUDA build, recover with:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

**Never `pip install flash-attn`** — it is a CUDA-only build. The script selects `sdpa`
automatically on ROCm.

### NVIDIA (CUDA)

Default PyPI wheels work — drop the ROCm index-url line. The `torch==2.11.0` pin in the
requirements file resolves to a native `+cu130` wheel on plain PyPI, so no `--index-url`
is needed (a bare `pip install torch` would give a newer build; either runs on a CUDA 13
driver):

```bash
python3 -m venv $DATA_DIR/envs/.env_pylate
source $DATA_DIR/envs/.env_pylate/bin/activate
pip install --upgrade pip
pip install "torch==2.11.0" numpy            # -> torch 2.11.0+cu130
pip install -r requirements_embedding_pylate.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
# expected: 2.11.0+cu130 13.0 None   <-- pylate did not swap torch
```

On CUDA you additionally get `--attn_implementation flash_attention_2` (needs
`flash-attn`), the fused MaxSim kernels via `pip install pylate[flash-maxsim]` or
`pylate[lik]` (`--scores_backend flash|lik`), and `--tf32`. All are CUDA-only; none are
required.

If the datasets cache lands on a slow or shared mount, point it at tmpfs:
`export HF_DATASETS_CACHE=/dev/shm/dscache_pylate`.

Verify either way:

```bash
python -c "import pylate, torch; from pylate import models, losses, indexes, retrieve, scores, rank; print('imports OK', pylate.__version__)"
```

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env` and loaded via `load_dotenv("dev.env")`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

None of the default models are gated, so the token is optional here. Point the Hub cache
at a volume with room for the model downloads:

```bash
export HF_HOME=/path/to/hf_cache
```

## Data

`--train_file` is a JSONL with one late-interaction training row per line:

```json
{"query": "...", "positive": "...", "negative": "..."}
```

Column *names* are free-form — the sentence-transformers collator feeds columns to the
loss **in order** (anchor, positive, then any negatives), so `negative_1 … negative_N`
also works and gives the contrastive loss more explicit hard negatives per query.

### Converter

The repo's shared sample (`../sentence_transformers/OTel_embedding_sample_100.jsonl`,
100 rows, `anchor` / `positive` / `negative_1..5`) is converted by `convert_data.py`:

```bash
python convert_data.py                          # 100 rows -> query/positive/negative
python convert_data.py --n_neg 5                # keep 5 hard negatives per row
python convert_data.py --n_neg 5 --explode      # 500 rows, one negative each
```

The converted default, **`OTel_embedding_pylate_100.jsonl` (100 rows), ships in this
folder** and is the default `--train_file`, so the smoke test runs with no arguments.

| Flag | Default | Meaning |
|---|---|---|
| `--src` | `../sentence_transformers/OTel_embedding_sample_100.jsonl` | Source JSONL |
| `--dst` | `OTel_embedding_pylate_100.jsonl` | Destination JSONL |
| `--n_neg` | `1` | Hard negatives per query to carry over |
| `--explode` | off | One row per negative instead of one wide row |
| `--max_chars` | `2000` | Truncate each text |

## Run

### Smoke test (single GPU)

```bash
source $DATA_DIR/envs/.env_pylate/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
python train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke1gpu \
  --epochs 1 --batch_size 8 --index_backend plaid
```

### Multi-GPU (2 GPUs)

```bash
source $DATA_DIR/envs/.env_pylate/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke2gpu \
  --epochs 3 --batch_size 4 --gather_across_devices --index_backend plaid
```

Budget the batch so `floor(train_rows / (batch × ranks)) > 0` — sentence-transformers
force-enables `dataloader_drop_last` under DDP.

### Full run

```bash
nohup torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --model_name lightonai/GTE-ModernColBERT-v1 \
  --train_file /path/to/your_train.jsonl \
  --epochs 3 --batch_size 16 --lr 1e-6 --n_negatives 5 --gather_across_devices \
  > train_embedding_pylate.log 2>&1 &

tail -f train_embedding_pylate.log
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `BAAI/bge-small-en-v1.5` | Base encoder or ColBERT checkpoint; selects the registry entry |
| `--train_file` | `OTel_embedding_pylate_100.jsonl` | Training JSONL (query/positive/negative) |
| `--output_dir` | `None` | Output dir; defaults to `<experiment_root>/<run_id>/colbert_<model>` |
| `--experiment_root` | `experiments` | Root for the default output dir |
| `--run_id` | `None` | Run id in the default output dir (default: UTC timestamp) |
| `--embedding_dim` | `128` (registry) | Per-token output dim of the ColBERT projection |
| `--query_length` | `32` (registry) | Query token budget, padded with expansion tokens |
| `--document_length` | `180` (registry) | Document token budget |
| `--batch_size` | registry | Per-device train batch size |
| `--eval_batch_size` | `32` | Per-device eval/encode batch size |
| `--epochs` | `1` (registry) | Training epochs |
| `--max_steps` | `-1` | Hard cap on optimizer steps; `-1` disables |
| `--lr` | registry | Learning rate |
| `--warmup_ratio` | `0.05` | Fraction of steps used for LR warmup |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--loss` | `contrastive` | `contrastive` / `cached_contrastive` / `distillation` |
| `--temperature` | `1.0` | Softmax temperature for the contrastive loss |
| `--gather_across_devices` | off | All-gather document embeddings across ranks for a bigger negative pool |
| `--score_mini_batch_size` | `None` | Chunk queries during loss scoring to cut transient memory |
| `--n_negatives` | `1` (registry) | Hard negative columns fed to the loss |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--eval_fraction` | `0.1` | Fraction held out for the ColBERT triplet evaluator |
| `--seed` | `42` | Seed for data splits and init |
| `--dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--attn_implementation` | `auto` | `auto` → `sdpa` on ROCm, `flash_attention_2` on CUDA |
| `--scores_backend` | `auto` | MaxSim kernel; `auto` → `torch` on ROCm (`flash`/`lik` are CUDA-only) |
| `--tf32` | off | tf32 matmuls; silently ignored on ROCm |
| `--gradient_checkpointing` | off | Trade compute for activation memory |
| `--logging_steps` | `1` | Steps between loss log lines |
| `--eval_strategy` | `epoch` | ColBERT triplet-evaluator cadence: `no` / `steps` / `epoch` |
| `--save_strategy` | `epoch` | Checkpoint cadence: `no` / `steps` / `epoch` |
| `--save_steps` | `500` | Steps between checkpoints when `save_strategy=steps` |
| `--dataloader_drop_last` | off | Drop the trailing partial batch (off so tiny samples still yield steps) |
| `--dataloader_num_workers` | `2` | Dataloader worker processes |
| `--skip_late_interaction_check` | off | Skip the post-training multi-vector / MaxSim proof |
| `--index_backend` | `none` | `plaid` builds an index and runs end-to-end retrieval after training |

## Output

Under the resolved output dir:

- `run.log` — the rank-0 log, including the late-interaction proof.
- `checkpoint-<step>/` — periodic checkpoints (`--save_strategy`, `save_total_limit=1`).
- `final_model/` — the trained ColBERT model, reloadable with
  `pylate.models.ColBERT(model_name_or_path=".../final_model")`. It contains the encoder
  **plus** the `Dense` projection module — reloading it with plain
  `SentenceTransformer` would give you a pooled vector and silently lose the paradigm.
- `indexes/` — the PLAID index, when `--index_backend plaid`.

If an eval split with negatives exists, a `ColBERTTripletEvaluator` reports MaxSim-based
triplet accuracy each epoch.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 2 GPUs (DDP over RCCL) |
| PyTorch | `torch==2.11.0` resolves to a `+cu130` wheel on plain PyPI — no `--index-url` | `torch==2.11.0` from `download.pytorch.org/whl/rocm7.2` |
| Attention | `sdpa`, or `flash_attention_2` with `pip install flash-attn` | `sdpa` only |
| MaxSim backend | `torch`, or the fused `flash`/`lik` kernels | `torch` only (the fused kernels are CUDA-only) |
| tf32 | `--tf32` engages | gated off (a no-op on HIP) |

No code changes are needed on either vendor: DDP uses the `nccl` backend string on both.

### Expected output

A single-GPU smoke on the shipped 100-row sample (90 train / 10 eval) with
`--epochs 4 --batch_size 8` runs `ceil(90/8) x 4 = 48` optimizer steps:

```
[rank=0] INFO: torch 2.11.0+cu130 (hip=None cuda=13.0) devices=1 attn=sdpa scores_backend=torch
[rank=0] WARNING: The checkpoint contains a final projection layer with output dimension (384, 96). Adding a dense layer with output dimensions (96, 128).
{'loss': '0.8416', 'grad_norm': '6.938', 'learning_rate': '5.556e-07', 'epoch': '2'}
{'loss': '0.5673', 'grad_norm': '6.938', 'learning_rate': '2.889e-07', 'epoch': '3'}
{'loss': '0.3965', 'grad_norm': '6.25', 'learning_rate': '2.222e-08', 'epoch': '4'}
{'eval_accuracy': '1', 'eval_runtime': '0.0448', 'epoch': '4'}
{'train_runtime': '9.2', 'train_samples_per_second': '39.13', 'train_loss': '1.379', 'epoch': '4'}
```

Read the **epoch-boundary** loss, not the per-step values — on 90 rows the per-step loss is
noisy.

The post-training check then exercises all three retrieval paths — `scores.colbert_scores`,
`rank.rerank`, and a PLAID index queried with `retrieve.ColBERT`:

```
[rank=0] INFO: MAXSIM rank 1: score 31.1195 | doc[0] Once the path message reaches the exit ASBR, any choice of inter-AS TE
[rank=0] INFO: Relevant document is rank 1: True
[rank=0] INFO: RERANK rank 1: id 0 score 31.1195
[rank=0] INFO: ✅ Index with FastPlaid backend.
[rank=0] INFO: PLAID rank 1: id 0 score 31.1191
```

## Notes & quirks

- **PyLate hard-pins sentence-transformers.** `pylate==1.6.0` requires
  `sentence-transformers==5.3.0` (exact `==`) and `transformers<=5.3.0`. Installing it
  into a venv that has the repo-proven ST 5.7.0 / transformers 5.5.0 **will downgrade
  both**. This is the main reason this folder gets its own venv. Both downgraded versions
  work fine on gfx950.
- **`pip install pylate` did not clobber the ROCm torch** — none of its
  dependencies pin torch. Re-check anyway; the recovery command is in *Install*.
- **`fast-plaid` needs no CUDA toolchain.** It ships a prebuilt `manylinux_2_28
  cp312` wheel and drives torch, so the PLAID index builds and queries on ROCm with no
  compilation. Its GPU path uses the ROCm torch you already installed.
- **PyLate's fused MaxSim kernels are CUDA-only.** PyLate's backend auto-detection tests
  `tensor.is_cuda`, which is **`True` on ROCm**, so `backend="auto"` could dispatch to a
  kernel that is not there. The script sets `PYLATE_SCORES_BACKEND=torch` explicitly on HIP
  builds via `--scores_backend auto` — leave it on `auto`.
- **PLAID prunes aggressively on tiny indexes** — centroid pruning drops candidates that
  share no centroid with the query, so a 3-document index can return fewer hits than `k`.
  Use `--index_backend plaid` as a smoke check; use `rank.rerank` when you need exhaustive
  scoring.
- **Never `pip install flash-attn`** — CUDA-only. `--attn_implementation auto` picks
  `sdpa` on ROCm.
- **`tf32=True` raises on ROCm.** `--tf32` is gated behind `torch.version.cuda is not None`
  and is silently a no-op on HIP.
- **Master port 29500 is often already in use on a shared box.** Use `--master_port 29830`
  (or any other free port).
- **MaxSim scores are unnormalized sums over query tokens**, so raw scores grow with query
  length and mildly favour longer documents. Compare scores only within one query; do not
  read absolute magnitudes as calibrated relevance.
- **DDP forces `dataloader_drop_last=True`.** sentence-transformers overrides the flag
  under DDP to avoid hangs on an uneven last batch, and logs a warning. Budget your batch
  size so `floor(train_rows / (batch × ranks))` is still > 0 — the same constraint documented in
  `../sentence_transformers`.
- **`warmup_ratio` is deprecated** in transformers v5 (warns, still works); switch to
  `warmup_steps` as a float when ST drops v4 support.
- **`destroy_process_group()` not called** — a benign torch shutdown warning at the end of
  the 2-GPU run; `rc=0` regardless.
- **Reload with `pylate.models.ColBERT`, not `SentenceTransformer`.** `final_model/`
  contains the encoder plus the `Dense` projection. Loading it with plain
  `SentenceTransformer` gives a pooled vector and silently discards the whole paradigm.
- **The shipped OTel sample is for pipeline validation only** — 100 rows produce no useful
  model. If you replicate it to get a longer run, write the copy outside the repo — it is a
  pipeline proof, not a learning result.
