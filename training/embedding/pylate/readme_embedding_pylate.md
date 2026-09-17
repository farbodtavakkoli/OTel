# `training/embedding/pylate` — ColBERT late-interaction retriever fine-tuner

`train_embedding_pylate.py` trains a multi-vector (late-interaction) retriever with
[PyLate](https://github.com/lightonai/pylate), starting either from a plain BERT-style
encoder (PyLate appends a fresh `Dense` projection down to the per-token dimension) or
from an existing ColBERT checkpoint.

Unlike the bi-encoder in `../sentence_transformers`, PyLate keeps **one vector per token**
and scores pairs with MaxSim, so `.encode()` returns a ragged list of `[n_tokens, 128]`
arrays rather than an `[n, dim]` matrix, and it can build a multi-vector PLAID index for
`retrieve` / `rank`. Pick it for ColBERT-style retrieval quality when you can afford the
larger index; pick `../sentence_transformers` for a single-vector model.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU) and AMD MI355X 288GB (ROCm 7.2.4, 1 and
2 GPUs, DDP over RCCL). No code changes on either vendor. On ROCm the MaxSim backend is
`torch` and attention is `sdpa`; the fused `flash`/`lik` MaxSim kernels, `flash-attn`, and
`--tf32` are CUDA-only and none are required.

## Files

- `train_embedding_pylate.py` — trainer entry point; `utils.py` builds the model, loss,
  evaluator and post-training retrieval checks.
- `convert_data.py` — shared triplet sample -> PyLate JSONL.
- `requirements_embedding_pylate.txt` — pinned deps.
- `OTel_embedding_pylate_100.jsonl` — 100-row converted sample, the default `--train_file`.

## Setup

Python 3.12, in a **dedicated** venv: `pylate==1.6.0` hard-pins
`sentence-transformers==5.3.0` and `transformers<=5.3.0`, so sharing a venv with the other
embedding recipes downgrades both. Put the venv on a larger volume if the root filesystem
is tight:

```bash
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export PIP_CACHE_DIR=$DATA_DIR/pip_cache

cd training/embedding/pylate
ln -sf ../../../dev.env dev.env        # HF_TOKEN; optional here, no default model is gated
python3 -m venv $DATA_DIR/envs/.env_pylate
source $DATA_DIR/envs/.env_pylate/bin/activate
pip install --upgrade pip
```

### AMD (ROCm 7.2)

Install the ROCm torch wheel **first**, then the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding_pylate.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda)"
# expected: 2.11.0+rocm7.2 7.2.26015 None
```

Do **not** install `flash-attn` on ROCm. If a CUDA torch lands on top, recover with:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

### NVIDIA (CUDA 13.0)

Drop the ROCm index — `torch==2.11.0` resolves to a native `+cu130` wheel on PyPI:

```bash
pip install "torch==2.11.0" numpy            # -> torch 2.11.0+cu130
pip install -r requirements_embedding_pylate.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
# expected: 2.11.0+cu130 13.0 None
```

Optional CUDA-only extras: `pip install flash-attn` for
`--attn_implementation flash_attention_2`, and `pip install pylate[flash-maxsim]` or
`pylate[lik]` for `--scores_backend flash|lik`.

Verify either install:

```bash
python -c "import pylate, torch; from pylate import models, losses, indexes, retrieve, scores, rank; print('imports OK', pylate.__version__)"
```

If the datasets cache lands on a slow or shared mount, point it at tmpfs:
`export HF_DATASETS_CACHE=/dev/shm/dscache_pylate`.

## Data

`--train_file` is a JSONL with one late-interaction row per line:

```json
{"query": "...", "positive": "...", "negative": "..."}
```

Column *names* are free-form — the sentence-transformers collator feeds columns to the
loss **in order** (anchor, positive, then any negatives), so `negative_1 ... negative_N`
also works and gives the contrastive loss more explicit hard negatives per query (set
`--n_negatives` to match).

`convert_data.py` produces this from the repo's shared triplet sample
(`../sentence_transformers/OTel_embedding_sample_100.jsonl`, with
`anchor` / `positive` / `negative_1..5`):

```bash
python convert_data.py                          # 100 rows -> query/positive/negative
python convert_data.py --n_neg 5                # keep 5 hard negatives per row
python convert_data.py --n_neg 5 --explode      # 500 rows, one negative each
```

| Flag | Default | Meaning |
|---|---|---|
| `--src` | `../sentence_transformers/OTel_embedding_sample_100.jsonl` | Source JSONL |
| `--dst` | `OTel_embedding_pylate_100.jsonl` | Destination JSONL |
| `--n_neg` | `1` | Hard negatives per query to carry over |
| `--explode` | off | One row per negative instead of one wide row |
| `--max_chars` | `2000` | Truncate each text |

The converted `OTel_embedding_pylate_100.jsonl` ships here and is the default
`--train_file`, so the smoke test runs with no data arguments. It validates the pipeline
only; 100 rows produce no useful model.

## Run

Smoke test, single GPU:

```bash
source $DATA_DIR/envs/.env_pylate/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
python train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke1gpu \
  --epochs 1 --batch_size 8 --index_backend plaid
```

Multi-GPU (2 GPUs). Use `--master_port 29830`; 29500 is often taken:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke2gpu \
  --epochs 3 --batch_size 4 --gather_across_devices --index_backend plaid
```

Full run:

```bash
nohup torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --model_name lightonai/GTE-ModernColBERT-v1 \
  --train_file /path/to/your_train.jsonl \
  --epochs 3 --batch_size 16 --lr 1e-6 --n_negatives 5 --gather_across_devices \
  > train_embedding_pylate.log 2>&1 &

tail -f train_embedding_pylate.log
```

Expected tail — single GPU, shipped sample (90 train / 10 eval), `--epochs 4
--batch_size 8` = `ceil(90/8) x 4 = 48` steps. Read the epoch-boundary loss; per-step loss
is noisy on 90 rows:

```
[rank=0] INFO: torch 2.11.0+cu130 (hip=None cuda=13.0) devices=1 attn=sdpa scores_backend=torch
{'train_runtime': '9.2', 'train_samples_per_second': '39.13', 'train_loss': '1.379', 'epoch': '4'}
[rank=0] INFO: Relevant document is rank 1: True
[rank=0] INFO: PLAID rank 1: id 0 score 31.1191
```

The post-training check exercises all three retrieval paths — `scores.colbert_scores`,
`rank.rerank`, and (with `--index_backend plaid`) a PLAID index queried through
`retrieve.ColBERT`. Skip it with `--skip_late_interaction_check`.

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
| `--attn_implementation` | `auto` | `auto` = `sdpa` on ROCm, `flash_attention_2` on CUDA |
| `--scores_backend` | `auto` | MaxSim kernel; `auto` = `torch` on ROCm (`flash`/`lik` are CUDA-only) |
| `--tf32` | off | tf32 matmuls; ignored on ROCm |
| `--gradient_checkpointing` | off | Trade compute for activation memory |
| `--logging_steps` | `1` | Steps between loss log lines |
| `--eval_strategy` | `epoch` | Triplet-evaluator cadence: `no` / `steps` / `epoch` |
| `--save_strategy` | `epoch` | Checkpoint cadence: `no` / `steps` / `epoch` |
| `--save_steps` | `500` | Steps between checkpoints when `save_strategy=steps` |
| `--dataloader_drop_last` | off | Drop the trailing partial batch |
| `--dataloader_num_workers` | `2` | Dataloader worker processes |
| `--skip_late_interaction_check` | off | Skip the post-training multi-vector / MaxSim proof |
| `--index_backend` | `none` | `plaid` builds an index and runs end-to-end retrieval after training |

## Output

Under the resolved output dir: `run.log` (the rank-0 log, including the late-interaction
proof), `checkpoint-<step>/` per `--save_strategy` (`save_total_limit=1`), `final_model/`
with the trained ColBERT model, and `indexes/` when `--index_backend plaid` is set.

With an eval split that has negatives, a `ColBERTTripletEvaluator` reports MaxSim triplet
accuracy each epoch.

## Notes

- **Reload with `pylate.models.ColBERT(model_name_or_path=".../final_model")`, not
  `SentenceTransformer`.** `final_model/` holds the encoder plus the `Dense` projection;
  plain `SentenceTransformer` returns a pooled single vector and silently discards the
  late-interaction paradigm.
- A ColBERT checkpoint brings its own projection, and PyLate still appends a
  `Dense(->128)` on top unless you pass the checkpoint's native `--embedding_dim` (e.g.
  `96` for `answerdotai/answerai-colbert-small-v1`).
- Under DDP, sentence-transformers force-enables `dataloader_drop_last`, so budget the
  batch so `floor(train_rows / (batch x ranks)) > 0` or the run trains nothing.
- PLAID prunes aggressively on tiny indexes — a 3-document index can return fewer than `k`
  hits. Use `--index_backend plaid` as a smoke check and `rank.rerank` when you need
  exhaustive scoring.
- MaxSim scores are unnormalized sums over query tokens, so they grow with query length
  and mildly favour longer documents. Compare scores only within one query.
