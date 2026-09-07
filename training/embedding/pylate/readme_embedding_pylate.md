# `train_embedding_pylate.py`

## Overview & when to use

**ColBERT late-interaction (multi-vector) retriever fine-tuner** built on
[PyLate](https://github.com/lightonai/pylate) (`lightonai/pylate`, PyPI `pylate`). One
script trains a late-interaction model either from a plain BERT-style encoder (PyLate
appends a fresh linear projection to a small per-token dimension) or from an existing
ColBERT checkpoint.

### What this does that the bi-encoder sibling cannot

Every other embedding/reranker folder in this repo produces **one vector per text**
(`../sentence_transformers` — a bi-encoder) or **one score per (query, document) pair**
(the `../../reranker/*` cross-encoders). PyLate is a third thing:

| | bi-encoder (`../sentence_transformers`) | **late interaction (this folder)** | cross-encoder (`../../reranker/*`) |
|---|---|---|---|
| Representation | 1 pooled vector per text | **1 vector per token** | none — scores pairs directly |
| Query/doc scoring | dot product / cosine of 2 vectors | **MaxSim**: for each query token take its best-matching document token, sum | full cross-attention over the concatenated pair |
| Documents encoded once, offline? | yes | **yes** | no — every pair needs a forward pass |
| Index unit | 1 vector/doc | **N vectors/doc** (N = tokens) | n/a |
| Typical accuracy | baseline | **higher than bi-encoder** | highest |
| Retrieval over 1M docs | milliseconds | tens of milliseconds (PLAID) | infeasible |

**The retrieval problem a pooled vector cannot solve.** A bi-encoder compresses a whole
passage into a single point, so *term-level* evidence is averaged away. If a query asks
about `inter-AS TE link` and one 200-word passage mentions exactly that phrase while
another is topically similar but never uses the terms, the pooled vectors can be nearly
indistinguishable — the signal is one phrase out of two hundred words. Late interaction
keeps every token's vector, so that one phrase gets its own high-similarity match and
contributes directly to the score. This is why ColBERT-family models beat single-vector
retrievers on out-of-domain and long-tail/entity-heavy queries, which is precisely the
regime where a domain-adapted retriever is worth training at all.

**What it costs.** You pay in index size and latency:

- **Index size.** One vector per *token*, not per document. A 180-token document at 128
  dims in fp16 is ~46 KB versus ~0.75 KB for a single 384-dim bi-encoder vector — roughly
  **50–100× larger** before compression. PyLate's PLAID index exists to claw this back
  (residual compression at `nbits=2/4`, typically ~10–30× smaller than raw), which is why
  you index with PLAID rather than dumping raw tensors.
- **Latency.** MaxSim is an `(Qt × Dt)` similarity matrix per candidate document instead
  of one dot product. PLAID handles this with centroid-based pruning, but retrieval is
  still meaningfully slower than a flat single-vector ANN search — while remaining orders
  of magnitude faster than running a cross-encoder over the corpus.
- **Ops.** Two extra moving parts: the per-token projection dimension (`--embedding_dim`,
  128 by default) and the index build itself.

Use this folder when bi-encoder recall is your bottleneck and a cross-encoder over the
whole corpus is too slow — i.e. as the *first-stage retriever*, or as a fast reranker
(`rank.rerank`, no index needed) sitting between a cheap retriever and an expensive
cross-encoder.

### Is PyLate just a sentence-transformers wrapper?

**No — verified against the source, not assumed.** PyLate imports
`SentenceTransformerTrainer` and `SentenceTransformerTrainingArguments`, so the *training
loop* is shared. Everything that defines the retrieval paradigm is PyLate's own:

- `pylate.models.ColBERT` subclasses `SentenceTransformer` but sets
  `similarity_fn_name="MaxSim"`, appends a `pylate.models.Dense` projection to
  `embedding_size` (128), and adds ColBERT-specific machinery a bi-encoder has no concept
  of: `query_prefix`/`document_prefix` marker tokens, **query expansion** (pad queries to
  `query_length` with `[MASK]` tokens that participate in scoring), a punctuation
  `skiplist`, and separate query/document length budgets. It also reads Stanford-NLP
  ColBERT checkpoints (`HF_ColBERT` architecture + `artifact.metadata`).
- **`.encode()` returns a ragged list of `[n_tokens, 128]` arrays** — one 2-D array per
  text, not a `[n, dim]` matrix. Proven below.
- `pylate.losses.{Contrastive, CachedContrastive, Distillation}` compute the loss over
  MaxSim scores, not cosine of pooled vectors.
- `pylate.scores` — `colbert_scores`, `colbert_scores_pairwise`, `colbert_kd_scores`,
  XTR variants, with pluggable MaxSim kernels (`torch` / `flash` / `lik`).
- `pylate.indexes` — **PLAID** (Rust `fast-plaid`), Voyager, ScaNN, WARP, Stanford PLAID.
  A bi-encoder index type does not apply here at all.
- `pylate.retrieve` / `pylate.rank` — end-to-end multi-vector retrieval and reranking.

Design notes:

- **Registry-driven** — `resolve_cfg` merges `DEFAULT_CFG` ← `MODELS[model_name]` ← CLI
  overrides, matching `../sentence_transformers`.
- **Two model routes** — a base encoder (`BAAI/bge-small-en-v1.5`) gets a *new* random
  `Dense(384→128)` projection; a ColBERT checkpoint
  (`lightonai/GTE-ModernColBERT-v1`, `colbert-ir/colbertv2.0`) brings its own.
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

### AMD (ROCm) — the verified route

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

**In practice it was not needed** — `pip install pylate==1.6.0` left
`torch 2.11.0+rocm7.2` untouched, because none of PyLate's dependencies pin torch.

**Never `pip install flash-attn`** — it is a CUDA-only build. The script selects `sdpa`
automatically on ROCm.

### NVIDIA (CUDA)

Default PyPI wheels work; drop the ROCm index-url line:

```bash
pip install -r requirements_embedding_pylate.txt
```

On CUDA you additionally get `--attn_implementation flash_attention_2` (needs
`flash-attn`) and the fused MaxSim kernels via `pip install pylate[flash-maxsim]` or
`pylate[lik]` (`--scores_backend flash|lik`). Both are CUDA-only.

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
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4
python train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke1gpu \
  --epochs 1 --batch_size 8 --index_backend plaid
```

### Multi-GPU (2 GPUs, master port 29830)

```bash
source $DATA_DIR/envs/.env_pylate/bin/activate
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke2gpu \
  --epochs 3 --batch_size 4 --gather_across_devices --index_backend plaid
```

`--gather_across_devices` makes multi-GPU more than data parallelism: document embeddings
are all-gathered (with gradients) across ranks, so each query is scored against every
rank's documents — an `N×` larger in-batch negative pool.

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

Full captured evidence for the runs below: **`smoke_mi355x.log`** in this folder.

## Hardware support & evidence

- **Tested:** 1× and 2× AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
  Python 3.12.3, `torch==2.11.0+rocm7.2`, `pylate==1.6.0`.
- **AMD:** **tested, works unmodified.** No PyLate source changes were needed. The ROCm
  torch build keeps the `torch.cuda` API semantics, so DDP over RCCL (`nccl` backend),
  bf16, `sdpa` attention, the MaxSim `torch` kernel, and the Rust `fast-plaid` index all
  behave as on CUDA.
- **NVIDIA:** untested here. Should work with default PyPI wheels; CUDA additionally
  unlocks `flash_attention_2` and the fused `flash`/`lik` MaxSim kernels.
- **Other hardware (upstream claims — not verified here):** LightOn claims PyLate trains
  "on hardware ranging from a single CPU to a multi-GPU node" — i.e. **CPU** training is
  claimed; Apple Silicon (Metal/MPS) is claimed only by the separate `pylate-rs`
  inference engine, not for training. No other accelerator is claimed upstream.

### Platform notes — AMD Instinct MI355X (ROCm 7.2)

This path works as documented on MI355X, with no code changes to PyLate. Both smoke tests
pass end to end.

Install that worked (Python 3.12.3, ROCm 7.2.4) — see *Install* for the full block. Notably
**`pip install pylate==1.6.0` did not swap in a CUDA torch**; `torch 2.11.0+rocm7.2`
survived untouched, so the documented `--force-reinstall` recovery was never needed.

#### 1. Single GPU

```bash
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4
python train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke1gpu \
  --epochs 1 --batch_size 8 --index_backend plaid
```

12 optimizer steps on the shipped 100-row sample (90 train / 10 eval), finite loss,
checkpoint + `final_model` written. **Expected output:**

```
[rank=0] INFO: torch 2.11.0+rocm7.2 (hip=7.2.26015 cuda=None) devices=1 attn=sdpa scores_backend=torch
[rank=0] INFO: The checkpoint does not contain a linear projection layer. Adding one with output dimensions (384, 128).
{'loss': '0.486', 'grad_norm': '20.62', 'learning_rate': '3e-06', 'epoch': '0.1667'}
{'loss': '0.1554', 'grad_norm': '5.281', 'learning_rate': '8.182e-07', 'epoch': '0.8333'}
{'eval_accuracy': '1', 'eval_runtime': '0.1697', 'epoch': '1'}
{'train_runtime': '8.034', 'train_samples_per_second': '11.2', 'train_loss': '0.4855', 'epoch': '1'}
```

`train_loss` 0.4855 over 12 steps; `ColBERTTripletEvaluator` accuracy 1.00 on the 10-row
held-out split. Per-step loss on a 90-row sample is noisy — the aggregate is the signal.

#### 2. Two GPUs (physical 4 and 5, master port 29830)

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
torchrun --nproc_per_node=2 --master_port=29830 train_embedding_pylate.py \
  --output_dir $OUTPUT_DIR/train_embedding_pylate/smoke2gpu \
  --epochs 3 --batch_size 4 --gather_across_devices --index_backend plaid
```

`rc=0`, 33 steps, `train_runtime 9.938s`, `train_samples_per_second 27.17`, and the
late-interaction check passed on rank 0. A longer run (120× replicated sample, 156 steps,
`--batch_size 32`) was used purely to get a clean utilization window:

```
{'loss': '1.854', 'grad_norm': '12.88', 'learning_rate': '1.524e-06', 'epoch': '1.042'}
{'loss': '1.829', 'grad_norm': '15.38', 'learning_rate': '1.129e-07', 'epoch': '1.935'}
{'eval_accuracy': '0.89', 'eval_runtime': '1.37', 'epoch': '2'}
{'train_runtime': '28.97', 'train_samples_per_second': '745.6', 'train_steps_per_second': '11.6', 'epoch': '2'}
```

`rocm-smi` sampled every 2s from outside the job, restricted to cards 4 and 5 — **both
GPUs busy simultaneously**:

```
card4 use=91% vram=2% card5 use=92% vram=2%  used=7.9GiB 7.9GiB
card4 use=92% vram=2% card5 use=93% vram=2%  used=7.9GiB 7.9GiB
card4 use=95% vram=2% card5 use=92% vram=2%  used=7.9GiB 7.9GiB
card4 use=93% vram=2% card5 use=97% vram=2%  used=7.9GiB 7.9GiB
card4 use=91% vram=2% card5 use=94% vram=2%  used=7.9GiB 7.9GiB
```

89–97% on both cards, 7.9 GiB each (~2.7% of 288GB) at `--batch_size 32` with a
`bge-small` backbone — enormous headroom for a bigger model or batch.

#### 3. The late-interaction property is real, not nominal

This is the evidence that matters most, because it is the capability no other folder in
this repo has. After training, on the **same `BAAI/bge-small-en-v1.5` backbone**, the same
query and the same 10 documents, encoded two ways:

```
--- PyLate ColBERT (late interaction) ---
query embedding: (32, 128) ndim 2
doc embedding shapes: [(38, 128), (67, 128), (159, 128), (90, 128), (66, 128),
                       (47, 128), (134, 128), (51, 128), (27, 128), (29, 128)]
MaxSim scores: [18.974, 19.422, 16.525, 15.714, 11.659, 12.812, 17.802, 17.49, 14.823, 13.179]

--- sentence-transformers bi-encoder (pooled) ---
query embedding: (1, 384) -> per text (384,) ndim 1
cosine scores: [0.722, 0.782, 0.539, 0.564, 0.506, 0.486, 0.521, 0.55, 0.469, 0.512]

--- index footprint for these 10 documents ---
late interaction: 708 token vectors x 128 dims = 90624 floats
bi-encoder:       10 doc vectors x 384 dims =  3840 floats
ratio: 23.6x more storage for late interaction
```

Read that carefully:

- The ColBERT output is **one 2-D array per text**, with a row count that tracks the
  document's token count (27 … 159 rows). `ndim == 2`. The bi-encoder output is
  `ndim == 1` — a single 384-vector, identical shape regardless of document length.
  This is a structural difference, not a configuration difference.
- Queries are always `(32, 128)` — padded to `--query_length` with `[MASK]` **expansion
  tokens** that participate in scoring. A bi-encoder has no analogue.
- The 23.6× storage ratio is the real, measured cost of the paradigm on this data (a
  128-dim per-token projection against a 384-dim pooled vector; it would be ~70× against
  an equal-dim baseline). PLAID compression is what makes it deployable.

And MaxSim retrieval works end to end. From the shipped in-script check
(positive, its hard negative, an unrelated passage):

```
[rank=0] INFO: MAXSIM rank 1: score 27.3844 | doc[0] Once the path message reaches the exit ASBR, any choice of inter-AS TE
[rank=0] INFO: MAXSIM rank 2: score 24.6381 | doc[1] Alternatively, if the next hop in the ERO is the entry ASBR for AS3 (s
[rank=0] INFO: MAXSIM rank 3: score 19.4948 | doc[2] Each chunk of audio data is preceded by an RTP header; RTP header and
[rank=0] INFO: Relevant document is rank 1: True
[rank=0] INFO: RERANK rank 1: id 0 score 27.3844
[rank=0] INFO: PLAID rank 1: id 0 score 27.3853
```

The relevant document ranks first; its hard negative second; the unrelated passage last.
The same ordering comes out of all three code paths — raw `scores.colbert_scores`,
`rank.rerank`, and a **PLAID index + `retrieve.ColBERT`** — with PLAID reproducing the
exact-MaxSim score to 4 decimal places (27.3853 vs 27.3844), which is a good end-to-end
check that the Rust index is correct on ROCm. (PLAID returned only the top 2 of the 3
indexed documents despite `k=3`; its centroid pruning drops candidates that share no
centroid with the query, which on a 3-document toy index is expected — see
*Notes & quirks*.)

### Platform notes — NVIDIA H100 80GB (CUDA 13.0)

This path works as documented on H100, with no code changes to PyLate. The single-GPU
ColBERT smoke passes end to end on one H100 (one GPU of a shared 8×H100 node; driver
580.173.02, Hopper cc(9,0), Python 3.12.3). This mirrors the MI355X result — the only differences are the
three ROCm accommodations *unwinding* on CUDA (flash-attn becomes available, `tf32`
engages instead of being gated off, and PyLate's MaxSim backend is no longer force-pinned).

**The `torch==2.11.0` pin is honorable on CUDA 13** — a `torch==2.11.0+cu130` wheel exists
on PyPI, so the requirements pin installs unchanged (a bare `pip install torch` would give
2.13.0+cu130 instead; either runs on driver 580). As on MI355X, **installing `pylate`
did NOT clobber torch** — none of its dependencies pin it, so `torch 2.11.0+cu130`
survived and the documented recovery was not needed. Re-verify anyway.

Install that worked (venv on tmpfs, deliberately outside the repo as on MI355X):

```bash
export PIP_CACHE_DIR=/dev/shm/pipcache
python3 -m venv /dev/shm/.env_pylate
source /dev/shm/.env_pylate/bin/activate
pip install --upgrade pip
pip install "torch==2.11.0" numpy            # -> torch 2.11.0+cu130 (default PyPI, NO --index-url)
pip install -r requirements_embedding_pylate.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
# expected: 2.11.0+cu130 13.0 None   <-- pylate did not swap torch
python -c "import pylate; from pylate import models, losses, indexes, retrieve, scores, rank; print('imports OK', pylate.__version__)"
```

Key versions: `torch 2.11.0+cu130` (CUDA 13.0), `pylate 1.6.0`, `sentence-transformers
5.3.0`, `transformers 5.3.0`, `tokenizers 0.22.2`, `accelerate 1.14.0`, `datasets 5.0.1`,
`fast-plaid 1.4.6.2110`; NVIDIA driver 580.173.02.

Exact smoke command (native ColBERT checkpoint, cleanest smoke — no random projection):

```bash
export CUDA_VISIBLE_DEVICES=4                      # plain CUDA; NO HIP_VISIBLE_DEVICES
export HF_HOME=/path/to/hf_cache                   # HF model cache
export HF_DATASETS_CACHE=/dev/shm/dscache_pylate
python train_embedding_pylate.py \
  --model_name answerdotai/answerai-colbert-small-v1 \
  --output_dir /dev/shm/pylate/smoke1gpu \
  --epochs 4 --batch_size 8 \
  --attn_implementation sdpa --scores_backend torch --tf32 \
  --index_backend plaid
```

**48 optimizer steps** on the shipped 100-row sample (90 train / 10 eval, `drop_last`
off → `ceil(90/8)=12` steps/epoch × 4 epochs). `epochs`/`batch_size` were bumped from the
1-epoch registry default specifically to clear the "non-trivial step count" bar.
**Expected output** (banner, per-epoch end loss decreasing, evaluator, final):

```
[rank=0] INFO: torch 2.11.0+cu130 (hip=None cuda=13.0) devices=1 attn=sdpa scores_backend=torch
[rank=0] WARNING: The checkpoint contains a final projection layer with output dimension (384, 96). Adding a dense layer with output dimensions (96, 128).
{'loss': '0.8416', 'grad_norm': '6.938', 'learning_rate': '5.556e-07', 'epoch': '2'}
{'loss': '0.5673', 'grad_norm': '6.938', 'learning_rate': '2.889e-07', 'epoch': '3'}
{'loss': '0.3965', 'grad_norm': '6.25', 'learning_rate': '2.222e-08', 'epoch': '4'}
{'eval_accuracy': '1', 'eval_runtime': '0.0448', 'epoch': '4'}
{'train_runtime': '9.2', 'train_samples_per_second': '39.13', 'train_steps_per_second': '5.217', 'train_loss': '1.379', 'epoch': '4'}
```

`train_loss` 1.379 aggregate; the epoch-boundary step (the full-epoch reduction, least
noisy) drops monotonically **0.8416 → 0.5673 → 0.3965** across epochs 2/3/4, and the
`ColBERTTripletEvaluator` holds 1.00 on the 10-row held-out split every epoch. Per-step
loss on 90 rows is noisy — the aggregate and the epoch-boundary trend are the signal, same
caveat as MI355X.

GPU residency, sampled by **VRAM-by-PID from inside the run** (`nvidia-smi -i <gpu>
--query-compute-apps=pid,used_memory`; when the card is otherwise idle the single listed
PID is the training job):

```
SMIPOLL | gpu_util,memused=[2 %, 799 MiB]  | compute-apps(pid,mem)=[<pid>, 690 MiB]
SMIPOLL | gpu_util,memused=[0 %, 1741 MiB] | compute-apps(pid,mem)=[<pid>, 1732 MiB]
SMIPOLL | gpu_util,memused=[5 %, 1741 MiB] | compute-apps(pid,mem)=[<pid>, 1732 MiB]
```

Peak ~1.7 GiB VRAM for a `bge`-class BERT backbone at `batch_size 8` — **~2% of the H100's
80 GB**, so batch/model/`document_length` have enormous headroom (VRAM is 80 GB here vs
288 GB on MI355X, but this workload is nowhere near either limit). Utilization reads low
because the whole train is ~9 s and the sampler ticks every 2 s; the by-PID VRAM residency
is the definitive proof, and `train_steps_per_second 5.2` confirms the GPU did the work.

The full late-interaction path works end to end on CUDA, identical structure to MI355X:

```
[rank=0] INFO: LATE-INTERACTION query embedding shape: (32, 128) (ndim=2)
[rank=0] INFO: LATE-INTERACTION document[0] embedding shape: (35, 128) (ndim=2)
[rank=0] INFO: MAXSIM rank 1: score 31.1195 | doc[0] Once the path message reaches the exit ASBR, any choice of inter-AS TE
[rank=0] INFO: Relevant document is rank 1: True
[rank=0] INFO: RERANK rank 1: id 0 score 31.1195
[rank=0] INFO: ✅ Index with FastPlaid backend.
[rank=0] INFO: PLAID rank 1: id 0 score 31.1191
```

`final_model/` saved with the encoder `model.safetensors` **plus the ColBERT `1_Dense` /
`2_Dense` projection modules** (`modules.json` lists `pylate.models.Dense.Dense`) — a
genuine multi-vector save, reloadable with `pylate.models.ColBERT`. Query embeddings are
`(32,128)` ndim-2, documents are per-token 2-D arrays — the paradigm holds. MaxSim ranks
the relevant doc first via all three paths (raw `colbert_scores`, `rank.rerank`, and a
**PLAID / `fast-plaid` index**), with PLAID reproducing the exact MaxSim score to ~4
decimals (31.1191 vs 31.1195). **On this run PLAID returned all 3 documents at `k=3`**
(the MI355X toy-index run returned only 2); centroid pruning on a 3-doc index is
non-deterministic at that scale — either outcome is expected, see *Notes & quirks*.

Full captured evidence: **`smoke_h100.log`** in this folder (train stdout + the GPU-4
by-PID sampler).

**Quirks specific to the H100 run:**

- **`answerdotai/answerai-colbert-small-v1` was used** instead of the `bge-small-en-v1.5`
  default: it is a native ColBERT checkpoint (loads with `similarity_fn_name="MaxSim"`
  already set), giving the cleanest smoke without bootstrapping a random projection onto a
  plain BERT encoder. It was not cached; downloaded once (21 files, tiny) with the box
  proxy unset (`unset HTTP_PROXY HTTPS_PROXY …`, which the proxy 403s for huggingface.co).
  `bge-small-en-v1.5` (cached) also works — it just adds a random `Dense(384→128)` first.
- PyLate reads the checkpoint's native `(384,96)` projection, then **appends a
  `Dense(96→128)`** because the script requests `--embedding_dim 128` (registry default).
  Per-token output is therefore 128, not the checkpoint's 96 — expected given the config;
  pass `--embedding_dim 96` to keep the checkpoint's native width.
- **flash-attn was NOT installed** — this backbone is BERT-architecture and `sdpa` is
  already fast at this size, so flash-attn buys little; used `--attn_implementation sdpa`.
  On CUDA `--attn_implementation flash_attention_2` (with `pip install flash-attn`) and the
  fused `--scores_backend flash|lik` MaxSim kernels (`pylate[flash-maxsim]`/`pylate[lik]`)
  are available if wanted — the MI355X-only reason to avoid them (CUDA-only builds) is gone.
- **`--tf32` engages here** (`resolve_tf32` returns True when `torch.version.cuda is not
  None`) — it is a no-op/gated on ROCm. No crash; ran clean.
- **Multi-GPU was not exercised on H100** (the node's other GPUs were held by a co-tenant
  job). A 2- or 8-GPU pass would reuse the MI355X recipe unchanged — `torchrun --nproc_per_node=N --master_port=PORT
  … --gather_across_devices` — over NCCL instead of RCCL (the `nccl` backend string is the
  same), with `dataloader_drop_last` force-enabled under DDP (budget batch so
  `floor(train_rows/(batch×ranks)) > 0`). Nothing in the code is ROCm/CUDA-specific there.

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
- **PyLate's fused MaxSim kernels are CUDA-only.** `pylate[flash-maxsim]` and `pylate[lik]`
  are NVIDIA builds. Worse, PyLate's backend auto-detection tests `tensor.is_cuda`, which
  is **`True` on ROCm** — so `backend="auto"` could dispatch to a kernel that is not there.
  The script therefore sets `PYLATE_SCORES_BACKEND=torch` explicitly on HIP builds via
  `--scores_backend auto`. The pure-torch einsum path is what was verified.
- **PLAID prunes aggressively on tiny indexes.** With 3 documents indexed and `k=3`, the
  retriever returned 2 hits: centroid-based pruning discards candidates that share no
  centroid with the query, and there are far too few points for `fastkmeans` to build a
  meaningful centroid set. Use `--index_backend plaid` as a smoke check, and expect exact
  recall only at realistic corpus sizes (or use `rank.rerank`, which is exhaustive).
- **Never `pip install flash-attn`** — CUDA-only. `--attn_implementation auto` picks
  `sdpa` on ROCm.
- **`tf32=True` raises on ROCm.** `--tf32` is gated behind `torch.version.cuda is not None`
  and is silently a no-op on HIP.
- **Master port 29500 is often already in use on a shared box.** Use `--master_port 29830`
  (or any other free port).
- **MaxSim scores are unnormalized sums over query tokens**, so raw scores grow with query
  length and mildly favour longer documents. In one probe on an over-fit
  120×-replicated run, an 11-token positive lost to a 22-token hard negative. This is
  inherent to ColBERT scoring, not a ROCm or PyLate bug — compare scores only within one
  query, and do not read absolute magnitudes as calibrated relevance.
- **DDP forces `dataloader_drop_last=True`.** sentence-transformers overrides the flag
  under DDP to avoid hangs on an uneven last batch, and logs a warning. Budget your batch
  size so `floor(train_rows / (batch × ranks))` is still > 0 — the same trap documented in
  `../sentence_transformers`.
- **`warmup_ratio` is deprecated** in transformers v5 (warns, still works); switch to
  `warmup_steps` as a float when ST drops v4 support.
- **`destroy_process_group()` not called** — a benign torch shutdown warning at the end of
  the 2-GPU run; `rc=0` regardless.
- **Reload with `pylate.models.ColBERT`, not `SentenceTransformer`.** `final_model/`
  contains the encoder plus the `Dense` projection. Loading it with plain
  `SentenceTransformer` gives a pooled vector and silently discards the whole paradigm.
- **The shipped OTel sample is for pipeline validation only** — 100 rows produce no useful
  model. The 120×-replicated file used for the utilization run was written outside the
  repo and deliberately not committed; it is a pipeline proof, not a learning result.

## Summary

### 1. Does it work on MI355X? **Yes — works, no code changes.**

`pylate==1.6.0` on `torch==2.11.0+rocm7.2` trains ColBERT models on gfx950 unmodified.
Verified on 1 GPU and on 2 GPUs (DDP/RCCL, port 29830, 89–97% utilization on both cards,
7.9 GiB each), with the loss finite, the triplet evaluator running, checkpoints and
`final_model` saved, and the whole late-interaction path — encode → `colbert_scores` →
`rank.rerank` → PLAID index → `retrieve.ColBERT` — working end to end. The only ROCm
accommodations are the three the repo already knows: `sdpa` instead of flash-attn, `tf32`
gated off, and here additionally the MaxSim kernel pinned to `torch` because PyLate's
`flash`/`lik` kernels are CUDA-only. All three are handled automatically by the script.

### 2. Is it worth keeping? **Yes — keep it. This is a distinct capability, not a duplicate.**

The concern worth testing was that PyLate, being built on sentence-transformers, might be
a thin wrapper around the same objective the repo already has three times over. It is not,
and the evidence is structural rather than a matter of opinion:

- **The output shape is different.** Same backbone, same text: PyLate returns
  `(n_tokens, 128)` per document — measured at 27 to 159 rows depending on length —
  where `../sentence_transformers` returns a single `(384,)` vector. `ndim` 2 vs 1.
  No flag on any existing folder's script produces that.
- **The scoring function is different.** MaxSim (per-query-token max over document
  tokens, then sum) versus a dot product. The training loss optimizes MaxSim directly.
- **The serving artifacts are different.** PyLate ships PLAID / Voyager / ScaNN / WARP
  multi-vector indexes and a `retrieve`/`rank` API. Nothing in this repo can index or
  query a multi-vector corpus; there is no way to serve a ColBERT model with the existing
  code even if you somehow trained one.
- **It occupies a real gap on the cost/accuracy curve.** Bi-encoders are fast and less
  accurate; cross-encoders are accurate and cannot scan a corpus. Late interaction is the
  standard middle point, and the repo had zero coverage of it. It is also usable as a
  *reranker* (`rank.rerank`, no index) — a cheaper alternative to the `../../reranker/*`
  cross-encoders.

The costs are real and measured — 23.6× the index storage on this sample, a slower scoring
kernel, and a dependency stack that downgrades sentence-transformers — but they buy a
retrieval mode the repo genuinely cannot express today. **Recommendation: keep.**
