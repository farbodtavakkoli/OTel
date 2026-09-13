# `train_embedding_tevatron.py`

## Overview & when to use

**Tevatron** ([texttron/tevatron](https://github.com/texttron/tevatron), v2.0) is the
reference toolkit for *dense retrieval* — DPR, RepLLaMA, and the BEIR/MS MARCO pipeline
(train → shard-encode → search → TREC run). This folder wraps its
`tevatron.retriever.driver.*` entrypoints in a thin launcher that follows this repo's
script conventions, plus a converter for the repo's triplet JSONL.

It is a **retrieval-first** trainer, not a general embedding trainer: one contrastive
objective (InfoNCE over `train_group_size` passages per query, optionally gathered across
ranks), no MTEB/STS evaluation, no Matryoshka, no multi-task loss zoo.

Use this folder when you want the **retrieval pipeline** (train → encode a corpus in
shards → search → run file) or a **LoRA decoder-LM retriever**. Use
`../sentence_transformers` when you want a general-purpose embedding model with
evaluation baked in.

## Install

Python 3.12, in its own venv. If the root filesystem is tight, put it outside the repo on
a larger volume:

```
$DATA_DIR/envs/.env_tevatron
```

The commands below refer to a few machine-specific locations through environment
variables — set them to suit your machine:

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Tevatron on PyPI (`pip install tevatron`) is **0.1.0 — the 2021 v1 package**. It has no
`tevatron.retriever` module and no `--grad_cache` flag. Install from git.

### AMD (ROCm)

Install the ROCm torch wheel **first**, then everything else, then re-check that pip did
not swap in a CUDA wheel:

```bash
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
python3 -m venv $DATA_DIR/envs/.env_tevatron
source $DATA_DIR/envs/.env_tevatron/bin/activate
pip install -U pip setuptools wheel

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding_tevatron.txt
pip install git+https://github.com/luyug/GradCache.git
pip install --no-deps git+https://github.com/texttron/tevatron.git

python -c "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda)"
# 2.11.0+rocm7.2 7.2.26015 None      <- hip set, cuda None
```

`--no-deps` on the Tevatron install is required: its `setup.py` asks for
`transformers>=4.10.0, datasets>=1.1.3`, which would resolve the whole stack away from the
pinned versions.

If pip ever replaces torch with the CUDA build (it can, depending on resolution order):

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

**Never `pip install flash-attn` here** — it is a CUDA-only build. Tevatron's
`ModelArguments.attn_implementation` **defaults to `flash_attention_2`**, so you must pass
`--attn_implementation sdpa` on every command (the launcher does this for you).

### NVIDIA (CUDA)

Same, minus the ROCm index. On a CUDA 13 box `pip install torch==2.11.0` resolves a
**native CUDA 13 wheel** (`2.11.0+cu130`) straight from PyPI — no `--index-url` needed:

```bash
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
python3 -m venv $DATA_DIR/envs/.env_tevatron
source $DATA_DIR/envs/.env_tevatron/bin/activate
pip install -U pip setuptools wheel
pip install torch==2.11.0 numpy          # -> 2.11.0+cu130, nvidia-*-cu13 deps
pip install -r requirements_embedding_tevatron.txt   # minus torch/numpy already satisfied
# GradCache + Tevatron: either the git+https form above, or clone and install locally
git clone --depth 1 https://github.com/luyug/GradCache.git   && pip install ./GradCache
git clone --depth 1 https://github.com/texttron/tevatron.git && pip install --no-deps ./tevatron
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0+cu130 13.0
```

Use `--attn_implementation sdpa` for the BERT backbones (transformers' FA2 path needs the
`flash-attn` package and gives BERT nothing). Reserve `flash_attention_2` +
`pip install flash-attn` for the Qwen `--pooling eos` LoRA recipe.

A BERT backbone logs `embeddings.position_ids UNEXPECTED` at load — benign. If Hub egress
is blocked, `google/embeddinggemma-300m` is a drop-in fallback with `--pooling mean`; with
a populated `HF_HOME` set `HF_HUB_OFFLINE=1`. Point `HF_DATASETS_CACHE` at tmpfs
(`/dev/shm/dscache_tevatron`) if the datasets cache lands on a slow or shared mount.

## Environment & secrets

`dev.env` is a symlink to the repo-root file and is loaded by `load_dotenv("dev.env")`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Only needed for gated models — `BAAI/bge-small-en-v1.5` and `Qwen/Qwen3-0.6B` are not
gated. Point the model cache at a shared cache directory:

```bash
export HF_HOME=/path/to/hf_cache
export HIP_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=2,3
```

## Data

Tevatron's `TrainDataset` accepts two formats:

1. **Self-contained (used here)** — `query` + inline `positive_passages` /
   `negative_passages`, each a `{docid, text}` object with an optional `title`. Needs no
   corpus file.
2. **Corpus-referencing** — `query_id`/`query_text` + `positive_document_ids` /
   `negative_document_ids`, resolved against a separate `--corpus_path` JSONL of
   `{docid, text}`.

`convert_data.py` emits format 1 from the repo's triplet schema:

```bash
python convert_data.py \
  --input_file ../sentence_transformers/OTel_embedding_sample_100.jsonl \
  --output_file OTel_tevatron_sample_100.jsonl
```

| Flag | Default | Meaning |
|---|---|---|
| `--input_file` | `../sentence_transformers/OTel_embedding_sample_100.jsonl` | Source JSONL with `anchor` / `positive` / `negative_1..N` columns |
| `--output_file` | `OTel_tevatron_sample_100.jsonl` | Destination JSONL in Tevatron format |
| `--anchor_field` | `anchor` | Column holding the query text |
| `--positive_field` | `positive` | Column holding the relevant passage |
| `--negative_prefix` | `negative_` | Prefix of the hard-negative columns |
| `--max_negatives` | `5` | Keep at most this many hard negatives per row |
| `--limit` | `None` | Convert only the first N rows |
| `--docid_prefix` | `otel` | Prefix for the generated `docid` values |

Two conversion rules matter when you point it at your own data. Negatives are taken in
**numeric** suffix order — `negative_2` before `negative_10`, not lexicographic — and empty
ones are dropped, so a row can yield fewer than `--max_negatives` passages. A row whose
`anchor` or `positive` is missing or blank is skipped entirely; the closing line prints the
written and skipped counts, so check it against your source row count.

Shipped sample: **`OTel_tevatron_sample_100.jsonl`** (100 rows, 296 KB) — the default
`--dataset_path`. One row:

```json
{"query_id": "q0", "query": "...",
 "positive_passages": [{"docid": "otel-0-pos", "text": "..."}],
 "negative_passages": [{"docid": "otel-0-neg0", "text": "..."}, ...]}
```

`title` is omitted on purpose: the reader does `title + ' ' + text` whenever the key is
present, so an empty title would prepend a stray space to every passage.

## Run

`train_embedding_tevatron.py` is a **launcher**: it resolves the attention kernel, counts
the visible GPUs, picks `python` vs `torchrun --master_port 29820`, clears a stale output
dir, prints the effective batch geometry, and execs
`tevatron.retriever.driver.train`. `--dry_run` prints the command without running it.

Smoke test on the shipped sample (single GPU):

```bash
source $DATA_DIR/envs/.env_tevatron/bin/activate

python train_embedding_tevatron.py \
  --devices 2 --model_name_or_path BAAI/bge-small-en-v1.5 \
  --batch_size 8 --train_group_size 6 --epochs 1 \
  --output_dir $OUTPUT_DIR/train_embedding_tevatron/smoke_1gpu --overwrite
```

Two GPUs (the launcher switches to `torchrun` automatically):

```bash
python train_embedding_tevatron.py --devices 2,3 \
  --batch_size 8 --train_group_size 6 --epochs 8 --save_strategy no \
  --output_dir $OUTPUT_DIR/train_embedding_tevatron/smoke_2gpu --overwrite
```

GradCache + LoRA on a decoder LM:

```bash
python train_embedding_tevatron.py --devices 2,3 \
  --model_name_or_path Qwen/Qwen3-0.6B --pooling eos --append_eos_token \
  --lora --lr 1e-4 \
  --grad_cache --gc_q_chunk_size 8 --gc_p_chunk_size 16 \
  --batch_size 32 --train_group_size 8 --epochs 30 --save_strategy no \
  --output_dir $OUTPUT_DIR/train_embedding_tevatron/gc_2gpu --overwrite
```

Full run — point `--dataset_path` at your own converted JSONL:

```bash
nohup python train_embedding_tevatron.py --devices 2,3 \
  --dataset_path /path/to/your_train.jsonl \
  --model_name_or_path Qwen/Qwen3-0.6B --pooling eos --append_eos_token --lora --lr 1e-4 \
  --grad_cache --gc_q_chunk_size 8 --gc_p_chunk_size 16 \
  --batch_size 64 --train_group_size 16 --epochs 1 \
  --output_dir $OUTPUT_DIR/train_embedding_tevatron/run \
  > train_embedding_tevatron.log 2>&1 &

tail -f train_embedding_tevatron.log
```

### After training: encode and search

Not wrapped by the launcher — these are the upstream drivers. Corpus encoding shards
across GPUs:

```bash
# queries
python -m tevatron.retriever.driver.encode --output_dir tmp \
  --model_name_or_path $TRAINED --attn_implementation sdpa --pooling cls --normalize --bf16 \
  --encode_is_query --dataset_name json --dataset_path queries.jsonl \
  --per_device_eval_batch_size 128 --encode_output_path emb/query.pkl

# corpus, one shard per GPU
for s in 0 1; do
  HIP_VISIBLE_DEVICES=$((s+2)) CUDA_VISIBLE_DEVICES=$((s+2)) \
  python -m tevatron.retriever.driver.encode --output_dir tmp \
    --model_name_or_path $TRAINED --attn_implementation sdpa --pooling cls --normalize --bf16 \
    --dataset_name json --dataset_path corpus.jsonl \
    --dataset_number_of_shards 2 --dataset_shard_index $s \
    --per_device_eval_batch_size 128 --encode_output_path emb/corpus.$s.pkl &
done; wait

set -f && python -m tevatron.retriever.driver.search \
  --query_reps emb/query.pkl --passage_reps 'emb/corpus*.pkl' \
  --depth 100 --batch_size 64 --save_text --save_ranking_to emb/run.txt
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name_or_path` | `BAAI/bge-small-en-v1.5` | Backbone model id or path |
| `--pooling` | `cls` | `cls` \| `mean` \| `eos` \| `last` — `eos` for decoder LMs |
| `--normalize` / `--no_normalize` | on | L2-normalize vectors (cosine) vs raw dot product |
| `--append_eos_token` | off | Append EOS to query/passage — required by the eos-pooling recipe |
| `--temperature` | `0.02` | Softmax temperature of the contrastive loss |
| `--attn_implementation` | `auto` | `auto` ⇒ sdpa on ROCm, flash_attention_2 on CUDA |
| `--dataset_name` | `json` | HF dataset name, or `json` to read `--dataset_path` |
| `--dataset_path` | `OTel_tevatron_sample_100.jsonl` | Local training JSONL (Tevatron format) |
| `--dataset_config` | `None` | HF dataset config name |
| `--dataset_split` | `train` | Split to train on |
| `--corpus_name` / `--corpus_path` | `None` | Corpus source for the docid-referencing data format |
| `--query_prefix` / `--passage_prefix` | `None` | Instruction prefixes prepended at tokenization |
| `--query_max_len` | `64` | Query truncation length |
| `--passage_max_len` | `192` | Passage truncation length |
| `--train_group_size` | `6` | Passages per query: 1 positive + n−1 hard negatives |
| `--output_dir` | `outputs/tevatron_run` | Checkpoints and the final encoder |
| `--overwrite` | off | Delete a non-empty `--output_dir` first (see Notes) |
| `--batch_size` | `8` | Per-device train batch size, in queries |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `--epochs` | `1` | Training epochs |
| `--max_steps` | `None` | Hard cap on optimizer steps |
| `--lr` | `1e-5` | Learning rate — use ~`1e-4` with `--lora` |
| `--lr_scheduler_type` | `None` | LR schedule, e.g. `linear`, `cosine` |
| `--warmup_ratio` | `0.1` | Warmup fraction of total steps |
| `--seed` | `42` | Random seed |
| `--bf16` / `--fp16` | bf16 on | Training precision |
| `--gradient_checkpointing` | off | Recompute activations (composes with GradCache) |
| `--dataloader_num_workers` | `None` | Dataloader worker processes |
| `--logging_steps` | `1` | Steps between loss lines |
| `--save_strategy` | `epoch` | `no` \| `steps` \| `epoch` |
| `--save_steps` | `None` | Steps between checkpoints when `--save_strategy steps` |
| `--report_to` | `none` | Trainer reporting integration, e.g. `tensorboard` |
| `--deepspeed` | `None` | Path to a DeepSpeed config JSON |
| `--grad_cache` | off | **GradCache** — chunked re-forward; batch size stops being memory-bound |
| `--gc_q_chunk_size` | `8` | Queries per GradCache sub-forward |
| `--gc_p_chunk_size` | `16` | Passages per GradCache sub-forward |
| `--lora` | off | LoRA fine-tuning (needs `peft`) |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | `16` / `64` / `0.1` | LoRA hyperparameters |
| `--lora_target_modules` | `q_proj,...,gate_proj` | Comma-separated LoRA target modules |
| `--devices` | `None` | GPU ids for `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, e.g. `2,3` |
| `--master_port` | `29820` | torchrun rendezvous port (29500 often collides on a shared box) |
| `--dry_run` | off | Print the resolved command and exit |

## Output

Under `--output_dir`:

- `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` — the
  **bare encoder**, reloadable with `AutoModel.from_pretrained(...)`. Tevatron's
  `_save` strips the `encoder.` prefix, so the artifact is a plain HF model, not a
  Tevatron wrapper.
- `training_args.bin`
- `checkpoint-<step>/` per `--save_strategy`.
- With `--lora`, an adapter (`adapter_model.safetensors` + `adapter_config.json`); pass it
  back to the encode driver via `--lora --lora_name_or_path <dir>`.

There is **no evaluation output** — no IR metrics, no MTEB, no best-checkpoint selection.
Quality is measured by running the encode → search drivers and scoring the run file
yourself (e.g. with `pytrec_eval`).

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 2 GPUs |
| PyTorch | `torch==2.11.0` resolves to `+cu130` on plain PyPI | `torch==2.11.0` from `download.pytorch.org/whl/rocm7.2` |
| Attention | `--attn_implementation sdpa` for BERT encoders; `flash_attention_2` is usable for decoder-LMs with `pip install flash-attn` | `--attn_implementation sdpa` only |
| Exercised | single-GPU SFT smoke | SFT, GradCache, LoRA, and 2-GPU DDP with the cross-rank negative gather |

**No Tevatron source changes are needed on either vendor** — the only argument changes are
`--attn_implementation sdpa` and dropping `--overwrite_output_dir` (use the launcher's
`--overwrite`).

### Expected output

Single GPU, `BAAI/bge-small-en-v1.5`, shipped 100-row sample, `--batch_size 8
--train_group_size 6 --epochs 4`. `drop_last` keeps 96 rows, so 96/8 = **13 steps/epoch x 4
= 52 optimizer steps**:

```text
{'loss': '0.4837', 'grad_norm': '19.38', 'learning_rate': '0', 'epoch': '0.07692'}
{'loss': '0.3523', 'grad_norm': '15.56', 'learning_rate': '4.348e-07', 'epoch': '3.923'}
{'loss': '0.0755', 'grad_norm': '5.906', 'learning_rate': '2.174e-07', 'epoch': '4'}
{'train_runtime': '12.27', 'train_samples_per_second': '32.6', 'train_loss': '0.8935', 'epoch': '4'}
INFO - training/embedding/tevatron - training finished; encoder written to ...
```

Per-step loss is noisy on 100 rows with 8-query batches; read the epoch means. The final
encoder reloads via `AutoModel.from_pretrained`; `--save_strategy epoch` also writes
`checkpoint-*` directories.

### GradCache

Without `--grad_cache`, activation memory scales linearly with `--batch_size`: for
`Qwen/Qwen3-0.6B` + LoRA at `--train_group_size 8 --passage_max_len 192` it OOMs on an
80 GB card above `--batch_size 16`. **Turn `--grad_cache` on whenever the per-device batch
exceeds 16**; memory then stays flat regardless of batch size, at the cost of a second
forward pass per step.

### Multi-GPU

Pass several devices (`--devices 0,1`) and the launcher flips to
`torchrun --nproc_per_node N --master_port <port>`; a single device runs plain `python`.
Under DDP the loss becomes `DistributedContrastiveLoss`, so the candidate pool is gathered
across ranks. GradCache, LoRA and DDP compose.

## Notes

- **Do not install `tevatron` from PyPI.** Version 0.1.0 is the 2021 v1 package — no
  `tevatron.retriever`, no `--grad_cache`. Install from git.
- **`--attn_implementation sdpa` is mandatory on ROCm.** `ModelArguments` defaults to
  `flash_attention_2` and flash-attn has no ROCm build here. The launcher's
  `--attn_implementation auto` resolves this from `torch.version.hip`.
- **`--overwrite_output_dir` no longer parses.** transformers 5.x removed it, so
  `HfArgumentParser` rejects the flag; the driver still reads
  `training_args.overwrite_output_dir` when the output dir is non-empty, which would raise
  `AttributeError`. The launcher clears the directory itself under `--overwrite`.
- **`warmup_ratio is deprecated and will be removed in v5.2`** — emitted by
  `TevatronTrainingArguments`, which redeclares the field. Harmless today; upstream will
  have to move to `warmup_steps`.
- **No `tf32` anywhere.** Tevatron never sets it, so the usual ROCm
  `tf32=True` crash does not apply here.
- **Port 29820** is the launcher default; 29500 often collides on a shared box.
- **`destroy_process_group() was not called`** warning at the end of every multi-GPU run —
  upstream never tears the group down. Cosmetic.
- **Loss goes *up* with batch size.** Expected, not a bug: a larger in-batch pool makes the
  InfoNCE softmax harder. Compare losses only at equal batch geometry.
- **`train_group_size` must fit the data.** The shipped sample has 5 negatives per query,
  so `--train_group_size 6` uses each exactly once; anything larger makes the reader sample
  negatives with replacement (`random.choices`), duplicating them inside the group.
- **BERT `embeddings.position_ids UNEXPECTED`** in the load report for
  `bge-small-en-v1.5` — a stale buffer in the checkpoint, ignorable.
- **`pillow` is a hard import** of `tevatron.retriever.dataset` even for pure-text
  retrieval; it is in the requirements file for that reason.
- **The shipped sample is a pipeline check, not a training result.** 100 rows of OTel/paper
  text produce no useful retriever.
