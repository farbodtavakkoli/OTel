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

**What it adds over the sentence-transformers sibling (`../sentence_transformers`):**

| Capability | `../sentence_transformers` | `training/embedding/tevatron` |
|---|---|---|
| GradCache-style large batches | `CachedMultipleNegativesRankingLoss` | `--grad_cache` + `--gc_q_chunk_size` / `--gc_p_chunk_size` |
| Cross-device negative gather | `gather_across_devices=True` | automatic under DDP (`DistributedContrastiveLoss`) |
| LoRA on a decoder-LM retriever | not wired | `--lora --lora_target_modules ...` (RepLLaMA recipe) |
| Sharded multi-GPU corpus encoding | not provided | `driver.encode` + `--dataset_number_of_shards/--dataset_shard_index` |
| Flat-index search → TREC run | not provided | `driver.search --save_text --save_ranking_to` |
| Evaluation during training | IR + MTEB evaluators, best-checkpoint reload | **none** — train only |
| Pooling choices | ST module config | `--pooling cls\|mean\|eos\|last` |

Use this folder when you want the **retrieval pipeline** (train → encode a corpus in
shards → search → run file) or a **LoRA decoder-LM retriever**. Use
`../sentence_transformers` when you want a general-purpose embedding model with
evaluation baked in.

### GradCache vs `CachedMultipleNegativesRankingLoss` — the honest take

They are the **same trick**: forward without grad in chunks, cache the representations,
compute the loss on the full similarity matrix, then re-forward chunk-by-chunk to get the
real gradients. sentence-transformers'
`CachedMultipleNegativesRankingLoss` is a reimplementation of Luyu Gao's GradCache; Tevatron
calls the original `grad_cache` package. The memory/quality behaviour is equivalent, so
**"Tevatron gives you GradCache and the sibling folder does not" is false** for a current
sentence-transformers install.

Two differences that are real but narrow:

- Tevatron's GradCache path is a **Trainer subclass** (`GradCacheTrainer` swaps
  `training_step`), so it composes with `--lora`, DeepSpeed, and its own DDP gather without
  a loss-class change. In sentence-transformers you swap the loss class instead.
- Tevatron exposes **separate query and passage chunk sizes** (`--gc_q_chunk_size`,
  `--gc_p_chunk_size`), which matters when queries are short and passages are long — the
  cached ST loss uses one `mini_batch_size` for both.

Neither is a capability gap. See **Verdict** at the bottom.

## Install

Python 3.12. **The venv lives outside the repo** — `/` on this host is at 94% and the
existing venvs already total >400 GB:

```
/mnt/data_450g/envs/.env_train_embedding_tevatron
```

> This venv survived the 2026-08 repo reorg (it lives outside the repo); the in-repo
> campaign venvs were removed in that reorg.

Tevatron on PyPI (`pip install tevatron`) is **0.1.0 — the 2021 v1 package**. It has no
`tevatron.retriever` module and no `--grad_cache` flag. Install from git.

### AMD (ROCm) — the verified route

Install the ROCm torch wheel **first**, then everything else, then re-check that pip did
not swap in a CUDA wheel:

```bash
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache
python3 -m venv /mnt/data_450g/envs/.env_train_embedding_tevatron
source /mnt/data_450g/envs/.env_train_embedding_tevatron/bin/activate
pip install -U pip setuptools wheel

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding_tevatron.txt
pip install git+https://github.com/luyug/GradCache.git
pip install --no-deps git+https://github.com/texttron/tevatron.git

python -c "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda)"
# 2.11.0+rocm7.2 7.2.26015 None      <- hip set, cuda None
```

`--no-deps` on the Tevatron install is deliberate: its `setup.py` asks for
`transformers>=4.10.0, datasets>=1.1.3`, which would let pip resolve the whole stack away
from the pinned versions. Nothing else in Tevatron's dependency set is missing from
`requirements_embedding_tevatron.txt`.

If pip ever replaces torch with the CUDA build (it did for other folders on this host):

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

**Never `pip install flash-attn` here** — it is a CUDA-only build. Tevatron's
`ModelArguments.attn_implementation` **defaults to `flash_attention_2`**, so you must pass
`--attn_implementation sdpa` on every command (the launcher does this for you).

### NVIDIA (CUDA) — verified on H100 (see the H100 section below)

Same, minus the ROCm index. On the H100 box `pip install torch==2.11.0` resolves a
**native CUDA 13 wheel** (`2.11.0+cu130`) straight from PyPI — no `--index-url` needed:

```bash
export PIP_CACHE_DIR=/dev/shm/h100/pipcache
python3 -m venv /dev/shm/h100/venv_tevatron
source /dev/shm/h100/venv_tevatron/bin/activate
pip install -U pip setuptools wheel
pip install torch==2.11.0 numpy          # -> 2.11.0+cu130, nvidia-*-cu13 deps
pip install -r requirements_embedding_tevatron.txt   # minus torch/numpy already satisfied
pip install <local GradCache clone>      # git clone https://github.com/luyug/GradCache.git
pip install --no-deps <local tevatron clone>   # git clone https://github.com/texttron/tevatron.git @ dd06310
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0+cu130 13.0
```

`flash_attention_2` is *usable* on H100 for decoder-LMs, but **`BAAI/bge-small-en-v1.5` is
a BERT encoder** — transformers' FA2 path needs the `flash-attn` package and gives BERT no
benefit, so the single-GPU smoke uses `--attn_implementation sdpa` (same as ROCm). Reserve
`flash_attention_2` + `pip install flash-attn` for the Qwen `--pooling eos` LoRA recipe.

## Environment & secrets

`dev.env` is a symlink to the repo-root file and is loaded by `load_dotenv("dev.env")`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Only needed for gated models — `BAAI/bge-small-en-v1.5` and `Qwen/Qwen3-0.6B` are not
gated. Point the model cache at the shared one:

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache
export HIP_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=2,3
```

## Data

Tevatron's `TrainDataset` accepts two formats. Verified against the current reader
(`src/tevatron/retriever/dataset.py`, commit `dd06310`):

1. **Self-contained (used here)** — `query` + inline `positive_passages` /
   `negative_passages`, each a `{docid, text}` object with an optional `title`. The reader
   branches on `if 'positive_passages' in group:` and needs no corpus file.
2. **Corpus-referencing** — `query_id`/`query_text` + `positive_document_ids` /
   `negative_document_ids`, resolved against a separate `--corpus_path` JSONL of
   `{docid, text}`. This is what the upstream README documents; it exists for the
   multi-modal datasets.

`convert_data.py` emits format 1 from the repo's triplet schema:

```bash
python convert_data.py \
  --input_file ../sentence_transformers/OTel_embedding_sample_100.jsonl \
  --output_file OTel_tevatron_sample_100.jsonl
```

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
source /mnt/data_450g/envs/.env_train_embedding_tevatron/bin/activate
export HF_HOME=/mnt/data_1.5t/hf_cache

python train_embedding_tevatron.py \
  --devices 2 --model_name_or_path BAAI/bge-small-en-v1.5 \
  --batch_size 8 --train_group_size 6 --epochs 1 \
  --output_dir /mnt/data_450g/outputs/train_embedding_tevatron/smoke_1gpu --overwrite
```

Two GPUs (the launcher switches to `torchrun` automatically):

```bash
python train_embedding_tevatron.py --devices 2,3 \
  --batch_size 8 --train_group_size 6 --epochs 8 --save_strategy no \
  --output_dir /mnt/data_450g/outputs/train_embedding_tevatron/smoke_2gpu --overwrite
```

GradCache — the reason this folder exists. `--batch_size` is no longer bounded by
activation memory:

```bash
python train_embedding_tevatron.py --devices 2,3 \
  --model_name_or_path Qwen/Qwen3-0.6B --pooling eos --append_eos_token \
  --lora --lr 1e-4 \
  --grad_cache --gc_q_chunk_size 8 --gc_p_chunk_size 16 \
  --batch_size 32 --train_group_size 8 --epochs 30 --save_strategy no \
  --output_dir /mnt/data_450g/outputs/train_embedding_tevatron/gc_2gpu --overwrite
```

Full run — point `--dataset_path` at your own converted JSONL:

```bash
nohup python train_embedding_tevatron.py --devices 2,3 \
  --dataset_path /path/to/your_train.jsonl \
  --model_name_or_path Qwen/Qwen3-0.6B --pooling eos --append_eos_token --lora --lr 1e-4 \
  --grad_cache --gc_q_chunk_size 8 --gc_p_chunk_size 16 \
  --batch_size 64 --train_group_size 16 --epochs 1 \
  --output_dir /mnt/data_450g/outputs/train_embedding_tevatron/run \
  > train_embedding_tevatron.log 2>&1 &

tail -f train_embedding_tevatron.log
```

### After training: encode and search

Not wrapped by the launcher — these are the upstream drivers, and they are the part
`../sentence_transformers` has no answer for. Corpus encoding shards across GPUs:

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
| `--master_port` | `29820` | torchrun rendezvous port (29500 collides on this host) |
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

## Hardware support & evidence

- **NVIDIA:** upstream's target. **Tested** — 1× H100 80GB HBM3 (Hopper cc9.0), CUDA 13.0,
  driver 580.173.02, Python 3.12.3, `torch==2.11.0+cu130` (native CUDA-13 PyPI wheel),
  `transformers==5.5.0`, `datasets==4.3.0`, `accelerate==1.14.0`, `peft==0.20.0`,
  `GradCache==0.1.0`, `faiss-cpu==1.13.0`, Tevatron at git `dd06310`. Single-GPU smoke
  verified 2026-08-22 (see the H100 section below).
- **Other hardware (upstream claims — not verified here):** Google **TPU** — Tevatron's
  paper and docs claim TPU training via the JAX/Flax path (`tevax` /
  `tevatron.driver.jax_train`, GradCache included), from the v1 era. The v2
  `tevatron.retriever` PyTorch path used here claims nothing beyond NVIDIA/AMD.
- **AMD:** **tested** — 1× and 2× MI355X (gfx950, 288 GB), ROCm 7.2.4, Python 3.12.3,
  `torch==2.11.0+rocm7.2`, `transformers==5.5.0`, `datasets==4.3.0`,
  `accelerate==1.14.0`, `peft==0.20.0`, `GradCache==0.1.0`, `faiss-cpu==1.13.0`,
  Tevatron at git `dd06310`. Verified 2026-08-20.

### Tested on AMD Instinct MI355X — ROCm 7.2 (verified 2026-08-20)

**Verdict: works, with two argument changes** (`--attn_implementation sdpa`, and drop
`--overwrite_output_dir`). No Tevatron source changes. GradCache, LoRA, DDP with the
cross-rank negative gather, and the model save all behave as documented.

Note on this host: GPUs 2 and 3 are **shared with a resident vLLM/SGLang tenant** holding
~213 GiB of each 288 GB card, so the memory budget in every number below is **~74 GiB per
GPU**, not 288 GB. That makes the OOM boundary below tighter than it would be on an idle
card — the *shape* of the result (flat vs linear memory) is what matters.

**1) Single GPU** — `BAAI/bge-small-en-v1.5`, shipped sample, `--batch_size 8
--train_group_size 6`, 13 steps:

```
{'loss': '1.093', 'grad_norm': '28', 'learning_rate': '5e-06', 'epoch': '0.1538'}
{'loss': '0.336', 'grad_norm': '15.25', 'learning_rate': '8.182e-06', 'epoch': '0.3846'}
{'train_runtime': '6.195', 'train_samples_per_second': '16.14', 'train_loss': '0.9597', 'epoch': '1'}
INFO - training/embedding/tevatron - training finished; encoder written to .../launcher_1gpu
```

`model.safetensors` (66.7 MB) + `config.json` + tokenizer written; `checkpoint-13/` from
`--save_strategy epoch`. The launcher reproduces the raw-CLI run's `train_loss` exactly.

**2) Two GPUs** — same model, `torchrun --nproc_per_node=2 --master_port=29820`, 8 epochs,
56 steps: `train_loss 0.9167`, `92.2 samples/s`, rc=0.

**3) GradCache** — `Qwen/Qwen3-0.6B` + LoRA, `--pooling eos --append_eos_token`,
`--train_group_size 8`, `--passage_max_len 192`, single GPU, 1 epoch,
`--skip_memory_metrics False` for peak-VRAM numbers:

| `--batch_size` | in-batch passages | no GradCache | with GradCache (q=8, p=16) |
|---|---|---|---|
| 8 | 64 | 31.27 GiB, 12.0 s | 7.65 GiB, 22.4 s |
| 16 | 128 | 62.49 GiB, 10.4 s | 7.65 GiB, 20.7 s |
| 32 | 256 | **OOM** (`torch.OutOfMemoryError`) | 7.65 GiB, 23.6 s |
| 64 | 512 | **OOM** | 7.65 GiB, 24.6 s |
| 128 | 1024 | **OOM** | 7.59 GiB, 22.4 s |

This is the headline result: **without GradCache the peak scales linearly with batch and
dies at 32; with GradCache it is flat at ~7.6 GiB from 8 all the way to 128.** GradCache
unlocked **8× the batch size (16 → 128) at 12% of the memory**, i.e. 1024 in-batch
passages per query instead of 128. The cost is ~2× wall-clock per step, exactly the
expected double-forward penalty.

**4) GradCache + LoRA + 2-GPU DDP together** — `--batch_size 32 --train_group_size 8`,
30 epochs, 60 steps. Under DDP the loss is `DistributedContrastiveLoss`, so the candidate
pool is gathered across both ranks: 32 × 2 = **64 queries and 512 passages per update**,
every query scored against all 512.

```
{'loss': '5.35',  'grad_norm': '1.959', 'learning_rate': '4.259e-05', 'epoch': '19'}
{'loss': '5.231', 'grad_norm': '2.39',  'learning_rate': '2.407e-05', 'epoch': '24'}
{'loss': '5.144', 'grad_norm': '1.928', 'learning_rate': '1.852e-06', 'epoch': '30'}
{'train_runtime': '285.9', 'train_samples_per_second': '10.49', 'train_loss': '5.541', 'epoch': '30'}
```

Monotone decreasing, `train_mem_gpu_peaked_delta` 7.65 GiB per rank.

`rocm-smi` sampled every 4 s from outside the run (`GPU[2],GPU[3]` busy % and total VRAM
used, which includes the co-tenant's ~213 GiB baseline):

```
07:32:35   0,0  | vram_used_GiB 212.8 213.6     <- before launch (co-tenant only)
07:33:54  95,94 | vram_used_GiB 226.0 226.8
07:35:18  91,96 | vram_used_GiB 226.0 226.8
07:36:41  99,96 | vram_used_GiB 226.0 227.0
07:39:40   0,0  | vram_used_GiB 212.8 213.6     <- after exit
```

Both assigned GPUs sit at **91–99% busy** for the whole run; the delta over the idle
baseline is ~13.2 GiB per card, matching the 7.65 GiB peak plus allocator reserve.

### Tested on NVIDIA H100 80GB — CUDA 13.0 (verified 2026-08-22)

**Verdict: works, with the same one argument change as ROCm** (`--attn_implementation sdpa`
for the BERT backbone). Single-GPU smoke only; multi-GPU deferred (box was co-tenanted — see
below). Ran on physical GPU 4 (`CUDA_VISIBLE_DEVICES=4`) of a shared 8×H100 node.

**Install that worked** (venv on tmpfs; `/` is tight on this box too):

```bash
export PIP_CACHE_DIR=/dev/shm/h100/pipcache HF_HOME=/mnt/gsma/gsma/gsma/models
python3 -m venv /dev/shm/h100/venv_tevatron && source /dev/shm/h100/venv_tevatron/bin/activate
pip install -U pip setuptools wheel
pip install torch==2.11.0 numpy          # PyPI ships a cu130 wheel for the 2.11.0 pin — no --index-url
pip install transformers==5.5.0 datasets==4.3.0 accelerate==1.14.0 peft==0.20.0 \
            faiss-cpu==1.13.0 pillow==12.3.0 python-dotenv==1.2.2 tensorboard==2.21.0
# GradCache + Tevatron: git clone each, then install from the local path
git clone --depth 1 https://github.com/luyug/GradCache.git   && pip install ./GradCache
git clone --depth 1 https://github.com/texttron/tevatron.git && pip install --no-deps ./tevatron   # -> dd06310
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
# 2.11.0+cu130 13.0 NVIDIA H100 80GB HBM3    <- cuda set, hip None; not clobbered by the git installs
```

Key versions: torch **2.11.0+cu130** / CUDA **13.0**, transformers 5.5.0, datasets 4.3.0,
accelerate 1.14.0, peft 0.20.0, GradCache 0.1.0, Tevatron @ dd06310, driver **580.173.02**.
No torch-clobber occurred — the two git installs left `2.11.0+cu130` intact (re-verified).

**Model:** the documented default `BAAI/bge-small-en-v1.5` was **not** in the shared cache, so
it was downloaded once (~130 MB; egress needs the box proxy unset) into `HF_HOME`. It is a
BERT encoder (33.4 M params) — the `embeddings.position_ids UNEXPECTED` load note is benign.
`google/embeddinggemma-300m` is cached and is a drop-in fallback (`--pooling mean`) if egress
is unavailable.

**Exact smoke command** (the launcher picks plain `python` for a single GPU, so no torchrun):

```bash
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 \
       HF_DATASETS_CACHE=/dev/shm/h100/dscache_tevatron CUDA_VISIBLE_DEVICES=4
python train_embedding_tevatron.py \
  --devices 4 --model_name_or_path BAAI/bge-small-en-v1.5 \
  --attn_implementation sdpa \
  --batch_size 8 --train_group_size 6 --epochs 4 \
  --output_dir /dev/shm/h100/out/tevatron/smoke_1gpu --overwrite
```

100 rows / (8 per-device × 1 GPU × 1 accum) = **13 steps/epoch × 4 = 52 optimizer steps**
(`drop_last` keeps 96 rows). Real output:

```text
{'loss': '0.4837', 'grad_norm': '19.38', 'learning_rate': '0', 'epoch': '0.07692'}
...
{'loss': '0.3523', 'grad_norm': '15.56', 'learning_rate': '4.348e-07', 'epoch': '3.923'}
{'loss': '0.0755', 'grad_norm': '5.906', 'learning_rate': '2.174e-07', 'epoch': '4'}
{'train_runtime': '12.27', 'train_samples_per_second': '32.6', 'train_steps_per_second': '4.238', 'train_loss': '0.8935', 'epoch': '4'}
```

Per-step loss is noisy on 100 rows / 8-query batches (identical behaviour to the MI355X run);
the trend is unambiguously **decreasing** — epoch-mean loss 0.948 → 0.916 → 0.893 → 0.826,
per-epoch min 0.350 → 0.075, final `train_loss 0.8935`, rc=0. Final encoder saved and reloads
via `AutoModel.from_pretrained` (`model.safetensors` 66.7 MB, matching the ROCm artifact).

GPU-4 residency, sampled by PID from **inside** the run (`nvidia-smi --query-compute-apps`):

```text
GPU4 UUID: GPU-e13d18b6-ccfb-6676-668a-cd489ad01b55
[04:12:19] util=14 % mem=3267 MiB | PID_on_GPU4: GPU-e13d18b6..., 1572032, python, 3120 MiB
[04:12:23] util=22 % mem=3291 MiB | PID_on_GPU4: GPU-e13d18b6..., 1572032, python, 3282 MiB
```

The training PID sits on GPU 4's UUID at ~3.3 GB (a 33 M-param BERT on 96 rows is light and
finishes in ~12 s, so utilisation stays modest). GPUs 0–3 were a co-tenant production job and
were never touched. Epoch checkpoints (`--save_strategy epoch`, 191 MB each) were deleted after
capturing evidence.

**Quirks / deviations from the MI355X recipe:**
- `torch==2.11.0` needs **no** `--index-url` on H100 — the plain PyPI wheel is already
  `+cu130`. (The MI355X route needs `--index-url .../rocm7.2`.)
- Same `--attn_implementation sdpa` as ROCm, but for a different reason: not a flash-attn gap,
  just that the BERT backbone gains nothing from FA2. `flash-attn` was **not** installed.
- Drop the ROCm `HIP_VISIBLE_DEVICES` var; plain `CUDA_VISIBLE_DEVICES=4` is enough (the
  launcher sets both anyway). tf32: torch default `allow_tf32=False`; bf16 (`--bf16`) is on and
  real bf16 matmul was confirmed on-GPU.
- VRAM is 80 GB here vs 288 GB on MI355X — irrelevant at this model size (peaked ~3.3 GB); it
  matters only for the Qwen/LoRA and GradCache recipes, where `--grad_cache` / offload apply.

**Multi-GPU (deferred):** a 2- or 8-GPU pass would use `--devices 4,5` (etc.), which flips the
launcher to `torchrun --nproc_per_node N --master_port 29644`. Not run this wave because GPUs
0–3 were a live production job; the lead coordinates the multi-GPU pass once they free up. DDP
gives the cross-rank in-batch-negative pool the ROCm 2-GPU run already exercised.

## Notes

- **PyPI `tevatron` is a trap.** Version 0.1.0 is the 2021 v1 package — no
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
- **Port 29820** is the launcher default; 29500 collides on this host.
- **`destroy_process_group() was not called`** warning at the end of every multi-GPU run —
  upstream never tears the group down. Cosmetic.
- **Loss goes *up* with batch size** in the sweep table. Expected, not a bug: a larger
  in-batch pool makes the InfoNCE softmax harder (ln 1024 ≈ 6.9 vs ln 128 ≈ 4.9), and with
  100 rows a bigger batch also means fewer optimizer steps. Compare losses only at equal
  batch geometry.
- **`train_group_size` must fit the data.** The shipped sample has 5 negatives per query,
  so `--train_group_size 6` uses each exactly once; anything larger makes the reader sample
  negatives with replacement (`random.choices`), duplicating them inside the group.
- **BERT `embeddings.position_ids UNEXPECTED`** in the load report for
  `bge-small-en-v1.5` — a stale buffer in the checkpoint, ignorable.
- **`pillow` is a hard import** of `tevatron.retriever.dataset` even for pure-text
  retrieval; it is in the requirements file for that reason.
- **The shipped sample is a pipeline check, not a training result.** 100 rows of OTel/paper
  text produce no useful retriever.

## Verdict

**Does it work on MI355X? Yes.** Every headline capability ran unmodified on gfx950 /
ROCm 7.2.4 with `torch==2.11.0+rocm7.2`: training, LoRA, DDP with the cross-rank negative
gather, model save, and — critically — **GradCache**, which is pure PyTorch autograd
plumbing with no custom kernels and therefore has no ROCm-specific failure mode. The only
adaptations are two CLI arguments, both handled by the launcher.

**Is it worth keeping next to `../sentence_transformers`? Yes, but not for GradCache.**

- **GradCache is *not* the differentiator.** sentence-transformers'
  `CachedMultipleNegativesRankingLoss` implements the same algorithm, and the sibling
  folder can adopt it by changing one loss class. Judged on GradCache alone, this folder
  is **redundant** — and the honest framing is that the original premise ("Tevatron gives
  you GradCache and the sibling doesn't") does not survive contact with a current
  sentence-transformers install. What Tevatron adds on that axis is narrow: separate
  query/passage chunk sizes, and GradCache living in the Trainer so it composes with LoRA
  and DeepSpeed without touching the loss.
- **What actually earns the folder is the retrieval pipeline.**
  `driver.encode` with `--dataset_number_of_shards/--dataset_shard_index` (sharded
  multi-GPU corpus encoding), `driver.search` producing a TREC run file, and the
  LoRA-on-a-decoder-LM (RepLLaMA / Qwen3-0.6B `--pooling eos`) recipe. `../sentence_transformers`
  has none of these — it trains and evaluates a model but never encodes a corpus at scale
  or produces a ranking. On a 288 GB × 8 box, "encode a 10 M-passage corpus in 8 shards"
  is exactly the thing you want and exactly the thing the other folder cannot do.
- **Do not use it as a general embedding trainer.** No evaluation of any kind, no
  best-checkpoint selection, no Matryoshka, no MTEB — and its transformers-5.x
  compatibility is incidental rather than maintained (two flags already broke). For "adapt
  an embedding model to my domain and tell me if it got better", `../sentence_transformers`
  is the better tool and should stay the default.

**Recommendation: keep**, scoped as *the dense-retrieval pipeline folder* (train → shard
encode → search → run file, plus LLM-scale LoRA retrievers), with the README stating
plainly that GradCache is not a reason to prefer it over the sentence-transformers sibling.
