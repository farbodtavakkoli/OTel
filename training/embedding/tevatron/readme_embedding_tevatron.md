# `training/embedding/tevatron` — dense-retrieval training (Tevatron v2)

[Tevatron](https://github.com/texttron/tevatron) is the reference toolkit for dense
retrieval — DPR, RepLLaMA, and the BEIR/MS MARCO pipeline (train -> shard-encode -> search
-> TREC run). `train_embedding_tevatron.py` wraps its `tevatron.retriever.driver.*`
entrypoints, and `convert_data.py` converts this repo's triplet JSONL into Tevatron format.

It is retrieval-first: one contrastive objective (InfoNCE over `train_group_size` passages
per query, gathered across ranks under DDP), no built-in evaluation. Pick it when you want
the full encode/search pipeline or a LoRA decoder-LM retriever; pick
`../sentence_transformers` when you want a general embedding model with NDCG@10 evaluation
baked in.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU) and AMD MI355X 288GB (ROCm 7.2.4, 1 and
2 GPUs, including GradCache, LoRA and the cross-rank negative gather). No Tevatron source
changes on either vendor.

## Files

- `train_embedding_tevatron.py` — launcher for the upstream train driver; `utils.py` builds
  its command and environment.
- `convert_data.py` — triplet JSONL -> Tevatron JSONL.
- `requirements_embedding_tevatron.txt` — pinned deps.
- `OTel_tevatron_sample_100.jsonl` — 100-row sample, the default `--dataset_path`.

## Setup

Python 3.12, one venv for this recipe. Put it on a larger volume if the root filesystem is
tight — the commands below refer to machine-specific locations by env var:

```bash
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export PIP_CACHE_DIR=$DATA_DIR/pip_cache

cd training/embedding/tevatron
ln -sf ../../../dev.env dev.env        # HF_TOKEN; optional here, bge/Qwen are not gated
python3 -m venv $DATA_DIR/envs/.env_tevatron
source $DATA_DIR/envs/.env_tevatron/bin/activate
pip install -U pip setuptools wheel
```

**Do not `pip install tevatron`.** PyPI 0.1.0 is the 2021 v1 package: no
`tevatron.retriever` module and no `--grad_cache`. Install from git, with `--no-deps` —
its `setup.py` bounds (`transformers>=4.10.0`, `datasets>=1.1.3`) would otherwise resolve
the whole stack away from the pins.

### AMD (ROCm 7.2)

Install the ROCm torch wheel **first**, then the requirements, then the two git packages:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding_tevatron.txt
pip install git+https://github.com/luyug/GradCache.git
pip install --no-deps git+https://github.com/texttron/tevatron.git

python -c "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda)"
# 2.11.0+rocm7.2 7.2.26015 None      <- hip set, cuda None
```

Do **not** install `flash-attn` on ROCm. If pip swapped in the CUDA torch wheel:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

### NVIDIA (CUDA 13.0)

Same order, no `--index-url` — `torch==2.11.0` resolves a native `+cu130` wheel:

```bash
pip install torch==2.11.0 numpy          # -> 2.11.0+cu130, nvidia-*-cu13 deps
pip install -r requirements_embedding_tevatron.txt
git clone --depth 1 https://github.com/luyug/GradCache.git   && pip install ./GradCache
git clone --depth 1 https://github.com/texttron/tevatron.git && pip install --no-deps ./tevatron
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0+cu130 13.0
```

Pass `--attn_implementation sdpa` for BERT backbones; FA2 needs the `flash-attn` package
and gives BERT nothing. Reserve `flash_attention_2` + `pip install flash-attn==2.8.3` for
the Qwen `--pooling eos` LoRA recipe.

## Data

Tevatron's `TrainDataset` accepts two formats: **self-contained** (used here) — `query`
plus inline `positive_passages` / `negative_passages`, each a `{docid, text}` object with
an optional `title`, no corpus file needed; or **corpus-referencing** —
`query_id`/`query_text` plus `positive_document_ids` / `negative_document_ids`, resolved
against a `--corpus_path` JSONL of `{docid, text}`.

`convert_data.py` emits the self-contained format from the repo's triplet schema
(`anchor` / `positive` / `negative_1..N`):

```bash
python convert_data.py \
  --input_file ../sentence_transformers/OTel_embedding_sample_100.jsonl \
  --output_file OTel_tevatron_sample_100.jsonl
```

| Flag | Default | Meaning |
|---|---|---|
| `--input_file` | `../sentence_transformers/OTel_embedding_sample_100.jsonl` | Source triplet JSONL |
| `--output_file` | `OTel_tevatron_sample_100.jsonl` | Destination JSONL in Tevatron format |
| `--anchor_field` | `anchor` | Column holding the query text |
| `--positive_field` | `positive` | Column holding the relevant passage |
| `--negative_prefix` | `negative_` | Prefix of the hard-negative columns |
| `--max_negatives` | `5` | Keep at most this many hard negatives per row |
| `--limit` | `None` | Convert only the first N rows |
| `--docid_prefix` | `otel` | Prefix for the generated `docid` values |

Shipped sample `OTel_tevatron_sample_100.jsonl` (100 rows) — one row:

```json
{"query_id": "q0", "query": "...",
 "positive_passages": [{"docid": "otel-0-pos", "text": "..."}],
 "negative_passages": [{"docid": "otel-0-neg0", "text": "..."}, ...]}
```

It validates the pipeline; it does not produce a useful retriever. Converting your own
data: negatives are taken in **numeric** suffix order (`negative_2` before `negative_10`)
and blank ones are dropped, so a row can yield fewer than `--max_negatives`; a row with a
missing or blank `anchor`/`positive` is skipped, so check the written/skipped counts on the
closing line. Omit `title` unless it is real — the reader does `title + ' ' + text`
whenever the key exists.

## Run

The launcher runs `python -m tevatron.retriever.driver.train` for one device and
`torchrun --nproc_per_node N --master_port 29820` for several, resolves the attention
kernel from the torch build, and prints the effective batch geometry. `--dry_run` prints
the resolved command and exits.

Smoke test on the shipped sample, one GPU (id 2):

```bash
source $DATA_DIR/envs/.env_tevatron/bin/activate

python train_embedding_tevatron.py \
  --devices 2 --model_name_or_path BAAI/bge-small-en-v1.5 \
  --batch_size 8 --train_group_size 6 --epochs 1 \
  --output_dir $OUTPUT_DIR/train_embedding_tevatron/smoke_1gpu --overwrite
```

Pass several ids (`--devices 2,3`) for multi-GPU: the launcher switches to `torchrun` and
the loss becomes `DistributedContrastiveLoss`, so negatives are pooled across ranks.
GradCache, LoRA and DDP compose.

Full run with GradCache + LoRA on a decoder LM — point `--dataset_path` at your own
converted JSONL:

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

Expected tail — `bge-small-en-v1.5`, 100-row sample, `--batch_size 8 --train_group_size 6
--epochs 4`; `drop_last` keeps 96 rows, so 13 steps/epoch x 4 = 52 optimizer steps:

```text
{'train_runtime': '12.27', 'train_samples_per_second': '32.6', 'train_loss': '0.8935', 'epoch': '4'}
INFO - training/embedding/tevatron - training finished; encoder written to ...
```

### Encode and search

Not wrapped by the launcher — call the upstream drivers directly. Corpus encoding shards
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

With `--lora`, pass the adapter to the encode driver as `--lora --lora_name_or_path <dir>`.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name_or_path` | `BAAI/bge-small-en-v1.5` | Backbone model id or path |
| `--pooling` | `cls` | `cls` \| `mean` \| `eos` \| `last` — `eos` for decoder LMs |
| `--normalize` / `--no_normalize` | on | L2-normalize vectors (cosine) vs raw dot product |
| `--append_eos_token` | off | Append EOS — required by the eos-pooling recipe |
| `--temperature` | `0.02` | Contrastive-loss softmax temperature |
| `--attn_implementation` | `auto` | `auto` = sdpa on ROCm/CPU, flash_attention_2 on CUDA |
| `--dataset_name` | `json` | HF dataset name, or `json` to read `--dataset_path` |
| `--dataset_path` | `OTel_tevatron_sample_100.jsonl` | Local training JSONL (Tevatron format) |
| `--dataset_config` | `None` | HF dataset config name |
| `--dataset_split` | `train` | Split to train on |
| `--corpus_name` / `--corpus_path` | `None` | Corpus source for the docid-referencing format |
| `--query_prefix` / `--passage_prefix` | `None` | Instruction prefixes added at tokenization |
| `--query_max_len` | `64` | Query truncation length |
| `--passage_max_len` | `192` | Passage truncation length |
| `--train_group_size` | `6` | Passages per query: 1 positive + n-1 hard negatives |
| `--output_dir` | `outputs/tevatron_run` | Checkpoints and the final encoder |
| `--overwrite` | off | Delete a non-empty `--output_dir` first |
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
| `--grad_cache` | off | GradCache chunked re-forward; batch size stops being memory-bound |
| `--gc_q_chunk_size` | `8` | Queries per GradCache sub-forward |
| `--gc_p_chunk_size` | `16` | Passages per GradCache sub-forward |
| `--lora` | off | LoRA fine-tuning (needs `peft`) |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | `16` / `64` / `0.1` | LoRA hyperparameters |
| `--lora_target_modules` | `q_proj,...,gate_proj` | Comma-separated LoRA target modules |
| `--devices` | `None` | GPU ids for `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, e.g. `2,3` |
| `--master_port` | `29820` | torchrun rendezvous port (29500 often collides) |
| `--dry_run` | off | Print the resolved command and exit |

## Output

Under `--output_dir`: `config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json` — the bare encoder, reloadable with
`AutoModel.from_pretrained(...)`, not a Tevatron wrapper — plus `training_args.bin` and
`checkpoint-<step>/` per `--save_strategy`. With `--lora` you get
`adapter_model.safetensors` + `adapter_config.json` instead.

There is **no evaluation output** — no IR metrics, no best-checkpoint selection. Score
quality by running the encode -> search drivers and scoring the run file yourself (e.g.
`pytrec_eval`).

## Notes

- Turn `--grad_cache` on whenever the per-device batch exceeds 16. Without it activation
  memory scales linearly with `--batch_size`: `Qwen/Qwen3-0.6B` + LoRA at
  `--train_group_size 8 --passage_max_len 192` OOMs above 16 on an 80GB card. With it,
  memory stays flat at the cost of a second forward pass per step.
- `--train_group_size` must fit the data. The shipped sample has 5 negatives per query, so
  `--train_group_size 6` uses each exactly once; larger values make the reader sample with
  replacement and duplicate negatives inside the group.
- Loss rises with batch size by design — a larger in-batch pool makes the InfoNCE softmax
  harder. Compare losses only at equal batch geometry.
- With a populated `HF_HOME` and no Hub egress, set `HF_HUB_OFFLINE=1`. Point
  `HF_DATASETS_CACHE` at `/dev/shm/dscache_tevatron` if the datasets cache lands on a slow
  mount.
