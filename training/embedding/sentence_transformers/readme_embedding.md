# `train_embedding_standalone.py`

## Overview & when to use

Unified **embedding fine-tuner** built on sentence-transformers. One script trains any of
four model families for retrieval with hard negatives:

- sentence-transformers (`all-mpnet-base-v2`, `all-MiniLM-*`)
- BGE (`BAAI/bge-small/large/m3`)
- EmbeddingGemma (`google/embeddinggemma-300m`)
- static model2vec (`minishlab/potion-*`)

Use it when you need a retrieval embedding model adapted to your domain corpus — the loss
(MultipleNegativesRankingLoss, dot-product, cross-device gather) treats each query's
positive as the target and the negatives as in-batch hard negatives, and is wrapped in
MatryoshkaLoss so one trained model serves multiple embedding dimensions.

Per-model defaults live in the `MODELS` registry in the script and can be overridden on
the CLI. Registry fields:

| Field | Meaning |
|---|---|
| `loader` | `"transformer"` (SentenceTransformer) or `"static"` (model2vec) |
| `query_prefix` / `doc_prefix` | Instruction prompts prepended to queries/documents at encode time |
| `train_batch` / `eval_batch` | Per-device batch sizes — tuned for H100 80GB |
| `epochs` | Number of training epochs |
| `learning_rate` | Optimizer LR — static lookup tables want a much higher LR |
| `use_checkpointing` | Gradient checkpointing on/off (n/a for static) |
| `max_seq_length` | Optional; `None` keeps the model default (Gemma uses 1024) |
| `matryoshka_dims` | Optional; `None` derives dims dynamically from the hidden size |
| `eval_corpus_size` | `None` — corpus = all positives; int — distractor pool of that size |
| `eval_on_start` | Run a baseline evaluation before training |

Design notes:

- **Registry-driven** — `resolve_cfg` merges `DEFAULT_CFG` ← `MODELS[model_name]` ← CLI
  overrides, so one code path serves every family.
- **Loader switch** — `transformer` ⇒ `SentenceTransformer(name)` with bf16 + flash-attn;
  `static` ⇒ `StaticEmbedding.from_model2vec(name)` — a lookup table, so no
  dtype/attention/tokenizer kwargs apply.
- **Best model** — `load_best_model_at_end` on `telco_unseen` nDCG@10; the final model is
  saved to `<output_dir>/final_model`.

## Install

Python 3.12, in its own venv — these deps differ from the other trainers:

```bash
python3.12 -m venv ~/.venv-embed && source ~/.venv-embed/bin/activate
pip install -r requirements_embedding.txt
```

`model2vec` is only needed for the static `potion-*` models. Verify:

```bash
python -c "import sentence_transformers, mteb; print('imports OK')"
```

### NVIDIA (CUDA)

The default `torch` wheels from PyPI ship with CUDA support: the `torch==2.11.0` pin
resolves to a native `2.11.0+cu130` wheel, so no `--index-url` is needed.

```bash
python3 -m venv .env_sentence_transformers
source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 numpy==2.5.1     # -> torch 2.11.0+cu130
pip install -r requirements_embedding.txt  # torch/numpy already satisfied
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: several deps clobber torch
```

Transformer-backed models load with `attn_implementation="flash_attention_2"` when
`flash_attn` imports and fall back to `sdpa` otherwise, so flash-attn is optional. There is
no prebuilt wheel for this torch/CUDA combination — build it from source if you want it:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3 --no-build-isolation
```

### AMD (ROCm)

Install torch from the ROCm wheel index **before** the rest of the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding.txt
```

Do **not** install `flash-attn` on ROCm — the pin here is a CUDA build. `load_model()`
detects a ROCm torch build (`torch.version.hip`) and selects
`attn_implementation="sdpa"` automatically.

## Environment & secrets

Optional `dev.env` next to the script — only needed for gated models on the Hub
(e.g. `google/embeddinggemma-300m`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`. `HF_HOME` (the model/dataset cache) is pointed at
`--experiment_root`.

## Data

`--train_file` is a JSONL with one triplet-style row per line — an `anchor` (query), one
`positive` (relevant passage), and hard negatives `negative_1 … negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

Shipped sample: `OTel_embedding_sample_100.jsonl` (100 rows) — the default
`--train_file`, there to smoke-test the pipeline end to end. To train for real, point
`--train_file` at your own JSONL with the same columns.

## Run

Launch with `accelerate` (single-node DDP) for multi-process runs, or `torchrun` for any
run including single-GPU — see *Single GPU* below for why `accelerate --num_processes=1`
does not work here. The model family is selected by `--model_name`; everything else
defaults from the registry.

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m \
  --train_file /path/to/your_train.jsonl \
  > train_embedding_standalone.log 2>&1 &

tail -f train_embedding_standalone.log
```

Any family works the same way:

```bash
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name sentence-transformers/all-mpnet-base-v2
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name minishlab/potion-retrieval-32M
```

Override registry defaults on the CLI:

```bash
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3 \
  --batch_size 24 --epochs 3 --lr 2e-5 --eval_corpus_size 100000
```

### Single GPU

Use **`torchrun`, not `accelerate launch`**: with `--num_processes=1` accelerate uses its
simple launcher and does not set `RANK`, which the script's explicit
`dist.init_process_group` requires. This is a launcher quirk, not a hardware one. Pass
`--master_port` if 29500 is already taken.

`--experiment_root` must point at the model cache, because the script sets
`HF_HOME = --experiment_root` internally; give `--output_dir` separately so checkpoints
stay off the cache directory.

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
export HF_HOME=/path/to/model/cache HF_DATASETS_CACHE=/tmp/dscache_emb_st
torchrun --nproc_per_node=1 --master_port=29610 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 8 --epochs 1 \
  --experiment_root /path/to/model/cache --output_dir /path/to/out
```

If an outbound proxy 403s huggingface.co (model resolution plus the MTEB SciFact/NFCorpus
dataset download), unset it first — `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy
https_proxy all_proxy`. A cache-resident model is unaffected.

**`--batch_size 8` is deliberate**: the shipped 100-row sample with
`dataloader_drop_last=True` yields **zero** optimizer steps at the registry default batch
of 96.

**Expected output** (11 steps):

```
{'loss': '5.097', 'grad_norm': '392', 'learning_rate': '0', 'epoch': '0.09091'}
{'loss': '1.44', 'grad_norm': '223', 'learning_rate': '1e-05', 'epoch': '0.2727'}
{'train_runtime': '70.47', 'train_samples_per_second': '1.277', 'train_loss': '2.286', 'epoch': '1'}
[rank=0] INFO: MTEB Success. Avg Score: 0.5609
```

Loss is noisy step to step but decreasing in the mean. **MTEB average ~0.56** is the useful
fingerprint that the run worked. `telco_unseen` / `telco_seen` nDCG@10 both read 1.0 —
saturated on a ~10-document eval corpus, so treat them as a smoke signal only.

### Multi-GPU

Plain single-node DDP via `torchrun`, one rank per GPU; no code change is needed.

```bash
source .env_sentence_transformers/bin/activate
# If bin/activate carries a leftover single-GPU device pin, sourcing it and launching N
# ranks silently trains on one GPU. Override after activating, and assert:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"
torchrun --nproc_per_node=8 --master_port=29631 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 1 --epochs 1
```

**Batch geometry.** Steps per epoch are
`floor(train_rows / (per_device_batch × ranks))` because `dataloader_drop_last=True`. Size
`--batch_size` so that this stays above zero, or the run silently trains nothing: the
shipped 100-row sample (90 train / 10 eval) needs `--batch_size 1` at 8 ranks.

**Checkpoint saving can be disabled** on a disk-constrained host by patching
`save_strategy="no"` / `load_best_model_at_end=False` onto the training args from an
external wrapper — no change to the repo script.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `google/embeddinggemma-300m` | Model id/path; selects the registry entry |
| `--train_file` | `OTel_embedding_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--output_dir` | `None` | Output dir; defaults to `<experiment_root>/<RUN_ID>/trained_<model>` |
| `--experiment_root` | `experiments` | Root for `HF_HOME` and the default output dir |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--seed` | `42` | Seed for data splits and eval sampling — fixed so all ranks split identically |
| `--loader` | `None` | `transformer` or `static` — overrides registry |
| `--batch_size` | `None` | Per-device train batch size — overrides registry (eval batch = 2×) |
| `--epochs` | `None` | Training epochs — overrides registry |
| `--lr` | `None` | Learning rate — overrides registry |
| `--max_seq_length` | `None` | Max sequence length — overrides registry |
| `--eval_corpus_size` | `None` | Distractor corpus size for eval — overrides registry; omit for all-positives corpus |

## Output

- Checkpoints, TensorBoard logs (`logs/`), and `run.log` under the resolved output dir.
- A backup of the training script (`train_script_backup.py`) alongside the checkpoints.
- The final (best) model under `<output_dir>/final_model`.

Each epoch runs a `SequentialEvaluator`:

- **`telco_unseen`** — IR evaluator over held-out eval queries; its
  `eval_telco_unseen_dot_ndcg@10` picks the best checkpoint.
- **`telco_seen`** — IR evaluator over a sample of training queries (memorization signal).
- **MTEB** — `SciFact` + `NFCorpus` as a general-capability check (rank-0 only).

`--eval_corpus_size` controls the retrieval corpus: unset ⇒ all known positives; an int ⇒
required answers plus that many sampled distractors (used for the larger Gemma/model2vec runs).

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 8 GPUs |
| PyTorch | `torch==2.11.0` resolves to `+cu130` on plain PyPI | `torch==2.11.0` from `download.pytorch.org/whl/rocm7.2` |
| Attention | `flash_attention_2` (source-built) or `sdpa` | `sdpa` (selected automatically) |

The script needs **no code changes** on either vendor; the flash-attn → sdpa switch is
automatic. Registry batch sizes are tuned for 80GB GPUs.

## Notes

- The evaluators and the MTEB wrapper only run their heavy work on rank 0; barriers keep
  the other ranks in sync.
- `transformers 5.0.0.dev0` misses `save_safetensors` — the script patches it onto the
  training args if absent.
- The shipped OTel sample is only for pipeline validation, not for producing a useful
  model.
