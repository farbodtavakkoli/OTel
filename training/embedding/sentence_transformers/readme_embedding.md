# `training/embedding/sentence_transformers` — bi-encoder embedding fine-tuner

`train_embedding_standalone.py` trains a dense retrieval embedding model on triplets with
hard negatives. One script serves four families: sentence-transformers
(`all-mpnet-base-v2`, `all-MiniLM-*`), BGE (`BAAI/bge-small/large/m3`), EmbeddingGemma
(`google/embeddinggemma-300m`), and static model2vec (`minishlab/potion-*`).

Pick it over `../tevatron` or `../pylate` when you want a standard single-vector
bi-encoder. The loss is `MultipleNegativesRankingLoss` (dot product, scale 20, cross-device
gather) wrapped in `MatryoshkaLoss`, so one trained model serves several embedding
dimensions. Per-model batch size, epochs, LR, prefixes, and eval corpus size come from the
`MODELS` registry in the script; the CLI flags under [Arguments](#arguments) override them.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU) and AMD MI355X 288GB (ROCm 7.2.4, 1 and
8 GPUs). No code change on either vendor. Registry batch sizes assume an 80GB GPU.

## Files

- `train_embedding_standalone.py` — trainer entry point.
- `requirements_embedding.txt` — pinned deps.
- `OTel_embedding_sample_100.jsonl` — 100-row sample, the default `--train_file`.

## Setup

Python 3.12, one venv for this recipe:

```bash
cd training/embedding/sentence_transformers
ln -sf ../../../dev.env dev.env          # supplies HF_TOKEN for gated models (embeddinggemma)
python3.12 -m venv .env_sentence_transformers
source .env_sentence_transformers/bin/activate
```

### NVIDIA (CUDA 13.0)

The `torch==2.11.0` pin resolves to a native `+cu130` wheel, so no `--index-url`:

```bash
pip install torch==2.11.0 numpy==2.5.1     # -> torch 2.11.0+cu130
pip install -r requirements_embedding.txt  # torch/numpy already satisfied
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: several deps clobber torch
```

`flash-attn` is optional — the script uses `sdpa` when it is not importable. No prebuilt
wheel exists for torch 2.11+cu130, so build it from source if you want it:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3 --no-build-isolation
```

### AMD (ROCm 7.2)

Install the ROCm torch wheel **before** the rest of the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding.txt
```

Do **not** install `flash-attn` on ROCm.

Verify either install:

```bash
python -c "import sentence_transformers, mteb; print('imports OK')"
```

`model2vec` (already in the requirements) is only used by the static `potion-*` models.

## Data

`--train_file` is a JSONL with one triplet-style row per line — an `anchor` (query), one
`positive` (relevant passage), and hard negatives `negative_1 ... negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

`OTel_embedding_sample_100.jsonl` is the shipped default and validates the pipeline only.
To train for real, point `--train_file` at your own JSONL with the same columns.

## Run

Use `accelerate launch` for multi-process runs and `torchrun` for single-GPU. The model
family is selected by `--model_name`; everything else defaults from the registry.

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m --batch_size 1
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m \
  --train_file /path/to/your_train.jsonl \
  > train_embedding_standalone.log 2>&1 &

tail -f train_embedding_standalone.log
```

Any family works the same way, and registry defaults are overridable:

```bash
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name sentence-transformers/all-mpnet-base-v2
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name minishlab/potion-retrieval-32M

accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3 \
  --batch_size 24 --epochs 3 --lr 2e-5 --eval_corpus_size 100000
```

### Single GPU

Launch with `torchrun`, not `accelerate launch --num_processes=1` — the simple launcher
does not set `RANK`, which this script requires. Pass `--master_port` if 29500 is taken.
`--experiment_root` doubles as `HF_HOME`, so pass `--output_dir` separately.

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
export HF_HOME=/path/to/model/cache HF_DATASETS_CACHE=/tmp/dscache_emb_st
torchrun --nproc_per_node=1 --master_port=29610 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 8 --epochs 1 \
  --experiment_root /path/to/model/cache --output_dir /path/to/out
```

Expected tail of an 11-step run:

```
{'train_runtime': '70.47', 'train_samples_per_second': '1.277', 'train_loss': '2.286', 'epoch': '1'}
[rank=0] INFO: MTEB Success. Avg Score: 0.5609
```

MTEB average around 0.56 is the useful fingerprint. On the 100-row sample the
`telco_unseen` / `telco_seen` nDCG@10 both read 1.0 — saturated on a ~10-document corpus,
so a smoke signal only.

### Multi-GPU

Single-node DDP, one rank per GPU:

```bash
source .env_sentence_transformers/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # override any device pin left in bin/activate
python -c "import torch; assert torch.cuda.device_count()==8"
torchrun --nproc_per_node=8 --master_port=29631 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 1 --epochs 1
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `google/embeddinggemma-300m` | Model id/path; selects the registry entry |
| `--train_file` | `OTel_embedding_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--output_dir` | `None` | Output dir; defaults to `<experiment_root>/<RUN_ID>/trained_<model>` |
| `--experiment_root` | `experiments` | Root for `HF_HOME` and the default output dir |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--seed` | `42` | Seed for data splits and eval sampling |
| `--loader` | `None` | `transformer` or `static` |
| `--batch_size` | `None` | Per-device train batch size (eval batch = 2x) |
| `--epochs` | `None` | Training epochs |
| `--lr` | `None` | Learning rate |
| `--max_seq_length` | `None` | Max sequence length |
| `--eval_corpus_size` | `None` | Distractor corpus size for eval; omit for an all-positives corpus |

## Output

Under the resolved output dir: checkpoints, TensorBoard logs (`logs/`), `run.log`, a copy
of the trainer (`train_script_backup.py`), and the best model at
`<output_dir>/final_model`.

Each epoch runs three evaluators:

- `telco_unseen` — IR evaluator on held-out queries; `eval_telco_unseen_dot_ndcg@10`
  selects the best checkpoint.
- `telco_seen` — same, on a sample of training queries (memorization signal).
- MTEB `SciFact` + `NFCorpus` — general-capability check, rank 0 only.

## Notes

- Steps per epoch are `floor(train_rows / (per_device_batch x ranks))` because
  `dataloader_drop_last=True`. Size `--batch_size` so this stays above zero or the run
  trains nothing: the 100-row sample (90 train / 10 eval) needs `--batch_size 1` at 8 ranks
  and `--batch_size 8` at one rank.
- `--eval_corpus_size` unset means the eval corpus is every known positive; set an int to
  use required answers plus that many distractors (what the Gemma/model2vec runs do).
- If an outbound proxy 403s huggingface.co (model resolution and the MTEB
  SciFact/NFCorpus download), `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy
  https_proxy all_proxy` first. A cache-resident model is unaffected.
