# `training/reranker/sentence_transformers` — cross-encoder reranker fine-tuner

`train_reranker_standalone.py` fine-tunes a `CrossEncoder` (default
`Qwen/Qwen3-Reranker-0.6B`, `num_labels=1`) to score (query, document) pairs with
`BinaryCrossEntropyLoss`, using one positive and N hard negatives per query.

Use it as the second stage of a retrieval pipeline: a cross-encoder scores the pair
jointly, which is more accurate than embedding similarity but too slow for first-stage
retrieval, so it reranks candidates produced by an embedding model (see
`../../embedding/sentence_transformers/`). Evaluation is `CrossEncoderRerankingEvaluator`
at k=10, reporting `reranking_map`, `reranking_mrr@10`, and `reranking_ndcg@10`; the last
one drives `load_best_model_at_end`.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU) and AMD MI355X 288GB (ROCm 7.2.4, 1 and
8 GPUs). No code change on either vendor. Batch defaults assume 80GB GPUs.

## Files

- `train_reranker_standalone.py` — trainer entry point.
- `requirements_reranker.txt` — pinned deps.
- `OTel_reranker_sample_100.jsonl` — 100-row sample, the default `--data`.

## Setup

Python 3.12, one venv for this recipe.

### NVIDIA (CUDA 13.0)

`torch==2.11.0` resolves to a native `+cu130` wheel on PyPI — no `--index-url`. Install
torch first, then the rest:

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_sentence_transformers && source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 numpy               # -> 2.11.0+cu130
pip install -r requirements_reranker.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # re-verify: not clobbered
ln -sf ../../../dev.env dev.env                # HF_TOKEN for the Hub
```

`flash-attn` is optional. No prebuilt wheel exists for torch 2.11/cu130/py3.12, so
building it needs the CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3 --no-build-isolation
```

### AMD (ROCm 7.2)

Install torch from the ROCm index **before** the rest of the requirements:

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_sentence_transformers && source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> 2.11.0+rocm7.2
pip install -r requirements_reranker.txt      # all pins install unchanged
ln -sf ../../../dev.env dev.env               # HF_TOKEN for the Hub
```

The `torch==2.11.0` pin also exists on the rocm7.1 index — match your installed ROCm. Do
**not** install `flash-attn` on ROCm.

Verify either install:

```bash
python -c "from sentence_transformers.cross_encoder import CrossEncoder; print('imports OK')"
```

`--hf_home` points the model cache somewhere other than `~/.cache/huggingface`.

## Data

`--data` is a JSONL, one row per query — an `anchor` (query), one `positive` (relevant
doc), and hard negatives `negative_1 ... negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

Each row expands into `1 + --n_neg` labeled cross-encoder pairs — the positive at label
`1.0`, each negative at `0.0` — as `{sentence_0=query, sentence_1=doc, label}`. Extra
columns (such as the `answer` field in the sample) are ignored.

`OTel_reranker_sample_100.jsonl` is the shipped default and validates the pipeline only.
To train for real, point `--data` at your own JSONL with the same columns.

## Run

Use `torchrun` at every world size: the script calls `dist.init_process_group`
unconditionally and `accelerate launch --num_processes=1` does not set `RANK`/`WORLD_SIZE`
(it dies with "environment variable RANK expected, but not set").
`accelerate launch --num_processes=8` does set them and works for multi-GPU.

Smoke test against the shipped sample, single GPU:

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # on NVIDIA set only CUDA_VISIBLE_DEVICES
torchrun --nproc_per_node=1 --master_port 29648 train_reranker_standalone.py \
  --test_mode --epochs 3 --batch 16 --eval_frac 0.05 --max_len 512 --out <out>
```

Multi-GPU (single-node DDP, one rank per GPU; NCCL maps to RCCL on ROCm):

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port 29632 train_reranker_standalone.py \
  --data /path/to/train.jsonl --out /path/to/out --batch 16 --epochs 3
```

Full run in the background:

```bash
nohup torchrun --nproc_per_node=8 --master_port 29632 train_reranker_standalone.py \
  --data /path/to/your_train.jsonl --out reranker_run1 --batch 16 --epochs 3 \
  > train_reranker_standalone.log 2>&1 &

tail -f train_reranker_standalone.log
```

Expected shape: a baseline evaluation prints before training, then a decaying loss, then
the best checkpoint is saved:

```
Baseline Results: {'reranking_map': 1.0, 'reranking_mrr@10': 1.0, 'reranking_ndcg@10': 1.0}
{'loss': '0.1686', 'grad_norm': '7.25', 'learning_rate': '3.125e-06', 'epoch': '2.229'}
{'train_runtime': '47.91', 'train_samples_per_second': '35.7', 'train_loss': '0.5844', 'epoch': '3'}
Training complete. Best model loaded and saved to <out>/final
```

### Batch geometry

`gradient_accumulation_steps` is fixed at 2 and `dataloader_drop_last=True`, so:

```
global_batch  = --batch x num_GPUs x 2
steps/epoch   = floor(pairs / (--batch x num_GPUs)) / 2
pairs         = train_rows x (1 + --n_neg)
```

At the default `--batch 64` on 8 ranks the global batch is 1024 against a 100-row sample
that expands to ~594 pairs, so the run performs **one optimizer step**, exits 0, and
trains nothing. Lower `--batch` (or enlarge the dataset) until the step count is
meaningful. Watch `--eval_frac` too: on 100 rows the default `0.003` leaves **one** eval
query, which makes `reranking_ndcg@10` read exactly `1.0` and `metric_for_best_model`
meaningless — use `--eval_frac 0.05` on small data.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder base model |
| `--data` | `OTel_reranker_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--out` | `output` | Output dir for checkpoints, logs, and the final model |
| `--max_len` | `1024` | Max sequence length (query + document) |
| `--batch` | `64` | Per-GPU train/eval batch size |
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

## Notes

- Budget disk: with `save_strategy="epoch"` and `load_best_model_at_end=True` a single
  epoch writes ~4.5 GB (checkpoint + optimizer state + final model). A full filesystem is
  the likeliest failure here.
- `predict()` on the saved model returns **raw logits** (e.g. -0.625 / -9.375), not 0-1
  probabilities — ST's `LogitScore` module with an `Identity` activation. When comparing
  against another implementation (e.g. the generative yes/no scorer in
  `inference/*/reranker/`), diff the rankings, not the absolute scores.
- Only `--batch` needs retuning across world sizes; `--max_len`, `--n_neg`, `--lr`, and the
  hardcoded `gradient_accumulation_steps=2` / `gradient_checkpointing=True` work as shipped.
- If a non-default model raises ST's `_verify_pair_roles_supported` `ValueError`, supply a
  Query/Document chat template via `model.processor.chat_template = ...`,
  `processor_kwargs={"chat_template": ...}`, or a `chat_template.jinja` beside the model.
  The default Qwen3-Reranker path needs no intervention.
