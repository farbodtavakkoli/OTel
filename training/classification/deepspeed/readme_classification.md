# `training/classification/deepspeed` — sequence classification with DeepSpeed ZeRO

`train_llm_classification.py` full fine-tunes a decoder model as a multi-class classifier
(`AutoModelForSequenceClassification`, custom `Dropout(0.1) -> Linear(hidden, num_labels)`
head) on a labeled CSV, using HF Transformers + DeepSpeed ZeRO-2. This is a classifier, not
a generative SFT trainer — for text generation use `../../llm/deepspeed_standalone/`.

Three steps: **train**, then **prepare** (the trained head is a `nn.Sequential`, so its
weight is stored as `score.1.weight`; `prepare_model_for_hf.py` renames it to
`score.weight` so the folder loads with a plain
`AutoModelForSequenceClassification.from_pretrained(...)`), then **infer/evaluate**.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU) and AMD MI355X 288GB (ROCm 7.2.4, 1, 4
and 8 GPUs), ZeRO stage 2. `deepspeed==0.19.4` installs as a pure-Python wheel on both
vendors — no nvcc/hipcc build, no `DS_BUILD_*` flags.

## Files

- `train_llm_classification.py` — trainer entry point; builds the DeepSpeed config
  in-memory from `--zero_stage` (there is no `ds_config.json`).
- `prepare_model_for_hf.py` — `score.1.weight` -> `score.weight` for a plain HF load.
- `inference_telelogs.py` — evaluate an HF-ready checkpoint on a labeled CSV (single GPU).
  Works on any CSV matching the column contract, not just TeleLogs.
- `make_classification_sample.py` — regenerates the shipped sample CSV.
- `requirements_classification.txt` — pinned deps.
- `data/classification_sample.csv` — 10-row sample, the default train and test file.

## Setup

Python 3.12, one venv for this recipe:

```bash
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache

cd training/classification/deepspeed
ln -sf ../../../dev.env dev.env        # HF_TOKEN; required for gated models such as gemma-3-1b-it
```

Install torch first and filter it (and `flash-attn`) out of the requirements so the
resolver cannot replace it.

### NVIDIA (CUDA 13.0)

`torch==2.11.0` resolves to a native `+cu130` wheel on plain PyPI — no `--index-url`:

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch==2.11.0 numpy
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_cuda.txt
pip install -r /tmp/reqs_cuda.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: not clobbered
```

`flash-attn` is optional. To build it, export the CUDA toolkit paths first:

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### AMD (ROCm 7.2)

Install torch from the ROCm index **before** the rest of the requirements:

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # rocm7.1 also published
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_rocm.txt
pip install -r /tmp/reqs_rocm.txt
```

Do **not** install `flash-attn` on ROCm.

Verify either install:

```bash
python -c "import torch, deepspeed, transformers, datasets, evaluate, sklearn; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

If `evaluate.load` fails behind a proxy that 403s huggingface.co, the metric scripts are
not cached yet. Fetch them once with the proxy unset:

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
python -c "import evaluate;[evaluate.load(m) for m in ('accuracy','f1','precision','recall')]"
```

## Data

`--train_file` / `--test_file` are CSVs. Each row needs a **text** column (`text`, or
`question`, which is renamed to `text`) and a **label** column named `label` or `answer`
holding string classes. The script derives the class set from the training split, builds
`label2id` / `id2label`, and maps labels to integer ids, so `num_labels` is inferred from
the data and nothing needs to be configured.

`data/classification_sample.csv` — 10 rows, columns `text,label`, 4 classes (`doc_direct`,
`key_facts`, `extractive_summary`, `detailed_summary`) — is the default for both
`--train_file` and `--test_file`. It is derived from
`../../llm/deepspeed_standalone/OTel_LLM_sample_10.jsonl`: the first user-turn content
becomes `text` (truncated to 2000 chars) and the row's `flow` field becomes `label`.
Regenerate it with:

```bash
python make_classification_sample.py
```

The sample only proves the pipeline runs — with `--test_file` equal to `--train_file` the
eval numbers are not a quality signal. For real training supply disjoint CSVs with the same
column contract, at least a few hundred rows so every rank gets full batches.

## Run

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_classification.py --num_epochs 1 --eval_steps 1 --save_steps 1
```

The accelerate config only needs `distributed_type: DEEPSPEED`; the ZeRO config comes from
`--zero_stage`. Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_classification.py \
  --train_file /path/to/train.csv --test_file /path/to/test.csv \
  > train_llm_classification.log 2>&1 &

tail -f train_llm_classification.log
```

Then prepare and evaluate:

```bash
python prepare_model_for_hf.py --source models/rnj-1-classifier --output best_model_hf_ready
python inference_telelogs.py --model_path best_model_hf_ready --test_file /path/to/test.csv
```

### Multi-GPU

Scales to 4 and 8 GPUs with no code change. Use a distinct `--main_process_port`; 29500
collides on a shared box.

```bash
source .env_deepspeed/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    # set AFTER sourcing, or a stale pin trains on one GPU
python -c "import torch; assert torch.cuda.device_count()==8"

accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29660 \
  train_llm_classification.py \
  --model_id google/gemma-3-1b-it --zero_stage 2 \
  --train_file /path/to/train.csv --test_file data/classification_sample.csv \
  --output_dir $OUTPUT_DIR/train_llm_classification/model_8gpu \
  --num_epochs 1 --batch_size 4 --grad_accum 1 --max_length 512 \
  --warmup_steps 2 --eval_steps 10 --save_steps 1000
```

The 10-row sample cannot feed 8 ranks — `DistributedSampler` pads it to 16, giving about
one micro-batch per rank. Use a few hundred rows so that per-device 4 x grad_accum 1 x
8 GPUs = global batch 32 yields real optimizer steps.

Expected output — single GPU, `google/gemma-3-1b-it`, 10-row sample, `--batch_size 1
--grad_accum 1 --num_epochs 5` = 50 optimizer steps:

```
{'loss': '0.3691','grad_norm': '13.19', 'learning_rate': '7.494e-09','epoch': '5'}
{'eval_loss': '0.4418', 'eval_accuracy': '1', 'eval_f1': '1', 'eval_precision': '1', 'eval_recall': '1', 'epoch': '5'}
Writing model shards: 100%|##########| 1/1 [00:00<00:00, 1.34it/s]
Model weights saved in .../model.safetensors
```

To confirm the DeepSpeed engine really ran (rather than a plain-Trainer fallback), look for
the ZeRO artifacts in the checkpoint: `checkpoint-50/global_step50/
bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt` and `checkpoint-50/zero_to_fp32.py`.

### Model choice

The default `EssentialAI/rnj-1` is a ~12B decoder; under ZeRO-2 a single-GPU full
fine-tune OOMs an 80GB card, so use multi-GPU or optimizer/parameter offload for it.
`google/gemma-3-1b-it` is the model the commands above use; it is gated, so `HF_TOKEN` is
required. `google/gemma-4-E4B-it` does not work here — `Gemma4Config` is not in
`AutoModelForSequenceClassification`'s registry in transformers 5.5.0.

## Arguments

### `train_llm_classification.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model_id` | `EssentialAI/rnj-1` | HF repo id or local path of the base model |
| `--output_dir` | `models/rnj-1-classifier` | Where checkpoints and the best model are written |
| `--train_file` | `data/classification_sample.csv` | Training CSV (text/question + label/answer columns) |
| `--test_file` | `data/classification_sample.csv` | Evaluation CSV with the same columns |
| `--learning_rate` | `7e-6` | Optimizer learning rate |
| `--batch_size` | `4` | Per-device batch; effective batch = batch x grad_accum x num_GPUs |
| `--grad_accum` | `2` | Gradient accumulation steps |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `150` | LR warmup steps |
| `--num_epochs` | `25` | Training epochs |
| `--max_grad_norm` | `0.5` | Gradient clipping norm |
| `--label_smoothing` | `0.1` | Softens targets to reduce overconfidence |
| `--max_length` | `5000` | Max input tokens |
| `--zero_stage` | `2` | DeepSpeed ZeRO stage |
| `--eval_steps` | `19` | Eval cadence — retune to your steps-per-epoch |
| `--save_steps` | `19` | Checkpoint cadence — retune to your steps-per-epoch |
| `--seed` | `42` | Random seed |

### `prepare_model_for_hf.py`

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `best_model` | Trained checkpoint dir (with `score.1.weight`) |
| `--output` | `best_model_hf_ready` | Output dir for the HF-ready model |

### `inference_telelogs.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model_path` | `best_model_hf_ready` | HF-ready checkpoint dir |
| `--test_file` | `data/classification_sample.csv` | Labeled test CSV |
| `--output_dir` | `inference_results` | Where result JSONs are written |
| `--max_length` | `5000` | Max input tokens |
| `--batch_size` | `8` | Inference batch size |
| `--sample_fraction` | `0.2` | Fraction of the test set to evaluate |

### `make_classification_sample.py`

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `../../llm/deepspeed_standalone/OTel_LLM_sample_10.jsonl` | Chat JSONL with `messages` + `flow` |
| `--out` | `data/classification_sample.csv` | Output CSV path |
| `--max_chars` | `2000` | Truncate the user-turn text to this length |

## Output

- **Training:** checkpoints + TensorBoard logs under `--output_dir`; the best model (by
  `accuracy`) and the tokenizer are saved there at the end. Each eval logs accuracy, macro
  F1/precision/recall, and the per-class prediction distribution (including zeros), which
  is how you spot a collapsed classifier.
- **Prepare:** an HF-ready model folder (`--output`) loadable with plain `from_pretrained`.
- **Inference:** metrics on stdout plus
  `inference_results/inference_results_<timestamp>.json` with per-record predictions, the
  label maps, and summary metrics.

## Notes

- **Keep `--zero_stage 2`.** Stage 3 trains correctly but writes no `model.safetensors`:
  `build_deepspeed_config()` emits no stage-3 keys, so `stage3_gather_16bit_weights_on_model_save`
  is never set and the HF save becomes a silent no-op (the run still exits 0). If you
  already have a stage-3 run, recover the weights with
  `python <out>/checkpoint-*/zero_to_fp32.py <checkpoint-dir> <out.safetensors>`.
- Budget disk: `load_best_model_at_end=True` makes `Trainer` write a checkpoint at the end
  of training even when `--save_steps` exceeds the total step count — roughly 17 GB per
  ZeRO-2 run on a 1B-class model (model + sharded optimizer + tokenizer).
