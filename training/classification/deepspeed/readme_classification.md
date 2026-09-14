# `train_llm_classification.py`

## Overview & when to use

Single-file **sequence-classification** trainer: full fine-tunes a decoder model as a
multi-class classifier (via `AutoModelForSequenceClassification`) on a labeled CSV,
using HF Transformers + DeepSpeed (ZeRO-2). This is a classifier, not a generative SFT
trainer — for text generation use `../../llm/deepspeed_standalone/`.

The folder is a three-step pipeline:

1. **Train** — `train_llm_classification.py` saves a checkpoint whose head is a
   `nn.Sequential(Dropout, Linear)`, so its weight is stored as `score.1.weight`.
2. **Prepare** — `python prepare_model_for_hf.py` renames `score.1.weight` →
   `score.weight` (and drops the parameter-free dropout entries), producing an HF-ready
   folder that loads with a plain `AutoModelForSequenceClassification.from_pretrained(...)`
   — no custom code.
3. **Infer / evaluate** — `python inference_telelogs.py` loads the HF-ready checkpoint,
   evaluates it on a labeled test CSV, prints accuracy / macro F1 / precision / recall,
   and writes per-record predictions to a timestamped JSON. Runs on a single GPU.

Design notes:

- **Sequence classification, not generation** — input is raw text, no chat template;
  `num_labels` is inferred from the data.
- **Left padding** — decoder-based classification reads the last non-pad token, so the
  tokenizer uses `padding_side="left"` and falls back to `eos` for the pad token.
- **Custom head** — the default score layer is replaced with `Dropout(0.1) →
  Linear(hidden, num_labels, bias=False)` to reduce overfitting; the linear is
  normal-initialized (std 0.01).
- **Full fine-tuning** — all parameters train in a single phase (no freeze/unfreeze,
  which DeepSpeed handles poorly mid-run). Gradient checkpointing is on to save memory,
  so `use_cache` is disabled.
- **DeepSpeed config in-memory** — `build_deepspeed_config(zero_stage)` returns the
  config dict passed to `TrainingArguments(deepspeed=...)`; no file is written, so every
  rank gets an identical config and `"auto"` fields are resolved by the HF Trainer at
  runtime.
- **Metrics** — accuracy plus macro F1/precision/recall; each eval logs the prediction
  distribution across all classes (including zeros) to catch collapsed classifiers.
- **Best-model selection** — `load_best_model_at_end` with
  `metric_for_best_model="accuracy"`.

## Install

Python 3.12. Install into a venv:

```bash
python3.12 -m venv ~/.venv && source ~/.venv/bin/activate
pip install -r requirements_classification.txt
```

The commands below refer to a couple of machine-specific locations through environment
variables — set them to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

`evaluate` and `scikit-learn` are required for the metrics and are installed by the
requirements file. Verify the import graph:

```bash
python -c "import torch, deepspeed, transformers, datasets, evaluate, sklearn; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### NVIDIA (CUDA)

flash-attn builds against the CUDA toolkit — export its paths first if you want it:

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

flash-attn is **optional**: `train_llm_classification.py` selects
`attn_implementation="flash_attention_2"` only when `flash_attn` is importable and falls
back to `"sdpa"` otherwise, which is sufficient for this pipeline. On CUDA 13 the pinned
`torch==2.11.0` resolves to a native CUDA 13 wheel on plain PyPI, so no `--index-url` is
needed. Install torch first, then the requirements without the `torch` / `flash-attn`
lines, so the resolver does not replace it:

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch==2.11.0 numpy
python -c "import torch;print(torch.__version__, torch.version.cuda)"
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_cuda.txt
pip install -r /tmp/reqs_cuda.txt            # deepspeed 0.19.4 installs clean, no nvcc build
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: not clobbered
```

> deepspeed 0.19.4 installs as a pure-Python wheel (ops are JIT-only, and ZeRO-2 bf16 needs
> none), on CUDA and ROCm alike — no `DS_BUILD_*` flags and no toolkit needed at install time.

If `evaluate.load` fails behind a proxy that 403s huggingface.co, the metric *scripts* are
simply not cached. Fetch them once with the proxy unset:

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
python -c "import evaluate;[evaluate.load(m) for m in ('accuracy','f1','precision','recall')]"
```

### AMD (ROCm)

Install torch from the ROCm index **before** the rest of the requirements:

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # rocm7.1 also published
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_rocm.txt
pip install -r /tmp/reqs_rocm.txt        # includes deepspeed==0.19.4 — installs cleanly, no hipcc build
```

`deepspeed==0.19.4` installs and initializes on ROCm with no env flags (no
`DS_BUILD_OPS=0`). Do **not** install `flash-attn` — the pinned build is CUDA-only;
`train_llm_classification.py` auto-falls back to `"sdpa"` when `flash_attn` is not
importable, so nothing needs editing.

Two ROCm guards are already in `train_llm_classification.py` and need no attention:

- **`tf32=True` crashes on ROCm** (`ValueError: --tf32 requires Ampere or a newer GPU
  arch`). The script passes `tf32=torch.version.cuda is not None`.
- **flash-attn is selected only when importable**, else `sdpa`.

Accelerate logs `ROCm + DeepSpeed + bf16 detected: setting communication_data_type='fp32'`
once per rank. Benign.

## `--zero_stage 3` silently produces no `model.safetensors`

**This is hardware-neutral and reproduces at a single GPU.** A ZeRO-3 run exits **0**, logs
`Model weights saved in .../model.safetensors`, and that file does not exist on disk. The
give-away in the log is:

```
Writing model shards: 0it [00:00, ?it/s]        # zero tensors -> transformers writes no file
```

**Root cause:** this folder's `build_deepspeed_config()` (top of
`train_llm_classification.py`) emits a `zero_optimization` block with no stage-3 keys — in
particular it never sets **`stage3_gather_16bit_weights_on_model_save: True`**. Under
stage 3 the parameters live sharded across ranks even at world size 1, so without that flag
DeepSpeed hands `Trainer` an empty state dict and the HF save becomes a no-op. Compare
`../../llm/deepspeed/utils.py`, whose `build_deepspeed_config` *does* set that key inside an
`if zero_stage == 3:` branch.

Workarounds:

- **Use `--zero_stage 2`** (the script default). It writes a directly loadable
  `model.safetensors` at every world size tested.
- If you need stage 3, the training itself is sound — loss, eval metrics and the per-rank
  shards are all correct. Recover the weights afterwards with the `zero_to_fp32.py` that
  DeepSpeed writes into the checkpoint:
  `python <out>/checkpoint-*/zero_to_fp32.py <checkpoint-dir> <out.safetensors>`.
- The permanent fix is to add the stage-3 keys to `build_deepspeed_config`, mirroring
  `../../llm/deepspeed/utils.py`.


## Environment & secrets

Optional `dev.env` next to the script, loaded via `load_dotenv("dev.env")` — only needed
for gated models on the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

## Data

`--train_file` / `--test_file` are CSVs. Each row needs a **text** column (`text`, or
`question` — auto-renamed to `text`) and a **label** column named `label` or `answer`
(string classes). The script derives the class set from the training split, builds
`label2id`/`id2label`, and maps labels to integer ids — the number of classes is
inferred from the data, so nothing needs to be set.

**Shipped sample:** `data/classification_sample.csv` — 10 rows, columns `text,label`,
4 classes (`doc_direct`, `key_facts`, `extractive_summary`, `detailed_summary`). It is
derived by the checked-in `make_classification_sample.py` from
`../../llm/deepspeed_standalone/OTel_LLM_sample_10.jsonl` (a chat JSONL): the first
user-turn content becomes `text` (truncated to 2000 chars) and the row's `flow` field
becomes `label`. Regenerate it any time with:

```bash
python make_classification_sample.py
```

The shipped sample only proves the pipeline runs — swap in your own train/test CSVs with
the same column contract for real training:

```bash
... train_llm_classification.py --train_file /path/to/train.csv --test_file /path/to/test.csv
```

## Run

The DeepSpeed config is built in-memory from `--zero_stage` (no `ds_config.json` file);
the accelerate config only needs `distributed_type: DEEPSPEED`.

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_classification.py --num_epochs 1 --eval_steps 1 --save_steps 1
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_classification.py \
  --train_file /path/to/train.csv --test_file /path/to/test.csv \
  > train_llm_classification.log 2>&1 &

tail -f train_llm_classification.log
```

**What "working" looks like:** the label map and class count are logged, then per-step
`{'loss': ...}` lines and periodic eval reports (accuracy / macro F1 / precision /
recall, plus a per-class prediction distribution). The best checkpoint (by accuracy) is
saved to `--output_dir` at the end.

Then prepare and evaluate:

```bash
python prepare_model_for_hf.py --source models/rnj-1-classifier --output best_model_hf_ready
python inference_telelogs.py --model_path best_model_hf_ready --test_file /path/to/test.csv
```

### Model choice

The script default `EssentialAI/rnj-1` is a ~12B decoder. Under ZeRO-2 a single-GPU full
fine-tune OOMs an 80 GB card — use multi-GPU ZeRO-3 (with the save fix above) or
optimizer/parameter offload for that model.

`google/gemma-3-1b-it` is the model the smoke commands below use. It is gated, so
`HF_TOKEN` from `dev.env` is required to download it.

`google/gemma-4-E4B-it` is **not** usable here — `Gemma4Config` is not in
`AutoModelForSequenceClassification`'s registry in transformers 5.5.0.

### Expected output

Single GPU, `google/gemma-3-1b-it`, shipped 10-row sample, `--batch_size 1 --grad_accum 1
--num_epochs 5` → global batch 1 → **50 optimizer steps**:

```
{'loss': '3.216', 'grad_norm': '301.3', 'learning_rate': '7e-06',    'epoch': '0.3'}
{'loss': '0.5745','grad_norm': '32.09', 'learning_rate': '5.048e-06','epoch': '2'}
{'loss': '0.3691','grad_norm': '13.19', 'learning_rate': '7.494e-09','epoch': '5'}
{'eval_loss': '0.4418', 'eval_accuracy': '1', 'eval_f1': '1', 'eval_precision': '1', 'eval_recall': '1', 'epoch': '5'}
Writing model shards: 100%|██████████| 1/1 [00:00<00:00, 1.34it/s]     # 1 shard -> real file
Model weights saved in .../model.safetensors
{'train_runtime': '64.17', 'train_samples_per_second': '0.779', 'train_loss': '0.7756'}
```

The eval file is the train file, so `eval_accuracy 1.0` shows the head is learning the
signal — it is **not** a generalization score.

**Check that the DeepSpeed engine really ran** (rather than a plain-Trainer fallback) by
looking for the ZeRO artifacts in the checkpoint:

```
checkpoint-50/global_step50/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
checkpoint-50/zero_to_fp32.py
model.safetensors                    # score.1.weight present = the trained head
```

### Multi-GPU

Scales to 4 and 8 GPUs with **no code change** — only the launch flags and the data slice
change.

```bash
source .env_deepspeed/bin/activate
# If activate ends with a stale device pin from an earlier single-GPU session, override it
# AFTER sourcing or you will silently train on one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"

accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29660 \
  train_llm_classification.py \
  --model_id google/gemma-3-1b-it --zero_stage 2 \
  --train_file <replicated slice> --test_file data/classification_sample.csv \
  --output_dir $OUTPUT_DIR/train_llm_classification/model_8gpu \
  --num_epochs 1 --batch_size 4 --grad_accum 1 --max_length 512 \
  --warmup_steps 2 --eval_steps 10 --save_steps 1000
```

Use a distinct `--main_process_port`; 29500 collides on a shared box.

**The data slice has to be big enough.** The shipped `data/classification_sample.csv` is
10 rows, which cannot feed 8 ranks — `DistributedSampler` pads 10 to 16 and gives about one
micro-batch per rank. Replicate it (e.g. x64 = 640 rows, written outside the repo) so that
per-device 4 x grad_accum 1 x 8 GPUs = global batch 32 yields 20 real optimizer steps. That
is a *pipeline* proof, not a learning result: the eval file's rows are contained in the
replicated train slice. For a real run supply `--train_file` / `--test_file` with disjoint
data, at least a few hundred rows so every rank gets full batches.

**Checkpointing is not optional.** `--save_steps` larger than the total step count does
*not* stop the final save: `load_best_model_at_end=True` makes `Trainer` write a checkpoint
at the end of training regardless, and with ZeRO-2 on this model that is ~17 GB per run
(2 GB model + 12 GB sharded optimizer + tokenizer). Budget disk, or set
`load_best_model_at_end=False` for smoke runs.

## Arguments

### `train_llm_classification.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model_id` | `EssentialAI/rnj-1` | HF repo id or local path of the base model |
| `--output_dir` | `models/rnj-1-classifier` | Where checkpoints and the best model are written |
| `--train_file` | `data/classification_sample.csv` | Training CSV (text/question + label/answer columns) |
| `--test_file` | `data/classification_sample.csv` | Evaluation CSV with the same columns |
| `--learning_rate` | `7e-6` | Optimizer learning rate |
| `--batch_size` | `4` | Per-device batch; effective batch = batch × grad_accum × num_GPUs |
| `--grad_accum` | `2` | Gradient accumulation steps |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `150` | LR warmup steps |
| `--num_epochs` | `25` | Training epochs |
| `--max_grad_norm` | `0.5` | Gradient clipping norm |
| `--label_smoothing` | `0.1` | Softens targets to reduce overconfidence |
| `--max_length` | `5000` | Max input tokens |
| `--zero_stage` | `2` | DeepSpeed ZeRO stage |
| `--eval_steps` | `19` | Eval cadence — tune to your steps-per-epoch |
| `--save_steps` | `19` | Checkpoint cadence |
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

- **Training:** checkpoints + TensorBoard logs under `--output_dir`; the best model and
  tokenizer are saved there at the end.
- **Prepare:** an HF-ready model folder (`--output`) loadable with plain
  `from_pretrained`.
- **Inference:** metrics on stdout and
  `inference_results/inference_results_<timestamp>.json` with per-record predictions,
  the label maps, and summary metrics.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1, 4 and 8 GPUs |
| PyTorch | `torch==2.11.0` resolves to a `+cu130` wheel on plain PyPI — no `--index-url` | `torch==2.11.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2` |
| DeepSpeed | `deepspeed==0.19.4`, pure-Python wheel, no nvcc build | `deepspeed==0.19.4`, pure-Python wheel, no hipcc build |
| ZeRO | stage 2 (stage 3 hits the no-save issue above) | stage 2 at 1/4/8 ranks; stage 3 trains but does not save |
| Attention | `flash_attention_2` when `flash_attn` imports, else `sdpa` | `sdpa` |
| tf32 | enabled by the `tf32=torch.version.cuda is not None` guard | disabled by the same guard (it raises on ROCm) |

The two guards in `train_llm_classification.py` (tf32, sdpa fallback) are what make the
script hardware-neutral; they no-op on CUDA.

## Notes

- Retune the eval/checkpoint cadence defaults (`--eval_steps`/`--save_steps` = 19) for your
  dataset size.
- The 10-row shipped sample is for pipeline validation only; with `--test_file` equal to
  the train file, eval numbers are meaningless as a quality signal.
- `inference_telelogs.py` works on any CSV matching the column contract, not just TeleLogs.
