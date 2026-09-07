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

DeepSpeed and flash-attn build against the CUDA toolkit — export its paths first:

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The scripts use `attn_implementation="flash_attention_2"`, so `flash-attn` must be
installed; if FA2 is unavailable, change it to `"sdpa"` in the scripts.

### AMD (ROCm)

HF Transformers runs on AMD GPUs via the official ROCm torch wheels — PyTorch publishes
pip wheels for ROCm-capable Linux systems (see
[pytorch.org — Get Started, "With ROCm"](https://pytorch.org/get-started/locally/)).
Install torch from the ROCm index **before** the rest of the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # rocm7.1 also published
pip install -r requirements_classification.txt   # comment out flash-attn first
```

DeepSpeed on ROCm: DeepSpeed supports AMD GPUs — its requirements list "a CUDA or ROCm
compiler such as nvcc or hipcc", its contributed hardware support includes AMD MI100 and
MI200, and an AMD MI200 workflow runs in DeepSpeed's own CI (see the
[DeepSpeed README](https://github.com/deepspeedai/DeepSpeed#readme)). In practice (see
the MI355X section below) `pip install deepspeed==0.19.4` builds a pure-Python wheel with
no hipcc compilation at install time — ops are JIT-compiled only if needed, and ZeRO-2
bf16 needs none. `flash-attn` as pinned here is a CUDA-only build — skip it;
`train_llm_classification.py` now auto-falls back to `"sdpa"` when `flash_attn` is not
importable (no manual edit needed).

### Platform notes — AMD MI355X (ROCm 7.2)

Verified end-to-end on an AMD Instinct MI355X (gfx950, 288 GB), ROCm 7.2.4, Python
3.12.3, single GPU. Exact install:

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_rocm.txt
pip install -r /tmp/reqs_rocm.txt        # includes deepspeed==0.19.4 — installs cleanly, no hipcc build
```

Exact smoke command (single GPU; scale `--num_processes` up for more):

```bash
accelerate launch --num_processes=1 --mixed_precision=bf16 --use_deepspeed \
  train_llm_classification.py --model_id google/gemma-3-1b-it \
  --num_epochs 1 --eval_steps 5 --save_steps 5 \
  --batch_size 1 --grad_accum 1 --max_length 512 --warmup_steps 2
```

**Expected output** (finite loss, real DeepSpeed ZeRO-2 engine, eval metrics, model saved):

```
INFO - accelerate.utils.dataclasses - ROCm + DeepSpeed + bf16 detected: setting `communication_data_type='fp32'` to avoid bf16 overflow corrupting weights.
{'loss': '2.396', 'grad_norm': '431.6', 'learning_rate': '0', 'epoch': '0.1'}
{'eval_loss': '0.9858', 'eval_accuracy': '0.6', 'eval_f1': '0.6095', 'eval_precision': '0.725', 'eval_recall': '0.7292', 'epoch': '1'}
INFO - __main__ - Training completed in 0h 0m 41s
```

Checkpoints contain `global_step*/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt` +
`zero_to_fp32.py` — proof the DeepSpeed ZeRO engine (not a plain-Trainer fallback) ran.
A 5-epoch run converged (loss 2.4 → 0.37, eval accuracy 1.0 on the 10-row sample).

ROCm quirks found and fixed in `train_llm_classification.py`:

- **`tf32=True` crashes on ROCm** — `TrainingArguments` raised
  `ValueError: --tf32 requires Ampere or a newer GPU arch`. TF32 is NVIDIA-only; the
  script now passes `tf32=torch.version.cuda is not None` (True on CUDA, False on ROCm).
- **flash-attn hardcode** — the script now selects `flash_attention_2` only when the
  `flash_attn` package is importable, else `sdpa`; nothing to edit by hand on AMD.
- **DeepSpeed on ROCm 7.2 / gfx950: works as-is.** No env flags (no `DS_BUILD_OPS=0`)
  and no version change were needed; `deepspeed==0.19.4` installed and initialized
  first try. Accelerate auto-sets DeepSpeed `communication_data_type='fp32'` on
  ROCm+bf16 (benign, logged).
- Model note: `google/gemma-4-E4B-it` is **not** usable here —
  `Gemma4Config` is not in `AutoModelForSequenceClassification`'s registry in
  transformers 5.5.0. `google/gemma-3-1b-it` (used above) and other registered decoder
  architectures work.

This path works on MI355X with the two small script fixes above (tf32 guard, sdpa
fallback); the stack itself (torch 2.11.0+rocm7.2, deepspeed 0.19.4, transformers
5.5.0) needed no patches.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

This path scales cleanly to 8 GPUs with no further code changes: the same
recipe, same model (`google/gemma-3-1b-it`), same two ROCm fixes as the 1-GPU run;
only the launch flags and the data slice changed. The run finishes with exit code 0 and
no teardown hang.

Exact launch (if the venv's `activate` ends with a stale `CUDA_VISIBLE_DEVICES` pin from
an earlier single-GPU session, **override it after sourcing or you will silently train on
one GPU**):

```bash
source .env_deepspeed/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # must come AFTER the source
python -c "import torch; assert torch.cuda.device_count()==8"

accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29660 \
  train_llm_classification.py \
  --model_id google/gemma-3-1b-it \
  --train_file <640-row slice> --test_file data/classification_sample.csv \
  --output_dir $OUTPUT_DIR/train_llm_classification/gpu8/model_8gpu \
  --num_epochs 1 --batch_size 4 --grad_accum 1 --max_length 512 \
  --warmup_steps 2 --eval_steps 10 --save_steps 1000
```

**Parallelism:** DeepSpeed **ZeRO stage 2**, world size 8, one process per GPU.
Batch geometry: per-device 4 x grad_accum 1 x 8 GPUs = **global batch 32**;
640 train rows / 32 = **20 optimizer steps**.

```
[assert] torch.cuda.device_count() = 8
[assert] gpu0: AMD Instinct MI355X gcn=gfx950:sramecc+:xnack- 288GiB      (... gpu1-gpu7 identical)
***** Running training *****
  Num examples = 640 | Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Total optimization steps = 20 | Number of trainable parameters = 999,890,560
{'loss': '17.99', 'grad_norm': '868.7', 'learning_rate': '0', 'epoch': '0.05'}
{'loss': '3.096', 'grad_norm': '51.22', 'learning_rate': '6.789e-06', 'epoch': '0.25'}
{'loss': '2.849', 'grad_norm': '16.53', 'learning_rate': '5.317e-08', 'epoch': '1'}
Pred Distribution (all 4 classes): {0: 1, 1: 4, 2: 2, 3: 3}
{'eval_loss': '3.019', 'eval_accuracy': '1', 'eval_f1': '1', ..., 'epoch': '0.5'}
{'eval_loss': '2.806', 'eval_accuracy': '1', 'eval_f1': '1', ..., 'epoch': '1'}
{'train_runtime': '18.69', 'train_samples_per_second': '34.25', 'train_steps_per_second': '1.07', 'train_loss': '5.525'}
```

Loss is finite and decreasing (17.99 -> 2.85), eval loss falls between the two evals,
and the head's prediction distribution `{0:1, 1:4, 2:2, 3:3}` exactly reproduces the
true class counts of the 10-row eval set — the classification head is really training.

**8-GPU residency — `rocm-smi` sampled *inside* the run** (run the sampler in-band, in the
background of the training script itself, so the sample is provably contemporaneous with
this job and not a neighbouring one). A healthy mid-training sample looks like:

```
GPU[0] VRAM Total Used Memory (B): 11910205440      GPU[4] ...: 12686188544
GPU[1] ...: 12803596288   GPU[2] ...: 12816175104   GPU[5] ...: 12501614592
GPU[3] ...: 12791042048                             GPU[6] ...: 12539387904
                                                    GPU[7] ...: 12518383616
----- rocm-smi --showpids -----
8 x python3   12 GB each        # one training process per GPU
  pt_elastic  0                 # the accelerate launcher
```

All 8 GPUs hold **~11.6-12.8 GB each** (~4% of the 288 GB HBM — this model is far from
memory-bound), and the 8 VRAM-holding PIDs are exactly this run's own
`train_llm_classification.py` children (cross-check them against `pgrep -f` in the same
sample), parented by the launcher. GPU utilisation sampled 24-26% on all eight
simultaneously.

**ZeRO-2 really is active at 8 ranks** (not a silent DDP fallback) — the step-20
checkpoint contains eight per-rank optimizer partitions:

```
checkpoint-20/global_step20/bf16_zero_pp_rank_{0..7}_mp_rank_00_optim_states.pt   1,499,842,885 B each
```

8 x 1.4998 GB = 11.999 GB, which is **exactly** the size of the single
`bf16_zero_pp_rank_0_...pt` (11,998,719,393 B) written by the 1-GPU reference run — a
clean 8-way partition of the optimizer state.

**Throughput vs 1 GPU** (same per-device batch 4, same `--max_length 512`, same model,
measured back to back):

| | world size | global batch | steps | `train_samples_per_second` |
|---|---|---|---|---|
| 1-GPU reference | 1 | 4  | 40 | 6.216 |
| 8-GPU run       | 8 | 32 | 20 | **34.25** |

**5.5x** on 8 GPUs. The gap to a linear 8x is dominated by fixed startup inside a
20-step run — the first step alone costs 10.35 s (RCCL rendezvous + ZeRO init), after
which steps settle to ~4.6 it/s. This is a smoke test, not a scaling benchmark.

What differed from the 1-GPU run:

- **Launch only:** `--num_processes=8` + `--main_process_port 29660`. No code change, no
  new package, no pin change — `requirements_classification.txt` is unchanged.
- **Data slice.** The shipped `data/classification_sample.csv` is **10 rows**, which
  cannot feed 8 ranks (`DistributedSampler` would pad 10 -> 16 and give ~1 micro-batch
  per rank). The run above used a **x64 replication of that same 10-row sample (640
  rows)**, written outside the repo, purely to get a meaningful 20 steps at global batch
  32. This is a *pipeline* proof, not a learning result.
- **Honest caveat on the eval metric:** the eval file is the original 10-row sample,
  whose rows are contained in the replicated train slice. `eval_accuracy: 1.0` therefore
  shows the head is learning the training signal — it is **not** a generalization score.
  For a real 8-GPU run, supply `--train_file`/`--test_file` with disjoint real data
  (>= a few hundred rows so every rank gets full batches).
- **Checkpointing left on by the framework.** `--save_steps 1000` (> total steps) does
  *not* stop the final save: `load_best_model_at_end=True` makes `Trainer` write a
  checkpoint at the end of training regardless, and with ZeRO-2 that is ~17 GB per run
  (2 GB model + 12 GB sharded optimizer + tokenizer). Budget disk accordingly, or set
  `load_best_model_at_end=False` for smoke runs.
- Accelerate again logged `ROCm + DeepSpeed + bf16 detected: setting
  communication_data_type='fp32'` — once per rank, benign, same as at 1 GPU.

### 4-GPU sharding run — **ZeRO-2 vs ZeRO-3** (4×MI355X, ROCm 7.2.4)

> Extends, and does not contradict, the 1-GPU and 8-GPU sections above — both of those
> ran **ZeRO stage 2 only**. This run repeats the recipe at 4 ranks and additionally
> exercises `--zero_stage 3`.

Verified on physical GPUs **4,5,6,7** of a node whose other four GPUs were held by a
sibling job, same venv and pins, `google/gemma-3-1b-it`, 640-row replicated slice, 40 steps.

ZeRO-2 holds at 4 GPUs (rc=0, correct output). ZeRO-3 trains correctly but
silently writes NO HuggingFace weights — see the finding below. Prefer `--zero_stage 2`
here until `build_deepspeed_config` is fixed.

```bash
source .env_deepspeed/bin/activate
export HIP_VISIBLE_DEVICES=4,5,6,7      # must come AFTER the source (stale pin, see above)
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==4"

accelerate launch --num_processes=4 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29794 \
  train_llm_classification.py \
  --model_id google/gemma-3-1b-it --zero_stage 3 \
  --train_file <640-row replicated slice> --test_file data/classification_sample.csv \
  --output_dir $OUTPUT_DIR/train_llm_classification_4gpu/zero3 \
  --num_epochs 1 --batch_size 4 --grad_accum 1 --max_length 512 \
  --warmup_steps 2 --eval_steps 10 --save_steps 1000
```

(The ZeRO-2 control was the identical command with `--zero_stage 2` and port 29795.)

**What a healthy 4-rank run looks like under either stage:**

```
[assert] device_count= 4
{'loss': '9.566', 'grad_norm': '560.2', 'learning_rate': '0',        'epoch': '0.025'}
{'loss': '1.623', 'grad_norm': '57.63', 'learning_rate': '6.952e-06','epoch': '0.125'}
{'loss': '1.809', 'grad_norm': '52.64', 'learning_rate': '6.43e-06', 'epoch': '0.25'}
{'eval_loss': '1.402', 'eval_accuracy': '1', 'eval_f1': '1', 'eval_precision': '1', 'eval_recall': '1', 'epoch': '1'}
Training completed in 0h 0m 54s        # rc=0, no hang, no rank divergence
```

`rocm-smi` sampled every 4 s *during* training (all four owned GPUs):

```
GPU[4]: GPU use (%): 99    VRAM Total Used Memory (B): 6877257728   # 6.4 GiB
GPU[5]: GPU use (%): 99    VRAM Total Used Memory (B): 6944395264   # 6.5 GiB
GPU[6]: GPU use (%): 98    VRAM Total Used Memory (B): 6877290496   # 6.4 GiB
GPU[7]: GPU use (%): 98    VRAM Total Used Memory (B): 6877257728   # 6.4 GiB
```

ZeRO-3 confirmed active at 4 ranks — the checkpoint holds a *four*-way partition of both
optimizer and model state (stage 2 writes only the `bf16_zero_pp_*optim*` set):

```
global_step40/bf16_zero_pp_rank_{0,1,2,3}_mp_rank_00_optim_states.pt   2,999,677,061 B each
global_step40/zero_pp_rank_{0,1,2,3}_mp_rank_00_model_states.pt
```

#### ⚠️ Finding: `--zero_stage 3` silently produces no `model.safetensors`

The ZeRO-3 run exits **0**, logs `Model weights saved in .../checkpoint-40/model.safetensors`,
and that file **does not exist on disk**. The give-away in the log is:

```
Writing model shards: 0it [00:00, ?it/s]        # zero tensors -> transformers writes no file
```

A back-to-back `--zero_stage 2` control run on the same 4 GPUs wrote a correct
**1,999,820,512 B (1.9 GiB)** `model.safetensors` in both `checkpoint-40/` and the output
root — so this is **specific to stage 3, not a 4-GPU regression**.

**Root cause:** this folder's `build_deepspeed_config()` (top of
`train_llm_classification.py`) emits a `zero_optimization` block with no stage-3 keys —
in particular it never sets **`stage3_gather_16bit_weights_on_model_save: True`**. Under
stage 3 the parameters live sharded across ranks, so without that flag DeepSpeed hands
`Trainer` an empty state dict and the HF save becomes a no-op. Compare
`../../llm/deepspeed/utils.py`, whose `build_deepspeed_config` *does* set that key
inside an `if zero_stage == 3:` branch — and which correctly wrote a consolidated 14.9 GiB
checkpoint under ZeRO-3 on these same 4 GPUs.

**Consequences and workarounds (no code was changed here):**

- Use **`--zero_stage 2`** (the script default) — it is what both prior sections tested
  and it produces directly loadable weights.
- If you need stage 3, recover the weights after the fact with the
  `zero_to_fp32.py` DeepSpeed writes into the output dir, which reconstructs a full
  checkpoint from the `global_step40/` shards. The training itself is not corrupted —
  loss, eval metrics and the per-rank shards are all sound.
- The permanent fix is to add the stage-3 keys to `build_deepspeed_config`, mirroring
  `../../llm/deepspeed/utils.py`.

Other notes at 4 ranks: no RCCL tuning, no new packages, no code change; ports
29794/29795 (29500 collides on a shared box); per-GPU VRAM even across all four
(6.4-6.5 GiB, ~0.1 GiB spread) and utilisation 98-99% on all four simultaneously.

### Platform notes — NVIDIA H100 80GB (CUDA 13.0)

This path works as documented on H100 (single-GPU, DeepSpeed ZeRO-2). The stack installs
and runs with **zero code changes** — the two guards added for MI355X (`tf32` on CUDA
only, `sdpa` fallback when `flash_attn` is absent) are exactly what a plain-CUDA box wants,
so they no-op correctly here. Verified end-to-end: train (ZeRO-2) → `model.safetensors`
with real weights → `prepare_model_for_hf.py` → `inference_telelogs.py` accuracy.
This validation was **single-GPU only** (the node's other GPUs were held by a co-tenant
job; see the MI355X 8-GPU / 4-GPU sections for what a multi-rank pass looks like).

Verified on one NVIDIA H100 80GB HBM3 (Hopper cc 9.0), driver **580.173.02**, CUDA 13.0,
Python 3.12.3, single physical GPU (`CUDA_VISIBLE_DEVICES=5`). Key versions: **torch
2.13.0+cu130**, deepspeed 0.19.4, transformers 5.5.0, accelerate 1.14.0, datasets 4.3.0.

Exact install (CUDA — no `--index-url`; the default PyPI wheel is already CUDA-13 native):

```bash
python3 -m venv .env_deepspeed && source .env_deepspeed/bin/activate
pip install torch numpy                      # -> torch 2.13.0+cu130 (verify below)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
# install the rest WITHOUT the torch/flash-attn pins (flash-attn is optional, see quirks):
grep -vE '^(flash-attn|torch)==' requirements_classification.txt | grep -vE '^\s*#' > /tmp/reqs_cuda.txt
pip install -r /tmp/reqs_cuda.txt            # deepspeed 0.19.4 installs clean, no nvcc build
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: NOT clobbered
```

> deepspeed 0.19.4 installs as a pure-Python wheel (ops JIT-only; ZeRO-2 bf16 needs none),
> exactly as on ROCm — no `DS_BUILD_*` flags, no CUDA toolkit needed at install time.

**Model note.** The script default `EssentialAI/rnj-1` is a **`Gemma3ForCausalLM` ~12B
decoder** (7 safetensors shards, hidden 4096). Full fine-tuning that at world-size 1 does
**not fit in 80 GB** — ZeRO-2 shards optimizer state but **not** parameters/grads, so a
1-GPU run holds the whole model + grads + Adam states on one card and OOMs. This validation
used `google/gemma-3-1b-it` (the exact model the MI355X sections used, and a decoder that matches
this pipeline's `.score`-head + left-padding contract). It is not in the shared HF cache;
download it once with the proxy unset (gated → needs `HF_TOKEN` from `dev.env`):
`unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy` then `huggingface-cli download
google/gemma-3-1b-it`. On this 80 GB card the 1B model + ZeRO-2 optimizer peaks at **~24 GB**
(see the residency sample below) — for the 12B `rnj-1` default you would need multi-GPU
ZeRO-3 (with the save fix below) or optimizer/param offload.

Exact smoke command (single GPU, port 29659, GPU 5 only):

```bash
export CUDA_VISIBLE_DEVICES=5                 # plain CUDA — no HIP_VISIBLE_DEVICES on NVIDIA
export HF_HOME=/path/to/hf_cache              # HF model cache (see "Install" above)
export HF_DATASETS_CACHE=/dev/shm/dscache_classification
accelerate launch --num_processes=1 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29659 \
  train_llm_classification.py \
  --model_id google/gemma-3-1b-it --zero_stage 2 \
  --output_dir <out>/model_zero2 \
  --num_epochs 5 --batch_size 1 --grad_accum 1 --max_length 512 \
  --warmup_steps 2 --eval_steps 10 --save_steps 1000
```

**Batch geometry / step count:** 10 rows × per-device 1 × grad_accum 1 × world 1 = global
batch 1 → **50 optimizer steps** over 5 epochs (`Total optimization steps = 50`,
`Number of trainable parameters = 999,890,560`). Non-trivial; every step is a real update.

**Expected output** (finite, **decreasing** loss; real ZeRO-2 engine; eval metrics; weights saved):

```
{'loss': '3.216', 'grad_norm': '301.3', 'learning_rate': '7e-06',    'epoch': '0.3'}
{'loss': '0.5745','grad_norm': '32.09', 'learning_rate': '5.048e-06','epoch': '2'}
{'loss': '0.3691','grad_norm': '13.19', 'learning_rate': '7.494e-09','epoch': '5'}
{'eval_loss': '0.4418', 'eval_accuracy': '1', 'eval_f1': '1', 'eval_precision': '1', 'eval_recall': '1', 'epoch': '5'}
Writing model shards: 100%|██████████| 1/1 [00:00<00:00, 1.34it/s]     # 1 shard -> real file
Model weights saved in .../model_zero2/model.safetensors
{'train_runtime': '64.17', 'train_samples_per_second': '0.779', 'train_steps_per_second': '0.779', 'train_loss': '0.7756'}
```

Loss ramps with the warmup LR (0.78→3.2) then falls to ~0.35-0.37 as LR decays; the head's
prediction distribution `{0:1, 1:4, 2:2, 3:3}` exactly reproduces the true class counts of
the 10-row set. (Eval file == train file, so `eval_accuracy 1.0` shows the head is learning
the signal, **not** a generalization score — same honest caveat as the MI355X runs.)

**The saved weights are real (this is the whole point given the ZeRO-3 trap below):**

```
-rw------- 1999820512  model_zero2/model.safetensors      # 1.9 GiB, 341 tensors, 999,890,560 params
                                                           # score.1.weight [4, 1152] present (the trained head)
model_zero2/checkpoint-50/global_step50/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt   11,998,719,393 B
model_zero2/checkpoint-50/zero_to_fp32.py                  # => real DeepSpeed ZeRO engine, not a Trainer fallback
```

Both the 1.9 GiB `model.safetensors` and the 12 GiB `bf16_zero_pp_*optim*` shard are
**byte-for-byte the same size as the MI355X 1-GPU reference** — same model, same ZeRO-2
optimizer partition.

**GPU residency — `nvidia-smi -i <gpu>` sampled *in-band* during the run** (run a background
sampler in the launching shell, so the sample is provably contemporaneous with the job and
scoped to the physical GPU in use; the card should read **0 MiB** immediately before launch):

```
util:  0 %   python3  1174 MiB     # warmup / model load
util: 29 %   python3 24636 MiB     # steady-state training
util:  0 %   python3 24638 MiB
```

Peak **~24.6 GB** on the training GPU, held by the run's own `train_llm_classification.py`
child (a `.env_deepspeed/bin/python3`). Co-tenant GPUs are never touched.

**Full pipeline (prepare → infer) also verified on H100:**

```
# prepare_model_for_hf.py: score.1.weight [4,1152] -> score.weight; reloads with plain from_pretrained
# inference_telelogs.py (--sample_fraction 1.0, cuda):
Pred Distribution (all 4 classes): {0: 1, 1: 4, 2: 2, 3: 3}
Accuracy: 1.0000 (100.00%)   F1 (macro): 1.0000   Precision: 1.0000   Recall: 1.0000
```

#### ⚠️ ZeRO-3 silent no-save trap reproduces on H100 (single GPU) — use `--zero_stage 2`

The `--zero_stage 3` bug documented in the 4-GPU MI355X section above is **hardware-neutral
and reachable at a single GPU on H100**: once ZeRO-3 sharding is on, params live sharded
even at world-size 1, and `build_deepspeed_config()` never sets
**`stage3_gather_16bit_weights_on_model_save: True`**, so `Trainer` is handed an empty state
dict and the HF save no-ops **while the run exits 0**. On H100 the identical smoke command
with `--zero_stage 3` reproduced it:

```
Writing model shards: 0it [00:00, ?it/s]                       # ZeRO-3: zero tensors gathered
Model weights saved in .../model_zero3/model.safetensors       # logged, but NO file on disk
$ ls .../model_zero3/model.safetensors  ->  No such file or directory      # rc still 0
```

**Workarounds (no code changed here):**
- **Use `--zero_stage 2`** (the script default and this section's passing smoke) — it writes
  a directly-loadable 1.9 GiB `model.safetensors`.
- If you must use stage 3, recover the weights from the shards DeepSpeed leaves in the
  checkpoint: `python <out>/model_zero3/checkpoint-*/zero_to_fp32.py <checkpoint-dir> <out.safetensors>`
  reconstructs the full tensor set (training itself is sound — loss/eval/shards are all fine).
- Permanent fix: add the stage-3 keys to `build_deepspeed_config`, mirroring
  `../../llm/deepspeed/utils.py` (which sets that flag inside an `if zero_stage == 3:` branch).

Quirks / what differed from the MI355X recipe:
- **No `communication_data_type='fp32'` line** — that accelerate log is ROCm+bf16-specific;
  on CUDA it (correctly) does not appear. bf16 comms are used directly.
- **`tf32` guard engaged as `True`** on CUDA (the MI355X guard `tf32=torch.version.cuda is
  not None` evaluates True here) — no crash, TF32 tensor cores enabled. This is the one
  place the ROCm workaround "reverses" itself automatically.
- **flash-attn not installed** — the pinned `flash-attn==2.8.3` is a source build; it was
  skipped here, so the script's `attn_implementation` auto-selected `sdpa`
  (as on ROCm). SDPA is fully sufficient for this smoke. To use FA2 on H100, add flash-attn
  (prebuilt cu13 wheel or a source build against CUDA 13) and the script picks it up
  automatically — no edit needed.
- **`evaluate` metric cache gap (not a code bug):** `precision`/`recall` metric *scripts*
  were not in the shared HF cache and the box proxy 403s huggingface.co, so `evaluate.load`
  failed until fetched once with the proxy unset (`unset HTTP_PROXY HTTPS_PROXY ...;
  python -c "import evaluate;[evaluate.load(m) for m in ('accuracy','f1','precision','recall')]"`).
  `accuracy`/`f1` were already cached.
- **Model swap** (see "Model note" above): `rnj-1` default is a 12B decoder that OOMs a
  single 80 GB card under ZeRO-2; used `google/gemma-3-1b-it` (downloaded once, proxy unset).
- **VRAM.** 80 GB here vs 288 GB on MI355X; the 1B model peaks ~24.6 GB single-GPU, well
  within budget. The 12B default would not fit single-GPU (needs multi-GPU ZeRO-3 or offload).

**What a multi-GPU pass on H100 would need:** free GPUs, `--num_processes=N`
+ a distinct `--main_process_port`, and a **replicated data slice** (the 10-row sample
can't feed N ranks — the MI355X 8-GPU run used a ×64 = 640-row slice for 20 steps at global
batch 32). Everything else is unchanged. If that run uses `--zero_stage 3`, apply the save
fix above first or its final `model.safetensors` will silently be empty.

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

**Original work:** this trainer was built and tested against the **TeleLogs** dataset
(`tele_logs-train.csv` / `tele_logs-test.csv`, with `question`/`answer` columns). The
shipped sample only proves the pipeline runs — swap in your own train/test CSVs with the
same column contract for real training:

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

## Hardware support & evidence

**Other hardware (upstream claims — not verified here):** DeepSpeed's accelerator abstraction claims Intel XPU, Intel Gaudi (HPU), Ascend NPU, and CPU (upstream accelerator docs).


- **Tested:** Azure cluster, 8×H100 80GB — single-node DeepSpeed ZeRO-2. Multi-node is
  possible via `accelerate`/`torchrun` rendezvous but has **not** been tested here.
- **NVIDIA:** default PyPI torch wheels + CUDA toolkit for the DeepSpeed/flash-attn builds.
- **AMD: tested** — MI355X (gfx950), ROCm 7.2.4, `torch==2.11.0+rocm7.2`,
  `deepspeed==0.19.4` (real ZeRO-2 engine, no fallback), single-GPU **and 8-GPU** smoke
  runs to completion with finite loss and saved checkpoints (8-GPU: ZeRO-2 sharded
  across all 8 ranks, 5.5x throughput). See "Platform notes — AMD MI355X
  (ROCm 7.2)" and "8-GPU run" above for exact commands and the two script fixes it required
  (tf32 guard, sdpa fallback). flash-attn is skipped; attention runs via SDPA.

## Notes

- Eval/checkpoint cadence defaults (`--eval_steps`/`--save_steps` = 19) were tuned to
  the original TeleLogs steps-per-epoch — retune them for your dataset size.
- The 10-row shipped sample is for pipeline validation only; with `--test_file` equal to
  the train file, eval numbers are meaningless as a quality signal.
- `inference_telelogs.py` keeps its historical name — it works on any CSV matching the
  column contract, not just TeleLogs.
