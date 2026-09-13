# `training/llm/lightning` — chat fine-tuning with PyTorch Lightning

> **Hardware coverage:** **AMD Instinct MI355X (ROCm 7.2.4)** at 1, 4 and 8 GPUs and
> **NVIDIA H100 80GB (CUDA 13.0)** at 1 GPU. See
> [Platform notes — AMD MI355X](#platform-notes--amd-mi355x-rocm-72) and
> [Platform notes — NVIDIA H100](#platform-notes--nvidia-h100-cuda-130). NVIDIA multi-GPU,
> `--strategy deepspeed`, `--activation_checkpointing` and `--cpu_offload` are untested.
> Lightning moves quickly across 2.x minors — run the smoke command below before committing
> to a long job.

## Overview & when to use

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). The model, loss,
optimizer and LR schedule live in one ~40-line `LightningModule`; sharding (FSDP or
DeepSpeed ZeRO), bf16 mixed precision, distributed samplers, checkpointing and resumption
are all supplied by `lightning.Trainer` flags. The same file runs unmodified on 1 GPU or 8
nodes.

The script loads `dev.env` right after its imports and reads `HF_TOKEN` from the
environment at model-download time.

Files in this folder:
- `train_llm_lightning.py` — the trainer: `LightningModule` + `Trainer`, chat-JSONL data pipeline, optional HF export.
- `requirements_lightning.txt` — deps, with notes on `deepspeed` and `flash-attn`.
- `data/OTel_LLM_sample_10.jsonl` — a 10-row sample dataset the script points at by default.
- `readme_lightning.md` — this file.

## Install

Python 3.12 in its own venv. Install `torch` first, matched to your accelerator, then the
rest.

### NVIDIA (CUDA)

```bash
python3.12 -m venv ~/.venv-lightning && source ~/.venv-lightning/bin/activate
pip install torch==2.11.0
pip install -r requirements_lightning.txt
```

**On CUDA 13 the `torch==2.11.0` pin resolves to a CUDA 13 build on PyPI**, so no
`--index-url` is needed. Install torch first and check it survives the requirements pass:

```bash
python3 -m venv .env_lightning && source .env_lightning/bin/activate
pip install torch==2.11.0 numpy               # CUDA 13 build, straight from PyPI
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
pip install -r requirements_lightning.txt     # torch must be unchanged afterward
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # still 2.11.0 13.0
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

If a sibling install drags the `kernels` package in, pin `kernels>=0.12,<0.13` —
`kernels 0.16.0` breaks all `transformers` imports. This folder's loader reads chat JSONL
directly and does **not** use HF `datasets`, so no `HF_DATASETS_CACHE` redirect is needed.

### AMD (ROCm)

Lightning has no ROCm-specific code path — it runs on whatever accelerator torch exposes,
and ROCm torch builds present the GPU through the standard `torch.cuda` API, so
`accelerator="gpu"` works unchanged. torch 2.11.0 ROCm wheels live on the **rocm7.2** index
(the older `rocm6.4` index stops at torch 2.9.1):

```bash
python3.12 -m venv ~/.venv-lightning && source ~/.venv-lightning/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_lightning.txt
```

On ROCm keep `--attn_implementation sdpa` (the default) and prefer `--strategy fsdp`. Do not
pip-install `flash-attn` on ROCm — the PyPI package is CUDA-only. See
[Platform notes — AMD MI355X](#platform-notes--amd-mi355x-rocm-72) for the run-time tweaks
this path needs.

### Either way

Install `lightning`, **not** `pytorch-lightning` — this script imports the `lightning` root,
and having both in one environment produces confusing import errors.

Only if you plan to pass `--strategy deepspeed`:

```bash
pip install "deepspeed>=0.16.0"
```

That floor is enforced by Lightning itself: with torch >= 2.6, `torch.load` defaults to
`weights_only=True`, and DeepSpeed only handles that from 0.16.0 onward.

## Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_lightning.py` calls `load_dotenv("dev.env")` at import time and passes the token
to `AutoTokenizer` and `AutoModelForCausalLM` — that is all gated checkpoints such as Llama
or Gemma need.

`dev.env` is git-ignored at the repo root. **Never commit a token.**

The multi-GPU commands below refer to two directories by environment variable — set them
to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints, logs and tfevents
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line, the same format as the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The loader reads only the `messages` key from each line, so extra bookkeeping columns
(`unmask`, `flow`, `source_id`, ...) are ignored — your own data may include or omit them
freely. To train on real data, pass `--train_file /path/to/train.jsonl` with the same
`messages` schema.

The whole file is read and tokenized up front with `tokenizer.apply_chat_template`, so
train-time formatting matches what you will send at inference. Behaviours worth knowing:

- **Prompt masking is on by default.** Every non-assistant token gets label `-100`, so
  loss falls on assistant turns only. Pass `--no_mask_prompt` to train on the full
  rendered sequence instead — that is the setting you want for *continued pre-training*,
  where the point is to model the whole text.
- **Over-length rows are dropped, never truncated.** A conversation longer than
  `--max_seq_len` is skipped, and the count is logged. Truncating a chat mid-answer
  teaches the model to stop mid-answer.
- Rows with no supervised tokens (no assistant turn) are also dropped. If the loader
  reports a large drop count, your `--max_seq_len` is too small for the data.

## Run

Smoke test first — one GPU, the shipped sample, one epoch, so failures surface in
minutes:

```bash
python3 train_llm_lightning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --devices 1 --strategy auto \
  --output_dir ./lightning_smoke
```

Then the real 8-GPU run:

```bash
nohup python3 train_llm_lightning.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./lightning_run \
  --export_hf_dir ./lightning_run/final_model \
  --strategy fsdp --devices 8 \
  --precision bf16-mixed \
  --max_seq_len 4096 \
  --batch_size 1 --grad_acc_steps 8 \
  --num_train_epochs 3 --learning_rate 1e-5 \
  --activation_checkpointing \
  > train_llm_lightning.log 2>&1 &

tail -f train_llm_lightning.log
```

`python3 train_llm_lightning.py` is the whole launch command — Lightning spawns and
configures the worker processes itself from `--devices` / `--num_nodes`. Do **not** wrap
it in `torchrun`. (If you are on SLURM, Lightning detects `SLURM_*` env vars and attaches
to the allocation instead of spawning.)

DeepSpeed ZeRO-3 instead of FSDP:

```bash
python3 train_llm_lightning.py ... --strategy deepspeed --zero_stage 3 --cpu_offload
```

**What "working" looks like:** the loader logs how many rows it kept and dropped; Lightning
prints the strategy and precision it selected, then a line listing the FSDP wrap classes
(e.g. `LlamaDecoderLayer`); `nvidia-smi` (or `rocm-smi`) shows roughly even memory across all
GPUs, not one GPU full and the rest idle. If memory is badly uneven, sharding is not engaged
— check that `--strategy fsdp` was accepted and that the wrap-class line appeared.

> **A healthy run looks like a hang.** This trainer logs no per-step loss to stdout and its
> progress bar is disabled (`enable_progress_bar=False`), so the console can sit silent while
> the model loads and trains. Read progress from `<output_dir>/lightning_logs/version_*/`
> with TensorBoard's `EventAccumulator` (tag `train_loss`) instead, and use per-PID VRAM
> occupancy — not utilisation — as the residency check.

On a shared box, set a unique `MASTER_PORT` before launching: Lightning defaults to 29500 and
parallel single-node jobs collide.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL: one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./lightning_run` | Lightning logs + `checkpoints/` (`.ckpt` files) |
| `--export_hf_dir` | `""` | If set, write a plain `from_pretrained`-loadable folder after training |
| `--resume_ckpt` | `""` | Lightning `.ckpt` to resume from (ignored if missing) |
| `--max_seq_len` | `4096` | Rows longer than this are dropped, not truncated |
| `--max_samples` | `None` | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for validation (0 = no val loop) |
| `--mask_prompt` / `--no_mask_prompt` | on | Assistant-only loss (default) vs full-sequence loss |
| `--num_workers` | `4` | Dataloader workers per rank |
| `--batch_size` | `1` | Per-device micro-batch size |
| `--grad_acc_steps` | `8` | Trainer `accumulate_grad_batches`; global batch = product x world size |
| `--num_train_epochs` | `3` | Max epochs |
| `--learning_rate` | `1e-5` | Peak LR (full fine-tune ~1e-5..2e-5); cosine schedule with warmup |
| `--weight_decay` | `0.0` | AdamW weight decay |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--grad_clip` | `1.0` | Global grad-norm clip; 0 disables |
| `--devices` | `-1` | GPUs per node; -1 = all visible |
| `--num_nodes` | `1` | Node count |
| `--precision` | `bf16-mixed` | `bf16-mixed`, `bf16-true`, `16-mixed`, `32-true` |
| `--log_every_n_steps` | `10` | Logging cadence in optimizer steps |
| `--seed` | `42` | Random seed |
| `--attn_implementation` | `sdpa` | `sdpa` (default), `flash_attention_2` (CUDA only), `eager` |
| `--strategy` | `fsdp` | `fsdp`, `deepspeed`, `ddp`, `ddp_find_unused_parameters_true`, or `auto` |
| `--sharding_strategy` | `FULL_SHARD` | FSDP: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` |
| `--state_dict_type` | `full` | FSDP: `full` consolidates on rank 0, `sharded` writes one file per rank |
| `--zero_stage` | `3` | DeepSpeed only: 1, 2 or 3 |
| `--cpu_offload` | off | Offload optimizer state (and ZeRO-3 params) to CPU |
| `--activation_checkpointing` | off | Recompute decoder-layer activations (FSDP path); big memory win |
| `--no_checkpoint` | off | Disable Lightning checkpoint writing entirely — use for smoke runs (a `.ckpt` of an 8B model with AdamW state is ~100 GB, written every epoch) |

## Output

```
lightning_run/
  lightning_logs/version_0/          # TensorBoard event files, hparams.yaml
  checkpoints/
    epoch=..-step=...ckpt            # Lightning checkpoints (weights + optimizer + schedule)
    last.ckpt
  final_model/                       # only if --export_hf_dir was passed
    config.json, model-*.safetensors, tokenizer files
```

A `.ckpt` is a *Lightning* checkpoint, not a Hugging Face model: it holds optimizer and
scheduler state so training can resume exactly, and `from_pretrained` cannot read it.
Resume with `--resume_ckpt ./lightning_run/checkpoints/last.ckpt`.

For serving, use the `--export_hf_dir` folder:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./lightning_run/final_model")
tok = AutoTokenizer.from_pretrained("./lightning_run/final_model")
```

Watch `train_loss` in TensorBoard (`tensorboard --logdir lightning_run/lightning_logs`),
and `val_loss` too if you passed `--eval_samples`.

## Quirks & troubleshooting

These are hardware-independent — they apply on ROCm and CUDA alike.

1. **FSDP refuses `gradient_clip_val`.** Under `--strategy fsdp`, Lightning 2.6.5 raises
   `MisconfigurationException: gradient_clip_algorithm='norm' is currently not supported for
   FSDPPrecision` during setup. Pass **`--grad_clip 0`** for every FSDP run — the gap is
   world-size independent and occurs at 1, 4 and 8 devices alike. If you need clipping under
   FSDP, override `configure_gradient_clipping` and call FSDP's own `clip_grad_norm_`.
   `--strategy auto`/`ddp` clip fine at the default `1.0`.
2. **Plain `--strategy ddp` fails on multimodal checkpoints** with
   `RuntimeError: It looks like your LightningModule has parameters that were not used in
   producing the loss returned by training_step.` A model such as `google/gemma-4-E4B-it` has
   vision/audio towers that receive no gradient from text-only chat data, so DDP's reducer
   never sees those buckets. Use `--strategy ddp_find_unused_parameters_true`. FSDP is
   unaffected — it wraps those towers and tolerates them being unused. Text-only checkpoints
   (Llama, Qwen, LFM2) never hit this.
3. **`flash_attention_2` may not import.** The prebuilt PyPI `flash-attn` wheel
   (2.8.3.post1) installs but fails to import against the CUDA 13 torch build:
   `ImportError: .../flash_attn_2_cuda...so: undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`
   — a c10 ABI mismatch. Build from source (`pip install flash-attn --no-build-isolation`) or
   stay on the `sdpa` default, which needs no extra package. On ROCm, `flash-attn` from PyPI
   is CUDA-only and must not be installed at all.
4. **Checkpoints are large — disable them for smoke runs.** Lightning writes a `.ckpt` every
   epoch, and `ModelCheckpoint` here uses `save_last=True` *plus* `save_top_k=1`, i.e. **two**
   files. For an 8B model with AdamW state that is ~100 GB per file, ~200 GB per run. Pass
   `--no_checkpoint`, or point `--output_dir` at scratch.
5. **transformers 5.x chat-template drift (already fixed in the script).**
   `apply_chat_template(tokenize=True, ...)` returns a `BatchEncoding` by default in
   transformers 5.x; the loader passes `return_dict=False` so it gets a token list back.
   Without that, `len()` measures the encoding's *keys*, the masking arithmetic collapses and
   every row is silently dropped. If you port this loader elsewhere, carry the flag with it.

## Hardware support

| Platform | Status |
|---|---|
| NVIDIA (CUDA 12/13) | works; `flash_attention_2` optional, `sdpa` is the default |
| AMD Instinct (ROCm 7.2) | works through torch's `torch.cuda` API; no `flash-attn` |

Hardware support is torch-level — Lightning has no vendor-specific code path.

## Platform notes — AMD MI355X (ROCm 7.2)

**This path works on AMD Instinct MI355X (gfx950), ROCm 7.2.4, at 1, 4 and 8 devices.** The
script has no ROCm-specific code and `requirements_lightning.txt` is unchanged at every world
size.

```bash
cd training/llm/lightning
python3 -m venv .env_lightning && source .env_lightning/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_lightning.txt
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

**Untested here:** `--activation_checkpointing`, `--export_hf_dir` under FSDP,
`--cpu_offload` and `--strategy deepspeed`.

**Device detection needs nothing special.** The ROCm torch build exposes the MI355X through
`torch.cuda`, so `accelerator="gpu"` and Lightning's "CUDA device" log lines work unchanged;
`distributed_backend=nccl` in the log is the ROCm build's RCCL. Keep
`--attn_implementation sdpa` (the default) and do not pip-install `flash-attn` — the PyPI
package is CUDA-only.

Launch — the whole command, no `torchrun`; Lightning spawns the ranks itself:

```bash
cd training/llm/lightning
source .env_lightning/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# HF_HOME / OUTPUT_DIR: see the "Set these to suit your machine" block above
export MASTER_PORT=29670          # keep off the default 29500 on a shared box
python -c "import torch; assert torch.cuda.device_count()==8"   # must read 8 BEFORE Trainer is built

python3 train_llm_lightning.py \
  --model_name google/gemma-4-E4B-it \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy fsdp --sharding_strategy FULL_SHARD \
  --devices 8 --num_nodes 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --num_train_epochs 10 --log_every_n_steps 1 \
  --grad_clip 0 --no_checkpoint \
  --output_dir $OUTPUT_DIR/lightning/gpu8/fsdp8
```

Export `HIP_VISIBLE_DEVICES` **and** `CUDA_VISIBLE_DEVICES` explicitly and assert the device
count before `Trainer` is constructed — if `torch.cuda.device_count()` reads low, Lightning
silently trains on fewer GPUs.

**Expected output** — every rank registered and the FSDP wrap policy resolved on each:

```
FSDP will wrap: ['Gemma4AudioLayer', 'Gemma4TextDecoderLayer', 'Gemma4VisionEncoderLayer', 'Gemma4VisionPatchEmbedder']
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8   ... GLOBAL_RANK: 7, MEMBER: 8/8
distributed_backend=nccl
All distributed processes registered. Starting with 8 processes
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
│ 0 │ model │ Gemma4ForConditionalGeneration │  992 M │      # per-rank, i.e. sharded
`Trainer.fit` stopped: `max_epochs=10` reached.
```

#### How to tell sharding is actually engaged

- The per-rank parameter count in Lightning's model summary is the full count **divided by
  the world size** — the most reliable check.
- No `FSDP is switching to use NO_SHARD ... since the world size is 1` line (that appears at
  `--devices 1`, where the wrapper is a no-op).
- Per-GPU VRAM is even across cards. One card holding the full model while the rest sit near
  zero means sharding is not engaged.

#### Consolidated checkpoints and DDP

The consolidated `--state_dict_type full` checkpoint works above world size 1: rank 0's
all-gather produces a `.ckpt` holding the whole *unsharded* model, not per-rank slices.
Checkpoints of a large model are big, so pass `--no_checkpoint` on smoke runs.

**DDP** on a multimodal checkpoint needs the find-unused-parameters variant:

```bash
python3 train_llm_lightning.py ... --devices 8 \
  --strategy ddp_find_unused_parameters_true --grad_clip 1.0 --no_checkpoint
```

Note the asymmetry: DDP keeps `--grad_clip 1.0` while FSDP requires `--grad_clip 0` — see
[Quirks & troubleshooting](#quirks--troubleshooting).

## Platform notes — NVIDIA H100 (CUDA 13.0)

**This path works on a single NVIDIA H100 80GB HBM3** (CUDA 13.0) with no code change, end to
end including `--export_hf_dir`. Use the CUDA-13 install in [Install](#nvidia-cuda)
(`torch==2.11.0`).

The one deviation from the MI355X recipe: the prebuilt `flash-attn` wheel does not import
against this torch, so stay on the `sdpa` default — see
[Quirks & troubleshooting](#quirks--troubleshooting) item 3.

### Smoke command (single GPU)

Plain `CUDA_VISIBLE_DEVICES` is the whole GPU-pinning story on NVIDIA — no
`HIP_VISIBLE_DEVICES`, no `RAY_EXPERIMENTAL_NOSET_*`. If the Hub is unreachable, point
`HF_HOME` at a local model cache and run offline against a pre-cached model that has a chat
template (e.g. `LiquidAI/LFM2.5-350M`):

```bash
source .env_lightning/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # HF_HOME points at the model cache (set above)
export MASTER_PORT=29646            # unique port; Lightning defaults to 29500 and collides

CUDA_VISIBLE_DEVICES=6 python3 train_llm_lightning.py \
  --model_name LiquidAI/LFM2.5-350M \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy auto --devices 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 3 --max_seq_len 2048 \
  --num_train_epochs 12 --log_every_n_steps 1 \
  --learning_rate 5e-5 --warmup_ratio 0.1 \
  --export_hf_dir ./lightning_smoke/final_model \
  --output_dir ./lightning_smoke
```

`--grad_clip` is left at its default 1.0 and works here — the FSDP `gradient_clip_val` gap
does **not** apply under `--strategy auto`/single-device.

### Expected output

```
INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
Using bfloat16 Automatic Mixed Precision (AMP)
You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. ...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [6]
│ 0 │ model │ Lfm2ForCausalLM │  354 M │
`Trainer.fit` stopped: `max_epochs=12` reached.
INFO - __main__ - Exported Hugging Face weights to .../lightning_smoke/final_model
INFO - __main__ - Training complete.
```

`train_loss` is read from the run's tfevents (`EventAccumulator`, tag `train_loss` — this
trainer logs loss to TensorBoard, **not** stdout).

Lightning writes `checkpoints/epoch=..-step=...ckpt` and `last.ckpt` (weights + AdamW state +
schedule), and `--export_hf_dir` produces a plain `from_pretrained`-loadable folder
(`model.safetensors`, `config.json`, `chat_template.jinja`, tokenizer files) with its chat
template intact. Point `--output_dir` at scratch and pass `--no_checkpoint` for throwaway
smoke runs.

### Multi-GPU on NVIDIA

Nothing NVIDIA-specific is needed:
`--strategy fsdp --sharding_strategy FULL_SHARD --devices N` with **`--grad_clip 0`**, a
distinct `MASTER_PORT`, and `--no_checkpoint` for smoke runs. Lightning spawns the ranks
itself — do **not** wrap in `torchrun`. For a text-only checkpoint like LFM2/Qwen/Llama, plain
`--strategy ddp` also works — `ddp_find_unused_parameters_true` is only needed for multimodal
towers.

## Notes

- **FSDP wraps at the decoder-block level, resolved from `_no_split_modules`.** If an exotic
  architecture does not declare it, the script fails loudly rather than silently
  mis-sharding.
- **`bf16-mixed` vs `bf16-true`.** `bf16-mixed` keeps fp32 master weights and autocasts the
  forward pass — the safe default for full fine-tuning. `bf16-true` puts the optimizer state
  in bf16 too: less memory, but small-magnitude updates can be lost to rounding.
- **Do not add a `DistributedSampler` yourself.** Lightning's `use_distributed_sampler`
  defaults to `True` and replaces the sampler on each rank; adding one is the classic
  double-sampler bug.
- **Export is a collective**, not a rank-0 operation: `export_hf` runs
  `lightning_module_state_dict()` on *every* rank, so guarding it behind
  `if trainer.is_global_zero` would hang. Export requires `--state_dict_type full`.
- **DeepSpeed export is a separate step.** ZeRO checkpoints are sharded; run the
  `zero_to_fp32.py` script DeepSpeed writes into the checkpoint folder to consolidate, then
  load the result and `save_pretrained`.
- **Checkpoint retention.** With no val split there is no monitored metric, so
  `ModelCheckpoint` keeps only the most recent checkpoint; with `--eval_samples` set it keeps
  the top 2 by `val_loss` plus `last.ckpt`.
- **OOM playbook.** In order: turn on `--activation_checkpointing`; drop `--batch_size` to 1
  and raise `--grad_acc_steps` to hold the global batch; lower `--max_seq_len`; add
  `--cpu_offload`; then switch to `--strategy deepspeed --zero_stage 3 --cpu_offload`.
