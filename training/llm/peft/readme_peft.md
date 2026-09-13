# `training/llm/peft` — chat SFT with standalone Hugging Face PEFT

> **Hardware coverage:** **AMD Instinct MI355X (ROCm 7.2.4)** at 1 GPU and at 8
> GPUs under DDP, and **NVIDIA H100 80GB (CUDA 13.0)** at 1 GPU — bf16 LoRA, QLoRA
> (`--load_in_4bit`), DoRA and `merge_adapter.py` all work on both vendors. See
> [Platform notes — AMD MI355X](#platform-notes--amd-instinct-mi355x-rocm-72) and
> [Platform notes — NVIDIA H100](#platform-notes--nvidia-h100-80gb-cuda-130). Multi-GPU on
> NVIDIA is not covered.
>
> Unlike the sharded trainers here this folder is genuinely useful on **one** GPU: QLoRA on a
> single card is the intended small-scale path. The multi-GPU route is plain DDP (one full
> model replica per GPU), not sharding.

## Overview & when to use

Chat-model SFT with Hugging Face [PEFT](https://github.com/huggingface/peft) used
directly — no Unsloth wrapper, no DeepSpeed. It covers **LoRA**, **QLoRA** (bitsandbytes
4-bit nf4 base weights), and the LoRA variants **DoRA** (`--use_dora`) and **rsLoRA**
(`--use_rslora`), all built from a plain `LoraConfig` + `get_peft_model` and trained by
the stock HF `Trainer`. It needs the model to fit on one GPU — quantized 4-bit weights
cannot be sharded by ZeRO-3 or FSDP2.

Both scripts load `dev.env` right after their imports and read `HF_TOKEN` from the
environment at model-download time — no token is ever hardcoded. The saved artifact is an
ordinary PEFT adapter directory that any PEFT-aware runtime can load; fold it into base
weights afterwards with `merge_adapter.py`.

Files in this folder:
- `train_llm_peft.py` — the training entrypoint (LoRA / QLoRA / DoRA / rsLoRA).
- `merge_adapter.py` — merges a trained adapter into base weights with `merge_and_unload()` and saves an HF-ready model.
- `requirements_peft.txt` — dependencies, with per-accelerator install notes.
- `data/OTel_LLM_sample_10.jsonl` — a 10-row sample dataset the script points at by default.

## Install

Python 3.12 in its own venv. Install `torch` first, matched to your accelerator, then the
rest.

### NVIDIA (CUDA)

```bash
python3.12 -m venv ~/.venv-peft && source ~/.venv-peft/bin/activate
pip install torch==2.11.0
pip install -r requirements_peft.txt
```

**On CUDA 13 the `torch==2.11.0` pin resolves to a CUDA 13 build on PyPI**, so no
`--index-url` is needed. Install torch first, then the rest of the file, and re-verify,
because a `bitsandbytes`/`peft` install can silently downgrade torch:

```bash
python3 -m venv .env_peft && source .env_peft/bin/activate
pip install torch==2.11.0 numpy          # CUDA 13 build, straight from PyPI
pip install -r requirements_peft.txt     # every pin installs unchanged
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

`bitsandbytes` 0.50.0 from plain PyPI ships `libbitsandbytes_cuda130.so` and auto-selects it,
so QLoRA needs no build step on CUDA.

### AMD (ROCm)

torch 2.11.0 ROCm wheels live on the **rocm7.2** index (the older `rocm6.4` index stops
at torch 2.9.1):

```bash
python3.12 -m venv ~/.venv-peft && source ~/.venv-peft/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_peft.txt
```

`bitsandbytes` — the QLoRA dependency — ships ROCm wheels on PyPI, so the same
`pip install bitsandbytes` works. Those wheels target specific GPU architectures (CDNA
gfx90a/gfx942/gfx950 plus current RDNA). If your card is not in that list, build from source
with `-DCOMPUTE_BACKEND=hip`, or drop `--load_in_4bit` and train plain bf16 LoRA —
everything else in this folder works without bitsandbytes.

### Verify

```bash
python -c "import torch, peft, transformers, bitsandbytes; \
print('peft', peft.__version__, '| transformers', transformers.__version__, '| bnb', bitsandbytes.__version__)"
```

Notes:
- `flash-attn` is **optional and not in the requirements file**: it is a CUDA-only source
  build (`pip install flash-attn --no-build-isolation`, with `CUDA_HOME` exported). Only
  then can you pass `--attn_implementation flash_attention_2`; the default `sdpa` needs
  nothing.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Both scripts load it with `load_dotenv("dev.env")` and read `HF_TOKEN` from the
environment, so run them from inside `training/llm/peft/`. `dev.env` is **git-ignored** at
the repo root. Never hardcode a token in the source and never commit one.

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The loader reads only the `messages` key from each line, so extra bookkeeping columns
(`unmask`, `flow`, `source_id`, ...) are ignored — your own data may include or omit them
freely.

To train on real data, pass `--train_file /path/to/train.jsonl` with the same
`messages` schema. A `system` turn and extra user/assistant turns are fine. Each row is
rendered with the tokenizer's own `apply_chat_template`, so training matches inference
formatting exactly; a model whose tokenizer has no chat template is rejected up front.

**Loss masking:** completion-only by default (`--mask_prompt`, on). Every token outside
an assistant turn is set to `-100`, so loss falls on assistant turns only — in a
multi-turn row, *all* assistant turns are supervised. Pass `--no_mask_prompt` to train on
the full rendered sequence. Rows longer than `--max_seq_len` are **dropped, never
truncated**, and rows with no supervised token are dropped too; both counts are logged.

## Run

Smoke test first — one GPU, the shipped sample, one epoch:

```bash
python train_llm_peft.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --logging_steps 1 \
  --output_dir ./peft_smoke
```

**QLoRA on a single GPU** (4-bit nf4 base + bf16 adapter), from inside `training/llm/peft/`:

```bash
nohup python train_llm_peft.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./qlora_run \
  --load_in_4bit --optim paged_adamw_8bit \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear \
  --batch_size 2 --grad_acc_steps 8 --num_train_epochs 3 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_peft.log 2>&1 &

tail -f train_llm_peft.log
```

**bf16 LoRA across 8 GPUs** (DDP — one replica per GPU, so the model must fit on one card):

```bash
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  train_llm_peft.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./lora_run \
  --lora_r 32 --lora_alpha 64 --use_rslora \
  --batch_size 2 --grad_acc_steps 4 --num_train_epochs 3 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_peft.log 2>&1 &
```

**Merge the adapter into base weights** when training finishes:

```bash
python merge_adapter.py \
  --adapter ./qlora_run/final_adapter \
  --output_dir ./qlora_run/merged_model
```

**What "working" looks like:** the row-count line (`Loaded N rows ... dropped M over K
tokens`), an adapter line naming r/alpha/targets, then `trainable params: ... || all
params: ... || trainable%: ...` showing well under 1% trainable. Training then logs
`{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}` every `--logging_steps` with a
finite `grad_norm` and a loss that moves. The run ends with
`Saved LoRA adapter to .../final_adapter` and `Training complete.`; the adapter dir is a
few hundred MB at most, not model-sized.

## Arguments

`train_llm_peft.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL: one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./peft_run` | Checkpoints + `final_adapter/` land here |
| `--resume_from_checkpoint` | `""` | Checkpoint dir to resume from (used only if it exists) |
| `--max_seq_len` | `4096` | Token cap per row; longer rows are dropped, never truncated |
| `--max_samples` | `None` | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for a per-epoch eval (0 = off) |
| `--mask_prompt` / `--no_mask_prompt` | on | Completion-only loss (default) vs full-sequence loss |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha (scaling); 2x rank is a good default |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` (QLoRA-style) or a comma-separated module list |
| `--modules_to_save` | `None` | Extra modules trained in full and saved with the adapter (e.g. `embed_tokens,lm_head`) |
| `--use_dora` | off | DoRA: decompose the update into magnitude + direction (better at low rank, slower) |
| `--use_rslora` | off | rsLoRA: scale by `alpha/sqrt(r)` instead of `alpha/r` (stabler at high rank) |
| `--load_in_4bit` | off | QLoRA: load the frozen base in 4-bit nf4 via bitsandbytes |
| `--bnb_4bit_quant_type` | `nf4` | 4-bit data type (`nf4` or `fp4`); nf4 is the QLoRA default |
| `--no_double_quant` | (double quant on) | Disable nested (double) quantization of the quantization constants |
| `--batch_size` | `2` | Per-device train/eval batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps |
| `--num_train_epochs` | `3.0` | Number of training epochs |
| `--learning_rate` | `2e-4` | Peak LR (LoRA/QLoRA typically 1e-4..2e-4, far higher than full FT) |
| `--lr_scheduler_type` | `cosine` | LR scheduler type |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--weight_decay` | `0.0` | Weight decay |
| `--optim` | `adamw_torch` | HF optimizer id; use `paged_adamw_8bit` with QLoRA to cut optimizer memory |
| `--logging_steps` | `10` | Log training metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints to keep |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Recompute activations; needed for long sequences on one GPU |
| `--ddp_find_unused_parameters` | off | Multi-GPU only: tolerate adapter params that get no gradient in a step. Required when `all-linear` adapts submodules a text-only batch never runs (a multimodal base's vision/audio towers) — see the 8-GPU section |
| `--attn_implementation` | `sdpa` | `sdpa` (default), `flash_attention_2` (CUDA only), `eager` |

`merge_adapter.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--adapter` | (required) | Trained adapter dir (e.g. `<output_dir>/final_adapter`) |
| `--base_model` | `None` | Base model id/path; defaults to `base_model_name_or_path` from `adapter_config.json` |
| `--output_dir` | (required) | Where the merged, HF-ready model is written |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` — the precision the merge happens in |
| `--device_map` | `cpu` | `cpu` (safest) or `auto` if the full model fits in HBM |
| `--max_shard_size` | `5GB` | Shard size for the saved safetensors files |

## Output

Under `--output_dir`:

- `checkpoint-<step>/` — per-epoch Trainer checkpoints (adapter + optimizer + scheduler
  state), capped by `--save_total_limit`; pass one back via `--resume_from_checkpoint`.
- `final_adapter/` — the trained adapter: `adapter_model.safetensors`,
  `adapter_config.json` (which records the base model), plus the tokenizer files. Load it
  with `PeftModel.from_pretrained(base_model, "<path>/final_adapter")`.
- `merged_model/` — only if you run `merge_adapter.py`: a standalone directory loadable
  with `AutoModelForCausalLM.from_pretrained(...)`, ready to serve or push to the Hub.
- `runs/` — TensorBoard event files; view with `tensorboard --logdir <output_dir>/runs`.
- `train_llm_peft.log` — the redirected stdout/stderr of the run, in this folder.

## Platform notes — AMD Instinct MI355X (ROCm 7.2)

**This path works on MI355X (gfx950), ROCm 7.2.4, with one change** — a transformers-v5
portability fix already applied to `train_llm_peft.py` and not AMD-specific:
`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers 5.x instead of
a flat id list, which made the prompt-masking span diff mark every row unsupervised
(`No usable rows ... unsupervised: 10`). The fix is `return_dict=False` on the three
`apply_chat_template` calls in `build_example`. If you port this loader elsewhere, carry the
flag with it.

Install:

```bash
cd training/llm/peft
python3 -m venv .env_peft && source .env_peft/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_peft.txt   # bitsandbytes 0.50.0 from plain PyPI — no special index needed
```

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts / adapter output
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Smoke commands (shipped 10-row sample; note the script has **no
`--max_steps`** — control step count via batch/accumulation/epochs; here
`batch 1 x grad_acc 1 x 1 epoch` = 9 steps, one row exceeds `--max_seq_len 2048` and is
dropped):

```bash
# bf16 LoRA
python train_llm_peft.py --model_name google/gemma-4-E4B-it \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir ./lora_smoke

# QLoRA (4-bit nf4 + paged_adamw_8bit)
python train_llm_peft.py --model_name google/gemma-4-E4B-it \
  --load_in_4bit --optim paged_adamw_8bit --gradient_checkpointing \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir ./qlora_smoke

# merge
python merge_adapter.py --adapter ./lora_smoke/final_adapter \
  --output_dir ./lora_smoke/merged_model --device_map auto
```

**Expected output** (LoRA run, gemma-4-E4B-it = 8.05 B params, 0.65 % trainable):

```
trainable params: 52,318,208 || all params: 8,048,474,656 || trainable%: 0.6500
{'loss': '9.64', 'grad_norm': '38.5', 'learning_rate': '0', 'epoch': '0.1111'}
{'loss': '5.931', 'grad_norm': '7.601', 'learning_rate': '7.612e-06', 'epoch': '1'}
Saved LoRA adapter to .../lora_smoke/final_adapter (base model: google/gemma-4-E4B-it)
```

Quirks and notes:

- **bitsandbytes on ROCm is confirmed on gfx950**: the plain `pip install
  bitsandbytes` (0.50.0) wheel bundles ROCm binaries alongside the CUDA ones and picks
  `libbitsandbytes_rocm72.so` for ROCm 7.2.4/gfx950 automatically. `quantize_4bit`/
  `dequantize_4bit` round-trip and full QLoRA training both work. No multi-backend
  index or source build is needed.
- Attention stays on the default `sdpa`; do **not** try `flash_attention_2` here —
  `pip install flash-attn` is a CUDA source build and fails on ROCm.
- On a multi-GPU node, pin one GPU with `export HIP_VISIBLE_DEVICES=<n>
  CUDA_VISIBLE_DEVICES=<n>` before running; torch then sees it as device 0.
- A benign transformers warning about PAD/BOS/EOS realignment for gemma-4-E4B-it
  appears at load; it is harmless.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

The same recipe scales from 1 GPU to all 8 on one node with **one required flag** and no new
package, pin or code edit. bf16 LoRA, QLoRA and DoRA all complete at world size 8. QLoRA
carries one caveat about rank-0 memory (see below).

**The one change: `--ddp_find_unused_parameters`.** Out of the box every variant dies
in the first backward pass on all 8 ranks:

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting
a new one. This error indicates that your module has parameters that were not used in
producing loss.
```

Cause, and it is not AMD-specific: `gemma-4-E4B-it` is multimodal, and
`--lora_target_modules all-linear` also adapts its vision and audio towers. A text-only batch
never runs those towers, so their adapters produce no gradient and DDP treats that as a
bucket-reduction error (a single GPU has no DDP wrapper, so the 1-GPU runs never see it).
Pass `--ddp_find_unused_parameters` whenever `all-linear` meets a multimodal base, or give an
explicit text-only `--lora_target_modules` list.

Commands (venv activated first; **note** if `.env_peft/bin/activate` has a
stale `export HIP_VISIBLE_DEVICES=3` / `CUDA_VISIBLE_DEVICES=3` appended by an earlier 1-GPU
session, override it *after* sourcing or you will silently train on one GPU):

```bash
source .env_peft/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"

# bf16 LoRA, world size 8
torchrun --nproc_per_node=8 --master_port 29690 train_llm_peft.py \
  --model_name google/gemma-4-E4B-it --train_file <replicated>.jsonl \
  --max_seq_len 2048 --lora_r 16 --lora_alpha 32 \
  --batch_size 2 --grad_acc_steps 1 --num_train_epochs 1 --logging_steps 5 \
  --ddp_find_unused_parameters --output_dir <out>/lora8

# QLoRA, world size 8   (add: --load_in_4bit --optim paged_adamw_8bit --gradient_checkpointing)
# DoRA,  world size 8   (add: --use_dora)
```

**Parallelism.** Plain PyTorch DDP via `torchrun` — no sharding; the full base is replicated
on each GPU and only the adapter params are all-reduced. `accelerate launch
--num_processes=8` is equivalent.

**Dataset.** The shipped `data/OTel_LLM_sample_10.jsonl` has 10 rows (9 usable), which at
world size 8 gives ~1 step/epoch. Replicate it (e.g. ×64 → 640 rows) into a directory
**outside the repo** and point `--train_file` there.

**Expected output** (bf16 LoRA, 8 GPUs, exit code 0):

```
trainable params: 52,318,208 || all params: 8,048,474,656 || trainable%: 0.6500
{'loss': '65.88', 'grad_norm': '62.87', 'learning_rate': '0.0001983', 'epoch': '0.1389'}
{'loss': '3.544', 'grad_norm': '17.5',  'learning_rate': '1.703e-06', 'epoch': '0.9722'}
{'train_runtime': '24.74', 'train_samples_per_second': '23.28', 'train_steps_per_second': '1.455'}
```

Grad norms are finite and non-zero at every log point, so the adapter really is training. The
high first value is expected: `warmup_ratio 0.03` of 36 steps is one step, so the LR is
already near peak and the loss spikes before falling.

To confirm all 8 GPUs are busy, sample `rocm-smi` **in-band** (from inside the run script,
while training) and cross-check `rocm-smi --showpids` against `pgrep -f train_llm_peft.py` in
the same sample: 8 `python3` processes, `GPU(s) = 1` each, even VRAM across cards
(`pt_elastic` is the torchrun launcher and holds 0).

**QLoRA under DDP — works, but memory is not symmetric.** With `--load_in_4bit`, rank 0
carries a much larger allocation than the other ranks: every rank also allocates on GPU 0
(`rocm-smi --showpids` reports `GPU(s) = 2` for the non-zero ranks). It is bitsandbytes'
4-bit path, not the optimizer — `--optim adamw_torch` shows the same skew, and plain bf16
LoRA is flat across all ranks. **Watch GPU 0 when scaling QLoRA**: this is the pattern that
OOMs rank 0 first while the other GPUs look idle.

**Under DDP, prefer `--optim adamw_torch` over `paged_adamw_8bit`.** The paged optimizer's
unified-memory paging costs far more with 8 ranks on one node than the memory it saves is
worth for a small adapter. Use `paged_adamw_8bit` at 1 GPU when memory is tight.

Everything else that differs from the 1-GPU run: nothing. Same install, same
`requirements_peft.txt` (no new package or pin is needed), same `sdpa` attention, no
`flash-attn`, no NCCL/RCCL tuning env vars, no `accelerate config` file. Checkpointing
stays at the default `save_strategy="epoch"`; the script has no
`load_best_model_at_end`, so only one adapter checkpoint plus `final_adapter/` is written
per run (~780 MB with optimizer state) — delete them after a smoke run.

## Platform notes — NVIDIA H100 80GB (CUDA 13.0)

**This path works on H100, QLoRA included** (CUDA 13.0), single-GPU. Pin the card with
`CUDA_VISIBLE_DEVICES` — no `HIP_VISIBLE_DEVICES`, that is ROCm-only. No code change is
needed beyond the hardware-neutral `return_dict=False` masking fix already in the tree.
Multi-GPU DDP is not covered (see below).

Use the CUDA-13 install: `torch==2.11.0` (see [Install](#nvidia-cuda)).
`bitsandbytes` auto-selects the CUDA backend, so QLoRA needs no build step:

```
bnb loaded lib: .../bitsandbytes/libbitsandbytes_cuda130.so
```

The smoke below uses the small `LiquidAI/LFM2.5-350M` rather than the MI355X section's gated
`google/gemma-4-E4B-it`. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` force cache-only loads.

### Smoke commands (one pinned GPU, offline, shipped 10-row sample → 9 usable)

The script has **no `--max_steps`** — step count is `rows / (batch × grad_acc)` per epoch.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # HF_HOME as exported above

# bf16 LoRA — PASS (sdpa; 4 epochs = 36 steps)
CUDA_VISIBLE_DEVICES=5 python train_llm_peft.py --model_name LiquidAI/LFM2.5-350M \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --attn_implementation sdpa \
  --output_dir /dev/shm/peft/lora_smoke

# QLoRA (4-bit nf4 + paged_adamw_8bit + gradient checkpointing) — PASS
CUDA_VISIBLE_DEVICES=5 python train_llm_peft.py --model_name LiquidAI/LFM2.5-350M \
  --load_in_4bit --optim paged_adamw_8bit --gradient_checkpointing \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir /dev/shm/peft/qlora_smoke

# merge — PASS
CUDA_VISIBLE_DEVICES=5 python merge_adapter.py \
  --adapter /dev/shm/peft/lora_smoke/final_adapter \
  --output_dir /dev/shm/peft/merged --device_map cpu
```

**Expected output** (bf16 LoRA, LFM2.5-350M, 1.66% trainable; loss is noisy per single-row
step but trends down):

```
Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
trainable params: 5,996,544 || all params: 360,480,512 || trainable%: 1.6635
{'loss': '2.211', 'grad_norm': '9.691', 'learning_rate': '0.0002', 'epoch': '0.3333'}
{'loss': '0.517', 'grad_norm': '3.114', 'learning_rate': '1.703e-06', 'epoch': '3.889'}
{'train_runtime': '6.483', 'train_samples_per_second': '5.553', 'train_steps_per_second': '5.553', 'train_loss': '0.7063', 'epoch': '4'}
Saved LoRA adapter to .../lora_smoke/final_adapter (base model: LiquidAI/LFM2.5-350M)
```

To confirm residency on a shared node, sample `nvidia-smi -i <n>` in-band from the driving
script and match the compute-apps table against this job's own worker PID — the VRAM held by
your PID is the signal, not `util%`.

### Quirks / deviations from the MI355X recipe

- **The torch pin needs no deviation on CUDA 13.** `torch==2.11.0` from PyPI is a CUDA 13
  build, so `requirements_peft.txt` installs unchanged.
- **bitsandbytes on CUDA needs no extra setup.** The plain PyPI wheel auto-selects the
  `cuda130` backend (on MI355X it picks `rocm72`), so QLoRA works as shipped.
- **flash-attn:** the script exposes `--attn_implementation flash_attention_2`, but
  `pip install flash-attn` is still a from-source nvcc build on CUDA 13, so the default
  **`sdpa`** (which needs nothing) is what these runs use. Pre-build the FA2 wheel offline if
  you want it; do not block on it.
- **Checkpoints** are adapter-only here (tens of MB for a sub-1B base). `save_strategy="epoch"`
  still writes a fit-end checkpoint; delete them after a smoke run.

### Multi-GPU on NVIDIA — not covered here

A multi-GPU DDP pass mirrors the MI355X 8-GPU section and, on 80 GB cards, needs care the
288 GB MI355X hid:

- The **one required flag stands**: `--ddp_find_unused_parameters` whenever
  `--lora_target_modules all-linear` meets a multimodal base (gemma-4's vision/audio
  towers get no gradient in a text-only batch). Use a distinct `--master_port` (e.g.
  29642), plain `CUDA_VISIBLE_DEVICES=0,1,...` (no `HIP_VISIBLE_DEVICES`).
- **QLoRA rank-0 memory skew is the H100 risk.** QLoRA under DDP puts an extra allocation on
  rank 0 (see the MI355X section) — **on an 80 GB card that is the pattern that OOMs rank 0
  first while the other GPUs look idle.** bf16 LoRA under DDP is symmetric, so it is the safe
  first multi-GPU step. This trainer never shards — a base that does not fit on one card needs
  `../fsdp` or `../deepspeed`, and neither shards 4-bit weights.

## Hardware support

| Platform | LoRA / DoRA / rsLoRA | QLoRA (`--load_in_4bit`) |
|---|---|---|
| NVIDIA CUDA | works | works — bitsandbytes picks the `cuda*` backend |
| AMD ROCm 7.2 (CDNA gfx90a/gfx942/gfx950, current RDNA) | works | works — bitsandbytes picks `rocm72` |

PEFT itself is pure PyTorch, so bf16 LoRA runs wherever torch/transformers run; the QLoRA path
is bounded by bitsandbytes' compiled GPU targets. If the bitsandbytes import or kernels fail on
your card, drop `--load_in_4bit` and train bf16 LoRA.

## Notes

- **Plain upstream PEFT.** `LoraConfig(...)` → `get_peft_model(model, config)` →
  `Trainer`. No custom kernels, no patched model classes, no import-order requirements.
- **`target_modules="all-linear"`** is the default because that is the QLoRA recipe:
  adapt every linear layer rather than only `q_proj`/`v_proj`. It is also model-agnostic,
  so no per-architecture module lists to maintain.
- **QLoRA path.** `--load_in_4bit` builds a `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=bfloat16)` and the model then goes through
  `prepare_model_for_kbit_training`, which upcasts norms and makes gradients flow from
  the frozen 4-bit base into the bf16 adapters. Pair it with `--optim paged_adamw_8bit`
  for the full memory win. The base stays frozen and quantized; only the adapter trains.
- **DoRA** (`--use_dora`) decomposes each update into a learned magnitude and a
  LoRA-handled direction; it adds runtime overhead, so merge the weights before serving.
  **rsLoRA** (`--use_rslora`) changes the scaling factor to `lora_alpha/sqrt(r)` — reach for
  it when raising rank stops helping. The two are independent and can be combined.
- **Chat-template masking.** Each row is tokenized once with `apply_chat_template`; each
  assistant span is located by re-rendering the conversation prefix with
  `add_generation_prompt=True` and diffing lengths. Labels outside those spans are
  `-100`. This assumes prefix-stable chat templates (true for mainstream instruct
  models).
- **Merging.** `merge_adapter.py` reloads the base in full precision (never 4-bit —
  bitsandbytes layers cannot absorb a LoRA delta; quantize the merged model afterwards if
  you need a quantized artifact), attaches the adapter with `PeftModel.from_pretrained`,
  and calls `merge_and_unload()`. Keep the original `final_adapter/` directory — after
  merging you no longer have a swappable adapter. It prefers the tokenizer saved next to the
  adapter (it may carry added tokens) and falls back to the base model's. aLoRA-style
  adapters cannot be merged at all.
- **Single-GPU first, DDP second.** There is no sharding here. `accelerate launch
  --num_processes 8` gives DDP, which replicates the whole model per GPU — fine for LoRA
  on a model that fits, useless for a model that does not. For sharded training use
  `../fsdp` (FSDP2) or `../deepspeed` (ZeRO); neither can shard 4-bit weights, so QLoRA
  belongs here.
