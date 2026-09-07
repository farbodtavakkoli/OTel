# `training/llm/peft` — chat SFT with standalone Hugging Face PEFT

> **Tested topology: single AMD Instinct MI355X (ROCm 7.2) — see the platform notes
> below.** Originally written against the upstream PEFT and
> Transformers docs for the pinned versions below, targeting a single node with 8x H100 80GB.
> Unlike the sharded trainers here it is
> genuinely useful on **one** GPU: QLoRA on a single card is the intended small-scale
> path. The multi-GPU route is plain DDP (one full model replica per GPU), not sharding.

## Overview & when to use

Chat-model SFT with Hugging Face [PEFT](https://github.com/huggingface/peft) used
directly — no Unsloth wrapper, no DeepSpeed. It covers **LoRA**, **QLoRA** (bitsandbytes
4-bit nf4 base weights), and the LoRA variants **DoRA** (`--use_dora`) and **rsLoRA**
(`--use_rslora`), all built from a plain `LoraConfig` + `get_peft_model` and trained by
the stock HF `Trainer`. Pick this folder over `../unsloth/` when you want the
vanilla upstream code path (any model Transformers supports, no patched kernels, adapters
that are exactly what PEFT writes), and over `../deepspeed/` or `../fsdp/`
when the model fits on one GPU — quantized 4-bit weights cannot be sharded by ZeRO-3 or
FSDP2 anyway.

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

### AMD (ROCm)

torch 2.11.0 ROCm wheels live on the **rocm7.2** index (the older `rocm6.4` index stops
at torch 2.9.1 — verified against download.pytorch.org):

```bash
python3.12 -m venv ~/.venv-peft && source ~/.venv-peft/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_peft.txt
```

`bitsandbytes` — the QLoRA dependency — **officially supports AMD ROCm**: upstream ships
ROCm wheels on PyPI (built for ROCm 6.4.4 through 7.14.0), so the same
`pip install bitsandbytes` works. The honest caveat: ROCm support landed much later than
CUDA and is less battle-tested in the wild, and the prebuilt wheels target specific GPU
architectures (CDNA: gfx90a/gfx942/gfx950 — MI200/MI300/MI350 class — plus current RDNA).
If your card is not in that list, build from source with `-DCOMPUTE_BACKEND=hip`, or drop
`--load_in_4bit` and train plain bf16 LoRA — everything else in this folder works
without bitsandbytes.

### Verify

```bash
python -c "import torch, peft, transformers, bitsandbytes; \
print('peft', peft.__version__, '| transformers', transformers.__version__, '| bnb', bitsandbytes.__version__)"
```

Notes:
- Everything here comes from PyPI. Nothing requires a git/source install.
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
the repo root. Never hardcode a token in the source and never commit one — if a token was
ever committed, rotate it on the Hub immediately.

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The sample rows also carry extra bookkeeping columns (`unmask`, `flow`, `source_id`,
`source_repo`, `source_spec_id`, `source_version`; `source_spec_id`/`source_version` are
null in most rows). The loader reads only the `messages` key from each line, so extra
columns are ignored — your own data may include or omit them freely.

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

Verified end to end on one MI355X (gfx950, 288 GB HBM) of an 8-GPU node — ROCm 7.2.4,
Ubuntu, Python 3.12.3. **This path works with one change.** The change is a
transformers-v5 portability fix (already applied to `train_llm_peft.py`, nothing
AMD-specific): `apply_chat_template(tokenize=True)` returns a `BatchEncoding` in
transformers 5.x instead of a flat id list, which made the prompt-masking span diff
silently mark every row unsupervised (`No usable rows ... unsupervised: 10`). The fix is
`return_dict=False` on the three `apply_chat_template` calls in `build_example`.

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
# bf16 LoRA — PASS (10 s train time)
python train_llm_peft.py --model_name google/gemma-4-E4B-it \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir ./lora_smoke

# QLoRA (4-bit nf4 + paged_adamw_8bit) — PASS on gfx950 (18 s train time)
python train_llm_peft.py --model_name google/gemma-4-E4B-it \
  --load_in_4bit --optim paged_adamw_8bit --gradient_checkpointing \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir ./qlora_smoke

# merge — PASS (13 s with --device_map auto on the MI355X)
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

A 5-epoch (45-step) LoRA run converges loss 8.73 → 1.18 with finite grad norms
throughout; `rocm-smi` mid-run shows the GPU active with ~17 GB VRAM allocated.

Status on gfx950, tested individually:

| Path | Status | Observed |
|---|---|---|
| bf16 LoRA (sdpa) | **works** | 9 steps, loss 9.64 → 5.93 |
| QLoRA (`--load_in_4bit`, nf4, double-quant, `paged_adamw_8bit`) | **works** | 9 steps, finite loss; bnb 0.50.0 PyPI wheel loads `libbitsandbytes_rocm72.so` |
| DoRA (`--use_dora`) | **works** | 3-step epoch, loss 9.79 → 5.97 |
| `merge_adapter.py` | **works** | 4-shard bf16 safetensors written |

Quirks and notes from the run:

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

The same recipe scales from 1 GPU to all 8 on the same node (torch 2.11.0+rocm7.2,
transformers 5.5.0, peft 0.20.0, accelerate 1.14.0), same base model
`google/gemma-4-E4B-it`. **Status per variant:**

| Variant at world size 8 | Status | Observed |
|---|---|---|
| bf16 LoRA (sdpa), DDP | **works with one change** | 36 steps, exit code 0, loss 65.88 → 3.544, 23.28 samples/s |
| QLoRA (`--load_in_4bit`, nf4, `paged_adamw_8bit`), DDP | **works with one change, with a caveat** | 36 steps, exit code 0, loss 66.99 → 4.039; but GPU 0 carries 76 % VRAM vs 17-19 % on ranks 1-7 |
| DoRA (`--use_dora`), DDP | **works with one change** | 36 steps, exit code 0, loss 67.29 → 6.396, 15.58 samples/s |

**The one change: `--ddp_find_unused_parameters`.** Out of the box every variant dies
in the first backward pass on all 8 ranks:

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting
a new one. This error indicates that your module has parameters that were not used in
producing loss.
```

Root cause, and it is not AMD-specific: `gemma-4-E4B-it` is multimodal (its `config.json`
has `vision_config` and `audio_config`), and `--lora_target_modules all-linear` adapts
every linear layer in the whole model. Counting the adapters on a meta-device
instantiation: **379 in the language model, 114 in the vision tower, 135 in the audio
tower.** A text-only batch never runs the two towers, so those 249 adapters produce no
gradient, and DDP's `ddp_find_unused_parameters=False` (previously hard-coded in
`train_llm_peft.py`) treats that as a bucket-reduction error. One GPU has no DDP wrapper,
so the 1-GPU runs above never saw it. `train_llm_peft.py` now exposes
`--ddp_find_unused_parameters` (default off, so the single-GPU path and its DDP
performance are unchanged); pass it whenever `all-linear` meets a multimodal base. The
alternative fix is to pass an explicit text-only `--lora_target_modules` list.

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

**Parallelism and batch geometry.** Plain PyTorch DDP via `torchrun` — no sharding; the
full 8.05 B base is replicated on each of the 8 GPUs and only the 52,318,208 adapter
params (0.65 %) are all-reduced. World size 8, per-device batch 2, `grad_acc 1` →
**global batch 16**, 576 rows → 36 optimizer steps/epoch. `accelerate launch
--num_processes=8` is equivalent; `torchrun` keeps the launcher explicit.

**Dataset change, stated plainly.** The shipped `data/OTel_LLM_sample_10.jsonl` has 10
rows (9 usable), which at world size 8 gives ~1 step/epoch — not enough to prove
anything. The 8-GPU runs use that same file **replicated 64x** (640 rows, 576 usable
after the one over-length row is dropped) written **outside the repo**, e.g. at
`$OUTPUT_DIR/train_llm_peft/gpu8/data/otel_sample_x64.jsonl`. This is a
**pipeline proof, not a learning result** — the loss curve is memorization of 9 repeated
conversations.

**Expected output** (bf16 LoRA, 8 GPUs):

```
ASSERT device_count=8 torch=2.11.0+rocm7.2 / GPU0: AMD Instinct MI355X
trainable params: 52,318,208 || all params: 8,048,474,656 || trainable%: 0.6500
{'loss': '65.88', 'grad_norm': '62.87', 'learning_rate': '0.0001983', 'epoch': '0.1389'}
{'loss': '3.544', 'grad_norm': '17.5',  'learning_rate': '1.703e-06', 'epoch': '0.9722'}
{'train_runtime': '24.74', 'train_samples_per_second': '23.28', 'train_steps_per_second': '1.455'}
```

The run should finish with exit code 0.
Grad norms are finite and non-zero at every log point, so the adapter really is training.
The first logged value is the mean of steps 1-5: `warmup_ratio 0.03` of 36 steps is one
step, so the LR is already 1.98e-4 at global batch 16 and the loss spikes before falling
monotonically. The 1-GPU baseline at global batch 2 starts at 8.18 instead — that
difference is the batch/LR schedule, not a distributed-training bug.

Sample `rocm-smi` **in-band** (from inside the run script, while training) to confirm all
8 GPUs are busy and that the PIDs holding VRAM are this job's own ranks (cross-check
against `pgrep -f train_llm_peft.py` in the same sample; `pt_elastic` is the torchrun
launcher). A healthy sample:

```
Device  ... PwrCap   VRAM%  GPU%          KFD process information:
0       ... 1400.0W  28%    98%           NAME       GPU(s)  VRAM USED
1       ... 1400.0W  30%    99%           python3    1       ~86-93 GB   (8 ranks,
2       ... 1400.0W  28%    99%           ...                             one per GPU)
3       ... 1400.0W  28%    98%           pt_elastic 0       0
4       ... 1400.0W  27%    99%
5-7     ... 1400.0W  28-30% 99%
```

**Throughput vs the 1-GPU run.** Measured back to back in the same lock hold, same
per-device geometry (the 1-GPU baseline runs 36 steps over a 1/8 slice of the same data):

| | 1 GPU | 8 GPUs (DDP) |
|---|---|---|
| optimizer steps | 36 | 36 |
| `train_samples_per_second` | 3.523 | **23.28** (6.61x, 83 % scaling efficiency) |
| `train_steps_per_second` | 1.762 | 1.455 (−17 %: the per-step cost of the adapter all-reduce) |
| per-GPU VRAM (`rocm-smi` VRAM%) | 27 % (~78 GB of 288 GB) | 28-30 % (~81-86 GB) |

**Per-GPU VRAM delta is essentially zero** for LoRA — DDP replicates, so each rank pays
the same memory as the single-GPU run plus a small gradient/communication bucket. The
scaling win is throughput, not capacity: this folder still cannot train a base that does
not fit on one card (use `../fsdp` or `../deepspeed` for that).

**QLoRA under DDP — works, but memory is not symmetric.** 4-bit training at world size 8
completes cleanly (exit code 0, 171.1 s, 3.366 samples/s, loss 66.99 → 4.039), so
bitsandbytes 0.50.0 on ROCm survives DDP. But the in-band sampler shows a real
asymmetry: **GPU 0 at 76 % VRAM (~219 GB) while ranks 1-7 sit at 17-19 % (~52 GB)**, and
`rocm-smi --showpids` reports `GPU(s) = 2` for every non-zero rank (rank 0 reports 1) —
i.e. each of the seven other ranks also allocates on GPU 0. Quantized base weights are
replicated per rank, never sharded, and on top of that the 4-bit path puts a second
allocation on device 0 from every rank. **This was isolated, not guessed:** a control run
of the identical job with `--optim adamw_torch` instead of `paged_adamw_8bit` shows the
same skew (GPU 0 75 %, ranks 1-7 17-19 %, `GPU(s) = 2` on every non-zero rank), so the
paged optimizer is *not* the cause — bitsandbytes' 4-bit path is. The plain bf16 LoRA run
shows `GPU(s) = 1` for all eight ranks and a flat 28-30 % across the node, so it is
specific to `--load_in_4bit`. On a 288 GB MI355X there is headroom; on a smaller card
this is the failure mode that OOMs rank 0 first while the other seven look idle. Watch
GPU 0 specifically when scaling QLoRA.

That control run also produced a **performance finding worth acting on**: under DDP,
`--optim paged_adamw_8bit` costs 4.5x throughput. Identical 36-step QLoRA job, only the
optimizer changed:

```
paged_adamw_8bit : {'train_runtime': '171.1',  'train_samples_per_second': '3.366'}
adamw_torch      : {'train_runtime': '37.68',  'train_samples_per_second': '15.29'}
```

The paged optimizer's unified-memory paging is far more expensive with 8 ranks on one
node than the memory it saves is worth here, since the 52 M adapter params make optimizer
state negligible either way. **Use `paged_adamw_8bit` at 1 GPU when memory is tight; drop
it to `adamw_torch` for multi-GPU QLoRA.** With that swap QLoRA at 8 GPUs is
only ~1.5x slower than bf16 LoRA (15.29 vs 23.28 samples/s), the rest being
`--gradient_checkpointing` plus per-step dequantization — a cost also present at 1 GPU,
not a DDP effect.

Everything else that differs from the 1-GPU run: nothing. Same install, same
`requirements_peft.txt` (no new package or pin is needed), same `sdpa` attention, no
`flash-attn`, no NCCL/RCCL tuning env vars, no `accelerate config` file. Checkpointing
stays at the default `save_strategy="epoch"`; the script has no
`load_best_model_at_end`, so only one adapter checkpoint plus `final_adapter/` is written
per run (~780 MB with optimizer state) — delete them after a smoke run.

## Platform notes — NVIDIA H100 80GB (CUDA 13.0)

Verified single-GPU on one **NVIDIA H100 80GB HBM3** (Hopper cc 9.0) of an 8-GPU node —
driver **580.173.02**, **CUDA 13.0**, Ubuntu, Python 3.12.3. Pinned to one free card with
`CUDA_VISIBLE_DEVICES=5` (no `HIP_VISIBLE_DEVICES` — that is ROCm-only). **This path works
(with the same transformers-v5 fix already in the tree), QLoRA included.** The
`return_dict=False` masking fix is hardware-neutral and is already applied; nothing else
in the code changes for CUDA. Only **single-GPU** was exercised; multi-GPU (DDP) is not
covered here (see below).

### Install (CUDA 13) + torch-clobber note

The `requirements_peft.txt` pin `torch==2.11.0` has **no cu130 wheel**, so on a CUDA-13
host install torch **unpinned** first, which resolves
`torch 2.13.0+cu130` (native CUDA 13), then the rest of the file:

```bash
cd training/llm/peft
python3 -m venv .env_peft && source .env_peft/bin/activate
pip install torch numpy                 # -> torch 2.13.0+cu130 (2.11.0 has no cu130 wheel)
pip install -r requirements_peft.txt     # skip the torch line; everything else installs as pinned
# ALWAYS re-verify torch afterward — bnb/peft installs can silently downgrade it:
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
```

In the validated run the framework install did **not** clobber torch (still `2.13.0+cu130`).
`bitsandbytes` 0.50.0 from plain PyPI ships a `libbitsandbytes_cuda130.so` and auto-selects
it — **no ROCm `.so` is loaded on CUDA**, and no build step is needed (QLoRA is genuinely
easier here than on ROCm):

```
bnb loaded lib: .../bitsandbytes/libbitsandbytes_cuda130.so
4bit nf4 round-trip mean abs err: 0.0723 | finite: True
```

Key versions: `torch 2.13.0+cu130` (cuda 13.0), `transformers 5.5.0`, `peft 0.20.0`,
`accelerate 1.14.0`, `bitsandbytes 0.50.0`, `datasets 4.3.0`, `tokenizers 0.22.2`, driver
`580.173.02`. tf32: torch 2.13 leaves the matmul tf32 flag off by default on Hopper
(`torch.backends.cuda.matmul.allow_tf32 == False`); training uses bf16 (`bf16=True`)
regardless, so this does not affect the runs.

### Model deviation (offline cache), stated plainly

The MI355X runs used `google/gemma-4-E4B-it`. On the H100 node that repo is present in
the shared HF cache (`$HF_HOME`) with **weights + config only —
its tokenizer files are not cached**, and the node's outbound Hub access is
proxy-blocked (`httpx.ProxyError: 403 Forbidden`), so the tokenizer cannot be fetched and
the trainer (which requires a chat template) cannot start on it offline. Rather than
mutate the shared cache, the H100 smoke uses a **fully-cached small instruct model,
`LiquidAI/LFM2.5-350M`** (360M params, complete tokenizer + chat template), against the
**same shipped `data/OTel_LLM_sample_10.jsonl`**. This is a pipeline/kernel proof on the
same code path, not a learning result. To reproduce on gemma-4-E4B-it, first populate its
tokenizer into the cache (or run on a host with Hub access). The runs set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to force cache-only loads.

### Smoke commands (GPU 5, offline, shipped 10-row sample → 9 usable)

The script has **no `--max_steps`** — step count is `rows / (batch × grad_acc)` per epoch.
9 usable rows at `batch 1 × grad_acc 1` = 9 steps/epoch; raise the epoch count to get a
non-trivial, clearly-decreasing curve.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # HF_HOME as exported above

# bf16 LoRA — PASS (sdpa; 4 epochs = 36 steps)
CUDA_VISIBLE_DEVICES=5 python train_llm_peft.py --model_name LiquidAI/LFM2.5-350M \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --attn_implementation sdpa \
  --output_dir /dev/shm/h100/out/peft/lora_smoke

# QLoRA (4-bit nf4 + paged_adamw_8bit + gradient checkpointing) — PASS
CUDA_VISIBLE_DEVICES=5 python train_llm_peft.py --model_name LiquidAI/LFM2.5-350M \
  --load_in_4bit --optim paged_adamw_8bit --gradient_checkpointing \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --logging_steps 1 \
  --lora_r 16 --lora_alpha 32 --max_seq_len 2048 --output_dir /dev/shm/h100/out/peft/qlora_smoke

# merge — PASS
CUDA_VISIBLE_DEVICES=5 python merge_adapter.py \
  --adapter /dev/shm/h100/out/peft/lora_smoke/final_adapter \
  --output_dir /dev/shm/h100/out/peft/merged --device_map cpu
```

**Expected output** (bf16 LoRA, LFM2.5-350M = 360.5M params, 1.66% trainable, 36
steps, loss noisy per single-row step but trending to ~0):

```
Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
trainable params: 5,996,544 || all params: 360,480,512 || trainable%: 1.6635
{'loss': '2.211', 'grad_norm': '9.691', 'learning_rate': '0.0002', 'epoch': '0.3333'}
{'loss': '0.517', 'grad_norm': '3.114', 'learning_rate': '1.703e-06', 'epoch': '3.889'}
{'train_runtime': '6.483', 'train_samples_per_second': '5.553', 'train_steps_per_second': '5.553', 'train_loss': '0.7063', 'epoch': '4'}
Saved LoRA adapter to .../lora_smoke/final_adapter (base model: LiquidAI/LFM2.5-350M)
```

A longer 60-epoch (540-step) bf16 LoRA run drives loss `0.877 → 1.2e-4` with finite,
shrinking grad norms throughout (train_loss 0.058); QLoRA over the same 540 steps gives
`0.972 → 1.2e-4` (train_loss 0.063) — bitsandbytes 4-bit on CUDA trains cleanly.

**GPU residency check, sampled in-band** (from the driving script while training, filtered
to the pinned GPU index with `nvidia-smi -i <n>` and matched to this job's own worker PID):

```
# QLoRA worker on GPU 5:
GPU5 [idx,mem,util]: 5, 887 MiB, 0 %
GPU5 compute-apps [pid,mem]: <pid>, 880 MiB     # <- the training PID holds VRAM on card 5
# bf16 LoRA worker on GPU 5:
GPU5 mem=761MiB util=0% | compute-app(pid,mem)=<pid>, 772 MiB
```

VRAM on the pinned GPU climbs from ~0 to ~0.7–0.9 GB and the compute-apps table names this
job's own python PID; on a shared node, leave the other cards alone.
(`util%` reads 0 in a 0.2 s snapshot because each optimizer step on a 1-row batch of a 360M
model is sub-millisecond of GPU time — the VRAM-by-PID residency is the signal, not util.)

Status on H100 (single GPU), tested individually:

| Path | Status | Observed |
|---|---|---|
| bf16 LoRA (sdpa) | **works** | 36-step run loss 2.21 → 0.5; 540-step run 0.877 → 1.2e-4; GPU5 PID holds ~0.76 GB |
| QLoRA (`--load_in_4bit`, nf4, double-quant, `paged_adamw_8bit`, grad-ckpt) | **works** | 36 steps finite loss; bnb 0.50.0 loads `libbitsandbytes_cuda130.so`; GPU5 PID holds ~0.88 GB |
| `merge_adapter.py` | **works** | 1-shard bf16 safetensors (681 MB standalone model) written, exit code 0 |

### Quirks / deviations from the MI355X recipe

- **torch pin does not exist for CUDA 13.** `torch==2.11.0` has no cu130 wheel; install
  torch unpinned → `2.13.0+cu130`. This is the only requirements deviation; every other
  pin installed unchanged. `numpy` came out at 2.5.2 (pin says 2.5.1) via the unpinned
  torch pull — harmless.
- **bitsandbytes on CUDA needs no thought.** The plain PyPI wheel auto-selects the
  `cuda130` backend (contrast the MI355X note where it picks `rocm72`). QLoRA "just
  works".
- **VRAM: 80 GB here vs 288 GB on MI355X.** Not a factor for a 360M smoke (peaks < 1 GB).
  It *would* matter at the documented QLoRA scale (Llama-3.1-8B) and especially at
  multi-GPU — see below. No OOM occurred; no batch/seq reduction was needed for the smoke.
- **flash-attn:** the script exposes `--attn_implementation flash_attention_2`, but
  `pip install flash-attn` is still a from-source nvcc build on CUDA 13, so the runs use
  the default **`sdpa`** (which needs nothing) and pass. Pre-build the FA2 wheel offline
  if you want it; do not block on it.
- **Offline model substitution** (tokenizer-not-cached) is documented above — the *only*
  reason gemma-4-E4B-it was not the smoke target.
- Checkpoints (adapter-only here, 40–74 MB since the base is 360M; `save_strategy="epoch"`
  still writes a fit-end checkpoint even conceptually with `"no"`) can be deleted after a
  smoke run.

### Multi-GPU on H100 — not covered here

Only single-GPU was exercised. A multi-GPU DDP pass would
mirror the MI355X 8-GPU section and, on 80 GB cards, needs care the 288 GB MI355X hid:

- The **one required flag stands**: `--ddp_find_unused_parameters` whenever
  `--lora_target_modules all-linear` meets a multimodal base (gemma-4's vision/audio
  towers get no gradient in a text-only batch). Use a distinct `--master_port` (e.g.
  29642), plain `CUDA_VISIBLE_DEVICES=0,1,...` (no `HIP_VISIBLE_DEVICES`).
- **QLoRA rank/VRAM skew is the H100 risk.** The MI355X run shows QLoRA under DDP puts a
  second allocation on rank 0 (GPU 0 ~76% VRAM vs 17–19% on ranks 1–7) — on a 288 GB card
  that is headroom; **on an 80 GB H100 that is exactly the pattern that OOMs rank 0 first
  while the other GPUs look idle.** Watch GPU 0 when scaling QLoRA; the MI355X finding to
  drop `paged_adamw_8bit` → `adamw_torch` for multi-GPU QLoRA (4.5× faster there) should be
  re-checked on CUDA. bf16 LoRA under DDP replicates and is symmetric, so it is the safe
  first multi-GPU step. This trainer never shards — a base that does not fit on one 80 GB
  card needs `../fsdp` or `../deepspeed`, and neither shards 4-bit weights.

## Hardware support & evidence

Claims above were checked against upstream sources:

- **bitsandbytes on AMD ROCm is official.** The upstream installation guide
  ([bitsandbytes-foundation/bitsandbytes `docs/source/installation.mdx`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx))
  states official support for NVIDIA GPUs, AMD GPUs, Intel XPUs, Apple Silicon and Intel
  Gaudi, with all features supported on RDNA and CDNA AMD hardware. PyPI wheels are built
  for ROCm 6.4.4 / 7.0.2 / 7.1.1 / 7.2.4 / 7.14.0, targeting CDNA gfx90a/gfx942/gfx950
  and current RDNA parts; source builds cover ROCm 6.3+.
- **QLoRA's AMD caveat, honestly:** support is real and official, but newer and less
  field-proven than CUDA, and limited to the wheel's compiled GPU targets — check your
  `gfx` arch against the list, and fall back to bf16 LoRA (drop `--load_in_4bit`) if the
  bitsandbytes import or kernels fail on your card.
- **ROCm torch wheels.** `torch==2.11.0+rocm7.2` wheels exist on
  [download.pytorch.org/whl/rocm7.2](https://download.pytorch.org/whl/rocm7.2/torch/);
  the `rocm6.4` index stops at torch 2.9.1.
- **Other hardware (upstream claims — not verified here):** PEFT itself is pure-PyTorch
  and device-agnostic — upstream publishes no hardware matrix and its README quickstart
  uses `torch.accelerator`, so bf16 LoRA runs wherever torch/transformers run (CPU,
  Apple MPS, Intel XPU/Gaudi, Ascend NPU torch builds). The QLoRA path is bounded by
  bitsandbytes, which officially claims NVIDIA CUDA, AMD ROCm, Intel XPU, Intel Gaudi,
  and Apple Silicon backends.

## Notes

- **Plain upstream PEFT.** `LoraConfig(...)` → `get_peft_model(model, config)` →
  `Trainer`. No custom kernels, no patched model classes, no import-order requirements —
  which is the trade against `../unsloth/`: slower and more memory-hungry, but it
  works with any model Transformers loads and the artifacts are exactly what upstream
  PEFT produces.
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
  LoRA-handled direction. It usually beats plain LoRA at low rank, supports linear
  layers, and adds real runtime overhead — merge the weights before serving. **rsLoRA**
  (`--use_rslora`) only changes the scaling factor to `lora_alpha/sqrt(r)`, which stops
  the effective learning rate from collapsing as `r` grows — the flag to reach for when
  raising rank stops helping. The two are independent and can be combined.
- **Chat-template masking.** Each row is tokenized once with `apply_chat_template`; each
  assistant span is located by re-rendering the conversation prefix with
  `add_generation_prompt=True` and diffing lengths. Labels outside those spans are
  `-100`. This assumes prefix-stable chat templates (true for mainstream instruct
  models).
- **Merging.** `merge_adapter.py` reloads the base in full precision (never 4-bit —
  bitsandbytes layers cannot absorb a LoRA delta; quantize the merged model afterwards if
  you need a quantized artifact), attaches the adapter with `PeftModel.from_pretrained`,
  and calls `merge_and_unload()`, which returns a new model with the adapter folded into
  the base weights and the PEFT modules removed. The return value is reassigned because
  the call is not in place. Merging removes all adapter inference overhead; the cost is
  that you no longer have a swappable adapter, so keep the original `final_adapter/`
  directory. It prefers the tokenizer saved next to the adapter (it may carry added
  tokens) and falls back to the base model's. The one exception upstream calls out:
  aLoRA-style adapters cannot be merged at all.
- **Single-GPU first, DDP second.** There is no sharding here. `accelerate launch
  --num_processes 8` gives DDP, which replicates the whole model per GPU — fine for LoRA
  on a model that fits, useless for a model that does not. (Measured on 8x MI355X: 6.61x
  throughput at world size 8, unchanged per-GPU VRAM, and one required flag — see the
  8-GPU section above.) For sharded training use
  `../fsdp` (FSDP2) or `../deepspeed` (ZeRO), and note that neither
  can shard 4-bit weights, so QLoRA belongs here.
