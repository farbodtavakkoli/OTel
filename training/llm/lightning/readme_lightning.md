# `training/llm/lightning` — chat fine-tuning with PyTorch Lightning

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). The model, loss,
optimizer and LR schedule live in one small `LightningModule`; sharding (FSDP or DeepSpeed
ZeRO), bf16 mixed precision, distributed samplers, checkpointing and resumption all come
from `lightning.Trainer` flags. The same file runs unmodified on 1 GPU or 8.

Pick this recipe when you want Lightning's Trainer ergonomics over a hand-rolled loop.
Lightning moves quickly across 2.x minors - run the smoke command before committing to a
long job.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 1/4/8 GPUs · NVIDIA H100 80GB (CUDA 13.0),
1 GPU.

## Files

| File | Purpose |
|---|---|
| `train_llm_lightning.py` | `LightningModule` + `Trainer`, chat-JSONL data pipeline, optional HF export |
| `requirements_lightning.txt` | Dependencies, with `deepspeed` / `flash-attn` notes |
| `data/OTel_LLM_sample_10.jsonl` | 10-row chat sample; the default `--train_file` |

## Setup

Python 3.12, one venv per recipe folder. Install `torch` first, matched to your
accelerator, then the rest, and confirm torch survived the second install.

Install `lightning`, not `pytorch-lightning` - this script imports the `lightning` root,
and having both in one environment produces confusing import errors.

### NVIDIA (CUDA 13)

```bash
python3.12 -m venv .env_lightning && source .env_lightning/bin/activate
pip install torch==2.11.0 numpy               # the PyPI wheel is the CUDA 13 build
pip install -r requirements_lightning.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

If a sibling install drags the `kernels` package in, pin `kernels>=0.12,<0.13` -
`kernels 0.16.0` breaks all `transformers` imports.

### AMD (ROCm 7.2)

Lightning has no ROCm-specific code path: ROCm torch exposes the GPU through the standard
`torch.cuda` API, so `accelerator="gpu"` and the `distributed_backend=nccl` log line (RCCL
underneath) work unchanged. torch 2.11.0 ROCm wheels live on the `rocm7.2` index (the older
`rocm6.4` index stops at torch 2.9.1).

```bash
python3.12 -m venv .env_lightning && source .env_lightning/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_lightning.txt
```

Keep `--attn_implementation sdpa` (the default) and prefer `--strategy fsdp`. Do not
install `flash-attn` on ROCm.

### Optional: DeepSpeed strategy

Only if you plan to pass `--strategy deepspeed`:

```bash
pip install "deepspeed>=0.16.0"
```

Lightning enforces that floor: with torch >= 2.6 `torch.load` defaults to
`weights_only=True`, which DeepSpeed only handles from 0.16.0 onward.

### Secrets

`HF_TOKEN` for gated checkpoints comes from `dev.env` in this folder, loaded by
`load_dotenv("dev.env")` at import time and passed to `AutoTokenizer` /
`AutoModelForCausalLM`:

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

## Data

`--train_file` takes a chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Only the `messages` key is read, so the shipped sample's extra columns (`unmask`, `flow`,
`source_id`, ...) are ignored and your own data may include or omit them. The whole file is
read and tokenized up front with `tokenizer.apply_chat_template`, so train-time formatting
matches inference.

Prompt masking is on by default: every non-assistant token gets label `-100`. Pass
`--no_mask_prompt` to train on the full rendered sequence - that is the setting for
continued pre-training, where the point is to model the whole text. Rows longer than
`--max_seq_len` and rows with no assistant turn are dropped, never truncated; both counts
are logged. A large drop count means `--max_seq_len` is too small for your data.

## Run

`python3 train_llm_lightning.py` is the whole launch command - Lightning spawns and
configures the worker processes itself from `--devices` / `--num_nodes`. Do not wrap it in
`torchrun`. (On SLURM, Lightning detects `SLURM_*` and attaches to the allocation instead
of spawning.)

Smoke test - one GPU, the shipped sample, one epoch:

```bash
python3 train_llm_lightning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --devices 1 --strategy auto \
  --output_dir ./lightning_smoke
```

Full run, 8 GPUs, FSDP:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # NVIDIA, or the subset you own
export MASTER_PORT=29670                        # Lightning defaults to 29500 and collides
python -c "import torch; assert torch.cuda.device_count()==8"   # assert BEFORE Trainer is built

nohup python3 train_llm_lightning.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./lightning_run \
  --export_hf_dir ./lightning_run/final_model \
  --strategy fsdp --sharding_strategy FULL_SHARD --devices 8 \
  --precision bf16-mixed \
  --max_seq_len 4096 \
  --batch_size 1 --grad_acc_steps 8 \
  --num_train_epochs 3 --learning_rate 1e-5 \
  --grad_clip 0 \
  --activation_checkpointing \
  > train_llm_lightning.log 2>&1 &

tail -f train_llm_lightning.log
```

**Pass `--grad_clip 0` for every FSDP run.** Under `--strategy fsdp`, Lightning 2.6.5
raises `MisconfigurationException: gradient_clip_algorithm='norm' is currently not
supported for FSDPPrecision` at setup, at every world size. `--strategy auto`/`ddp` clip
fine at the default `1.0`.

DeepSpeed ZeRO-3 instead of FSDP:

```bash
python3 train_llm_lightning.py ... --strategy deepspeed --zero_stage 3 --cpu_offload
```

Plain `--strategy ddp` fails on a multimodal checkpoint (e.g. `google/gemma-4-E4B-it`) with
`RuntimeError: It looks like your LightningModule has parameters that were not used in
producing the loss...`: vision/audio towers get no gradient from text-only chat data. Use
`--strategy ddp_find_unused_parameters_true` (keeping `--grad_clip 1.0`). FSDP is
unaffected; text-only checkpoints never hit this.

**A healthy run looks like a hang.** This trainer logs no per-step loss to stdout and its
progress bar is disabled, so the console sits silent while the model loads and trains. Read
progress from `<output_dir>/lightning_logs/version_*/` in TensorBoard (tag `train_loss`),
and use per-PID VRAM occupancy, not utilisation, as the residency check.

To confirm sharding is actually engaged, check that the per-rank parameter count in
Lightning's model summary is the full count divided by the world size, and that per-GPU
VRAM is even across cards. One full card while the rest sit near zero means it is not.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL, one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./lightning_run` | Lightning logs + `checkpoints/` (`.ckpt` files) |
| `--export_hf_dir` | `""` | If set, write a `from_pretrained`-loadable folder after training |
| `--resume_ckpt` | `""` | Lightning `.ckpt` to resume from (ignored if missing) |
| `--max_seq_len` | `4096` | Rows longer than this are dropped, not truncated |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--eval_samples` | `0` | Rows held out for validation (0 = no val loop) |
| `--mask_prompt` / `--no_mask_prompt` | on | Assistant-only loss vs full-sequence loss |
| `--num_workers` | `4` | Dataloader workers per rank |
| `--batch_size` | `1` | Per-device micro-batch size |
| `--grad_acc_steps` | `8` | `accumulate_grad_batches`; global batch = product x world size |
| `--num_train_epochs` | `3` | Max epochs |
| `--learning_rate` | `1e-5` | Peak LR (full FT ~1e-5..2e-5); cosine schedule with warmup |
| `--weight_decay` | `0.0` | AdamW weight decay |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--grad_clip` | `1.0` | Global grad-norm clip; 0 disables - required under FSDP |
| `--devices` | `-1` | GPUs per node; -1 = all visible |
| `--num_nodes` | `1` | Node count |
| `--precision` | `bf16-mixed` | `bf16-mixed`, `bf16-true`, `16-mixed`, `32-true` |
| `--log_every_n_steps` | `10` | Logging cadence in optimizer steps |
| `--seed` | `42` | Random seed |
| `--attn_implementation` | `sdpa` | `sdpa`, `flash_attention_2` (CUDA only), `eager` |
| `--strategy` | `fsdp` | `fsdp`, `deepspeed`, `ddp`, `ddp_find_unused_parameters_true`, `auto` |
| `--sharding_strategy` | `FULL_SHARD` | FSDP: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` |
| `--state_dict_type` | `full` | FSDP: `full` consolidates on rank 0, `sharded` writes one file per rank |
| `--zero_stage` | `3` | DeepSpeed only: 1, 2 or 3 |
| `--cpu_offload` | off | Offload optimizer state (and ZeRO-3 params) to CPU |
| `--activation_checkpointing` | off | Recompute decoder-layer activations (FSDP path) |
| `--no_checkpoint` | off | Disable Lightning checkpoint writing entirely |

## Output

```
lightning_run/
  lightning_logs/version_0/          # TensorBoard event files, hparams.yaml
  checkpoints/
    epoch=..-step=...ckpt            # Lightning checkpoints (weights + optimizer + schedule)
    last.ckpt
  final_model/                       # only if --export_hf_dir was passed
    config.json, model-*.safetensors, chat_template.jinja, tokenizer files
```

A `.ckpt` is a Lightning checkpoint, not a Hugging Face model: `from_pretrained` cannot
read it. Resume with `--resume_ckpt ./lightning_run/checkpoints/last.ckpt`. For serving,
use the `--export_hf_dir` folder:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./lightning_run/final_model")
tok = AutoTokenizer.from_pretrained("./lightning_run/final_model")
```

Watch `train_loss` with `tensorboard --logdir lightning_run/lightning_logs`, plus
`val_loss` if you passed `--eval_samples`.

## Notes

- **Checkpoints are large - disable them for smoke runs.** `ModelCheckpoint` uses
  `save_last=True` plus `save_top_k=1`, i.e. two files per run; for an 8B model with AdamW
  state that is ~100 GB each. Pass `--no_checkpoint`, or point `--output_dir` at scratch.
- With no val split there is no monitored metric, so only the most recent checkpoint is
  kept; with `--eval_samples` set, the top 2 by `val_loss` plus `last.ckpt`.
- `--export_hf_dir` requires `--state_dict_type full`.
- **`flash_attention_2` may not import.** The prebuilt PyPI `flash-attn` wheel
  (2.8.3.post1) installs but fails against the CUDA 13 torch build with an `undefined
  symbol` c10 ABI mismatch. Build from source (`pip install flash-attn
  --no-build-isolation`) or stay on the `sdpa` default.
- `bf16-mixed` keeps fp32 master weights and autocasts the forward - the safe default for
  full fine-tuning. `bf16-true` puts the optimizer state in bf16 too: less memory, but
  small-magnitude updates can be lost to rounding.
- **DeepSpeed export is a separate step.** ZeRO checkpoints are sharded; run the
  `zero_to_fp32.py` script DeepSpeed writes into the checkpoint folder, then load the result
  and `save_pretrained`.
- **OOM playbook,** in order: turn on `--activation_checkpointing`; drop `--batch_size` to 1
  and raise `--grad_acc_steps` to hold the global batch; lower `--max_seq_len`; add
  `--cpu_offload`; then switch to `--strategy deepspeed --zero_stage 3 --cpu_offload`.
