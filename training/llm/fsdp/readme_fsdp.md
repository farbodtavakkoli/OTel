# `training/llm/fsdp` — chat SFT on PyTorch-native FSDP2

Chat-model SFT on PyTorch-native FSDP2 (`fully_shard`), driven by HF Transformers'
`Trainer` and launched with accelerate. Full fine-tuning by default, with an optional LoRA
path. It shards parameters, gradients and optimizer state the way DeepSpeed ZeRO-3 does,
with no DeepSpeed in the stack: FSDP2 ships inside torch, so there are no JIT-compiled
kernels and no `CUDA_HOME` toolchain. Pick this over `../deepspeed/` for that reason.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4, torch 2.11.0+rocm7.2), 2/4/8 GPUs · NVIDIA
H100 80GB (CUDA 13.0, torch 2.13.0+cu130), 1 and 2 GPUs.

## Files

| File | Purpose |
|---|---|
| `train_llm_fsdp.py` | Training entrypoint: data loading, chat-template masking, Trainer setup |
| `fsdp2_config.yaml` | accelerate config enabling FSDP2 (`fsdp_version: 2`); the primary launch route |
| `fsdp2_offload.yaml` | Low-HBM variant: `reshard_after_forward: false` + `fsdp_offload_params: true` |
| `fsdp2_torchrun.json` | Trainer-side FSDP config (`"version": 2`) for the raw `torchrun` route |
| `untie_launch.py` | Launch shim setting `tie_word_embeddings=False`; needed for tied-embedding models on torch 2.13 |
| `requirements_fsdp.txt` | Pinned dependencies (Python 3.12) |
| `data/OTel_LLM_sample_10.jsonl` | 10-row chat sample; the default `--train_file` |

## Setup

Python 3.12, one venv per recipe folder. Install `torch` FIRST from the index matching your
accelerator - the plain PyPI wheel is the CUDA build - then the requirements on top, which
resolve unchanged.

```bash
python3.12 -m venv .env_fsdp && source .env_fsdp/bin/activate
```

### NVIDIA (CUDA 13)

```bash
pip install torch==2.11.0
pip install -r requirements_fsdp.txt
```

Add `--index-url https://download.pytorch.org/whl/cu130` to the first command if you want
the explicit `+cu130` local version tag. On torch 2.13 two model-specific workarounds
apply - see [Model workarounds](#model-workarounds).

### AMD (ROCm 7.2)

torch 2.11.0 ROCm wheels are on the `rocm7.2` index (the older `rocm6.4` index stops at
torch 2.9.1).

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_fsdp.txt
```

### Verify

```bash
python -c "import torch; from torch.distributed.fsdp import fully_shard; \
print('fully_shard OK', torch.accelerator.current_accelerator(), torch.accelerator.device_count())"
```

### Secrets

`HF_TOKEN` for gated models comes from `dev.env` in this folder, loaded by
`load_dotenv("dev.env")` - so run the script from inside `training/llm/fsdp/`:

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

### What changes per vendor

FSDP2 shards each parameter as a `DTensor` over a `DeviceMesh` whose device type comes from
the running accelerator, so the training code and `fsdp2_config.yaml` are identical on both
vendors. Only the environment differs:

| | NVIDIA CUDA | AMD ROCm |
|---|---|---|
| torch wheel | default PyPI | `--index-url .../whl/rocm7.2` |
| Device API | `torch.cuda` | `torch.cuda` (ROCm reuses the CUDA surface) |
| Collective backend | NCCL | RCCL (still selected as `nccl`) |
| Attention | `sdpa`, or `flash_attention_2` if you built `flash-attn` | `sdpa` (FA2 wheels are CUDA-only) |
| Visible-device env var | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` (also `ROCR_VISIBLE_DEVICES`) |

`bf16` is used throughout on both; do not switch to `fp16` for full fine-tuning.

## Data

`--train_file` takes a chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Only the `messages` key is read, so the shipped sample's extra columns (`unmask`, `flow`,
`source_id`, ...) are ignored and your own data may include or omit them. A `system` turn
and extra turns are fine. Each row is rendered with the tokenizer's own
`apply_chat_template`, so a model whose tokenizer has no chat template is rejected up front.

Loss is completion-only by default (`--mask_prompt`): every token outside an assistant turn
is set to `-100`, and in a multi-turn row all assistant turns are supervised. Pass
`--no_mask_prompt` to train on the full rendered sequence. Rows longer than `--max_seq_len`
and rows with no supervised token are dropped, never truncated; both counts are logged.

## Run

Smoke test - one GPU, the shipped sample, one epoch:

```bash
python train_llm_fsdp.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --logging_steps 1 \
  --output_dir ./fsdp_smoke
```

Full run - accelerate with the FSDP2 config, 8 GPUs, from inside `training/llm/fsdp/`:

```bash
nohup accelerate launch --config_file fsdp2_config.yaml --num_processes 8 \
  --main_process_port 29650 \
  train_llm_fsdp.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_full_ft \
  --max_seq_len 4096 \
  --batch_size 1 --grad_acc_steps 8 \
  --num_train_epochs 3 --learning_rate 1e-5 \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &

tail -f train_llm_fsdp.log
```

LoRA instead of full fine-tuning - same launcher, add `--use_lora` and raise the LR:

```bash
nohup accelerate launch --config_file fsdp2_config.yaml --num_processes 8 \
  train_llm_fsdp.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_lora \
  --use_lora --lora_r 32 --lora_alpha 64 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &
```

Raw torchrun route - pass `--fsdp_config`, or the run is plain DDP and will OOM on a large
model:

```bash
nohup torchrun --standalone --nproc_per_node=8 \
  train_llm_fsdp.py \
  --fsdp_config fsdp2_torchrun.json \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_full_ft \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &
```

Scaling across ranks on ROCm needs no new flags and no RCCL/NCCL tuning. Re-export the GPU
mask after activating the venv and assert the device count before training - a leftover
`export CUDA_VISIBLE_DEVICES=4,5` in `.env_fsdp/bin/activate` silently caps you at 2 GPUs:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ROCm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch;assert torch.cuda.device_count()==8"
```

Pass `--main_process_port` on a shared box; the default 29500 collides.

### Model workarounds

Required on torch 2.13; torch 2.11.0 (the ROCm recipe) needs neither.

- **gemma-4-E4B-it:** pass `--fsdp_cpu_ram_efficient_loading false` on the
  `accelerate launch` line (or flip it in `fsdp2_config.yaml`) at every world size.
  accelerate 1.14.0's `fsdp2_load_full_state_dict` otherwise raises
  `AttributeError: 'Tensor' object has no attribute 'device_mesh'`.
- **Tied embeddings** (`tie_word_embeddings: True` - gemma-4, LFM2): launch
  `untie_launch.py` instead of `train_llm_fsdp.py`, with the same arguments. It must be the
  thing accelerate launches, because the patch has to land in the worker process. Affects
  full FT and LoRA alike.
- **gemma-4 forward under `reshard_after_forward: true`:** use `fsdp2_offload.yaml`
  instead of `fsdp2_config.yaml`. Model-specific - LFM2 runs fine on the default config.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL, one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./fsdp_run` | Checkpoints and `final_model/` land here |
| `--resume_from_checkpoint` | `""` | Checkpoint dir to resume from (used only if it exists) |
| `--max_seq_len` | `4096` | Token cap per row; longer rows are dropped |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--eval_samples` | `0` | Rows held out for a per-epoch eval (0 = off) |
| `--mask_prompt` / `--no_mask_prompt` | on | Completion-only loss vs full-sequence loss |
| `--batch_size` | `1` | Per-device train/eval batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps (effective batch = batch x acc x world size) |
| `--num_train_epochs` | `3.0` | Epochs |
| `--learning_rate` | `1e-5` | Peak LR (full FT ~1e-5..2e-5; LoRA ~1e-4..2e-4) |
| `--lr_scheduler_type` | `cosine` | LR scheduler |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--weight_decay` | `0.0` | Weight decay |
| `--logging_steps` | `10` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Recompute activations; keep `fsdp_activation_checkpointing: false` |
| `--no_save` | off | Disable checkpointing and the final `save_model` (smoke runs) |
| `--attn_implementation` | `sdpa` | `sdpa` (portable), `flash_attention_2` (CUDA only), `eager` |
| `--fsdp_config` | `None` | Trainer FSDP JSON for the torchrun route; leave unset under accelerate |
| `--use_lora` | off | Train a LoRA adapter instead of full fine-tuning |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or a comma-separated module list |

## Output

Under `--output_dir`:

- `checkpoint-<step>/` - per-epoch Trainer checkpoints (model + optimizer + scheduler),
  capped by `--save_total_limit`; pass one back via `--resume_from_checkpoint`.
- `final_model/` - consolidated `safetensors` weights loadable with
  `AutoModelForCausalLM.from_pretrained`, plus tokenizer files. With `--use_lora` it is a
  LoRA adapter instead: apply it with `PeftModel.from_pretrained`, or merge it with
  `../peft/merge_adapter.py`.
- `runs/` - TensorBoard event files; view with `tensorboard --logdir <output_dir>/runs`.

## Notes

- **Saving is the memory and disk cliff.** `fsdp_state_dict_type: FULL_STATE_DICT` gathers
  the fp32 master weights, so artifacts are ~4 bytes/param regardless of bf16 training -
  tens of GB for an 8B model, roughly triple that per epoch checkpoint. Pass `--no_save`
  for smoke runs and `--save_total_limit 1` otherwise.
- On a single 80 GB card, full FT of an 8B model needs `fsdp_offload_params: true` or the
  optimizer step OOMs, and the save-time gather then spikes HBM to the full card. Use
  `SHARDED_STATE_DICT`, or defer saving to a multi-GPU run where the gather is spread
  across ranks.
- Non-writing ranks sit at 100% GPU during the save; that is a busy-wait inside the
  collective barrier, not a hang.
- **`reshard_after_forward` is the memory dial:** `true` = ZeRO-3-like, lowest memory;
  `false` = ZeRO-2-like, higher peak. Coming from `../deepspeed/`: ZeRO-3 ->
  `reshard_after_forward: true`, ZeRO-2 -> `false`, CPU optimizer offload ->
  `fsdp_offload_params: true`.
- **Enable activation checkpointing once.** Either `--gradient_checkpointing` (Trainer) or
  `fsdp_activation_checkpointing: true` (FSDP) - both at once raises an error. The shipped
  configs keep the FSDP one off.
- `SHARDED_STATE_DICT` produces per-rank `.distcp` shards that only load back into FSDP;
  consolidate them with `accelerate merge-weights <sharded_dir> <output>`.
- 4-bit QLoRA is not supported here (quantized params are not shardable) - use `../peft/`.
- Build the venv on a local filesystem: `python3 -m venv` on an SMB mount fails with
  `Operation not permitted: '.../bin/Activate.ps1'`.
- Copy model snapshots to local or tmpfs storage before training; mmap page-faults over a
  network mount block the first forward for minutes.
- On offline boxes pass the explicit snapshot directory as `--model_name`, not the repo id,
  with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Under transformers 5.5.0 a repo id does
  not resolve offline and fails with `ValueError: Couldn't instantiate the backend tokenizer`.
