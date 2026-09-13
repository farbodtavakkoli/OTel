# `training/llm/fsdp` — chat SFT on PyTorch-native FSDP2

> **Coverage:** FSDP2 full-shard SFT is verified on **2, 4 and 8× AMD Instinct MI355X**
> (gfx950, ROCm 7.2.4, torch 2.11.0+rocm7.2) and on **1 and 2× NVIDIA H100 80GB**
> (CUDA 13.0, torch 2.13.0+cu130), including the collective `save_model` at 2 and 4 ranks.
> The Intel XPU path is untested and its commands are a starting point, not a proven
> recipe. Model-specific workarounds differ by torch version — see
> [Platform notes](#platform-notes).

## Overview & when to use

Chat-model SFT on **PyTorch-native FSDP2** (`fully_shard`), driven by HF Transformers'
`Trainer` and launched with accelerate. It supports **full fine-tuning** (default) and an
optional **LoRA** path, and shards parameters, gradients and optimizer state across GPUs
the way DeepSpeed ZeRO-3 does — with **no DeepSpeed in the stack at all**. That is the
reason to pick this folder over `../deepspeed/`: DeepSpeed drags in JIT-compiled
CUDA kernels and a `CUDA_HOME` toolchain, while FSDP2 ships inside torch itself and runs
unmodified on NVIDIA CUDA, AMD ROCm and Intel XPU.

The training script loads `dev.env` right after its imports and reads `HF_TOKEN` from the
environment at model-download time — no token is ever hardcoded. `save_model` runs on
**all** ranks at the end of training because under FSDP it is a collective that
reconstructs the sharded weights; guarding it with `if rank == 0` would hang the other
ranks. `device_map` stays `None` on load — FSDP2 places and shards the parameters itself.

Files in this folder:
- `train_llm_fsdp.py` — the training entrypoint (data loading, masking, Trainer setup).
- `fsdp2_config.yaml` — accelerate config that turns on FSDP2 (`fsdp_version: 2`); the primary launch route.
- `fsdp2_torchrun.json` — Trainer-side FSDP config (`"version": 2`) for the raw `torchrun` route.
- `fsdp2_offload.yaml` — same as `fsdp2_config.yaml` but with `reshard_after_forward: false` + `fsdp_offload_params: true`, for models/torch versions that need it.
- `untie_launch.py` — launch shim that sets `tie_word_embeddings=False` before running the trainer; required for tied-embedding models under torch 2.13 (see Platform notes).
- `requirements_fsdp.txt` — dependencies, with per-accelerator torch install notes.
- `data/OTel_LLM_sample_10.jsonl` — a 10-row sample dataset the script points at by default.

## Install

Python 3.12 in its own venv. **Install `torch` first**, using the index that matches your
accelerator — the plain PyPI wheel is the CUDA build:

```bash
python3.12 -m venv ~/.venv-fsdp && source ~/.venv-fsdp/bin/activate
```

### NVIDIA (CUDA)

```bash
pip install torch==2.11.0
pip install -r requirements_fsdp.txt
```

The `torch==2.11.0` wheel on PyPI is the CUDA 13 build, so no `--index-url` is needed. Add
`--index-url https://download.pytorch.org/whl/cu130` to the first command if you want the
explicit `+cu130` local version tag.

On torch 2.13, two FSDP2 regressions apply that this folder works around — tied embeddings
and `reshard_after_forward: true`; see [Platform notes](#platform-notes).

### AMD (ROCm)

torch 2.11.0 ROCm wheels are published on the **rocm7.2** index (the older `rocm6.4`
index stops at torch 2.9.1):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_fsdp.txt
```

### Intel (XPU)

torch 2.11.0 XPU wheels are on the dedicated `xpu` index:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements_fsdp.txt
```

Verify the accelerator and the FSDP2 entrypoint are both visible:

```bash
python -c "import torch; from torch.distributed.fsdp import fully_shard; \
print('fully_shard OK', torch.accelerator.current_accelerator(), torch.accelerator.device_count())"
```

### Portability: what changes per vendor

FSDP2 is device-agnostic — it shards each parameter as a `DTensor` over a `DeviceMesh`
whose device type comes from the running accelerator, so the training code is identical
on all three vendors. Only the environment changes:

| | NVIDIA CUDA | AMD ROCm | Intel XPU |
|---|---|---|---|
| torch wheel | default PyPI | `--index-url .../whl/rocm7.2` | `--index-url .../whl/xpu` |
| Device API | `torch.cuda` | `torch.cuda` (ROCm reuses the CUDA API surface) | `torch.xpu` |
| Collective backend | NCCL | RCCL (still selected as `nccl`) | XCCL |
| Attention | `sdpa`, or `flash_attention_2` if you can build it | `sdpa` (FA2 wheels are CUDA-only) | `sdpa` |
| Visible-device env var | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` (also `ROCR_VISIBLE_DEVICES`) | `ZE_AFFINITY_MASK` |

Notes that matter in practice:
- `--attn_implementation` defaults to **`sdpa`** because it is the one kernel path
  available everywhere. Only pass `flash_attention_2` on CUDA, and only if `flash-attn`
  actually built (`--no-build-isolation` plus a matching `CUDA_HOME`).
- accelerate selects the process-group backend from the detected device, so
  `fsdp2_config.yaml` contains nothing vendor-specific.
- `bf16` is used throughout; all three targets support it. Do not switch to `fp16` for
  full fine-tuning.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_fsdp.py` loads it with `load_dotenv("dev.env")` and reads `HF_TOKEN` from the
environment, so run the script from inside `training/llm/fsdp/`. `dev.env` is **git-ignored**
at the repo root. Never hardcode a token in the source and never commit one — if a token
was ever committed, rotate it on the Hub immediately.

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The loader reads only the `messages` key from each line, so the sample's extra bookkeeping
columns (`unmask`, `flow`, `source_id`, ...) are ignored — your own data may include or omit
them freely.

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

Smoke test first — one GPU, the shipped sample, one epoch, so failures surface in minutes:

```bash
python train_llm_fsdp.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --logging_steps 1 \
  --output_dir ./fsdp_smoke
```

Full run — **accelerate with the FSDP2 config**, 8 GPUs, from inside `training/llm/fsdp/`:

```bash
nohup accelerate launch --config_file fsdp2_config.yaml --num_processes 8 \
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

Two launch-time adjustments are needed for some model/torch combinations, both covered in
[Platform notes](#platform-notes): add `--fsdp_cpu_ram_efficient_loading false` (required
for gemma-4-E4B-it at every world size), and launch `untie_launch.py` instead of
`train_llm_fsdp.py` — with the same arguments — for any model whose config sets
`tie_word_embeddings: True` when running on torch 2.13. Also pass `--main_process_port` on
a shared box.

LoRA instead of full fine-tuning — same launcher, add `--use_lora` and raise the LR:

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

**Raw torchrun route** (no accelerate config; the Trainer builds the FSDP2 plugin from
its own JSON — pass `--fsdp_config`, otherwise the run is plain DDP and will OOM on a
large model):

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

**What "working" looks like:** a startup line naming the accelerator and model, then the
row-count line (`Loaded N rows ... dropped M over K tokens`), then `{'loss': ...,
'grad_norm': ..., 'learning_rate': ...}` every `--logging_steps` with a finite
`grad_norm` and a loss that moves. Per-GPU memory should sit well below a
single-GPU-replica run — that is the sharding working. At the end you get
`Saved consolidated weights to .../final_model` (or `Saved a LoRA ADAPTER ...`) followed
by `Training complete.`

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL: one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./fsdp_run` | Checkpoints + `final_model/` land here |
| `--resume_from_checkpoint` | `""` | Checkpoint dir to resume from (used only if it exists) |
| `--max_seq_len` | `4096` | Token cap per row; longer rows are dropped, never truncated |
| `--max_samples` | `None` | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for a per-epoch eval (0 = off) |
| `--mask_prompt` / `--no_mask_prompt` | on | Completion-only loss (default) vs full-sequence loss |
| `--batch_size` | `1` | Per-device train/eval batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps (effective batch = batch x acc x world size) |
| `--num_train_epochs` | `3.0` | Number of training epochs |
| `--learning_rate` | `1e-5` | Peak LR (full FT ~1e-5..2e-5; LoRA ~1e-4..2e-4) |
| `--lr_scheduler_type` | `cosine` | LR scheduler type |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--weight_decay` | `0.0` | Weight decay |
| `--logging_steps` | `10` | Log training metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints to keep |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Recompute activations; leave `fsdp_activation_checkpointing: false` in the YAML when using it |
| `--no_save` | off | Disable checkpointing **and** the final `save_model` (use it for smoke runs — see Platform notes) |
| `--attn_implementation` | `sdpa` | `sdpa` (portable default), `flash_attention_2` (CUDA only), `eager` |
| `--fsdp_config` | `None` | Trainer FSDP JSON for the torchrun route; leave unset under accelerate |
| `--use_lora` | off | Train a LoRA adapter instead of full fine-tuning |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or a comma-separated module list |

## Output

Under `--output_dir`:

- `checkpoint-<step>/` — per-epoch Trainer checkpoints (model + optimizer + scheduler
  state), capped by `--save_total_limit`; pass one back via `--resume_from_checkpoint`.
- `final_model/` — the end-of-training artifact plus the tokenizer files. Full
  fine-tuning writes consolidated `safetensors` weights loadable straight with
  `AutoModelForCausalLM.from_pretrained`. With `--use_lora` it is a LoRA adapter instead:
  load the base model and apply it with `PeftModel.from_pretrained`, or merge it with
  `../peft/merge_adapter.py`.
- `runs/` — TensorBoard event files (`report_to="tensorboard"`); view with
  `tensorboard --logdir <output_dir>/runs`.
- `train_llm_fsdp.log` — the redirected stdout/stderr of the run, in this folder.

## Platform notes

Scaling across ranks on ROCm needs **no new flags, no RCCL/NCCL tuning and no OOM
workarounds**. On torch 2.13 (CUDA) two model-specific workarounds below are required.

### Model-specific workarounds

1. **`fsdp_cpu_ram_efficient_loading: true` crashes with gemma-4-E4B-it.** accelerate
   1.14.0's `fsdp2_load_full_state_dict` raises `AttributeError: 'Tensor' object has no
   attribute 'device_mesh'` — some parameters of this architecture stay plain tensors after
   `fully_shard`. Pass `--fsdp_cpu_ram_efficient_loading false` on the `accelerate launch`
   line, or flip the flag in `fsdp2_config.yaml`. It is required at every world size tested.
   Mainstream llama-style models may not need it; the crash is architecture-specific.
2. **Tied embeddings are rejected under FSDP2 + torch 2.13.** Any model with
   `tie_word_embeddings: True` (gemma-4, LFM2) fails the first forward with
   `ValueError: Parameter '...embed_tokens.weight' is shared with a parameter already
   managed by another FSDP group`. PEFT and `TRANSFORMER_BASED_WRAP` put the tied weight
   into two `fully_shard` groups, which FSDP2 forbids — this hits **full FT and LoRA
   alike**, and setting `fsdp_transformer_layer_cls_to_wrap` does not help. Workaround: the
   shipped **`untie_launch.py`**, a shim that sets `tie_word_embeddings=False` at load and
   then runs the training script unchanged. The patch must live in the accelerate *worker*,
   so it has to be the thing accelerate launches — a monkeypatch in the parent process, or
   `sitecustomize.py`, does not take on the Auto metaclass. This is a torch-2.13 regression:
   torch 2.11.0 (the ROCm recipe) tolerates the tied weight.
3. **`fsdp_reshard_after_forward: true` breaks gemma-4's forward under torch 2.13.** After
   untieing, the first forward raises `RuntimeError: aten.where.self got mixed torch.Tensor
   and DTensor` in `modeling_gemma4.py` — the resharded params are DTensors but a
   mask/buffer in the gemma-4 forward is a plain tensor. Workaround: use the shipped
   **`fsdp2_offload.yaml`** (`reshard_after_forward: false`, i.e. ZeRO-2-like, plus
   `fsdp_offload_params: true`). This is model-specific, not universal on torch 2.13 —
   LFM2 runs fine on the shipped `fsdp2_config.yaml` with `reshard_after_forward: true`.
4. **transformers 5.x `apply_chat_template` returns a `BatchEncoding`, not a token list**
   (any hardware, any model), which silently drops every row as "unsupervised". Already
   fixed in `train_llm_fsdp.py` by passing `return_dict=False` at the three `build_example`
   call sites.

### Saving is the memory and disk cliff

`fsdp_state_dict_type: FULL_STATE_DICT` gathers the **fp32** master weights, so saves are
~4 bytes/param regardless of the bf16 training dtype — expect tens of GB per artifact for
an 8B model, and roughly triple that per epoch checkpoint (weights + Adam state).

- Pass `--no_save` for smoke runs, and `--save_total_limit 1` otherwise.
- On a single 80 GB card, full FT of an 8B model needs `fsdp_offload_params: true`
  (without it the optimizer step OOMs), and the save-time gather then spikes HBM to the
  **full card**. Switch to `fsdp_state_dict_type: SHARDED_STATE_DICT`, or defer saving to a
  multi-GPU run where the fp32 gather is spread across ranks.
- During the save phase the non-writing ranks sit at 100% GPU utilisation — that is a
  busy-wait inside the collective barrier, not a hang.
- `--no_save` skips checkpointing **and** the final `save_model`, on *all* ranks. That
  symmetry is mandatory: `save_model` is a collective under FSDP, so a rank-0 guard would
  hang every other rank.

### Environment traps

- **Re-export the GPU mask after activating the venv**, and assert the device count before
  training. A leftover `export CUDA_VISIBLE_DEVICES=4,5` in `.env_fsdp/bin/activate`,
  easily accumulated from an earlier session, silently caps you at 2 GPUs:

  ```bash
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ROCm
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -c "import torch;assert torch.cuda.device_count()==8"   # assert BEFORE training
  ```

- **Pass `--main_process_port`** (accelerate's spelling of torchrun's `--master_port`) to
  avoid rendezvous collisions with other jobs on the box; the default 29500 collides.
- **A venv cannot live on an SMB mount.** `python3 -m venv .env_fsdp` inside a share fails
  with `Operation not permitted: '.../bin/Activate.ps1'` (check with `stat -f`). Build it on
  a local filesystem instead; only the venv location moves.
- **Loading weights over a network mount is very slow** — mmap page-faults block
  the first forward for minutes. Copy the snapshot to local/tmpfs storage first.
- **Offline boxes:** pass the explicit snapshot directory as `--model_name`, not the repo
  id. Under transformers 5.5.0 the cached blob layout does not resolve offline from a repo
  id and falls through to `ValueError: Couldn't instantiate the backend tokenizer`. Set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` with a fully-cached model.

## Hardware support

- **NVIDIA CUDA** — verified (H100 80GB, torch 2.13.0+cu130); `torch==2.11.0` from PyPI is
  also a CUDA 13 build.
- **AMD ROCm** — verified (MI355X / gfx950, torch 2.11.0+rocm7.2 from the `rocm7.2` index;
  the `rocm6.4` index stops at torch 2.9.1).
- **Intel XPU** — `torch==2.11.0+xpu` wheels exist and FSDP2 is device-agnostic, but the XPU
  commands above are untested here.

## Notes

- **FSDP2, not FSDP1.** The current API is `torch.distributed.fsdp.fully_shard`; the legacy
  `FullyShardedDataParallel` wrapper is deliberately not used.
- **Who calls `fully_shard`.** This script does not call it directly; accelerate's FSDP
  plugin (`fsdp_version: 2`) applies it over the transformer blocks selected by
  `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`.
- **`reshard_after_forward` is the memory dial.** `true` = ZeRO-3-like, lowest memory;
  `false` = ZeRO-2-like, higher peak memory.
- **DeepSpeed-free by design.** Nothing here imports `deepspeed`, and no `CUDA_HOME` or
  op-builder JIT step is needed. The mapping if you are coming from
  `../deepspeed/`: ZeRO-3 → `reshard_after_forward: true`, ZeRO-2 →
  `reshard_after_forward: false`, ZeRO-0/DDP → don't use FSDP, CPU optimizer offload →
  `fsdp_offload_params: true`.
- **Chat-template masking** assumes chat templates are prefix-stable (true for mainstream
  instruct models); labels outside assistant spans are `-100`.
- **Checkpointing.** `fsdp_state_dict_type: FULL_STATE_DICT` makes the Trainer write one
  consolidated, directly loadable HF checkpoint. `SHARDED_STATE_DICT` is faster for very
  large models but produces per-rank `.distcp` shards that only load back into FSDP —
  consolidate those with `accelerate merge-weights <sharded_dir> <output>`.
- **Activation checkpointing, once.** Either `--gradient_checkpointing` (Trainer) or
  `fsdp_activation_checkpointing: true` (FSDP) — enabling both raises an error. The
  shipped configs keep the FSDP one off.
- **LoRA under FSDP2.** `--use_lora` wraps the model with PEFT *before* the Trainer hands
  it to accelerate, so the frozen base and the trainable adapters are sharded together.
  The saved artifact is then an adapter, not a full model. 4-bit QLoRA is **not**
  supported here (quantized params are not shardable) — use `../peft/` for that.
