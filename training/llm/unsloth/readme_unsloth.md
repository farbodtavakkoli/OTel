# `training/llm/unsloth` — Unsloth LoRA/QLoRA SFT + GRPO trainer

Standalone Unsloth trainer for Gemma-4-31B: supervised fine-tuning by default, with an
optional GRPO (RL) mode in the same script. Pick it when you want memory-lean LoRA/QLoRA
adapter training (or full FT) on one node without DeepSpeed. Unsloth's OSS build is DDP only —
one full replica per GPU, no FSDP/ZeRO — so the model must fit on a single GPU.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2.4) at 1 and 8 GPUs · NVIDIA H100 80GB
(CUDA 13.0). Upstream lists Instinct MI350-series as fully supported on Linux
(<https://unsloth.ai/docs/basics/amd>).

## Files

- `train_llm_unsloth.py` — the trainer (SFT + GRPO modes).
- `grpo_rewards.py` — swappable GRPO reward-function module.
- `data/OTel_LLM_sample_10.jsonl` — 10-row sample, the default `--train_file`.
- `requirements_unsloth.txt` — pinned environment.
- `vllm_unsloth_setup.md` — the separate `.grpo_env` venv for vLLM-backed GRPO rollouts.

## Setup

Python 3.12, one venv for this folder. Do not install `flash-attn` on either platform —
Unsloth manages attention internally and uses SDPA for gemma-4.

### NVIDIA (CUDA 13)

`torch==2.11.0` / `xformers==0.0.35` / `triton==3.6.0` are genuine `+cu130` wheels, so the
requirements file installs verbatim.

```bash
python3.12 -m venv .env_unsloth && source .env_unsloth/bin/activate
pip install -r requirements_unsloth.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # expect 2.11.0 13.0
```

Installing this file pins torch back to `2.11.0+cu130` if you had a newer one — a downgrade,
still CUDA 13. On a different CUDA, install torch/triton/xformers via Unsloth's official
installer and keep the HF-stack pins from `requirements_unsloth.txt`.

### AMD (ROCm 7.2)

Do **not** install `requirements_unsloth.txt` verbatim — its torch/triton/xformers pins are
CUDA wheels.

```bash
python3 -m venv .env_unsloth && source .env_unsloth/bin/activate

# 1) ROCm PyTorch FIRST (brings triton-rocm 3.6.0 as the `triton` module)
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2

# 2) Unsloth via upstream's AMD extra, plus this folder's HF-stack pins
pip install "unsloth[amd]==2026.8.9" unsloth_zoo==2026.8.6 \
    transformers==5.5.0 trl==0.24.0 peft==0.20.0 datasets==4.3.0 accelerate==1.14.0 \
    bitsandbytes==0.50.0 tokenizers==0.22.2 huggingface_hub==1.27.0 safetensors==0.8.0 \
    sentencepiece==0.2.2 hf_transfer==0.1.9 hf-xet==1.6.0 python-dotenv==1.2.2 \
    numpy==2.3.5 protobuf==6.33.6 tqdm==4.70.0 PyYAML==6.0.3 filelock==3.32.2 packaging==26.3

# 3) REQUIRED cleanup: unsloth[amd] drags in the CUDA triton 3.7.1 (which shadows the ROCm
#    triton module) and the CUDA xformers wheel. Evict both, restore ROCm triton.
pip uninstall -y triton xformers
pip install --force-reinstall --no-deps triton-rocm==3.6.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

torch survives step 2 — the `+rocm7.2` local build satisfies unsloth's requirement — so only
`triton`/`xformers` need evicting. Without step 3, `import triton` resolves to the CUDA build
and Unsloth's Triton kernels break.

`bitsandbytes==0.50.0`'s stock PyPI wheel bundles `libbitsandbytes_rocm72.so` and auto-loads it
on gfx950, so `--load_in_4bit` (QLoRA) and `adamw_8bit` work with the pinned version. Do not
set `HSA_OVERRIDE_GFX_VERSION` on MI355X — that workaround is for gfx942.

### Verify and secrets

```bash
python -c "from unsloth import FastModel; import torch, trl, transformers, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"

ln -sf ../../../dev.env dev.env        # HF_TOKEN; only needed for gated models
export HF_HOME=/path/with/space/huggingface
```

## Data

JSONL, one `{"messages": [...]}` per line, with an optional per-row `"unmask"` flag:

- `unmask: true` — loss on the **entire sequence** (prompt + response).
- `unmask: false` or absent — **completion-only**; the row must contain the response marker
  (`--response_marker`, default `<|turn>model\n`) or the run hard-fails.

The bundled sample carries extra metadata columns (`flow`, `source_repo`, `source_id`,
`source_spec_id`, `source_version`); the SFT path drops all original columns after
tokenization, so extras in your own data are harmless.

GRPO needs a different schema — `prompt` / `answer` / `family` rows, see GRPO mode below.

## Run

### Smoke test, 1 GPU

`--max_steps 5` proves the pipeline end to end and exits:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
python train_llm_unsloth.py \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 5 --learning_rate 2e-4 \
  --experiment_root experiments --output_subdir smoke_1gpu
```

### 8 GPUs, 1 node

Plain DDP (RCCL/NCCL), one full replica per rank, no code or dependency change versus 1 GPU —
only the launcher. Launch multi-GPU with `torchrun`/`accelerate`, never plain `python` (except
1 GPU or `--device_map balanced`).

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # and HIP_VISIBLE_DEVICES too on ROCm
export RUN_ID=$(date -u +%Y%m%d_%H%M%S)       # mandatory: every rank reads it for the output dir
python -c "import torch; assert torch.cuda.device_count()==8"

torchrun --nproc_per_node 8 --master_port 29590 train_llm_unsloth.py \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 5 --learning_rate 2e-4 \
  --experiment_root experiments --output_subdir smoke_8gpu_1node
```

Re-export the device list **after** activating the venv: a stale `CUDA_VISIBLE_DEVICES=4` (or
`HIP_VISIBLE_DEVICES`) left in `bin/activate` pins every rank to one card and quietly produces
a fake "N-GPU" result.

Use a dataset with at least `batch_size x world_size` rows for a multi-GPU smoke test; the
10-row sample is only sufficient at 1 GPU.

```
==((====))==  Unsloth 2026.8.9: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    AMD Instinct MI355X. Num GPUs = 8. Max memory: 287.984 GB. Platform: Linux.
\        /    Data Parallel GPUs = 8 | Total batch size (2 x 1 x 8) = 16
{'loss': '0.3762', 'grad_norm': '0.3065', 'learning_rate': '0', 'epoch': '1'}
```

Ignore the `Num GPUs used = 1` banner line — Unsloth prints it per rank. The authoritative
lines are `Num GPUs = N` and `Data Parallel GPUs = N | Total batch size (b x a x N)`. An
occasional `grad_norm: nan` on a single step is a benign bf16-QLoRA logging artifact if loss
keeps dropping.

### Full run

Drop `--max_steps` (epochs take over via `--num_train_epochs`) and point `--train_file` at your
real dataset. Effective batch = `batch_size x grad_acc_steps x total_GPUs`.

Multi-node: the same `torchrun` command on every node with
`--nnodes N --node_rank <i> --master_addr <node0-ip> --master_port <port>`, an identical
`RUN_ID`, and `--experiment_root` on a shared filesystem. Each GPU still holds a full replica,
so multi-node adds throughput, not room for a bigger model.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--train_mode` | `sft` | `sft` or `grpo` (RL) |
| `--model_name` | `unsloth/gemma-4-31B-it` | Hub id or local snapshot dir |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Training JSONL |
| `--output_dir` | None | Full output-dir override |
| `--experiment_root` | `experiments` | Root for run outputs (`<root>/<RUN_ID>/<subdir>`) |
| `--output_subdir` | `gsma_sft_lora_unsloth` | Subdir under `experiment_root/<RUN_ID>` |
| `--max_seq_length` | 3100 | Over-length rows are dropped, not truncated |
| `--load_in_4bit` | off | On-the-fly 4-bit (QLoRA) or load a 4-bit ckpt |
| `--load_in_8bit` | off | On-the-fly 8-bit; mutually exclusive with 4-bit |
| `--full_finetuning` | off | Full FT instead of LoRA (heavy) |
| `--device_map` | None | `balanced` splits one replica across GPUs (single process); unset = DDP |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 8 | LoRA alpha |
| `--lora_dropout` | 0.0 | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or comma-separated module names |
| `--batch_size` | 1 | Per-device batch size |
| `--grad_acc_steps` | 8 | Gradient-accumulation steps |
| `--num_train_epochs` | 3 | Epochs |
| `--max_steps` | -1 | >0 = smoke test (overrides epochs) |
| `--learning_rate` | 2e-4 | LR (auto-drops to 5e-6 for GRPO if left at the SFT default) |
| `--lr_scheduler_type` | `cosine` | Scheduler |
| `--weight_decay` | 0.01 | Weight decay |
| `--warmup_ratio` | 0.03 | Warmup ratio |
| `--optim` | `adamw_8bit` | Optimizer (8-bit optimizer states) |
| `--logging_steps` | 1 | Log interval |
| `--save_strategy` | `epoch` | `no` / `epoch` / `steps` |
| `--save_steps` | 250 | Checkpoint interval for `--save_strategy steps` |
| `--seed` | 42 | Seed |
| `--mask_prompt` / `--no-mask_prompt` | on | Per-row masking policy (off = full-sequence loss) |
| `--response_marker` | `<\|turn>model\n` | Marker opening the assistant turn |
| `--loss_weight` | 1.0 | Per-token loss multiplier for completion-only rows |
| `--report_to` | `none` | Trainer reporting backend |
| `--eval_samples` | 1000 | Held-out eval split size |
| `--test_mode` | off | Use only the first `--test_mode_count` rows |
| `--test_mode_count` | 10000 | Row cap for `--test_mode` |
| `--num_generations` | 8 | [GRPO] Rollouts per prompt |
| `--max_prompt_length` | 1024 | [GRPO] Longer prompts are dropped |
| `--max_completion_length` | None | [GRPO] Default `max_seq_length - max_prompt_length` |
| `--grpo_temperature` | 1.0 | [GRPO] Rollout sampling temperature (>0 required) |
| `--turn_end_token` | `<turn\|>` | [GRPO] End-of-turn token set as EOS so rollouts stop |
| `--reward_mode` | `rule` | [GRPO] `rule` / `llm` / `hybrid` |
| `--reward_module` | `grpo_rewards` | [GRPO] Module exposing `build_reward_funcs(reward_mode, judge_model)` |
| `--judge_model` | `claude-sonnet-5` | [GRPO] LLM-judge model id (llm/hybrid only) |
| `--fast_inference` | off | [GRPO] Colocated vLLM rollouts (see GRPO mode) |
| `--gpu_memory_utilization` | 0.9 | [GRPO] vLLM KV-cache VRAM fraction |
| `--vllm_server` | off | [GRPO] Route rollouts to an external vLLM server |
| `--vllm_server_host` | `127.0.0.1` | [GRPO] vLLM server host |
| `--vllm_server_port` | 8000 | [GRPO] vLLM server port |

## Output

Checkpoints land in `experiment_root/<RUN_ID>/<output_subdir>/` (or `--output_dir`); rank 0
saves the final LoRA adapter plus tokenizer to `.../final_model/`. `RUN_ID` comes from the
environment so all ranks agree — set it explicitly for multi-process runs.

## Quantization

The base model is frozen and only the LoRA adapter trains, so quantizing the base cuts VRAM at
little quality cost:

| Flag | Base precision | ~VRAM (31B, bs8, 4200 ctx) | Notes |
|---|---|---|---|
| `--load_in_4bit` | 4-bit NF4 (QLoRA) | ~22 GB/GPU | Best VRAM; quality approx. 16-bit for LoRA |
| `--load_in_8bit` | 8-bit int8 | ~36 GB/GPU | Higher fidelity, slower steps |
| *(neither)* | bf16 | ~60+ GB/GPU | Full-precision base |

Two ways to a 4-bit base: a pre-quantized checkpoint
(`unsloth/gemma-4-31b-it-unsloth-bnb-4bit`) or a 16-bit base plus `--load_in_4bit`. `adamw_8bit`
quantizes *optimizer states*, independent of base dtype. Compressed-tensors checkpoints
(`*-qat-w4a16*`, `*-FP8-dynamic`) are inference-only — use bitsandbytes NF4/int8 for training.
`--full_finetuning` needs a 16-bit base and no quantization.

## GRPO mode

`--train_mode grpo` runs GRPO with the same launch shape as SFT: each step samples
`--num_generations` completions per prompt from the current policy, scores them with reward
functions, computes group-relative advantages, and updates the LoRA. On-device rollouts (no
vLLM) are the path that works in this venv.

Data is `prompt` / `answer` / `family` rows:

| field | meaning |
|---|---|
| `prompt` | chat messages **without** the assistant turn |
| `answer` | gold answer the reward checks (`""` for open-ended) |
| `family` | tag your reward function can branch on |

The batching invariant `batch_size x grad_acc_steps x world_size % num_generations == 0` is
enforced with a hard exit.

**On ROCm, export `UNSLOTH_GRPO_SEQ_PACKING=0` before a GRPO run.** With Unsloth's GRPO
sequence-packing on (the default), the first backward dies with
`torch.utils.checkpoint.CheckpointError: Recomputed values ... different metadata`. Leave
`UNSLOTH_GRPO_PREFIX_GROUPER` alone — disabling only the grouper does not help.

### Reward modules

Rewards come from a swappable module (`--reward_module`, default `grpo_rewards`) exposing
`build_reward_funcs(reward_mode, judge_model)`. TRL calls each reward as
`reward_fn(prompts, completions, **columns) -> list[float]`, where every extra dataset column
arrives by keyword as an aligned list. Conversational datasets hand completions back as
message-dict lists, so normalize to text first (see `_text` in `grpo_rewards.py`).

Two templates ship: `reward_verifiable` (normalized exact match against `answer`) and
`make_llm_judge_reward` (scores open-ended answers 0-10 via the OpenAI SDK; configured by
`JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY`, falling back to `OPENAI_API_KEY`). Register
yours in `build_reward_funcs()` or point `--reward_module` at your own module. **Never raise
inside a reward** — a raised error kills the whole GRPO step; catch and return 0.0.

Watch `reward` / `rewards/*/mean` trending up (movement can take a few hundred steps). A
`frac_reward_zero_std` near 1 means every rollout in a group scored the same, so there is no
gradient — raise `--num_generations`. A high `completions/clipped_ratio` means rollouts hit the
length cap without terminating — raise `--max_completion_length`.

### vLLM rollouts

Colocated `--fast_inference` is blocked for gemma-4: Unsloth's `VLLM_SUPPORTED_VLM` allowlist
excludes the multimodal `gemma4` arch, and the script fails fast with an actionable message.
For server-mode rollouts (`--vllm_server`), use the separate `.grpo_env` venv — vLLM/TRL pin
against each other in ways that conflict with this venv's pins. See
[`vllm_unsloth_setup.md`](vllm_unsloth_setup.md).

On-device GRPO is slow: bound experiments with `--max_steps` and keep
`--max_completion_length` tight.

## Notes

- **Import order is load-bearing.** `from unsloth import FastModel` is deliberately the first
  import in `train_llm_unsloth.py` — Unsloth patches torch/transformers/trl at import time. Do
  not reorder it below other imports.
- Multi-GPU DDP works on the pinned `trl==0.24.0` + `unsloth==2026.8.9`. If a later upgrade
  starts raising `RuntimeError: Unsloth currently does not support multi GPU setups`, pin back
  to those versions.
- Do not upgrade transformers past 5.5.x in this venv: 5.15+ breaks the gemma-4 load with a
  per-layer `head_dim` regression.
- If model downloads fail with 403s from huggingface.co, unset the proxy variables for the run
  (`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python train_llm_unsloth.py ...`).
  Offline: pre-download the model, pass its local snapshot dir as `--model_name`, and set
  `HF_HUB_OFFLINE=1`.
- `--loss_weight 1.0` uses Unsloth's fused CE (bs8 fine at 4200 ctx); `>1.0` uses a custom
  `compute_loss` that materializes logits — drop to bs4 at 4200 ctx to avoid OOM.
- A `unsloth_compiled_cache/` directory appears next to the script on first run; it is safe to
  delete.
