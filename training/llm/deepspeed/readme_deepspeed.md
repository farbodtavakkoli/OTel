# `training/llm/deepspeed` — generic DeepSpeed post-training (SFT / DPO / GRPO)

## Overview & when to use

Generic post-training for conversational `messages` models on **HF Transformers +
DeepSpeed (ZeRO)**. Supports **SFT**, **DPO**, and **GRPO**, LoRA or full fine-tuning,
and any model whose tokenizer ships a chat template — no chat template means the run
hard-fails, by design, to guarantee train/inference parity. Data/eval/loss helpers and
the DPO/GRPO dataset builders live in `utils.py`; the training recipe itself lives in
`train_llm_deepspeed.py`.

Use this folder when you want chat-template parity, LoRA/QLoRA, or preference/RL modes.
Use `../deepspeed_standalone/` when you want a single portable file with
hand-written prompt templates and SFT only.

Design notes absorbed from the code:

- **DeepSpeed, not one-replica-per-GPU** — the DeepSpeed config is built in-memory by
  `utils.build_deepspeed_config` from `--zero_stage`/`--offload_optimizer` and passed as
  `SFTConfig(deepspeed=...)`, so every rank gets an identical config and there is no
  `ds_config.json` file (an in-memory dict takes priority over any file).
- **Chat-template masking** — each row is tokenized via the tokenizer's official
  `apply_chat_template` (the same rendering inference uses), with
  `add_special_tokens=False` because the template already injects BOS. Completion-only
  masking sets prompt labels to `-100`; `_stable_causal_lm_loss` averages cross-entropy
  over supervised tokens only and clamps the denominator so a micro-batch with zero
  supervised tokens yields 0.0 instead of NaN.
- **Data pipeline** (`utils.get_datasets`) — load → `messages`-contract preflight on an
  evenly-spaced sample → optional subsetting (`--test_mode` / `--sample_fraction` /
  `--max_samples`) → tokenize + mask → drop rows over `--max_token_length` (never
  truncate a completion) → seed-stable train/eval split with a distributed-safe floor
  (at least one eval row per rank).
- **SFT loss configuration** — `loss_type="nll"` is used instead of TRL's default
  `chunked_nll`, which needs an `outputs.num_valid_tokens` field Gemma-4 doesn't
  provide; `max_length=None` / `packing=False` keep the pre-tokenized, length-filtered
  tensors as-is (TRL's 1024 default would truncate long rows).
- **`--load_best_model_at_end` is OFF by default** — for memorization-style training the
  last checkpoint is what you want: held-out eval_loss *rises* as the train set is
  memorized, so the "best" eval-loss checkpoint is the least-memorized one.
- **QLoRA guard** — `--load_in_4bit` requires `--use_lora` and is incompatible with
  ZeRO-3 (4-bit params are not shardable); the script fails fast on that combination.
- **DPO memory note** — `--dpo_precompute_ref_log_probs` computes the reference model's
  log-probs once up front and drops the ref model from the training loop, halving the
  concurrent forward passes so a larger batch or max_length fits.
- **GRPO reward is a placeholder** — `utils.grpo_reward_funcs` rewards non-empty
  completions; replace it with task-specific checks or an LLM judge before real GRPO.
- **Optimizer names** — `--optim adamw` maps to HF `adamw_torch` (DeepSpeed-native);
  `adamw_bnb_8bit` is available but has a known sharp edge with ZeRO-3.

> **Tested topology:** Azure cluster, 8×H100 80GB — single-node DeepSpeed ZeRO-3.
> Also verified on 2×AMD MI355X (ROCm 7.2), single-node ZeRO-2 — see
> "MI355X (ROCm 7.2) — tested" below. Multi-node is possible via
> `accelerate`/`torchrun` rendezvous but has **not** been tested here.

## Install

Python 3.12. All pins in `requirements_deepspeed.txt` are the tested set.

### NVIDIA (CUDA)

```bash
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r requirements_deepspeed.txt
```

DeepSpeed JIT-compiles its C++/CUDA ops against the CUDA toolkit, so export the toolkit
paths (the script defaults `CUDA_HOME` to `/usr/local/cuda-13` if unset — a pre-set
`CUDA_HOME` wins):

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify the import graph:

```bash
python -c "import torch, deepspeed, trl, transformers, peft, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### AMD / ROCm

**Tested on 2×MI355X (gfx950), ROCm 7.2.4 — see "MI355X (ROCm 7.2) — tested" below for
the verified commands and per-algorithm verdicts.** The differences from the NVIDIA
install:

- **PyTorch** — install from the ROCm wheel index instead of PyPI, **before** the rest
  of the requirements:

  ```bash
  pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
  ```

  The rocm7.2 index satisfies the pins exactly (`torch 2.11.0+rocm7.2`,
  `torchvision 0.26.0+rocm7.2`) — no version lag against the CUDA pins anymore.
- **DeepSpeed** — same `pip install deepspeed==0.19.4`; it lands as a **pure-python
  wheel with no hipcc compile at install time**. Ops are JIT-only, and none were needed
  in the tested ZeRO-2 runs (`adamw_torch` is DeepSpeed-native), so no `CUDA_HOME`/
  toolkit export is required — the script's `/usr/local/cuda-13` default is harmless
  on ROCm.
- **flash-attn** — this trainer does **not** use FlashAttention (`--flash_attention
  sdpa` is required; see Notes), so nothing to install. If you adapt the code for FA2 on
  ROCm, the upstream flash-attention repo provides ROCm backends (composable_kernel and
  Triton) for MI200x–MI355x with head dims up to 256 — a source build with
  `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`, not the CUDA wheel.
- **bitsandbytes** — official ROCm support; the regular PyPI wheels include ROCm builds
  (ROCm 6.4.4–7.14, Instinct gfx90a/gfx942/gfx950). Needed only for
  `--optim adamw_bnb_8bit` or `--load_in_4bit`.

## Environment & secrets

Optional `dev.env` **in this folder**, only needed to pull gated models from the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The script loads it via `load_dotenv("dev.env")` — run the script from inside
`training/llm/deepspeed/` so the relative path resolves. `dev.env` is git-ignored at the
repo root; never commit a token. The script also sets `HF_HOME` (`--hf_home`), NCCL env
vars, and clears proxy vars automatically (a proxy that 403s huggingface.co was seen on
the tested cluster).

Launch with `accelerate launch --use_deepspeed`. The only accelerate-config fields that
matter are `distributed_type: DEEPSPEED` and `zero3_init_flag` — a minimal
`~/.cache/huggingface/accelerate/default_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero3_init_flag: true
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 8
use_cpu: false
```

## Data

- **SFT / GRPO** — JSONL, one `{"messages": [{"role": ..., "content": ...}, ...]}` per
  line; the final message must be a non-empty `assistant` turn. `--mask_prompt` trains
  completion-only; without it the whole sequence is supervised. Over-length rows
  (`> --max_token_length`) are dropped, never truncated.
- **DPO** — a `--pref_file` JSONL of `{system?, prompt, chosen, rejected}`.

A 10-row sample ships at `data/OTel_LLM_sample_10.jsonl` and is the default
`--train_file`. Its schema is the canonical `messages` format plus extra metadata
columns: `unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
`source_version`.

**The null-column trap** — in the full dataset this sample comes from,
`source_spec_id` and `source_version` are NULL in every row, so a naive
`datasets`/pyarrow load infers a `null` dtype for them and can break schema-sensitive
pipelines. This trainer is immune: it only reads the `messages` column and
`remove_columns=<all>` drops every extra column at tokenization time. If you write your
own loader for this data, ignore or drop the extra columns rather than relying on their
inferred dtype.

To swap in real data, point `--train_file` at your own `messages` JSONL (the tested
runs used absolute paths like `/mnt/gsma/FTaaS_data/UC524_combined_op_clean_smoke_10k.jsonl`
— any path works). Any row failing the `messages` contract fails the preflight with the
row index and reason.

## Run

Run from inside `training/llm/deepspeed/`. Effective batch =
`batch_size × grad_acc_steps × num GPUs`.

**Smoke test against the shipped sample** (1 GPU; the 10-row sample leaves 8 train /
2 eval rows):

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=1 --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed.py \
  --train_mode sft --train_file data/OTel_LLM_sample_10.jsonl \
  --model_name <small-chat-model-id> \
  --flash_attention sdpa --gradient_checkpointing --mask_prompt \
  --eval_samples 2 --num_train_epochs 1 --batch_size 1 --grad_acc_steps 1 \
  --zero_stage 2 \
  --experiment_root experiments/ --output_subdir smoke_sample \
  > smoke_sample.log 2>&1
```

**Full example — 8 GPUs, LoRA SFT, ZeRO-2** (recommended for LoRA; the model fits
per-GPU):

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed.py \
  --train_mode sft --train_file /path/to/train.jsonl \
  --model_name google/gemma-4-31B-it \
  --max_token_length 3100 --mask_prompt --gradient_checkpointing --flash_attention sdpa \
  --use_lora --zero_stage 2 \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.0 --lora_target_modules all-linear \
  --optim adamw_bnb_8bit \
  --batch_size 4 --grad_acc_steps 2 --num_train_epochs 1 --learning_rate 2e-4 \
  --test_mode --test_mode_count 256 \
  --experiment_root experiments/ --output_subdir smoke_sft_lora \
  > smoke_sft.log 2>&1
```

**What "working" looks like:** the `CHECKPOINT SAVING LOCATION` banner, a tokenization
sanity block (per-dataset masked/trained token counts), then `{'loss': ...}` lines with
a finite `grad_norm`, and finally `Saving final ...` + `Training Complete.`

For a full run, drop `--test_mode*` and set `--num_train_epochs 3`. DPO/GRPO can
continue on top of an existing SFT adapter via
`--use_lora --init_adapter /path/to/sft/final_model`.

## Arguments

Defaults are the tested values, except paths, which were absolute on the tested cluster
(old defaults noted in Data / Output).

| Flag | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Training data — canonical `messages` JSONL |
| `--model_name` | `google/gemma-4-E4B-it` | HF model id or local path (tokenizer must have a chat template) |
| `--experiment_root` | `experiments/` | Root under which run outputs are created |
| `--hf_home` | `hf_cache/` | `HF_HOME` cache root for models/datasets |
| `--output_subdir` | `gemma4_ftaas_uc524_finetuned` | Subdirectory under `experiment_root/<RUN_ID>` |
| `--output_dir` | `None` | Full output-dir override (skips the construction above) |
| `--resume_from_checkpoint` | `""` | Checkpoint path to resume from (used only if it exists) |
| `--load_best_model_at_end` | off | Load the lowest-eval-loss checkpoint as final (see Overview for why off) |
| `--max_token_length` | `32768` | Max tokens per example; longer rows are dropped, never truncated |
| `--eval_samples` | `1000` | Target absolute eval-set size (floored at world size) |
| `--preflight_sample_size` | `256` | Rows validated in the fail-fast `messages` preflight |
| `--supervision_sample_size` | `16` | Rows sampled for the supervision sanity check |
| `--test_mode` | off | Cap the dataset at `--test_mode_count` rows |
| `--test_mode_count` | `10000` | Row cap applied when `--test_mode` is set |
| `--seed` | `42` | Seed for `set_seed` and dataset shuffling/splitting |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use (seeded shuffle + subsample) |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--num_proc` | `8` | Worker processes for dataset map/filter steps |
| `--batch_size` | `4` | Per-device train/eval batch size |
| `--grad_acc_steps` | `2` | Gradient accumulation steps |
| `--num_train_epochs` | `3` | Epochs (memorization: start 3, extend to ~8 by the per-epoch eval curve) |
| `--learning_rate` | `2e-4` | Peak LR (LoRA default; use ~1e-5..2e-5 for full FT) |
| `--lr_scheduler_type` | `cosine` | LR scheduler |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `0.03` | <1 = fraction of total steps; >=1 = absolute step count |
| `--logging_steps` | `50` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--optim` | `adamw` | `adamw` (→ `adamw_torch`), `sgd`, `rmsprop`, or `adamw_bnb_8bit` |
| `--mask_prompt` | off | Completion-only loss (mask the prompt) |
| `--gradient_checkpointing` | off | Enable gradient checkpointing |
| `--flash_attention` | `flash_attention_2` | `flash_attention_2` or `sdpa` — **must pass `sdpa`** (see Notes) |
| `--use_lora` | off | Train a LoRA adapter instead of full FT |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha (typically 2× rank) |
| `--lora_dropout` | `0.0` | LoRA dropout (0.0 when the goal is to fit the data) |
| `--lora_target_modules` | `all-linear` | `all-linear` or comma-separated module list |
| `--load_in_4bit` | off | QLoRA 4-bit base; requires `--use_lora` and `--zero_stage 0` |
| `--custom_eval` | off | Generation-based per-epoch eval via `--scorer_module` |
| `--test_dir` | `test` | Dir with `<name>_eval.jsonl` files for `--custom_eval` |
| `--scorer_module` | `step8_score_eval` | Module exposing `EVAL_DATASETS`/`load_eval`/`score_dataset`/`macro_average` |
| `--eval_max_new_tokens` | `768` | Max new tokens per prompt during custom eval |
| `--empty_cache_steps` | `0` | Flush CUDA cache every N steps on all ranks (0 = off) |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage (see table below) |
| `--offload_optimizer` | off | Offload optimizer state to CPU pinned memory |
| `--train_mode` | `sft` | `sft`, `dpo`, or `grpo` |
| `--init_adapter` | `None` | Existing LoRA adapter to continue training from |
| `--pref_file` | `None` | DPO preference JSONL — required for `--train_mode dpo` |
| `--dpo_beta` | `0.1` | DPO KL strength (lower = closer to the reference policy) |
| `--dpo_max_length` | `4096` | DPO max total sequence length (prompt+completion) |
| `--dpo_max_prompt_length` | `3072` | DPO max prompt length (not passed to this TRL version's DPOConfig) |
| `--dpo_precompute_ref_log_probs` | off | Cache ref log-probs up front; drop the ref model from the loop |
| `--grpo_num_generations` | `8` | GRPO candidates sampled per prompt |
| `--grpo_max_completion_length` | `512` | GRPO max new tokens per candidate |
| `--grpo_temperature` | `1.0` | GRPO sampling temperature |

ZeRO stages (`--zero_stage`):

| Stage | Shards | When to use |
|---|---|---|
| `0` | nothing (plain DDP) | small models; **required with `--load_in_4bit`** |
| `1` | optimizer state | mild memory relief |
| `2` | + gradients | **recommended for LoRA** / any model that fits on one GPU |
| `3` | + parameters | models too large for one GPU; adds param-gather overhead |

## Output

Outputs land in `<experiment_root>/<RUN_ID>/<output_subdir>/` (or `--output_dir`
verbatim). `RUN_ID` comes from the environment or a UTC timestamp. The tested runs used
`--experiment_root /mnt/gsma/gsma/gsma/experiments/` and
`--hf_home /mnt/gsma/gsma/gsma/models/` — the new relative defaults (`experiments/`,
`hf_cache/`) keep the same layout under the working directory.

- Per-epoch checkpoints (`save_strategy="epoch"`, capped by `--save_total_limit`).
- `final_model/` — the LoRA adapter (with `--use_lora`) or consolidated 16-bit weights
  (full FT). A LoRA adapter needs the base model + `PeftModel.from_pretrained` at
  inference.
- `train_script_backup.py` — a copy of the training script for provenance.
- TensorBoard logs (`report_to="tensorboard"`).
- On failure, every rank writes `train_err_rank<N>.log` with its full traceback.

## Hardware support & evidence

| | NVIDIA | AMD |
|---|---|---|
| Status here | **Tested (SFT, single GPU)** — 1×H100 80GB, CUDA 13.0 (see "H100 (CUDA 13.0) — tested" below); multi-GPU deferred | **Tested** — 2×MI355X **and** all 8×MI355X 288GB (gfx950), ROCm 7.2.4 (see section below) |
| PyTorch | PyPI pins in requirements (`torch 2.11.0+cu130` — the pin resolves natively on CUDA 13, no `--index-url`) | `download.pytorch.org/whl/rocm7.2` (satisfies the pins exactly) |
| DeepSpeed ops | pure-python install; JIT via `nvcc` only if an op is requested (**none needed** for ZeRO-2 + `adamw_torch` — confirmed on H100) | pure-python install; JIT via `hipcc` only if an op is requested (none needed for ZeRO-2 + `adamw_torch`) |

Verified upstream sources (fetched 2026-08-19):

- **DeepSpeed README** (github.com/deepspeedai/DeepSpeed) — lists a ROCm compiler
  (`hipcc`) alongside `nvcc` as a supported requirement; names AMD MI100 and MI200 among
  the GPUs it develops and tests against; runs a dedicated `amd-mi200` CI pipeline; and
  its 2026/05 news item covers SDMA offload for ZeRO-3 collectives specifically on AMD
  GPUs.
- **AMD ROCm AI-ecosystem docs** (rocm.docs.amd.com, "Install PyTorch for ROCm 7.14.0")
  — documents PyTorch 2.10–2.12 for ROCm on Python 3.11–3.14 via `rocm/pytorch` Docker
  images and pip.
- **PyTorch ROCm wheel index** (download.pytorch.org/whl/rocm7.0) — carries
  `torch-2.10.0+rocm7.0` cp312 manylinux wheels.
- **flash-attention** (github.com/ROCm/flash-attention README) — ROCm ≥ 6.0 with
  composable_kernel and Triton backends; CK backend supports MI200x, MI250x, MI300x,
  MI355x, fp16/bf16, head dims up to 256. Not used by this trainer (SDPA only).
- **bitsandbytes installation docs** (github.com/bitsandbytes-foundation/bitsandbytes)
  — official AMD ROCm support; PyPI wheels built for ROCm 6.4.4–7.14 covering Instinct
  gfx90a/gfx942/gfx950; ROCm needs no special install (`pip install bitsandbytes`).

**Other hardware (upstream claims — not verified here):** Intel Gaudi/HPU (upstream CI),
Intel XPU (upstream CI), Intel Xeon CPU (upstream CI), Huawei Ascend NPU (contributor),
Tecorigin SDAA (contributor) — per the DeepSpeed README's "Contributed HW support"
accelerator table (checked August 2026). This repo provides setup instructions for
NVIDIA and AMD only.

## MI355X (ROCm 7.2) — tested

Verified 2026-08-19 on 2×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
Python 3.12.3, single node. All three algorithms were smoke-tested with LoRA + ZeRO-2,
bf16, `--flash_attention sdpa`, per-device batch 1, 1 epoch over the shipped 10-row
sample (`accelerate launch --use_deepspeed`, 2 processes).

**Install (exactly what worked):**

```bash
python3 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_deepspeed.txt   # minus the torch/torchvision lines
# DPO/GRPO only (platform-independent; see Notes):
pip install mergekit && pip install accelerate==1.14.0   # restore the pin mergekit downgrades
# + llm_blender/weave import stubs in site-packages (see Notes)
```

Resolved set: `torch 2.11.0+rocm7.2`, `torchvision 0.26.0+rocm7.2`, all other pins
unchanged (`deepspeed 0.19.4`, `transformers 5.5.0`, `trl 0.24.0`, `peft 0.20.0`,
`accelerate 1.14.0`). DeepSpeed installed as a pure-python wheel — no hipcc compile at
install and no JIT build triggered during any run.

> **Venv note:** transcripts below reference the campaign venv
> (`.env_train_llm_deepspeed`) verbatim. The campaign venvs were removed during the
> 2026-08 reorg — rebuild from `requirements_deepspeed.txt` (new convention:
> `.env_deepspeed`).

**Launch (2 GPUs; pass a non-default port when the box is shared):**

```bash
accelerate launch --num_processes=2 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29620 \
  train_llm_deepspeed.py \
  --train_mode sft --train_file data/OTel_LLM_sample_10.jsonl \
  --model_name google/gemma-4-E4B-it \
  --flash_attention sdpa --gradient_checkpointing --mask_prompt \
  --use_lora --zero_stage 2 \
  --eval_samples 2 --num_train_epochs 1 --batch_size 1 --grad_acc_steps 1
```

For DPO swap in `--train_mode dpo --pref_file <pairs.jsonl> --model_name
google/gemma-3-1b-it`; for GRPO `--train_mode grpo --grpo_num_generations 2` (the
global generation batch — `num_processes × batch_size` — must be divisible by
`--grpo_num_generations`).

**Per-algorithm verdicts:**

| Mode | Model | Verdict | Evidence (from the run logs) |
|---|---|---|---|
| SFT (LoRA, ZeRO-2) | `google/gemma-4-E4B-it` | **Works, unmodified** | `{'loss': '10.48', 'grad_norm': '37.87', ...}` → `{'loss': '6.555', 'grad_norm': '8.469', ...}` over 4/4 steps; `Saving final LoRA adapter`; `Training Complete.` |
| DPO (LoRA, ZeRO-2) | `google/gemma-3-1b-it` | **Works with changes** (import fixes + `warnings_issued` shim; gemma-4 blocked — see Notes) | `{'loss': '0.6914', ...}` → `{'loss': '0.05249', 'rewards/margins': '2.172', 'rewards/accuracies': '1', ...}` over 5/5 steps; `Training Complete.` |
| GRPO (LoRA, ZeRO-2) | `google/gemma-3-1b-it` | **Works with changes** (same fixes; placeholder reward ⇒ constant reward ⇒ zero advantage, `loss 0` by design) | `GRPO prompts: 10 | num_gen=2 max_completion=32`; 10/10 generate+train steps, `reward: 1`, `frac_reward_zero_std: 1`; `Training Complete.` |

DeepSpeed engine confirmed active on both runs: per-rank ZeRO-2 shards
(`bf16_zero_pp_rank_{0,1}_mp_rank_00_optim_states.pt`) in every checkpoint, both GPUs
busy in `rocm-smi` mid-run, and accelerate logging:

```
ROCm + DeepSpeed + bf16 detected: setting `communication_data_type='fp32'` to avoid bf16 overflow corrupting weights.
```

**ROCm-specific quirks (all minor):**

- DeepSpeed on ROCm is JIT-only and nothing JIT-compiled in these runs; the script's
  `CUDA_HOME=/usr/local/cuda-13` default points at a nonexistent path on this box and
  is harmless.
- `--flash_attention sdpa` is mandatory here as on NVIDIA; do **not** pip-install
  flash-attn on ROCm (the CUDA wheel cannot build; a ROCm build exists upstream but is
  unused by this trainer).
- This script sets no `tf32` flag (which would raise on ROCm) and its chat-template
  tokenization uses `tokenize=False`, so neither of those transformers-5.x sharp edges
  applies.
- Default master port 29500 collides with parallel jobs — pass
  `--main_process_port` (29620 used in the tested runs).

**Platform-independent issues found (would also occur on CUDA with these pins):**

1. **DPO/GRPO imports fail out of the box** — trl 0.24.0 probes optional deps via
   transformers' private `_is_package_available`, which returns a `(bool, version)`
   tuple in transformers 5.5.0; a tuple is always truthy, so trl unconditionally
   imports `mergekit`, `llm_blender`, and `weave`. Fix used: install real `mergekit`
   (0.1.4; then force `accelerate==1.14.0` back), and drop two tiny stub packages for
   `llm_blender` and `weave` into site-packages (the real ones are incompatible with
   transformers 5.5.0; TRL only touches them in `PairRMJudge`/callbacks this trainer
   never uses).
2. **`warnings_issued` shim** — transformers 5.x removed
   `PreTrainedModel.warnings_issued`, but trl 0.24.0's `DPOTrainer`/`GRPOTrainer`
   write to it in `__init__`. `train_llm_deepspeed.py` now sets an empty dict when the
   attribute is missing.
3. **DPO cannot run gemma-4 under these pins** — trl 0.24.0 classifies `gemma4`
   (image-text-to-text `model_type`) as a vision model and routes DPO rows through
   `process_row`, which requires an `AutoProcessor` and an `images` column
   (`AttributeError: GemmaTokenizer has no attribute tokenizer`). Text-only pref data
   therefore needs a text-only model (verified with `google/gemma-3-1b-it`). SFT with
   gemma-4 is unaffected.

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

Verified 2026-08-19 on **all 8** MI355X (gfx950, 288GB each), ROCm 7.2.4, single node,
same venv and same pins as the 2-GPU run above. **All three tiers passed at world size 8
with zero code and zero requirements changes.**

| Tier | Model | Verdict | Steps | Final loss | Wall time |
|---|---|---|---|---|---|
| SFT (LoRA, ZeRO-2) | `google/gemma-4-E4B-it` | **WORKS** — unmodified | 32 | `10.58 → 0.02386` (eval 0.01136) | 28.31 s |
| DPO (LoRA, ZeRO-2) | `google/gemma-3-1b-it` | **WORKS** — unmodified at 8 ranks (still needs the trl import fixes from the section above) | 32 | `0.6914 → 1.188e-24`, margins `60`, accuracies `1` | 18.63 s |
| GRPO (LoRA, ZeRO-2) | `google/gemma-3-1b-it` | **WORKS** — unmodified at 8 ranks; generation loop scales cleanly | 20 | `0` **by design** (placeholder reward ⇒ `reward: 1`, `reward_std: 0`) | 35.80 s |

**Launch (exactly what ran — all three tiers back to back in one job):**

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # see the CUDA_VISIBLE_DEVICES warning below
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
COMMON="--flash_attention sdpa --gradient_checkpointing --use_lora --zero_stage 2 \
 --logging_steps 2 --eval_samples 8 --num_train_epochs 1 \
 --batch_size 1 --grad_acc_steps 1 --save_total_limit 1 --num_proc 8"

# SFT
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29650 train_llm_deepspeed.py \
  --train_mode sft --model_name google/gemma-4-E4B-it --mask_prompt \
  --train_file <264-row messages jsonl> --output_dir <out>/sft $COMMON

# DPO (same launcher; text-only model, see issue 3 above)
  --train_mode dpo --model_name google/gemma-3-1b-it \
  --pref_file <264-row pref jsonl> --dpo_max_length 2048

# GRPO (same launcher)
  --train_mode grpo --model_name google/gemma-3-1b-it \
  --grpo_num_generations 2 --grpo_max_completion_length 32 --max_samples 80
```

**Parallelism actually used:** DeepSpeed **ZeRO stage 2**, world size **8** (1 rank per
GPU, pure data parallel — no TP/PP), bf16, LoRA r=64 α=128 `all-linear`, per-device batch
1 × grad-acc 1 × 8 ranks = **global batch 8**; 256 train rows ⇒ 32 optimizer steps.
GRPO used `num_generations=2` because the global generation batch
(`num_processes × batch_size` = 8) must be divisible by it.

**Evidence the run really used 8 GPUs** (`rocm-smi` sampled *inside* the job, not after):

```
[assert] torch 2.11.0+rocm7.2 | hip 7.2.26015 | device_count=8
[assert] OK device_count == 8
[rank=0] INFO: Model weights loaded on all 8 ranks.
[rank=0] INFO: Mode=sft | DeepSpeed: ZeRO stage=2, optimizer_offload=none
{'loss': '10.58', 'grad_norm': '73.07', 'mean_token_accuracy': '0.06932', 'epoch': '0.0625'}
{'loss': '0.02386', 'grad_norm': '0.4118', 'mean_token_accuracy': '0.9989', 'epoch': '1'}
{'train_runtime': '28.31', 'train_samples_per_second': '9.044', 'train_steps_per_second': '1.13', 'train_loss': '1.985'}
```

```
########## 17:08:43 UTC tier=sft
GPU[0..7]: GPU use (%): 100        # all eight, every sample taken during training
---- rocm-smi --showpids ----      # 8 distinct KFD python3 PIDs, one GPU each
3502667..3502674  python3  GPU(s) 1  VRAM ~4.3-4.8 GB (early step)
3501680           pt_elastic
```

PID cross-check: the `pt_elastic` agent (pid `3501680`) carried the runner's own process
group id, and the eight KFD PIDs holding VRAM (`3502667`–`3502674`) were exactly its eight
children — i.e. the VRAM belonged to *this* job, not to another tenant of the box. 32 of 32
in-training samples showed 8 busy GPUs and exactly 8 KFD PIDs.

**ZeRO-2 confirmed active at 8 ranks** (not just the flag): every checkpoint contained one
optimizer shard *per rank* —

```
sft/checkpoint-32/global_step32/bf16_zero_pp_rank_{0,1,2,3,4,5,6,7}_mp_rank_00_optim_states.pt
```

8 shards × 3 tiers = 24, versus 2 shards on the 2-GPU run.

**Per-GPU VRAM: 8 ranks vs 2 ranks (matched baseline, identical per-device geometry)**

| | 2×MI355X | 8×MI355X | Δ |
|---|---|---|---|
| Peak per-GPU VRAM (SFT, gemma-4-E4B) | 43.98 GiB | 49.13 GiB | **+5.2 GiB (≈ +12%)** |
| `train_samples_per_second` | 2.148 | 9.044 | **4.21× on 4× the GPUs** |
| `train_steps_per_second` | 1.074 | 1.13 | flat (as expected — fixed per-device batch) |

> **ZeRO-2 does *not* reduce per-GPU VRAM in this LoRA recipe, and that is correct
> behaviour, not a bug.** ZeRO-2 shards optimizer state and gradients only — with LoRA the
> trainable state is ~0.4 GB, so there is almost nothing to shard. Per-GPU memory is
> dominated by the frozen bf16 base weights, which ZeRO-2 *replicates* on every rank, plus
> activations, which are driven by the per-device batch and therefore unchanged. The small
> *increase* at 8 ranks is RCCL communication buffers growing with world size. If you need
> per-GPU memory to fall as you add GPUs, you need `--zero_stage 3` (shards parameters) or
> full fine-tuning, where the optimizer state is large enough for stage 2 to matter.
> Throughput, by contrast, scales near-linearly (4.21× on 4× GPUs).

Peak per-GPU VRAM for the other tiers at 8 ranks: DPO **19.54 GiB**, GRPO **12.25 GiB**
(both `gemma-3-1b-it`).

**What differed from the 1-2 GPU run:**

- **Nothing in the code, requirements, or DeepSpeed config.** Same venv, same pins, same
  `--zero_stage 2`; only `--num_processes` and `--main_process_port` changed. DeepSpeed
  still JIT-compiled nothing (no hipcc invocation, no slow first step).
- ⚠️ **`.env_train_llm_deepspeed/bin/activate` ends with a stale
  `export CUDA_VISIBLE_DEVICES=0,1` from the 2-GPU session.** Sourcing the venv silently
  pins you to 2 GPUs, so an "8-GPU" launch would quietly run on 2. **Always re-export both
  `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after activating**, and assert
  `torch.cuda.device_count() == 8` before training.
- **The shipped 10-row sample is too small for 8 ranks.** The eval split is floored at
  `world_size`, so 10 rows leaves 8 eval / 2 train. The tested run used a 264-row file
  (the shipped sample tiled) to get 256 train rows ⇒ 32 real steps. Use ≥ `world_size × 32`
  rows for a meaningful 8-GPU smoke test.
- **DPO/GRPO needed no additional multi-GPU fixes** — the trl↔transformers skew fixes
  documented above are sufficient at 8 ranks. GRPO's generation-heavy loop, the most likely
  place for multi-GPU trouble, ran clean: 20/20 generate+train steps, no hang, no rank
  divergence, exit 0.
- Port `29650` used (29500 collides on a shared box).
- All three tiers exited **rc=0** with no teardown hang and no NCCL/RCCL timeout.

**Reproduction notes:** run on a machine-wide lock if the box is shared, disable/prune
checkpoints for smoke runs (this trainer hard-codes `save_strategy="epoch"`, so a 1-epoch
run writes one checkpoint plus `final_model` — ~4.9 GB for the gemma-4-E4B LoRA, ~1.6 GB
for gemma-3-1b), and delete the weights afterwards.

### 4-GPU sharding run — **ZeRO-3 full fine-tune** (4×MI355X, ROCm 7.2.4) — tested August 2026

> Closes the gap left open by the two sections above: both of them ran **ZeRO-2 with
> LoRA**, where there is almost nothing to shard. This run is the opposite corner —
> **`--zero_stage 3` with no `--use_lora`**, i.e. real parameter sharding of all 8B
> weights plus the `stage3_gather_16bit_weights_on_model_save` collective all-gather on
> save. Nothing below contradicts the ZeRO-2 results; it extends them.

Verified 2026-08-19 on physical GPUs **4,5,6,7** of the same 8×MI355X node (a sibling job
owned 0-3), same venv and same pins. **Verdict: ZeRO-3 holds at 4 GPUs — no code, config,
or requirements changes. Three runs, all rc=0, no hang, no all-gather deadlock.**

```bash
source .env_train_llm_deepspeed/bin/activate
export HIP_VISIBLE_DEVICES=4,5,6,7      # MUST re-export: activate pins 0,1 (see warning above)
export CUDA_VISIBLE_DEVICES=4,5,6,7     # renumber to 0-3 inside the process
export HF_HOME=/mnt/data_1.5t/hf_cache

accelerate launch --num_processes=4 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29792 train_llm_deepspeed.py \
  --train_mode sft --model_name google/gemma-4-E4B-it --mask_prompt \
  --train_file <200-row messages jsonl: the shipped 10-row sample tiled ×20> \
  --output_dir /mnt/data_1.5t/outputs/train_llm_deepspeed_4gpu/zero3_sft \
  --flash_attention sdpa --gradient_checkpointing --zero_stage 3 \
  --logging_steps 1 --eval_samples 4 --num_train_epochs 1 \
  --batch_size 1 --grad_acc_steps 1 --save_total_limit 1 --num_proc 4 \
  --learning_rate 1e-5
```

Note there is **no `--use_lora`** — this is a full fine-tune of all 8B parameters, which
is what makes stage 3 meaningful.

**Evidence (from the run logs):**

```
[rank=0] INFO: Mode=sft | DeepSpeed: ZeRO stage=3, optimizer_offload=none
[rank=0] INFO: ROCm + DeepSpeed + bf16 detected: setting `communication_data_type='fp32'` ...
{'loss': '10.6',      'grad_norm': '349.6', 'mean_token_accuracy': '0.08021', 'epoch': '0.1111'}
{'loss': '6.697',     'grad_norm': '84.55', 'mean_token_accuracy': '0.2194',  'epoch': '0.3333'}
{'loss': '0.001053',  'grad_norm': '0.09613','mean_token_accuracy': '1',      'epoch': '1'}
{'train_runtime': '139.6', 'train_samples_per_second': '1.404', 'train_steps_per_second': '0.351', 'train_loss': '1.649'}
[rank=0] INFO: Saving final model to .../zero3_sft/final_model
Writing model shards: 100%|██████████| 1/1 [00:07<00:00,  7.86s/it]
Model weights saved in .../final_model/model.safetensors
[rank=0] INFO: Training Complete.        # rc=0
```

`rocm-smi` sampled every 5 s *during* training (all four owned GPUs, mid-run sample):

```
=== 20:34:41 ===
GPU[4]: GPU use (%): 78     VRAM Total Used Memory (B): 80206942208   # 74.7 GiB
GPU[5]: GPU use (%): 73     VRAM Total Used Memory (B): 85438681088   # 79.6 GiB
GPU[6]: GPU use (%): 72     VRAM Total Used Memory (B): 80206180352   # 74.7 GiB
GPU[7]: GPU use (%): 77     VRAM Total Used Memory (B): 83828494336   # 78.1 GiB
```

**What this proves that the ZeRO-2 runs did not:**

- **The ZeRO-3 all-gather save works on ROCm and does not deadlock.** `save_model` on all
  four ranks reconstructed the sharded 8B weights into one consolidated
  `model.safetensors` of **15,992,595,884 bytes (~14.9 GiB)**, directly loadable — in
  ~8 s. This is the exact pattern that deadlocked in `../deepspeed_standalone/`
  when it was guarded with `if rank == 0`; this trainer's unguarded call is correct and
  is now verified at world size 4. Do not "optimise" it back behind a rank check.
- **Parameter sharding is really happening.** Per-GPU VRAM is ~75-80 GiB for a full 8B
  fine-tune (params + fp32 master + Adam moments ≈ 112 GB of state, sharded 4 ways),
  versus the ~49 GiB the *LoRA* ZeRO-2 run needed — i.e. stage 3 is absorbing an
  optimizer state that stage 2 with LoRA never had. Confirms the prediction made in the
  8-GPU section ("if you need per-GPU memory to fall as you add GPUs, you need
  `--zero_stage 3` or full fine-tuning").
- **VRAM across ranks is even to within ~5 GiB** (74.7 / 79.6 / 74.7 / 78.1 GiB) and the
  spread was stable across all 45 samples — a fixed asymmetry in the partitioning, not a
  leak. GPU utilisation held **68-78 % on all four** for the whole training phase.
- Loss converges normally under stage 3 (10.6 → 0.001 over 49 steps, token accuracy → 1),
  so `communication_data_type='fp32'` is doing its job — no bf16 collective overflow.

**Caveats / unchanged limitations at 4 GPUs:**

- `--custom_eval` remains **unusable under ZeRO-3** (documented in Notes below); it was
  not enabled in this run and the guard correctly skipped it.
- Ports 29790-29792 used; 29500 still collides on a shared box.
- Disk: each 1-epoch ZeRO-3 run writes `checkpoint-N` **plus** a ~15 GiB `final_model`.
  Budget ~35 GB per run and delete immediately after verifying.

## H100 (CUDA 13.0) — tested

Verified 2026-08-22 on 1×NVIDIA H100 80GB HBM3 (Hopper cc 9.0, native FP8), driver
**580.173.02**, **CUDA 13.0** toolkit at `/usr/local/cuda-13.0`, Ubuntu, Python 3.12.3,
single node. SFT was smoke-tested with LoRA + ZeRO-2, bf16, `--flash_attention sdpa`,
per-device batch 1, 4 epochs over the shipped 10-row sample (8 train / 2 eval) via
`accelerate launch --use_deepspeed`, 1 process on a shared box (GPUs 0–3 were running a
sibling job — this run was pinned to physical **GPU 4** with `CUDA_VISIBLE_DEVICES=4` and
a non-default `--main_process_port 29641`).

**Install (exactly what worked):**

```bash
python3 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 numpy   # plain PyPI = the cu130 build
pip install -r requirements_deepspeed.txt             # minus the torch/torchvision lines
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # re-check: NOT clobbered
```

**The requirements pin resolves natively on CUDA 13 — no `--index-url` and no deviation.**
Plain PyPI `torch==2.11.0 torchvision==0.26.0` installs `torch 2.11.0+cu130` /
`torchvision 0.26.0+cu130` with the `nvidia-*-cu13` runtime deps (`nccl-cu13 2.28.9`,
`cudnn-cu13 9.19`, `nvjitlink 13.0.88`). The base `pip install torch numpy` recipe (which
the box was validated with) yields the same cu130 line — the exact pin is a strict subset,
so this trainer needs no fallback to a newer torch. Resolved set: `torch 2.11.0+cu130`,
`torchvision 0.26.0+cu130`, all other pins unchanged (`deepspeed 0.19.4`,
`transformers 5.5.0`, `trl 0.24.0`, `peft 0.20.0`, `accelerate 1.14.0`,
`datasets 4.3.0`). Installing the requirements did **not** clobber torch — re-checked
`torch 2.11.0+cu130` after the second `pip install`. A real bf16 matmul and a sustained
bf16 GEMM loop both ran clean on GPU 4 (cc `(9, 0)`, `torch.float8_e4m3fn` present).

DeepSpeed installed as a normal wheel and **JIT-compiled nothing** in this run (no
`ninja`/`nvcc`/"Building extension" — ZeRO-2 with `adamw_torch` is DeepSpeed-native), so
the toolkit-path export from the NVIDIA install block above was never exercised for a
build. The script's `CUDA_HOME` default is the string `/usr/local/cuda-13`; the box's
toolkit is at `/usr/local/cuda-13.0` (with a `/usr/local/cuda` → 13.0 symlink) — harmless
here because no op JIT-built. If you enable an op that JIT-compiles (e.g. a fused/CPU-Adam
path), export `CUDA_HOME=/usr/local/cuda-13.0` first.

**Launch (single GPU on a shared box; pin your GPU and pass a non-default port):**

```bash
export CUDA_VISIBLE_DEVICES=4                       # your free GPU only
export HF_HOME=/path/to/model/cache                 # reuse a pre-cached gemma-4-E4B-it
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache  # writable fs for the arrow cache (see quirks)

accelerate launch --num_processes=1 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29641 \
  train_llm_deepspeed.py \
  --train_mode sft --train_file data/OTel_LLM_sample_10.jsonl \
  --model_name google/gemma-4-E4B-it \
  --hf_home /path/to/model/cache \
  --flash_attention sdpa --gradient_checkpointing --mask_prompt \
  --use_lora --zero_stage 2 \
  --eval_samples 2 --num_train_epochs 4 --batch_size 1 --grad_acc_steps 1 \
  --logging_steps 1 --output_dir /dev/shm/out/deepspeed/sft
```

`--num_train_epochs 4` (not 1) is deliberate: 8 train rows × 4 epochs at global batch 1
= **32 real optimizer steps**, enough to see the loss curve. One epoch would give only 8.

**Result — SFT (LoRA, ZeRO-2), `google/gemma-4-E4B-it`: WORKS, unmodified.** rc=0,
`Training Complete.`, LoRA adapter saved. 32/32 steps ran; loss finite and decreasing;
memorization visible (`mean_token_accuracy` → 1.0 on several late steps, held-out
`eval_loss` bottoming at epoch 2 then ticking up as the tiny set is memorized — exactly
the behaviour the Overview predicts with `--load_best_model_at_end` off).

**Real log lines (from the run, not paraphrased):**

```
[rank=0] INFO: Model weights loaded on all 1 ranks.
[rank=0] INFO: Mode=sft | DeepSpeed: ZeRO stage=2, optimizer_offload=none
[rank=0] INFO: train supervision over 8 sampled rows: ... | trained min=11 mean=183 max=455
{'loss': '11.02',   'grad_norm': '75.18', 'mean_token_accuracy': '0.05502', 'epoch': '0.125'}
{'loss': '5.567',   'grad_norm': '8.194', 'mean_token_accuracy': '0.244',   'epoch': '1'}
{'loss': '3.946',   'grad_norm': '8.878', 'mean_token_accuracy': '0.3404',  'epoch': '2'}
{'loss': '0.08252', 'grad_norm': '2.108', 'mean_token_accuracy': '1',       'epoch': '3.625'}
{'eval_loss': '4.316', ... 'epoch': '1'}   →   {'eval_loss': '3.952', ... 'epoch': '2'}
{'train_runtime': '87.36', 'train_samples_per_second': '0.366', 'train_steps_per_second': '0.366', 'train_loss': '3.59', 'epoch': '4'}
[rank=0] INFO: Saving final LoRA adapter to .../sft/final_model
[rank=0] INFO: Training Complete.        # rc=0
```

**GPU residency proof — this run owned only GPU 4** (`nvidia-smi -i 4` VRAM-by-PID,
sampled *inside* the job; the launcher's own venv python held the VRAM):

```
-- compute-apps on phys GPU4 --   (nvidia-smi --query-compute-apps ... -i 4, mid-run)
GPU-e13d18b6-...  1446849  .../.env_deepspeed/bin/python   6726 MiB   # rising to ~15.8 GiB peak
```

Peak per-GPU VRAM was **~15.8 GiB** (gemma-4-E4B LoRA, bf16, seq ≤ ~2.3k, gradient
checkpointing) — well within the H100's 80 GB, no OOM, no batch/seq reduction needed. A
GPU-utilisation cross-check on GPU 4 pegged **100 %** under a sustained bf16 GEMM loop
(112/154 high-frequency samples at 100 %); during the training itself the coarse 3 s
sampler mostly landed in the idle gaps between the fast ~1 s steps, so the VRAM-by-PID
above is the authoritative residency evidence.

**ZeRO-2 confirmed active** (not just the flag) — the fit-end checkpoint carried the
DeepSpeed optimizer shard for the single rank:

```
sft/checkpoint-32/global_step32/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
sft/checkpoint-32/global_step32/mp_rank_00_model_states.pt
```

(1 rank ⇒ 1 shard; the tested 2×/8× MI355X runs show ranks `0..N` here — the same file
pattern, one per rank.)

**CUDA-specific quirks / what differed from the MI355X recipe:**

- **PyTorch source flips back to plain PyPI** — no `--index-url https://.../rocm7.2`; the
  cu130 wheel is the PyPI default. The exact requirements pin (`torch==2.11.0
  torchvision==0.26.0`) works as-is, so unlike the general H100 note about falling back to
  `torch 2.13.0`, **no torch deviation was needed for this folder.**
- **No `communication_data_type='fp32'` line** — that ROCm log line (a bf16-collective
  overflow guard) does **not** appear on CUDA; H100 bf16 collectives are used directly and
  loss stayed finite. This is expected, not a regression.
- **tf32:** despite the general expectation that a tf32 guard "auto-enables on CUDA," **this
  trainer sets no tf32 flag at all** (grep confirms — same finding the MI355X section
  records). There is nothing to verify engaging; bf16 mixed precision does the compute. If
  you want TF32 on the fp32 residuals, set `torch.set_float32_matmul_precision("high")`
  yourself.
- **FlashAttention-2 stays off, by design, even though flash-attn exists on H100** — the
  script hard-fails `--flash_attention flash_attention_2` for Gemma-4 because its attention
  head_dim exceeds FA2's 256 limit (see Notes). This is a *model* constraint, not a
  platform one, so the H100 recipe keeps `--flash_attention sdpa` exactly like ROCm; do not
  bother installing flash-attn for this trainer + gemma-4.
- **`--flash_attention sdpa`, `HIP_*` dropped** — plain `CUDA_VISIBLE_DEVICES` only; no
  `HIP_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` needed.
- **Datasets arrow-cache location matters on some mounts** — on the tested box the shared
  model-cache filesystem rejected `datasets` arrow-cache writes with `PermissionError:
  Operation not permitted` (the `num_proc` tokenization map hard-crashes the run at
  `cache-*.arrow`). Fix: point `HF_DATASETS_CACHE` at a writable fs (tmpfs here) while
  leaving `--hf_home` on the pre-cached model store. This is an environment/mount quirk, not
  a CUDA issue, but it *will* stop the run cold if the model cache is read-only-ish.
- Default master port 29500 collides on a shared box — `29641` used here.

**Multi-GPU (2, then 8) is deferred** (GPUs 0–3 were busy with a production job during this
wave). No code, requirements, or DeepSpeed-config change is expected to scale up — only
`--num_processes` and `--main_process_port`. A meaningful N-GPU smoke needs a larger file:
the eval split is floored at `world_size`, so the 10-row sample leaves too few train rows
past ~4 ranks; tile the sample to ≥ `world_size × 32` rows (see the MI355X 8-GPU section).
On 80 GB cards, LoRA + ZeRO-2 replicates the frozen base per rank (per-GPU VRAM ~flat as
you add GPUs — that is correct ZeRO-2 behaviour, see the MI355X 8-GPU note); use
`--zero_stage 3` or full FT if you need per-GPU memory to fall.

**DPO/GRPO on H100 not run this wave** (SFT was prioritised). They need the same
platform-independent trl↔transformers 5.5.0 import fixes the MI355X "Platform-independent
issues" section documents (mergekit install + `llm_blender`/`weave` stubs +
`warnings_issued` shim; DPO also needs a text-only model, not gemma-4). Those fixes are
hardware-neutral and would apply identically on CUDA.

**Verdict: SFT (LoRA, ZeRO-2) WORKS on H100 / CUDA 13.0, unmodified — the requirements pin
resolves to a native cu130 build with no deviation.** The only adjustment the *box* forced
was redirecting `HF_DATASETS_CACHE` off a read-only-ish shared mount; nothing in the trainer
changed. Multi-GPU and DPO/GRPO are deferred, not blocked.

### 2-GPU run (2× H100, CUDA 13.0) — tested August 2026

> Supersedes the "Multi-GPU is deferred" note in the single-GPU section above **for the
> 2-GPU case**: real 2-rank DeepSpeed ZeRO-2 sharding is now measured on H100, not
> extrapolated. Same venv, same pins, same code and DeepSpeed config as the single-GPU run
> — only `--num_processes`, `--main_process_port`, and the GPU pinning changed. The 8-GPU
> tier remains extrapolated-not-measured (see the note at the end of this subsection).

Verified 2026-08-22 on **physical GPUs 4 and 7** of the same shared 8×H100 80GB node
(driver **580.173.02**, CUDA 13.0, Hopper cc 9.0), single node, Python 3.12.3. GPUs 0–3
were running a co-tenant production job (PIDs 1273508–1273511, one rank per GPU) throughout
— this run never touched them (verified by sampling `nvidia-smi -i 0,1,2,3` before and
during the run; no foreign PID ever appeared on 0–3). SFT smoke test: LoRA + ZeRO-2, bf16,
`--flash_attention sdpa`, per-device batch 1, **4 epochs** over the shipped ~10-row sample
(8 train / 2 eval), `accelerate launch --use_deepspeed`, **2 processes**, port `29670`.

**Model:** `LiquidAI/LFM2.5-350M` (fully cached under `HF_HOME`; the offline-safe default
for this box — `gemma-4-E4B-it`, used in the single-GPU section above, has no cached
tokenizer when `HF_HUB_OFFLINE=1`). Its tokenizer ships a ChatML (`<|im_start|>`) chat
template, so the trainer's chat-template masking works unmodified.

**The DeepSpeed device-0 trap (why the launch looks like this):** the bare
`deepspeed --num_gpus N` / `--include localhost` launcher can ignore
`CUDA_VISIBLE_DEVICES` and grab GPU 0 — which here is the co-tenant's production job. Two
safe patterns avoid it: (a) `accelerate launch` with `CUDA_VISIBLE_DEVICES` exported
(accelerate honors it — used here; `torch.cuda.device_count()` reported **2**, confirming
only GPUs 4,7 were visible), or (b) `unset CUDA_VISIBLE_DEVICES` then
`deepspeed --include localhost:4,7 --master_port 29670` (explicit device list).

**Launch (exactly what ran — accelerate pattern, pins to GPUs 4 and 7 only):**

```bash
source .env_deepspeed/bin/activate
export CUDA_VISIBLE_DEVICES=4,7                       # accelerate honors this — ONLY 4 and 7
unset HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES   # ROCm leftovers, unused on CUDA
export HF_HOME=/mnt/gsma/gsma/gsma/models             # 1.1 TB pre-cached model store
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # offline: use only cached weights
export HF_DATASETS_CACHE=/dev/shm/h100/dscache_deepspeed2   # writable tmpfs (see arrow-cache quirk)

accelerate launch --num_processes=2 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29670 \
  train_llm_deepspeed.py \
  --train_mode sft --train_file data/OTel_LLM_sample_10.jsonl \
  --model_name LiquidAI/LFM2.5-350M \
  --hf_home /mnt/gsma/gsma/gsma/models \
  --flash_attention sdpa --gradient_checkpointing --mask_prompt \
  --use_lora --zero_stage 2 \
  --eval_samples 2 --num_train_epochs 4 --batch_size 1 --grad_acc_steps 1 \
  --logging_steps 1 --save_total_limit 1 --num_proc 4 \
  --output_dir /dev/shm/h100/out/deepspeed/sft2
```

**Batch geometry / step count:** global batch = per_device(1) × world(2) × grad_accum(1)
= **2**. With 8 train rows (the eval split is floored at `world_size`, so `--eval_samples 2`
leaves 8 train / 2 eval) and `drop_last`, steps/epoch = floor(8/2) = **4**; **4 epochs ⇒ 16
optimizer steps** — deliberately bumped from 1 epoch (which would give only 4) so the loss
curve is visible and non-trivial (≥8 steps).

**Result — SFT (LoRA, ZeRO-2), `LiquidAI/LFM2.5-350M`: WORKS, unmodified.** rc=0
(`TRAIN_EXIT=0`), `Training Complete.`, LoRA adapter saved. 16/16 steps ran; loss finite
and decreasing (`1.288 → 0.0074` by step 15, `train_loss 0.8238`), `mean_token_accuracy`
reaching `1.0` on late steps.

**Real log lines (from the run, not paraphrased):**

```
[rank=0] 2026-08-23 05:55:24 INFO: Model weights loaded on all 2 ranks.
[rank=0] 2026-08-23 05:55:24 INFO: Mode=sft | DeepSpeed: ZeRO stage=2, optimizer_offload=none
[rank=0] 2026-08-23 05:55:26 INFO: Using 2 samples for evaluation
{'loss': '1.288',    'grad_norm': '6.545', 'mean_token_accuracy': '0.6657', 'epoch': '0.25'}
{'loss': '1.871',    'grad_norm': '7.659', 'mean_token_accuracy': '0.5859', 'epoch': '1'}
{'loss': '0.1208',   'grad_norm': '3.852', 'mean_token_accuracy': '0.9672', 'epoch': '2.25'}
{'loss': '0.007425', 'grad_norm': '0.7237','mean_token_accuracy': '1',      'epoch': '3.75'}
{'eval_loss': '1.348', ... 'epoch': '1'}  →  {'eval_loss': '1.375', ... 'epoch': '4'}
{'train_runtime': '9.153', 'train_samples_per_second': '3.496', 'train_steps_per_second': '1.748', 'train_loss': '0.8238', 'epoch': '4'}
[rank=0] 2026-08-23 05:55:44 INFO: Saving final LoRA adapter to .../sft2/final_model
[rank=0] 2026-08-23 05:55:45 INFO: Training Complete.        # rc=0
```

(Held-out `eval_loss` bottoms early then ticks up as the 8-row set is memorized — the
`--load_best_model_at_end` off behaviour the Overview predicts.)

**GPU residency proof — this run owned exactly GPUs 4 and 7, one training PID each**
(`nvidia-smi` sampled *inside* the job every 4 s; UUID→index map confirmed
`GPU-e13d18b6…` = phys **4**, `GPU-9eb34eec…` = phys **7**):

```
== 05:55:37 ==                       # index, mem.used, util  (query-gpu -i 4,7)
4, 5755 MiB, 10 %
7, 5929 MiB, 42 %
-- compute-apps on GPU 4,7 --        # pid, used_memory, gpu_uuid
1726992, 5746 MiB, GPU-e13d18b6-ccfb-6676-668a-cd489ad01b55   # phys GPU 4
1726993, 5920 MiB, GPU-9eb34eec-449b-7839-d362-d1e995e95239   # phys GPU 7
```

Two distinct training PIDs (`1726992` on GPU 4, `1726993` on GPU 7) held VRAM concurrently
for the whole run (rising ~1.5 → ~7.4 GiB peak, then both back to 0 MiB at teardown). Peak
per-GPU VRAM was ~**7.4 GiB** (GPU 4) / ~**5.9 GiB** (GPU 7) — LFM2.5-350M is small; both
well within 80 GB. GPU-utilisation samples read low because the coarse 4 s sampler kept
landing in the idle gaps between the fast (~0.5 s) steps — `train_runtime` was only 9.15 s
for 16 steps — so the VRAM-by-PID above is the authoritative residency evidence (same
caveat as the single-GPU section). **Co-tenant check: `nvidia-smi -i 0,1,2,3` showed only
PIDs 1273508–1273511 for the entire run — the production job on GPUs 0–3 was never touched.**

**ZeRO-2 sharding confirmed active across both ranks** (not just the flag) — the fit-end
checkpoint carried **one optimizer-state shard per rank**:

```
sft2/checkpoint-16/global_step16/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt   # 143,930,693 B
sft2/checkpoint-16/global_step16/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt   # 143,931,461 B
sft2/checkpoint-16/global_step16/mp_rank_00_model_states.pt
```

`rank_0` **and** `rank_1` optimizer shards = genuine 2-way ZeRO-2 partitioning of the
optimizer state — exactly the `rank_{0,1,…,N}` file pattern the MI355X 2-/8-GPU sections
show, here at world size 2 on H100. (The single-GPU H100 run above shows only `rank_0`.)

**What differed from the single-GPU H100 run:** nothing in the code, requirements, or
DeepSpeed config — only `--num_processes=2`, `--main_process_port 29670`, and pinning to
GPUs 4,7. DeepSpeed JIT-compiled nothing (ZeRO-2 + `adamw_torch` is native), and no
`communication_data_type='fp32'` line appeared (that is a ROCm-only bf16-collective guard;
H100 bf16 NCCL collectives were used directly and loss stayed finite). The model was swapped
to `LiquidAI/LFM2.5-350M` (cached, offline-safe) vs the single-GPU section's
`gemma-4-E4B-it`; SFT + ZeRO-2 + LoRA is model-agnostic here, so this is an offline-cache
convenience, not a code change. The `HF_DATASETS_CACHE` redirect off the read-only-ish
model mount was still required (a prior 2-GPU attempt died at `cache-*.arrow` with
`PermissionError: Operation not permitted` — the exact quirk the single-GPU section
documents).

**8-GPU on H100 is extrapolated, NOT measured.** The co-tenant production job held GPUs 0–3
for this entire wave, so a world-size-8 run was not possible without risking someone else's
job. The 2-GPU result plus the MI355X 8-GPU evidence (same code, same pins, ZeRO-2 scales
1→2→8 with only `--num_processes`/port changing) make an H100 8-GPU pass very likely to work
unchanged, but it has not been run here. When GPUs 0–3 free up: tile the sample to ≥
`world_size × 32` rows (the eval split floors at `world_size`, so the ~10-row sample leaves
too few train rows past ~4 ranks — see the MI355X 8-GPU section), and re-export
`CUDA_VISIBLE_DEVICES` for all 8 after activating the venv.

**Verdict: real 2-rank DeepSpeed ZeRO-2 sharding PROVEN on 2× H100 / CUDA 13.0, unmodified**
— 2-process init, per-rank optimizer shards (`rank_0` + `rank_1`), both GPUs 4 and 7 busy by
distinct PID, 16 steps, decreasing loss, saved LoRA adapter. Same code/pins/config as the
single-GPU run; only the launcher's process count, port, and GPU pinning changed. 8-GPU is
extrapolated, not measured (co-tenant held GPUs 0–3).

## Notes

- **FlashAttention-2 is deliberately blocked** — Gemma-4's attention head_dim exceeds
  FA2's 256 limit, so the script fails fast on `--flash_attention flash_attention_2`;
  always pass `--flash_attention sdpa`.
- `--custom_eval` is **disabled under ZeRO-3** — rank-0 generation would hang on the
  sharded-param all-gather. Use `--zero_stage 2` for in-loop custom eval, or evaluate
  the saved adapter offline.
- `save_model` runs on **all** ranks under ZeRO-3 (a collective all-gather reconstructs
  the sharded weights; the file write is rank-0-guarded internally) — never wrap it in
  `if rank == 0`.
- The NCCL process group uses a 2-hour timeout to tolerate the slow model AllGather
  under ZeRO-3; `TORCH_CPP_LOG_LEVEL=ERROR` is set before `import torch` to silence
  non-fatal c10 allocator warnings.
- Optional callbacks (`CustomEvalCallback`, `EmptyCacheCallback`) are imported
  defensively — with an older `utils.py` the corresponding flags become no-ops instead
  of crashing.
