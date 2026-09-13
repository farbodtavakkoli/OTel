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

Behaviour worth knowing before you launch:

- **No `ds_config.json`** — the DeepSpeed config is built in-memory from
  `--zero_stage`/`--offload_optimizer`, so an on-disk config file is ignored.
- **Over-length rows are dropped, never truncated** (`--max_token_length`), and the
  train/eval split floors the eval set at one row per rank.
- **`--load_best_model_at_end` is OFF by default** — for memorization-style training the
  last checkpoint is the one you want, not the lowest-eval-loss one.
- **QLoRA guard** — `--load_in_4bit` requires `--use_lora` and is incompatible with
  ZeRO-3; the script fails fast on that combination.
- **`--dpo_precompute_ref_log_probs`** drops the reference model from the training loop,
  so a larger batch or max_length fits.
- **GRPO reward is a placeholder** — `utils.grpo_reward_funcs` rewards non-empty
  completions; replace it before a real GRPO run.
- **Optimizer names** — `--optim adamw` maps to HF `adamw_torch` (DeepSpeed-native);
  `adamw_bnb_8bit` is available but has a known issue with ZeRO-3.

> **Coverage:** single-node SFT, DPO and GRPO with LoRA + ZeRO-2 on MI355X (gfx950,
> ROCm 7.2.4) and H100 80GB (CUDA 13.0), plus a ZeRO-3 full fine-tune on MI355X. Scaling
> world size needs no code, requirements or DeepSpeed-config change — only
> `--num_processes` and `--main_process_port`. Multi-node is untested.

## Install

Python 3.12. All pins in `requirements_deepspeed.txt` are the tested set.

### NVIDIA (CUDA)

```bash
python3.12 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 numpy   # plain PyPI = the cu130 build
pip install -r requirements_deepspeed.txt             # minus the torch/torchvision lines
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # re-check: NOT clobbered
```

The requirements pins resolve natively on CUDA 13 — no `--index-url` needed. Installing
the requirements afterwards does not clobber torch, but re-check as above.

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

Same order as the NVIDIA install, but torch comes from the ROCm wheel index:

```bash
python3 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_deepspeed.txt   # minus the torch/torchvision lines
```

The differences from the NVIDIA install:

- **PyTorch** — the rocm7.2 index satisfies the pins exactly (`torch 2.11.0+rocm7.2`,
  `torchvision 0.26.0+rocm7.2`); every other pin is unchanged.
- **DeepSpeed** — same `pip install deepspeed==0.19.4`; ops are JIT-only and none are
  needed for ZeRO-2 + `adamw_torch`, so no toolkit export is required.
- **flash-attn** — do not install it; this trainer requires `--flash_attention sdpa`.
- **bitsandbytes** — the regular PyPI wheels include ROCm builds. Needed only for
  `--optim adamw_bnb_8bit` or `--load_in_4bit`.

### Extra step for DPO / GRPO (both platforms)

`--train_mode dpo` and `--train_mode grpo` fail to import out of the box with these pins,
on CUDA and ROCm alike: trl 0.24.0 unconditionally imports `mergekit`, `llm_blender` and
`weave` under transformers 5.5.0. Fix:

```bash
pip install mergekit && pip install accelerate==1.14.0   # restore the pin mergekit downgrades
```

plus two tiny stub packages named `llm_blender` and `weave` dropped into `site-packages`
— the real ones are incompatible with transformers 5.5.0, and TRL only touches them in
code paths this trainer never uses. `train_llm_deepspeed.py` already carries the third fix
(the missing `PreTrainedModel.warnings_issued` attribute).

**DPO needs a text-only model under these pins.** trl 0.24.0 treats `gemma4` as a vision
model and fails with `AttributeError: GemmaTokenizer has no attribute tokenizer`. Use a
text-only model such as `google/gemma-3-1b-it` for preference data; SFT with gemma-4 is
unaffected.

## Environment & secrets

Optional `dev.env` **in this folder**, only needed to pull gated models from the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The script loads it via `load_dotenv("dev.env")` — run the script from inside
`training/llm/deepspeed/` so the relative path resolves. `dev.env` is git-ignored at the
repo root; never commit a token. The script also sets `HF_HOME` (`--hf_home`), NCCL env
vars, and clears proxy vars automatically (a proxy that 403s huggingface.co has been seen
on some clusters).

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
columns (`unmask`, `flow`, `source_id`, ...), which this trainer drops at tokenization
time. If you write your own loader, drop them too — some are NULL in every row, so a
naive `datasets`/pyarrow load infers a `null` dtype and breaks schema-sensitive pipelines.

To swap in real data, point `--train_file` at your own `messages` JSONL (relative or
absolute — any path works). Any row failing the `messages` contract fails the preflight
with the row index and reason.

## Run

Run from inside `training/llm/deepspeed/`. Effective batch =
`batch_size × grad_acc_steps × num GPUs`.

The commands below use two placeholders — set them once to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts (checkpoints, adapters, logs)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

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

**DPO / GRPO variants of the same launch:** for DPO swap in
`--train_mode dpo --pref_file <pairs.jsonl> --model_name google/gemma-3-1b-it`
(text-only model — see Install); for GRPO use
`--train_mode grpo --grpo_num_generations 2`. The global generation batch
(`num_processes × batch_size`) **must be divisible by `--grpo_num_generations`**.

### Scaling up and shared boxes

Scaling from 1 → 2 → 8 ranks needs no change to the code, the requirements, or the
DeepSpeed config — only `--num_processes` and `--main_process_port`.

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD; also export CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # NVIDIA, or the subset you own
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29650 train_llm_deepspeed.py ...
```

- **Always re-export both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after
  activating the venv**, and assert `torch.cuda.device_count() == N` before training. A
  venv `activate` script that ends with a stale `export CUDA_VISIBLE_DEVICES=0,1` will
  silently pin an "8-GPU" launch to 2 GPUs.
- **Pass a non-default `--main_process_port`** (e.g. `29650`); the default 29500 collides
  with any parallel job on the same box.
- **Do not use the bare `deepspeed --num_gpus N` / `--include localhost` launcher on a
  shared box** — it can ignore `CUDA_VISIBLE_DEVICES` and grab GPU 0, which may be
  someone else's job. Either use `accelerate launch` with `CUDA_VISIBLE_DEVICES` exported
  (accelerate honors it), or `unset CUDA_VISIBLE_DEVICES` and pass an explicit device
  list: `deepspeed --include localhost:4,7 --master_port 29670`.
- **Size the dataset to the world size.** The eval split is floored at `world_size`, so
  the shipped 10-row sample leaves too few train rows past ~4 ranks. Use at least
  `world_size × 32` rows (tiling the sample is fine) for a meaningful N-GPU smoke test.
- A ZeRO-3 full fine-tune writes a consolidated `final_model` (~15 GiB for an 8B model)
  in addition to each checkpoint — budget ~35 GB per run and delete after verifying.

## Arguments

Defaults are the tested values.

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
verbatim). `RUN_ID` comes from the environment or a UTC timestamp. `--experiment_root`
and `--hf_home` accept absolute paths (e.g. `$OUTPUT_DIR/experiments/`, `$HF_HOME`); the
relative defaults (`experiments/`, `hf_cache/`) keep the same layout under the working
directory.

- Per-epoch checkpoints (`save_strategy="epoch"`, capped by `--save_total_limit`).
- `final_model/` — the LoRA adapter (with `--use_lora`) or consolidated 16-bit weights
  (full FT). A LoRA adapter needs the base model + `PeftModel.from_pretrained` at
  inference.
- `train_script_backup.py` — a copy of the training script for provenance.
- TensorBoard logs (`report_to="tensorboard"`).
- On failure, every rank writes `train_err_rank<N>.log` with its full traceback.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 and 2 GPUs | MI355X 288GB (gfx950), ROCm 7.2.4, 1/2/4/8 GPUs |
| PyTorch | PyPI pins in requirements (`torch 2.11.0+cu130` — the pin resolves natively on CUDA 13, no `--index-url`) | `download.pytorch.org/whl/rocm7.2` (satisfies the pins exactly) |
| DeepSpeed ops | pure-python install; JIT via `nvcc` only if an op is requested (none needed for ZeRO-2 + `adamw_torch`) | pure-python install; JIT via `hipcc` only if an op is requested (none needed for ZeRO-2 + `adamw_torch`) |

DPO/GRPO need the trl import fixes in
[Install](#extra-step-for-dpo--grpo-both-platforms) on both platforms. Multi-node is
untested.

## Notes

- **FlashAttention-2 is deliberately blocked** — Gemma-4's attention head_dim exceeds
  FA2's 256 limit, so the script fails fast on `--flash_attention flash_attention_2`;
  always pass `--flash_attention sdpa`.
- `--custom_eval` is **disabled under ZeRO-3** — rank-0 generation would hang on the
  sharded-param all-gather. Use `--zero_stage 2` for in-loop custom eval, or evaluate
  the saved adapter offline.
- `save_model` runs on **all** ranks under ZeRO-3 (a collective all-gather reconstructs
  the sharded weights; the file write is rank-0-guarded internally) — never wrap it in
  `if rank == 0`. Guarding it deadlocks.
- **ZeRO-2 does not reduce per-GPU VRAM in a LoRA recipe, and that is correct behaviour.**
  If you need per-GPU memory to *fall* as you add GPUs, use `--zero_stage 3` (shards
  parameters) or full fine-tuning.
- **GRPO's placeholder reward** makes every completion equally good, so `reward: 1`,
  `reward_std: 0` and `loss 0` are correct output for an unmodified GRPO run, not a
  failure.
- **On ROCm the trainer sets `communication_data_type='fp32'`** to avoid bf16 overflow
  corrupting weights; the log line does not appear on CUDA, which is expected.
- **Do not pip-install flash-attn on ROCm** — the CUDA wheel cannot build, and this
  trainer uses SDPA only.
- **No tf32 flag is set on either platform** (a tf32 flag would raise on ROCm). bf16
  mixed precision does the compute; if you want TF32 on the fp32 residuals, call
  `torch.set_float32_matmul_precision("high")` yourself.
- **`PermissionError: Operation not permitted` at `cache-*.arrow`** — the `datasets`
  arrow cache is written next to the model cache, and a shared, read-only-ish model mount
  rejects it, hard-crashing the `num_proc` tokenization map. Point `HF_DATASETS_CACHE` at
  a writable filesystem (e.g. `/dev/shm/hf_datasets_cache`) while leaving `--hf_home` on
  the pre-cached model store.
- For fully offline runs, export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and use a model
  whose tokenizer is already cached.
- The NCCL process group uses a 2-hour timeout to tolerate the slow model AllGather
  under ZeRO-3; `TORCH_CPP_LOG_LEVEL=ERROR` is set before `import torch` to silence
  non-fatal c10 allocator warnings.
- Optional callbacks (`CustomEvalCallback`, `EmptyCacheCallback`) are imported
  defensively — with an older `utils.py` the corresponding flags become no-ops instead
  of crashing.
