# `training/llm/deepspeed` — DeepSpeed post-training (SFT / DPO / GRPO)

Post-training for conversational `messages` models on HF Transformers + DeepSpeed (ZeRO).
SFT, DPO and GRPO, LoRA/QLoRA or full fine-tuning, on any model whose tokenizer ships a
chat template (no chat template = hard fail, to guarantee train/inference parity).

This is the recommended starting point for chat post-training. Use
`../deepspeed_standalone/` instead for a single portable file, hand-written prompt
templates and SFT only.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 1/2/4/8 GPUs · NVIDIA H100 80GB (CUDA 13.0),
1 and 2 GPUs.

## Files

| File | Purpose |
|---|---|
| `train_llm_deepspeed.py` | Trainer entrypoint: arg parsing, model/LoRA setup, SFT/DPO/GRPO branches |
| `utils.py` | Dataset loaders, DPO/GRPO dataset builders, in-memory DeepSpeed config, eval callbacks |
| `requirements_deepspeed.txt` | Pinned environment (Python 3.12) |
| `data/OTel_LLM_sample_10.jsonl` | 10-row `messages` sample; the default `--train_file` |

## Setup

Python 3.12. One venv per recipe folder; all pins live in `requirements_deepspeed.txt`.
Install torch first, then the rest - the requirements install must not re-resolve torch.

### NVIDIA (CUDA 13)

```bash
python3.12 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 numpy   # plain PyPI = the cu130 build
pip install -r requirements_deepspeed.txt             # minus the torch/torchvision lines
```

DeepSpeed JIT-compiles its C++/CUDA ops against the CUDA toolkit, so export the toolkit
paths (the script defaults `CUDA_HOME` to `/usr/local/cuda-13` if unset; a pre-set
`CUDA_HOME` wins):

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### AMD / ROCm 7.2

Same order; torch comes from the ROCm wheel index, which satisfies the pins exactly
(`torch 2.11.0+rocm7.2`, `torchvision 0.26.0+rocm7.2`). Every other pin is unchanged, and
`deepspeed==0.19.4` is a pure-python wheel on both platforms.

```bash
python3.12 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_deepspeed.txt   # minus the torch/torchvision lines
```

No toolkit export is needed on ROCm: no DeepSpeed op is JIT-built for ZeRO-2 +
`adamw_torch`. Do not install `flash-attn`. `bitsandbytes` is already pinned (PyPI wheels
include ROCm builds) and is only used by `--optim adamw_bnb_8bit` and `--load_in_4bit`.

### Verify

```bash
python -c "import torch, deepspeed, trl, transformers, peft, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### Extra step for DPO / GRPO (both platforms)

Under these pins trl 0.24.0 unconditionally imports `mergekit`, `llm_blender` and `weave`,
so `--train_mode dpo|grpo` fails at import without:

```bash
pip install mergekit && pip install accelerate==1.14.0   # restore the pin mergekit downgrades
```

plus empty stub packages named `llm_blender` and `weave` in `site-packages` - the real
packages are incompatible with transformers 5.5.0, and TRL only touches them in code paths
this trainer never uses.

Use a text-only model for DPO (e.g. `google/gemma-3-1b-it`); trl 0.24.0 treats `gemma4` as
a vision model and fails on `GemmaTokenizer`. SFT with gemma-4 is unaffected.

### Secrets

`HF_TOKEN` for gated models comes from `dev.env` in this folder, loaded by
`load_dotenv('dev.env')` - so run the script from inside `training/llm/deepspeed/`:

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

### accelerate config

Launch with `accelerate launch --use_deepspeed`. The only fields that matter are
`distributed_type` and `zero3_init_flag`. Minimal
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

SFT and GRPO take a `messages` JSONL, one row per line; the final message must be a
non-empty `assistant` turn. DPO takes a `--pref_file` JSONL instead.

```jsonc
// SFT / GRPO --train_file
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
// DPO --pref_file
{"system": "...", "prompt": "...", "chosen": "...", "rejected": "..."}
```

The shipped sample `data/OTel_LLM_sample_10.jsonl` carries extra metadata columns
(`unmask`, `flow`, `source_id`, ...) that this trainer drops at tokenization. Some are NULL
in every row, so drop them in any loader of your own - a naive `datasets`/pyarrow load
infers a `null` dtype and breaks schema-sensitive pipelines.

Point `--train_file` at your own JSONL (relative or absolute) to swap in real data. Rows
that break the `messages` contract fail the preflight with a row index and reason.
Over-length rows (`> --max_token_length`) are dropped, never truncated.

## Run

Run from inside `training/llm/deepspeed/`. Effective batch =
`batch_size x grad_acc_steps x num GPUs`. Always pass `--flash_attention sdpa`; the
default (`flash_attention_2`) fails fast because Gemma-4's head_dim exceeds FA2's 256
limit.

```bash
export OUTPUT_DIR=/path/to/outputs     # training artifacts (checkpoints, adapters, logs)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Smoke test, 1 GPU (the 10-row sample leaves 8 train / 2 eval rows):

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

Full run, 8 GPUs, LoRA SFT, ZeRO-2:

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  --main_process_port 29650 train_llm_deepspeed.py \
  --train_mode sft --train_file /path/to/train.jsonl \
  --model_name google/gemma-4-31B-it \
  --max_token_length 3100 --mask_prompt --gradient_checkpointing --flash_attention sdpa \
  --use_lora --zero_stage 2 \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.0 --lora_target_modules all-linear \
  --optim adamw_bnb_8bit \
  --batch_size 4 --grad_acc_steps 2 --num_train_epochs 3 --learning_rate 2e-4 \
  --experiment_root experiments/ --output_subdir sft_lora \
  > sft_lora.log 2>&1
```

Add `--test_mode --test_mode_count 256` to cap the dataset for a quick verification run.

DPO: swap in `--train_mode dpo --pref_file <pairs.jsonl> --model_name google/gemma-3-1b-it`.
GRPO: `--train_mode grpo --grpo_num_generations 2`; the global generation batch
(`num_processes x batch_size`) must be divisible by `--grpo_num_generations`. Either can
continue from an SFT adapter with `--use_lora --init_adapter /path/to/sft/final_model`.

### Scaling

Going from 1 to 8 ranks needs only `--num_processes` and `--main_process_port` - no change
to the code, requirements or DeepSpeed config.

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # NVIDIA, or the subset you own
```

- Re-export both variables after activating the venv and assert
  `torch.cuda.device_count() == N`; a stale `export CUDA_VISIBLE_DEVICES=0,1` left in an
  `activate` script silently pins an "8-GPU" launch to 2 GPUs.
- Pass a non-default `--main_process_port` (e.g. `29650`); 29500 collides with any parallel
  job on the box.
- On a shared box use `accelerate launch`, which honors `CUDA_VISIBLE_DEVICES`. The bare
  `deepspeed --num_gpus N` launcher can ignore it and grab GPU 0; if you must use it,
  `unset CUDA_VISIBLE_DEVICES` and pass `deepspeed --include localhost:4,7 --master_port 29670`.
- Size the dataset to the world size: the eval split is floored at `world_size`, so use at
  least `world_size x 32` rows (tiling the sample is fine) for a meaningful N-GPU run.

## Arguments

Defaults are the tested values.

| Flag | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Training data (`messages` JSONL) |
| `--model_name` | `google/gemma-4-E4B-it` | HF model id or local path; tokenizer must have a chat template |
| `--experiment_root` | `experiments/` | Root under which run outputs are created |
| `--hf_home` | `hf_cache/` | `HF_HOME` cache root |
| `--output_subdir` | `gemma4_ftaas_uc524_finetuned` | Subdirectory under `experiment_root/<RUN_ID>` |
| `--output_dir` | `None` | Full output-dir override |
| `--resume_from_checkpoint` | `""` | Checkpoint to resume from (used only if it exists) |
| `--load_best_model_at_end` | off | Load lowest-eval-loss checkpoint as final; leave off for memorization-style runs |
| `--max_token_length` | `32768` | Max tokens per example; longer rows are dropped |
| `--eval_samples` | `1000` | Target eval-set size (floored at world size) |
| `--preflight_sample_size` | `256` | Rows validated in the fail-fast preflight |
| `--supervision_sample_size` | `16` | Rows sampled for the supervision sanity check |
| `--test_mode` | off | Cap the dataset at `--test_mode_count` rows |
| `--test_mode_count` | `10000` | Row cap applied with `--test_mode` |
| `--seed` | `42` | Seed for `set_seed` and dataset shuffle/split |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--num_proc` | `8` | Workers for dataset map/filter |
| `--batch_size` | `4` | Per-device train/eval batch size |
| `--grad_acc_steps` | `2` | Gradient accumulation steps |
| `--num_train_epochs` | `3` | Epochs |
| `--learning_rate` | `2e-4` | Peak LR (LoRA; use ~1e-5..2e-5 for full FT) |
| `--lr_scheduler_type` | `cosine` | LR scheduler |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `0.03` | <1 = fraction of total steps; >=1 = absolute steps |
| `--logging_steps` | `50` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--optim` | `adamw` | `adamw` (-> `adamw_torch`), `sgd`, `rmsprop`, `adamw_bnb_8bit` |
| `--mask_prompt` | off | Completion-only loss |
| `--gradient_checkpointing` | off | Enable gradient checkpointing |
| `--flash_attention` | `flash_attention_2` | Must pass `sdpa`; `flash_attention_2` fails fast |
| `--use_lora` | off | Train a LoRA adapter instead of full FT |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha (typically 2x rank) |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or comma-separated module list |
| `--load_in_4bit` | off | QLoRA 4-bit base; requires `--use_lora` and `--zero_stage 0` |
| `--custom_eval` | off | Generation-based per-epoch eval via `--scorer_module`; skipped under ZeRO-3 |
| `--test_dir` | `test` | Dir with `<name>_eval.jsonl` files for `--custom_eval` |
| `--scorer_module` | `step8_score_eval` | Module exposing `EVAL_DATASETS`/`load_eval`/`score_dataset`/`macro_average` |
| `--eval_max_new_tokens` | `768` | Max new tokens per prompt during custom eval |
| `--empty_cache_steps` | `0` | Flush CUDA cache every N steps (0 = off) |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage (see below) |
| `--offload_optimizer` | off | Offload optimizer state to CPU pinned memory |
| `--train_mode` | `sft` | `sft`, `dpo`, or `grpo` |
| `--init_adapter` | `None` | Existing LoRA adapter to continue training from |
| `--pref_file` | `None` | DPO preference JSONL; required for `--train_mode dpo` |
| `--dpo_beta` | `0.1` | DPO KL strength (lower = closer to the reference policy) |
| `--dpo_max_length` | `4096` | DPO max total sequence length |
| `--dpo_max_prompt_length` | `3072` | Parsed but unused - this TRL version's `DPOConfig` has no `max_prompt_length` |
| `--dpo_precompute_ref_log_probs` | off | Cache ref log-probs up front, drop the ref model from the loop |
| `--grpo_num_generations` | `8` | GRPO candidates per prompt |
| `--grpo_max_completion_length` | `512` | GRPO max new tokens per candidate |
| `--grpo_temperature` | `1.0` | GRPO sampling temperature |

ZeRO stages (`--zero_stage`):

| Stage | Shards | When to use |
|---|---|---|
| `0` | nothing (plain DDP) | small models; required with `--load_in_4bit` |
| `1` | optimizer state | mild memory relief |
| `2` | + gradients | recommended for LoRA / any model that fits on one GPU |
| `3` | + parameters | models too large for one GPU; adds param-gather overhead |

## Output

Artifacts land in `<experiment_root>/<RUN_ID>/<output_subdir>/`, or in `--output_dir`
verbatim. `RUN_ID` comes from the environment or a UTC timestamp. `--experiment_root` and
`--hf_home` accept absolute paths.

- Per-epoch checkpoints, capped by `--save_total_limit`.
- `final_model/` - the LoRA adapter, or consolidated 16-bit weights for full FT. A LoRA
  adapter needs the base model plus `PeftModel.from_pretrained` at inference.
- `train_script_backup.py`, TensorBoard logs, and `train_err_rank<N>.log` per rank on
  failure.

A ZeRO-3 full fine-tune writes a consolidated `final_model` (~15 GiB for an 8B model) on
top of each checkpoint - budget ~35 GB per run.

## Notes

- There is no `ds_config.json`. The DeepSpeed config is built in memory from
  `--zero_stage` and `--offload_optimizer`; an on-disk config file is ignored.
- `--custom_eval` is skipped under `--zero_stage 3` (rank-0 generation would hang on the
  param all-gather). Use `--zero_stage 2`, or evaluate the saved adapter offline.
- The shipped GRPO reward (`utils.grpo_reward_funcs`) only rewards non-empty completions.
  Replace it before a real GRPO run; until you do, `reward: 1`, `reward_std: 0` and
  `loss 0` are the correct output, not a failure.
- `PermissionError` on `cache-*.arrow`: export
  `HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache` when the model cache mount is read-only,
  leaving `--hf_home` on the pre-cached model store.
- For fully offline runs export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and use a model
  whose tokenizer is already cached.
- No TF32 flag is set on either platform (it would raise on ROCm). bf16 does the compute;
  call `torch.set_float32_matmul_precision("high")` yourself if you want TF32 on the fp32
  residuals.
