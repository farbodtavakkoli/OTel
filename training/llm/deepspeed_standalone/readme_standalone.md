# `training/llm/deepspeed_standalone` — single-file DeepSpeed SFT trainer

Single-file SFT trainer on HF Transformers + DeepSpeed (ZeRO). It imports only
third-party packages, so this folder runs on its own. Full fine-tuning only (no
LoRA/PEFT) and SFT only - for LoRA, DPO or GRPO use `../deepspeed/`.

Prompt formatting is hand-written per model in `format_conversation` rather than taken
from the tokenizer's chat template, and over-length rows are truncated rather than
dropped. Pick this recipe when you want one portable file; pick `../deepspeed/` when you
want chat-template parity.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 2 and 8 GPUs, ~8B full FT · NVIDIA H100 80GB
(CUDA 13.0), 1 GPU, sub-1B full FT.

## Files

| File | Purpose |
|---|---|
| `train_llm_deepspeed_standalone.py` | The entire trainer: templates, tokenization/masking, in-memory DeepSpeed config, training loop |
| `requirements_standalone.txt` | Pinned environment (Python 3.12) |
| `OTel_LLM_sample_10.jsonl` | 10-row `messages` sample; the default `--train_file` |

## Setup

Python 3.12. One venv per recipe folder; all pins live in `requirements_standalone.txt`.
Install torch first, then the requirements - the second install must not re-resolve torch.

```bash
export OUTPUT_DIR=/path/to/outputs     # training artifacts / experiment root
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export DATA_DIR=/path/to/data          # training JSONL files
```

### NVIDIA (CUDA 13)

The `torch==2.11.0` pin resolves to the CUDA 13 build on plain PyPI, so no `--index-url`
is needed. Every other pin installs unchanged (transformers 5.5.0, trl 0.24.0,
datasets 4.3.0, accelerate 1.14.0, deepspeed 0.19.4, bitsandbytes 0.50.0).

```bash
python3.12 -m venv .env_deepspeed_standalone
source .env_deepspeed_standalone/bin/activate
pip install -r requirements_standalone.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

To pin the `+cu130` local version tag explicitly, install torch first with
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

DeepSpeed JIT-compiles its kernels against the CUDA toolkit, so export the toolkit paths
(the script defaults `CUDA_HOME` to `/usr/local/cuda-13` if unset; a pre-set `CUDA_HOME`
wins):

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### AMD / ROCm 7.2

```bash
python3.12 -m venv .env_deepspeed_standalone
source .env_deepspeed_standalone/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_standalone.txt   # torch pin already satisfied; all other pins unchanged
export CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm   # else the script prepends phantom CUDA paths
```

`deepspeed==0.19.4` installs as a pure-python wheel and JIT-compiles ops on demand, so
there is no `hipcc` build at install time. The default optimizer is `adamw_bnb_8bit`, and
the PyPI `bitsandbytes==0.50.0` wheel ships working ROCm binaries for gfx950 (DeepSpeed
logs an "untested optimizer" warning and proceeds); `--optim adamw_torch` avoids it
entirely. Do not pip-install the CUDA `flash-attn` wheel.

### Verify

```bash
python -c "import torch, deepspeed, trl, transformers, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### Secrets

`HF_TOKEN` for gated models comes from `dev.env` in this folder, loaded by
`load_dotenv('dev.env')` - so run the script from inside
`training/llm/deepspeed_standalone/`:

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token. Ungated models need no
`dev.env` at all.

## Data

`format_conversation` renders each row into one training string using the template
selected by `--model_type`. Two input shapes:

```jsonc
// most model types - flat rows (optional "reasoning" for the gpt-oss reasoning template)
{"prompt": "...", "completion": "..."}
// --model_type lfm_ftaas (the default) - messages rows, flattened by role
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Built-in templates: `qwen3`, `llama3`, `gemma3`, `gemma-4`, `mistral`, `olmo3`, `rnj-1`,
`lfm` / `lfm_ftaas`, `phi4`, `gpt-oss_reasoning`, `gpt_oss_it`. Anything else falls back to
a plain `User:/Assistant:` format; add a branch to `format_conversation` for a new model.

The shipped `OTel_LLM_sample_10.jsonl` carries extra metadata columns (`unmask`, `flow`,
`source_id`, `source_repo`, `source_spec_id`, `source_version`). This trainer drops them
all before tokenization. Drop them in any loader of your own too: `source_spec_id` and
`source_version` are NULL in every row, so a naive `datasets`/pyarrow load infers a `null`
dtype and breaks schema-sensitive pipelines.

To swap in real data, point `--train_file` at your own JSONL (relative or absolute) and
pick the `--model_type` matching your target model's prompt format.

## Run

Run from inside `training/llm/deepspeed_standalone/`. Launch with
`accelerate launch --use_deepspeed`; the accelerate config only needs
`distributed_type: DEEPSPEED` and `zero3_init_flag` (minimal YAML in `../deepspeed/`'s
README). There is no `ds_config.json` - the config is built in memory from `--zero_stage`
and `--offload_optimizer` and takes priority over any `deepspeed_config_file`.

Smoke test, 1 GPU (`--test_size 0.2` gives 8 train / 2 eval rows from the sample):

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=1 --main_process_port=29645 \
  --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file OTel_LLM_sample_10.jsonl \
  --model_name LiquidAI/LFM2.5-350M --model_type lfm_ftaas \
  --zero_stage 3 \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --test_size 0.2 \
  --max_token_length 4096 --logging_steps 1 --warmup_steps 2 --num_proc 4 \
  > smoke_standalone.log 2>&1
```

Full run, 8 GPUs, ~8B full fine-tune under ZeRO-3:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # NVIDIA, or the subset you own

RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
nohup accelerate launch --num_processes=8 --main_process_port=29620 \
  --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file /path/to/data.jsonl \
  --model_name google/gemma-4-E4B-it --model_type gemma-4 \
  --zero_stage 3 \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --test_size 0.2 \
  --max_token_length 4096 --logging_steps 1 --warmup_steps 2 --num_proc 8 \
  > train_llm_deepspeed_standalone.log 2>&1 &

tail -f train_llm_deepspeed_standalone.log
```

Scaling from 1 to 8 ranks needs no code, requirements or DeepSpeed-config change - only
`--num_processes` and a distinct `--main_process_port`. No RCCL/NCCL variables are needed,
and per-device batch stays at 1 (the global batch grows with the ranks).

- Re-export both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after activating the venv
  and assert `torch.cuda.device_count() == N`; a stale `export CUDA_VISIBLE_DEVICES=0,1`
  left in `bin/activate` silently pins an "8-GPU" launch to 2 cards.
- Add `--no_save` for smoke tests: a ZeRO-3 checkpoint of an 8B model is >100 GB.
- Replicate the 10-row sample (e.g. 25x -> 250 rows) so a global batch of 8 yields more
  than one optimizer step.
- An ~8B full FT under ZeRO-3 will OOM on a single 80 GB card. Use enough ranks, or
  `--offload_optimizer`.

## Arguments

Path defaults are relative to the working directory.

| Flag | Default | Meaning |
|---|---|---|
| `--train_file` | `OTel_LLM_sample_10.jsonl` | Training data JSONL |
| `--model_name` | `LiquidAI/LFM2.5-1.2B-Instruct` | HF repo id or local path |
| `--model_type` | `lfm_ftaas` | Prompt template selector in `format_conversation` |
| `--experiment_root` | `experiments/` | Root under which run outputs are created |
| `--hf_home` | `hf_cache/` | `HF_HOME` cache root |
| `--output_subdir` | `lfm_ftaas_uc524_finetuned` | Subdirectory under `experiment_root/<RUN_ID>` |
| `--output_dir` | `None` | Full output-dir override |
| `--resume_from_checkpoint` | `""` | Checkpoint to resume from (used only if it exists) |
| `--max_token_length` | `16192` | Tokenizer max length; longer examples are truncated |
| `--test_mode` | off | Cap the dataset at `--test_mode_count` rows |
| `--test_mode_count` | `10000` | Row cap applied with `--test_mode` |
| `--seed` | `42` | Seed for `set_seed` and dataset shuffle/split |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--test_size` | `0.0002` | Eval split as a fraction of rows |
| `--max_eval_samples` | `None` | Cap on eval rows |
| `--num_proc` | `8` | Workers for dataset map/filter |
| `--no_mask_prompt` | (mask on) | Train on the full sequence instead of completion-only loss |
| `--batch_size` | `16` | Per-device train/eval batch size |
| `--grad_acc_steps` | `4` | Gradient accumulation steps |
| `--num_train_epochs` | `2` | Epochs |
| `--learning_rate` | `1e-5` | Peak learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--optim` | `adamw_bnb_8bit` | HF optim id (needs bitsandbytes; `adamw_torch` avoids it) |
| `--warmup_steps` | `100` | LR warmup steps |
| `--logging_steps` | `50` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--no_load_best_model_at_end` | (best on) | Keep the last checkpoint instead of the lowest-eval-loss one |
| `--no_save` | off | Disable all checkpointing and the final save (smoke tests) |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage (0-3; 3 = param+grad+optimizer sharding) |
| `--offload_optimizer` | off | Offload optimizer state to CPU pinned memory |

## Output

Artifacts land in `<experiment_root>/<RUN_ID>/<output_subdir>/`, or in `--output_dir`
verbatim. `RUN_ID` comes from the environment or a UTC timestamp. Absolute paths work
(`--experiment_root $OUTPUT_DIR/experiments/`, `--hf_home $HF_HOME`).

- Per-epoch checkpoints capped by `--save_total_limit`, eval each epoch, best model loaded
  at the end unless `--no_load_best_model_at_end`. `--no_save` turns all of this off.
- `final_model/` - a full model directory, not an adapter:
  `model.safetensors` plus `config.json`, `generation_config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`.
- `train_script_backup.py` and TensorBoard logs.

## Notes

- Export `HF_DATASETS_CACHE` to writable storage (e.g. `/dev/shm/ds_cache`) whenever
  `HF_HOME` sits on a read-mostly shared mount; otherwise `Dataset.map` dies with
  `PermissionError: [Errno 1] Operation not permitted`.
- The character pre-filter runs at `max_token_length * 4` chars, so a small
  `--max_token_length` silently drops long rows - `--max_token_length 1024` keeps only ~3
  of the 10 sample rows. Use `4096` for the shipped sample.
- Rows whose completion is truncated away end up all-`-100` and log `loss: 0`; an eval
  shard made entirely of such rows makes `eval_loss` come back `nan`. Use a real eval set
  or `--test_size 0` beyond a smoke test.
- `--model_type lfm_ftaas` triggers the script's `Duplicate BOS token detected!` warning:
  the template embeds `<|startoftext|>` and the LFM2.5 tokenizer prepends its own. Drop the
  leading token from the template, or set the tokenizer's `add_bos_token=False`, for real
  LFM training.
- With the default `--test_size 0.0002` a tiny dataset still gets at least one eval row;
  pass a larger `--test_size` for small files.
- For fully offline nodes export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and use a model
  already present in the cache (the default `LiquidAI/LFM2.5-1.2B-Instruct` may not be).

Relationship to `../deepspeed/`:

| | this folder | `../deepspeed/` |
|---|---|---|
| Dependencies | none (single file) | `utils.py` |
| Modes | SFT only | SFT / DPO / GRPO |
| Fine-tuning | full FT only | LoRA or full FT |
| Prompt format | hand-written per-`--model_type` strings | tokenizer's official chat template |
| Over-length rows | truncated | dropped |
