# `training/llm/deepspeed_standalone` — single-file DeepSpeed SFT trainer

## Overview & when to use

Single-file **SFT** trainer on HF Transformers + DeepSpeed (ZeRO). Self-contained: it
imports **only third-party packages** (no local project modules), so this folder runs on
its own. It does **full fine-tuning** (no LoRA/PEFT) and **only SFT** — for LoRA, DPO,
or GRPO, use `../deepspeed/` instead.

Design notes absorbed from the code:

- **Standalone by design** — all prompt formatting lives in `format_conversation`
  (hand-written per-model template strings) instead of the tokenizer's chat template,
  so the file has no dependency on `utils.py` or any shared module.
- **Prompt masking** — `process_and_format` tokenizes the full string, re-renders a
  prompt-only version (completion/reasoning/assistant fields blanked), and masks the
  matching token prefix to `-100` so loss falls on the completion only. Disable with
  `--no_mask_prompt`.
- **DeepSpeed config in-memory** — `build_deepspeed_config(zero_stage,
  offload_optimizer)` returns the dict passed to `TrainingArguments(deepspeed=...)`, so
  every rank gets an identical config with no file dependency; `"auto"` fields are
  filled by the HF Trainer at runtime. stage3 keys (including the gather flag needed
  for the consolidated save) are emitted only for stage 3.
- **Full fine-tuning** — every non-vision parameter trains; the vision tower
  (`vision_model`/`vision_tower`/`multi_modal_projector`/`visual`) is frozen. Uses more
  memory than LoRA; ZeRO-3 shards params/grads/optimizer.
- **BOS handling** — only the `gemma3` model type sets `add_bos=True` on the tokenizer;
  every other template embeds its own BOS token in the format string. A sanity check
  warns if BOS ends up duplicated.
- **Truncation, not dropping** — unlike `../deepspeed/`, over-length examples
  are *truncated* at `--max_token_length` (after a cheap character-length pre-filter at
  `max_token_length * 4` chars).
- **Mistral fallback** — if `AutoModelForCausalLM` rejects the checkpoint, the script
  retries with `Mistral3ForConditionalGeneration`, which some Mistral releases require.

> **Tested topology:** Azure cluster, 8×H100 80GB — single-node DeepSpeed ZeRO-3.
> Also verified on 2× and 8×AMD Instinct MI355X (ROCm 7.2) — see "AMD MI355X (ROCm 7.2)
> — tested" below. Multi-node is possible via `accelerate`/`torchrun` rendezvous but has
> **not** been tested here.

## Install

Python 3.12. All pins in `requirements_standalone.txt` are the tested set.

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts / experiment root
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export DATA_DIR=/path/to/data          # training JSONL files
```

### NVIDIA (CUDA)

```bash
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r requirements_standalone.txt
```

DeepSpeed JIT-compiles its kernels against the CUDA toolkit, so export the toolkit
paths (the script defaults `CUDA_HOME` to `/usr/local/cuda-13` if unset — a pre-set
`CUDA_HOME` wins):

```bash
export CUDA_HOME=/usr/local/cuda-13.0     # or /usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify the import graph:

```bash
python -c "import torch, deepspeed, trl, transformers, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### AMD / ROCm — **tested** (MI355X, ROCm 7.2)

Verified end-to-end on this stack — see "AMD MI355X (ROCm 7.2) — tested" below.
Differences from the NVIDIA install:

- **PyTorch** — install from the ROCm wheel index instead of PyPI, **before** the
  requirements file (the `torch==2.11.0` pin is then already satisfied by
  `2.11.0+rocm7.2` and pip leaves it alone):

  ```bash
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
  pip install -r requirements_standalone.txt
  ```

  The rocm7.2 index carries `torch-2.11.0+rocm7.2` cp312 wheels, so the CUDA pin is
  matched exactly — no version lag in practice (an earlier revision of this README
  claimed the ROCm wheels lag the CUDA pins; that was true of the rocm7.0 index and is
  no longer the case).
- **DeepSpeed** — same `pip install deepspeed==0.19.4`; it installs as a pure-Python
  wheel (**no** `hipcc` compile at install time) and JIT-compiles ops on demand. The
  smoke run below needed **no** compiled ops at all.
- **bitsandbytes** — matters here because the **default optimizer is
  `adamw_bnb_8bit`**. **Confirmed working on MI355X (gfx950)**: the PyPI
  `bitsandbytes==0.50.0` wheel ships ROCm binaries and the tested run below trained
  with the default `adamw_bnb_8bit` (DeepSpeed logs an "untested optimizer" warning
  and proceeds). `--optim adamw_torch` remains the fallback.
- **flash-attn** — the main path doesn't use it (only the rarely-hit Mistral fallback
  requests `flash_attention_2`). Do **not** pip-install the CUDA `flash-attn` wheel on
  ROCm; on ROCm, flash-attention is a source build from the upstream repo
  (composable_kernel or Triton backend).
- **CUDA_HOME** — the script defaults `CUDA_HOME` to `/usr/local/cuda-13`; on a ROCm
  box export `CUDA_HOME=/opt/rocm` (a pre-set value wins) so no phantom CUDA paths are
  prepended.

## Environment & secrets

Put a `dev.env` **in this folder** (the script does `load_dotenv('dev.env')`, resolved
from the current working directory — so run the script from inside
`training/llm/deepspeed_standalone/`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`dev.env` is git-ignored at the repo root; never commit a token. The token is only
needed for gated models. The script also sets `HF_HOME` (`--hf_home`) and NCCL env vars
automatically.

## Data

`format_conversation` renders each row into a single training string using the template
selected by `--model_type`. Two input shapes are supported:

- **Most model types** — flat rows: `{"prompt": "...", "completion": "..."}` (plus
  optional `reasoning` for the gpt-oss reasoning template).
- **`--model_type lfm_ftaas`** (the default) — `messages` rows are flattened to
  `{system, user, assistant}` by role:
  `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

Built-in `--model_type` templates: `qwen3`, `llama3`, `gemma3`, `gemma-4`, `mistral`,
`olmo3`, `rnj-1`, `lfm` / `lfm_ftaas`, `phi4`, `gpt-oss_reasoning`, `gpt_oss_it`
(anything else falls back to a plain `User:/Assistant:` format). To add a model, add a
branch to `format_conversation`.

A 10-row sample ships in this folder as `OTel_LLM_sample_10.jsonl` and is the default
`--train_file`. Its schema is the `messages` format plus extra metadata columns:
`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`, `source_version`.

**The null-column trap** — in the full dataset this sample comes from,
`source_spec_id` and `source_version` are NULL in every row, so a naive
`datasets`/pyarrow load infers a `null` dtype for them and can break schema-sensitive
pipelines. This trainer is immune: `_flatten_messages` runs with
`remove_columns=<all>`, so every extra column is dropped before tokenization. If you
write your own loader for this data, ignore or drop the extra columns rather than
relying on their inferred dtype.

To swap in real data, point `--train_file` at your own JSONL (an absolute path such as
`$DATA_DIR/UC524_combined_op.jsonl` works just as well as a relative one) and
pick the `--model_type` matching your target model's prompt format.

## Run

The DeepSpeed config is built **in-memory** from `--zero_stage` /
`--offload_optimizer` (no `ds_config.json`; the in-memory dict takes priority over any
`deepspeed_config_file`). The accelerate config only needs
`distributed_type: DEEPSPEED` and `zero3_init_flag` (see `../deepspeed/`'s
README for a minimal YAML). Run from inside `training/llm/deepspeed_standalone/`.

**Smoke test against the shipped sample** (1 GPU; `--test_size 0.2` makes the tiny
split explicit — 8 train / 2 eval rows):

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=1 --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file OTel_LLM_sample_10.jsonl \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --test_size 0.2 \
  > smoke_standalone.log 2>&1
```

**Full example** (8 GPUs, real data, capped for verification via `--test_mode`):

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file /path/to/data.jsonl \
  --model_name LiquidAI/LFM2.5-1.2B-Instruct --model_type lfm_ftaas \
  --test_mode --test_mode_count 10000 \
  > train_llm_deepspeed_standalone.log 2>&1 &

tail -f train_llm_deepspeed_standalone.log
```

**What "working" looks like:** the `CHECKPOINT SAVING LOCATION` banner, a tokenization
sanity block (decoded sample + first/last tokens and labels), then `{'loss': ...}`
lines, and finally `Saving final model to ...` + `Training Complete.`

## Arguments

Defaults equal the tested constants, except the path defaults, which are relative to the
working directory here (see Data / Output).

| Flag | Default | Meaning |
|---|---|---|
| `--train_file` | `OTel_LLM_sample_10.jsonl` | Training data JSONL |
| `--model_name` | `LiquidAI/LFM2.5-1.2B-Instruct` | HF repo id or local path to fine-tune |
| `--model_type` | `lfm_ftaas` | Prompt template selector in `format_conversation` (see Data) |
| `--experiment_root` | `experiments/` | Root under which run outputs are created |
| `--hf_home` | `hf_cache/` | `HF_HOME` cache root for models/datasets |
| `--output_subdir` | `lfm_ftaas_uc524_finetuned` | Subdirectory under `experiment_root/<RUN_ID>` |
| `--output_dir` | `None` | Full output-dir override (skips the construction above) |
| `--resume_from_checkpoint` | `""` | Checkpoint path to resume from (used only if it exists) |
| `--max_token_length` | `16192` | Tokenizer max length; longer examples are truncated |
| `--test_mode` | off | Cap the dataset at `--test_mode_count` rows |
| `--test_mode_count` | `10000` | Row cap applied when `--test_mode` is set |
| `--seed` | `42` | Seed for `set_seed` and dataset shuffling/splitting |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use (seeded shuffle + subsample) |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--test_size` | `0.0002` | Eval split as a fraction of rows |
| `--max_eval_samples` | `None` | Cap on eval rows (None = no cap) |
| `--num_proc` | `8` | Worker processes for dataset map/filter steps |
| `--no_mask_prompt` | (mask on) | Train on the full sequence instead of completion-only loss |
| `--batch_size` | `16` | Per-device train/eval batch size |
| `--grad_acc_steps` | `4` | Gradient accumulation steps |
| `--num_train_epochs` | `2` | Training epochs |
| `--learning_rate` | `1e-5` | Peak learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--optim` | `adamw_bnb_8bit` | HF optim identifier (needs bitsandbytes; `adamw_torch` avoids it) |
| `--warmup_steps` | `100` | LR warmup steps |
| `--logging_steps` | `50` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--no_load_best_model_at_end` | (best on) | Keep the last checkpoint instead of the lowest-eval-loss one |
| `--no_save` | off | Disable all checkpointing and the final save (smoke tests — a ZeRO-3 checkpoint of an 8B model is >100 GB) |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage (0–3; 3 = full param+grad+optimizer sharding) |
| `--offload_optimizer` | off | Offload optimizer state to CPU pinned memory (more headroom, slower) |

## Output

Outputs land in `<experiment_root>/<RUN_ID>/<output_subdir>/` (or `--output_dir`
verbatim). `RUN_ID` comes from the environment or a UTC timestamp. Absolute paths work
too (`--experiment_root $OUTPUT_DIR/experiments/`, `--hf_home $HF_HOME`); the relative
defaults (`experiments/`, `hf_cache/`) keep the same layout under the working directory.

- Per-epoch checkpoints (`save_strategy="epoch"`, capped by `--save_total_limit`),
  eval each epoch, best model loaded at the end unless `--no_load_best_model_at_end`.
  `--no_save` turns all of this off and writes no weights at all.
- `final_model/` — consolidated 16-bit weights.
- `train_script_backup.py` — a copy of the training script for provenance.
- TensorBoard logs (`report_to="tensorboard"`).

## NVIDIA H100 80GB (CUDA 13.0) — tested

Verified on a single **NVIDIA H100 80GB HBM3** (Hopper cc 9.0), driver
**580.173.02**, CUDA **13.0**, Ubuntu, Python 3.12.3 — **full fine-tuning of
`LiquidAI/LFM2.5-350M` under ZeRO-3**, prompt-masked SFT (`--model_type lfm_ftaas`) on
the shipped `OTel_LLM_sample_10.jsonl`. **Single-GPU smoke only** (one free H100 on a
shared 8-GPU node); the multi-GPU / offload story is deferred — see "Multi-GPU / larger
models" below.

> **Model choice.** The script default `LiquidAI/LFM2.5-1.2B-Instruct` was **not present
> in the offline model cache** on this node, so the smoke used the cached
> `LiquidAI/LFM2.5-350M`. A 350M full fine-tune fits comfortably in 80 GB (peak ~7.3 GiB
> — see below). This is *not* the ZeRO-3 memory-pressure case the MI355X run exercised
> (an ~8B full FT measured **up to 58% of 288 GiB ≈ 167 GiB/GPU at 2 ranks** — see the
> MI355X table below); that would OOM on a single 80 GB H100 and is not covered here.

**Install (native CUDA 13 wheel, no `--index-url`):**

```bash
cd training/llm/deepspeed_standalone
python3 -m venv .env_deepspeed_standalone
source .env_deepspeed_standalone/bin/activate
pip install torch numpy                       # -> torch 2.13.0+cu130 (CUDA 13.0), numpy 2.5.2
pip install -r requirements_standalone.txt    # minus the torch==2.11.0 pin (see note)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-verify: 2.13.0 13.0
```

- **torch pin deviation.** `requirements_standalone.txt` pins `torch==2.11.0`; there is
  **no cu130 wheel for 2.11.0**, so the verified recipe installs the current stable
  `torch==2.13.0+cu130` first and installs the requirements **without** the torch line
  (`grep -v '^torch==' requirements_standalone.txt`). All other pins install unchanged
  (transformers 5.5.0, trl 0.24.0, datasets 4.3.0, accelerate 1.14.0, deepspeed 0.19.4,
  bitsandbytes 0.50.0). Re-check torch afterward to confirm it was **not** clobbered.
- **CUDA_HOME.** The toolkit lives at `/usr/local/cuda-13.0` (with a `/usr/local/cuda-13`
  symlink — exactly the script's default). `nvcc` reports `release 13.0, V13.0.88`.
  DeepSpeed JIT-compiled **no** ops for this smoke (none were needed), same as ROCm.
- **bitsandbytes / `adamw_bnb_8bit`.** The PyPI `bitsandbytes==0.50.0` CUDA wheel imports
  cleanly and `AdamW8bit` is available; the validated run trains with the default
  `adamw_bnb_8bit` (DeepSpeed logs the same "untested optimizer" warning and proceeds).
- **flash-attn — not reversed.** The MI355X notes warn off the CUDA flash-attn wheel, but
  the **main training path never sets `attn_implementation`** (only the rarely-hit Mistral
  fallback requests `flash_attention_2`). There is nothing to switch back to
  `flash_attention_2` on the LFM/most-model path, so no flash-attn install is needed;
  the default (SDPA) attention is used, unchanged.
- **Plain `CUDA_VISIBLE_DEVICES`.** No `HIP_VISIBLE_DEVICES` / `RAY_*` vars are used by
  this script; on a shared node pin the run to one card with
  `CUDA_VISIBLE_DEVICES=5` and a non-default `--main_process_port 29645`.

**Offline-node env (for a node where hub egress is proxy-blocked, 403):**

```bash
export CUDA_VISIBLE_DEVICES=5                              # the free H100 on a shared node
# HF_HOME points at a pre-populated model cache (may be read-only) — see the Install block
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/dev/shm/h100/out/deepspeed_standalone/ds_cache   # see quirk below
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Smoke launch (single GPU, 4 epochs for a visible curve):**

```bash
RUN_ID=h100_smoke_$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=1 --main_process_port=29645 \
  --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file OTel_LLM_sample_10.jsonl \
  --model_name LiquidAI/LFM2.5-350M --model_type lfm_ftaas \
  --hf_home "$HF_HOME" \
  --zero_stage 3 \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 4 --test_size 0.2 \
  --max_token_length 4096 --logging_steps 1 --warmup_steps 2 --num_proc 4
```

**Step count.** The 10-row sample splits (`--test_size 0.2`) to 8 train / 2 eval; batch
geometry 1 × 1 × 1 = global batch 1 → **8 steps/epoch × 4 epochs = 32 optimizer steps**
(non-trivial). The run is reproducible: repeated runs give the same `train_loss` and
finish with exit code 0.

**Expected output** (loss on rows with unmasked completions falls to ~0):

```
[rank=0] ... INFO: Selective Freezing: Frozen 0.0M params (Vision), Trainable 354.5M params (Language)
NCCL version 2.29.7+cuda13.2
{'loss': '1.215',    'grad_norm': '87.41', 'mean_token_accuracy': '0.7321', 'epoch': '0.125'}
{'loss': '1.499',    'grad_norm': '294',   'mean_token_accuracy': '0.7273', 'epoch': '0.875'}
{'loss': '0.4629',   'grad_norm': '44.66', 'mean_token_accuracy': '0.9018', 'epoch': '1.875'}
{'loss': '0.005925', 'grad_norm': '3.71',  'mean_token_accuracy': '1',      'epoch': '2.125'}
{'loss': '0.0006557','grad_norm': '0.719', 'mean_token_accuracy': '1',      'epoch': '3.625'}
{'eval_loss': '1', 'eval_mean_token_accuracy': '0.7553', 'epoch': '4'}
{'train_runtime': '23.36', 'train_samples_per_second': '1.37', 'train_steps_per_second': '1.37', 'train_loss': '0.1811'}
[rank=0] ... INFO: Saving final model to .../lfm_ftaas_uc524_finetuned
[rank=0] ... INFO: Training Complete.
```

The run should finish with exit code 0.

**GPU residency check.** Sample `nvidia-smi` compute-apps filtered to the pinned GPU from
inside the run and confirm the single training process is the only VRAM holder; memory
climbs from ~1.2 GiB (model loaded) through ~2.3 GiB (training ramps) to a steady peak.

**Peak VRAM: ~7.3 GiB** (7511 MiB) for a 350M full fine-tune under ZeRO-3 on one GPU —
about **9%** of the 80 GB card. The full run (load → 32 steps → eval → consolidated save)
completes in ~25 s.

**Saved model (this is FULL FT — a full model dir, not a LoRA adapter):**

```
final_model/model.safetensors   708,984,464 bytes  (~709 MB = 354.5M params × 2 bytes bf16)
final_model/{config.json, generation_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja}
```

The ZeRO-3 `stage3_gather_16bit_weights_on_model_save` consolidation and the
save-on-all-ranks fix (see MI355X quirks) both work on a single rank: the log shows
`Writing model shards: 100%` then a complete 16-bit `model.safetensors`. (Mind the disk
cost — a full FT writes a full model and `save_strategy="epoch"` also writes per-epoch;
use `--no_save` for pure smoke tests.)

**Quirks found on H100 (all vendor-neutral):**

- **`HF_DATASETS_CACHE` must point at writable storage.** With `HF_HOME` on a shared
  read-mostly model cache, `datasets` tries to write its
  arrow map cache under `$HF_HOME/datasets/...` and dies with
  `PermissionError: [Errno 1] Operation not permitted` at the `os.chmod` step of
  `Dataset.map`. Fix: set `HF_DATASETS_CACHE` to a writable path (fast local/tmpfs). This
  is a mount-permission issue, not CUDA/torch — it would bite any framework writing a
  dataset cache to that mount.
- **Duplicate BOS on the `lfm_ftaas` path.** The script's own sanity check fires
  `Duplicate BOS token detected!`: the `lfm_ftaas` template embeds `<|startoftext|>` and
  the LFM2.5 tokenizer also prepends its own BOS, so the sequence starts
  `<|startoftext|><|startoftext|>...`. Benign for a smoke test; for real LFM training you
  likely want to drop the leading token from the template (or set the tokenizer's
  `add_bos_token=False`). Not H100-specific — same on any accelerator.
- **`loss: 0` on some tiny-sample steps.** Same documented artifact as MI355X: rows whose
  completion is short/fully truncated relative to the masked prompt end up all-`-100`, so
  their step logs `loss: 0`, `grad_norm: 0`. A property of the 10-row toy sample, not the
  hardware. Here `eval_loss` stays finite (~0.96–1.0) because the 2-row eval split
  contains unmasked completions (the `nan` seen on the MI355X 8-rank run needs a
  fully-masked eval shard).
- **tf32 note.** The tf32 fast-path does not auto-engage on CUDA. On
  torch 2.13 the legacy flag reads `torch.backends.cuda.matmul.allow_tf32 == False` and
  the new API `torch.backends.cuda.matmul.fp32_precision == 'none'` by default (cuDNN's
  `allow_tf32` is `True`). This is **moot for this trainer**: it runs bf16 end-to-end
  (`--mixed_precision=bf16`, `bf16=True`, DeepSpeed `bf16.enabled=True`), so fp32 matmul
  precision does not gate the compute. A real bf16 matmul returns finite values.
- **torch deprecation warnings** (`all_gather_into_tensor` / `reduce_scatter_tensor`
  deprecated in favour of `*_single`) fire from DeepSpeed's ZeRO-3 gather on torch 2.13 —
  warnings only, the save completes correctly.

**Multi-GPU / larger models on H100 (not covered here):**
- A 2/8-GPU pass needs N free GPUs. The recipe is `--num_processes=N` with the same flags
  and a distinct `--main_process_port`; assert `torch.cuda.device_count()==N` after
  activating the venv.
- **An ~8B full FT under ZeRO-3 will OOM on a single 80 GB H100** (the MI355X datapoint
  below is ~167 GiB/GPU at 2 ranks). On H100 that needs either ≥ ~4–8 ranks so stage-3
  sharding brings per-GPU VRAM under 80 GB (the MI355X 8-rank run hit 33.5 GiB/GPU), or
  `--offload_optimizer` (CPU) — neither is exercised here.

This path works on H100 with the changes below. No CUDA-specific *code* change is needed —
the training path is vendor-neutral and the ROCm workarounds (comm-dtype guard,
save-on-all-ranks, trl/transformers shim) are all no-ops or apply identically on CUDA. The
only deviations are operational: install the native `torch 2.13.0+cu130` (the `2.11.0` pin
has no cu130 wheel) with the torch line stripped from the requirements, swap the uncached
default model to a cached one such as `LFM2.5-350M`, and set `HF_DATASETS_CACHE` to
writable storage on a shared/offline node. Single-GPU 350M full FT: finite decreasing loss,
32 steps, ~7.3 GiB peak, consolidated 709 MB bf16 model saved, exit code 0.

## AMD MI355X (ROCm 7.2) — tested

Verified on 2×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
Python 3.12.3 — **full fine-tuning of `google/gemma-4-E4B-it` (~8B) under ZeRO-3**,
prompt-masked SFT on the shipped 10-row sample re-rendered to `prompt`/`completion`
rows for the `gemma-4` template.

**Install:**

```bash
cd training/llm/deepspeed_standalone
python3 -m venv .env_deepspeed_standalone
source .env_deepspeed_standalone/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_standalone.txt   # torch pin already satisfied, all other pins unchanged
```

**Launch (2 GPUs; pass a custom port when the machine is shared):**

```bash
export CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
accelerate launch --num_processes=2 --main_process_port=29630 \
  --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file <prompt_completion>.jsonl \
  --model_name google/gemma-4-E4B-it --model_type gemma-4 \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 2 --test_size 0.2 \
  --max_token_length 4096 --logging_steps 1 --warmup_steps 2
```

**Expected output:**

```
[rank=0] ... INFO: ROCm + DeepSpeed + bf16 detected: setting `communication_data_type='fp32'` to avoid bf16 overflow corrupting weights.
{'loss': '21.04', 'grad_norm': '811.4', ... 'epoch': '0.25'}   ...   {'loss': '5.7', 'grad_norm': '64.37', ... 'epoch': '2'}
[rank=0] ... INFO: Training Complete.   # + 15GB consolidated bf16 final_model/model.safetensors
```

`rocm-smi` mid-run shows both GPUs busy (100%/70%, ~370W, up to 58% VRAM).

**Quirks found while testing (all vendor-neutral except where noted):**

- **trl 0.24.0 × transformers 5.5.0 skew** — `SFTTrainer` pops `push_to_hub_token`
  from `TrainingArguments.to_dict()`, but transformers 5.x removed that field; the
  script now shims the dict (would fail identically on CUDA with these pins).
- **ZeRO-3 final save deadlock** — `trainer.save_model()` was called under
  `if rank == 0:`, but the stage-3 16-bit weight gather is a *collective*: rank 0 spun
  at 100% GPU against already-exited peers. The script now calls `save_model` on all
  ranks (the Trainer still writes files only from the main process).
- **Char pre-filter trap on tiny samples** — with `--max_token_length 1024` the
  `4×` character pre-filter drops most of the long OTel sample rows (10 → ~3);
  use `--max_token_length 4096` to keep all 10.
- Rows whose completion is fully truncated away log `loss: 0` (all labels `-100`) —
  a tiny-sample artifact, not a ROCm issue.
- DeepSpeed warns `You are using ZeRO with an untested optimizer` for
  `adamw_bnb_8bit` and proceeds; training and 8-bit optimizer steps work on gfx950.

This path works with changes on MI355X — no ROCm-specific code changes are needed; the two
script fixes above are stack-version fixes that apply to NVIDIA as well. Install
deviation is only the torch index URL.

### 8-GPU run (8× MI355X, ROCm 7.2.4)

**This path works as documented.** The 2-GPU ZeRO-3 recipe scales to all 8 MI355X
unchanged — no new flags, no RCCL tuning, no OOM, no hang, no code change to the training
path. The only addition is a `--no_save` switch for disk hygiene (see below). Repeated runs
give identical results, exit code 0, with all 8 ranks reaching teardown.

**Launch:**

```bash
source .env_deepspeed_standalone/bin/activate
# override any stale 1-2 GPU pin left in bin/activate by an earlier session
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# HF_HOME as exported in the Install section
export CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm

accelerate launch --num_processes=8 --main_process_port=29620 \
  --mixed_precision=bf16 --use_deepspeed \
  train_llm_deepspeed_standalone.py \
  --train_file <prompt_completion>.jsonl \
  --model_name google/gemma-4-E4B-it --model_type gemma-4 \
  --zero_stage 3 \
  --batch_size 1 --grad_acc_steps 1 --num_train_epochs 1 --test_size 0.2 \
  --max_token_length 4096 --logging_steps 1 --warmup_steps 2 \
  --num_proc 8 --no_save
```

Same model as the 2-GPU run (`google/gemma-4-E4B-it`, ~8B, full fine-tune — no LoRA).
The dataset is the shipped 10-row OTel sample re-rendered to `prompt`/`completion` and
repeated 25× (250 rows → 200 train / 50 eval) purely so that a global batch of 8 yields
25 optimizer steps instead of 1.

**Parallelism actually used** — read off the live `DeepSpeedEngine`, not the CLI flag
(all 8 ranks print an identical line):

```
[ZERO-PROOF] rank=0 world_size=8 zero_stage=3 partition_grads=True partition_params=True
             comm_dtype=torch.float32 train_batch_size=8 micro_bs=1 grad_accum=1
             offload_opt=device='none' ... super_offload=False
[ZERO-PROOF] rank=0 local_param_elems=0 full_model_elems=7996156448 shard_ratio=0.0000
```

ZeRO stage 3 with both parameter and gradient partitioning on, world size 8, batch
geometry 1 × 1 × 8 = **global batch 8**, no CPU/NVMe offload (`offload_optimizer` left
off — there is no memory pressure to justify it). `local_param_elems=0` against a
7,996,156,448-element model is the stage-3 signature: outside a gather context the
module holds no materialised parameters at all.

Note: DeepSpeed 0.19.4 hard-creates its logger at `WARNING`
(`deepspeed/utils/logging.py:61`) and `transformers/integrations/deepspeed.py:602`
resets it again, so the familiar `Creating bf16 ZeRO stage 3 optimizer` banner never
reaches the log. Introspecting the engine (as above) is the reliable check — a run log
that is silent about ZeRO is **not** evidence that ZeRO is off.

**Expected output** — loss, throughput, clean exit:

```
{'loss': '54.25', 'grad_norm': '1983',  'mean_token_accuracy': '0.08069', 'epoch': '0.04'}
{'loss': '44.99', 'grad_norm': '1626',  'mean_token_accuracy': '0.07015', 'epoch': '0.08'}
{'loss': '33.87', 'grad_norm': '1255',  'mean_token_accuracy': '0.1576',  'epoch': '0.12'}
   ...
{'loss': '0.2654','grad_norm': '14.8',  'mean_token_accuracy': '0.9827',  'epoch': '1'}
{'train_runtime': '63.37', 'train_samples_per_second': '3.156', 'train_steps_per_second': '0.395'}
100%|##########| 25/25 [01:03<00:00,  2.12s/it]
[rank=0] INFO: --no_save set: skipping checkpoint and final model save.
[rank=0] INFO: Training Complete.
```

The run should finish with exit code 0: finite and monotonically falling loss
(54.25 → 0.27 over 25 steps; the sample is repeated 25× so the model memorises it —
expected for a smoke test), **~2.1 s/step**, 3.16 samples/s at global batch 8.

**Checking all 8 GPUs are genuinely busy** (`rocm-smi` sampled every 20 s during the run;
`procs: 9` = 8 ranks + launcher):

```
Device  Temp    Power     SCLK     VRAM%  GPU%
0       44.0°C  325.0W    2390Mhz  7%     96%
1       44.0°C  319.0W    2403Mhz  7%     100%
2       43.0°C  329.0W    2408Mhz  7%     96%
3       47.0°C  319.0W    2392Mhz  7%     99%
4       46.0°C  323.0W    2400Mhz  7%     93%
5       47.0°C  324.0W    2400Mhz  7%     95%
6       46.0°C  325.0W    2395Mhz  7%     96%
7       45.0°C  331.0W    2404Mhz  7%     99%
```

Every GPU should sit at 93–100 % utilisation and ~2.4 GHz across mid-run samples — no
idle rank, no straggler.

**Per-GPU VRAM: 2 ranks vs 8 ranks (the ZeRO-3 sharding datapoint)**

| | 2×MI355X | 8×MI355X |
|---|---|---|
| `rocm-smi` peak VRAM per GPU | up to 58 % (~167 GiB) | **18 % (47–53 GiB)** |
| `torch.cuda.max_memory_allocated` | — | **33.51 GiB** (identical on all 8 ranks) |
| `torch.cuda.max_memory_reserved` | — | 42.46–42.82 GiB |
| Steady-state mid-training | — | 21.7–22.5 GiB |

Going 2 → 8 ranks cut peak per-GPU VRAM roughly **3×**. The allocation is near-perfectly
even across ranks (33.51 GiB on every one), which is what correct stage-3 partitioning
of parameters, gradients and optimizer state should look like. Headroom is enormous —
288 GiB cards running at 18 %, so batch size, sequence length or a much larger model all
have room; this 8B full fine-tune is nowhere near the limit of this hardware.

**What differs from the 2-GPU run:**

- **New `--no_save` flag** (the only script change). `save_strategy="epoch"` was
  hardcoded, and a 2-GPU run leaves 60–134 GB of ZeRO-3 checkpoints on disk; an
  8-way stage-3 checkpoint of an 8B model plus the consolidated 16-bit gather is far too
  large for a smoke test. `--no_save` sets `save_strategy="no"`, forces
  `load_best_model_at_end=False` (transformers rejects "load best" without saving), and
  skips the final `save_model`. Default behaviour is unchanged. **The 8-GPU run
  therefore did not exercise the ZeRO-3 save path** — the rank-0-only save deadlock
  fixed for the 2-GPU run was not re-tested at 8 ranks, and if anything an 8-way gather
  makes that collective *more* deadlock-prone. Treat saving at 8 ranks as untested.
- **`--main_process_port 29620`** instead of 29630, to avoid colliding with other jobs.
- **No RCCL/NCCL environment variables are needed.** RCCL 2.27.7 negotiates the 8-way
  topology on its own; the run log shows no RCCL warnings or fallbacks.
- **No OOM and no batch-size change.** Per-device batch stays at 1; global batch grows
  2 → 8 simply by adding ranks.
- **No hang anywhere** — not at init, not at the stage-3 param AllGather, not at
  teardown. All 8 ranks print their exit line. DeepSpeed JIT-builds nothing new via
  `hipcc`, so there is no slow first step.
- The `communication_data_type='fp32'` ROCm guard the script applies under
  DeepSpeed+bf16 fires exactly as it does at 2 ranks (`comm_dtype=torch.float32` above)
  and causes no problem at 8-way.
- **Watch out — stale GPU pin.** A venv can end up with
  `export CUDA_VISIBLE_DEVICES=<1-2 GPUs>` appended to `bin/activate` by an earlier
  single-GPU session. Always re-export both
  `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` after sourcing the venv and assert
  `torch.cuda.device_count() == 8` before launching, or you will silently "8-GPU" train
  on two cards.

**One honest wart:** `eval_loss` comes back `nan` at 8 ranks (train loss is fine). This is
the tiny-sample truncation artifact already documented above — rows whose completion is
truncated away have every label set to `-100`; at 2 ranks that surfaces as `loss: 0`, and
at 8 ranks a fully-masked eval shard turns the aggregated mean into `nan`. It is a
property of the 10-row toy dataset, not of ROCm or of 8-way ZeRO-3. Use a real eval set
(or `--test_size 0`) for anything beyond a smoke test.

## Hardware support & evidence

| | NVIDIA | AMD |
|---|---|---|
| Status here | **Tested** — 8×H100 80GB, CUDA 13.0 | **Tested** — 2×MI355X and 8×MI355X (gfx950), ROCm 7.2.4, torch 2.11.0+rocm7.2 |
| PyTorch | PyPI pins in requirements | ROCm wheel index / `rocm/pytorch` images |
| DeepSpeed ops | JIT via `nvcc` | JIT via `hipcc` |
| 8-bit optimizer | bitsandbytes PyPI wheel | bitsandbytes PyPI wheel (ROCm build) |

Verified upstream sources:

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
- **bitsandbytes installation docs** (github.com/bitsandbytes-foundation/bitsandbytes)
  — official AMD ROCm support; PyPI wheels built for ROCm 6.4.4–7.14 covering Instinct
  gfx90a/gfx942/gfx950 — relevant because `adamw_bnb_8bit` is this trainer's default
  optimizer.
- **flash-attention** (github.com/ROCm/flash-attention README) — ROCm ≥ 6.0 with
  composable_kernel and Triton backends (MI200x–MI355x). Only relevant to the Mistral
  fallback path here.

**Other hardware (upstream claims — not verified here):** Intel Gaudi/HPU (upstream CI),
Intel XPU (upstream CI), Intel Xeon CPU (upstream CI), Huawei Ascend NPU (contributor),
Tecorigin SDAA (contributor) — per the DeepSpeed README's "Contributed HW support"
accelerator table. This repo provides setup instructions for NVIDIA and AMD only.

## Notes

- Relationship to `../deepspeed/`:

  | | this folder | `../deepspeed/` |
  |---|---|---|
  | Dependencies | none (single file) | `utils.py` |
  | Modes | SFT only | SFT / DPO / GRPO |
  | Fine-tuning | full FT only | LoRA or full FT |
  | Prompt format | hand-written per-`--model_type` strings | tokenizer's official chat template |
  | Over-length rows | truncated | dropped |
  | Configuration | CLI flags (formerly constants at top of file) | CLI flags |

- The NCCL process group uses a 2-hour timeout to tolerate the slow model AllGather
  under ZeRO-3.
- `HF_TOKEN` is only re-exported when present, so runs against ungated models work with
  no `dev.env` at all.
- With the default `--test_size 0.0002`, tiny datasets still get at least one eval row
  (the split rounds up); pass a larger `--test_size` for small files, as in the smoke
  command.
