# `training/llm/composer` — LLM fine-tuning with MosaicML Composer

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[MosaicML Composer](https://github.com/mosaicml/composer). Efficiency methods (gradient
clipping, low-precision LayerNorm, sequence-length warmup) are `Algorithm` objects handed
to the Trainer; sharding is PyTorch FSDP through `parallelism_config`, plus
auto-microbatching that finds the largest microbatch that fits.

Pick this recipe for Composer's Algorithm/Time API and auto-microbatching. The cost is
packaging: Composer pins `torch<2.7.1`, which forces a torch downgrade on CUDA and an
explicit override on ROCm - see Setup.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 1 and 8 GPUs, DDP and FSDP `FULL_SHARD` -
NVIDIA H100 80GB (CUDA 13.0, driver 580), 1 GPU with `--no_fsdp`.

## Files

| File | Purpose |
|---|---|
| `train_llm_composer.py` | `HuggingFaceModel` + `composer.Trainer`, chat-JSONL pipeline, FSDP wrap tagging |
| `requirements_composer.txt` | Dependencies (PyPI distribution `mosaicml`, import root `composer`) |
| `data/OTel_LLM_sample_10.jsonl` | 10-row chat sample; the default `--train_file` |

## Setup

The PyPI distribution is **`mosaicml`** and it imports as **`composer`**.
`pip install composer` fetches an unrelated project.

### NVIDIA (CUDA 13)

No override is needed. Let `mosaicml` resolve torch: the `torch<2.7.1` pin lands on
`torch 2.7.0+cu126`, which runs on a CUDA-13 / driver-580 host via driver
backward-compatibility, so `pip check` stays clean. Installing `mosaicml` downgrades torch
from `2.13.0+cu130` to `2.7.0+cu126` - that is expected; accept it and re-verify rather
than override it.

```bash
python3 -m venv .env_composer && source .env_composer/bin/activate
pip install torch numpy                 # base check: the current CUDA 13 build
pip install "mosaicml>=0.27.0"          # resolves the pin -> reinstalls torch 2.7.0+cu126
pip install "transformers>=4.51.0" tokenizers torchmetrics python-dotenv huggingface_hub

CUDA_VISIBLE_DEVICES=<your_gpu> python -c "import torch, composer; from composer.utils import get_device; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(type(get_device(None)).__name__)"
# -> 2.7.0+cu126 12.6 True
# -> DeviceGPU
```

For native CUDA-13 torch instead, reinstall it *after* mosaicml with
`pip install --force-reinstall --no-deps torch==2.13.0`; `pip check` then reports the
`torch<2.7.1` pin as violated, which Composer 0.32.1 tolerates. Do not blind-uninstall the
CUDA `triton` 3.3.0 that ships with torch 2.7.0. If a sibling install drags in
`kernels 0.16.0` (which breaks all `transformers` imports), pin `kernels>=0.12,<0.13`.

Upstream's pre-built images (`mosaicml/pytorch:2.7.0_cu128-python3.12-ubuntu22.04` and
friends) are an alternative if you want flash-attn without a build step.

### AMD / ROCm 7.2

Composer ships no ROCm install path but runs on ROCm unchanged - every device call goes
through `torch.cuda.*`. The obstacle is packaging: `mosaicml` pins `torch<2.7.1` and the
ROCm 7.2 index only publishes torch 2.11/2.12/2.13, so the pin is unsatisfiable and must be
overridden. Do not `pip install -r requirements_composer.txt` on ROCm - pip would resolve
the pin and install CUDA torch. Run these steps in this exact order:

```bash
python3 -m venv .env_composer && source .env_composer/bin/activate

# 1. Composer first. This pulls CUDA torch 2.7.0 + ~3GB of nvidia-* wheels; step 2
#    overwrites them.
pip install "mosaicml>=0.27.0" python-dotenv
pip install transformers tokenizers huggingface_hub

# 2. Force the ROCm build over the CUDA one. --no-deps is REQUIRED, otherwise pip
#    re-resolves the torch<2.7.1 pin and reinstalls the CUDA wheel.
pip install --force-reinstall --no-deps \
  torch==2.13.0+rocm7.2 torchvision==0.28.0+rocm7.2 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# 3. Drop the now-dead CUDA runtime wheels (~3GB). Do NOT blind-uninstall `triton`.
pip uninstall -y $(pip list --format=freeze | grep -oE "^nvidia-[a-z0-9-]+" | tr '\n' ' ')

# 4. Verify: must print a ROCm build and DeviceGPU.
python -c "import torch, composer; from composer.utils import get_device; \
print(torch.__version__, torch.version.hip, torch.cuda.is_available()); \
print(type(get_device(None)).__name__)"
# -> 2.13.0+rocm7.2 7.2.53211 True
# -> DeviceGPU
```

`pip check` will now permanently report two violations. Expected and safe to ignore:

```
mosaicml 0.32.1 has requirement torch<2.7.1,>=2.6.0, but you have torch 2.13.0+rocm7.2.
mosaicml 0.32.1 has requirement torchvision<0.22.1,>=0.21.0, but you have torchvision 0.28.0+rocm7.2.
```

**The triton package conflict.** The CUDA `triton` wheel installs into the same `triton/`
package directory that ROCm's `triton-rocm` owns, so step 1 silently clobbers it. If you
then `pip uninstall triton` while purging CUDA packages you delete ROCm's copy too and
every `import torch` dies with `AttributeError: module 'triton' has no attribute
'language'`. Recover with the exact pin torch asks for - note it is `triton-rocm`, not
`pytorch-triton-rocm`:

```bash
pip install --force-reinstall --no-deps triton-rocm==3.7.1 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

Do not `pip install flash-attn` on ROCm; `--attn_implementation sdpa` (the default) needs
no build.

### Secrets

`HF_TOKEN` for gated checkpoints comes from `dev.env` in this folder, loaded by
`load_dotenv("dev.env")` and passed explicitly to `AutoTokenizer` /
`AutoModelForCausalLM`. The `composer` launcher inherits your shell environment, so the
token reaches every rank with no extra plumbing.

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

## Data

Chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Only the `"messages"` key is read, so extra metadata columns are ignored. Rows are
rendered with `tokenizer.apply_chat_template`, so training matches inference. Prompt
masking is on by default (loss on assistant turns only); pass `--no_mask_prompt` for
continued pre-training. Rows longer than `--max_seq_len` and rows with no assistant turn
are dropped rather than truncated; both counts are logged.

To train on your own data, point `--train_file` at any JSONL in this schema. Read the
*survivor* count in the log, not your input count.

## Run

Launch with the `composer` launcher, not `torchrun`. It sets the `torch.distributed`
environment variables and spawns one process per device; on a single node it autodetects
the device count.

Smoke test - the shipped sample, one GPU:

```bash
export HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3      # never leave these empty on ROCm

composer -n 1 --master_port 29710 train_llm_composer.py \
  --model_name Qwen/Qwen3-0.6B \
  --max_samples 8 --max_duration 40ba \
  --global_train_batch_size 2 --device_train_microbatch_size 1 \
  --max_seq_len 2048 --no_fsdp \
  --save_folder ./composer_run --save_interval 1000ba
```

Full run, 8 GPUs:

```bash
source .env_composer/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # NVIDIA, or the subset you own
python -c "import torch,sys; sys.exit(torch.cuda.device_count()!=8)" || exit 1

nohup composer -n 8 --master_port 29740 train_llm_composer.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --save_folder ./composer_run \
  --max_duration 3ep \
  --global_train_batch_size 64 \
  --device_train_microbatch_size 2 \
  --max_seq_len 4096 \
  --learning_rate 1e-5 \
  --sharding_strategy FULL_SHARD --fsdp_mixed_precision PURE \
  --activation_checkpointing \
  --low_precision_layernorm \
  > train_llm_composer.log 2>&1 &

tail -f train_llm_composer.log
```

Add `--no_fsdp` for the DDP variant (and at world size 1, where FSDP is a no-op).

- **`--global_train_batch_size` must be divisible by the world size**; the per-device
  minibatch is that quotient.
- **Pin `--device_train_microbatch_size` to an integer at 8 ranks.** `auto` discovers the
  microbatch by catching OOMs, which races across processes and can leave the device in an
  irrecoverable state on long unattended runs.
- Re-export both GPU masks after activating the venv and assert the device count; a stale
  single-GPU export in `bin/activate` makes `composer` autodetect one device while you
  believe you ran on eight.
- Size the dataset to the world size: the shipped 10-row sample cannot feed 8 ranks.
  Replicate it (e.g. x64 -> 640 rows) outside the repo and point `--train_file` there.
- Use a distinct `--master_port` on a shared box.

To keep all ranks' logs, not just rank zero's:

```bash
composer -n 8 --stdout stdout_{rank}.log --stderr stderr_{rank}.log train_llm_composer.py ...
```

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL, one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--save_folder` | `./composer_run` | Where Composer `.pt` checkpoints are written |
| `--save_interval` | `1ep` | Checkpoint cadence as a Time string (`1ep`, `500ba`) |
| `--load_path` | (empty) | Composer checkpoint to resume from (ignored if missing) |
| `--run_name` | `composer-sft` | Run name in logs and checkpoints |
| `--max_seq_len` | `4096` | Max tokens per example; longer rows are dropped |
| `--max_samples` | none | Hard cap on rows loaded |
| `--eval_samples` | `0` | Rows held out for eval (0 = no eval loop) |
| `--mask_prompt` / `--no_mask_prompt` | on | Assistant-only loss vs full-sequence loss |
| `--num_workers` | `4` | DataLoader workers per rank |
| `--global_train_batch_size` | `64` | Batch across ALL ranks; per-device minibatch derives from it |
| `--device_train_microbatch_size` | `auto` | Per-device microbatch, or `auto` to find the largest that fits |
| `--max_duration` | `3ep` | Training length as a Time string (`3ep`, `2000ba`, `10000000tok`) |
| `--learning_rate` | `1e-5` | `DecoupledAdamW` peak LR |
| `--weight_decay` | `0.0` | `DecoupledAdamW` weight decay |
| `--t_warmup` | `0.03dur` | Cosine warmup: fraction of training (`0.03dur`) or a Time string |
| `--alpha_f` | `0.1` | Final LR as a fraction of peak |
| `--grad_clip` | `1.0` | Global grad-norm clip via the `GradientClipping` algorithm; 0 disables |
| `--seed` | `42` | Seed for `reproducibility.seed_all` |
| `--attn_implementation` | `sdpa` | `sdpa`, `flash_attention_2`, `eager` |
| `--no_fsdp` | off | Fall back to DDP (single GPU or small models) |
| `--sharding_strategy` | `FULL_SHARD` | `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` |
| `--fsdp_mixed_precision` | `PURE` | `PURE`, `DEFAULT`, `FULL` - see Notes |
| `--state_dict_type` | `full` | `full` (one file) or `sharded` (one file per rank, elastic resume) |
| `--activation_checkpointing` | off | Recompute decoder-layer activations |
| `--low_precision_layernorm` | off | Algorithm: keep LayerNorm in the autocast dtype |
| `--seq_length_warmup` | off | Algorithm: ramp sequence length over the first 30% of training |

## Output

```
composer_run/
  ep0-ba500-rank0.pt        # Composer checkpoints (model + optimizer + schedule + RNG)
  latest-rank0.pt
```

With `--state_dict_type sharded` you get a directory per checkpoint event holding a
`.metadata` file and one `.distcp` per rank - the format that supports elastic resume (save
on 8, resume on 16). `full` gives a single consolidated file that is easier to copy.

Resume with `--load_path composer_run/latest-rank0.pt`.

**A Composer checkpoint is not a Hugging Face model** - `from_pretrained` cannot read a
`.pt`. Two ways across:

```python
from composer.models import HuggingFaceModel

model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
    "composer_run/latest-rank0.pt",
)
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
```

```bash
python scripts/inference/convert_composer_to_hf.py \
  --composer_path composer_run/latest-rank0.pt \
  --hf_output_path ./final_model \
  --output_precision bf16
```

Console output carries loss, `LanguageCrossEntropy`, `Perplexity`, learning rate, memory
and throughput; add `composer.loggers.WandBLogger` or `MLFlowLogger` to the Trainer's
`loggers=` to send them elsewhere.

## Notes

- **Composer writes a fit-end checkpoint regardless of `--save_interval`**, even on a
  40-batch run. Point `--save_folder` at scratch for smoke runs and clean up.
- `google/gemma-3-1b-it` fails at the `HuggingFaceModel` constructor with "The number of
  tokens in the tokenizer is greater than the number of tokens in the model" (262145 vs
  262144) - Composer's strict embedding check meeting Gemma-3's off-by-one vocab, on any
  hardware. Use a model without the mismatch (Qwen3-0.6B is clean), or add
  `allow_embedding_resizing=True` to the `HuggingFaceModel` call in the script.
- **Time is a first-class type.** `3ep`, `2000ba`, `500sp`, `10000000tok` are valid for
  `--max_duration`, `--save_interval` and warmup. `0.03dur` means "3% of total training",
  which keeps warmup correct when you change the duration.
- `--global_train_batch_size` is what the optimizer sees; `--device_train_microbatch_size`
  is what fits on a card. Composer derives gradient accumulation from the two plus the
  world size.
- **`--fsdp_mixed_precision`:** `DEFAULT` all-reduces gradients in fp32, `PURE` (the
  default) in bf16, `FULL` keeps everything fp32. Drop to `DEFAULT` if you see loss
  instability.
- **`--seq_length_warmup` caveat.** It truncates every tensor in the batch, labels
  included, to a sequence length that starts small, so with prompt masking on, early
  batches can end up with no supervised tokens if your prompts are long. It suits
  continued pre-training (`--no_mask_prompt`) better than short-answer SFT.
- Log lines mentioning "NCCL" / "ProcessGroupNCCL" are RCCL on ROCm - cosmetic.
- **OOM playbook,** in order: `--device_train_microbatch_size auto` (if you had pinned it);
  `--activation_checkpointing`; lower `--max_seq_len`; `--fsdp_mixed_precision PURE`; then
  raise `--global_train_batch_size` only once the per-device footprint is under control.
