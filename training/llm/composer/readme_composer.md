# `training/llm/composer` — LLM fine-tuning with MosaicML Composer

## 1. Overview & when to use

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[MosaicML Composer](https://github.com/mosaicml/composer). Efficiency methods (gradient
clipping, low-precision LayerNorm, sequence-length warmup, ...) are **Algorithm** objects you
hand to the Trainer. Sharding is PyTorch FSDP configured through `parallelism_config`, plus
auto-microbatching that finds the largest microbatch that fits.

Files in this folder:
- `train_llm_composer.py` — the trainer: `HuggingFaceModel` + `composer.Trainer`, chat-JSONL pipeline, FSDP wrap tagging.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample so the folder is self-contained.
- `requirements_composer.txt` — deps (note: the PyPI package is `mosaicml`, the import is `composer`).
- `readme_composer.md` — this file.

> **Hardware coverage:**
>
> - **NVIDIA H100 80GB (CUDA 13.0, driver 580) — works, 1 GPU** (`--no_fsdp`, DDP degenerate
>   at world size 1): finite decreasing loss, checkpoint written, work resident on the
>   assigned GPU. **No packaging override is required.** The documented `pip install mosaicml`
>   path resolves Composer's `torch<2.7.1` pin to **torch 2.7.0+cu126**, a cu12.6 build that
>   runs on a CUDA-13 host via driver backward-compatibility, so `pip check` stays clean.
>   Multi-GPU on NVIDIA is not covered here.
> - **AMD MI355X (ROCm 7.2.4) — works, 1 and 8 GPUs**, both DDP and FSDP `FULL_SHARD` at world
>   size 8 via the stock `composer -n 8` launcher, despite upstream's "AMD + RoCM coming soon"
>   claim. Composer's device detection handles ROCm-as-CUDA transparently. **One packaging
>   override is required** (the `torch<2.7.1` pin is unsatisfiable on the ROCm 7.2 wheel
>   index); no source change and no extra package is needed to go from 1 GPU to 8.
>
> Exact commands are in [section 2](#2-install); per-platform detail and caveats in
> [section 8](#8-hardware-support). Run the smoke command in section 5 first.

## 2. Install

The PyPI distribution is **`mosaicml`** and it imports as **`composer`**. `pip install
composer` fetches an unrelated project.

### NVIDIA (CUDA)

```bash
python -m venv composer-env && source composer-env/bin/activate

# torch first, matched to your CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements_composer.txt
```

**On CUDA 13 (driver 580) no override is needed.** Let `mosaicml` resolve torch from the
default index: the `torch<2.7.1` pin lands on **torch 2.7.0+cu126**, which runs on a CUDA-13 /
driver-580 host via driver backward-compatibility. The pin is *satisfied*, so `pip check` is
clean and no `--force-reinstall --no-deps` dance is required:

```bash
python -m venv .env_composer && source .env_composer/bin/activate
pip install torch numpy                 # base check: gives the current CUDA 13 build on a CUDA-13 host
pip install "mosaicml>=0.27.0"          # RESOLVES pin -> reinstalls torch 2.7.0+cu126 (expected)
pip install "transformers>=4.51.0" tokenizers torchmetrics python-dotenv huggingface_hub
# Re-verify torch was reconciled to a build mosaicml accepts AND that runs here:
CUDA_VISIBLE_DEVICES=<your_gpu> python -c "import torch, composer; from composer.utils import get_device; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
a=torch.randn(1024,1024,dtype=torch.bfloat16,device='cuda'); print('bf16 matmul', float((a@a).float().sum())); \
print(type(get_device(None)).__name__)"
# -> 2.7.0+cu126 12.6 True
# -> bf16 matmul <finite>
# -> DeviceGPU
```

`pip install mosaicml` **silently downgrades** torch from `2.13.0+cu130` to `2.7.0+cu126`.
That is expected — accept mosaicml's torch and re-verify it rather than override it.

If you want native-CUDA-13 torch instead, reinstall it *after* mosaicml with
`pip install --force-reinstall --no-deps torch==2.13.0`; `pip check` will then report the
`torch<2.7.1` pin as violated, which Composer 0.32.1 tolerates. Note the **triton package
conflict** below: torch 2.7.0 ships CUDA `triton` 3.3.0; do not blind-uninstall it. If a sibling
install drags in `kernels 0.16.0` (which breaks all `transformers` imports), pin
`kernels>=0.12,<0.13`.

Upstream's pre-built Docker images
(`mosaicml/pytorch:2.7.0_cu128-python3.12-ubuntu22.04` and friends) are an alternative if you
want flash-attn without a build step.

### AMD / ROCm (MI355X / gfx950, ROCm 7.2.4)

Composer ships no ROCm install path, but it works: every device call goes through
`torch.cuda.*`, which ROCm implements.

The one obstacle is packaging, not code: **`mosaicml` pins `torch<2.7.1`, and the ROCm 7.2
wheel index only publishes torch 2.11/2.12/2.13.** The pin is therefore *unsatisfiable* on
ROCm 7.2 and must be overridden. Install Composer first (it will drag in the CUDA torch), then
force the ROCm wheels back over the top:

```bash
python -m venv .env_composer && source .env_composer/bin/activate

# 1. Composer first. This pulls CUDA torch 2.7.0 + ~3GB of nvidia-* wheels.
#    Let it happen; step 2 overwrites them.
pip install "mosaicml>=0.27.0" python-dotenv
pip install transformers tokenizers huggingface_hub

# 2. Force the ROCm build back over the CUDA one. --no-deps is REQUIRED, otherwise
#    pip re-resolves mosaicml's torch<2.7.1 pin and reinstalls the CUDA wheel.
pip install --force-reinstall --no-deps \
  torch==2.13.0+rocm7.2 torchvision==0.28.0+rocm7.2 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# 3. Drop the now-dead CUDA runtime wheels (~3GB). Do NOT blind-uninstall `triton`
#    (see the triton package conflict below).
pip uninstall -y $(pip list --format=freeze | grep -oE "^nvidia-[a-z0-9-]+" | tr '\n' ' ')

# 4. Verify: must print a ROCm build and DeviceGPU.
python -c "import torch, composer; from composer.utils import get_device; \
print(torch.__version__, torch.version.hip, torch.cuda.is_available()); \
print(type(get_device(None)).__name__)"
# -> 2.13.0+rocm7.2 7.2.53211 True
# -> DeviceGPU
```

`pip check` will now permanently report the two pins as violated. **This is expected and
safe to ignore** — nothing in Composer 0.32.1 actually uses a torch API that changed:

```
mosaicml 0.32.1 has requirement torch<2.7.1,>=2.6.0, but you have torch 2.13.0+rocm7.2.
mosaicml 0.32.1 has requirement torchvision<0.22.1,>=0.21.0, but you have torchvision 0.28.0+rocm7.2.
```

**The triton package conflict.** The CUDA `triton` wheel installs into the same `triton/` package
directory that ROCm's `triton-rocm` owns, so step 1 silently clobbers it. If you then
`pip uninstall triton` while purging CUDA packages, you delete ROCm's copy too and every
`import torch` dies with `AttributeError: module 'triton' has no attribute 'language'`
from `torch._dynamo`. Recover with the exact pin torch asks for — note it is
`triton-rocm`, **not** `pytorch-triton-rocm`:

```bash
pip install --force-reinstall --no-deps triton-rocm==3.7.1 \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

Do **not** `pip install flash-attn` on ROCm; `--attn_implementation sdpa` (the default)
needs no build and works.

The run commands below refer to an output directory and the Hugging Face cache by
environment variable — set them to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints and generated training data
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

## 3. Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_composer.py` calls `load_dotenv("dev.env")` at startup and passes the token
explicitly to `AutoTokenizer` and `AutoModelForCausalLM`. That covers gated checkpoints
such as Llama or Gemma.

`dev.env` is git-ignored at the repo root. **Never commit a token.**

The `composer` launcher inherits your shell environment, so the token reaches every rank
without extra plumbing.

## 4. Data

Chat JSONL, one conversation per line — the same format as the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The folder ships a 10-row sample at `data/OTel_LLM_sample_10.jsonl` (the default
`--train_file`), so the smoke run below works with no data prep. The loader takes only the
`"messages"` key from each line, so extra metadata columns are ignored.

Rows are rendered with `tokenizer.apply_chat_template`, so training matches inference.
Prompt masking is on by default (loss on assistant turns only); pass `--no_mask_prompt`
for continued pre-training, where you want to model the whole sequence. Rows longer than
`--max_seq_len` are dropped rather than truncated, and rows with no assistant turn are
dropped; both counts are logged.

To train on your own data, point `--train_file` at any JSONL in this schema — nothing else
changes. One Composer-specific requirement the script handles for you: **map-style
datasets need an explicit `DistributedSampler`**. Composer raises an error rather than
silently feeding every rank the same batches, so the dataloaders are built with
`composer.utils.dist.get_sampler`.

## 5. Run

Launch with the **`composer` launcher**, not `torchrun`. It sets the `torch.distributed`
environment variables and spawns one process per device; on a single node it autodetects
the device count, so `-n 8` is belt-and-braces.

Smoke test first — this uses the shipped sample by default:

```bash
composer -n 8 train_llm_composer.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --max_samples 200 --max_duration 20ba
```

Then the real run:

```bash
nohup composer -n 8 train_llm_composer.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --save_folder ./composer_run \
  --max_duration 3ep \
  --global_train_batch_size 64 \
  --device_train_microbatch_size auto \
  --max_seq_len 4096 \
  --learning_rate 1e-5 \
  --activation_checkpointing \
  --low_precision_layernorm \
  > train_llm_composer.log 2>&1 &

tail -f train_llm_composer.log
```

By default only rank-zero logs reach the console. To keep all of them (worth it the first
time you debug a hang):

```bash
composer -n 8 --stdout stdout_{rank}.log --stderr stderr_{rank}.log train_llm_composer.py ...
```

**What "working" looks like:** the loader reports kept/dropped row counts; a line confirms
how many decoder blocks were tagged for FSDP wrapping (**if this says 0, sharding is not
happening**); with `--device_train_microbatch_size auto` Composer may try a microbatch, catch
an OOM, halve it and continue — that is the feature working, not a failure. Then per-batch
console lines with `loss/train/total`, `LanguageCrossEntropy` and `Perplexity`. Memory should
be roughly even across all GPUs.

## 6. Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL, one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--save_folder` | `./composer_run` | Where Composer `.pt` checkpoints are written |
| `--save_interval` | `1ep` | Checkpoint cadence as a Time string: `1ep`, `500ba` |
| `--load_path` | (empty) | Composer checkpoint to resume from (ignored if missing) |
| `--run_name` | `composer-sft` | Run name in logs and checkpoints |
| `--max_seq_len` | `4096` | Max tokens per example; longer rows are dropped, never truncated |
| `--max_samples` | none | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for eval (0 = no eval loop) |
| `--mask_prompt` | on | Completion-only loss: supervise assistant turns only |
| `--no_mask_prompt` | — | Train on the full rendered sequence (continued pre-training style) |
| `--num_workers` | `4` | DataLoader workers per rank |
| `--global_train_batch_size` | `64` | Batch across ALL ranks; per-device minibatch is derived from it |
| `--device_train_microbatch_size` | `auto` | Per-device microbatch, or `auto` to let Composer find the largest that fits |
| `--max_duration` | `3ep` | Training length as a Time string: `3ep`, `2000ba`, `10000000tok` |
| `--learning_rate` | `1e-5` | `DecoupledAdamW` peak LR |
| `--weight_decay` | `0.0` | `DecoupledAdamW` weight decay |
| `--t_warmup` | `0.03dur` | Cosine warmup: fraction of training (`0.03dur`) or a Time string (`100ba`) |
| `--alpha_f` | `0.1` | Final LR as a fraction of peak |
| `--grad_clip` | `1.0` | Global grad-norm clip via the `GradientClipping` algorithm; 0 disables |
| `--seed` | `42` | Seed for `reproducibility.seed_all` |
| `--attn_implementation` | `sdpa` | `sdpa`, `flash_attention_2`, or `eager` |
| `--no_fsdp` | off | Fall back to DDP (single GPU or small models) |
| `--sharding_strategy` | `FULL_SHARD` | `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` |
| `--fsdp_mixed_precision` | `PURE` | `PURE`, `DEFAULT`, `FULL` — see Notes |
| `--state_dict_type` | `full` | `full` (one file) or `sharded` (one file per rank, elastic resume) |
| `--activation_checkpointing` | off | Recompute decoder-layer activations to save memory |
| `--low_precision_layernorm` | off | Algorithm: keep LayerNorm in the autocast dtype |
| `--seq_length_warmup` | off | Algorithm: ramp sequence length over the first 30% of training |

## 7. Output

```
composer_run/
  ep0-ba500-rank0.pt        # Composer checkpoints (model + optimizer + schedule + RNG)
  latest-rank0.pt
```

With `--state_dict_type sharded` you instead get a directory per checkpoint event
containing a `.metadata` file and one `.distcp` per rank. That is the format that supports
elastic resume (save on 8, resume on 16); `full` gives you a single consolidated file that
is easier to copy around.

**A Composer checkpoint is not a Hugging Face model.** `from_pretrained` cannot read a
`.pt`. Two supported ways across:

```python
# 1) In Python, from a monolithic checkpoint
from composer.models import HuggingFaceModel

model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
    "composer_run/latest-rank0.pt",
)
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
```

```bash
# 2) With LLM Foundry's converter, if you have that repo checked out
python scripts/inference/convert_composer_to_hf.py \
  --composer_path composer_run/latest-rank0.pt \
  --hf_output_path ./final_model \
  --output_precision bf16
```

Resume a run with `--load_path composer_run/latest-rank0.pt`. Console output carries loss,
`LanguageCrossEntropy`, `Perplexity`, learning rate, memory and throughput; add
`composer.loggers.WandBLogger` or `MLFlowLogger` to the Trainer's `loggers=` if you want
those elsewhere.

## 8. Hardware support

| Platform | Status |
|---|---|
| NVIDIA / CUDA | **Works** — the documented path; no packaging override (H100, CUDA 13.0) |
| AMD / ROCm | **Works** at 1 and 8 GPUs (MI355X, ROCm 7.2.4) with the packaging override in section 2 |
| CPU | Runs, for debugging only |

### NVIDIA H100 (CUDA 13.0, driver 580)

**This path works as documented.** No packaging override, no source changes — see the
[NVIDIA install](#nvidia-cuda). Covered: single GPU with `--no_fsdp` (FSDP is bypassed at
world size 1). Multi-GPU on NVIDIA is not covered.

Any small model with a chat template works for the smoke; the example below uses
`LiquidAI/LFM2.5-350M` because the documented `meta-llama/Llama-3.1-8B-Instruct` is gated.
Keep `--attn_implementation sdpa`: LFM2 is a hybrid conv/attention architecture and sdpa is
the safe default.

Smoke command (offline cache, one pinned GPU, non-default master port):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1            # HF_HOME points at the model cache (set above)
export HF_DATASETS_CACHE=/dev/shm/dscache_composer   # datasets .arrow writes can fail on a network model-cache mount
export CUDA_VISIBLE_DEVICES=7                             # plain CUDA_VISIBLE_DEVICES; no HIP_* on NVIDIA

composer -n 1 --master_port 29648 train_llm_composer.py \
  --model_name LiquidAI/LFM2.5-350M \
  --max_seq_len 2048 --max_duration 40ba \
  --global_train_batch_size 2 --device_train_microbatch_size 1 \
  --learning_rate 5e-5 --no_fsdp \
  --attn_implementation sdpa \
  --run_name composer-h100-smoke \
  --save_folder /dev/shm/composer/composer_run --save_interval 1000ba
```

Expected output:

```
- INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
- INFO - __main__ - Starting training: 40ba, fsdp=False, algorithms=['GradientClipping']
[batch=1/40]:   Train loss/train/total: 2.0366    Train metrics/train/LanguagePerplexity: 6.7352
[batch=10/40]:  Train loss/train/total: 0.6773    Train metrics/train/LanguagePerplexity: 1.9369
[batch=20/40]:  Train loss/train/total: 0.0280    Train metrics/train/LanguagePerplexity: 1.0409
[batch=40/40]:  Train loss/train/total: 0.0006    Train metrics/train/LanguagePerplexity: 1.0010
- INFO - __main__ - Training complete. Composer checkpoints are in .../composer_run
### composer exit_code=0
```

**Checkpoint / cleanup.** Composer writes a **fit-end checkpoint regardless of
`--save_interval`**, even for a 40-batch run. Point `--save_folder` at scratch and clean up.

**For a multi-GPU pass on NVIDIA:** `composer -n <N>` with the same script; drop `--no_fsdp`
and add `--sharding_strategy FULL_SHARD --fsdp_mixed_precision PURE` for the FSDP variant. The
FSDP wrap path (`tag_blocks_for_fsdp`) keys on the model's `_no_split_modules`; verify it tags
more than 0 blocks before trusting a sharded run.

### AMD MI355X (ROCm 7.2)

**This path works on MI355X with one change** — the packaging override (Composer's
`torch<2.7.1` pin is unsatisfiable on the ROCm 7.2 wheel index). **No source change to
`train_llm_composer.py` is needed for ROCm.** Both **DDP** (`--no_fsdp`) and **FSDP
`FULL_SHARD`** work at 1 and 8 GPUs with no new package and no new pin — only the launcher's
`-n` and the batch geometry change.

Install: see section 2 "AMD / ROCm". Single-GPU smoke (no sharding, small model, few
batches):

```bash
source .env_composer/bin/activate
export HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3      # never leave these empty on ROCm
export $(grep -v '^#' dev.env | xargs)                   # HF_TOKEN

composer -n 1 --master_port 29710 train_llm_composer.py \
  --model_name Qwen/Qwen3-0.6B \
  --max_samples 8 --max_duration 40ba \
  --global_train_batch_size 2 --device_train_microbatch_size 1 \
  --max_seq_len 2048 --no_fsdp \
  --save_folder $OUTPUT_DIR/composer/composer_run \
  --save_interval 1000ba
```

Expected output:

```
INFO - __main__ - Loaded 8 rows from data/OTel_LLM_sample_10.jsonl (dropped 0 over 2048 tokens, 0 with no supervised tokens)
INFO - __main__ - Starting training: 40ba, fsdp=False, algorithms=['GradientClipping']
[batch=1/5]:   Train loss/train/total: 2.5845   LanguageCrossEntropy: 2.2906   LanguagePerplexity: 9.8809
[batch=5/5]:   Train loss/train/total: 0.2626   LanguageCrossEntropy: 0.2697   LanguagePerplexity: 1.3096
[batch=40/40]: Train throughput/device/tokens_per_sec: 4884.9990   memory/peak_reserved_mem: 14.7660
```

**Device detection needs nothing special on ROCm.** `composer.utils.get_device(None)` returns
`DeviceGPU`, the launcher autodetects the device count from `HIP_VISIBLE_DEVICES`, and
`precision='amp_bf16'` works unchanged. Log lines mentioning "NCCL" / "ProcessGroupNCCL" are
RCCL on ROCm — cosmetic.

**Auto-microbatching works**, but Composer's OOM-catching can leave the device in an
irrecoverable state, so pin `--device_train_microbatch_size` to an integer for long unattended
runs.

**Quirks and caveats found on ROCm:**

1. **The `torch<2.7.1` pin cannot be satisfied on ROCm 7.2** — the index carries only
   2.11.0 / 2.12.0 / 2.12.1 / 2.13.0. Overriding it is required, and `pip check` keeps
   reporting the violation. Nothing breaks as a result.
2. **The triton clobber** (see section 2) — the most likely way to break this environment.
3. **`--save_interval` does not suppress the end-of-run checkpoint.** Composer writes a full
   `.pt` even on a 5-batch run. Point `--save_folder` at scratch.
4. **Not a ROCm issue, but it will stop you:** `google/gemma-3-1b-it` fails at the
   `HuggingFaceModel` constructor with "The number of tokens in the tokenizer is greater
   than the number of tokens in the model" (262145 vs 262144). That is Composer's strict
   embedding check meeting Gemma-3's off-by-one vocab, on any hardware. Pass
   `allow_embedding_resizing=True` or use a model without the mismatch (Qwen3-0.6B is
   clean).
5. **transformers 5.x chat-template fix was required** (applies on every platform):
   `apply_chat_template(tokenize=True)` now returns a `BatchEncoding`, so the
   length-diff prompt masking silently masked everything and every row was dropped as
   "no supervised tokens". `build_example()` now passes `return_dict=False`. The
   `dropped ... 0 with no supervised tokens` line above is the fix confirming itself.

#### 8-GPU launch

Run both variants inside one `flock`-serialised session so the job owns all 8 GPUs:

```bash
source .env_composer/bin/activate
# CRITICAL: if the venv's bin/activate pins HIP/CUDA_VISIBLE_DEVICES to a single GPU (an
# easy leftover from a 1-GPU run), override it AFTER sourcing or you silently train on one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch,sys; sys.exit(torch.cuda.device_count()!=8)" || exit 1   # assert

# 1) DDP (priority run)
composer -n 8 --master_port 29740 train_llm_composer.py \
  --train_file $OUTPUT_DIR/composer/gpu8/data/OTel_sample_x64_640.jsonl \
  --model_name Qwen/Qwen3-0.6B \
  --max_seq_len 2048 --max_duration 50ba \
  --global_train_batch_size 64 --device_train_microbatch_size 2 \
  --learning_rate 1e-5 --run_name composer-gpu8-ddp \
  --save_folder $OUTPUT_DIR/composer/gpu8/ckpt_ddp --save_interval 1000ba \
  --no_fsdp

# 2) FSDP FULL_SHARD (bonus run) — same command minus --no_fsdp, plus:
#    --sharding_strategy FULL_SHARD --fsdp_mixed_precision PURE
```

**Batch geometry** (the Trainer takes a *global* batch and the script divides it by world
size before building the dataloader): **`--global_train_batch_size` must
stay divisible by the world size**, and the per-device minibatch is that quotient.

**Data.** The shipped 10-row sample cannot feed 8 ranks meaningfully; replicate it (e.g. ×64
to 640 rows) into a directory outside the repo and point `--train_file` there. Read the
*survivor* count in the log, not the input count — rows over `--max_seq_len` are dropped:

```
INFO - __main__ - Loaded 576 rows from .../OTel_sample_x64_640.jsonl (dropped 64 over 2048 tokens, 0 with no supervised tokens)
INFO - __main__ - Starting training: 50ba, fsdp=False, algorithms=['GradientClipping']
```

To confirm all ranks really landed on distinct cards, sample `rocm-smi --showpids` **in-band**
(while your ranks are live) and cross-check against `pgrep -af` in the *same* sample: exactly
N KFD processes, one GPU each, every PID a child of your `composer -n N` launcher.

**What differs from the 1-GPU run:**

1. `-n 1` → `-n 8`, a different `--master_port`, and `--no_fsdp` dropped for the FSDP run.
2. **Any stale single-GPU `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` export inside the
   venv's `bin/activate` must be overridden after sourcing.** Leave one in place and
   `composer` autodetects **one** device while you believe you ran on eight. Assert
   `torch.cuda.device_count() == 8`.
3. Batch geometry: raise `--global_train_batch_size` and pin the microbatch to an integer —
   do not use `auto` at 8 ranks, where OOM-catching races across processes.
4. Data: replicated outside the repo so each rank gets real work.
5. Checkpointing: point `--save_folder` at scratch — the end-of-run checkpoint is written
   regardless of `--save_interval`.
6. Nothing else. No new package, no new pin, no code edit — `requirements_composer.txt` is
   unchanged for the 8-GPU path.

**Still not tested anywhere here:** multi-node, `SHARD_GRAD_OP` / `HYBRID_SHARD`,
`--state_dict_type sharded` elastic resume, checkpoint resume, `SeqLengthWarmup`,
`LowPrecisionLayerNorm`, `--device_train_microbatch_size auto` at world size 8, models larger
than 0.6B, and multi-GPU on NVIDIA.

## 9. Notes

- **`parallelism_config` is the current API; `fsdp_config=` is not.** The flat
  `fsdp_config` kwarg was replaced by `parallelism_config={'fsdp': {...}, 'tp': {...}}` —
  older examples showing the flat form no longer work.
- **You must tell Composer which modules to wrap.** Its auto-wrap policy looks for
  `module._fsdp_wrap = True` or a `fsdp_wrap_fn` on the root module, and a stock HF model has
  neither, so a naive `HuggingFaceModel` + FSDP run wraps almost nothing.
  `tag_blocks_for_fsdp()` sets `_fsdp_wrap` (and `_activation_checkpointing`) on each block
  class found in HF's `_no_split_modules`.
- **`load_dotenv` runs at startup** so `HF_TOKEN` is in the environment before any Hub
  download; the token is also passed explicitly to `from_pretrained`.
- **`mixed_precision: PURE` vs `DEFAULT`.** `DEFAULT` all-reduces gradients in fp32; `PURE`
  (the default here) all-reduces in bf16, and `FULL` keeps everything fp32. Drop to `DEFAULT`
  if you see loss instability.
- **Time is a first-class type.** `3ep`, `2000ba`, `500sp`, `10000000tok` are all valid for
  `--max_duration`, `--save_interval` and warmup. `0.03dur` means "3% of total training",
  which keeps the warmup correct when you change the duration.
- **Batch knobs.** `global_train_batch_size` is what the optimizer sees;
  `device_train_microbatch_size` is what fits on a card. Composer derives gradient
  accumulation from the two plus the world size; `auto` discovers the microbatch by catching
  OOMs and retrying.
- **`SeqLengthWarmup` caveat.** It truncates every tensor in the batch, labels included, to
  a sequence length that starts small. With prompt masking on, early batches can end up with
  no supervised tokens at all if your prompts are long. It suits continued pre-training
  (`--no_mask_prompt`) better than short-answer SFT.
- **`shift_labels=True` is not optional.** Causal LMs predict token i+1 from token i.
  Composer infers this from the model class name, but the script sets it explicitly so the
  logged `LanguageCrossEntropy` cannot silently disagree with the training loss.
- **OOM playbook.** In order: `--device_train_microbatch_size auto` (if you had pinned it);
  `--activation_checkpointing`; lower `--max_seq_len`; `--fsdp_mixed_precision PURE`; then
  raise `--global_train_batch_size` only after the per-device footprint is under control.
