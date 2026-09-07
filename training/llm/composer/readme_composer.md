# `training/llm/composer` — LLM fine-tuning with MosaicML Composer

## 1. Overview & when to use

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[MosaicML Composer](https://github.com/mosaicml/composer). Composer is a PyTorch training
library whose distinguishing feature is the **Algorithm** system: speedup and efficiency
methods (gradient clipping, low-precision LayerNorm, sequence-length warmup, EMA, selective
backprop, and more) are composable objects you hand to the Trainer, rather than code you
weave into a loop. Sharding is PyTorch FSDP configured through `parallelism_config`, plus
auto-microbatching that finds the largest microbatch that fits so you stop hand-tuning
around OOMs.

Files in this folder:
- `train_llm_composer.py` — the trainer: `HuggingFaceModel` + `composer.Trainer`, chat-JSONL pipeline, FSDP wrap tagging.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample so the folder is self-contained.
- `requirements_composer.txt` — deps (note: the PyPI package is `mosaicml`, the import is `composer`).
- `readme_composer.md` — this file.

> **Tested topology:** Written against the Composer upstream README,
> `docs/source/notes/distributed_training.rst`, and the `HuggingFaceModel` /
> `composer.algorithms` / `composer.optim` sources on the `dev` branch, for the pinned
> versions below.
> Targets a single node of 8x H100 80GB (NVIDIA, CUDA).
>
> **NVIDIA H100 (CUDA 13.0, driver 580): TESTED AND WORKING** on **1x H100 80GB single
> process (`--no_fsdp`, DDP degenerate at world size 1)** — finite decreasing loss,
> checkpoint written, work resident on the assigned GPU. The documented `pip install
> mosaicml` path pulls **torch 2.7.0+cu126**, which satisfies Composer's `torch<2.7.1` pin
> natively AND runs on a CUDA-13/driver-580 host (driver is backward-compatible) — so on
> H100 **no packaging override is required**, unlike the ROCm path below. See section 2
> "NVIDIA (CUDA)" and the **H100** subsection in section 8 for the exact commands and
> evidence. Multi-GPU (2, 8) is deferred (shared node). Run the smoke command in section 5
> first.
>
> **AMD MI355X (ROCm 7.2): TESTED AND WORKING** on **1x MI355X (gfx950) single process
> (no sharding) AND on all 8x MI355X of one node — both DDP and FSDP `FULL_SHARD`,
> world size 8, via the stock `composer -n 8` launcher** — despite upstream's "AMD +
> RoCM coming soon" claim. Composer's device detection handles ROCm-as-CUDA
> transparently. One packaging override is required (the `torch<2.7.1` pin is
> unsatisfiable on ROCm 7.2); **no source change and no extra package was needed to go
> from 1 GPU to 8.** See section 2 "AMD / ROCm" and section 8 (including "8-GPU run")
> for the exact commands, evidence and caveats. Run the smoke command in section 5 first.

### An honest note on the Mosaic stack

Composer's development slowed considerably after Databricks acquired MosaicML. It is
maintained rather than rapidly evolving, its most-publicised results (ResNet-50, BERT,
Stable Diffusion speedups) are from the pre-LLM era, and the ecosystem assumes you are
heading toward Databricks Mosaic AI Training. **Do not pick this as your default LLM SFT
trainer.** For an ordinary chat fine-tune, `../unsloth/` (LoRA, single GPU),
`../fsdp/` (PyTorch-native, no framework), or `../llamafactory/` /
`../axolotl/` (recipe-driven) will get you to a good model with less friction and
more community answers when something breaks.

Where Composer still genuinely earns its place:

- **The Algorithm system.** Nothing else lets you A/B a training-efficiency method by
  adding one object to a list. If your work is *comparing* efficiency methods rather than
  shipping one model, this is the right harness.
- **Auto-microbatching.** `device_train_microbatch_size="auto"` makes a config genuinely
  hardware-agnostic; the same run works on 40GB and 80GB cards without you re-tuning.
- **Elastic sharded checkpointing.** Save on 8 GPUs, resume on 16. This is first-class and
  well-tested here, and it is the reason the checkpoint format is `.pt` rather than HF.
- **It is the substrate under LLM Foundry.** If you are heading for `../llmfoundry/`
  (YAML-driven pretraining), understanding this file is what makes those YAMLs readable.

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

**H100 (CUDA 13.0, driver 580) — tested, works, no override needed.** On a
box where the `download.pytorch.org` index is unreachable, the `--index-url .../cu128`
line above cannot fetch a wheel — but you do not need it. Just let `mosaicml` resolve
torch from the default index. Composer's `torch<2.7.1` pin lands on **torch 2.7.0+cu126**,
which is a **cu12.6 build that runs on this CUDA-13/driver-580 host via driver
backward-compatibility** — `torch.cuda.is_available()` is True and a bf16 matmul executes
on the GPU. Because the pin is *satisfied* (not violated as on ROCm), **`pip check` is
clean and no `--force-reinstall --no-deps` dance is required**:

```bash
python -m venv .env_composer && source .env_composer/bin/activate
pip install torch numpy                 # base check: gives torch 2.13.0+cu130 on this box
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

If you instead WANT torch 2.13.0+cu130 (the box's native-CUDA-13 stable), reinstall it
*after* mosaicml with `pip install --force-reinstall --no-deps torch==2.13.0` (the CUDA
analog of the ROCm dance below); `pip check` will then report the `torch<2.7.1` pin as
violated — expected, and Composer 0.32.1 runs fine against it. **This is not needed on
H100** — 2.7.0+cu126 works out of the box, so the simpler no-override path is
recommended. Watch the same **triton trap** noted below: torch 2.7.0 ships CUDA `triton`
3.3.0; do not blind-uninstall it.

Upstream recommends their pre-built Docker images
(`mosaicml/pytorch:2.7.0_cu128-python3.12-ubuntu22.04` and friends) over a hand-built
environment, and that advice is sound if you want flash-attn without a build step.

### AMD / ROCm — TESTED, WORKS (MI355X / gfx950, ROCm 7.2.4)

Upstream still lists AMD as unsupported ("System with CUDA-compatible GPUs (AMD + RoCM
coming soon!)") and ships no ROCm install path, image or CI. **It works anyway.**
Composer is a thin layer over PyTorch, and every device call it makes goes
through `torch.cuda.*`, which ROCm implements. Verified on 1x AMD Instinct
MI355X, ROCm 7.2.4, Python 3.12.3.

The one real obstacle is packaging, not code: **`mosaicml` pins `torch<2.7.1`, and the
ROCm 7.2 wheel index only publishes torch 2.11/2.12/2.13.** The pin is therefore
*unsatisfiable* on ROCm 7.2 and must be overridden. Install Composer first (it will drag
in the CUDA torch), then force the ROCm wheels back over the top:

```bash
python -m venv .env_composer && source .env_composer/bin/activate

# 1. Composer first. This WILL pull CUDA torch 2.7.0 + ~3GB of nvidia-* wheels.
#    Let it — we overwrite them next.
pip install "mosaicml>=0.27.0" python-dotenv
pip install transformers tokenizers huggingface_hub

# 2. Force the ROCm build back over the CUDA one. --no-deps is REQUIRED, otherwise
#    pip re-resolves mosaicml's torch<2.7.1 pin and reinstalls the CUDA wheel.
pip install --force-reinstall --no-deps \
  torch==2.13.0+rocm7.2 torchvision==0.28.0+rocm7.2 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# 3. Drop the now-dead CUDA runtime wheels (~3GB). Do NOT blind-uninstall `triton`
#    (see the triton trap below).
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

**The triton trap.** The CUDA `triton` wheel installs into the same `triton/` package
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

If you would rather stay on a supported path, LLM Foundry (which sits on Composer) has a
documented beta AMD path — see `../llmfoundry/` — or use a PyTorch-native folder
such as `../fsdp/`.

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

`dev.env` is git-ignored at the repo root. **Never commit a token**; rotate it on the Hub
if one ever lands in a commit.

The `composer` launcher inherits your shell environment, so the token reaches every rank
without extra plumbing.

## 4. Data

Chat JSONL, one conversation per line — the same format as the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The folder ships a 10-row sample at `data/OTel_LLM_sample_10.jsonl` (the default
`--train_file`), so the smoke run below works with no data prep. The sample rows carry
extra metadata columns alongside `messages` (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version` — the last two null in most rows). That is fine here:
the loader reads each line with `json.loads` and takes only the `"messages"` key, so extra
columns are ignored.

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
how many decoder blocks were tagged for FSDP wrapping (if this says 0, or the tagging step
raises, sharding is not happening); with `--device_train_microbatch_size auto` you will see
Composer try a microbatch, possibly catch an OOM, halve it and continue — that is the
feature working, not a failure. Then per-batch console lines every 10 batches with
`loss/train/total`, `LanguageCrossEntropy`, `Perplexity`, plus throughput from
`SpeedMonitor` (tokens/sec/device) and a shrinking ETA from `RuntimeEstimator`. Memory
should be roughly even across all 8 GPUs.

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

## 8. Hardware support & evidence

- **NVIDIA / CUDA — supported.** The Composer README
  ([github.com/mosaicml/composer](https://github.com/mosaicml/composer)) states the
  hardware requirement as "System with CUDA-compatible GPUs"; MosaicML's published results
  and Docker images (`mosaicml/pytorch:2.7.0_cu128-*`) are all NVIDIA/CUDA. This folder
  targets 8x H100 80GB.
- **AMD / ROCm — unsupported upstream, but VERIFIED WORKING here.** The same README
  sentence continues "(AMD + RoCM coming soon!)", and upstream still documents no ROCm
  install path, image or CI. That claim describes upstream's *support commitment*, not a
  technical limitation: Composer 0.32.1 runs end-to-end on an MI355X. See the
  MI355X subsection below for commands and evidence.
- **Other hardware (upstream claims — not verified here):** none claimed beyond
  NVIDIA/AMD — the upstream README's hardware requirement is "CUDA-compatible GPUs",
  with CPU runs possible for debugging only.

### NVIDIA H100 (CUDA 13.0, driver 580)

**This path works as documented on H100.** No packaging override, no source changes. The documented `pip install
mosaicml` path resolves Composer's `torch<2.7.1` pin to **torch 2.7.0+cu126**, which runs
unchanged on this CUDA-13.0 / driver-580.173.02 host (driver backward-compatibility), so
`pip check` stays clean. Single-GPU smoke (`--no_fsdp`, DDP degenerate at world size 1)
produced finite decreasing loss, a saved checkpoint, and GPU-resident work. Multi-GPU (2,
then 8) is deferred — the node's other GPUs were running a production job.

| Field | Value |
| --- | --- |
| GPU | 1x NVIDIA H100 80GB HBM3, Hopper cc(9,0), native bf16/FP8 |
| Driver / CUDA | 580.173.02 / 13.0 |
| torch | **2.7.0+cu126** (mosaicml-resolved from the default index; runs on driver 580) |
| Base-case torch | `pip install torch` alone gives 2.13.0+cu130 — clobbered back to 2.7.0 by mosaicml (expected) |
| composer (`mosaicml`) | 0.32.1 |
| transformers / tokenizers | 5.15.1 / 0.22.2 |
| torchmetrics / numpy | 1.7.4 / 2.2.6 |
| triton | 3.3.0 (CUDA build shipped with torch 2.7.0 — do NOT blind-uninstall) |
| Python | 3.12.3 |
| attn_implementation | `sdpa` (LFM2 is a hybrid conv/attention arch; sdpa is the safe default. flash-attn deferred) |
| Model | **`LiquidAI/LFM2.5-350M`** (chat template present) — swapped in for the documented `meta-llama/Llama-3.1-8B-Instruct`, which is NOT in the offline model cache |

**torch reconciliation (the central H100 gotcha).** `pip install torch` gives
`2.13.0+cu130`; then `pip install mosaicml` **silently downgrades it to `2.7.0+cu126`** to
satisfy `torch<2.7.1`. Unlike ROCm — where the pin is *unsatisfiable* and you must
`--force-reinstall --no-deps` a ROCm wheel back — on H100 the resolved cu126 build simply
**works** (bf16 matmul executes on-GPU, `composer.utils.get_device(None)` returns
`DeviceGPU`). So the correct action is to *accept* mosaicml's torch and re-verify it, not
to override it. `pip check` reports **no broken requirements**. `torch.backends.cuda.matmul.allow_tf32`
auto-enables to `True` on import. (`kernels` was not pulled, so the transformers-5.x
`kernels 0.16.0` import crash never triggered; if it does, pin `kernels>=0.12,<0.13`.)

Exact smoke command (offline env, GPU 7, master port 29648):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1            # HF_HOME points at the model cache (set above)
export HF_DATASETS_CACHE=/dev/shm/h100/dscache_composer   # datasets .arrow writes can fail on a network model-cache mount
export CUDA_VISIBLE_DEVICES=7                             # plain CUDA_VISIBLE_DEVICES; no HIP_* on NVIDIA

composer -n 1 --master_port 29648 train_llm_composer.py \
  --model_name LiquidAI/LFM2.5-350M \
  --max_seq_len 2048 --max_duration 40ba \
  --global_train_batch_size 2 --device_train_microbatch_size 1 \
  --learning_rate 5e-5 --no_fsdp \
  --attn_implementation sdpa \
  --run_name composer-h100-smoke \
  --save_folder /dev/shm/h100/out/composer/composer_run --save_interval 1000ba
```

Step count: 9 usable rows (1 of 10 dropped for exceeding `--max_seq_len 2048`),
`global_train_batch_size 2` / world 1 / microbatch 1 → 3 batches/epoch, `40ba` =
**40 optimizer steps over ~4.4 epochs** (non-trivial). Expected output:

```
- INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
- INFO - __main__ - Starting training: 40ba, fsdp=False, algorithms=['GradientClipping']
[batch=1/40]:   Train loss/train/total: 2.0366    Train metrics/train/LanguagePerplexity: 6.7352
[batch=10/40]:  Train loss/train/total: 0.6773    Train metrics/train/LanguagePerplexity: 1.9369
[batch=20/40]:  Train loss/train/total: 0.0280    Train metrics/train/LanguagePerplexity: 1.0409
[batch=40/40]:  Train loss/train/total: 0.0006    Train metrics/train/LanguagePerplexity: 1.0010
	 Train memory/peak_reserved_mem: 7.8895    Train throughput/tokens_per_sec: 27609.4498
- INFO - __main__ - Training complete. Composer checkpoints are in .../composer_run
### composer exit_code=0
```

Loss falls monotonically 2.04 → 0.68 → 0.028 → 0.0006 (perplexity 6.74 → 1.001): the
expected fast memorization of 9 rows. GPU-7 residency, sampled by PID from *inside* the
run (GPU 7 was 0 MiB before launch):

```
=== nvidia-smi GPU7 sample 2 ===  7, 4 MiB, 0 %, 74.17 W          # pre-load
=== nvidia-smi GPU7 sample 4 ===  7, 8841 MiB, 26 %, 259.58 W     # mid-train
                                  1406030, 8832 MiB               # our python PID on GPU 7
```

Peak reserved VRAM was **7.89 GB** on 80 GB — headroom is enormous; the 350M model
fits trivially and no OOM/batch-reduction was needed (contrast the VRAM-driven tuning that
larger models would require on this 80 GB card vs 288 GB on MI355X).

**Checkpoint / cleanup.** Composer wrote a **fit-end checkpoint regardless of
`--save_interval 1000ba`** — `ep9-ba40-rank0.pt` (2.0 GB) plus a `latest-rank0.pt`
symlink. Evidence captured, then the checkpoint was deleted per policy (put `--save_folder`
under `/dev/shm` and clean up).

**What a multi-GPU pass would need (DEFERRED):** free GPUs, then `composer -n <N>` with the
same script; drop `--no_fsdp` and add `--sharding_strategy FULL_SHARD --fsdp_mixed_precision
PURE` for the FSDP variant. The FSDP wrap path (`tag_blocks_for_fsdp`) keys on LFM2's
`_no_split_modules`; verify it tags >0 blocks before trusting a sharded run. At world size 1
FSDP is bypassed (`use_fsdp` requires world size > 1), which is why the smoke uses
`--no_fsdp`.

### AMD MI355X (ROCm 7.2)

**This path works on MI355X with one change.** That change is a packaging override (Composer's
`torch<2.7.1` pin is unsatisfiable on the ROCm 7.2 wheel index). **Zero source changes to
`train_llm_composer.py` were needed for ROCm** — no tf32 hardcode, no FlashAttention-2
hardcode, no device-string assumptions.

| Component | Version |
|---|---|
| GPU | 1x AMD Instinct MI355X, `gfx950:sramecc+:xnack-` |
| ROCm | 7.2.4 (`torch.version.hip` = `7.2.53211`) |
| torch | `2.13.0+rocm7.2` (`torch.version.cuda` is `None`) |
| mosaicml (Composer) | `0.32.1` |
| transformers | `5.15.0` |
| triton-rocm | `3.7.1` |
| Python | 3.12.3 |

Install: see section 2 "AMD / ROCm". Smoke run (single GPU, no sharding, small model, few
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

`rocm-smi -d 3` sampled mid-run confirms the work landed on the physical GPU:

```
GPU[3] : Current Socket Graphics Package Power (W): 351.0
GPU[3] : GPU use (%): 18
GPU[3] : GPU Memory Allocated (VRAM%): 6
```

**Does Composer's device detection handle ROCm? Yes — transparently.**
`composer.utils.get_device(None)` returns `DeviceGPU` (`name='gpu'`) with no ROCm-specific
configuration. `DeviceGPU` is written entirely against `torch.cuda.*` and
`torch.distributed` with the NCCL backend, all of which ROCm implements (RCCL is
API-compatible and reports itself as NCCL). Consequences observed:

- The `composer` launcher autodetects the visible device count correctly — with
  `HIP_VISIBLE_DEVICES=3` it sees exactly one GPU.
- `MemoryMonitor` works: `memory/peak_reserved_mem`, `alloc_retries` etc. are populated
  from `torch.cuda` memory stats.
- `precision='amp_bf16'` works on gfx950 with no changes. bf16 is native on MI355X.
- `SpeedMonitor` / `RuntimeEstimator` / `LRMonitor` all report normally.
- Log lines mention "NCCL" and "ProcessGroupNCCL" — on ROCm that is RCCL. Cosmetic.

**Auto-microbatching works.** `--device_train_microbatch_size auto` completed and settled
on a microbatch of 4 (with 288GB of VRAM it never needed to back off). Composer still
prints its standard warning that OOM-catching can leave CUDA in an irrecoverable state;
that caveat applies equally on ROCm, so pin an integer for long unattended runs.

**Quirks and caveats found on ROCm:**

1. **The `torch<2.7.1` pin cannot be satisfied on ROCm 7.2** — the index carries only
   2.11.0 / 2.12.0 / 2.12.1 / 2.13.0. Overriding it is mandatory, and `pip check` stays
   unhappy forever. Nothing broke as a result across install, model construction,
   training, metrics and checkpointing.
2. **The triton clobber** (see section 2) — the single most likely way to break this env.
3. **`--save_interval` does not suppress the end-of-run checkpoint.** Composer wrote
   `ep2-ba5-rank0.pt` at 3.6GB even with `--save_interval 1000ba` on a 5-batch run. Three
   short runs cost 11GB. Budget disk accordingly, or point `--save_folder` at scratch.
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

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**Works unchanged.** The same recipe that passes on 1 GPU scales to all
8 MI355X of the node with **zero source changes, zero new packages and zero new pins**:
only the launcher's `-n` and the batch geometry change. Both **DDP** (`--no_fsdp`) and
**FSDP `FULL_SHARD`** complete 50 batches at world size 8 with exit code 0, all 8
ranks finishing and no teardown hang. RCCL (reported as "NCCL") works out of the box.

Launch (run both variants inside one `flock`-serialised session so the job owns all 8 GPUs):

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

**Batch geometry, as Composer actually ran it** (this is the classic 8-GPU trap — the
Trainer takes a *global* batch and the script divides it by world size before building the
dataloader):

| Knob | Value |
|---|---|
| World size | 8 (one process per MI355X, spawned by `composer -n 8`) |
| `--global_train_batch_size` | 64 → **must stay divisible by 8** |
| Per-device minibatch | 64 / 8 = **8** (`device_batch_size` in `main()`) |
| `trainer/device_train_microbatch_size` (logged) | **2** → 4 grad-accum steps per rank per batch |
| Samples consumed at batch 50 (logged `Train time/sample`) | **3136** = 49 x 64 — confirms the global batch really was 64 across 8 ranks |

**Data — this is a pipeline proof, not a learning result.** The shipped 10-row sample
cannot feed 8 ranks meaningfully, so the 10 rows were replicated 64x to 640 rows written
**outside the repo** (`$OUTPUT_DIR/composer/gpu8/data/`). Report the
survivors, not the input count — at `--max_seq_len 2048` the loader drops one of the ten
distinct rows (2242 tokens) and therefore **576 of 640 rows survived**:

```
INFO - __main__ - Loaded 576 rows from .../OTel_sample_x64_640.jsonl (dropped 64 over 2048 tokens, 0 with no supervised tokens)
INFO - __main__ - Starting training: 50ba, fsdp=False, algorithms=['GradientClipping']
```

The loss falls smoothly and finitely, but on 64 copies of 9 conversations that is
**memorisation, not generalisation**. Use it as proof the 8-rank pipeline is correct.

Expected output — DDP (rank 0 console, `console_log_interval=10ba`):

```
[batch=1/50]:  Train loss/train/total: 1.6896  LanguageCrossEntropy: 1.6547  LanguagePerplexity: 5.2316
[batch=20/50]: Train loss/train/total: 0.5505  throughput/tokens_per_sec: 155825.3348  device/tokens_per_sec: 19478.1669
[batch=50/50]: Train loss/train/total: 0.2242  throughput/tokens_per_sec: 153504.0970  device/tokens_per_sec: 19188.0121
[batch=50/50]: Train throughput/batches_per_sec: 1.2936  memory/peak_reserved_mem: 27.2940
### END ddp exit_code=0
```

FSDP `FULL_SHARD` (same data, same geometry):

```
[batch=1/50]:  Train loss/train/total: 1.6870   memory/peak_reserved_mem: 21.6430
[batch=50/50]: Train loss/train/total: 0.2236   throughput/tokens_per_sec: 134413.3856  device/tokens_per_sec: 16801.6732
[batch=50/50]: Train memory/peak_reserved_mem: 21.6450
### END fsdp exit_code=0
```

**DDP vs FSDP on 8x MI355X, measured:** peak reserved VRAM **27.29 GB/GPU (DDP) →
21.64 GB/GPU (FSDP FULL_SHARD)**, a **5.65 GB (-20.7%) per-GPU saving**, paid for with
~13% throughput (153.5k → 134.4k aggregate tokens/sec) from the all-gather traffic. Loss
curves match to 3 decimal places (0.2242 vs 0.2236 at batch 50), i.e. sharding did not
change the math. At 0.6B params on 288 GB cards FSDP is pure overhead; the number that
matters is that `tag_blocks_for_fsdp()` + `parallelism_config` work unmodified on ROCm.

**8-GPU evidence — `rocm-smi` sampled IN-BAND (from inside the training script, while our
own ranks were live), not from a separate shell afterwards:**

```
Device  Temp     Power    SCLK     ...  VRAM%  GPU%
0       45.0°C   915.0W   2363Mhz       11%    100%
1       45.0°C   883.0W   2376Mhz       10%    100%
2       44.0°C   905.0W   2363Mhz       12%    100%
3       48.0°C   899.0W   2375Mhz       10%    100%
4       47.0°C   864.0W   2381Mhz       12%    100%
5       47.0°C   884.0W   2374Mhz       12%    100%
6       46.0°C   863.0W   2372Mhz       12%    100%
7       46.0°C   867.0W   2375Mhz       10%    100%
```

All eight cards reach 100% busy and ~900 W (vs a 1400 W cap) simultaneously, with even
VRAM (10-12% of 288 GB) — the "memory roughly even across all 8 GPUs" that section 5
promises.

**PID cross-check** — `rocm-smi --showpids` in the *same* sample as a `pgrep -af`, so the
VRAM holders are provably our own children and not another tenant's job:

```
--- pgrep -af train_llm_composer.py ---
611699 .../bin/composer -n 8 --master_port 29740 train_llm_composer.py ... --no_fsdp   <- launcher
611930 611931 611932 611933 611934 611935 611936 611937  <- the 8 rank processes
--- rocm-smi --showpids ---
PID       PROCESS NAME   GPU(s)   VRAM USED
611932    python3        1        38908776448
611930    python3        1        34151780352
611937    python3        1        31910543360
611935    python3        1        38751453184
611933    python3        1        32046858240
611931    python3        1        32053149696
611936    python3        1        38952198144
611934    python3        1        39025561600
```

Exactly 8 KFD processes, each on exactly **1** GPU, and every PID matches a `composer -n 8`
child from the same sample. (The FSDP run repeats this with PIDs 627676-627683.) Sampling
`rocm-smi` from a separate bash call after a 60-second run is worthless on a shared box —
it will happily photograph someone else's job.

**What differed from the 1-GPU run:**

1. `-n 1` → `-n 8`, `--master_port 29710` → `29740`, and `--no_fsdp` dropped for the FSDP run.
2. **Any stale `HIP_VISIBLE_DEVICES=3` / `CUDA_VISIBLE_DEVICES=3` exports inside
   `.env_composer/bin/activate` must be overridden after sourcing.** They are an easy
   leftover from a 1-GPU run; leave them and `composer` autodetects **one** device and you
   report an 8-GPU pass that never happened. Assert `torch.cuda.device_count() == 8`.
3. Batch geometry: global 2 → 64, microbatch pinned 1 → 2 (`auto` was deliberately avoided —
   OOM-catching is not something you want racing on 8 ranks).
4. Data: the 10-row sample was replicated to 640 rows outside the repo so each rank gets
   real work; 576 survived the 2048-token filter.
5. Checkpointing: `--save_folder` pointed at scratch. Confirming caveat 3 below, the
   end-of-run checkpoint is written regardless of `--save_interval 1000ba` — **3.4 GB (DDP)
   and 3.7 GB (FSDP, `state_dict_type=full`) for 50-batch runs.** Deleted after evidence
   capture.
6. Nothing else. No new package, no new pin, no code edit — `requirements_composer.txt` is
   unchanged by the 8-GPU work.

**Still not tested:** multi-node (this is one node of 8), `SHARD_GRAD_OP` / `HYBRID_SHARD`,
`--state_dict_type sharded` elastic resume, `SeqLengthWarmup`, `LowPrecisionLayerNorm`,
`--device_train_microbatch_size auto` at world size 8, models larger than 0.6B, and any
NVIDIA path.

**Not covered by the 1-GPU run (now covered by the 8-GPU section above):**
multi-GPU FSDP sharding on ROCm — the 1-GPU run used `--no_fsdp` throughout and owned a
single GPU. Still untested anywhere here: `SeqLengthWarmup`, `LowPrecisionLayerNorm`,
checkpoint resume, and any NVIDIA path.
- **CPU** — Composer runs on CPU per the README, but that is for debugging, not LLM
  training.
- **Maintenance status:** post-Databricks-acquisition, the repo is maintained (release
  cadence continues) but development has visibly slowed and the README steers users toward
  Databricks Mosaic AI Training. Its headline speedup results date from the pre-LLM era.

Sources: Composer README (`README.md`, `dev` branch);
`docs/source/notes/distributed_training.rst`; Docker Hub `mosaicml` org.

## 9. Notes

- **`parallelism_config` is the current API; `fsdp_config=` is not.** Composer used to take
  a top-level `fsdp_config` kwarg on the Trainer. That was replaced by
  `parallelism_config={'fsdp': {...}, 'tp': {...}}`, which is what the current distributed
  training docs and LLM Foundry both use. Old blog posts and StackOverflow answers still
  show the flat form — ignore them. (Confusingly, LLM Foundry's *YAML* still has a
  top-level `fsdp_config:` block; its train script folds that into `parallelism_config`
  before constructing the Trainer. Both are correct at their own layer.)
- **You must tell Composer which modules to wrap.** Composer's auto-wrap policy looks for
  `module._fsdp_wrap = True`, or a `fsdp_wrap_fn` on the root module. A stock HF model has
  neither, so a naive `HuggingFaceModel` + FSDP run wraps almost nothing and saves almost no
  memory. `tag_blocks_for_fsdp()` walks the model, finds the block class from HF's
  `_no_split_modules` (e.g. `["LlamaDecoderLayer"]`), and sets `_fsdp_wrap` (and
  `_activation_checkpointing`) on each one. This is the single most important detail in
  this file.
- **`load_dotenv` runs at startup** so `HF_TOKEN` is in the environment before any Hub
  download; the token is also passed explicitly to `from_pretrained`.
- **`mixed_precision: PURE` vs `DEFAULT`.** `DEFAULT` emulates classic AMP: parameters and
  buffers are cast to bf16 for compute but gradients are all-reduced in fp32. `PURE`
  all-reduces in bf16 too — meaningfully faster, and what the Mosaic reference LLM configs
  use. `FULL` keeps everything fp32. `PURE` is the default here; drop to `DEFAULT` if you
  see loss instability.
- **Time is a first-class type.** `3ep`, `2000ba`, `500sp`, `10000000tok` are all valid for
  `--max_duration`, `--save_interval` and warmup. `0.03dur` means "3% of total training",
  which keeps the warmup correct when you change the duration.
- **Auto-microbatching is the headline ergonomic win.** `global_train_batch_size` is the
  *math* (what the optimizer sees); `device_train_microbatch_size` is the *execution* (what
  fits on a card). Composer derives gradient accumulation from the two plus the world size,
  so the same config trains identically on 8 or 16 GPUs. With `auto`, it discovers the
  microbatch by catching OOMs and retrying.
- **Speedup algorithms, honestly scoped.** Most of Composer's catalogue is vision-oriented
  (BlurPool, ColOut, ProgressiveResizing, MixUp) and irrelevant here. The LLM-relevant ones
  exposed by this script are `GradientClipping` (always on), `LowPrecisionLayerNorm` (small,
  safe throughput win), and `SeqLengthWarmup` (larger win, more caveats). `Alibi` only
  applies to models with ALiBi position embeddings; `EMA` and `SelectiveBackprop` are
  available in `composer.algorithms` if you want to experiment.
- **`SeqLengthWarmup` caveat.** It truncates every tensor in the batch, labels included, to
  a sequence length that starts small. With prompt masking on, early batches can end up with
  no supervised tokens at all if your prompts are long. It suits continued pre-training
  (`--no_mask_prompt`) much better than short-answer SFT. Measure before you keep it.
- **`shift_labels=True` is not optional.** Causal LMs predict token i+1 from token i.
  Composer infers this from the model class name, but the script sets it explicitly so the
  logged `LanguageCrossEntropy` cannot silently disagree with the training loss.
- **OOM playbook.** In order: `--device_train_microbatch_size auto` (if you had pinned it);
  `--activation_checkpointing`; lower `--max_seq_len`; `--fsdp_mixed_precision PURE`; then
  raise `--global_train_batch_size` only after the per-device footprint is under control.
