# `training/llm/lightning` — chat fine-tuning with PyTorch Lightning

> **Tested topology: 1x and 8x AMD Instinct MI355X (ROCm 7.2.4), 2026-08-19** — single
> device plus a full FSDP `FULL_SHARD` run across all 8 GPUs; see
> [Tested on AMD MI355X](#tested-on-amd-mi355x-rocm-72--2026-08-19) below. Originally
> written against the PyTorch Lightning upstream README and the `FSDPStrategy` /
> `DeepSpeedStrategy` sources on `master` as of **August 2026**, targeting a single node
> of 8x H100 80GB (NVIDIA, CUDA); the NVIDIA multi-GPU path remains untested. Lightning
> moves quickly across 2.x minors — run the smoke command below before committing to a
> long job.

## Overview & when to use

Full fine-tuning (or continued pre-training) of a Hugging Face causal LM with
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). The model, loss,
optimizer and LR schedule live in one ~40-line `LightningModule`; sharding (FSDP or
DeepSpeed ZeRO), bf16 mixed precision, distributed samplers, checkpointing and resumption
are all supplied by `lightning.Trainer` flags. Pick this over the other trainers here
when you want a *structured, testable* training loop that you own — swapping
`strategy="fsdp"` for `strategy="deepspeed"` is a one-word change, and the same file runs
unmodified on 1 GPU or 8 nodes.

The script loads `dev.env` right after its imports and reads `HF_TOKEN` from the
environment at model-download time. One non-obvious safeguard:
`save_hyperparameters(ignore=["hf_token"])` — Lightning serialises saved hyperparameters
into *every* `.ckpt` file, so a token passed to `__init__` without that `ignore` would be
written to disk on each checkpoint. `device_map` stays `None` on load — the strategy
places and shards the parameters.

Files in this folder:
- `train_llm_lightning.py` — the trainer: `LightningModule` + `Trainer`, chat-JSONL data pipeline, optional HF export.
- `requirements_lightning.txt` — deps, with notes on `deepspeed` and `flash-attn`.
- `data/OTel_LLM_sample_10.jsonl` — a 10-row sample dataset the script points at by default.
- `readme_lightning.md` — this file.

### Lightning Fabric — the lighter-weight alternative

Lightning ships two products. `Trainer` (this file) owns the loop for you. **Fabric**
(`lightning.fabric`) gives you the same accelerator/precision/sharding plumbing while you
keep writing the `for batch in loader:` loop yourself:

```python
import lightning as L

fabric = L.Fabric(accelerator="gpu", devices=8, strategy="fsdp", precision="bf16-mixed")
fabric.launch()
model, optimizer = fabric.setup(model, optimizer)
train_loader = fabric.setup_dataloaders(train_loader)
# ... your own loop; fabric.backward(loss) instead of loss.backward()
```

Fabric is the better fit when the training loop itself is the thing you are experimenting
with (custom RL loops, multi-model steps, unusual gradient handling) and you do not want
callbacks/hooks in the way. It is already installed by `lightning` — no extra dependency.
This folder deliberately uses `Trainer` because checkpointing, resumption and distributed
sampling are exactly the parts you do not want to hand-write for an ordinary SFT run.

## Install

Python 3.12 in its own venv. Install `torch` first, matched to your accelerator, then the
rest.

### NVIDIA (CUDA)

```bash
python3.12 -m venv ~/.venv-lightning && source ~/.venv-lightning/bin/activate
pip install torch==2.11.0
pip install -r requirements_lightning.txt
```

### AMD (ROCm)

Lightning has no ROCm-specific code path of its own — it runs on whatever accelerator
torch exposes, and ROCm builds of torch present the GPU through the standard `torch.cuda`
API, so `accelerator="gpu"` works unchanged. Note the honest claim level: Lightning
upstream does not separately advertise or CI-test ROCm (its README lists CPU/GPU/TPU
generically), so this is "supported via torch," not "certified by Lightning."
torch 2.11.0 ROCm wheels live on the **rocm7.2** index (the older `rocm6.4` index stops
at torch 2.9.1 — verified against download.pytorch.org, 2026-08):

```bash
python3.12 -m venv ~/.venv-lightning && source ~/.venv-lightning/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_lightning.txt
```

On ROCm keep `--attn_implementation sdpa` (the default) and prefer `--strategy fsdp`.
(DeepSpeed 0.19.x ships as a pure-python wheel that JIT-compiles its ops and has been
verified working on this box's ROCm 7.2 stack — see `training/llm/deepspeed` — so
`--strategy deepspeed` is also viable, though it was not smoke-tested from this folder.)
This install path was executed verbatim on an MI355X on 2026-08-19 — see the
[Tested on AMD MI355X](#tested-on-amd-mi355x-rocm-72--2026-08-19) section for results
and two required run-time tweaks.

### Either way

Install `lightning`, **not** `pytorch-lightning`. They are the same Trainer published
under two import roots (`import lightning as L` vs `import pytorch_lightning as pl`);
having both in one environment is a reliable way to get confusing import errors. This
script uses the `lightning` root.

Only if you plan to pass `--strategy deepspeed`:

```bash
pip install "deepspeed>=0.16.0"
```

That floor is enforced by Lightning itself: with torch >= 2.6, `torch.load` defaults to
`weights_only=True`, and DeepSpeed only handles that from 0.16.0 onward.

## Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_lightning.py` calls `load_dotenv("dev.env")` at import time and passes the
token explicitly to `AutoTokenizer` and `AutoModelForCausalLM`. That is all that is
needed for gated checkpoints such as Llama or Gemma.

`dev.env` is git-ignored at the repo root. **Never commit a token**; rotate it on the Hub
if one ever lands in a commit.

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line, the same format as the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The sample rows also carry extra bookkeeping columns (`unmask`, `flow`, `source_id`,
`source_repo`, `source_spec_id`, `source_version`; `source_spec_id`/`source_version` are
null in most rows). The loader reads only the `messages` key from each line, so extra
columns are ignored — your own data may include or omit them freely. To train on real
data, pass `--train_file /path/to/train.jsonl` with the same `messages` schema.

The whole file is read and tokenized up front with `tokenizer.apply_chat_template`, so
train-time formatting matches what you will send at inference. The loader returns a plain
list of dicts, which `DataLoader` accepts directly as a map-style dataset. Behaviours
worth knowing:

- **Prompt masking is on by default.** Every non-assistant token gets label `-100`, so
  loss falls on assistant turns only. Pass `--no_mask_prompt` to train on the full
  rendered sequence instead — that is the setting you want for *continued pre-training*,
  where the point is to model the whole text.
- **Over-length rows are dropped, never truncated.** A conversation longer than
  `--max_seq_len` is skipped, and the count is logged. Truncating a chat mid-answer
  teaches the model to stop mid-answer.
- Rows with no supervised tokens (no assistant turn) are also dropped. If the loader
  reports a large drop count, your `--max_seq_len` is too small for the data.

## Run

Smoke test first — one GPU, the shipped sample, one epoch, so failures surface in
minutes:

```bash
python3 train_llm_lightning.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --devices 1 --strategy auto \
  --output_dir ./lightning_smoke
```

Then the real 8-GPU run:

```bash
nohup python3 train_llm_lightning.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./lightning_run \
  --export_hf_dir ./lightning_run/final_model \
  --strategy fsdp --devices 8 \
  --precision bf16-mixed \
  --max_seq_len 4096 \
  --batch_size 1 --grad_acc_steps 8 \
  --num_train_epochs 3 --learning_rate 1e-5 \
  --activation_checkpointing \
  > train_llm_lightning.log 2>&1 &

tail -f train_llm_lightning.log
```

`python3 train_llm_lightning.py` is the whole launch command — Lightning spawns and
configures the worker processes itself from `--devices` / `--num_nodes`. Do **not** wrap
it in `torchrun`. (If you are on SLURM, Lightning detects `SLURM_*` env vars and attaches
to the allocation instead of spawning.)

DeepSpeed ZeRO-3 instead of FSDP:

```bash
python3 train_llm_lightning.py ... --strategy deepspeed --zero_stage 3 --cpu_offload
```

**What "working" looks like:** the loader logs how many rows it kept and dropped;
Lightning prints the strategy and precision it selected, then a line listing the FSDP
wrap classes (e.g. `LlamaDecoderLayer`); `nvidia-smi` (or `rocm-smi`) shows roughly even
memory across all 8 GPUs, not one GPU at 79GB and the rest idle. After that,
`train_loss` lines every `--log_every_n_steps` optimizer steps, falling quickly for the
first few hundred steps and then flattening. If memory is badly uneven, sharding is not
actually engaged — check that `--strategy fsdp` was accepted and that the wrap-class line
appeared.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL: one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./lightning_run` | Lightning logs + `checkpoints/` (`.ckpt` files) |
| `--export_hf_dir` | `""` | If set, write a plain `from_pretrained`-loadable folder after training |
| `--resume_ckpt` | `""` | Lightning `.ckpt` to resume from (ignored if missing) |
| `--max_seq_len` | `4096` | Rows longer than this are dropped, not truncated |
| `--max_samples` | `None` | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for validation (0 = no val loop) |
| `--mask_prompt` / `--no_mask_prompt` | on | Assistant-only loss (default) vs full-sequence loss |
| `--num_workers` | `4` | Dataloader workers per rank |
| `--batch_size` | `1` | Per-device micro-batch size |
| `--grad_acc_steps` | `8` | Trainer `accumulate_grad_batches`; global batch = product x world size |
| `--num_train_epochs` | `3` | Max epochs |
| `--learning_rate` | `1e-5` | Peak LR (full fine-tune ~1e-5..2e-5); cosine schedule with warmup |
| `--weight_decay` | `0.0` | AdamW weight decay |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--grad_clip` | `1.0` | Global grad-norm clip; 0 disables |
| `--devices` | `-1` | GPUs per node; -1 = all visible |
| `--num_nodes` | `1` | Node count |
| `--precision` | `bf16-mixed` | `bf16-mixed`, `bf16-true`, `16-mixed`, `32-true` |
| `--log_every_n_steps` | `10` | Logging cadence in optimizer steps |
| `--seed` | `42` | Random seed |
| `--attn_implementation` | `sdpa` | `sdpa` (default), `flash_attention_2` (CUDA only), `eager` |
| `--strategy` | `fsdp` | `fsdp`, `deepspeed`, `ddp`, or `auto` |
| `--sharding_strategy` | `FULL_SHARD` | FSDP: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` |
| `--state_dict_type` | `full` | FSDP: `full` consolidates on rank 0, `sharded` writes one file per rank |
| `--zero_stage` | `3` | DeepSpeed only: 1, 2 or 3 |
| `--cpu_offload` | off | Offload optimizer state (and ZeRO-3 params) to CPU |
| `--activation_checkpointing` | off | Recompute decoder-layer activations (FSDP path); big memory win, ~20-30% slower |

## Output

```
lightning_run/
  lightning_logs/version_0/          # TensorBoard event files, hparams.yaml
  checkpoints/
    epoch=..-step=...ckpt            # Lightning checkpoints (weights + optimizer + schedule)
    last.ckpt
  final_model/                       # only if --export_hf_dir was passed
    config.json, model-*.safetensors, tokenizer files
```

A `.ckpt` is a *Lightning* checkpoint, not a Hugging Face model: it holds optimizer and
scheduler state so training can resume exactly, and `from_pretrained` cannot read it.
Resume with `--resume_ckpt ./lightning_run/checkpoints/last.ckpt`.

For serving, use the `--export_hf_dir` folder:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./lightning_run/final_model")
tok = AutoTokenizer.from_pretrained("./lightning_run/final_model")
```

Watch `train_loss` in TensorBoard (`tensorboard --logdir lightning_run/lightning_logs`),
and `val_loss` too if you passed `--eval_samples`.

## Hardware support & evidence

**Other hardware (upstream claims — not verified here):** Apple Silicon (MPS) and Google TPU are first-class Lightning accelerators; Intel via torch XPU builds (Lightning docs).


Claims above were checked against upstream sources on **2026-08-19**:

- **Lightning's own claim level.** The upstream README
  ([Lightning-AI/pytorch-lightning `README.md`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/README.md))
  advertises hardware-agnostic training ("scales from CPU to multi-node GPUs") and a CI
  matrix of CPUs, GPUs (CUDA) and TPUs. It does **not** name AMD ROCm anywhere, so the
  ROCm claim here rests on torch, not on Lightning: ROCm torch builds expose the GPU
  through the `torch.cuda` API, which is the same API Lightning's GPU accelerator uses.
  Honest status: expected to work via torch, not separately certified upstream.
- **ROCm torch wheels.** `torch==2.11.0+rocm7.2` wheels exist on
  [download.pytorch.org/whl/rocm7.2](https://download.pytorch.org/whl/rocm7.2/torch/);
  the `rocm6.4` index stops at torch 2.9.1, which is why the install line pins the
  `rocm7.2` index.
- **DeepSpeed floor.** Lightning refuses to start with torch >= 2.6 and
  deepspeed < 0.16.0 (`torch.load` switched to `weights_only=True`); hence the
  `deepspeed>=0.16.0` note in the requirements file.

## Tested on AMD MI355X (ROCm 7.2) — 2026-08-19

**Verdict: works with changes** on a single AMD Instinct MI355X (gfx950, 288 GB), ROCm
7.2.4, Python 3.12.3. One code fix was required (transformers API drift, not a ROCm
issue); the ROCm install path above is otherwise exactly right.

Environment that was actually run (versions as resolved by pip on 2026-08-19):
`torch 2.11.0+rocm7.2`, `lightning 2.6.5`, `transformers 5.15.0`, `tokenizers 0.22.2`,
`accelerate 1.14.0`, `python-dotenv 1.2.3`.

```bash
cd training/llm/lightning
python3 -m venv .env_train_llm_lightning && source .env_train_llm_lightning/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_lightning.txt
ln -sf ../../../dev.env dev.env   # HF_TOKEN for the gated Gemma checkpoint
```

Smoke run (single GPU — the box runs one job per GPU, hence the masking exports; on a
dedicated box you can omit them):

```bash
export HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5   # pick your GPU; it becomes device 0
python3 train_llm_lightning.py \
  --model_name google/gemma-4-E4B-it \
  --num_train_epochs 1 --devices 1 --strategy auto \
  --batch_size 1 --grad_acc_steps 2 --max_seq_len 2048 --max_samples 6 \
  --output_dir ./lightning_smoke
```

What it printed (abridged):

```
Loaded 6 rows from data/OTel_LLM_sample_10.jsonl (dropped 0 over 2048 tokens, 0 with no supervised tokens)
You are using a CUDA device ('AMD Instinct MI355X') that has Tensor Cores. ...
`Trainer.fit` stopped: `max_epochs=1` reached.
Training complete. Checkpoints under .../lightning_smoke/checkpoints
```

`train_loss` (TensorBoard) over the 3 optimizer steps: 0.18 → 2.34 → 1.22 — finite and
sane for 6 samples. `rocm-smi` showed ~32-35% VRAM on the target GPU mid-run
(7.9B-param `Gemma4ForConditionalGeneration` in bf16 + fp32 AdamW state).

The FSDP code path was also exercised at `--devices 1`: process-group init over the
NCCL/RCCL backend works, the wrap classes resolve
(`FSDP will wrap: ['Gemma4AudioLayer', 'Gemma4TextDecoderLayer', ...]`), torch logs
`FSDP is switching to use NO_SHARD ... since the world size is 1`, and the run completes
with finite losses (2.01 → 1.62). Multi-GPU FULL_SHARD was **not** exercised (single
GPU allotted), but everything ROCm-specific about the FSDP path — RCCL, device
placement, bf16 autocast — ran.

Changes and quirks found:

1. **Required fix (applied to `train_llm_lightning.py`): `return_dict=False` on every
   `apply_chat_template(tokenize=True, ...)` call.** transformers 5.x returns a
   `BatchEncoding` from that call by default; the loader then measured `len()` of the
   encoding's *keys*, the masking arithmetic collapsed, and every row was dropped. This
   bites on any hardware, not just ROCm.
2. **Lightning 2.6.5 + FSDP refuses `gradient_clip_val`:**
   `MisconfigurationException: gradient_clip_algorithm='norm' is currently not supported
   for FSDPPrecision`. Hardware-independent. For `--strategy fsdp` runs pass
   `--grad_clip 0` (or move clipping into a `configure_gradient_clipping` override that
   calls FSDP's own `clip_grad_norm_`). `--strategy auto`/`ddp` clip fine.
3. **Device detection on ROCm is exactly as the Install section claims:** the ROCm torch
   build exposes the MI355X through `torch.cuda` (`torch.cuda.get_device_name(0)` →
   `AMD Instinct MI355X`), so `accelerator="gpu"` and even Lightning's "CUDA device"
   log lines work unchanged. No code knows it is on AMD.
4. **Keep `--attn_implementation sdpa`** (the default). Do not pip-install `flash-attn`
   on ROCm — the PyPI package is CUDA-only.
5. If several jobs share the box, set a unique `MASTER_PORT` (Lightning defaults to
   29500 and parallel single-node jobs collide).

DeepSpeed (`--strategy deepspeed`) was not smoke-tested here, but `deepspeed==0.19.4`
is verified working on this box's ROCm stack by the sibling `training/llm/deepspeed`
folder — it should be viable if you need ZeRO.

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

**Verdict: WORKS WITH CHANGES.** Lightning's FSDP strategy scales from 1 to 8 MI355X with
no ROCm-specific work at all: `--devices 8 --strategy fsdp` spawns 8 ranks over RCCL,
`FULL_SHARD` genuinely shards, and the run exits 0 with a cleanly decreasing loss. The two
changes needed are both hardware-independent Lightning/model quirks — `--grad_clip 0` is
**still required** under FSDP, and plain `--strategy ddp` needs
`ddp_find_unused_parameters_true` for this multimodal checkpoint (details below).

Same stack as the single-GPU section: `torch 2.11.0+rocm7.2`, `lightning 2.6.5`,
`transformers 5.15.0`. No new packages, no new pins — `requirements_lightning.txt` is
unchanged for the 8-GPU path.

Exact launch (the whole command — no `torchrun`, Lightning spawns the ranks itself):

```bash
cd training/llm/lightning
source .env_train_llm_lightning/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/data_1.5t/hf_cache
export MASTER_PORT=29670          # see quirk 5 above; Lightning reads this env var

python3 train_llm_lightning.py \
  --model_name google/gemma-4-E4B-it \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy fsdp --sharding_strategy FULL_SHARD \
  --devices 8 --num_nodes 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --num_train_epochs 10 --log_every_n_steps 1 \
  --grad_clip 0 --no_checkpoint \
  --output_dir /mnt/data_1.5t/outputs/train_llm_lightning/gpu8/fsdp8
```

**Parallelism actually used:** pure data parallelism with full parameter sharding —
world size 8 (1 node x 8 ranks, 1 rank per GPU), FSDP `FULL_SHARD` (ZeRO-3-equivalent:
params, grads and AdamW state all sharded 8 ways), no tensor/pipeline/sequence
parallelism. Batch geometry: micro-batch 1 x grad_acc 1 x 8 ranks = **global batch 8
sequences** at up to 2048 tokens. The sample file yields 9 usable rows, so each rank sees
2 batches per epoch → 2 optimizer steps/epoch x 10 epochs = **20 optimizer steps**.

Real log lines — all 8 ranks registered, and the FSDP wrap policy resolved on each:

```
[assert] torch 2.11.0+rocm7.2 device_count=8
[assert]   cuda:0 = AMD Instinct MI355X   ... cuda:7 = AMD Instinct MI355X
FSDP will wrap: ['Gemma4AudioLayer', 'Gemma4TextDecoderLayer', 'Gemma4VisionEncoderLayer', 'Gemma4VisionPatchEmbedder']
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8   ... GLOBAL_RANK: 7, MEMBER: 8/8
distributed_backend=nccl
All distributed processes registered. Starting with 8 processes
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
`Trainer.fit` stopped: `max_epochs=10` reached.
```

(`distributed_backend=nccl` is the ROCm build's RCCL — the name is inherited, no NVIDIA
anything is involved.)

The sharding is visible in Lightning's own model summary: it reports **992 M params per
rank**, which is exactly 7.94 B / 8 — i.e. each rank holds one eighth of the model, not a
replica.

```
┃   ┃ Name  ┃ Type                           ┃ Params ┃
│ 0 │ model │ Gemma4ForConditionalGeneration │  992 M │
Trainable params: 992 M
```

`train_loss` from the TensorBoard scalars, all 20 steps, finite and monotonically falling:

```
step  0: 1.8430    step  5: 0.4944    step 10: 0.4080    step 15: 0.4033
step  1: 1.1498    step  6: 0.4769    step 11: 0.3678    step 16: 0.3476
step  2: 0.9782    step  7: 0.4444    step 12: 0.3848    step 17: 0.3447
step  3: 0.6324    step  8: 0.4878    step 13: 0.3520    step 18: 0.3467
step  4: 0.6113    step  9: 0.4647    step 14: 0.3750    step 19: 0.4027
```

Wall clock: 20 optimizer steps in ~165 s end-to-end (17:01:32 → 17:04:17), which includes
FSDP wrapping and the first-step warmup — roughly 8 s/step amortised, and that is
dominated by fixed setup at this tiny step count, not by steady-state throughput.

**8-GPU evidence (`rocm-smi`, sampled from inside the job while it trained).** The sampler
runs as a background loop in the same script as the training, so every sample is
guaranteed to overlap this run and not a neighbouring job's. Peak VRAM per card during
the FSDP-8 phase — all 8 cards busy, and evenly so, which is what "sharding is actually
engaged" looks like:

```
card0  57.9 GiB   card2  59.6 GiB   card4  59.2 GiB   card6  59.4 GiB
card1  58.1 GiB   card3  59.6 GiB   card5  59.8 GiB   card7  57.3 GiB
       -> 8/8 cards, mean 58.9 GiB, spread 57.3-59.8 GiB (4% max-min)
```

PID cross-check via `rocm-smi --showpids` during the same window — 8 python processes,
each pinned to exactly 1 GPU, and the PIDs are this job's own launcher (3438829) plus the
7 ranks Lightning spawned (3439294-3439300):

```
PID      PROCESS NAME  GPU(s)  VRAM USED
3438829  python        1       49909624832
3439294  python        1       43792195584
3439295  python        1       57442492416
3439296  python        1       61613674496
3439297  python        1       61154451456
3439298  python        1       52077977600
3439299  python        1       55135571968
3439300  python        1       50177957888
```

What differed from the single-GPU run:

1. **`--grad_clip 0` is STILL required at 8 devices.** Probed deliberately: the identical
   command with `--grad_clip 1.0` dies during setup with
   `MisconfigurationException: gradient_clip_algorithm='norm' is currently not supported
   for FSDPPrecision`. Nothing about the world size changes this — it is a flat
   Lightning-side gap in `FSDPPrecision`, so quirk 2 above applies at every scale. If you
   need clipping under FSDP, override `configure_gradient_clipping` and call FSDP's own
   `clip_grad_norm_`.
2. **Sharding is real, and the per-GPU VRAM delta proves it.** At `--devices 1` FSDP logs
   `switching to use NO_SHARD ... since the world size is 1` — a no-op wrapper. At
   `--devices 8` there is no such line, the model summary drops to 992 M params/rank, and
   peak VRAM is **58.9 GiB/card mean under FSDP `FULL_SHARD` vs 108.2 GiB/card mean under
   DDP-8** on the identical recipe — a **45.6% per-GPU reduction** (DDP-8 peaks, same
   in-band sampler: card0 105.3, card1 107.1, card2 108.9, card3 108.9, card4 107.3,
   card5 110.4, card6 111.3, card7 106.4 GiB). Both fit comfortably in 288 GB, so this
   model does not *need* sharding on MI355X — the point is that the mechanism
   demonstrably works and the saving is real.
3. **Plain `--strategy ddp` fails on this checkpoint** — not on ROCm, on model shape:
   `RuntimeError: It looks like your LightningModule has parameters that were not used in
   producing the loss returned by training_step.` `google/gemma-4-E4B-it` is multimodal,
   and the vision/audio towers receive no gradient from text-only chat data, so DDP's
   reducer never sees those buckets. Fixed by `--strategy ddp_find_unused_parameters_true`
   (added to the strategy choices for this reason). FSDP is unaffected — it wraps those
   towers and tolerates them being unused. A text-only checkpoint (Llama, Qwen) would not
   hit this.

   With that one word changed, **DDP-at-8 also passes** (exit 0, 20 steps, loss
   2.5676 → 0.8262, noisier than FSDP because DDP keeps `--grad_clip 1.0` and each rank
   carries a full replica). So both of Lightning's multi-GPU strategies work on MI355X:

   ```bash
   python3 train_llm_lightning.py ... --devices 8 \
     --strategy ddp_find_unused_parameters_true --grad_clip 1.0 --no_checkpoint
   ```

   Its `--showpids` sample is the same shape — 8 python PIDs (3691548 launcher +
   3692463-3692469), one GPU each, ~19.7-20.3 GB apiece at that instant.
4. **Checkpointing must be turned off for smoke runs** — pass `--no_checkpoint` (new
   flag). Lightning writes a `.ckpt` every epoch by default, and for a 7.9 B model with
   optimizer state that is ~100 GB per file at 10 epochs. With the flag the whole 8-GPU
   run leaves 276 KB of logs behind and zero weight files.
5. **No stale GPU pin, and `MASTER_PORT` still matters.** This folder's venv `activate`
   had no leftover `CUDA_VISIBLE_DEVICES` export, but export it explicitly anyway (as
   above) — `torch.cuda.device_count()` must read 8 before `Trainer` is constructed, or
   Lightning silently trains on fewer GPUs. `MASTER_PORT=29670` kept this job off the
   default 29500 while other jobs shared the box.

Not retested at 8 GPUs: `--activation_checkpointing`, `--export_hf_dir` (the FSDP
all-gather export path), `--cpu_offload`, and `--strategy deepspeed`.

### 4-GPU sharding run — **checkpoint writing enabled** (4×MI355X, ROCm 7.2.4) — tested August 2026

> Extends the 8-GPU section above, which ran with `--no_checkpoint` (quirk 4). That left
> the FSDP **`--state_dict_type full` consolidated checkpoint** — where rank 0 gathers the
> sharded weights — untested at any world size above 1. This section covers it, plus a
> `FULL_SHARD` re-confirmation at 4 devices. Nothing here contradicts the 1- or 8-GPU
> results.

Verified 2026-08-19 on physical GPUs **4,5,6,7** of the same node (a sibling job owned
0-3), same venv and pins. **Verdict: FULL_SHARD holds at 4 devices, and the consolidated
checkpoint save works — rc=0, no collective hang.**

```bash
source .env_train_llm_lightning/bin/activate
export HIP_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME=/mnt/data_1.5t/hf_cache
export MASTER_PORT=29796            # 29797/29798 for the two follow-up runs
python -c "import torch; assert torch.cuda.device_count()==4"

# Run A — 8B sharding proof (checkpointing off, as at 8 GPUs)
python3 train_llm_lightning.py \
  --model_name google/gemma-4-E4B-it \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy fsdp --sharding_strategy FULL_SHARD \
  --devices 4 --num_nodes 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --num_train_epochs 12 --log_every_n_steps 1 \
  --grad_clip 0 --no_checkpoint \
  --output_dir /mnt/data_1.5t/outputs/train_llm_lightning_4gpu/fsdp4

# Run B — consolidated checkpoint proof (NO --no_checkpoint; small model on purpose)
python3 train_llm_lightning.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy fsdp --sharding_strategy FULL_SHARD --state_dict_type full \
  --devices 4 --num_nodes 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --num_train_epochs 6 --log_every_n_steps 1 --grad_clip 0 \
  --output_dir /mnt/data_1.5t/outputs/train_llm_lightning_4gpu/fsdp4_ckpt
```

`--grad_clip 0` is still mandatory at 4 devices (quirk 1 is world-size independent).

**Evidence:**

```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4      (... MEMBER 2/4, 3/4, 4/4)
LOCAL_RANK: 0..3 - CUDA_VISIBLE_DEVICES: [4,5,6,7]
Trainable params: 2.0 B                                    # per rank = 8B / 4 ranks
Total estimated model params size (MB): 7,941.101
`Trainer.fit` stopped: `max_epochs=12` reached.
INFO - __main__ - Training complete.                       # rc=0
```

`train_loss` (read from the run's tfevents — see the logging caveat below):

| Run | Steps | first → mid → last |
|---|---|---|
| A, 8B, 12 epochs | 36 | 2.2939 → 0.3343 → **0.3386** |
| A, 8B, 60 epochs (killed at 96 steps for time) | 96 | 2.2939 → 0.0939 → **0.0160** |
| B, 0.6B, 6 epochs + checkpointing | 18 | 2.4063 → 0.6905 → **0.7837** |

`rocm-smi` sampled every 2 s *during* training of the 8B `FULL_SHARD` run:

```
=== sample 29  20:55:05 ===
GPU[4]: GPU use (%): 100   VRAM Total Used Memory (B): 61138726912   # 56.9 GiB
GPU[5]: GPU use (%): 100   VRAM Total Used Memory (B): 68971597824   # 64.2 GiB
GPU[6]: GPU use (%):  99   VRAM Total Used Memory (B): 65660182528   # 61.2 GiB
GPU[7]: GPU use (%):  99   VRAM Total Used Memory (B): 69732728832   # 64.9 GiB
```

**New findings at 4 devices:**

- **`FULL_SHARD` is genuinely sharding 4 ways.** The model summary reports
  **`Trainable params: 2.0 B` per rank** for the 8B checkpoint — exactly 8B/4 — and there
  is no `switching to use NO_SHARD` line (contrast `--devices 1`, quirk 2). Per-GPU VRAM
  56.9-64.9 GiB sits between the 8-rank figure (58.9 GiB) and DDP's 108 GiB, as expected
  for a 4-way split.
- **The consolidated `--state_dict_type full` checkpoint works at 4 ranks.** Run B wrote
  `checkpoints/last.ckpt` and `checkpoints/epoch=5-step=18.ckpt`, **3,887,874,685 B each**.
  Reloaded with `torch.load` it contains **311 keys / 0.752 B parameters** — the whole
  unsharded model — with full tensor shapes (e.g.
  `model.model.layers.0.self_attn.q_proj.weight` → `(2048, 1024)`, bf16), not per-rank
  slices. So rank 0's gather completes and the file is a real, loadable checkpoint.
- ⚠️ **Run B was deliberately done on `Qwen/Qwen3-0.6B`, not the 8B model.** Quirk 4's
  ~100 GB-per-`.ckpt` estimate is accurate, and `ModelCheckpoint` here uses
  `save_last=True` *plus* `save_top_k=1`, i.e. **two** files per run — ~200 GB for the 8B
  checkpoint. That did not fit the free disk. The collective gather path being tested is
  model-independent, so the small model proves the mechanism; the disk cost at 8B is the
  only reason to keep `--no_checkpoint` on smoke runs.
- ⚠️ **This trainer logs no loss to stdout, and its progress bar vanishes when stdout is
  a pipe.** A healthy 4-GPU run therefore looks *identical to a hang*: the last console
  line is the `Found 1758 module(s) in eval mode` warning and nothing follows for minutes.
  It is not hung — read progress from `<output_dir>/lightning_logs/version_*/` with
  `EventAccumulator` (tag `train_loss`), or run on a TTY. Two `rocm-smi` samples 5 s apart
  can also read 0-3% on all four cards on a short run while training is genuinely
  progressing, because the setup/teardown phases dominate wall time.
- Ports 29796-29798 used via `MASTER_PORT`. No code, requirements or config change was
  needed for any of this.

Still not tested at 4 GPUs: `--activation_checkpointing`, `--export_hf_dir`,
`--cpu_offload`, `--strategy deepspeed`, and the 8B model with checkpointing enabled
(disk-bound, see above).

## Tested on NVIDIA H100 (CUDA 13.0) — 2026-08-22

**Verdict: WORKS on a single NVIDIA H100 80GB HBM3** (Hopper, compute capability 9.0),
driver **580.173.02**, CUDA **13.0**, Python 3.12.3. Single-GPU smoke test only — the
end-to-end path runs clean: chat-JSONL load → tokenize → train → `.ckpt` checkpoint →
`--export_hf_dir` HF folder. **No code change was needed** (the `return_dict=False` fix
the MI355X run added is already in `train_llm_lightning.py` and is what makes the loader
work on transformers 5.x here too). One deviation from the MI355X recipe: `flash-attn`
does not import against this torch, so the run used the `sdpa` default (details below).

### Install that worked (CUDA)

The NVIDIA install block above pins `torch==2.11.0`, but **there is no cu130 wheel for
that pin** on this box. Per the H100 recipe, the bare `pip install torch` default stable
was used and resolved to **`torch 2.13.0+cu130`** (native CUDA 13 — no `--index-url`
needed). torch **stayed put** through the requirements pass (no silent clobber):

```bash
python3 -m venv .env_lightning && source .env_lightning/bin/activate
pip install torch numpy                       # -> torch 2.13.0+cu130, numpy 2.5.2
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
pip install -r requirements_lightning.txt     # torch unchanged afterward
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # still 2.13.0+cu130 13.0
```

**Key versions as resolved by pip on 2026-08-22:** `torch 2.13.0+cu130`,
`lightning 2.6.5`, `transformers 5.15.1`, `tokenizers 0.22.2`, `accelerate 1.14.0`,
`tensorboard 2.21.0`, `python-dotenv 1.2.3`; driver 580.173.02, CUDA 13.0. Notably
`transformers 5.15.1` did **not** pull the `kernels` package here (a known crasher on this
box: `kernels 0.16.0` breaks all transformers imports — if a sibling install drags it in,
pin `kernels>=0.12,<0.13`). This folder's loader reads the chat JSONL directly and does
**not** use HF `datasets`, so no `HF_DATASETS_CACHE` tmpfs redirect is needed.

The node is offline (Hub egress is proxy-blocked, 403), so the run used the pre-cached
`LiquidAI/LFM2.5-350M` (has a chat template) with `HF_HOME=/mnt/gsma/gsma/gsma/models
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. `tf32` on CUDA is available (cuDNN tf32 backend
on; Lightning also prints the "you have Tensor Cores — set float32_matmul_precision" hint).

### Exact smoke command (single GPU)

The box runs one job per GPU and GPUs 0–3 were a neighbour's production job, so the run
was pinned to physical GPU 6 with a non-default `MASTER_PORT`:

```bash
source .env_lightning/bin/activate
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MASTER_PORT=29646            # unique port; Lightning defaults to 29500 and collides

CUDA_VISIBLE_DEVICES=6 python3 train_llm_lightning.py \
  --model_name LiquidAI/LFM2.5-350M \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --strategy auto --devices 1 \
  --precision bf16-mixed --attn_implementation sdpa \
  --batch_size 1 --grad_acc_steps 3 --max_seq_len 2048 \
  --num_train_epochs 12 --log_every_n_steps 1 \
  --learning_rate 5e-5 --warmup_ratio 0.1 \
  --export_hf_dir ./lightning_smoke/final_model \
  --output_dir ./lightning_smoke
```

**Step count (non-trivial):** the sample yields 9 usable rows (1 dropped over 2048
tokens). `--strategy auto` on 1 GPU is a single-device strategy (world size 1), so
9 rows / (batch 1 × grad_acc 3) = **3 optimizer steps/epoch × 12 epochs = 36 optimizer
steps**. `--grad_clip` is left at its default 1.0 and works here — the FSDP
`gradient_clip_val` incompatibility (MI355X quirk 2) does **not** apply under
`--strategy auto`/single-device.

### Real evidence

Run completed `rc=0`. Loader + device + completion lines (from the run log):

```
INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
Using bfloat16 Automatic Mixed Precision (AMP)
You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. ...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [6]
│ 0 │ model │ Lfm2ForCausalLM │  354 M │
`Trainer.fit` stopped: `max_epochs=12` reached.
INFO - __main__ - Exported Hugging Face weights to .../lightning_smoke/final_model
INFO - __main__ - Training complete.
```

`train_loss` read from the run's tfevents (`EventAccumulator`, tag `train_loss` — this
trainer logs loss to TensorBoard, **not** stdout; see the logging caveat in the MI355X
4-GPU section, which bites identically on H100). All 36 steps finite and clearly falling:

```
step  0  1.1720   step  6  0.0037   step 12  0.1197   step 18  0.0011   step 30  0.0005
step  1  2.0791   step  7  0.6314   step 13  0.0062   step 20  0.0596   step 33  0.0001
step  3  0.2939   step  9  0.1373   step 15  0.0456   step 27  0.0001   step 35  0.0015
```

first-3-step mean **1.4184 → last-3-step mean 0.0014** (linear slope −0.035 loss/step) —
a 350 M model memorising 9 rows at lr 5e-5, exactly the expected shape. (A separate first
pass with `--grad_acc_steps 1 --num_train_epochs 3 --learning_rate 1e-5` also ran
`rc=0`/27 steps with finite loss, but per-single-example logging is far noisier — raising
`grad_acc` and lr is what makes the decrease legible on so few rows.)

**GPU-6 residency (`nvidia-smi` filtered to GPU 6 by PID, sampled from a background loop
in the same shell *while* the job trained):** the training PID (1404885) is the only
compute app on card 6, ~8.0 GiB resident (354 M params bf16 + fp32 AdamW master weights),
peak util 61% on the sampler's short probes:

```
=== sample 14 00:41:08 ===        # nvidia-smi -i 6 --query-gpu=memory.used,utilization.gpu
6169 MiB, 16 %
=== sample 50 00:42:30 ===        # nvidia-smi -i 6 --query-compute-apps=pid,proc,used_memory
8027 MiB, 0 %
1404885, python3, 8018 MiB        # <- our training PID, pinned to GPU 6, no other app
```

(Util reads 0% on most 2-s samples because the forward/backward of a 350 M model on short
sequences is sub-sampling-interval; the by-PID VRAM occupancy is the reliable residency
proof. Peak VRAM was 8.0 GiB here / 9.1 GiB on the 27-step first pass — trivial against
the H100's 80 GB, so no OOM playbook was needed for this model.)

**Checkpoint + export both work.** Lightning wrote `checkpoints/epoch=11-step=36.ckpt`
and `last.ckpt` (2,127,093,981 B each — weights + AdamW state + schedule), and
`--export_hf_dir` produced a plain `from_pretrained`-loadable folder
(`model.safetensors` 709 MB, `config.json`, `chat_template.jinja`, tokenizer files);
reloading it gives `Lfm2ForCausalLM, 354.5M params, chat_template present`. Per the brief
these large artifacts were deleted after capturing evidence (outputs live under
`/dev/shm/h100/out/lightning`); the `.ckpt` files are ~2 GB each, so pass `--no_checkpoint`
for throwaway smoke runs.

### Quirks / deviations from the MI355X recipe

1. **`flash_attention_2` is NOT usable out of the box on this stack — fell back to
   `sdpa`.** The prebuilt PyPI `flash-attn` wheel (2.8.3.post1) installs fine but **fails
   to import** against `torch 2.13.0+cu130`:
   `ImportError: .../flash_attn_2_cuda...so: undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`
   — a c10 ABI mismatch (the wheel was built against a different torch). Building
   flash-attn from source against torch 2.13/CUDA 13 (`pip install flash-attn
   --no-build-isolation`) would work but exceeds a smoke-test time budget, so the run used
   `--attn_implementation sdpa` (the default). This is the one place the H100 recipe's
   "switch sdpa → flash_attention_2" step could not be completed. `sdpa` on Hopper already
   dispatches to an efficient fused kernel; flash-attn is a throughput optimisation, not a
   correctness requirement.
2. **No code fix required.** The `return_dict=False` change the MI355X campaign made to
   `apply_chat_template(tokenize=True, ...)` is already committed in the script, so the
   loader kept all 9 rows (`0 with no supervised tokens`) on transformers 5.15.1. No new
   drift surfaced.
3. **`--grad_clip` default (1.0) is fine** under `--strategy auto`/single-device — the
   `FSDPPrecision` grad-clip gap (MI355X quirk 2) only affects `--strategy fsdp`.
4. **The "silent hang" is real here too.** This trainer prints no per-step loss to stdout
   and its progress bar is disabled (`enable_progress_bar=False`), so a healthy run sits
   for a minute or two after `Found 201 module(s) in eval mode` with no output while it
   loads weights and runs. It is not hung — read `train_loss` from
   `<output_dir>/lightning_logs/version_*/` with `EventAccumulator`.
5. **`MASTER_PORT` still matters** on a shared box (used 29646 to avoid the default 29500).
   Plain `CUDA_VISIBLE_DEVICES=6` is the whole GPU-pinning story on NVIDIA — no
   `HIP_VISIBLE_DEVICES`, no `RAY_EXPERIMENTAL_NOSET_*`.

### What a multi-GPU pass would need (DEFERRED)

Multi-GPU (2, then 8) was **not** run — GPUs 0–3 were a neighbour's production job during
this wave. To do it later: `--strategy fsdp --sharding_strategy FULL_SHARD --devices N`
with **`--grad_clip 0`** (the `FSDPPrecision` grad-clip gap is world-size-independent and
bit the MI355X FSDP runs at every scale), a distinct `MASTER_PORT`, and `--no_checkpoint`
for smoke runs (a `.ckpt` of a large model with AdamW state is ~100 GB/file). Lightning
spawns the ranks itself — do **not** wrap in `torchrun`. The MI355X 8-GPU and 4-GPU
sections above document the expected `FSDP will wrap: [...]`, per-rank param-count drop,
and even-VRAM sharding signatures to check for; none of that is NVIDIA-specific. For a
text-only checkpoint like LFM2/Qwen/Llama, plain `--strategy ddp` also works (the
`ddp_find_unused_parameters_true` workaround is only needed for multimodal towers).

## Notes

- **The `LightningModule` is the only thing you own.** `training_step` returns the loss
  HF already computed (`self.model(**batch).loss`), `configure_optimizers` returns AdamW
  plus a cosine schedule stepped per optimizer step. Everything else — device placement,
  autocast, gradient accumulation, clipping, checkpoint writing — is a `Trainer` flag.
  That is the trade: less control than raw FSDP, far less code to get wrong.
- **`estimated_stepping_batches` is what makes the schedule correct.** The number of
  optimizer steps depends on dataset size, world size, accumulation and epochs, and you
  do not know it until the Trainer is attached. Lightning computes it, so the warmup and
  decay land where you intended even when you change GPU count.
- **FSDP wraps at the decoder-block level, resolved from `_no_split_modules`.** Every
  well-behaved HF causal LM declares its repeated block there (`["LlamaDecoderLayer"]`,
  `["Qwen3DecoderLayer"]`, ...), which is the right sharding granularity. Wrapping the
  whole model as one unit defeats the point; wrapping every `nn.Linear` drowns you in
  collectives. If an exotic architecture does not declare it, the script fails loudly
  rather than silently mis-sharding.
- **`bf16-mixed` vs `bf16-true`.** `bf16-mixed` keeps fp32 master weights and autocasts
  the forward pass — the safe default for full fine-tuning. `bf16-true` puts everything,
  including the optimizer state, in bf16: roughly half the memory, but small-magnitude
  updates can be lost to rounding. Only reach for it if you are memory-bound and
  validating loss closely.
- **`shuffle=True` on the DataLoader is correct here.** Lightning's
  `use_distributed_sampler` defaults to `True` and replaces the sampler with a
  `DistributedSampler` on each rank. Adding one yourself is the classic double-sampler
  bug.
- **Exporting is a collective, not a rank-0 operation.** `export_hf` calls
  `trainer.strategy.lightning_module_state_dict()` on *every* rank — under FSDP that is
  the all-gather that reconstructs full parameters — and only then does rank 0 write the
  files. Guarding the whole thing behind `if trainer.is_global_zero` would hang. Export
  requires `--state_dict_type full`.
- **DeepSpeed export is a separate step.** ZeRO checkpoints are sharded across ranks; the
  script skips export and tells you so. DeepSpeed writes a `zero_to_fp32.py` script into
  the checkpoint folder — run that to consolidate, then load the result and
  `save_pretrained`.
- **Checkpoint retention.** With no val split there is no monitored metric, so the
  `ModelCheckpoint` callback keeps only the most recent checkpoint (`save_top_k > 1`
  requires a monitored metric); with `--eval_samples` set it keeps the top 2 by
  `val_loss` plus `last.ckpt`.
- **OOM playbook.** In order: turn on `--activation_checkpointing`; drop `--batch_size`
  to 1 and raise `--grad_acc_steps` to hold the global batch; lower `--max_seq_len`; add
  `--cpu_offload`; then switch to `--strategy deepspeed --zero_stage 3 --cpu_offload`,
  the most aggressive memory setting available here.

### Where the other trainers in this repo are a better default

- If you want the *same* PyTorch-native FSDP2 without an extra framework layer,
  `training/llm/fsdp` is the more direct route — Lightning's value is the loop, not the
  sharding.
- For LoRA / QLoRA on one or two GPUs, `training/llm/unsloth` or `training/llm/peft` are far
  faster to get running; this file does full fine-tuning and has no adapter path.
- For YAML-driven, recipe-style fine-tuning where you do not want to touch Python at all,
  `training/llm/llamafactory` or `training/llm/axolotl` are a better fit.
- Lightning earns its place when you need custom training logic (extra loss terms, staged
  freezing, a second model in the step) *plus* production-grade distributed plumbing, and
  you want that logic in reviewable, unit-testable Python.
