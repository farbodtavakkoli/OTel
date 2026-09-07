# training/llm/redhat — OSFT continual fine-tuning via Red Hat's training_hub

## Overview & when to use

Continual fine-tuning with **OSFT** (Orthogonal Subspace Fine-Tuning) via Red Hat's [`training_hub`](https://github.com/Red-Hat-AI-Innovation-Team/training_hub), on a single-node multi-GPU setup. OSFT adapts a base or instruct model to a new domain — e.g. observability / OTel telemetry — **without catastrophic forgetting**, which makes it ideal for:

- Adapting a base model to specialized domains,
- Adding new knowledge without degrading general capabilities,
- Fine-tuning without complex replay mechanisms or data mixing.

Works with any HuggingFace causal LM supported by training_hub's dependencies. `train_llm_redhat.py` wraps `training_hub.osft` with argument logging, an optional live speed/ETA monitor, and an EOS-override staging step for models whose chat template ends turns with a non-default token.

Files in this folder:
- `train_llm_redhat.py` — the OSFT training entrypoint (wraps `training_hub.osft`).
- `speed_monitor.py` — optional live speed/ETA reporter (`--speed-steps N`).
- `check_memory.py` — post-hoc run summary (peak memory, step, duration, loss plot).
- `data/OTel_LLM_sample_10.jsonl` — 10-row sample dataset (default `--data-path`).
- `requirements_redhat.txt` — this folder's direct deps (training_hub manages the rest).

> **Tested topology:** single node, **8 GPUs — measured**, on **AMD MI355X (ROCm 7.2.4)** with `torch==2.11.0+rocm7.2` (see "8-GPU run" below for the evidence; an earlier 2-GPU smoke run is documented above it). 8×H100 is the upstream reference environment and is *not* re-verified here. Multi-node is exposed upstream via `nnodes`/`rdzv_*` but has not been tested here.

## Install

`training_hub` is on PyPI (upstream: `pip install training-hub`; the base package installs without the CUDA-only GPU extras — see github.com/Red-Hat-AI-Innovation-Team/training_hub). Use Python 3.12+ in a `venv`/`uv` environment:

```bash
python3.12 -m venv ~/.venv && source ~/.venv/bin/activate
```

### NVIDIA (CUDA)

Install in order — upstream recommends sequential installs because the `[grpo]` extras constrain torch/vllm/transformers and can conflict with `[cuda]` if solved together:

```bash
pip install torch torchvision
pip install training-hub[grpo,lora]
pip install training-hub[cuda] --no-build-isolation   # builds flash-attn — the slow/finicky step
pip install -r requirements_redhat.txt                # this folder's extra direct deps
```

The flash-attn build in the `[cuda]` step is the one most likely to give trouble; make sure `CUDA_HOME` points at your toolkit first:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### AMD (ROCm)

This folder's reference environment was AMD MI355X with ROCm 7 (`torch==2.11.0`) — the code path is the same; only the install differs:

1. Install a **ROCm build of PyTorch** (from the ROCm wheel index for your ROCm version) instead of the CUDA wheels.
2. Install `training-hub[grpo,lora]` as above.
3. **Skip the `[cuda]` extra** — it exists to build CUDA flash-attn (upstream documents it as the CUDA-support step requiring `--no-build-isolation`); on ROCm use the attention implementation your ROCm PyTorch/flash-attn ROCm port provides instead. `CUDA_HOME` is not applicable.
4. `pip install -r requirements_redhat.txt`.

Follow the upstream repo's README if the install steps have changed.

### ✅ Platform notes — MI355X, ROCm 7.2.4

Validated end-to-end on 8× AMD Instinct MI355X (gfx950, 288 GB), ROCm 7.2.4, Ubuntu,
Python 3.12.3 — smoke run used 2 GPUs. The MI355X claim above **still holds**, with two
small additions the generic AMD steps don't mention (liger + `TESTING=true`, below).

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints / training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Validated install:

```bash
cd training/llm/redhat
python3 -m venv .env_redhat && source .env_redhat/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install "training-hub[grpo,lora]"      # keeps the ROCm torch (requires only torch>=2.6)
pip install "liger-kernel>=0.5.10"         # REQUIRED on ROCm — see quirk 1 below
pip install -r requirements_redhat.txt
```

Versions landed: `torch 2.11.0+rocm7.2`, `training-hub 0.9.7`, `instructlab-training
0.16.2`, `rhai-innovation-mini-trainer 0.8.1`, `transformers 5.5.0`, `trl 0.24.0`,
`liger-kernel 0.8.2`. **No flash-attn installed** — do not pip-install it on ROCm.

Smoke command (bundled 10-row sample, 1 epoch = 5 steps):

```bash
export TESTING=true    # REQUIRED on ROCm — see quirk 2 below
python3 train_llm_redhat.py \
  --model-path google/gemma-4-E4B-it \
  --ckpt-output-dir checkpoints_smoke \
  --num-epochs 1 --effective-batch-size 2 --nproc-per-node 2 \
  --eos-token "<turn|>"
```

**Expected output** (finite, decreasing loss; `rocm-smi` mid-run shows both GPUs active,
46–84 GB VRAM used, 100% GPU busy during OSFT weight reconstruction):

```
Epoch 1: ──━━━━━━━━  20% │ 1/5 │ loss: 10.7442 │ lr: 5.00e-06 │ 112 tok/s
Epoch 1: ────────── 100% │ 5/5 │ loss: 6.0593 │ lr: 4.77e-07 │ 2385 tok/s
✅ Saved model at 10.0 samples in 162.99 seconds
```

**Quirks found on ROCm (both required, neither in the generic AMD steps above):**

1. **`liger-kernel` must be installed explicitly.** The script hardcodes
   `use_liger=True`, but upstream packages liger only inside the CUDA-only `[cuda]`
   extra, so the base install lacks it and mini-trainer raises at startup. liger is
   Triton-based and runs fine on ROCm — `pip install "liger-kernel>=0.5.10"`.
2. **`export TESTING=true` before running.** mini-trainer 0.8.1 hard-requires
   `import flash_attn` for standard causal LMs and only falls back to SDPA when
   `TESTING=true` (one code site, `mini_trainer/setup_model_for_training.py`). With no
   flash-attn on ROCm this env var is the supported no-code-change route; PyTorch ROCm's
   SDPA uses AOTriton flash attention on gfx950. (Models mini-trainer classifies as
   SDPA-only — M-RoPE / timm-vision — don't need it.)
3. Benign at import time: `torchao` prints `Failed to load ..._C_mxfp8...so` /
   `_C_cutlass_90a...so` warnings (CUDA-only kernels); harmless on ROCm.
4. `pip` resolves PyPI `triton 3.7.1` over the wheel-index `triton-rocm 3.6.0` (pulled
   in via unsloth/xformers). No breakage observed — liger's Triton kernels compiled and
   ran on gfx950.
5. The gemma-4-E4B-it smoke checkpoint is ~16 GB — point `--ckpt-output-dir` at a large
   disk.

**This path works with changes** — OSFT training via training_hub runs on MI355X /
ROCm 7.2.4 exactly as the reference-environment claim says, provided you add the
explicit `liger-kernel` install and `TESTING=true` (SDPA) listed above.

> Scope note: the smoke run above used **2 GPUs**, not 8 (`--nproc-per-node 2`). The full
> 8-GPU verification is below.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This path works with changes.** The recipe scales from 2 to all 8 MI355X GPUs with no
code edits — but two environment/data changes are mandatory (stale venv GPU pin, and a
dataset large enough to feed 8 ranks). Clean exit (code 0), all 8 ranks finished, no
teardown hang.

**Launch** (serialized behind a machine-wide GPU mutex so the job owns all 8 GPUs; the
runner script sources the venv, overrides the GPU pin, asserts `device_count()==8`, and
samples `rocm-smi` in-band):

```bash
nohup flock -w 25200 /tmp/mi355x_gpu8.lock bash run8_redhat.sh \
    > $OUTPUT_DIR/train_llm_redhat/gpu8/run.log 2>&1 &
```

```bash
# inside the runner, AFTER `source .env_redhat/bin/activate`:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ← overrides the venv's stale 6,7 pin
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TESTING=true                              # ROCm: SDPA instead of flash-attn
python3 -c "import torch; assert torch.cuda.device_count()==8"

python3 train_llm_redhat.py \
  --model-path google/gemma-4-E4B-it \
  --data-path  $OUT/data_rep_640.jsonl \
  --ckpt-output-dir $OUT/ckpt8 --data-output-dir $OUT/data_output \
  --num-epochs 1 --effective-batch-size 64 --nproc-per-node 8 \
  --max-seq-len 4096 --max-tokens-per-gpu 8192 \
  --eos-token "<turn|>" --seed 42 --speed-steps 2
```

**Parallelism / batch geometry.** mini-trainer wraps the model with **FSDP2**
(`torch.distributed.fsdp.fully_shard`), launched by `torchrun --nnodes=1
--nproc-per-node=8 --rdzv-id=105 --rdzv-endpoint=127.0.0.1:29500` — i.e. the
`osft_params` single-node block, unchanged. World size 8; `effective_batch_size 64`
= 8 ranks x 4 samples x `grad_accum 2` (mini-trainer batches by token budget,
`max_tokens_per_gpu=8192`, ~72k–90k loss-counted tokens per step). Peak memory
reported per rank: 40.1 GB at step 1, 53.1 GB at step 10.

**Expected output** (`--speed-steps 2`; the run's own `training_metrics_0.jsonl` has the full series):

```
Epoch 1: ─━━━━━━━━━  10% │  1/10 │ loss: 10.8526 │ lr: 5.00e-06 │  5306 tok/s
Epoch 1: ────━━━━━━  40% │  4/10 │ loss:  5.8742 │ lr: 3.97e-06 │ 37704 tok/s
Epoch 1: ────────── 100% │ 10/10 │ loss:  4.8896 │ lr: 1.22e-07 │ 18575 tok/s
✅ Saved model at 640.0 samples in 167.21 seconds
```

The run should finish with exit code 0. Loss is finite and monotonically decreasing
(10.85 → 4.89); `grad_norm` 240 → 9.8.
OSFT itself runs: `Reconstructing OSFT weights, this may take a while...` over **294 OSFT
parameters**, with mini-trainer receiving `--osft --osft-unfreeze-rank-ratio=0.3
--osft-upcast-dtype=float32 --use-liger-kernels`.

**Checking all 8 GPUs are yours — `rocm-smi` sampled *in-band*** (run the sampler inside
the flock-held runner, so the sample cannot capture another job). Across consecutive
samples all 8 ranks should be resident. Peak VRAM used per card:

```
card0 181.8   card1 185.9   card2 180.2   card3 183.8
card4 221.0   card5 220.4   card6 221.7   card7 181.1     (GiB used of 288)
all 8 devices: SCLK 2388-2404 MHz, 312-322 W, GPU% 36-47 mid-step
```

**PID cross-check** (same sample). `rocm-smi --showpids` should list exactly 8 `python3`
holders, 35.3–41.5 GB each; `pgrep -af` in that same sample shows those
pids are `mini_trainer/train.py` children of `torchrun … --nproc-per-node=8`,
itself a child of the launcher `python3 train_llm_redhat.py … --nproc-per-node 8`
— i.e. the VRAM holders are your own processes, not a co-tenant's.

**What differs from the 2-GPU run:**

1. **A venv's `bin/activate` may pin `HIP_VISIBLE_DEVICES=6,7` / `CUDA_VISIBLE_DEVICES=6,7`**
   — a leftover from a 2-GPU session. Sourcing the venv and passing
   `--nproc-per-node 8` without re-exporting these silently runs on 2 GPUs (or fails
   rendezvous). **Always re-export both after `source …/bin/activate`.**
2. **The bundled 10-row sample cannot feed 8 ranks.** Replicate it ×64 → **640 rows**,
   written **outside the repo** (e.g. `$OUTPUT_DIR/train_llm_redhat/gpu8/data_rep_640.jsonl`).
   Sample survival: **640 in → 640 processed → "Saved model at 640.0 samples"**, i.e. 0 rows
   dropped by `max_seq_len=4096` filtering. Because only 10 rows are unique, the falling
   loss here is **a pipeline proof, not a learning result** (it is memorisation).
3. `--effective-batch-size 64` (instead of 2) so the global batch divides across 8 ranks;
   10 steps in 1 epoch. No batch-geometry assert is hit — mini-trainer's token-budget
   batching derives `grad_accum` itself.
4. Outputs redirected off the repo: `--ckpt-output-dir` / `--data-output-dir` under
   `$OUTPUT_DIR/train_llm_redhat/gpu8/` (delete the 16 GB checkpoint afterwards; see below).
5. Throughput scales as expected: peak **37.7k tok/s on 8 GPUs** vs 2.4k tok/s on the
   2-GPU smoke run (different batch geometry, so treat as indicative, not a clean speedup
   measurement).

**Gotchas worth knowing at 8 GPUs:**

- **Checkpointing cannot be turned off from the CLI.** `osft_params` hardcodes
  `checkpoint_at_epoch=True` and `save_final_checkpoint=True` (there is no flag), so a
  1-epoch smoke run still writes a full **16 GB** `hf_format/samples_640.0`. Point
  `--ckpt-output-dir` at a big disk and delete afterwards, or edit `osft_params` to
  disable both.
- **The rendezvous port is fixed at `127.0.0.1:29500`** in `osft_params`. On a shared
  machine a concurrent torchrun on that port collides; there is no CLI flag, so set
  `RDZV_ENDPOINT` (see the CUDA quirks below) or edit the script.
- No new packages and no pins changed for the 8-GPU path — `requirements_redhat.txt` is
  unchanged. `TESTING=true` and the explicit `liger-kernel` install are still required,
  exactly as for 2 GPUs.

### ✅ Platform notes — NVIDIA H100, CUDA 13.0

Single-GPU OSFT smoke validated on **1× NVIDIA H100 80GB HBM3** (Hopper cc 9.0),
driver **580.173.02**, **CUDA 13.0**, Ubuntu, Python 3.12.3. **This path works with changes**
— OSFT training via training_hub runs on H100, and unlike ROCm the CUDA `[cuda]` extra
**does build and engage flash-attn** (so `TESTING=true` is *not* needed), but the `[cuda]`
extra also drags in a `kernels` version that breaks the transformers import and must be
pinned back (see quirk 1).

**Model swap (offline node):** on a node where the folder's documented default
(`google/gemma-4-*-it`) is not in the HF cache and the Hub is proxy-blocked (403), the smoke
runs against a fully-cached
model instead: **`LiquidAI/LFM2.5-350M`** (arch `lfm2`; a hybrid conv+attention model —
its `[cuda]` deps `mamba-ssm`/`causal-conv1d` also built). training_hub/mini-trainer/OSFT
accepted the `lfm2` arch with no code change. **No `--eos-token` needed:** LFM2's chat
template closes turns with `<|im_end|>`, which already equals the tokenizer's `eos_token`
(contrast Gemma 4's `<turn|>`), so the EOS-staging path — which would `snapshot_download`
and fail offline — was correctly skipped.

CUDA install (to run offline, point `HF_HOME` at a populated cache and export
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; keep the venv on a fast local FS, e.g. tmpfs):

```bash
cd training/llm/redhat
python3 -m venv .env_redhat && source .env_redhat/bin/activate
pip install torch numpy                       # lands torch 2.13.0+cu130 (native CUDA 13)
pip install "training-hub[grpo,lora]"         # DOWNGRADES torch -> 2.11.0+cu130 (still CUDA 13, cuda.is_available()==True)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install "training-hub[cuda]" --no-build-isolation   # BUILDS flash-attn 2.8.3 (+mamba-ssm, causal-conv1d, liger) — ~24 min via nvcc, MAX_JOBS=32
pip install "kernels>=0.12,<0.13"             # QUIRK 1: undo the kernels 0.16 that [cuda] pulled (breaks transformers 5.5.0 import)
pip install -r requirements_redhat.txt
```

**Torch-clobber outcome:** `training-hub[grpo,lora]` uninstalls the base `torch 2.13.0+cu130`
and installs **`torch 2.11.0+cu130`** (its pin is `torch>=2.6`; pip resolves a cu130 wheel,
so it stays on **CUDA 13** — `torch.cuda.is_available()==True`, real bf16 matmul confirmed
on-GPU). This is the same 2.11.0 the MI355X run landed. The `[cuda]` extra does **not**
further clobber torch (verify `2.11.0+cu130` after it). No force-reinstall is required;
keep 2.11.0+cu130 to respect training_hub's pins.

Key versions landed: `torch 2.11.0+cu130`, `flash-attn 2.8.3.post1`, `training-hub 0.9.7`,
`instructlab-training 0.16.2`, `rhai-innovation-mini-trainer 0.8.1`, `transformers 5.5.0`,
`trl 0.24.0`, `accelerate 1.14.0`, `liger-kernel 0.8.2`, **`kernels 0.12.3`** (pinned down
from 0.16.0), driver 580.173.02.

Smoke command (bundled sample, single GPU, port + GPU pinned for a shared machine;
3 epochs to make the loss trend obvious on only ~9 unique rows):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # HF_HOME as exported above
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache   # QUIRK 2: keep datasets' .arrow cache OFF the read-only-ish network mount
export CUDA_VISIBLE_DEVICES=7                          # your assigned free GPU
export RDZV_ENDPOINT=127.0.0.1:29647                  # override the hardcoded :29500 (new env hook — see below)
python3 train_llm_redhat.py \
  --model-path LiquidAI/LFM2.5-350M \
  --ckpt-output-dir /dev/shm/out/ckpt_smoke \
  --data-output-dir /dev/shm/out/data_output \
  --num-epochs 3 --effective-batch-size 2 --nproc-per-node 1 \
  --max-seq-len 4096 --max-tokens-per-gpu 8192 \
  --learning-rate 5e-6 --seed 42 --speed-steps 1
```

**Expected output — finite, DECREASING loss over 15 optimizer steps** (5 steps/epoch × 3
epochs; `grad_norm` collapses 217 → ~12), as recorded in `training_metrics_0.jsonl`:

```
step  2 epoch 0 loss 5.0361 lr 4.95e-06 grad_norm 217.85
step  5 epoch 0 loss 2.7385 lr 4.17e-06 grad_norm 60.16
step 11 epoch 2 loss 1.5709 lr 1.25e-06 grad_norm 16.52
✅ OSFT Training completed successfully!   Most recent checkpoint: .../hf_format/samples_30.0
```

OSFT itself runs: `Reconstructing OSFT weights ...` over **18 OSFT parameters**; peak memory
**8.24 GB**, peak **23,877 tok/s**. A full HF checkpoint (`hf_format/samples_30.0`,
~809 MB safetensors) is written (delete it afterwards — see the checkpointing
gotcha, unchanged from the MI355X notes).

**flash-attn built AND engaged.** mini-trainer selects `flash_attention_2` on its own
(flash_attn imports, so its gate passes) — **no `TESTING=true`**. The log shows FA2's
"only supports fp16/bf16" info-warning because the model is loaded fp32 and the attention
runs under bf16 autocast; training is unaffected and the loss falls as above.

**GPU residency check — `nvidia-smi` sampled *in-band* (filtered with `-i <n>`)**, so the
sample can only see the pinned GPU. The training child (a child of `python3
train_llm_redhat.py`) should be the holder; VRAM climbs 0.5 → **8.1 GB** as the model
loads/trains:

```
7, 8887 MiB, 4 %, 127.53 W          # GPU 7: mem.used, util, power
<pid>, 8110 MiB, GPU-<uuid>         # the training pid holding 8.1 GB on GPU 7
```

(util% reads low because each step is ~1.2 s and the 4 s sampler cadence lands between the
fast steps of this tiny model; VRAM occupancy is the reliable residency signal here, and it
matches the run's reported 8.24 GB peak.)

**Quirks found on H100 / CUDA 13 (in addition to the ROCm ones above):**

1. **`kernels` must be pinned to `<0.13`.** The `training-hub[cuda]` extra transitively
   pulls **`kernels 0.16.0`**, but `transformers 5.5.0` pins `kernels<0.13,>=0.12.0` and
   its `integrations/hub_kernels.py` builds `LayerRepository(repo_id=…, layer_name=…)` with
   no `revision`/`version` — which `kernels 0.16` rejects with
   `ValueError: Either a revision or a version must be specified`, crashing *every* import of
   transformers. Fix: `pip install "kernels>=0.12,<0.13"` (lands 0.12.3). ROCm never hit
   this because it skips the `[cuda]` extra (which is what drags `kernels` in).
2. **Redirect the HF *datasets* cache off a network mount.** With `HF_HOME` on a shared
   network mount, `datasets` tries to write its `cache-*.arrow` under
   `$HF_HOME/datasets/...` and dies with `OSError: [Errno 1] Operation not permitted` (the
   mount rejects the op). Set `HF_DATASETS_CACHE=/dev/shm/...` (or any writable local dir);
   `HF_HOME` can stay on the mount for read-only model loads.
3. **`RDZV_ENDPOINT` env hook added.** `osft_params` hardcoded `rdzv_endpoint=127.0.0.1:29500`;
   on a shared machine concurrent torchrun jobs collide on that port. The script now reads
   `os.environ.get("RDZV_ENDPOINT", "127.0.0.1:29500")` — default behavior is unchanged; set
   `RDZV_ENDPOINT=127.0.0.1:<your-port>` to move it. (This is the only code change; it is
   NVIDIA-neutral and also helps the ROCm multi-tenant case.)
4. **flash-attn build is slow** (~24 min: flash-attn + mamba-ssm + causal-conv1d compile
   via nvcc). It succeeds on CUDA 13 with `CUDA_HOME=/usr/local/cuda-13.0` and
   `--no-build-isolation`. If it exceeds your time budget, `pip install "training-hub[grpo,lora]"`
   + `pip install "liger-kernel>=0.5.10"` + `export TESTING=true` gives the SDPA path (as on
   ROCm) with no flash-attn build.

**Multi-GPU on H100 is not covered here.** Only the single-GPU smoke was run. A multi-GPU
pass would use `--nproc-per-node N` (mini-trainer wraps FSDP2
via `torchrun`), needs `--effective-batch-size` ≥ world size and a dataset large enough to
feed N ranks (the 9-row sample must be replicated, exactly as in the MI355X 8-GPU run), and
should set `RDZV_ENDPOINT` to a free port.

**Summary: works with changes** on H100 / CUDA 13.0 — the two ROCm workarounds are *reversed*
(flash-attn builds & engages; no `TESTING=true`; plain `CUDA_VISIBLE_DEVICES`), at the cost of
one new pin (`kernels<0.13`), one env redirect (`HF_DATASETS_CACHE`), and the slow flash-attn
build. torch resolves to `2.11.0+cu130` (CUDA 13 intact).

### Verify

```bash
python -c "from training_hub import osft; print('training_hub OK')"
```

## Environment & secrets

Put a `dev.env` in this folder with your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_redhat.py` loads it via `load_dotenv("dev.env")` before the training_hub import, since transformers/datasets resolve tokens and cache locations at import time. Never commit `dev.env` — and if a token was ever committed, rotate it on the Hub.

**Performance note — HF cache on a RAM disk:** model loads are much faster when the HF cache sits on a RAM disk. Optionally export before running:

```bash
export HF_HOME=/dev/shm/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
```

Make sure the RAM disk is large enough for your model; leave these unset to use the default `~/.cache/huggingface`.

## Data

`--data-path` is a JSONL file of chat conversations — one `{"messages": [...]}` per line, the format training_hub/instructlab expects. The bundled sample `data/OTel_LLM_sample_10.jsonl` (the default) follows this schema and carries extra metadata columns (`flow`, `source_repo`, `source_id`, `source_spec_id`, `source_version`, `unmask`) — some mostly null; training_hub's processing keys on `messages` and tolerates extras. Swap in your own data with `--data-path path/to/train.jsonl`.

`--unmask-messages` (default on) trains on all turns; pass `--no-unmask-messages` for standard SFT (assistant turns only).

**EOS caveat (important for Gemma 4):** instructlab keys label-unmasking on the tokenizer's `eos_token`. If the chat template closes the assistant turn with a *different* token, the terminator never gets unmasked and the model never learns to stop. Gemma 4 closes turns with `<turn|>` (not `<eos>`), so pass `--eos-token "<turn|>"`. The script then stages a copy of the model with the corrected EOS — weights symlinked, only tokenizer/config regenerated — and trains from that; the base model is never modified.

## Run

### Smoke test

A quick end-to-end check on the bundled sample (small batch so 10 rows produce steps):

```bash
python3 train_llm_redhat.py \
  --model-path <hf-model-id-or-path> \
  --ckpt-output-dir checkpoints_smoke \
  --num-epochs 1 --effective-batch-size 2 --nproc-per-node 1
```

### Full run

```bash
nohup python3 train_llm_redhat.py \
  --model-path google/gemma-4-31b-it \
  --data-path path/to/train.jsonl \
  --ckpt-output-dir checkpoints \
  --num-epochs 4 --effective-batch-size 512 \
  --learning-rate 2e-5 --unfreeze-rank-ratio 0.35 \
  --max-seq-len 5056 --max-tokens-per-gpu 37312 \
  --nproc-per-node 8 --eos-token "<turn|>" --seed 42 \
  --speed-steps 20 \
  > train_llm_redhat.log 2>&1 &

tail -f train_llm_redhat.log
```

The run's exact arguments are recorded to `logs/<timestamp>/run_args.json`.

## Arguments

Every flag, with defaults (`python train_llm_redhat.py --help` shows the same):

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` | *(required)* | Base/instruct model — HF id or local path. |
| `--data-path` | `data/OTel_LLM_sample_10.jsonl` | Training JSONL. |
| `--ckpt-output-dir` | *(required)* | Where checkpoints (`hf_format/samples_*`) are written. |
| `--num-epochs` | 4 | Epochs. |
| `--unfreeze-rank-ratio` | 0.3 | OSFT adaptation vs preservation (0.2–0.35 typical). |
| `--effective-batch-size` | 128 | Global batch size (good for >10k-sample datasets). |
| `--learning-rate` | 5e-6 | Learning rate. |
| `--max-seq-len` | 4096 | Max sequence length in tokens. |
| `--max-tokens-per-gpu` | 8192 | Lower on OOM (large/large-vocab models); raise with headroom. |
| `--nproc-per-node` | 8 | Number of GPUs. |
| `--data-output-dir` | `data_output` | Processed-data / EOS-staging dir — point at a RAM disk (e.g. `/dev/shm`) for speed. |
| `--unmask-messages` / `--no-unmask-messages` | on | Train on all turns vs assistant-only (standard SFT). |
| `--eos-token` | None | EOS override for data processing + checkpoint (see Data). |
| `--seed` | 42 | Random seed. |
| `--speed-steps` | 0 | Print a live speed/ETA report every N steps (0 = off). |
| `--validation-split` | 0.0 | Held-out fraction, [0.0, 1.0); 0.0 disables validation. |
| `--validation-frequency` | None | Validate every N steps — required when split > 0. |
| `--save-best-val-loss` | off | Checkpoint whenever validation loss improves. |

## Output

Checkpoints are written under `<ckpt-output-dir>/hf_format/samples_*` (plus `samples_*_best_val_loss` when `--save-best-val-loss` is set). On success the script prints the most recent checkpoint path. Run arguments land in `logs/<timestamp>/run_args.json`; the backend writes step metrics to `<ckpt-output-dir>/training_metrics_0.jsonl`.

## Monitoring

- **Live (`--speed-steps N`)** — `speed_monitor.py` polls `training_metrics_0.jsonl` in the checkpoint dir and prints progress, per-step rate, ETA (epoch + total), peak memory, peak tokens/sec, and last validation loss. Steps-per-epoch is derived as `ceil(num_samples / effective_batch_size)`; the per-step rate comes from the wall-clock span between the first and last logged step, which excludes load/warmup time.
- **After the fact** — `python check_memory.py <ckpt_output_dir>` prints peak memory, current step, and wall-clock duration, and renders a loss plot via `training_hub.plot_loss`.

## Hardware support & evidence

**Other hardware (upstream claims — not verified here):** none claimed beyond NVIDIA CUDA and AMD ROCm.


- **AMD/ROCm — first-party validated:** this folder's reference training environment was **AMD MI355X with ROCm 7** (`torch==2.11.0`), i.e. OSFT training through training_hub ran on AMD hardware here. On AMD, skip the CUDA-specific `[cuda]`/flash-attn install step (see Install). **Verified on MI355X + ROCm 7.2.4** (`torch==2.11.0+rocm7.2`, training-hub 0.9.7) — see the MI355X platform notes under Install for the two extra steps ROCm needs (`liger-kernel`, `TESTING=true`), and **"8-GPU run (8x MI355X, ROCm 7.2.4)"** for the full-node run: FSDP2 world size 8, exit code 0, with in-band `rocm-smi` + PID checks confirming all 8 GPUs were the job's own and busy.
- **NVIDIA:** the same code runs on 8×H100; upstream training_hub documents the CUDA path explicitly — `pip install training-hub[cuda] --no-build-isolation` for GPU training with flash-attn, with the base PyPI package excluding "the CUDA-related dependencies which are required for GPU training" (github.com/Red-Hat-AI-Innovation-Team/training_hub README, Installation section).
- OSFT in training_hub is backed by the RHAI Innovation Mini-Trainer per upstream's support matrix.

## Notes & troubleshooting

- **OOM:** reduce `--max-tokens-per-gpu` first (large / large-vocabulary models are memory-hungry), then `--effective-batch-size`. `use_liger` (Liger kernels) and `osft_memory_efficient_init` are already enabled in the script's `osft_params`.
- For domain adaptation, `--unfreeze-rank-ratio` between 0.2 and 0.3 is a good starting band.
- The script pins single-node settings (`nnodes=1`, `rdzv_endpoint=127.0.0.1:29500`) in `osft_params`; multi-node would need those edited and is untested here.
