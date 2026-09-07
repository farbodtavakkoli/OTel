# Setup & usage — `train_llm_axolotl.py`

## 1. Overview & when to use

Config-driven post-training with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl):
one YAML file describes the model, the dataset, the algorithm, and the distributed
strategy, and the `axolotl` CLI runs it. Axolotl covers full fine-tuning, LoRA, QLoRA,
QAT, preference tuning (DPO, IPO, KTO, ORPO, SimPO), online RL (GRPO, GDPO) and reward
modelling, over DDP / DeepSpeed ZeRO / FSDP2. Pick it over the hand-written trainers in
this repo (`../deepspeed/`, `../unsloth/`) when you want to sweep recipes by
editing YAML instead of code, and when you want multipack sample packing and the newer
attention/LoRA kernels without wiring them yourself.

Files in this folder:
- `train_llm_axolotl.py` — thin launcher: loads `dev.env`, validates the YAML, shells out to `axolotl train`.
- `config_sft_lora.yaml` — LoRA SFT on chat JSONL (the default recipe).
- `config_sft_qlora.yaml` — QLoRA SFT (4-bit frozen base + LoRA) for models that do not fit in bf16.
- `config_sft_full.yaml` — full-parameter SFT with DeepSpeed ZeRO-3.
- `config_dpo_lora.yaml` — DPO preference tuning with LoRA.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample the SFT configs point at (self-contained smoke data).
- `data/dpo_sample.jsonl` — 10-row **synthetic** preference sample the DPO config points at (see section 4).
- `requirements_axolotl.txt` — dependency list plus the required install order, with a commented ROCm variant.

> **Tested topology:** the 8x H100 path is **UNTESTED** — the configs were written against
> the upstream Axolotl documentation (docs.axolotl.ai config reference, dataset formats,
> RLHF and multi-GPU guides) for the pinned versions below and target a single node with
> **8x H100 80GB**; multi-node is supported upstream (torchrun / Ray) but is not configured
> here. Treat every batch-size and learning-rate number as a starting point, not a result.
> **AMD MI355X (ROCm 7.2): TESTED and working** — single-GPU LoRA SFT smoke verified
> (see section 2a). The H100 configs need three ROCm overrides
> (`sdpa`, `tf32: false`, ROCm torch wheel) documented there.

## 2. Install

### NVIDIA (CUDA)

Axolotl is uv-first upstream and the install is order-sensitive: torch first, then Axolotl
with `--no-build-isolation`. Python 3.12 recommended (>= 3.11 required), PyTorch >= 2.11.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv
export UV_TORCH_BACKEND=cu130                       # match your CUDA toolkit (cu128 also documented)

uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.12.0 torchvision            # step 1: torch BEFORE axolotl
uv pip install --no-build-isolation axolotl[deepspeed]   # step 2: builds against that torch
uv pip install -r requirements_axolotl.txt          # step 3: this folder's launcher deps

axolotl fetch deepspeed_configs   # writes deepspeed_configs/*.json into the cwd
axolotl fetch examples            # optional: upstream reference configs
```

`config_sft_full.yaml` references `deepspeed_configs/zero3_bf16.json`, so run the
`axolotl fetch deepspeed_configs` step from this folder (or fix the path in the YAML).

Docker is the lower-friction alternative and ships flash-attn prebuilt:

```bash
docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest
```

Verify:
```bash
axolotl --help
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

### AMD / ROCm

Upstream lists AMD as a supported GPU class — the requirements line in both the README and
the installation guide reads "NVIDIA GPU (Ampere architecture or newer for `bf16` and Flash
Attention) or AMD GPU". The only AMD-specific walkthrough upstream publishes is the
"AMD GPUs on HPC Systems" guide (<https://docs.axolotl.ai/docs/amd_hpc.html>), which is a
manual, source-based path rather than a one-liner:

```bash
# ROCm torch wheels (the upstream guide pins the rocm5.7 index — check the current tag)
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 --force-reinstall

# Axolotl from source
git clone https://github.com/axolotl-ai-cloud/axolotl && cd axolotl
pip install packaging ninja
pip install --no-build-isolation -e .

pip install -r ../requirements_axolotl.txt   # this folder's launcher deps
```

ROCm caveats documented in that guide, which affect the configs in this folder:
- **flash-attn** comes from the ROCm fork (`github.com/ROCm/flash-attention`), built with
  `GPU_ARCHS` set for your card — or drop `attn_implementation: flash_attention_2` to `sdpa`.
- **DeepSpeed "did not work at the time of testing"** upstream on AMD; use FSDP for sharded
  runs instead — i.e. replace the `deepspeed:` key in `config_sft_full.yaml` with the
  commented `fsdp_version: 2` block.
- xformers is incompatible with ROCm and needs the SwiGLU workaround described in the guide.

See section 8 for the full evidence trail.

### 2a. Platform notes — AMD MI355X (ROCm 7.2) ✅

Verified on 8x AMD Instinct MI355X (gfx950, 288 GB), ROCm 7.2.4, Ubuntu, Python 3.12.3.
This path works with changes — a plain `pip install axolotl` (no source build, no
extras) trains LoRA SFT on ROCm once the torch wheel is corrected and two config keys
are overridden. The upstream AMD HPC guide's source-build path was NOT needed.

```bash
# Set this to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
# Training artifacts need no env var: every config writes to its own relative
# output_dir (./outputs/<recipe>) under training/llm/axolotl.
```

```bash
cd training/llm/axolotl
python3 -m venv .env_axolotl && source .env_axolotl/bin/activate

# step 1: ROCm torch (any recent rocm7.2 wheel; axolotl will re-pin it in step 2)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# step 2: base axolotl — NO extras: [deepspeed]/[flash-attn]/[vllm] pull CUDA-only bits.
pip install packaging ninja
pip install --no-build-isolation axolotl        # installed axolotl 0.18.0

# step 3: CRITICAL — step 2 silently REPLACES torch with the CUDA wheel (axolotl pins
# torch==2.12.1 and pip resolves it from PyPI). Put the ROCm build back:
pip install --force-reinstall --no-deps torch==2.12.1 --index-url https://download.pytorch.org/whl/rocm7.2

# step 4: axolotl pins bitsandbytes==0.49.1, whose wheel has no ROCm-7.2 binary
# (load error: libbitsandbytes_rocm72.so not found; fatal for QLoRA). 0.50.0 ships it:
pip install -U bitsandbytes==0.50.0

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.12.1+rocm7.2  AMD Instinct MI355X
```

Tested versions: torch 2.12.1+rocm7.2, axolotl 0.18.0, transformers 5.14.1, trl 1.8.0,
peft 0.19.1, accelerate 1.13.0, datasets 4.8.4, bitsandbytes 0.50.0, triton-rocm 3.7.1.

Smoke run (single GPU, `config_sft_lora_smoke_mi355x.yaml` in this folder — a copy of
`config_sft_lora.yaml` with Qwen3-0.6B instead of Qwen3-8B, `sdpa`, `tf32: false`,
`sample_packing: false`, seq 512, batch 1, `max_steps: 5`):

```bash
HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 \
python3 train_llm_axolotl.py --config config_sft_lora_smoke_mi355x.yaml --num-processes 1
```

**Expected output** (5/5 steps, finite loss dropping, LoRA kernels auto-patched on ROCm):

```
[axolotl.monkeypatch.lora_kernels] Patched attention class with LoRA optims: Qwen3Attention
{'loss': '1.692', 'grad_norm': '7.348', 'learning_rate': '0.0002', 'ppl': '5.432', ...}
{'loss': '0.08245', 'grad_norm': '1.043', 'learning_rate': '1.91e-05', 'ppl': '1.086', ...}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-mi355x
```

`rocm-smi -d 6` mid-run shows VRAM climbing to ~6 GB on the MI355X; the adapter
(`adapter_model.safetensors`) and checkpoints land in the output dir.

ROCm quirks found (apply these when running the shipped H100 configs on MI355X):

- **torch replacement (step 3 above)** is the one real trap: without the
  force-reinstall you train nothing — the CUDA torch cannot see the AMD GPUs.
- **`attn_implementation: flash_attention_2` → `sdpa`** in every YAML. flash-attn is not
  installed and must NOT be pip-installed (CUDA-only source build; the ROCm fork is a
  separate manual build nobody needs for sdpa).
- **`tf32: true` → `false`** in every YAML — tf32 raises on ROCm.
- **bitsandbytes**: upgrade to >= 0.50.0 (step 4) or QLoRA fails at import.
- **Extras skipped**: `[deepspeed]`, `[flash-attn]`, `[vllm]`. Base axolotl was enough
  for LoRA SFT. The pip resolver also installs CUDA-only `xformers` 0.0.35 and a stack
  of `nvidia-*-cu13` wheels as axolotl deps — they sat inert and caused no harm on the
  sdpa path; don't fight them, just don't enable xformers attention.
- **DeepSpeed / sharding**: not exercised in this smoke (single GPU). The upstream AMD
  guide reports DeepSpeed broken with Axolotl on ROCm — for multi-GPU `config_sft_full`
  runs on MI355X, prefer the commented `fsdp_version: 2` block over the `deepspeed:` key.
- Parallel-job note: when the machine runs several trainings at once, pass a unique
  `--main-process-port` (e.g. 29660) to avoid the 29500 collision.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**Works with changes.** The 1-GPU recipe above scales to all 8 MI355X with
**no new packages and no source build** — `requirements_axolotl.txt` is unchanged. The
changes are all config/launch-side: a new 8-GPU YAML, a replicated dataset (the shipped
10-row sample cannot feed 8 ranks), and an env override for a stale GPU pin in the venv.
Parallelism is **plain torch DDP** (accelerate multi-GPU, RCCL) — **no DeepSpeed, no FSDP**,
so the upstream "DeepSpeed is broken on ROCm" caveat never comes into play for LoRA.

```bash
# runner script, executed under a machine-wide mutex so the job owns all 8 GPUs:
#   flock -w 25200 /tmp/mi355x_gpu8.lock bash run8_axolotl.sh
cd training/llm/axolotl
source .env_axolotl/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # MUST override: see "stale pin" below
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# HF_HOME as exported in the "Set these to suit your machine" block above
python -c "import torch;assert torch.cuda.device_count()==8"   # assert BEFORE training

python3 train_llm_axolotl.py \
    --config config_sft_lora_smoke_mi355x_8gpu.yaml \
    --num-processes 8 --main-process-port 29700
```

The folder's own wrapper works unmodified at 8 processes: it forwards `--num-processes` and
the passthrough `--main-process-port` to the axolotl CLI, which expands to
`accelerate launch --num-processes 8 --main-process-port 29700 -m axolotl.cli.train <cfg>`.

**Parallelism / batch geometry (as axolotl logs it, not as intended):**
`"world_size": 8`, `"tensor_parallel_size": 1`, `"context_parallel_size": 1` →
pure data parallel. `micro_batch_size: 2` x `gradient_accumulation_steps: 1` x 8 ranks =
**global batch 16**; 20 steps. Axolotl does not rewrite the geometry, but it *does* reinterpret
the epoch count — see the dataset note.

**Checking that all 8 GPUs are really in use.** Run a `rocm-smi --showpids` sampler
alongside training (inside the same flock) and cross-check the VRAM holders against
`pgrep -f 'axolotl.cli.train'`, so you know the 8 busy GPUs are your ranks and not another
tenant's job. A healthy sample looks like this:

```
ASSERT device_count = 8    (gpu0..gpu7: AMD Instinct MI355X)
RCCL version : 2.27.7-HEAD:96a25b5    HIP 7.2.53211    ROCm 7.2.1.0-81
PROCESS NAME  GPU(s)  VRAM USED
python3       0..7    ~11-12 GB each   <- 8 axolotl.cli.train ranks, one per MI355X
pt_elastic    -       0                <- accelerate launcher, holds no VRAM
```
(`rocm-smi --showmeminfo vram` in the same sample: 11.6-14.9 GB on each of GPU[0]..GPU[7],
up from the 0.3 GB idle baseline.)

**Expected output** — finite, monotonically decreasing loss, clean exit:

```
{'loss': '1.695', 'grad_norm': '7.532', 'ppl': '5.444', 'epoch': '0.25'}
{'loss': '1.105', 'grad_norm': '5.17',  'ppl': '3.02',  'tokens/train_per_sec_per_gpu': '3049'}
{'loss': '0.0004104', 'grad_norm': '0.009908', 'ppl': '1', 'epoch': '5'}
{'train_runtime': '35.87', 'train_samples_per_second': '8.921', 'train_steps_per_second': '0.558',
 'train_loss': '0.1716', 'memory/max_allocated (GiB)': '3.17'}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-mi355x-8gpu
```

~1.0-2.2 s/step steady-state, ~3100 train tokens/s/GPU, 3.17 GiB peak allocated per rank
(1.2% of the MI355X's 288 GB — this model is far too small to say anything about scaling
efficiency). All 8 ranks should finish, RCCL should tear down with no hang, and the wrapper
should exit with code 0.

**What differs from the 1-GPU run:**

1. **New config `config_sft_lora_smoke_mi355x_8gpu.yaml`** (the working 1-GPU
   `config_sft_lora_smoke_mi355x.yaml` is left untouched). Deltas: dataset path,
   `micro_batch_size` 1 -> 2, `max_steps` 5 -> 20, `num_epochs` 3 -> 1,
   `save_strategy: "no"` + `saves_per_epoch: null`, separate 8-GPU `output_dir` /
   `dataset_prepared_path` (`./outputs/sft-lora-smoke-mi355x-8gpu`,
   `./last_run_prepared_smoke_mi355x_8gpu`) so nothing collides with the 1-GPU run. Same
   `Qwen/Qwen3-0.6B`, same `sdpa`, same `tf32: false`. **No `deepspeed:` / `fsdp:` block** —
   DDP is the right choice for a 0.6B LoRA and sidesteps the ROCm DeepSpeed problem.
2. **Stale GPU pin in the venv (the trap).** If a single-GPU run left
   `export HIP_VISIBLE_DEVICES=6` / `CUDA_VISIBLE_DEVICES=6` at the end of
   `.env_axolotl/bin/activate`, sourcing the venv and launching 8 processes puts all 8 ranks
   on GPU 6 — a silently wrong "8-GPU pass". Always re-export both vars after `source`, and assert
   `torch.cuda.device_count()==8` before training. The configs themselves pin no devices.
3. **The dataset has to be replicated, and axolotl still shrinks it — read this honestly.**
   `./data/OTel_LLM_sample_10.jsonl` (10 rows) cannot feed a global batch of 16. Generate a
   640-row copy (the same 10 rows x64) at `./outputs/data/otel_sample_x64.jsonl` — the path
   `config_sft_lora_smoke_mi355x_8gpu.yaml` already points at — once, from this folder:

   ```bash
   mkdir -p outputs/data
   # awk, NOT cat: the shipped sample has no trailing newline, so a `cat` loop glues the
   # last row of one copy onto the first row of the next -> 576 lines, 63 of them invalid
   # JSON. `awk '{print}'` terminates every record it emits.
   for i in $(seq 64); do awk '{print}' data/OTel_LLM_sample_10.jsonl; done \
       > outputs/data/otel_sample_x64.jsonl
   wc -l outputs/data/otel_sample_x64.jsonl    # -> 640 (10 rows x 64), all valid JSON
   ```

   Axolotl then runs
   `Dropping Invalid Sequences (<None or >512)` over the 640 and keeps **64** — only 1 of the
   10 sample rows fits `sequence_len: 512`; the rest are long IETF mail dumps. So 20 steps at
   batch 16 = **5 epochs over 64 copies of a single conversation**, and the loss collapse to
   4e-4 / ppl 1.0 is **memorisation of one row, not learning**. This is a *pipeline* proof
   (8 ranks, RCCL all-reduce, DDP step loop) and nothing more. For a real multi-GPU run,
   raise `sequence_len` (or set `long_sequences_strategy: truncate`) and use real data.
4. **Port**: `--main-process-port 29700` (use a unique port when several trainings share a machine).
5. QLoRA at 8 GPUs was **not** attempted here; 4-bit + DDP remains untested.

Not exercised: DeepSpeed and FSDP (unnecessary for LoRA at this size), multi-node, and any
model large enough to make sharding meaningful.

### 2b. Platform notes — NVIDIA H100 80GB (CUDA 13.0) ✅

Verified on 1x NVIDIA H100 80GB HBM3 (Hopper, cc 9.0), driver **580.173.02**, **CUDA 13.0**,
Ubuntu, Python 3.12.3. This path works as documented — a plain `pip install axolotl` (base,
no extras) trains LoRA SFT on CUDA once one LoRA-kernel autopatch is disabled for the model in use.
Both **`sdpa`** (zero-build) and **`flash_attention_2`** (source-built flash-attn 2.8.3, ~15 min
nvcc compile) were verified end-to-end. The MI355X ROCm workarounds were all **reversed /
unnecessary** on H100: flash-attn builds & runs here (MI355X used sdpa), `tf32: true` is safe,
and the torch wheel + bitsandbytes bumps aren't needed.

```bash
cd training/llm/axolotl
python3 -m venv .env_axolotl && source .env_axolotl/bin/activate

# step 1: torch FIRST. Where plain PyPI serves native cu130 wheels, NO --index-url
# and NO UV_TORCH_BACKEND are needed. (The README's uv/cu130 flow also works; plain pip is
# simpler here.) The upstream `torch==2.12.0` pin DOES resolve to a cu130 wheel on PyPI now,
# but the base install is left to settle on what axolotl wants:
pip install torch numpy                 # -> torch 2.13.0+cu130 (CUDA 13.0), bf16 matmul OK on H100

# step 2: base axolotl — NO extras. [deepspeed] is not needed for a single-GPU LoRA smoke.
pip install packaging ninja
pip install --no-build-isolation axolotl    # installed axolotl 0.18.0

# step 3: RE-VERIFY torch — axolotl DOES replace it (pins torch==2.12.1), BUT on CUDA hosts
# 2.12.1 also resolves to a cu130 wheel, so NO recovery reinstall is required (contrast MI355X,
# where the pin pulled a CUDA wheel blind to the AMD GPUs). Just confirm cuda == 13.0:
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# -> 2.12.1+cu130  13.0     <- still CUDA 13, still sees the H100. Leave it.
```

Tested versions: **torch 2.12.1+cu130** (after axolotl's re-pin; 2.13.0+cu130 pre-pin, both
fine), axolotl 0.18.0, transformers 5.14.1, trl 1.8.0, peft 0.19.1, accelerate 1.13.0,
datasets 4.8.4, kernels 0.15.2, bitsandbytes 0.49.1 (untouched — QLoRA not exercised),
xformers 0.0.35, driver 580.173.02, CUDA 13.0.

**Model swap (offline node).** On a node where the Hub is 403-blocked, the configs'
`base_model: Qwen/Qwen3-8B` is unreachable. The smoke runs instead against the fully-cached
**`LiquidAI/LFM2.5-350M`** (env: `HF_HOME=$HF_HOME` pointing at a populated local cache, plus
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`). LFM2 ships a chat template (both `chat_template.jinja` and a
`chat_template` key in `tokenizer_config.json`), so `chat_template: tokenizer_default` works
unchanged.

Smoke run (single GPU, `config_sft_lora_smoke_h100_sdpa.yaml` in this folder — a copy of
`config_sft_lora.yaml` with LFM2.5-350M, `sdpa`, `tf32: true`, `sample_packing: false`,
seq 2048, batch 1, `max_steps: 20`, and the LoRA-kernel autopatch disabled — see quirk 1):

```bash
CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_CACHE=/dev/shm/dscache_axolotl AXOLOTL_DO_NOT_TRACK=1 \
python3 train_llm_axolotl.py --config config_sft_lora_smoke_h100_sdpa.yaml \
    --num-processes 1 --main-process-port 29645
```

**Expected output** (20/20 steps, finite loss trending down, clean save):

```
{'loss': '1.325', 'grad_norm': '99.88', 'learning_rate': '0.0002',    'ppl': '3.761', 'epoch': '0.1111'}
{'loss': '0.5884','grad_norm': '15.02', 'learning_rate': '6.91e-05',  'ppl': '1.801', 'epoch': '1.444'}
{'loss': '0.1121','grad_norm': '7.018', 'learning_rate': '1.231e-06', 'ppl': '1.119', 'epoch': '2.222'}
{'train_runtime': '15.86', 'train_samples_per_second': '1.261', 'train_steps_per_second': '1.261', 'train_loss': '0.9872'}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-h100-sdpa
```

The run should finish with exit code 0.

**GPU residency check.** Sample `nvidia-smi` by-PID from inside the run and confirm the
training child is the VRAM holder on the pinned GPU (memory climbing from a few hundred MiB
to a few GiB as it loads and trains) rather than another tenant's job.

Peak reported by axolotl is `memory/max_allocated (GiB): 2.54` — trivial next to the H100's
80 GB (this 350M model says nothing about scaling). The adapter
(`adapter_model.safetensors`, 96 MB) + `checkpoint-20/` + resolved chat template land in
the output dir; `adapter_config.json` records `base_model_name_or_path: LiquidAI/LFM2.5-350M`.

**Step count is real.** 9 of the 10 sample rows survive `sequence_len: 2048` (one long IETF
row dropped; `min_input_len 390 / max 2212`). batch 1 x grad_accum 1 x 1 GPU x 3 epochs would
be 27 steps, capped by `max_steps: 20` → **20 optimizer steps actually run** (not a 0-step
exit).

**H100 quirks / deltas from the MI355X recipe:**

1. **LoRA-kernel autopatch vs LFM2 (the one real trap on H100).** Axolotl auto-enables its
   fused LoRA kernels and monkeypatches the attention QKV forward; with `LiquidAI/LFM2.5-350M`
   (a hybrid conv+attention `lfm2` arch) that patch aborts at load with
   `AssertionError: Original QKV code not found` (`axolotl/monkeypatch/lora_kernels.py`),
   because its regex targets Qwen/Llama-style attention. Fix per axolotl's own warning: set
   `lora_mlp_kernel: false`, `lora_qkv_kernel: false`, `lora_o_kernel: false` in the YAML —
   plain PEFT LoRA then trains fine. **This is model-specific, not H100-specific:** the MI355X
   smoke used Qwen3-0.6B, whose `Qwen3Attention` *does* match the patch (its log even shows
   `Patched attention class with LoRA optims: Qwen3Attention`), so it never hit this. A Qwen
   base on H100 would not need the override.
2. **No torch-clobber recovery needed.** Axolotl replaces torch (2.13.0 → 2.12.1) exactly as
   on MI355X, but here *both* are `+cu130` builds that see the H100, so the MI355X step-3
   force-reinstall is unnecessary. Still re-verify `torch.version.cuda == 13.0` after install.
3. **`tf32: true` (reversed from MI355X's `false`).** tf32 does not raise on CUDA; left on.
   Note torch 2.13's default is `allow_tf32=False` until the framework flips it.
4. **`attn_implementation` (reversed from MI355X's sdpa):** BOTH paths verified. `sdpa` is
   drop-in/zero-build (`config_sft_lora_smoke_h100_sdpa.yaml`); `flash_attention_2`
   (`config_sft_lora_smoke_h100.yaml`) also trains once flash-attn 2.8.3 is source-built and is
   ~2.4x faster on this smoke — see the flash-attn note below. sdpa is the guaranteed fallback.
5. **Extras skipped** (`[deepspeed]`/`[flash-attn]`/`[vllm]`), same as MI355X — base axolotl
   was enough for single-GPU LoRA. `xformers 0.0.35` and the `nvidia-*-cu13` wheels install as
   deps and sit inert on the sdpa path.
6. **bitsandbytes left at the 0.49.1 pin** — the MI355X 0.50.0 bump was only for the missing
   `rocm72` binary; the CUDA wheel is fine. (QLoRA not exercised here.)
7. **Telemetry noise:** on the offline node axolotl's posthog/HF telemetry spams
   `403 Forbidden` proxy warnings — harmless. `AXOLOTL_DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1`
   quiets it.
8. **Datasets cache on a shared/network mount can fail** the `.arrow` write path — point
   `HF_DATASETS_CACHE` at tmpfs (`/dev/shm/...`) if you hit it.

**flash-attn on H100 — BUILT + TRAINED ✅ (a genuine reversal from MI355X's sdpa).**
flash-attn has **no prebuilt wheel** on this PyPI index (`pip install --only-binary=:all:
flash-attn` → no distribution), so it needs a source build via the on-box `nvcc`:

```bash
export CUDA_HOME=/usr/local/cuda            # nvcc 13.0 (release V13.0.88)
MAX_JOBS=16 pip install --no-build-isolation flash-attn==2.8.3
# ~15.5 min nvcc compile with 16 jobs; builds a flash_attn-2.8.3-cp312 wheel, exit code 0.
# torch stays 2.12.1+cu130 afterward (flash-attn does NOT clobber it). Re-verify anyway.
```

Then the shipped **`config_sft_lora_smoke_h100.yaml`** (identical to the sdpa variant but
`attn_implementation: flash_attention_2`) trains cleanly on GPU 5 — axolotl confirms
`attn_uses_flash_lib: true`, and it is markedly faster than sdpa on the same 20-step smoke:

```
[axolotl.monkeypatch.attention.flash_attn_4] Flash Attention 4 is available for your GPU ... pip install flash-attn-4
{'loss': '1.338', 'grad_norm': '99.68', 'epoch': '0.1111'}   # step 1
{'loss': '0.07975','grad_norm': '2.969', 'epoch': '2.222'}   # step 20
{'train_runtime': '6.649', 'train_steps_per_second': '3.008', 'train_loss': '0.9844'}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-h100
```

`train_runtime` 6.6 s (fa2) vs 15.9 s (sdpa) for the identical run — ~2.4x, though at this
toy scale that is dominated by fixed overhead, not a throughput benchmark. Axolotl also flags
that **Flash Attention 4** is available for Hopper (`pip install flash-attn-4`) for even faster
training — not attempted here. Both paths are documented; **sdpa remains the zero-build
fallback** if you can't spare the ~15 min compile (or hit an arch mismatch on a different
base model).

**Multi-GPU on H100 is not covered here** — no multi-GPU pass was run on this
platform. What a multi-GPU LoRA pass would need — mirroring the MI355X 8-GPU section — is
purely config/launch-side: a new N-GPU YAML (bump `micro_batch_size`/`max_steps`, drop
`num_epochs` to 1), a **replicated dataset** (10 rows cannot feed a global batch across N
ranks), `--num-processes N --main-process-port <unique>`, and a pre-flight
`assert torch.cuda.device_count()==N`. Parallelism is plain torch DDP (accelerate + NCCL) —
**no DeepSpeed/FSDP** for a model this small. No new packages expected.

## 3. Environment & secrets

Put a `dev.env` in **this folder** containing your Hub token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_axolotl.py` calls `load_dotenv("dev.env")` before anything else, so the token
lands in the environment the `axolotl` child process inherits. `dev.env` is **git-ignored**
at the repo root — **never commit a token**. There are no secrets hardcoded in the script or
the YAML files; if a token was ever committed anywhere, rotate it on the Hub.

The launcher warns (and continues) when `HF_TOKEN` is unset — that only matters for gated
models such as the Llama or Gemma repos.

## 4. Data

Chat JSONL, one conversation per line, consistent with the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Shipped SFT sample.** The three SFT configs point at `./data/OTel_LLM_sample_10.jsonl`
(10 rows, copied from `../deepspeed_standalone/` so this folder is self-contained).
Each row carries extra columns beyond `messages` — `unmask`, `flow`, `source_id`,
`source_repo`, and two all-null fields (`source_spec_id`, `source_version`).

**Extra-column verification:** per the upstream conversation-format docs
(<https://docs.axolotl.ai/docs/dataset-formats/conversation.html>), the `chat_template`
strategy reads the column named by `field_messages` (default `messages`) and the keys
declared in `message_property_mappings`; columns not declared in the dataset block play no
role in prompt construction. The extra columns in the sample are therefore ignored. This
matches the docs but is **untested here** — the honesty note in section 1 applies.

**Shipped DPO sample.** `./data/dpo_sample.jsonl` is a tiny **synthetic** preference set
generated from the SFT sample: `chosen` is the real assistant answer, `rejected` is a
truncated variant explicitly labeled `[TRUNCATED — synthetic rejected sample for smoke
testing only]`. It exists so `config_dpo_lora.yaml` is runnable end-to-end as a smoke test —
it encodes no real human preference and must be replaced before any meaningful DPO run.
DPO rows are one preference pair per line:

```json
{"messages": [{"role": "user", "content": "..."}],
 "chosen": {"role": "assistant", "content": "..."},
 "rejected": {"role": "assistant", "content": "..."}}
```

`chosen` / `rejected` are single message **objects**, not lists — that is what the
`chat_template.default` DPO strategy expects.

**Swapping in your own data:** point `datasets[].path` at your file (a local path, an HF
repo id, or `s3://` / `gs://`). The launcher refuses to start when a local dataset path
does not exist. The dataset block that does the wiring:

```yaml
datasets:
  - path: ./data/OTel_LLM_sample_10.jsonl   # local file (may also be an HF repo id, s3://, gs://)
    ds_type: json                  # required when `path` is a file
    split: train
    type: chat_template            # the OpenAI-style messages strategy
    field_messages: messages       # column holding the turn list (default: messages)
    message_property_mappings:
      role: role
      content: content
    roles_to_train: ["assistant"]  # loss only on assistant turns
    train_on_eos: turn             # train on the terminator of each trainable turn
chat_template: tokenizer_default   # use the template in tokenizer_config.json
```

Notes and gotchas:
- `chat_template: tokenizer_default` errors if the tokenizer has no template. Use
  `tokenizer_default_fallback_chatml` (or a named template such as `gemma`, `llama3`,
  `qwen3`) in that case.
- Upstream also documents a `data_files:` form for local files
  (`- ds_type: json` + `data_files: [train.jsonl]` + `split: train`); either works.
- If the chat template ends a turn with a token that is not `tokenizer.eos_token`, declare
  it under `eot_tokens:` — otherwise the terminator is never trained and the model will not
  learn to stop. This is the same class of bug documented in `../redhat/`.

Optional but recommended for large datasets: tokenize once up front with

```bash
python3 train_llm_axolotl.py --config config_sft_lora.yaml --task preprocess
```

which fills `dataset_prepared_path` so later runs skip tokenization.

## 5. Run

**Smoke run with the shipped sample** — validates the whole path (config, data, model
download, one tiny training run) without touching your own data:

```bash
python3 train_llm_axolotl.py --config config_sft_lora.yaml --validate-only   # no GPU needed
python3 train_llm_axolotl.py --config config_sft_lora.yaml --num-processes 1 # tiny real run
```

**Full example:**

```bash
nohup python3 train_llm_axolotl.py \
  --config config_sft_lora.yaml \
  --num-processes 8 \
  > train_llm_axolotl.log 2>&1 &

tail -f train_llm_axolotl.log
```

Swap the config for the other recipes: `config_sft_qlora.yaml`, `config_sft_full.yaml`,
`config_dpo_lora.yaml`. Any unrecognized flags are forwarded verbatim to the `axolotl` CLI,
so config fields can be overridden on the command line (e.g. `--learning-rate 1e-5`).

**What "working" looks like:** the launcher first prints its own `INFO` summary block
(config path, base model, objective, method, batch shape, output dir, then the exact
`axolotl train ...` command). Axolotl then prints its config validation, dataset loading and
tokenization progress bars (or a "loading prepared dataset" line on a re-run), a packing
efficiency estimate when `sample_packing: true`, the trainable-parameter count, and finally
a tqdm training bar with `{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}` dicts every
`logging_steps`. Loss should be finite and drifting down, `grad_norm` finite and O(0.1-10).
Checkpoints appear under `output_dir/checkpoint-*` at each save. A `CUDA out of memory`
traceback means lower `micro_batch_size` or `sequence_len`, or move to a higher ZeRO stage.

Merging a LoRA adapter back into the base model afterwards:

```bash
python3 train_llm_axolotl.py --config config_sft_lora.yaml \
  --task merge-lora --lora-model-dir ./outputs/sft-lora
```

The merged weights land in `<output_dir>/merged`.

## 6. Arguments

Every launcher flag:

| Arg | Meaning |
|---|---|
| `--config` | YAML recipe to run (required) |
| `--task` | `train` (default), `preprocess` (tokenize only), `merge-lora` |
| `--num-processes` | Training processes = GPUs on this node (default 8) |
| `--deepspeed` | Override the config's DeepSpeed json for this run |
| `--lora-model-dir` | Adapter directory, required for `--task merge-lora` |
| `--validate-only` | Check the config, print the command, exit without training |

Anything else on the command line is passed through to the `axolotl` CLI unchanged.

Key config fields (full list: <https://docs.axolotl.ai/docs/config-reference.html>):

| Field | Meaning |
|---|---|
| `base_model` | Hub repo id or local path of the model to train |
| `datasets[].path` / `ds_type` / `type` | Where the data is, its file type, and the prompt strategy |
| `chat_template` | Which chat template renders the turns (`tokenizer_default` by default) |
| `roles_to_train`, `train_on_eos` | Which turns and terminators contribute to the loss |
| `sequence_len`, `sample_packing` | Max tokens per row; multipack packing for throughput |
| `adapter` | `lora`, `qlora`, or absent for full fine-tuning |
| `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_linear` | LoRA shape |
| `load_in_4bit` | 4-bit base weights (pair with `adapter: qlora`) |
| `rl` | `dpo`, `ipo`, `kto`, `orpo`, `simpo`, `grpo`, `gdpo`; absent = SFT |
| `rl_beta`, `dpo_loss_type` | Preference-loss knobs |
| `micro_batch_size`, `gradient_accumulation_steps`, `num_epochs` | Batch schedule (effective batch = micro x accum x GPUs) |
| `learning_rate`, `optimizer`, `lr_scheduler`, `warmup_ratio` | Optimization |
| `bf16`, `gradient_checkpointing`, `attn_implementation` | Precision, memory, attention backend |
| `deepspeed` / `fsdp_version` + `fsdp_config` | Multi-GPU sharding strategy (mutually exclusive) |
| `val_set_size`, `evals_per_epoch`, `saves_per_epoch` | Eval split and checkpoint cadence |
| `output_dir`, `dataset_prepared_path` | Where weights and the tokenized cache go |

## 7. Output

Everything lands under the config's `output_dir` — always a path relative to
`training/llm/axolotl` (`./outputs/<recipe>`, e.g. `./outputs/sft-lora`,
`./outputs/sft-lora-smoke-h100`, `./outputs/sft-lora-smoke-mi355x-8gpu`):

- `checkpoint-<step>/` — periodic checkpoints (`saves_per_epoch` / `save_total_limit`).
- Final weights at the top level of `output_dir`: a PEFT adapter
  (`adapter_model.safetensors` + `adapter_config.json`) for LoRA/QLoRA runs, or full
  `model-*.safetensors` shards for full fine-tuning, plus the tokenizer files and the
  resolved chat template.
- `merged/` — appears only after `--task merge-lora`.
- `dataset_prepared_path` (`./last_run_prepared`, or a per-recipe variant such as
  `./last_run_prepared_smoke_h100` in the smoke configs) holds the tokenized arrow cache;
  delete it if you change the data or the template, otherwise the stale cache is reused.
- The console log you redirected to `train_llm_axolotl.log` is the run record; add
  `use_tensorboard: true` or the `wandb_*` fields to the YAML for a metrics backend.

## 8. Hardware support & evidence

What upstream actually documents (checked against the pinned versions below):

- **NVIDIA** — first-class. Requirements in the upstream README and installation guide:
  "NVIDIA GPU (Ampere architecture or newer for `bf16` and Flash Attention) or AMD GPU",
  Python >= 3.11, PyTorch >= 2.11.0. Quick install is uv-based with
  `UV_TORCH_BACKEND=cu130` (or `cu128`); Blackwell GPUs need PyTorch >= 2.11 + CUDA 13.0.
  Sources: <https://github.com/axolotl-ai-cloud/axolotl> (README "Requirements"),
  <https://docs.axolotl.ai/docs/installation.html>.
- **AMD / ROCm** — supported but manual. The requirements line explicitly includes
  "or AMD GPU", and the dedicated guide "AMD GPUs on HPC Systems"
  (<https://docs.axolotl.ai/docs/amd_hpc.html>) documents the working recipe: ROCm torch
  wheels (the guide pins the `rocm5.7` index), source install with `--no-build-isolation`,
  the ROCm flash-attention fork built with `GPU_ARCHS`, an xformers SwiGLU workaround, and
  FSDP for multi-node — with the explicit note that DeepSpeed "did not work at the time of
  testing". There is no ROCm uv backend in the quick-install docs and no prebuilt ROCm
  docker image is advertised.
- **This folder**: the 8x H100 80GB target is **unrun**, but the AMD path is now
  **verified first-hand**: single-GPU LoRA SFT trains on MI355X / ROCm 7.2 with a plain
  pip install of axolotl 0.18.0 — no source build and no ROCm flash-attn fork needed
  (section 2a). Contrary to the AMD HPC guide's rocm5.7-era recipe, the only mandatory
  deviations were the ROCm torch re-pin, `sdpa` instead of `flash_attention_2`,
  `tf32: false`, and bitsandbytes >= 0.50.0. The guide's DeepSpeed-broken-on-ROCm
  finding was not re-tested here (single-GPU smoke); still switch
  `deepspeed: deepspeed_configs/zero3_bf16.json` to the FSDP2 block for sharded AMD runs.
- **Other hardware (upstream claims — not verified here):** none claimed beyond
  NVIDIA/AMD — the upstream README's requirements line lists only "NVIDIA GPU
  (Ampere or newer …) or AMD GPU".

## 9. Notes

- **The YAML is the program.** Axolotl builds the model, tokenizer, dataset pipeline,
  trainer and distributed strategy from one config; the same file is reused for
  `preprocess`, `train`, `inference` and `merge-lora`. Keeping recipes in YAML is why this
  folder ships four configs and one short launcher instead of a large Python trainer.
- **The launcher is deliberately dumb.** It loads `dev.env`, parses the YAML, checks the
  required keys and that local dataset files exist, warns about incoherent adapter /
  quantization combinations, prints a summary, then `subprocess.run`s the CLI and
  propagates its exit code. No training logic lives in Python here — that avoids the
  failure mode where a wrapper silently disagrees with the config.
- **Process launch is Axolotl's job.** `axolotl train cfg.yaml --num-processes 8` spawns the
  distributed workers itself (accelerate under the hood), so there is no `accelerate launch`
  or `torchrun` in the command line.
- **Sharding choices.** LoRA/QLoRA fit per-GPU on H100, so DDP or ZeRO-2 is usually fastest;
  full fine-tuning uses `deepspeed: deepspeed_configs/zero3_bf16.json` to shard parameters,
  gradients and optimizer state. DeepSpeed, FSDP and DDP are mutually exclusive — set one.
  Note the checkpointing detail: upstream uses `use_reentrant: true` with ZeRO-3 and
  `false` elsewhere, which is why the configs differ on that key.
- **Masking.** `roles_to_train: ["assistant"]` plus `train_on_eos: turn` is standard
  completion-only SFT. Set `train_on_inputs: true` to supervise the whole sequence instead.
- **DPO reference model.** With an adapter, TRL uses the adapter-disabled base as the
  implicit reference, so no second copy of the weights is loaded; that is why the DPO config
  uses LoRA. Set `rl_adapter_ref_model: true` to force a real reference model.

**Upstream API uncertainty (do not assume these are verified):**
- `deepspeed_configs/zero2.json` and `zero3_bf16.json` are the filenames upstream documents
  for `axolotl fetch deepspeed_configs`; confirm what actually lands in your working
  directory before uncommenting those lines.
- `transformer_layer_cls_to_wrap: Qwen3DecoderLayer` in the commented FSDP2 blocks must
  match the decoder-layer class name of the model you actually train.
- Sharded QLoRA (4-bit + FSDP) has its own recipe upstream
  (<https://docs.axolotl.ai/docs/fsdp_qlora.html>); the QLoRA config here defaults to DDP
  rather than guessing at that combination.
- GRPO is supported by Axolotl but is not configured here: it needs a separate
  `axolotl vllm-serve` process, GPU partitioning between generation and training, and a
  local `rewards.py`. See <https://docs.axolotl.ai/docs/grpo.html> before attempting it.
