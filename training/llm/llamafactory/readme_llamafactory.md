# Setup & usage — `train_llm_llamafactory.py`

## 1. Overview & when to use

Config-driven post-training with
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): one YAML file per recipe, run by
`llamafactory-cli`. It covers continued pre-training, SFT, reward modelling, PPO, DPO, KTO,
ORPO and SimPO, each of them with full-parameter, freeze (partial), LoRA or QLoRA tuning
(2/3/4/8-bit via bitsandbytes / HQQ / EETQ / GPTQ / AWQ), plus GaLore, APOLLO, BAdam, Muon,
LoRA+, DoRA and PiSSA. Pick it over the other trainers in this repo when you want the widest
menu of algorithms behind one flag (`stage:` + `finetuning_type:`), a large library of
built-in chat templates, and a dataset registry that keeps data wiring out of the code.

Files in this folder:
- `train_llm_llamafactory.py` — thin launcher: loads `dev.env`, validates the YAML and the dataset registry, shells out to `llamafactory-cli`.
- `config_sft_lora.yaml` — LoRA SFT on chat JSONL (the default recipe).
- `config_sft_lora_smoke_mi355x.yaml` — single-GPU 5-step smoke variant, verified on MI355X/ROCm (section 2).
- `config_sft_qlora.yaml` — QLoRA SFT (4-bit bitsandbytes base + LoRA).
- `config_sft_full.yaml` — full-parameter SFT with DeepSpeed ZeRO-3.
- `config_dpo_lora.yaml` — DPO preference tuning with LoRA.
- `data/dataset_info.json` — the dataset registry; LLaMA-Factory will not read a custom file that is not declared here.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample registered as `otel_chat_sft` (self-contained smoke data).
- `data/dpo_sample.jsonl` — 10-row **synthetic** preference sample registered as `otel_chat_dpo` (see section 4).
- `requirements_llamafactory.txt` — dependency list plus install notes, with commented ROCm / NPU variants.

> **Tested topology:** the LoRA SFT path is **verified on 1x AMD Instinct MI355X
> (ROCm 7.2)** — see the MI355X platform notes below. The configs as shipped still target a
> single node with **8x H100 80GB**, and everything beyond single-GPU LoRA SFT (QLoRA,
> ZeRO-3 full FT, DPO, multi-GPU/multi-node) remains **unrun**. They were written against
> the upstream README, `data/README.md`, `examples/` YAMLs and the
> `src/llamafactory/hparams/*` dataclasses for the pinned versions below. Multi-node is supported
> upstream (`FORCE_TORCHRUN=1 NNODES=... NODE_RANK=... MASTER_ADDR=...`) but is not
> configured here. Treat every batch-size and learning-rate number as a starting point,
> not a result.

## 2. Install

### NVIDIA (CUDA)

Install from a **source checkout** — `config_sft_full.yaml` references
`examples/deepspeed/ds_z3_config.json`, which only exists in the git tree. (A `llamafactory`
PyPI wheel exists, but the examples/, data/ and DeepSpeed JSONs do not ship with it.)

```bash
python3.12 -m venv ~/.venv && source ~/.venv/bin/activate

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt     # optional: BLEU / ROUGE eval
pip install -r requirements/deepspeed.txt   # needed for config_sft_full.yaml

cd -                                        # back to this folder
pip install -r requirements_llamafactory.txt
```

Docker is the lower-friction alternative (ships flash-attn prebuilt):

```bash
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

Verify:
```bash
llamafactory-cli help
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

Because `deepspeed: examples/deepspeed/ds_z3_config.json` is resolved relative to the
**current working directory**, run the full-FT recipe from inside the LLaMA-Factory
checkout (pass this folder's config by absolute path), or change that key to an absolute
path. Everything else runs fine from this folder.

### AMD / ROCm

Upstream documents ROCm as a supported path, primarily through Docker: the README ships a
dedicated `docker/docker-rocm/` compose file, and AMD publishes an official LLaMA-Factory
fine-tuning guide on `rocm.docs.amd.com`. There is no ROCm pip extra — the non-Docker route
is ROCm PyTorch wheels plus the same source install as above.

```bash
# Option A (upstream-documented): ROCm docker compose
cd LLaMA-Factory/docker/docker-rocm/
docker compose up -d
docker compose exec llamafactory bash

# Option B: ROCm wheels + source install
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocmX.Y  # current ROCm tag
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .
```

ROCm caveats for the configs in this folder: `flash_attn: fa2` assumes a working
flash-attention build for your stack — set `flash_attn: sdpa` (or `auto`) if it is not
available, and treat `quantization_method: bnb` (bitsandbytes) as CUDA-first. Upstream also
documents **Ascend NPU** support (Python 3.12 + `requirements/npu.txt` + CANN toolkit, with
prebuilt docker images); that path is out of scope here. See section 8 for sources.

#### ✅ Platform notes — AMD MI355X (ROCm 7.2)

**This path works with changes** (pip/venv route, Option B above). Two config deltas from the
shipped YAML: `flash_attn: sdpa` instead of `fa2` (no flash-attn wheel on ROCm — never
`pip install flash-attn` there), and single-GPU sizing. The upstream ROCm docker compose
route (Option A) was **not** exercised — documented-but-untried here.

Host: 8x MI355X (gfx950, 288 GB), ROCm 7.2.4, Python 3.12.3.

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Validated install:

```bash
cd training/llm/llamafactory
python3 -m venv .env_llamafactory && source .env_llamafactory/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .env_llamafactory/LLaMA-Factory
pip install -e .env_llamafactory/LLaMA-Factory   # keeps ROCm torch; verify below
pip install python-dotenv                                   # launcher dep, not pulled by upstream
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.13.0+rocm7.2 AMD Instinct MI355X
```

Resolved versions: torch **2.13.0+rocm7.2**, llamafactory **0.9.6.dev0** (source,
`ff6d4d1`), transformers **5.8.0**, peft 0.18.1, trl 0.24.0, accelerate 1.11.0,
datasets 4.0.0. Upstream now pins `transformers>=4.55.0,<=5.8.0`, so no version bump is
needed for transformers-5.x. Install torch/torchvision/torchaudio from the ROCm index
**before** `pip install -e .` (upstream depends on all three) or pip will pull CUDA wheels
over your ROCm torch. The checkout lives inside the git-ignored venv dir to keep
`git status` clean; for `config_sft_full.yaml` remember `deepspeed:` resolves relative to
the CWD, i.e. run from `.env_llamafactory/LLaMA-Factory/`.

Smoke run (single GPU, Qwen3-0.6B stand-in — same `qwen3_nothink` template as the shipped
Qwen3-8B recipe):

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7      # pick your GPU
python3 train_llm_llamafactory.py --config config_sft_lora_smoke_mi355x.yaml --num-gpus 1
```

**Expected output** (5-step smoke, plus a 200-step run at cutoff_len 2048 / batch 4 for GPU load):

```
[INFO] llamafactory.model.model_utils.attention >> Using torch SDPA for faster training and inference.
[INFO] llamafactory.model.loader >> trainable params: 40,370,176 || all params: 636,420,096 || trainable%: 6.3433
{'loss': '1.747', 'grad_norm': '5.769', 'learning_rate': '0', 'epoch': '0.1'}
{'train_runtime': '66.57', 'train_samples_per_second': '12.02', 'train_steps_per_second': '3.004', 'train_loss': '0.08945'}
```

`rocm-smi` mid-run shows the selected GPU at 2404 MHz / 330 W / 36% busy; adapter +
checkpoints land in `output_dir` as described in section 7. Quirks hit:

- **CLI `key=value` overrides are YAML-parsed**: `eval_strategy=no` becomes boolean
  `False` and crashes transformers' `IntervalStrategy` — hence the smoke YAML copy, where
  `eval_strategy: "no"` is quoted.
- transformers 5.8.0 warns `warmup_ratio is deprecated ... use warmup_steps` (harmless).
- Not retested here but known on ROCm: TF32 toggles raise (none of these
  configs set `tf32:` — keep it that way), and `config_sft_qlora.yaml`'s bitsandbytes
  path should work (bnb 0.50.0 stock wheel loads its ROCm binary on gfx950) but was not
  run.

#### ✅ Platform notes — NVIDIA H100 80GB (CUDA 13.0)

**This path works with changes** (pip/venv route, source `pip install -e .`). Single-GPU LoRA
SFT trains with finite, decreasing loss and saves an adapter. Two deltas from the shipped
`config_sft_lora.yaml`: `flash_attn: sdpa` instead of `fa2` (the *prebuilt* flash-attn wheel
ABI-mismatches this very new torch — see below), and single-GPU sizing. Everything else
(model stand-in, template, LoRA rank/alpha/target, lr, bf16) is identical to the MI355X smoke.

Host: 1x NVIDIA H100 80GB HBM3 (Hopper cc 9.0, native FP8), driver **580.173.02**, CUDA
**13.0**, Python 3.12.3. Validated install (tmpfs venv):

```bash
cd training/llm/llamafactory
python3 -m venv .env_llamafactory && source .env_llamafactory/bin/activate
pip install torch numpy            # -> torch 2.13.0+cu130 (native CUDA 13, NO --index-url)
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .env_llamafactory/LLaMA-Factory
pip install -e .env_llamafactory/LLaMA-Factory   # pulls transformers 5.8.0 etc; verify torch below
pip install python-dotenv                         # launcher dep, not pulled by upstream
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130 13.0
```

**Torch-clobber check:** unlike the ROCm route (where `pip install -e .` pulls CUDA wheels
over ROCm torch), on CUDA the editable install adds matching `torchvision 0.28.0+cu130` /
`torchaudio 2.11.0+cu130` and leaves `torch 2.13.0+cu130` intact — no recovery needed. Still
re-verify after install; the SDK's other frameworks do clobber. No `--index-url` is needed
for the base case (the default PyPI torch is already cu130). `kernels` was **not** pulled
(no `>=0.12,<0.13` trap here).

Resolved versions: torch **2.13.0+cu130**, llamafactory **0.9.6.dev0** (source, `c4e09c7`),
transformers **5.8.0**, peft 0.18.1, trl 0.24.0, accelerate 1.11.0, datasets 4.0.0 — the
**same** stack the MI355X run resolves (upstream pins `transformers<=5.8.0`), so no version
bump is needed and the transformers-5.x `apply_chat_template` trap does not bite (upstream
handles masking internally).

**flash-attn on H100 (the one real deviation):** a prebuilt wheel *installs*
(`pip install flash-attn --no-build-isolation` → `flash-attn 2.8.3.post1`) but **fails to
import** against torch 2.13.0+cu130:
`ImportError: flash_attn_2_cuda...so: undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`
— the wheel was built against an older libc10 ABI. A from-source rebuild might fix it; the
validated run instead uses `flash_attn: sdpa` (the same fallback as ROCm, for a
different reason). SDPA trains fine. Uninstall `flash-attn` to keep the env clean; it is
left commented in `requirements_llamafactory.txt` with this ABI note. `flash_attn: fa2` should
work if you build flash-attn from source against 2.13/cu130, or use the upstream docker image.

**TF32 note:** these configs set no `tf32:` key (correct — leave it unset). On CUDA the guard
does not raise (unlike ROCm); torch's default `matmul.allow_tf32` is `False` and training uses
`bf16: true` regardless, so TF32 is moot for this bf16 LoRA run.

Smoke run (single GPU 7, port 29663, Qwen3-0.6B stand-in — same `qwen3_nothink` template as
the shipped Qwen3-8B recipe). A dedicated `config_sft_lora_smoke_h100.yaml` is provided
(the 1-GPU MI355X config is left untouched); it bumps to `num_train_epochs: 3.0` over the
~10-row sample = **30 optimizer steps** (vs the MI355X copy's `max_steps: 5`) so the loss
trend is visible, and writes to `output_dir: /dev/shm/h100/out/llamafactory/sft_lora_smoke`:

```bash
export CUDA_VISIBLE_DEVICES=7        # plain CUDA_VISIBLE_DEVICES — no HIP_VISIBLE_DEVICES on NVIDIA
export MASTER_PORT=29663
# HF_HOME as exported in the "Set these to suit your machine" block above
export HF_DATASETS_CACHE=/dev/shm/h100/dscache_llamafactory
python3 train_llm_llamafactory.py --config config_sft_lora_smoke_h100.yaml --num-gpus 1
```

**Expected output** (30-step run):

```
[INFO|llamafactory.model.model_utils.attention:144] Using torch SDPA for faster training and inference.
[INFO|llamafactory.model.loader:144] trainable params: 40,370,176 || all params: 636,420,096 || trainable%: 6.3433
[INFO|trainer.py:1470]   Num examples = 10
[INFO|trainer.py:1478]   Total optimization steps = 30
{'loss': '4.168', 'grad_norm': '24.51', 'learning_rate': '9.738e-05', 'epoch': '0.5'}   # early
{'loss': '1.078', 'grad_norm': '6.427', 'learning_rate': '5.809e-05', 'epoch': '1.5'}   # mid
{'loss': '0.7526','grad_norm': '4.786', 'learning_rate': '4.621e-06', 'epoch': '2.7'}   # late
{'train_runtime': '11.09', 'train_samples_per_second': '2.706', 'train_steps_per_second': '2.706', 'train_loss': '1.7511', 'epoch': '3'}
```

The run should finish with exit code 0. Loss is finite and falls from the ~2.5–4 band
(epochs 0–1, LR still warming) to ~0.3–0.8 by
epochs 2–3; `train_loss = 1.7511` mean over the noisy 30-step curve (batch 1×1×1, one 0.6B
model seeing ~10 conversations — a plumbing proof, not a learning result). Artifacts land in
`output_dir`: `adapter_model.safetensors` (161 MB), `adapter_config.json`
(`base_model_name_or_path: Qwen/Qwen3-0.6B`), `checkpoint-30/`, `trainer_log.jsonl`,
tokenizer files.

**GPU residency check.** On a shared node, run a background `nvidia-smi` sampler from
*inside* the launching script, filtered to the GPU you pinned, so the sample is provably
concurrent with your own job. With `--query-compute-apps=pid,process_name,used_memory` a
healthy sample shows a single compute PID — your venv's python — on that GPU:

```
GPU7 (idx, mem.used, util, power):  7, 1787 MiB, 0 %, 120.57 W
-- compute-apps on GPU7 --
<pid>, <venv>/bin/python, 1798 MiB      <- the training process, only PID
```

VRAM on the pinned GPU climbs 518 → 1656 → 3742 MiB across the run; power 73 W idle →
120–125 W under load, with no foreign PID. As on the
MI355X short runs, `util.gpu` reads `0 %` in most polls because a ~0.37 s/step job idles
between 3 s samples — VRAM occupancy + power + the per-step loss log are the load-bearing
signals, not the instantaneous util poll. Peak VRAM ~3.7 GB of 80 GB, so **no VRAM pressure
at all** on H100 for this 0.6B LoRA (the 80 GB vs 288 GB gap is irrelevant at this size; a
full Qwen3-8B LoRA at `cutoff_len: 4096` batch 4 would need the shipped 8-GPU sizing revisited
for 80 GB, but that is untested here).

**Multi-GPU on H100 is not covered here** — no H100 multi-GPU run was made.
What it would need mirrors the MI355X 8-GPU section below:
`llamafactory-cli` auto-re-execs under `torchrun` for `device_count > 1` (plain DDP for LoRA,
no DeepSpeed), a dataset large enough to feed N ranks (the x256 replication trick), a distinct
`MASTER_PORT`, and — critically — overriding any stale `CUDA_VISIBLE_DEVICES` left in the
venv/shell after a 1-GPU run.

**H100 summary:** LoRA SFT **works with changes** — the only functional change vs the shipped
CUDA config is `flash_attn: fa2 → sdpa` (prebuilt flash-attn wheel ABI-incompatible with torch
2.13/cu130; sdpa trains cleanly), plus single-GPU smoke sizing. torch 2.13.0+cu130 installs
with a plain `pip install torch` (no index URL) and survives the editable LLaMA-Factory
install. Same resolved framework stack as MI355X.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This path works with changes.** LoRA SFT scales from 1 to 8 MI355X unmodified in *code* —
`llamafactory-cli` auto-detects `get_device_count() > 1` and re-execs itself under
`torchrun`, so plain **DDP (world size 8, no DeepSpeed, no FSDP)** is what you get for free.
The changes are all environment/data plumbing, not framework fixes: override the venv's
stale GPU pin, and give the run a dataset big enough to feed 8 ranks.

Launch (run under a machine-wide GPU mutex so the job owns all 8 GPUs):

```bash
cd training/llm/llamafactory
source .env_llamafactory/bin/activate
# CRITICAL: if bin/activate ends with `export HIP_VISIBLE_DEVICES=7` / `CUDA_VISIBLE_DEVICES=7`
# left over from a 1-GPU smoke, sourcing it silently pins you to ONE GPU. Override after:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29710 NPROC_PER_NODE=8   # HF_HOME as exported above
python3 -c "import torch; assert torch.cuda.device_count()==8"   # assert before training
python3 train_llm_llamafactory.py \
  --config config_sft_lora_smoke_mi355x_8gpu.yaml --num-gpus 8
```

There is **no `--nproc_per_node` to pass** — upstream `src/llamafactory/launcher.py` builds the
torchrun command itself, reading `NPROC_PER_NODE` (default = device count) and `MASTER_PORT`
(default = a random free port) from the environment. The folder's own wrapper
`train_llm_llamafactory.py` needs no change: it only forces `FORCE_TORCHRUN=1` for
multi-GPU *full* fine-tuning, and LoRA does not need it.

Parallelism and batch geometry as actually logged (not as intended):

```
[INFO|llamafactory.launcher:144] Initializing 8 distributed tasks at: 127.0.0.1:29710
[INFO|llamafactory.hparams.parser:651] Process rank: 0, world size: 8, device: cuda:0,
    distributed training: True, compute dtype: torch.bfloat16      # ... ranks 1-7 identical
[INFO|trainer.py:1470]   Num examples = 2,560
[INFO|trainer.py:1473]   Instantaneous batch size per device = 4
[INFO|trainer.py:1476]   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:1477]   Gradient Accumulation steps = 2
[INFO|trainer.py:1478]   Total optimization steps = 20
```

So the real global batch is **4 x 2 x 8 = 64** (the 1-GPU smoke is 1 x 1 x 1 = 1). Loss should
be finite and fall monotonically, and the run should exit with code 0 and all 8 ranks torn
down cleanly (no NCCL/RCCL hang, no `ddp_timeout` trip):

```
{'loss': '2.301', 'grad_norm': '3.848', 'learning_rate': '0',        'epoch': '0.025'}
{'loss': '1.276', 'grad_norm': '2.457', 'learning_rate': '9.397e-05', 'epoch': '0.125'}
{'loss': '0.2919','grad_norm': '1.369', 'learning_rate': '6.227e-05', 'epoch': '0.25'}
{'loss': '0.0283','grad_norm': '0.5202','learning_rate': '6.819e-07', 'epoch': '0.5'}
***** train metrics *****  epoch = 0.5   total_flos = 1760880GF   train_loss = 0.6782
  train_runtime = 0:00:13.91   train_samples_per_second = 91.962   train_steps_per_second = 1.437
```

**Checking all 8 GPUs are really yours.** Run the `rocm-smi --showpids` sampler as a
background loop in the same script that holds the GPU lock, so the sample is provably
concurrent with your own training — an externally-timed `rocm-smi` on a shared machine can
easily catch a different tenant's job. A healthy mid-training sample:

```
PROCESS NAME     GPU(s)  VRAM USED
python3          0..7    ~12.6-13.4 GB each   <- the 8 DDP ranks, one per GPU
pt_elastic       -       0                    <- torchrun agent
llamafactory-cl  -       0                    <- the CLI process
```

That is **8 python3 processes, one GPU each, ~12.6-13.4 GB VRAM apiece** (~4% of each 288 GB
card — Qwen3-0.6B + LoRA is tiny), with `GPU Memory Allocated (VRAM%) = 4` on GPU[0]..GPU[7]
and no other GPU in use. Cross-check the PIDs with `pgrep` in the **same** sample: the
launcher script -> `train_llm_llamafactory.py` -> `llamafactory-cli` -> `pt_elastic` -> the
contiguous 8-worker block. Every VRAM-holding PID in the KFD table should be a descendant of
your own launcher, with no foreign PID.

A **second, longer run** gives utilisation (not just occupancy) evidence —
same config with CLI overrides `max_steps=60 per_device_train_batch_size=8 cutoff_len=1024`,
i.e. global batch **8 x 2 x 8 = 128**, 3 epochs over the 2,560 rows, `train_runtime = 0:00:40.24`,
`train_samples_per_second = 190.821`, `train_steps_per_second = 1.491`, `total_flos = 21123853GF`,
loss 2.119 -> 0.000297, exit code 0. Its in-band samples (every 5 s) show all eight cards loaded
together:

```
busy% per sample, GPU[0..7]   98 98 98 98 98 98 98 98
                              96 96 95 95 94 95 94 93
                              95 96 95 95 96 95 96 96
                              98 98 98 98 98 98 99 98
power (W)         GPU[0..7]:  951 918 941 951 889 892 906 928
VRAM%             GPU[0..7]:  9 9 9 9 9 9 9 9     (~29.5-29.9 GB each, 8 python3 ranks
                                                   under one llamafactory-cli / pt_elastic)
```

Note `rocm-smi` busy% is an instantaneous poll: the 20-step run's samples read
`0 100 100 100 100 100 100 100` (init) then `6 6 7 6 6 5 6 6`, because a 0.7 s/step job idles
between polls. For a short run **VRAM occupancy + the per-rank logs are the load-bearing
signal**; do not read a 0% sample as "GPUs unused" — a longer run like the 60-step one above
is what shows sustained 8-way load.

**What differs from the 1-GPU run** (everything else — model, template, LoRA rank/alpha/target,
`flash_attn: sdpa`, lr, bf16 — is identical):

1. **New config file `config_sft_lora_smoke_mi355x_8gpu.yaml`.** The working 1-GPU
   `config_sft_lora_smoke_mi355x.yaml` is left untouched. Deltas: `max_steps: 20`,
   `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 2`,
   `preprocessing_num_workers: 8`, `dataloader_num_workers: 2`, `save_strategy: "no"`
   (replacing `save_steps`/`save_total_limit`), and the x256 dataset below. No `deepspeed:`
   key — DDP is sufficient for a 0.6B LoRA and avoids the ZeRO config surface entirely.
2. **A bigger data slice, because 10 rows cannot feed 8 ranks.** 8 ranks x batch 4 x accum 2
   = 64 samples per optimizer step; the shipped `OTel_LLM_sample_10.jsonl` would be exhausted
   before rank 7 sees a batch. Replicate the 10 rows x256 -> 2,560 rows written
   **outside the repo**, e.g. at
   `$OUTPUT_DIR/train_llm_llamafactory/gpu8/data/otel_chat_sft_x256.jsonl`, and
   register it as `otel_chat_sft_x256` in `data/dataset_info.json` with an **absolute**
   `file_name` (upstream `data/loader.py` does `os.path.join(dataset_dir, file_name)`, so an
   absolute path wins and `dataset_dir: ./data` still resolves the registry). This is a
   **pipeline proof, not a learning result** — the loss curve above is a 0.6B model memorising
   10 repeated conversations, which is exactly why it reaches 0.028.
3. **The venv GPU pin must be overridden** (see the launch block). This is the single
   highest-risk trap in this folder: source the venv, forget the override, and you get a
   *successful-looking* run that silently used one GPU only.
4. `save_strategy: "no"` still leaves a final `adapter_model.safetensors` — HF `Trainer` saves
   the model at the end of `do_train` regardless. Delete it after a smoke run; 165 MB per run
   here. (No config in this folder sets `load_best_model_at_end`, which would force a save.)

Not attempted, deliberately: **QLoRA at 8 GPUs** — `config_sft_qlora.yaml` + ZeRO-3 is already
known broken (a quantized base cannot be ZeRO-3 sharded; see section 7), and upstream ships a
separate FSDP+QLoRA recipe for that. No new package or pin was needed for the 8-GPU path, so
`requirements_llamafactory.txt` is unchanged.

## 3. Environment & secrets

Put a `dev.env` in **this folder** containing your Hub token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_llamafactory.py` calls `load_dotenv("dev.env")` before anything else, so the
token lands in the environment the `llamafactory-cli` child process inherits. `dev.env` is
**git-ignored** at the repo root — **never commit tokens**. No secret is hardcoded in the
script or the YAML files; if a token was ever committed anywhere, rotate it on the Hub.

The launcher warns (and continues) when `HF_TOKEN` is unset — that only matters for gated
models. Upstream also accepts `hf_hub_token:` in the YAML; do **not** use it, that would put
the secret in a tracked file.

## 4. Data

Chat JSONL, one conversation per line, consistent with the rest of this repo:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

LLaMA-Factory calls this the **OpenAI format**, which upstream treats as a special case of
`sharegpt` formatting.

**Registry mechanics.** A custom file is invisible until it is registered in
`dataset_info.json` inside `dataset_dir` — that file is shipped here as
`data/dataset_info.json`:

```json
"otel_chat_sft": {
  "file_name": "OTel_LLM_sample_10.jsonl",
  "formatting": "sharegpt",
  "columns": { "messages": "messages" },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```

Then the config says `dataset: otel_chat_sft` and `dataset_dir: ./data`, and `file_name` is
resolved relative to `dataset_dir` — so the shipped registry points at
`data/OTel_LLM_sample_10.jsonl` and this folder is fully self-contained.

**Shipped SFT sample.** `data/OTel_LLM_sample_10.jsonl` (10 rows, copied from
`training/llm/deepspeed_standalone`). Each row carries extra columns beyond `messages` —
`unmask`, `flow`, `source_id`, `source_repo`, and two all-null fields (`source_spec_id`,
`source_version`).

**Extra-column verification:** per upstream `data/README.md`, dataset parsing is driven by
the registry's `columns` mapping (here only `messages` for SFT; `messages` + `chosen` +
`rejected` for DPO) and the `tags` inside each message; columns not named in that mapping
are not part of the prompt/response construction. The extra columns in the sample are
therefore ignored. This matches the docs but is **untested here** — the honesty note in
section 1 applies.

**Shipped DPO sample.** `data/dpo_sample.jsonl` is a tiny **synthetic** preference set
generated from the SFT sample: `chosen` is the real assistant answer, `rejected` is a
truncated variant explicitly labeled `[TRUNCATED — synthetic rejected sample for smoke
testing only]`. It exists so `config_dpo_lora.yaml` is runnable end-to-end as a smoke test —
it encodes no real human preference and must be replaced before any meaningful DPO run.
It is registered as `otel_chat_dpo` with `"ranking": true`; one row per preference pair:

```json
{"messages": [{"role": "user", "content": "..."}],
 "chosen": {"role": "assistant", "content": "..."},
 "rejected": {"role": "assistant", "content": "..."}}
```

`chosen` / `rejected` are single message **objects**, not lists. The launcher refuses to
start a `stage: dpo` run against a dataset that is not marked `ranking: true`.

**Swapping in your own data:** drop your JSONL into `data/`, change (or add) the
`file_name` in `data/dataset_info.json`, and reference the registered name from the
config's `dataset:` key. Notes:

- `file_name` is resolved relative to `dataset_dir`; allowed types are json, jsonl, csv,
  parquet, arrow. Use `hf_hub_url` instead of `file_name` for a Hub dataset.
- The optional `system` message may be the first element of `messages` — that is exactly
  what `system_tag: system` enables. `user` and `assistant` turns must alternate.
- `template:` must match the model family (`qwen3_nothink`, `llama3`, `gemma3`, `deepseek3`,
  ...); the README's model table lists the template per model. Use the **same** template at
  inference time or output quality falls apart.

For very large datasets, tokenize once with `tokenized_path: <dir>` in the config: the first
run writes the tokenized dataset there and later runs load it instead of re-tokenizing.

## 5. Run

**Smoke run with the shipped sample** — validates the whole path (config, registry, data,
model download, one tiny training run) without touching your own data:

```bash
python3 train_llm_llamafactory.py --config config_sft_lora.yaml --validate-only  # no GPU needed
python3 train_llm_llamafactory.py --config config_sft_lora.yaml --gpus 0         # tiny real run
```

**Full example:**

```bash
nohup python3 train_llm_llamafactory.py \
  --config config_sft_lora.yaml \
  --num-gpus 8 \
  > train_llm_llamafactory.log 2>&1 &

tail -f train_llm_llamafactory.log
```

Swap the config for the other recipes: `config_sft_qlora.yaml`, `config_sft_full.yaml`,
`config_dpo_lora.yaml`. Unrecognized arguments are forwarded verbatim, and upstream accepts
`key=value` overrides, so `... --config config_sft_lora.yaml learning_rate=1e-5
logging_steps=1` works.

**What "working" looks like:** the launcher prints its own `INFO` summary first (config,
model, stage/method, template, resolved dataset names, batch shape, output dir, then the
exact `llamafactory-cli ...` command). LLaMA-Factory then logs the tokenizer/model load, a
dataset preview — it prints one formatted example with its `input_ids`, decoded `inputs`,
`label_ids` and `labels`, and the labels should show `-100`-masked prompt tokens with only
the assistant turn supervised — then the trainable-parameter count (`trainable params: ...
|| all params: ... || trainable%: ...`), and finally a tqdm bar with
`{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` every `logging_steps`.
Loss should be finite and drifting down. For DPO you additionally get `rewards/chosen`,
`rewards/rejected` and `rewards/accuracies` — accuracy climbing above 0.5 is the signal that
preference training is doing something. `CUDA out of memory` means lower
`per_device_train_batch_size` or `cutoff_len`, or move to ZeRO-3.

Merging a LoRA adapter into the base model afterwards needs a small export config
(`model_name_or_path`, `adapter_name_or_path`, `template`, `finetuning_type: lora`,
`export_dir`), then:

```bash
python3 train_llm_llamafactory.py --config config_export.yaml --task export
```

Do **not** merge against a quantized base or with `quantization_bit` set.

## 6. Arguments

Every launcher flag:

| Arg | Meaning |
|---|---|
| `--config` | YAML recipe to run (required) |
| `--task` | `train` (default), `export` (merge LoRA into the base), `chat`, `api` |
| `--gpus` | Sets `CUDA_VISIBLE_DEVICES`, e.g. `0,1,2,3`. Default: all visible GPUs |
| `--num-gpus` | GPU count for the summary and the torchrun decision (default 8) |
| `--force-torchrun` | Always set `FORCE_TORCHRUN=1` (automatic for multi-GPU full FT) |
| `--validate-only` | Check config + dataset registry, print the command, exit |

Anything else on the command line is passed through to `llamafactory-cli` unchanged
(upstream `key=value` overrides).

Key config fields (source of truth: `src/llamafactory/hparams/` upstream):

| Field | Meaning |
|---|---|
| `model_name_or_path` | Hub repo id or local path of the model to train |
| `stage` | `pt`, `sft`, `rm`, `ppo`, `dpo`, `kto` |
| `finetuning_type` | `lora`, `oft`, `freeze`, `full` |
| `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target` | LoRA shape (`lora_target: all` = all linear modules; `lora_alpha` defaults to `lora_rank * 2`) |
| `quantization_bit`, `quantization_method`, `quantization_type` | QLoRA: 4/8-bit, `bnb`/`hqq`/`eetq`, `nf4`/`fp4` |
| `pref_beta`, `pref_loss`, `pref_ftx` | DPO family: beta, loss variant (`sigmoid`/`ipo`/`orpo`/`simpo`/...), optional SFT-loss mix |
| `dataset`, `dataset_dir`, `eval_dataset` | Registered dataset name(s), the folder holding `dataset_info.json` |
| `template` | Chat template — must match the model |
| `cutoff_len`, `packing`, `neat_packing` | Max tokens per sample; sequence packing (`neat_packing` also blocks cross-attention) |
| `train_on_prompt`, `mask_history` | Unmask the prompt / train only on the last turn |
| `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs` | Batch schedule (effective batch = per-device x accum x GPUs) |
| `learning_rate`, `lr_scheduler_type`, `warmup_ratio` | Optimization |
| `bf16`, `pure_bf16`, `flash_attn`, `enable_liger_kernel` | Precision and kernels (`flash_attn`: `auto`/`disabled`/`sdpa`/`fa2`/`fa3`) |
| `deepspeed` | Path to a ZeRO json (`examples/deepspeed/ds_z{0,2,3}_config.json`) |
| `val_size`, `eval_strategy`, `eval_steps` | Held-out split and eval cadence (mutually exclusive with `eval_dataset`) |
| `output_dir`, `save_steps`, `save_total_limit`, `plot_loss` | Checkpointing and the loss plot |
| `report_to` | `none`, `wandb`, `tensorboard`, `swanlab`, `mlflow` |

## 7. Output

Everything lands under the config's `output_dir` (`saves/<model>/<method>/<stage>`):

- `checkpoint-<step>/` — periodic checkpoints (`save_steps` / `save_total_limit`).
- Final artifacts at the top level: a PEFT adapter (`adapter_model.safetensors` +
  `adapter_config.json`) for LoRA/QLoRA, or full `model-*.safetensors` shards for full
  fine-tuning, plus tokenizer files.
- `trainer_log.jsonl` — per-logging-step metrics (loss, learning rate, epoch, throughput).
- `training_args.yaml` / `all_results.json` / `train_results.json` — the resolved
  configuration and final metrics for the run.
- `training_loss.png` — written when `plot_loss: true`.
- `running_log.txt` — the framework's own log, alongside the console log you redirected to
  `train_llm_llamafactory.log`.
- Merged weights land in the `export_dir` of your export config, not in `output_dir`.

## 8. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** Ascend NPU is officially documented upstream, in addition to NVIDIA/AMD.


What upstream actually documents (upstream README at
<https://github.com/hiyouga/LLaMA-Factory>):

- **NVIDIA** — first-class. Requirement table: Python >= 3.11, torch minimum 2.0.0 /
  recommended 2.6.0, CUDA 11.6 minimum / 12.2 recommended; the official docker image is
  built on CUDA 12.4 + PyTorch 2.6.0 + flash-attn 2.7.4.
- **AMD / ROCm** — documented upstream **and verified here**: single-GPU LoRA SFT runs on
  an MI355X (ROCm 7.2) via the pip/venv route — see the MI355X platform notes
  in section 2. Upstream's README ships a dedicated ROCm docker path
  (`docker/docker-rocm/` with `docker compose up -d`, run with `--device /dev/kfd
  --device /dev/dri`), links AMD's official fine-tuning guide
  (<https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html>)
  under "Documentation (AMD GPU)", and links an AMD GPU Cloud free-credits notebook. No
  ROCm-specific pip extra exists; outside Docker you bring ROCm PyTorch yourself. The
  docker route remains untried here.
- **Ascend NPU** — documented. Python 3.12 + `pip install -r requirements/npu.txt` + the
  Ascend CANN Toolkit and Kernels, with prebuilt docker images
  (`hiyouga/llamafactory:latest-910b-ubuntu` etc.) and a dedicated install tutorial at
  `llamafactory.readthedocs.io` (`multibackend/npu/`). QLoRA on NPU needs a source-built
  bitsandbytes (`multi-backend-refactor` branch) and `double_quantization: false`.
- **This folder**'s shipped configs target 8x H100 80GB. Single-GPU LoRA SFT is verified
  on MI355X/ROCm (section 2); QLoRA, full-FT ZeRO-3, DPO and any multi-GPU topology remain
  **unrun**. On ROCm, `flash_attn: fa2` must become `sdpa`; the bitsandbytes QLoRA config
  is expected to work on gfx950 (bnb ≥0.50 ships ROCm binaries) but is untested.

## 9. Notes

- **The YAML is the program.** `stage` picks the algorithm, `finetuning_type` picks how many
  parameters move, and the rest are HF `Seq2SeqTrainingArguments` plus LLaMA-Factory's own
  dataclasses (`ModelArguments`, `DataArguments`, `FinetuningArguments`). That is why this
  folder ships four configs and one short launcher instead of a large Python trainer.
- **The launcher is deliberately dumb**, with one exception: it reads
  `dataset_info.json` and cross-checks it against the config. An unregistered dataset name,
  a missing data file, or preference data without `ranking: true` are the three failures that
  cost the most time in this framework, so they are caught before any GPU is touched. Beyond
  that it loads `dev.env`, checks required keys, prints a summary, `subprocess.run`s the CLI
  and propagates its exit code.
- **Launcher semantics.** `llamafactory-cli train` uses its own device logic; upstream
  requires `FORCE_TORCHRUN=1` for multi-GPU **full** fine-tuning, so the script sets it
  automatically when `finetuning_type: full` and more than one GPU. Device selection is
  `CUDA_VISIBLE_DEVICES` (`--gpus`), not a `--num-processes` flag.
- **Masking.** Default SFT supervises assistant turns only. `train_on_prompt: true`
  supervises the prompt as well; `mask_history: true` trains only on the final turn. The two
  are mutually exclusive upstream.
- **DPO reference model.** With LoRA, the adapter-disabled base acts as the reference, so no
  second copy of the weights is loaded — that is why the DPO recipe here uses LoRA. Set
  `ref_model:` to point at an explicit reference. Note `pref_loss: orpo` / `simpo` need no
  reference model at all.
- **Quantized bases do not shard.** QLoRA runs one 4-bit replica per GPU (DDP); combining it
  with ZeRO-3 does not work. Upstream has a separate FSDP+QLoRA recipe under
  `examples/extras/fsdp_qlora/`.

**Upstream API uncertainty (do not assume these are verified):**
- `template: qwen3_nothink` in these configs is the upstream template name for the
  non-reasoning Qwen3 variants; change it to match whatever model you actually train
  (the README's model table maps model family to template) — a wrong template trains
  against a chat format the model never sees at inference.
- The `deepspeed:` path is relative to the process CWD, and the exact filenames
  (`ds_z0_config.json`, `ds_z2_config.json`, `ds_z3_config.json`) come from the upstream
  examples — confirm them in your checkout.
- `packing` / `neat_packing` are real keys but interact with `cutoff_len` (upstream
  decrements `cutoff_len` by 1 when packing is on); they are left `false` here.
- PPO, KTO and reward modelling are supported by LLaMA-Factory but are **not** configured in
  this folder; they need extra pieces (a reward model, `kto_tag` data) — see
  `examples/train_lora/` upstream.
- Upstream has renamed the repository to `hiyouga/LlamaFactory`; the old
  `hiyouga/LLaMA-Factory` URL redirects, so the clone commands above keep working.
