# `training/llm/fsdp` — chat SFT on PyTorch-native FSDP2

> **Tested topology:** verified on **2x and 8x AMD Instinct MI355X (gfx950, ROCm 7.2.4)**
> on 2026-08-19 — see the tested section below, including the 8-GPU subsection. The folder was written against the upstream
> PyTorch, accelerate and Transformers docs as of **August 2026**; the NVIDIA (8x H100)
> and Intel XPU paths remain untested and their commands are a starting point, not a
> proven recipe.

## ✅ Tested on AMD MI355X (ROCm 7.2) — 2026-08-19

**Verdict: works with changes** (two fixes below; both are already applied / documented).
2-rank FSDP2 SFT of `google/gemma-4-E4B-it` (~8B params) ran end-to-end on 2x MI355X
(gfx950, 288GB), ROCm 7.2.4, Python 3.12.3: finite decreasing loss, both GPUs busy
(~41% VRAM each mid-run per `rocm-smi`), clean collective `save_model`, exit code 0.

Install — exactly as the AMD section below says (venv `.env_fsdp/` inside this
folder, git-ignored):

```bash
cd training/llm/fsdp && python3 -m venv .env_fsdp && source .env_fsdp/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> torch 2.11.0+rocm7.2
pip install -r requirements_fsdp.txt                                             # transformers 5.5.0, accelerate 1.14.0
```

> **Venv note:** the as-run transcripts below reference the campaign venv
> (`.env_train_llm_fsdp`) verbatim. The campaign venvs were removed during the 2026-08
> reorg — rebuild from `requirements_fsdp.txt` (new convention: `.env_fsdp`).

Launch (2 GPUs; scale `--num_processes` to your node — the YAML's default is 8):

```bash
accelerate launch --config_file fsdp2_config.yaml --num_processes 2 --fsdp_cpu_ram_efficient_loading false \
  train_llm_fsdp.py \
  --model_name google/gemma-4-E4B-it \
  --output_dir /mnt/data_1.5t/outputs/train_llm_fsdp/smoke_final \
  --num_train_epochs 2 --logging_steps 1 \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --gradient_checkpointing
```

Observed log lines:

```
INFO - __main__ - Accelerator: cuda | model: google/gemma-4-E4B-it | LoRA: False
INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
{'loss': '21.31', 'grad_norm': '1028', 'learning_rate': '0', 'epoch': '0.2'}
{'loss': '6.5', 'grad_norm': '96.13', 'learning_rate': '3.015e-07', 'epoch': '2'}
INFO - __main__ - Saved consolidated weights to .../smoke_final/final_model
INFO - __main__ - Training complete.
```

Quirks found and their fixes:

1. **transformers 5.x `apply_chat_template` change** (any hardware, any model): with the
   pinned `transformers==5.5.0`, `apply_chat_template(tokenize=True)` returns a
   `BatchEncoding` dict, not a token list — every row was dropped as "unsupervised".
   **Fixed in `train_llm_fsdp.py`** by passing `return_dict=False` at the three call
   sites in `build_example`.
2. **`fsdp_cpu_ram_efficient_loading: true` crashes with gemma-4-E4B-it**: accelerate
   1.14.0's `fsdp2_load_full_state_dict` raises
   `AttributeError: 'Tensor' object has no attribute 'device_mesh'` — some parameters of
   this architecture stay plain tensors after `fully_shard`. Workaround: add
   `--fsdp_cpu_ram_efficient_loading false` to the `accelerate launch` line (as above)
   or flip the flag in `fsdp2_config.yaml`. Mainstream llama-style models may not need
   this; the crash is architecture-specific.
3. **Non-quirks (worked as documented)**: RCCL selected as `nccl` with zero config,
   `--attn_implementation sdpa` (never install flash-attn on ROCm), bf16, GPU masking
   via `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`, and the collective
   `save_model` across both ranks.

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

**Verdict: WORKS.** The exact 2-GPU recipe above scales to all 8 MI355X with **no new
flags, no RCCL tuning and no OOM workarounds** — the only additions are operational
(`--no_save` for disk, an explicit GPU mask). 80 optimizer steps of FSDP2 full-shard SFT
on `google/gemma-4-E4B-it` (~8B) across `world_size=8`: finite monotonically-decreasing
loss, all 8 GPUs at 100% utilisation, clean teardown, **exit code 0**, no hang.

Launch (exactly what was run, under a machine-wide `flock` so it owned all 8 GPUs):

```bash
cd training/llm/fsdp && source .env_train_llm_fsdp/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # see the GPU-mask warning below
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/data_1.5t/hf_cache
source ./dev.env                                # HF_TOKEN
python -c "import torch;assert torch.cuda.device_count()==8"   # assert BEFORE training

accelerate launch \
  --config_file fsdp2_config.yaml \
  --num_processes 8 \
  --main_process_port 29610 \
  --fsdp_cpu_ram_efficient_loading false \
  train_llm_fsdp.py \
  --model_name google/gemma-4-E4B-it \
  --output_dir /mnt/data_1.5t/outputs/train_llm_fsdp/gpu8/run \
  --num_train_epochs 40 --logging_steps 1 \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --gradient_checkpointing \
  --no_save
```

`accelerate launch` is used rather than raw `torchrun` because `fsdp2_config.yaml` is what
supplies the FSDP2 plugin; `--main_process_port 29610` is the accelerate spelling of
torchrun's `--master_port` and avoids rendezvous collisions with other jobs on the box.

**Parallelism actually used**

| | |
|---|---|
| strategy | FSDP2 `fully_shard`, full sharding (params + grads + optimizer state), ZeRO-3 equivalent |
| world size | 8 ranks, 1 node, 1 rank per GPU |
| wrapping | `TRANSFORMER_BASED_WRAP`, `fsdp_reshard_after_forward: true`, `fsdp_offload_params: false` |
| batch geometry | per-device 1 x grad_acc 1 x 8 ranks = **global batch 8** |
| steps | 9 rows -> 2 steps/epoch x 40 epochs = **80 steps** |
| precision | bf16, `sdpa` attention, `--gradient_checkpointing` |

**Real log lines** (`/mnt/data_1.5t/outputs/train_llm_fsdp/gpu8/run.log`):

```
ASSERT device_count=8 torch=2.11.0+rocm7.2
INFO - __main__ - Loaded 9 rows from data/OTel_LLM_sample_10.jsonl (dropped 1 over 2048 tokens, 0 with no supervised tokens)
{'loss': '83.94', 'grad_norm': '2726', 'learning_rate': '0', 'epoch': '0.5'}
{'loss': '76.31', 'grad_norm': '2175', 'learning_rate': '6.667e-06', 'epoch': '1.5'}
{'loss': '0.02608', 'grad_norm': '19.11', 'learning_rate': '5.509e-06', 'epoch': '20'}
{'loss': '2.782e-05', 'grad_norm': '0.006154', 'learning_rate': '4.161e-09', 'epoch': '40'}
{'train_runtime': '101.6', 'train_samples_per_second': '3.544', 'train_steps_per_second': '0.788', 'train_loss': '6.252', 'epoch': '40'}
80/80 [01:41<00:00,  1.27s/it]
INFO - __main__ - --no_save set: skipped checkpointing and final save_model.
=== accelerate exit code: 0 ===
```

(The loss collapsing to ~3e-05 is the 8-row sample dataset being memorised over 40 epochs —
expected for a smoke test, not a quality signal.)

**8-GPU rocm-smi evidence** — full capture in
`/mnt/data_1.5t/outputs/train_llm_fsdp/gpu8/rocm_smi_8gpu.txt` (26 samples, 4s apart,
spanning the whole training loop). All 8 GPUs busy simultaneously, e.g.:

```
########## sample 20  2026-08-19T16:34:44+00:00 ##########
GPU[0..7] : GPU use (%):                  96  96  98  97  97  44  96  98
GPU[0..7] : GPU Memory Allocated (VRAM%): 18  18  19  19  18  19  18  18
GPU[0..7] : Socket Power (W):            419 414 419 417 404 425 412 425
GPU[0..7] : Temp junction (C):            46  46  45  49  48  49  48  47
```

Aggregate over the run: every GPU 0-7 reached **100%** utilisation; **16 of 26** samples
had *all eight* GPUs at >=80% simultaneously (the dips are collective waits between
micro-steps on an 80-step job, not idle GPUs); all 8 held VRAM in **every** sample.

**What differed from the 1-2 GPU run**

1. **No new flags, env vars or pins were needed.** RCCL initialised at 8 ranks with zero
   configuration — no `NCCL_*`/`RCCL_*` tuning, no init hang, no NUMA or topology warnings
   in the log. `requirements_fsdp.txt` is unchanged.
2. **`--fsdp_cpu_ram_efficient_loading false` is still required** at 8 ranks, for the same
   architecture-specific reason as quirk 2 above. Carried forward unchanged.
3. **Per-GPU VRAM dropped from ~41% to ~18.5%** (of 288 GB) going from 2 ranks to 8 —
   roughly **118 GB -> ~53 GB per GPU**. Sharding is genuinely dividing the parameter,
   gradient and optimizer-state footprint, which is the datapoint this folder exists to
   prove. Nowhere near OOM; there is ample headroom to raise `--batch_size` or
   `--max_seq_len` well beyond this smoke configuration.
4. **New `--no_save` flag (added to `train_llm_fsdp.py`)**: `save_strategy` was hardcoded
   to `"epoch"`, so a 40-epoch 8B run would have written dozens of consolidated
   FULL_STATE_DICT checkpoints and filled the disk. `--no_save` sets `save_strategy="no"`
   **and** skips the final `save_model`. It skips it on *all* ranks — that symmetry is
   mandatory, because `save_model` is a collective under FSDP and a rank-0 guard would
   hang the other seven. Total disk written by this run: **396 KB** (logs + tensorboard).
   The 2-GPU run's collective `save_model` path is untouched and still the default.
5. **Beware the venv's GPU pin.** `.env_train_llm_fsdp/bin/activate` had a leftover
   `export CUDA_VISIBLE_DEVICES=4,5` from the 2-GPU session, which silently caps you at
   2 GPUs after `source`. It has been commented out; regardless, always set the mask
   explicitly after activating and assert `torch.cuda.device_count() == 8` before
   training, or you will report an 8-GPU pass you never ran.

### 4-GPU sharding run — **collective `save_model` enabled** (4×MI355X, ROCm 7.2.4) — tested August 2026

> Closes the one gap the 8-GPU run above deliberately left open: it ran with `--no_save`,
> so the collective FULL_STATE_DICT `save_model` had only ever been proven at **2** ranks.
> This run repeats FSDP2 full-shard SFT at **4** ranks with saving **on** (default
> `save_strategy="epoch"` plus the final `save_model`). Nothing below contradicts the
> 2-GPU or 8-GPU results.

Verified 2026-08-19 on physical GPUs **4,5,6,7** of the same node (a sibling job owned
0-3), same venv and pins. **Verdict: FSDP2 holds at 4 GPUs, and the collective
`save_model` completes — no deadlock, no rank divergence, rc=0.**

```bash
cd training/llm/fsdp && source .env_train_llm_fsdp/bin/activate
export HIP_VISIBLE_DEVICES=4,5,6,7      # re-export after activate (see quirk 5 above)
export CUDA_VISIBLE_DEVICES=4,5,6,7     # renumber to 0-3 inside the process
export HF_HOME=/mnt/data_1.5t/hf_cache
source ./dev.env                        # HF_TOKEN
python -c "import torch;assert torch.cuda.device_count()==4"

accelerate launch \
  --config_file fsdp2_config.yaml \
  --num_processes 4 \
  --main_process_port 29793 \
  --fsdp_cpu_ram_efficient_loading false \
  train_llm_fsdp.py \
  --model_name google/gemma-4-E4B-it \
  --output_dir /mnt/data_1.5t/outputs/train_llm_fsdp_4gpu/run \
  --num_train_epochs 1 --save_total_limit 1 --logging_steps 1 \
  --train_file <200-row messages jsonl: the shipped 10-row sample tiled ×20> \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 2048 \
  --gradient_checkpointing
```

Note the **absence of `--no_save`** — that is the entire point of this run.

**Evidence (from the run logs):**

```
[assert] device_count= 4
{'loss': '41.06',  'grad_norm': '1475',  'learning_rate': '0',        'epoch': '0.02222'}
{'loss': '1.483',  'grad_norm': '55.25', 'learning_rate': '5.908e-06','epoch': '0.4889'}
{'loss': '0.0247', 'grad_norm': '1.588', 'learning_rate': '1.334e-08','epoch': '1'}
{'train_runtime': '122.6', 'train_samples_per_second': '1.468', 'train_steps_per_second': '0.367', 'train_loss': '7.332'}
[RANK 0] Saving model to .../checkpoint-45/pytorch_model_fsdp.bin
[RANK 0] Model saved to .../checkpoint-45/pytorch_model_fsdp.bin
[RANK 0] Saving Optimizer state to .../checkpoint-45/optimizer.bin
INFO - __main__ - Saved consolidated weights to .../run/final_model
INFO - __main__ - Training complete.        # rc=0
```

`rocm-smi` sampled every 5 s *during* training (all four owned GPUs):

```
=== 20:39:34 ===
GPU[4]: GPU use (%): 100    VRAM Total Used Memory (B): 79404371968   # 73.9 GiB
GPU[5]: GPU use (%): 100    VRAM Total Used Memory (B): 77335080960   # 72.0 GiB
GPU[6]: GPU use (%): 100    VRAM Total Used Memory (B): 79404269568   # 73.9 GiB
GPU[7]: GPU use (%): 100    VRAM Total Used Memory (B): 80386899968   # 74.9 GiB
```

**Findings specific to saving at 4 ranks:**

- **The collective `save_model` completes and scales past 2.** 45 steps then a full
  FULL_STATE_DICT save; `final_model/model.safetensors` is
  **34,669,265,436 bytes (~32.3 GiB)** and directly loadable. Confirms quirk 4's warning
  from the other direction: because the call is symmetric on all ranks it simply works —
  a rank-0 guard here would hang the other three, exactly as it would at 8.
- ⚠️ **The consolidated save is written in fp32, not bf16.** ~32 GiB for an 8B model —
  *twice* what DeepSpeed ZeRO-3 writes for the same model on this box (~14.9 GiB, which
  uses `stage3_gather_16bit_weights_on_model_save`). FSDP2's FULL_STATE_DICT gathers the
  fp32 master weights. Budget accordingly; this is a real difference between the two
  folders, not a measurement error.
- ⚠️ **Full disk cost of one saving run is ~99 GB, not ~32 GB.** The epoch checkpoint
  (`checkpoint-45`) is **84 GB** on its own, because `optimizer.bin` is a consolidated
  fp32 Adam state (~2× params again) on top of the 15 GB `pytorch_model_fsdp.bin`; the
  `final_model` adds another 32 GB. Use `--save_total_limit 1` and delete immediately, or
  keep `--no_save` for smoke runs as the 8-GPU section does.
- **During the save phase, ranks 1-3 sit at 100 % GPU utilisation while rank 0 writes.**
  That is a busy-wait inside the collective barrier, not useful work and not a hang — it
  resolved on its own each time. Do not mistake it for a live training step when sampling
  `rocm-smi`; cross-check against the log.
- **Per-GPU VRAM even across ranks** (73.9 / 72.0 / 73.9 / 74.9 GiB, ~2.9 GiB spread,
  stable across every in-training sample). Higher than the 8-rank figure (~53 GB) purely
  because the same model is sharded 4 ways instead of 8 — sharding is behaving as
  advertised.
- No RCCL tuning, no new flags, no code change. Port 29793 (29500 collides on a shared box).

## ✅ Tested on NVIDIA H100 (CUDA 13.0) — 2026-08-22 (single-GPU smoke)

**Verdict: WORKS-WITH-CHANGES.** Single-GPU FSDP2 SFT of `google/gemma-4-E4B-it` (~8B)
runs end-to-end on **1x H100 80GB HBM3** (driver 580.173.02, CUDA 13.0, Hopper cc(9,0),
Python 3.12.3): 20 optimizer steps, **finite decreasing loss (20.1 → ~1–8)**, consolidated
fp32 model saved, GPU-6 residency confirmed by PID, clean exit 0. **But the shipped
`fsdp2_config.yaml` does not run this model as-is** — it took four changes: (1) venv off
the SMB repo mount, (2) **untie** gemma-4's word embeddings (tied-param wrap error, both
full-FT and LoRA), (3) load the model from a **local snapshot copy by explicit path**,
(4) `reshard_after_forward: false` + `fsdp_offload_params: true` (a DTensor forward bug at
`reshard:true`, then OOM without offload). Two of these (2, 4) look like **torch-2.13
regressions** vs the MI355X torch-2.11 recipe. Single-GPU FSDP is degenerate
(`world_size=1` shards nothing) but still exercises the `fully_shard` wrap + fwd/bwd +
FULL_STATE_DICT save path — a valid smoke. Multi-GPU (2/8) is DEFERRED (a sibling
production job owned GPUs 0-3); see the deferred note below.

Four H100-specific changes were needed vs the MI355X recipe — all documented here.

### Install (CUDA 13.0) — what actually worked

The repo pins `torch==2.11.0`, but **there is no cu130 wheel for 2.11.0**; the verified
H100 recipe is the current stable, which ships native CUDA 13:

```bash
# venv on a LOCAL fs, NOT the repo dir — see change (1) below
python3 -m venv /dev/shm/h100/venvs/fsdp && source /dev/shm/h100/venvs/fsdp/bin/activate
pip install torch numpy                 # -> torch 2.13.0+cu130 (NO --index-url needed)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 / 13.0
# install the HF stack WITHOUT the torch/numpy lines (torch already satisfied):
grep -vE '^\s*torch==|^\s*numpy==' requirements_fsdp.txt | grep -vE '^\s*#|^\s*$' > /tmp/r.txt
pip install -r /tmp/r.txt                # transformers 5.5.0, accelerate 1.14.0, peft 0.20.0
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # RE-VERIFY: still 2.13.0+cu130 (HF stack did NOT clobber torch)
```

**Key versions:** torch **2.13.0+cu130** (cuda 13.0), transformers 5.5.0, accelerate
1.14.0, datasets 4.3.0, peft 0.20.0, numpy 2.5.2, driver **580.173.02**. `fully_shard`
imports; bf16 matmul verified on-GPU. tf32 note: `torch.backends.cuda.matmul.allow_tf32`
is **False** by default in 2.13.0 and the trainer runs **bf16** mixed precision (not fp32),
so the tf32 path is not the active matmul path here — nothing to "turn on" for this run.

### Exact smoke command (single GPU, GPU 6, port 29643)

**The shipped `fsdp2_config.yaml` does NOT run this model as-is** — full FT and LoRA
both hit the tied-embedding wrap error, and after untieing, the `reshard_after_forward:
true` default hits a DTensor/tensor forward bug (see changes 2–4). The command that
actually trains to a decreasing loss uses `reshard_after_forward: false` +
`fsdp_offload_params: true` and an untie shim:

```bash
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./dev.env
# load from the explicit snapshot DIR, not the repo id (change 3); a LOCAL tmpfs copy (change 3b)
MODEL=/dev/shm/h100/models/gemma-4-E4B-it
python -c "import torch;assert torch.cuda.device_count()==1"

# fsdp2_offload.yaml = fsdp2_config.yaml with reshard_after_forward:false + offload_params:true
# untie_launch.py = 6-line shim that sets tie_word_embeddings=False then runs the shipped script
accelerate launch --config_file fsdp2_offload.yaml --num_processes 1 \
  --main_process_port 29643 --fsdp_cpu_ram_efficient_loading false \
  untie_launch.py \
  --model_name "$MODEL" \
  --output_dir /dev/shm/h100/out/fsdp/offload_run \
  --num_train_epochs 5 --logging_steps 1 \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 1024 \
  --attn_implementation sdpa --gradient_checkpointing --no_save
```

Full fine-tuning (not LoRA) mirrors the MI355X campaign. `--no_save` for the
loss/residency smoke; the consolidated-save behavior is in its own run below.

**Real log (single H100, GPU 6, offload config) — 20 optimizer steps, exit 0:**

```
INFO __main__ Loaded 4 rows from data/OTel_LLM_sample_10.jsonl (dropped 6 over 1024 tokens, 0 with no supervised tokens)
INFO __main__ Starting training
{'loss': '20.12', 'grad_norm': '598.4', 'learning_rate': '0',        'epoch': '0.25'}
{'loss': '15.5',  'grad_norm': '147.3', 'learning_rate': '9.932e-06','epoch': '0.75'}
{'loss': '8.5',   'grad_norm': '307',   'learning_rate': '8.946e-06','epoch': '1.5'}
{'loss': '1.031', 'grad_norm': '85.3',  'learning_rate': '1.054e-06','epoch': '4.25'}
{'train_runtime': '608', 'train_samples_per_second': '0.033', 'train_steps_per_second': '0.033', 'train_loss': '9.532', 'epoch': '5'}
INFO __main__ --no_save set: skipped checkpointing and final save_model.  /  Training complete.
```

Loss trends 20.1 → ~1–8 (noisy: 4-row micro-sample, batch 1); grad_norm 598 → ~46.
`train_runtime` 608 s — slow because CPU offload shuttles the full 8B optimizer state
each step (the price of fitting full FT on one card). **GPU-6 residency proof**, sampled
by PID from inside the run with `nvidia-smi -i 6`:

```
$ nvidia-smi -i 6 --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
1342078, /dev/shm/h100/venvs/fsdp/bin/python3, 45898 MiB     # our worker on GPU 6
$ nvidia-smi -i 6 --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
NVIDIA H100 80GB HBM3, 0 %, 46089 MiB, 81559 MiB
```

### H100-specific changes vs the MI355X recipe

1. **venv cannot live in the repo folder — it is an SMB2 mount.** `python3 -m venv
   .env_fsdp` inside `training/llm/fsdp/` fails with `Operation not permitted:
   '.../bin/Activate.ps1'` (the share rejects creating the activate scripts;
   `stat -f` reports `smb2`). Build the venv on a local fs (`/dev/shm/h100/venvs/fsdp`)
   and activate that. Functionally identical; only the venv location moves.
2. **LoRA (`--use_lora`) is BLOCKED on gemma-4-E4B-it under FSDP2 + torch 2.13.** The
   first forward raises `ValueError: Parameter
   'base_model.model.model.language_model.embed_tokens.weight' is shared with a parameter
   already managed by another FSDP group. For shared/tied parameters, use
   fully_shard([module_a, module_b]) ...`. gemma-4 **ties input/output embeddings**;
   PEFT + `TRANSFORMER_BASED_WRAP` place the tied weight into two `fully_shard` groups,
   which FSDP2 forbids. The MI355X campaign only ever ran **full FT** on this model, so
   this path was never exercised there — it is newly surfaced on H100. **Full FT hits the
   exact same error** (`Parameter 'model.language_model.embed_tokens.weight' is shared ...`)
   — it is NOT LoRA-specific: gemma-4 `tie_word_embeddings: True` collides with
   `TRANSFORMER_BASED_WRAP` under FSDP2/torch-2.13 regardless. Setting
   `fsdp_transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer` does **not** help.
   **Workaround used:** untie the embeddings at load (`tie_word_embeddings=False`), via a
   6-line shim `untie_launch.py` that patches
   `transformers ... _BaseAutoModelClass.from_pretrained` and then runs the shipped script
   unchanged (the patch must live in the accelerate *worker*, so a wrapper that accelerate
   launches — not a monkeypatch in the parent — is required; `sitecustomize.py` did not
   take on the Auto metaclass). This is a **torch-2.13 regression**: 2.11.0 (MI355X)
   tolerated the tied weight; 2.13.0's `_validate_no_duplicate_params` rejects it.
3. **Load the model from a LOCAL snapshot copy, and by explicit path.** Two sub-issues:
   (a) `--model_name google/gemma-4-E4B-it` fails offline under transformers 5.5.0 —
   the cached blob layout doesn't resolve, and it falls through to
   `ValueError: Couldn't instantiate the backend tokenizer ...`. Passing the explicit
   snapshot dir (`.../snapshots/<hash>`) loads cleanly (GemmaTokenizer + chat template).
   (b) The HF cache lives on the **slow SMB share**; loading 15 GB of weights by mmap
   made the first forward block for minutes in `folio_wait_bit_common` (page-fault wait
   on SMB). Copying the snapshot to tmpfs (`cp -rL <snap> /dev/shm/h100/models/...`,
   ~15 GB) removes the bottleneck. Network to huggingface.co is 403-blocked on this box,
   so `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` + a fully-cached model are mandatory.
4. **`fsdp_reshard_after_forward: true` (the shipped default) breaks gemma-4's forward
   under FSDP2.** After untieing, the first forward raises `RuntimeError: aten.where.self
   got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor ...`
   (in `modeling_gemma4.py`) — the resharded params are DTensors but a mask/buffer in the
   gemma-4 forward is a plain tensor, and `torch.where` refuses the mix. Also a plausible
   torch-2.13 DTensor-strictness regression. **Workaround:** set
   `fsdp_reshard_after_forward: false` (keep params gathered after forward, ~ZeRO-2). The
   forward then runs. This is the config change that made training actually step. (A
   `reshard: true`/ZeRO-3 fix would need a gemma-4 modeling patch to DTensor-ify the mask;
   out of scope.)

Non-changes (worked as-is): NCCL selected with zero config; `sdpa` attention (flash-attn
FA2 not attempted — network is blocked so no wheel/build is possible offline; `sdpa` is
the portable default and works); bf16; `--fsdp_cpu_ram_efficient_loading false` carried
forward from MI355X quirk 2 (kept for safety on this tied-weight architecture).

### VRAM — the 80 GB constraint (vs 288 GB on MI355X)

Single-GPU FSDP does **not** shard across ranks (`world_size=1`), so one card holds a full
replica. Measured on GPU 6 (`nvidia-smi -i 6`, sampled through the run):

| Config (full FT, 8B, bf16, grad-ckpt) | Result | Peak HBM |
|---|---|---|
| `reshard:false`, offload:**false** | **OOM** during optimizer step | 79.16 GiB used, **8 MiB free** → `torch.OutOfMemoryError` |
| `reshard:false`, offload:**true**, `--no_save` | trains, 20 steps, exit 0 | **~46 GB** steady |
| `reshard:false`, offload:**true**, save ON | trains + saves, exit 0 | **~80 GB peak at save** (46 GB training → **80389 MiB** during consolidation) |

**The consolidated save is the memory cliff.** With `fsdp_state_dict_type: FULL_STATE_DICT`
the fit-end save gathers the model to **fp32**: the written
`final_model/model.safetensors` is **34.7 GB** (dtypes `{F32, BF16}` — matches the MI355X
note's "~32 GB for an 8B model"), and the fit-end training checkpoint (`save_strategy:
epoch`, weights **+ fp32 Adam optimizer state**) is **126 GB on disk**. During that gather
HBM spiked from the 46 GB training plateau to the **full 80 GB card** — right at the OOM
edge even *with* CPU offload enabled. This is the single-card echo of the MI355X campaign's
"a saving run cost ~99 GB with optimizer state on 288 GB cards": on 80 GB the save, not the
training, is what nearly kills you.

**Practical guidance for 1x 80 GB:** full FT of an 8B model needs `offload_params: true`
(without it the optimizer step OOMs). To avoid the save-time spike entirely on a single
card, either `--no_save` (as in the smoke), use `fsdp_state_dict_type: SHARDED_STATE_DICT`,
or defer saving to a multi-GPU run where the fp32 gather is spread across ranks. **Delete
the fit-end checkpoint after** — 126 GB per epoch fills a disk fast. (`save_strategy: "no"`
/ `--no_save` does stop the fit-end save here — verified exit 0 with "skipped ...
save_model".)

### 2-GPU run (2× H100) — 2026-08-23 (REAL multi-rank sharding proven)

**This is the run that proves FSDP2's whole reason to exist: params really split across
ranks.** The single-GPU smoke above is degenerate (`world_size=1` shards nothing); here
`fully_shard` splits the model across **2× H100 80GB** and each rank holds **exactly half**.
GPUs 4 AND 5 only (`CUDA_VISIBLE_DEVICES=4,5`); GPUs 0-3 were a co-tenant production job
(PIDs 1273508-11) and were **never touched** — sampled `-i 0,1,2,3` throughout, only those
four PIDs ever appeared. Port 29671, same reused venv (torch 2.13.0+cu130 / transformers
5.5.0 / accelerate 1.14.0).

**Model change vs the single-GPU section:** this pass runs **`LiquidAI/LFM2.5-350M`**
(`lfm2`, 16 `Lfm2DecoderLayer` blocks, fully cached with a working tokenizer), *not*
gemma-4-E4B-it. Reason: at 2 GPUs we want to (a) prove the per-rank param split cheaply and
(b) exercise the consolidated `FULL_STATE_DICT` save without the 80 GB save-cliff that the
8B gemma model hits — LFM2.5-350M's fp32 save is ~1.6 GB and peaks at ~6.7 GB/GPU. The two
carried-over H100 fixes still apply: **untie the embeddings** (LFM2 has
`tie_word_embeddings: True`; torch-2.13 FSDP2 rejects the shared `embed_tokens`/`lm_head`,
same error class as gemma-4 — see change 2 above) via the shipped `untie_launch.py`, and
`--attn_implementation sdpa`, `--max_seq_len 1024`. **Unlike gemma-4, LFM2 does NOT need the
`reshard_after_forward: false` workaround** — it runs on the shipped `fsdp2_config.yaml`
(`fsdp_reshard_after_forward: true`, the true ZeRO-3 full-shard, no offload). That gemma-4
DTensor/`aten.where` forward bug (change 4) is model-specific, not universal on torch-2.13.

Batch geometry: `batch_size 1 × world 2 × grad_acc 1` → **global batch 2**. The sample has
10 rows; 6 exceed 1024 tokens under the LFM2 tokenizer and are dropped, leaving **4 rows →
2 optimizer steps/epoch**, so `--num_train_epochs 6` gives **12 steps** (≥8, decreasing).

```bash
export CUDA_VISIBLE_DEVICES=4,5 HF_HOME=/mnt/gsma/gsma/gsma/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE=/dev/shm/h100/dscache_fsdp2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./dev.env
python -c "import torch;assert torch.cuda.device_count()==2"   # 2 GPUs visible

# fsdp2_config.yaml = shipped config (reshard_after_forward:TRUE, no offload — full ZeRO-3)
# untie_launch.py   = shim that sets tie_word_embeddings=False, then runs the shipped script
accelerate launch --config_file fsdp2_config.yaml --num_processes 2 \
  --main_process_port 29671 \
  untie_launch.py \
  --model_name LiquidAI/LFM2.5-350M \
  --output_dir /dev/shm/h100/out/fsdp2/save_run \
  --num_train_epochs 6 --logging_steps 1 \
  --batch_size 1 --grad_acc_steps 1 --max_seq_len 1024 \
  --attn_implementation sdpa --gradient_checkpointing
# (drop --output_dir/keep --no_save for a save-free smoke; add --no_save to skip the save.)
```

**Proof #1 — genuine 2-rank full-shard (the key signal).** A per-rank probe on the first
optimizer step (patched into the accelerate *worker*, same mechanism as `untie_launch.py`)
reports each rank's LOCAL DTensor shard vs the GLOBAL param count. Both ranks hold **exactly
0.500** of the model — `fully_shard` split every one of the 149 params across GPU 4 and
GPU 5:

```
[SHARD-PROBE rank 0/2] DTensor_params=149 local_param_elems=210,796,416 global_param_elems=421,592,832 local/global=0.500 (expect ~0.500 for full-shard)
[SHARD-PROBE rank 1/2] DTensor_params=149 local_param_elems=210,796,416 global_param_elems=421,592,832 local/global=0.500 (expect ~0.500 for full-shard)
```

(Global is 421 M, not 350 M, because untieing adds a separate `lm_head.weight`
≈ vocab 65536 × hidden 1024 ≈ 67 M — the `LOAD REPORT: lm_head.weight MISSING` line
confirms it was materialized as an independent, newly-initialized tensor.)

**Proof #2 — finite decreasing loss, 12 steps, exit 0** (identical curve in the no_save and
the save runs):

```
2026-08-23 - INFO - __main__ - Loaded 4 rows from data/OTel_LLM_sample_10.jsonl (dropped 6 over 1024 tokens, 0 with no supervised tokens)
{'loss': '12.94', 'grad_norm': '54.81', 'learning_rate': '0',        'epoch': '0.5'}
{'loss': '11.11', 'grad_norm': '38.37', 'learning_rate': '9.206e-06','epoch': '2'}
{'loss': '6.391', 'grad_norm': '171.5', 'learning_rate': '4.288e-06','epoch': '4'}
{'loss': '3.918', 'grad_norm': '108.1', 'learning_rate': '2.025e-07','epoch': '6'}
{'train_runtime': '4.661', 'train_samples_per_second': '5.149', 'train_steps_per_second': '2.574', 'train_loss': '8.735', 'epoch': '6'}
```

Loss 12.94 → 3.918 over 12 steps (noisy: 4-row micro-sample, global batch 2). `train_runtime`
was **4.7 s** for the `--no_save` run; the save run measured 42.5 s (activation-checkpointing
recompute + the fit-end collective gather — not a per-step regression).

**Proof #3 — BOTH GPU 4 and GPU 5 busy by PID** (sampled `-i 4,5` from inside the run; two
distinct worker PIDs, one per GPU, VRAM rising as training progressed):

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 4,5
1740250, 2926 MiB     # rank-0 worker on GPU 4
1740251, 3142 MiB     # rank-1 worker on GPU 5
# meanwhile GPUs 0-3 held ONLY the production PIDs, every sample:
$ nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1,2,3 | sort -u
1273508  1273509  1273510  1273511
```

**Proof #4 — consolidated (FULL_STATE_DICT) save works at world_size=2.** Dropping
`--no_save`, the fit-end collective gathers the 2 shards to one fp32 checkpoint on rank 0:

```
2026-08-23 - INFO - __main__ - Saved consolidated weights to /dev/shm/h100/out/fsdp2/save_run/final_model
2026-08-23 - INFO - __main__ - Training complete.
```

`final_model/model.safetensors` is **1.57 GiB** (fp32; reloads cleanly with
`tie_word_embeddings: False`, lm_head ≠ embed, chat template intact). VRAM during the save
peaked at just **~6.7 GB/GPU** (6490 / 6746 MiB on GPU 4 / 5) — no save-cliff, because the
model is small and the fp32 gather is spread across the 2 ranks (contrast the single-GPU 8B
run, which spiked to the full 80 GB card at save). The fit-end training checkpoints
(`checkpoint-*/optimizer.bin`, ~3.1 GB fp32 Adam each + sharded `pytorch_model_fsdp.bin`)
were **deleted after capturing this evidence** (`save_strategy: epoch` writes one per epoch —
they fill a disk fast even for a 350M model).

**Verdict (2-GPU): WORKS.** Real FSDP2 full-shard across 2× H100 proven by the exact 0.500
per-rank param split, decreasing loss over 12 steps, both GPUs resident by PID, and a working
consolidated save — all on the *shipped* `fsdp2_config.yaml` (no offload, `reshard:true`),
with only the model-agnostic untie shim carried over. **8-GPU is projected, not measured:**
the co-tenant production job holds GPUs 0-3, so a full 8-rank pass (`--num_processes 8`,
per-rank fraction → ~0.125) was not run this wave; the 2-rank 0.500 split is the direct
evidence that it would shard as advertised.

### Multi-GPU (deferred)

Not run this wave (GPUs 0-3 were a sibling production job; only GPU 6 was free). A 2- or
8-GPU pass would be the exact MI355X `accelerate launch --num_processes N` recipe with
`CUDA_VISIBLE_DEVICES` set to the free GPUs and `--main_process_port 29643`, asserting
`torch.cuda.device_count()==N` before launch. Only at `world_size>=2` does FSDP2 actually
shard params/grads/optimizer state across ranks — that is where the per-GPU VRAM drop
(the datapoint this folder exists to prove) becomes visible, and where the collective
`save_model` should be re-tested on CUDA.

## Overview & when to use

Chat-model SFT on **PyTorch-native FSDP2** (`fully_shard`), driven by HF Transformers'
`Trainer` and launched with accelerate. It supports **full fine-tuning** (default) and an
optional **LoRA** path, and shards parameters, gradients and optimizer state across GPUs
the way DeepSpeed ZeRO-3 does — with **no DeepSpeed in the stack at all**. That is the
reason to pick this folder over `../deepspeed/`: DeepSpeed drags in JIT-compiled
CUDA kernels and a `CUDA_HOME` toolchain, while FSDP2 ships inside torch itself and runs
unmodified on NVIDIA CUDA, AMD ROCm and Intel XPU.

The training script loads `dev.env` right after its imports and reads `HF_TOKEN` from the
environment at model-download time — no token is ever hardcoded. `save_model` runs on
**all** ranks at the end of training because under FSDP it is a collective that
reconstructs the sharded weights; guarding it with `if rank == 0` would hang the other
ranks. `device_map` stays `None` on load — FSDP2 places and shards the parameters itself.

Files in this folder:
- `train_llm_fsdp.py` — the training entrypoint (data loading, masking, Trainer setup).
- `fsdp2_config.yaml` — accelerate config that turns on FSDP2 (`fsdp_version: 2`); the primary launch route.
- `fsdp2_torchrun.json` — Trainer-side FSDP config (`"version": 2`) for the raw `torchrun` route.
- `requirements_fsdp.txt` — dependencies, with per-accelerator torch install notes.
- `data/OTel_LLM_sample_10.jsonl` — a 10-row sample dataset the script points at by default.

## Install

Python 3.12 in its own venv. **Install `torch` first**, using the index that matches your
accelerator — the plain PyPI wheel is the CUDA build:

```bash
python3.12 -m venv ~/.venv-fsdp && source ~/.venv-fsdp/bin/activate
```

### NVIDIA (CUDA)

```bash
pip install torch==2.11.0
pip install -r requirements_fsdp.txt
```

### AMD (ROCm)

torch 2.11.0 ROCm wheels are published on the **rocm7.2** index (the older `rocm6.4`
index stops at torch 2.9.1 — verified against download.pytorch.org, 2026-08):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_fsdp.txt
```

### Intel (XPU)

torch 2.11.0 XPU wheels are on the dedicated `xpu` index:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements_fsdp.txt
```

Verify the accelerator and the FSDP2 entrypoint are both visible:

```bash
python -c "import torch; from torch.distributed.fsdp import fully_shard; \
print('fully_shard OK', torch.accelerator.current_accelerator(), torch.accelerator.device_count())"
```

### Portability: what changes per vendor

FSDP2 is device-agnostic — it shards each parameter as a `DTensor` over a `DeviceMesh`
whose device type comes from the running accelerator, so the training code is identical
on all three vendors. Only the environment changes:

| | NVIDIA CUDA | AMD ROCm | Intel XPU |
|---|---|---|---|
| torch wheel | default PyPI | `--index-url .../whl/rocm7.2` | `--index-url .../whl/xpu` |
| Device API | `torch.cuda` | `torch.cuda` (ROCm reuses the CUDA API surface) | `torch.xpu` |
| Collective backend | NCCL | RCCL (still selected as `nccl`) | XCCL |
| Attention | `sdpa`, or `flash_attention_2` if you can build it | `sdpa` (FA2 wheels are CUDA-only) | `sdpa` |
| Visible-device env var | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` (also `ROCR_VISIBLE_DEVICES`) | `ZE_AFFINITY_MASK` |

Notes that matter in practice:
- `--attn_implementation` defaults to **`sdpa`** because it is the one kernel path
  available everywhere. Only pass `flash_attention_2` on CUDA, and only if `flash-attn`
  actually built (`--no-build-isolation` plus a matching `CUDA_HOME`).
- accelerate selects the process-group backend from the detected device, so
  `fsdp2_config.yaml` contains nothing vendor-specific.
- `bf16` is used throughout; all three targets support it. Do not switch to `fp16` for
  full fine-tuning.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hub token (needed for gated models):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_fsdp.py` loads it with `load_dotenv("dev.env")` and reads `HF_TOKEN` from the
environment, so run the script from inside `training/llm/fsdp/`. `dev.env` is **git-ignored**
at the repo root. Never hardcode a token in the source and never commit one — if a token
was ever committed, rotate it on the Hub immediately.

## Data

`--train_file` defaults to the shipped sample, `data/OTel_LLM_sample_10.jsonl` — 10
single-turn conversations for smoke-testing the pipeline. It is a chat JSONL — one
conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The sample rows also carry extra bookkeeping columns (`unmask`, `flow`, `source_id`,
`source_repo`, `source_spec_id`, `source_version`; `source_spec_id`/`source_version` are
null in most rows). The loader reads only the `messages` key from each line, so extra
columns are ignored — your own data may include or omit them freely.

To train on real data, pass `--train_file /path/to/train.jsonl` with the same
`messages` schema. A `system` turn and extra user/assistant turns are fine. Each row is
rendered with the tokenizer's own `apply_chat_template`, so training matches inference
formatting exactly; a model whose tokenizer has no chat template is rejected up front.

**Loss masking:** completion-only by default (`--mask_prompt`, on). Every token outside
an assistant turn is set to `-100`, so loss falls on assistant turns only — in a
multi-turn row, *all* assistant turns are supervised. Pass `--no_mask_prompt` to train on
the full rendered sequence. Rows longer than `--max_seq_len` are **dropped, never
truncated**, and rows with no supervised token are dropped too; both counts are logged.

## Run

Smoke test first — one GPU, the shipped sample, one epoch, so failures surface in minutes:

```bash
python train_llm_fsdp.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --logging_steps 1 \
  --output_dir ./fsdp_smoke
```

Full run — **accelerate with the FSDP2 config**, 8 GPUs, from inside `training/llm/fsdp/`:

```bash
nohup accelerate launch --config_file fsdp2_config.yaml --num_processes 8 \
  train_llm_fsdp.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_full_ft \
  --max_seq_len 4096 \
  --batch_size 1 --grad_acc_steps 8 \
  --num_train_epochs 3 --learning_rate 1e-5 \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &

tail -f train_llm_fsdp.log
```

LoRA instead of full fine-tuning — same launcher, add `--use_lora` and raise the LR:

```bash
nohup accelerate launch --config_file fsdp2_config.yaml --num_processes 8 \
  train_llm_fsdp.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_lora \
  --use_lora --lora_r 32 --lora_alpha 64 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &
```

**Raw torchrun route** (no accelerate config; the Trainer builds the FSDP2 plugin from
its own JSON — pass `--fsdp_config`, otherwise the run is plain DDP and will OOM on a
large model):

```bash
nohup torchrun --standalone --nproc_per_node=8 \
  train_llm_fsdp.py \
  --fsdp_config fsdp2_torchrun.json \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./fsdp_full_ft \
  --gradient_checkpointing \
  > train_llm_fsdp.log 2>&1 &
```

**What "working" looks like:** a startup line naming the accelerator and model, then the
row-count line (`Loaded N rows ... dropped M over K tokens`), then `{'loss': ...,
'grad_norm': ..., 'learning_rate': ...}` every `--logging_steps` with a finite
`grad_norm` and a loss that moves. Per-GPU memory should sit well below a
single-GPU-replica run — that is the sharding working. At the end you get
`Saved consolidated weights to .../final_model` (or `Saved a LoRA ADAPTER ...`) followed
by `Training complete.`

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL: one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./fsdp_run` | Checkpoints + `final_model/` land here |
| `--resume_from_checkpoint` | `""` | Checkpoint dir to resume from (used only if it exists) |
| `--max_seq_len` | `4096` | Token cap per row; longer rows are dropped, never truncated |
| `--max_samples` | `None` | Hard cap on rows loaded (quick smoke runs) |
| `--eval_samples` | `0` | Rows held out for a per-epoch eval (0 = off) |
| `--mask_prompt` / `--no_mask_prompt` | on | Completion-only loss (default) vs full-sequence loss |
| `--batch_size` | `1` | Per-device train/eval batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps (effective batch = batch x acc x world size) |
| `--num_train_epochs` | `3.0` | Number of training epochs |
| `--learning_rate` | `1e-5` | Peak LR (full FT ~1e-5..2e-5; LoRA ~1e-4..2e-4) |
| `--lr_scheduler_type` | `cosine` | LR scheduler type |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--weight_decay` | `0.0` | Weight decay |
| `--logging_steps` | `10` | Log training metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints to keep |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Recompute activations; leave `fsdp_activation_checkpointing: false` in the YAML when using it |
| `--attn_implementation` | `sdpa` | `sdpa` (portable default), `flash_attention_2` (CUDA only), `eager` |
| `--fsdp_config` | `None` | Trainer FSDP JSON for the torchrun route; leave unset under accelerate |
| `--use_lora` | off | Train a LoRA adapter instead of full fine-tuning |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` or a comma-separated module list |

## Output

Under `--output_dir`:

- `checkpoint-<step>/` — per-epoch Trainer checkpoints (model + optimizer + scheduler
  state), capped by `--save_total_limit`; pass one back via `--resume_from_checkpoint`.
- `final_model/` — the end-of-training artifact plus the tokenizer files. Full
  fine-tuning writes consolidated `safetensors` weights loadable straight with
  `AutoModelForCausalLM.from_pretrained`. With `--use_lora` it is a LoRA adapter instead:
  load the base model and apply it with `PeftModel.from_pretrained`, or merge it with
  `../peft/merge_adapter.py`.
- `runs/` — TensorBoard event files (`report_to="tensorboard"`); view with
  `tensorboard --logdir <output_dir>/runs`.
- `train_llm_fsdp.log` — the redirected stdout/stderr of the run, in this folder.

## Hardware support & evidence

Claims above were checked against upstream sources on **2026-08-19**:

- **FSDP2 is device-agnostic.** The `fully_shard` documentation
  ([pytorch/pytorch `docs/source/distributed.fsdp.fully_shard.md`](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.fsdp.fully_shard.md))
  describes DTensor dim-0 per-parameter sharding with hooks registered on the original
  module — nothing CUDA-specific in the user contract; the device comes from the mesh.
- **ROCm wheels.** `torch==2.11.0+rocm7.2` wheels (cp310–cp314) exist on
  [download.pytorch.org/whl/rocm7.2](https://download.pytorch.org/whl/rocm7.2/torch/).
  The `rocm6.4` index stops at torch 2.9.1, which is why the install line above pins the
  `rocm7.2` index.
- **XPU wheels.** `torch==2.11.0+xpu` wheels exist on
  [download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu/torch/).
- **DeepSpeed comparison.** DeepSpeed's ZeRO on ROCm/XPU requires vendor-patched builds
  and a working op-builder toolchain; FSDP2 needs neither — the portability argument for
  this folder.
- **Other hardware (upstream claims — not verified here):** Intel XPU (official
  `torch==2.11.0+xpu` wheels; Intel also documents FSDP for XPU) plus, by the
  device-agnostic `fully_shard`/DTensor contract, any accelerator with a PyTorch device
  and collective backend. Only NVIDIA and AMD have instructions in this repo; the XPU
  commands above are untested here.

## Notes

- **FSDP2, not FSDP1.** The current API is `torch.distributed.fsdp.fully_shard`, which
  converts each parameter into a `DTensor` sharded on dim-0 and registers pre/post
  forward-backward hooks on the *original* module — parameter FQNs are unchanged, and
  `type(model)` is unioned with `FSDPModule` in place. The legacy
  `FullyShardedDataParallel` wrapper (FSDP1, one flat `FlatParameter` per unit) is
  deliberately not used: per-parameter sharding is what makes interspersed
  frozen/trainable params (i.e. LoRA) work without `use_orig_params` gymnastics, and it
  gives communication-free sharded state dicts.
- **Who calls `fully_shard`.** This script does not call it directly; accelerate's FSDP
  plugin (`fsdp_version: 2`) applies it, bottom-up over the transformer blocks selected
  by `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`. Each `fully_shard` call is one
  communication group, so per-layer wrapping lets the next layer's all-gather overlap
  with the current layer's compute; wrapping only the root would collapse the run into
  two giant blocking collectives.
- **`reshard_after_forward` is the memory/speed dial.** `true` frees the unsharded params
  after forward and re-all-gathers them in backward (ZeRO-3-like, lowest memory); `false`
  keeps them resident (ZeRO-2-like, fewer collectives, higher peak memory). FSDP2 already
  prefetches backward all-gathers and runs reduce-scatters on a separate stream by
  default, so there is no `backward_prefetch` knob to tune anymore.
- **DeepSpeed-free by design.** Nothing here imports `deepspeed`, and no `CUDA_HOME` or
  op-builder JIT step is needed. The mapping if you are coming from
  `../deepspeed/`: ZeRO-3 → `reshard_after_forward: true`, ZeRO-2 →
  `reshard_after_forward: false`, ZeRO-0/DDP → don't use FSDP, CPU optimizer offload →
  `fsdp_offload_params: true`.
- **Chat-template masking.** Each row is tokenized once with `apply_chat_template`; the
  span of every assistant turn is located by re-rendering the conversation prefix with
  `add_generation_prompt=True` and diffing lengths. This assumes chat templates are
  prefix-stable (true for mainstream instruct models). Labels outside those spans are
  `-100`.
- **Checkpointing.** `fsdp_state_dict_type: FULL_STATE_DICT` makes the Trainer write one
  consolidated, directly loadable HF checkpoint. `SHARDED_STATE_DICT` is faster for very
  large models but produces per-rank `.distcp` shards that only load back into FSDP —
  consolidate those with `accelerate merge-weights <sharded_dir> <output>`.
- **Activation checkpointing, once.** Either `--gradient_checkpointing` (Trainer) or
  `fsdp_activation_checkpointing: true` (FSDP) — enabling both raises an error. The
  shipped configs keep the FSDP one off.
- **LoRA under FSDP2.** `--use_lora` wraps the model with PEFT *before* the Trainer hands
  it to accelerate, so the frozen base and the trainable adapters are sharded together.
  The saved artifact is then an adapter, not a full model. 4-bit QLoRA is **not**
  supported here (quantized params are not shardable) — use `../peft/` for that.
