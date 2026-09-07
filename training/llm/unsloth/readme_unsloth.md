# training/llm/unsloth — Unsloth LoRA/QLoRA SFT + GRPO trainer

## Overview & when to use

Standalone Unsloth trainer for Gemma-4-31B — supervised fine-tuning (SFT) by default, with an optional GRPO (RL) mode in the same script. Imports only third-party packages; no local project modules required. Use it when you want fast, memory-lean LoRA/QLoRA adapter training (or full FT) on one node without DeepSpeed — Unsloth's patched kernels give roughly 2× speed and large VRAM savings over stock TRL.

Files:
- `train_llm_unsloth.py` — the trainer (SFT + GRPO modes).
- `grpo_rewards.py` — swappable GRPO reward-function module (generic template).
- `data/OTel_LLM_sample_10.jsonl` — 10-row sample dataset (schema below).
- `requirements_unsloth.txt` — pinned environment.
- `vllm_unsloth_setup.md` — separate venv notes for the GRPO/vLLM tooling.

**Import-order note (load-bearing):** `from unsloth import FastModel` is deliberately the *first* import in `train_llm_unsloth.py` — Unsloth patches torch/transformers/trl at import time to install its Triton kernels and fast path. Do not reorder it below other imports; the fast path silently degrades or breaks.

> **Tested topology:** 8×H100 80GB (CUDA) — validated single-GPU and single-node DDP. Also validated single-GPU on 1× AMD Instinct MI355X (ROCm 7.2) — see the MI355X section below. Unsloth's OSS build is DDP (one full replica per GPU, no FSDP/ZeRO), but with quantization that is often enough: a quantized replica fits one 80GB GPU for roughly LoRA training up to ~120B and full-parameter training up to ~50B. Real ceilings are lower with longer sequences, higher LoRA rank, an 8-bit base, or larger batches. Multi-node is documented below but has **not** been tested here.

## Install

Python 3.12, in a venv:

```bash
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate
```

### NVIDIA (CUDA)

```bash
pip install -r requirements_unsloth.txt
```

- `torch==2.11.0` / `xformers==0.0.35` / `triton==3.6.0` are CUDA-13 wheels (tested on H100). On a different CUDA, install torch/triton/xformers via Unsloth's official installer and keep the HF-stack pins from `requirements_unsloth.txt`.
- Flash-Attention 2 is **not** required — Unsloth auto-falls back to `xformers`.

### AMD (ROCm) — ✅ tested on MI355X, ROCm 7.2

Upstream Unsloth officially supports AMD (https://unsloth.ai/docs/basics/amd lists CDNA 4 / Instinct MI350 series / gfx950 as **Full** support, Linux only), and this folder has now been **validated on an AMD Instinct MI355X (gfx950, 288 GB, ROCm 7.2.4, Python 3.12)**. Do **not** install the CUDA-13 pins from `requirements_unsloth.txt` verbatim — the torch/triton/xformers wheels there are CUDA builds. The exact sequence that worked:

```bash
python3 -m venv .env_unsloth && source .env_unsloth/bin/activate
# 1) ROCm PyTorch FIRST (brings triton-rocm 3.6.0 as the `triton` module):
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
# 2) Unsloth via upstream's official AMD extra, plus this folder's HF-stack pins:
pip install "unsloth[amd]==2026.8.9" unsloth_zoo==2026.8.6 \
    transformers==5.5.0 trl==0.24.0 peft==0.20.0 datasets==4.3.0 accelerate==1.14.0 \
    bitsandbytes==0.50.0 tokenizers==0.22.2 huggingface_hub==1.27.0 safetensors==0.8.0 \
    sentencepiece==0.2.2 hf_transfer==0.1.9 hf-xet==1.6.0 python-dotenv==1.2.2 \
    numpy==2.3.5 protobuf==6.33.6 tqdm==4.70.0 PyYAML==6.0.3 filelock==3.32.2 packaging==26.3
# 3) REQUIRED cleanup: unsloth[amd] drags in the CUDA `triton` 3.7.1 (which shadows the
#    ROCm triton module) and the CUDA xformers 0.0.35 wheel. Evict both, restore ROCm triton:
pip uninstall -y triton xformers
pip install --force-reinstall --no-deps triton-rocm==3.6.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Notes from the MI355X validation:

- **torch survives step 2** — the `+rocm7.2` local build satisfies unsloth's `torch` requirement, so pip does not replace it. Only `triton`/`xformers` need the step-3 eviction; without it `import triton` resolves to the CUDA build and Unsloth's Triton kernels break.
- **bitsandbytes 0.50.0 stock PyPI wheel is ROCm-ready** — it bundles `libbitsandbytes_rocm72.so` and auto-loads it on gfx950, so `--load_in_4bit` (QLoRA) and `adamw_8bit` work with the pinned version. Upstream's "install a pre-release bnb" note only applies to bnb ≤ 0.49.2 (4-bit decode NaN bug).
- **Attention runs on SDPA** — no xformers/FA2 on ROCm (`Xformers = None. FA2 = False` in the Unsloth banner). Training is correct; do NOT pip-install `flash-attn`.
- Do **not** set `HSA_OVERRIDE_GFX_VERSION` on MI355X — that workaround is for MI300X (gfx942); gfx950 is natively supported.

### Verify

```bash
python -c "from unsloth import FastModel; import torch, trl, transformers, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

## Environment & secrets

Optional `dev.env` next to the script (loaded via `load_dotenv("dev.env")`) — only needed to pull gated models from the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Never commit `dev.env`. Optionally point the HF model cache somewhere with room before running:

```bash
export HF_HOME=/path/with/space/huggingface
```

## Data

Training data is JSONL, one `{"messages": [...]}` per line, with an optional per-row `"unmask": true|false` flag:

- `unmask: true` — loss on the **entire sequence** (prompt + response).
- `unmask: false` / absent — **completion-only** (prompt masked). Each such row must contain the response marker (`--response_marker`, default `<|turn>model\n`) or the run hard-fails.

The bundled sample `data/OTel_LLM_sample_10.jsonl` (the default `--train_file`) has this schema plus extra metadata columns (`flow`, `source_repo`, `source_id`, `source_spec_id`, `source_version`) — some mostly null. The loader tolerates and drops extras: the SFT path removes all original columns after tokenization, so any extra columns in your own data are harmless. Swap in your own data with `--train_file path/to/your.jsonl`.

GRPO mode needs a different schema — `prompt` / `answer` / `family` rows (see the GRPO section below).

## Run

`$BASE` below is `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (pre-quantized 4-bit; downloads on first run).

### Smoke test — 1 GPU

`--max_steps 5` proves the pipeline end-to-end and exits:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
python train_llm_unsloth.py \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 5 --learning_rate 2e-4 \
  --experiment_root experiments --output_subdir smoke_1gpu
```

### Smoke test — 8 GPUs, 1 node (tested topology)

```bash
RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
torchrun --nproc_per_node 8 --master_port 29590 train_llm_unsloth.py \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 5 --learning_rate 2e-4 \
  --experiment_root experiments --output_subdir smoke_8gpu_1node
```

> The 10-row sample is fine for a 1-GPU smoke test; for multi-GPU smoke tests use a dataset with at least `batch_size × world_size` rows so every rank gets a full batch.

### Full run

Drop `--max_steps` (epochs take over via `--num_train_epochs`) and point `--train_file` at your real dataset. Effective batch = `batch_size × grad_acc_steps × total_GPUs`.

**What "working" looks like:** all ranks load, then `{'loss': ...}` lines appear with a finite `grad_norm`. An occasional `grad_norm: nan` on a single step is a benign bf16-QLoRA logging artifact if loss keeps dropping.

### Multi-node (untested here)

Run the same torchrun command on every node with `--nnodes N --node_rank <i> --master_addr <node0-ip> --master_port <port>`, an identical `RUN_ID`, and `--experiment_root` on a shared filesystem. Unsloth composes with standard torchrun multi-node DDP — each GPU still holds a full replica, so multi-node adds throughput, not room for a bigger model. Fast interconnect (InfiniBand/RoCE) matters for the all-reduce at this model size.

## Arguments

Every flag, with defaults:

| Flag | Default | Meaning |
|---|---|---|
| `--train_mode` | `sft` | `sft` or `grpo` (RL). |
| `--model_name` | `unsloth/gemma-4-31B-it` | Hub id or local snapshot dir. |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Training JSONL. |
| `--output_dir` | None | Full output-dir override. |
| `--experiment_root` | `experiments` | Root for run outputs (`<root>/<RUN_ID>/<subdir>`). |
| `--output_subdir` | `gsma_sft_lora_unsloth` | Subdir under `experiment_root/<RUN_ID>`. |
| `--max_seq_length` | 3100 | Over-length rows are dropped, not truncated. |
| `--load_in_4bit` | off | On-the-fly 4-bit (QLoRA) or load a 4-bit ckpt. |
| `--load_in_8bit` | off | On-the-fly 8-bit; mutually exclusive with 4-bit. |
| `--full_finetuning` | off | Full FT instead of LoRA (heavy). |
| `--device_map` | None | `balanced` splits one replica across GPUs (single-process launch); unset = DDP. |
| `--lora_r` | 8 | LoRA rank. |
| `--lora_alpha` | 8 | LoRA alpha. |
| `--lora_dropout` | 0.0 | LoRA dropout. |
| `--lora_target_modules` | `all-linear` | `all-linear` or comma-separated module names. |
| `--batch_size` | 1 | Per-device batch size. |
| `--grad_acc_steps` | 8 | Gradient-accumulation steps. |
| `--num_train_epochs` | 3 | Epochs. |
| `--max_steps` | -1 | >0 = smoke test (overrides epochs). |
| `--learning_rate` | 2e-4 | LR (auto-drops to 5e-6 for GRPO if left at the SFT default). |
| `--lr_scheduler_type` | `cosine` | Scheduler. |
| `--weight_decay` | 0.01 | Weight decay. |
| `--warmup_ratio` | 0.03 | Warmup ratio. |
| `--optim` | `adamw_8bit` | Optimizer (8-bit optimizer states). |
| `--logging_steps` | 1 | Log interval. |
| `--save_strategy` | `epoch` | `no` / `epoch` / `steps`. |
| `--save_steps` | 250 | Checkpoint interval for `--save_strategy steps`. |
| `--seed` | 42 | Seed. |
| `--mask_prompt` / `--no-mask_prompt` | on | Per-row masking policy on/off (off = full-sequence loss for all rows). |
| `--response_marker` | `<\|turn>model\n` | Marker opening the assistant turn; completion-only rows mask up to and including it. |
| `--loss_weight` | 1.0 | Per-token loss multiplier for completion-only rows (>1 up-weights them; 1.0 = stock loss path). |
| `--report_to` | `none` | Trainer reporting backend. |
| `--eval_samples` | 1000 | Held-out eval split size. |
| `--test_mode` | off | Use only the first `--test_mode_count` rows. |
| `--test_mode_count` | 10000 | Row cap for `--test_mode`. |
| `--num_generations` | 8 | [GRPO] Rollouts per prompt; effective batch must be divisible by it. |
| `--max_prompt_length` | 1024 | [GRPO] Prompts longer than this are dropped. |
| `--max_completion_length` | None | [GRPO] Max tokens per rollout (default: `max_seq_length − max_prompt_length`). |
| `--grpo_temperature` | 1.0 | [GRPO] Rollout sampling temperature (>0 needed). |
| `--turn_end_token` | `<turn\|>` | [GRPO] End-of-turn token set as EOS so rollouts stop. |
| `--reward_mode` | `rule` | [GRPO] `rule` / `llm` / `hybrid`. |
| `--reward_module` | `grpo_rewards` | [GRPO] Module exposing `build_reward_funcs(reward_mode, judge_model)`. |
| `--judge_model` | `claude-sonnet-5` | [GRPO] LLM-judge model id (llm/hybrid only). |
| `--fast_inference` | off | [GRPO] Colocated vLLM rollouts (blocked for gemma-4; see limitations). |
| `--gpu_memory_utilization` | 0.9 | [GRPO] vLLM KV-cache VRAM fraction (with `--fast_inference`). |
| `--vllm_server` | off | [GRPO] Route rollouts to an external `trl vllm-serve` server. |
| `--vllm_server_host` | `127.0.0.1` | [GRPO] vLLM server host. |
| `--vllm_server_port` | 8000 | [GRPO] vLLM server port. |

## Output

Checkpoints land in `experiment_root/<RUN_ID>/<output_subdir>/` (or `--output_dir`); the final LoRA adapter + tokenizer are saved by rank 0 to `.../final_model/`. `RUN_ID` comes from the environment so all ranks share it — set it explicitly for multi-process runs.

## Hardware support & evidence

**Other hardware (upstream claims — not verified here):** upstream claims CPU, Apple (macOS training + MLX/GGUF inference), Intel GPUs, and Vulkan GGUF inference, alongside NVIDIA/AMD (Unsloth requirements docs).


- **NVIDIA** — validated in this folder on 8×H100 80GB (CUDA 13 wheels in `requirements_unsloth.txt`).
- **AMD/ROCm** — **validated in this folder on 1× and 8× AMD Instinct MI355X** (gfx950, ROCm 7.2.4); 8-GPU LoRA SFT runs as plain DDP via `torchrun` with no code changes. See the AMD install section above and the MI355X tested + 8-GPU sections below. Upstream lists Instinct MI350-series (gfx950) as fully supported on Linux (https://unsloth.ai/docs/basics/amd).
- Unsloth OSS is DDP-only (no FSDP/ZeRO) — one full replica per GPU regardless of node count.

## Platform notes — AMD MI355X (ROCm 7.2)

Validated single-GPU on 1× MI355X (gfx950, 288 GB VRAM), ROCm 7.2.4, Python 3.12.3, with the install sequence from the AMD (ROCm) section above (torch 2.11.0+rocm7.2, unsloth 2026.8.9 via `unsloth[amd]`, triton-rocm 3.6.0, bitsandbytes 0.50.0). Model: `google/gemma-4-E4B-it` (8B; Unsloth transparently remaps to its `unsloth/gemma-4-E4B-it` mirror) — same gemma-4 family/chat template as the folder's default 31B, sized for a fast smoke.

Smoke command (LoRA SFT, 5 steps — the folder's 1-GPU smoke with only the model swapped):

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 RUN_ID=$(date -u +%Y%m%d_%H%M%S) \
python train_llm_unsloth.py \
  --model_name google/gemma-4-E4B-it \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 5 --learning_rate 2e-4 \
  --experiment_root experiments --output_subdir smoke_1gpu
```

Expected output (bf16 LoRA run; the `--load_in_4bit` QLoRA rerun matched within noise, 3.618 → 2.386):

```
==((====))==  Unsloth 2026.8.9: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    AMD Instinct MI355X. Num GPUs = 1. Max memory: 287.984 GB. Platform: Linux.
{'loss': '3.601', 'grad_norm': '2.702', 'learning_rate': '0', 'epoch': '0.2'}
{'loss': '2.364', 'grad_norm': '2.91', 'learning_rate': '2.929e-05', 'epoch': '1'}
{'train_runtime': '26.73', ...} → Saved adapter + tokenizer to .../final_model
```

This path works on MI355X (LoRA SFT and QLoRA 4-bit out of the box; GRPO works with one env var — see below).

Status by tier, single MI355X:

| Tier | Status | Notes |
|---|---|---|
| LoRA SFT (bf16 base) | ✅ pass | 5/5 steps, finite loss/grad_norm, adapter saved. |
| QLoRA (`--load_in_4bit`) | ✅ pass | bitsandbytes 0.50.0 ROCm binaries; losses match bf16 run. |
| GRPO (on-device rollouts) | ✅ pass with `UNSLOTH_GRPO_SEQ_PACKING=0` | Default crashes — see quirks. |
| DDP / multi-GPU | ✅ pass on 8× MI355X | Real DDP world_size=8 via `torchrun`, VRAM on all 8 cards; see the 8-GPU section below. |

MI355X quirks (in addition to the install notes above):

- **GRPO needs `export UNSLOTH_GRPO_SEQ_PACKING=0`.** With Unsloth's GRPO sequence-packing enabled (the default), the first backward dies with `torch.utils.checkpoint.CheckpointError: Recomputed values ... different metadata` (saved `[1, 8, 112, 112]` vs recomputed `[2560, 1024]`) — the packed forward is not recompute-stable under Unsloth's gradient checkpointing on ROCm. Disabling seq-packing alone fixes it (`UNSLOTH_GRPO_PREFIX_GROUPER` can stay on; disabling only the grouper does NOT help). Verified end-to-end after the fix: rollouts → `reward_verifiable` → advantages → LoRA update → adapter saved, exit 0.
- **What Unsloth patched/compiled on ROCm:** import-time patching works exactly as on CUDA ("Fast Gemma4 patching"); Triton kernels compile via `triton-rocm` 3.6.0; a `unsloth_compiled_cache/` directory appears next to the script on first run (safe to delete).
- First GRPO step is slow (~30 s) while Triton autotunes/compiles the generation path; subsequent steps are fast.

### 8-GPU run (8× MI355X, ROCm 7.2.4)

**This path works as documented.** LoRA SFT scales to all 8 MI355X cards as plain DDP (one full replica per GPU) with **no code, flag, or dependency change** vs the 1-GPU run — only the launcher differs (`torchrun` instead of `python`). Despite Unsloth OSS's reputation for gating multi-GPU, the installed build (unsloth 2026.8.9 / unsloth_zoo 2026.8.6) did **not** refuse, and it did **not** silently fall back to one GPU: all 8 cards held ~24 GB of VRAM under the job's own PIDs for the whole run.

Launch (world size 8; run under a machine-wide `flock` so the job owns all 8 GPUs on a shared box):

```bash
cd training/llm/unsloth && source .env_unsloth/bin/activate
# MUST override if bin/activate carries a stale 1-GPU pin from an earlier session
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/path/to/hf_cache
export RUN_ID=$(date -u +%Y%m%d_%H%M%S)   # shared by all ranks

torchrun --nproc_per_node=8 --master_port 29680 train_llm_unsloth.py \
  --model_name google/gemma-4-E4B-it \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 15 --learning_rate 2e-4 \
  --save_strategy no \
  --experiment_root experiments --output_subdir sft_8gpu
```

**Parallelism:** DDP (RCCL backend), world size 8, one full replica per rank — no FSDP/ZeRO, no sharding (Unsloth OSS is DDP-only). Batch geometry `2 × 1 × 8 = 16`. Each rank is pinned by the script to `cuda:LOCAL_RANK` via `device_map={"": f"cuda:{local_rank}"}`; 73.4 M LoRA params trainable of 8.07 B.

What a healthy run looks like (15/15 steps, exit 0):

```
ASSERT device_count = 8
   \\   /|    AMD Instinct MI355X. Num GPUs = 8. Max memory: 287.984 GB. Platform: Linux.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 9 | Num Epochs = 15 | Total steps = 15
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 1
\        /    Data Parallel GPUs = 8 | Total batch size (2 x 1 x 8) = 16
{'loss': '0.3762', 'grad_norm': '0.3065', 'learning_rate': '0', 'epoch': '1'}
{'loss': '0.1976', 'grad_norm': '0.0453', 'learning_rate': '2.507e-06', 'epoch': '15'}
{'train_runtime': '28.91', 'train_samples_per_second': '8.302', 'train_loss': '0.2421', 'epoch': '15'}
=== torchrun exit code: 0 ===
```

GPU residency check — sample `rocm-smi` **in-band** (from inside the training script, while the run is live) and cross-check against the job's own PIDs so a co-tenant's job cannot be mistaken for yours:

```
--- pgrep -f train_llm_unsloth.py ---
<pid> <pid> ... <pid>                             # 1 torchrun + 8 ranks

--- rocm-smi --showmeminfo vram --csv (card, total_B, used_B) ---
card0,309220868096,32087703552     card4,309220868096,24091340800
card1,309220868096,24093143040     card5,309220868096,23843500032
card2,309220868096,24000659456     card6,309220868096,23959506944
card3,309220868096,24466399232     card7,309220868096,24141996032

--- rocm-smi --showpids ---
PID     PROCESS NAME  GPU(s)  VRAM USED
<pid>   python3       1       25879388160
<pid>   python3       2       24716886016
<pid>   python3       2       25054515200
<pid>   python3       2       25543204864
<pid>   python3       2       24333090816
<pid>   python3       2       24582651904
<pid>   python3       2       25769684992
<pid>   pt_elastic    0       0
```

All 8 cards carry ~24 GB (card0 ~32 GB — rank 0 also holds the torchrun/NCCL overhead), and every VRAM-holding PID should be one of the job's own 8 ranks. That is a genuine 8-way run, not a rank-0-only fallback.

**What differed from the 1-GPU run:**

- **Launcher only** — `torchrun --nproc_per_node=8 --master_port 29680` instead of `python`. Same model, same dataset, same LoRA/optimizer flags, same `requirements_unsloth.txt` (no new package or pin needed).
- **`export RUN_ID=...` is mandatory**, not optional: all ranks must agree on the output dir (the script reads `RUN_ID` from the environment).
- **A venv `bin/activate` that carries a stale `export HIP_VISIBLE_DEVICES=4` / `CUDA_VISIBLE_DEVICES=4`** (appended during an earlier 1-GPU session) is a real trap. Sourcing it and launching would pin every rank to a single card and quietly produce a fake "8-GPU" result. Always re-export the full device list after activating, and assert `torch.cuda.device_count() == 8` before training.
- `--save_strategy no` for the smoke run (the script still writes the final adapter from rank 0 — ~311 MB — regardless; there is no `load_best_model_at_end`, so nothing else is forced).
- Effective batch is `world_size ×` larger, so with the tiny 9-row sample file one step ≈ one epoch; loss values are not comparable step-for-step with the 1-GPU smoke.

**Reading the banner (important, this is the trap):** Unsloth prints `Num GPUs used = 1` **per rank** — that is each process reporting *its own* device, not a fallback. The authoritative lines are the loader's `Num GPUs = 8` and the trainer's `Data Parallel GPUs = 8 | Total batch size (2 x 1 x 8) = 16`. Do not read `Num GPUs used = 1` as a single-GPU run; confirm with `rocm-smi` instead.

**About Unsloth's multi-GPU guard (why this could have failed):** the installed package *does* still ship the block — `unsloth/tokenizer_utils.py` (~line 1703) injects into `SFTTrainer.train`, for hosts where `nvidia-smi` is absent (i.e. every ROCm box), `if torch.cuda.device_count() > 1: raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')`, and `unsloth/_gpu_init.py:120` still reads "Multi-GPU is not yet supported (beta available on request)". On this stack that guard **never lands**: the patcher rebuilds `train` from `getsource(trl.trainer.sft_trainer.SFTTrainer.train)`, and with **trl 0.24.0** that patch path bails out, so the runtime `SFTTrainer.train` contains no such check (verified by inspecting the patched source: `'does not support multi GPU' in inspect.getsource(...)` → `False`). Relatedly, `unsloth/models/_utils.py:2729` only forces `DistributedType.NO` when `DEVICE_COUNT == 1 and WORLD_SIZE <= 1`, so `WORLD_SIZE=8` leaves real DDP intact. **Caveat: this is version-luck, not a supported guarantee** — a trl/unsloth upgrade that restores the patch could re-enable the refusal. If a future upgrade starts raising that RuntimeError, pin back to trl 0.24.0 + unsloth 2026.8.9, and re-run the one-liner check above before trusting a multi-GPU run.

Not retested at 8 GPUs: QLoRA `--load_in_4bit` and GRPO. GRPO's TRL trainer is not in Unsloth's patch list at all, so the guard is a non-issue there, but its `batch_size × grad_acc_steps × world_size % num_generations == 0` invariant does bind at world size 8.

## Platform notes — NVIDIA H100 80GB (CUDA 13.0)

Validated **single-GPU** on 1× NVIDIA H100 80GB HBM3 (Hopper cc 9.0, driver 580.173.02, **CUDA 13.0**, Python 3.12.3) on a shared 8-GPU node (pinned to physical GPU 4 via `CUDA_VISIBLE_DEVICES=4`; GPUs 0–3 were a co-tenant job and untouched). Model: **`unsloth/gemma-4-31b-it-unsloth-bnb-4bit`** — the folder's pre-quantized 4-bit default (QLoRA), which fits comfortably in 80 GB (~24 GB peak). This mirrors the folder's documented 1-GPU smoke exactly, run in 4-bit.

### NVIDIA install that worked (CUDA 13)

The pinned `requirements_unsloth.txt` installs **verbatim** on CUDA 13 — the `torch==2.11.0` / `triton==3.6.0` / `xformers==0.0.35` pins ARE genuine CUDA-13 wheels (confirmed `+cu130`), so no Unsloth-installer detour is needed here:

```bash
python3 -m venv .env_unsloth && source .env_unsloth/bin/activate
pip install torch numpy                       # baseline: torch 2.13.0+cu130 (sanity), CUDA avail
pip install -r requirements_unsloth.txt        # pulls torch 2.11.0+cu130, triton 3.6.0, xformers 0.0.35
```

- **Torch-clobber check (do this):** `pip install unsloth==2026.8.9 …` (or `-r requirements_unsloth.txt`) **downgraded torch 2.13.0+cu130 → 2.11.0+cu130** to honor the pin. Crucially it stayed **`+cu130`** (CUDA 13), NOT an older-CUDA or CPU/ROCm build — so no recovery was needed. Re-verify after install: `python -c "import torch;print(torch.__version__, torch.version.cuda)"` → `2.11.0 13.0`.
- **bitsandbytes 0.50.0 stock wheel works on CUDA out of the box** — `Linear4bit` forward is finite on H100; `--load_in_4bit` (QLoRA) and `adamw_8bit` work with no special build (QLoRA is easier on NVIDIA than ROCm).
- **flash-attn: not installed, not needed.** Unsloth manages attention internally. For this **gemma-4 4-bit** path its runtime banner reports `Xformers = None. FA2 = False` (i.e. it uses **SDPA** internally for gemma-4 even though `xformers==0.0.35` is importable) — training is correct. Installing `flash-attn` is unnecessary for this model; leave it out.
- **tf32:** `torch.backends.cuda.matmul.allow_tf32` reads `False` by default here; the trainer runs **bf16** (`bf16=True` in `SFTConfig`, banner `Bfloat16 = TRUE`), so tf32 is not on the matmul path for training anyway.

### Verify

```bash
CUDA_VISIBLE_DEVICES=4 python -c "from unsloth import FastModel; import torch, trl, transformers, datasets; print('imports OK', torch.cuda.device_count(), 'GPUs')"
# -> imports OK 1 GPUs   (transformers 5.5.0, trl 0.24.0)
```

### Smoke command (LoRA/QLoRA SFT — the folder's 1-GPU smoke, 4-bit, steps bumped to 15 for a visible loss curve)

```bash
CUDA_VISIBLE_DEVICES=4 RUN_ID=$(date -u +%Y%m%d_%H%M%S) HF_HOME=/path/to/hf_cache \
python train_llm_unsloth.py \
  --model_name unsloth/gemma-4-31b-it-unsloth-bnb-4bit \
  --load_in_4bit \
  --train_file data/OTel_LLM_sample_10.jsonl \
  --max_seq_length 4200 --mask_prompt \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear --optim adamw_8bit \
  --batch_size 2 --grad_acc_steps 1 --max_steps 15 --learning_rate 2e-4 \
  --save_strategy no \
  --experiment_root experiments --output_subdir smoke_1gpu
```

**Step count:** 9 train rows (10-row sample, 1 held out for eval), `batch_size 2 × grad_acc 1 × world 1 = 2` → `Total steps = 15` over 3 epochs. Non-trivial (15 real optimizer steps).

### Expected output (QLoRA 4-bit, 15/15 steps, exit 0)

```
==((====))==  Unsloth 2026.8.9: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA H100 80GB HBM3. Num GPUs = 1. Max memory: 79.179 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 9.0. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
   \\   /|    Num examples = 9 | Num Epochs = 3 | Total steps = 15
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 1 x 1) = 2
 "-____-"     Trainable parameters = 244,858,880 of 31,517,945,392 (0.78% trained)
{'loss': '9.091', 'grad_norm': '42.03', 'learning_rate': '0',        'epoch': '0.2'}   # step 1
{'loss': '3.168', 'grad_norm': '5.277', 'learning_rate': '0.0001975', 'epoch': '0.6'}  # step 3
{'loss': '1.063', 'grad_norm': '1.555', 'learning_rate': '2.507e-06', 'epoch': '3'}    # step 15
{'train_runtime': '63.19', 'train_samples_per_second': '0.475', 'train_loss': '2.715', 'epoch': '3'}
[unsloth-sft] INFO: Saved adapter + tokenizer to .../qlora_1gpu/final_model   # adapter_model.safetensors = 935 MB
```

Loss falls **9.091 → 1.063** (finite `grad_norm` throughout; the transient bumps at steps 5/12 are the benign bf16-QLoRA logging artifact the README already notes — the trend is clearly down).

GPU residency check — sample `nvidia-smi` VRAM-by-PID **in-band** (a background loop inside the run script, live during training), filtered to physical **GPU 4's UUID** and **the job's own PID** so a co-tenant on GPUs 0–3 cannot be mistaken for yours:

```
# resolve your training PID and physical GPU 4's UUID first
--- nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid (grep the GPU-4 UUID & your PID) ---
<pid>, 18646 MiB, <gpu-uuid>   # during load
<pid>, 23819 MiB, <gpu-uuid>   # peak, mid-training
--- nvidia-smi --id=4 --query-gpu=memory.used,utilization.gpu ---
4, 23819 MiB, 12 %      # GPUs 0-3 (co-tenant) never touched
```

The only VRAM-holding PID on GPU 4 should be your own training process — a genuine GPU-4 run, ~24 GB peak (well under 80 GB).

**This path works as documented on H100.** QLoRA (4-bit) LoRA SFT runs on H100/CUDA-13 out of the box with the pinned `requirements_unsloth.txt` verbatim — no code change, no torch recovery, no flash-attn build. Single-GPU validated.

Status by tier, single H100:

| Tier | Status | Notes |
|---|---|---|
| QLoRA (`--load_in_4bit`, pre-quant 4-bit base) | ✅ pass | 15/15 steps, loss 9.09→1.06, finite grad_norm, 935 MB adapter saved, ~24 GB peak. |
| LoRA SFT (bf16 base) | ⚪ not run | bf16 31B base ≈ 60+ GB — fits 80 GB but tight; QLoRA is the folder's default and was the smoke. |
| GRPO | ⚪ not run | On-device path expected to work as on MI355X; on CUDA the ROCm `UNSLOTH_GRPO_SEQ_PACKING=0` workaround should NOT be needed (that bug was ROCm-specific). |
| DDP / multi-GPU (2, then 8) | ⚪ **not run** | Node was shared (GPUs 0–3 busy). See below. |

**What differed from the MI355X recipe:**

- **Install is simpler on CUDA** — `pip install -r requirements_unsloth.txt` **verbatim** works; the whole ROCm 3-step dance (ROCm-torch-first, `unsloth[amd]`, then evicting the CUDA triton/xformers and force-reinstalling `triton-rocm`) is **not** needed. The CUDA-13 pins in the file are exactly what unsloth wants.
- **torch is downgraded, not clobbered** — unsloth pins `torch 2.11.0+cu130` (still CUDA 13); on MI355X the `+rocm7.2` build survived untouched. Either way, re-verify `torch.version.cuda` after install.
- **Attention:** SDPA on both (Unsloth reports `Xformers = None. FA2 = False` for gemma-4 4-bit on H100 too) — so, like ROCm, do **not** bother installing `flash-attn` for this model.
- **VRAM ceiling is 80 GB vs 288 GB on MI355X** — QLoRA 31B (~24 GB peak) is comfortable; bf16 full-precision 31B base (~60+ GB) would be tight but should fit. No OOM hit in this smoke.
- **Device flags:** plain `CUDA_VISIBLE_DEVICES=4`; no `HIP_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`.

**Multi-GPU on H100 — NOT run here:** the node's GPUs 0–3 were a co-tenant job, so only 1 free GPU was used. To run multi-GPU on H100, use the existing 8-GPU recipe (the "8-GPU run" section above) with `torchrun --nproc_per_node=N`, an explicit shared `RUN_ID`, and `CUDA_VISIBLE_DEVICES` listing the free cards. On CUDA the multi-GPU guard analysis still applies (trl 0.24.0 makes the refusal path bail out), but here it is even less likely to bite since `nvidia-smi` *is* present. This was not exercised on H100 — verify `torch.cuda.device_count()` matches your device list and confirm per-GPU residency with `nvidia-smi` before trusting a multi-GPU H100 run.

## Quantization (4-bit / 8-bit / 16-bit)

The base model is frozen and only the LoRA adapter trains, so quantizing the base cuts VRAM at little quality cost:

| Flag | Base precision | ~VRAM (31B, bs8, 4200 ctx) | Notes |
|---|---|---|---|
| `--load_in_4bit` | 4-bit NF4 (QLoRA) | ~22 GB/GPU | Default choice — best VRAM; quality ≈ 16-bit for LoRA. |
| `--load_in_8bit` | 8-bit int8 | ~36 GB/GPU | Higher fidelity, ~50% slower steps. |
| *(neither)* | bf16 | ~60+ GB/GPU | Full-precision base. |

- **QLoRA = quantized base + bf16 adapter.** The frozen base stays quantized; only the LoRA tensors are bf16.
- **Two ways to a 4-bit base:** a pre-quantized checkpoint (`unsloth/gemma-4-31b-it-unsloth-bnb-4bit` — loads ~3× faster) or a 16-bit base + `--load_in_4bit` (bitsandbytes quantizes at load).
- `adamw_8bit` quantizes *optimizer states*, independent of base dtype.
- Compressed-tensors checkpoints (`*-qat-w4a16*`, `*-FP8-dynamic`) are inference-only — use bitsandbytes NF4/int8 for training.
- `--full_finetuning` needs a 16-bit base and no quantization; normally leave off.

## GRPO mode — how it works, limitations

`--train_mode grpo` runs GRPO (RL) with the same launch shape as SFT. GRPO has no token-level labels — each step samples `--num_generations` completions per prompt from the current policy, scores them with reward functions, computes group-relative advantages, and updates the LoRA. Data must be `prompt` / `answer` / `family` rows:

| field | meaning |
|---|---|
| `prompt` | chat messages **without** the assistant turn |
| `answer` | gold answer the reward checks (`""` for open-ended) |
| `family` | tag your reward function can branch on |

### Reward modules (porting guide)

Rewards come from a swappable module (`--reward_module`, default `grpo_rewards`) exposing `build_reward_funcs(reward_mode, judge_model)`. TRL calls each reward function as `reward_fn(prompts, completions, **columns) -> list[float]` — `prompts`/`completions` are aligned lists, and every extra dataset column is forwarded by keyword as an aligned list (an `answer` column arrives as an `answer` kwarg). Conversational datasets hand completions back as message-dict lists, so normalize to text first (see `_text` in `grpo_rewards.py`).

To port to your own data you need:
1. **A dataset** with the columns your reward reads — verifiable (`prompt` + gold `answer` → rule reward) or open-ended (`prompt` only → LLM-judge reward).
2. **A reward function** with the signature above returning one float per completion. Two templates ship in `grpo_rewards.py`: `reward_verifiable` (normalized exact-match against `answer`) and `make_llm_judge_reward` (scores open-ended answers 0–10 with an external LLM via the OpenAI SDK; config from `JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY` env vars, falling back to `OPENAI_API_KEY`).

Register functions in `build_reward_funcs()` or point `--reward_module` at your own module. Keep the signature, and **never raise inside a reward** — a raised error kills the whole GRPO step; catch and return 0.0.

### What works

On-device GRPO (no vLLM) is the reliable path — verified end-to-end (rollouts → rewards → advantages → LoRA update → adapter saved). Launch like SFT plus `--train_mode grpo` and a GRPO dataset. The batching invariant `batch_size × grad_acc_steps × world_size % num_generations == 0` is enforced (hard exit otherwise).

### Limitations (verified on unsloth 2026.8.9 + vllm 0.26.0)

1. **Colocated `--fast_inference` does not work for gemma-4** — Unsloth's `VLLM_SUPPORTED_VLM` allowlist excludes the multimodal `gemma4` arch (RuntimeError at load; the text-only escape hatch crashes on the bnb-4bit config). vLLM itself supports the arch; it is Unsloth's integration that lacks it. The script fails fast with an actionable message.
2. **`--vllm_server` connects but is not viable for a quantized base** — TRL's PEFT weight-sync merges and streams the entire model every step; on a 4-bit base each merge is a slow dequantize→add→re-quantize that also OOMs the server GPU. Use the on-device path.
3. **GRPO needs a separate venv from SFT/eval** — GRPO-with-vLLM tooling wants `vllm 0.26` + `trl 0.29` + `transformers 5.5.4`, which conflicts with the main venv's pins (see `vllm_unsloth_setup.md`). Do not upgrade the main venv; transformers ≥5.15 breaks the gemma-4 load (per-layer `head_dim` regression).
4. **On-device GRPO is slow** — generation and the optimizer step run sequentially per rank. Bound experiments with `--max_steps`, keep `--max_completion_length` tight, and raise `--num_generations` only if `frac_reward_zero_std` is high.
5. **AMD/ROCm: GRPO requires `export UNSLOTH_GRPO_SEQ_PACKING=0`** — Unsloth's GRPO sequence-packing crashes the backward with a `torch.utils.checkpoint.CheckpointError` on ROCm (verified on MI355X/gfx950; details in the MI355X section).

### Health metrics to watch

- `reward` / `rewards/*/mean` should trend up — movement can take many steps (~300 is a common rule of thumb).
- `frac_reward_zero_std` near 1 means all rollouts in a group scored the same (no gradient) — raise `--num_generations`.
- `completions/clipped_ratio` high means rollouts hit the length cap without terminating — raise `--max_completion_length` or curb generation length.

## Notes & troubleshooting

- **Corporate/Azure proxies:** some environments set `HTTP_PROXY`/`HTTPS_PROXY` to a proxy that 403s huggingface.co. If model downloads fail with 403s, unset the proxy vars for the run (`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python train_llm_unsloth.py ...`) or export empty values in your shell profile.
- **Restricted environments** (no internet, or a shared mount where the Hub prefetch stalls): pre-download the model, pass its local snapshot dir as `--model_name`, and set `HF_HUB_OFFLINE=1`.
- **Design notes:** each DDP rank is pinned to `cuda:LOCAL_RANK` with an explicit `device_map` (auto placement would stack all ranks on `cuda:0` and OOM). Labels are baked per row during tokenization so `unmask` can switch masking policy per row. Gemma-4 loads as a multimodal *processor* whose `__call__` is patched to `(images, text, videos)` — the script tokenizes via the underlying fast tokenizer with `text=` as a keyword and renders with `apply_chat_template`. Gradient checkpointing is always on (Unsloth's variant). Loss weighting off (1.0) uses Unsloth's fused CE (bs8 OK at 4200 ctx); on (>1.0) uses a custom `compute_loss` that materializes logits — use bs4 at 4200 ctx to avoid OOM.
- **GRPO EOS fix:** the multimodal processor can report an EOS that is not the turn-end marker, so rollouts never stop; the script points the tokenizer at `--turn_end_token` before training.
- Launch with `torchrun`/`accelerate` for multi-GPU, never plain `python` (except 1-GPU or `--device_map balanced`).
