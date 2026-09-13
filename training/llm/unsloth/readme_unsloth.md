# training/llm/unsloth — Unsloth LoRA/QLoRA SFT + GRPO trainer

## Overview & when to use

Standalone Unsloth trainer for Gemma-4-31B — supervised fine-tuning (SFT) by default, with an optional GRPO (RL) mode in the same script. Imports only third-party packages; no local project modules required. Use it when you want memory-lean LoRA/QLoRA adapter training (or full FT) on one node without DeepSpeed.

Files:
- `train_llm_unsloth.py` — the trainer (SFT + GRPO modes).
- `grpo_rewards.py` — swappable GRPO reward-function module (generic template).
- `data/OTel_LLM_sample_10.jsonl` — 10-row sample dataset (schema below).
- `requirements_unsloth.txt` — pinned environment.
- `vllm_unsloth_setup.md` — separate venv notes for the GRPO/vLLM tooling.

**Import-order note (load-bearing):** `from unsloth import FastModel` is deliberately the *first* import in `train_llm_unsloth.py` — Unsloth patches torch/transformers/trl at import time to install its Triton kernels and fast path. Do not reorder it below other imports; the fast path silently degrades or breaks.

> **Coverage:** LoRA/QLoRA SFT is verified on NVIDIA H100 80GB (CUDA 13.0) and on 1× and 8× AMD Instinct MI355X (gfx950, ROCm 7.2.4), with GRPO verified on MI355X. Unsloth's OSS build is DDP only (one full replica per GPU, no FSDP/ZeRO), so the model must fit on a single GPU. Multi-node is documented below but untested.

## Install

Python 3.12, in a venv:

```bash
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate
```

### NVIDIA (CUDA)

```bash
pip install -r requirements_unsloth.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # expect 2.11.0 13.0
```

- `torch==2.11.0` / `xformers==0.0.35` / `triton==3.6.0` are genuine CUDA-13 (`+cu130`) wheels, so the file installs **verbatim** on CUDA 13 — none of the ROCm steps below are needed. On a different CUDA, install torch/triton/xformers via Unsloth's official installer and keep the HF-stack pins from `requirements_unsloth.txt`.
- **Torch-downgrade check:** installing this file pins torch back to `2.11.0+cu130` if you already had a newer one. That is a downgrade, not a clobber — it stays on CUDA 13 — but re-verify with the one-liner above.
- `bitsandbytes==0.50.0`'s stock wheel works on CUDA out of the box, so `--load_in_4bit` (QLoRA) and `adamw_8bit` need no special build.
- Flash-Attention 2 is **not** required — Unsloth manages attention internally and uses SDPA for gemma-4. Do not install `flash-attn` for this model.

### AMD (ROCm)

Upstream Unsloth officially supports AMD (https://unsloth.ai/docs/basics/amd lists CDNA 4 / Instinct MI350 series / gfx950 as **Full** support, Linux only). Do **not** install the CUDA-13 pins from `requirements_unsloth.txt` verbatim — the torch/triton/xformers wheels there are CUDA builds. Use this sequence:

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

Notes on that sequence:

- **torch survives step 2** — the `+rocm7.2` local build satisfies unsloth's `torch` requirement, so pip does not replace it. Only `triton`/`xformers` need the step-3 eviction; without it `import triton` resolves to the CUDA build and Unsloth's Triton kernels break.
- **bitsandbytes 0.50.0 stock PyPI wheel is ROCm-ready** — it bundles `libbitsandbytes_rocm72.so` and auto-loads it on gfx950, so `--load_in_4bit` (QLoRA) and `adamw_8bit` work with the pinned version.
- **Attention runs on SDPA** — no xformers/FA2 on ROCm (`Xformers = None. FA2 = False` in the Unsloth banner). Training is correct; do not pip-install `flash-attn`.
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

### Smoke test — 8 GPUs, 1 node

Multi-GPU is plain DDP (RCCL/NCCL), one full replica per rank, and needs **no code, flag or
dependency change** versus the 1-GPU run — only the launcher. Three rules:

- **`export RUN_ID=...` is mandatory, not optional** — every rank reads it from the
  environment, and they must agree on the output directory.
- **Re-export the full device list after activating the venv**, and assert
  `torch.cuda.device_count() == N` before training. A venv `bin/activate` carrying a stale
  `export CUDA_VISIBLE_DEVICES=4` (or `HIP_VISIBLE_DEVICES`) from an earlier single-GPU
  session will pin every rank to one card and quietly produce a fake "N-GPU" result.
- **Ignore the `Num GPUs used = 1` banner line.** Unsloth prints it *per rank* — each
  process reporting its own device, not a fallback. The authoritative lines are the loader's
  `Num GPUs = N` and the trainer's `Data Parallel GPUs = N | Total batch size (b x a x N)`.
  Confirm residency with `rocm-smi` / `nvidia-smi` if in doubt.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # and HIP_VISIBLE_DEVICES too on ROCm
export RUN_ID=$(date -u +%Y%m%d_%H%M%S)       # shared by all ranks
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

## Hardware support

- **NVIDIA** — verified on H100 80GB (CUDA 13 wheels in `requirements_unsloth.txt`, installed verbatim).
- **AMD/ROCm** — verified on 1× and 8× AMD Instinct MI355X (gfx950, ROCm 7.2.4); 8-GPU LoRA SFT runs as plain DDP via `torchrun` with no code changes. Upstream lists Instinct MI350-series (gfx950) as fully supported on Linux (https://unsloth.ai/docs/basics/amd).
- Unsloth OSS is DDP-only (no FSDP/ZeRO) — one full replica per GPU regardless of node count.

## Platform notes

A healthy run prints the Unsloth banner, then decreasing `{'loss': ...}` lines with finite
`grad_norm`, and ends by saving the adapter + tokenizer to `.../final_model`:

```
==((====))==  Unsloth 2026.8.9: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    AMD Instinct MI355X. Num GPUs = 8. Max memory: 287.984 GB. Platform: Linux.
\        /    Data Parallel GPUs = 8 | Total batch size (2 x 1 x 8) = 16
{'loss': '0.3762', 'grad_norm': '0.3065', 'learning_rate': '0', 'epoch': '1'}
{'loss': '0.1976', 'grad_norm': '0.0453', 'learning_rate': '2.507e-06', 'epoch': '15'}
```

### AMD / ROCm quirks

- **GRPO needs `export UNSLOTH_GRPO_SEQ_PACKING=0`.** With Unsloth's GRPO sequence-packing enabled (the default), the first backward dies with `torch.utils.checkpoint.CheckpointError: Recomputed values ... different metadata` — the packed forward is not recompute-stable under Unsloth's gradient checkpointing on ROCm. Disabling seq-packing alone fixes it; `UNSLOTH_GRPO_PREFIX_GROUPER` can stay on, and disabling only the grouper does **not** help.
- **Import-time patching works exactly as on CUDA**, and Triton kernels compile via `triton-rocm` 3.6.0. A `unsloth_compiled_cache/` directory appears next to the script on first run; it is safe to delete.
- Attention is SDPA (no xformers/FA2 on ROCm) — training is correct; do not install `flash-attn`.

### Unsloth's multi-GPU guard — why 8-GPU works, and the caveat

The installed package still ships a refusal (`RuntimeError: Unsloth currently does not
support multi GPU setups`), but on this pin set the guard never lands and real DDP survives.
**This holds for this pin set only; it is not a supported guarantee.** Verify it yourself
before trusting a multi-GPU run:

```python
import inspect, trl
'does not support multi GPU' in inspect.getsource(trl.trainer.sft_trainer.SFTTrainer.train)   # -> False
```

If a future upgrade starts raising that RuntimeError, pin back to trl 0.24.0 +
unsloth 2026.8.9. GRPO is unaffected either way — its TRL trainer is not in Unsloth's patch
list at all — but GRPO's `batch_size × grad_acc_steps × world_size % num_generations == 0`
invariant does bind at world size 8.

Not retested at 8 GPUs: QLoRA `--load_in_4bit` and GRPO.

## Quantization (4-bit / 8-bit / 16-bit)

The base model is frozen and only the LoRA adapter trains, so quantizing the base cuts VRAM at little quality cost:

| Flag | Base precision | ~VRAM (31B, bs8, 4200 ctx) | Notes |
|---|---|---|---|
| `--load_in_4bit` | 4-bit NF4 (QLoRA) | ~22 GB/GPU | Default choice — best VRAM; quality ≈ 16-bit for LoRA. |
| `--load_in_8bit` | 8-bit int8 | ~36 GB/GPU | Higher fidelity, slower steps. |
| *(neither)* | bf16 | ~60+ GB/GPU | Full-precision base. |

- **QLoRA = quantized base + bf16 adapter.** The frozen base stays quantized; only the LoRA tensors are bf16.
- **Two ways to a 4-bit base:** a pre-quantized checkpoint (`unsloth/gemma-4-31b-it-unsloth-bnb-4bit`) or a 16-bit base + `--load_in_4bit` (bitsandbytes quantizes at load).
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
4. **On-device GRPO is slow** — bound experiments with `--max_steps`, keep `--max_completion_length` tight, and raise `--num_generations` only if `frac_reward_zero_std` is high.
5. **AMD/ROCm: GRPO requires `export UNSLOTH_GRPO_SEQ_PACKING=0`** — Unsloth's GRPO sequence-packing crashes the backward with a `torch.utils.checkpoint.CheckpointError` on ROCm (details under [AMD / ROCm quirks](#amd--rocm-quirks)).

### Health metrics to watch

- `reward` / `rewards/*/mean` should trend up — movement can take many steps (~300 is a common rule of thumb).
- `frac_reward_zero_std` near 1 means all rollouts in a group scored the same (no gradient) — raise `--num_generations`.
- `completions/clipped_ratio` high means rollouts hit the length cap without terminating — raise `--max_completion_length` or curb generation length.

## Notes & troubleshooting

- **Corporate/Azure proxies:** some environments set `HTTP_PROXY`/`HTTPS_PROXY` to a proxy that 403s huggingface.co. If model downloads fail with 403s, unset the proxy vars for the run (`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python train_llm_unsloth.py ...`) or export empty values in your shell profile.
- **Restricted environments** (no internet, or a shared mount where the Hub prefetch stalls): pre-download the model, pass its local snapshot dir as `--model_name`, and set `HF_HUB_OFFLINE=1`.
- **Design notes:** each DDP rank is pinned to `cuda:LOCAL_RANK` with an explicit `device_map` (auto placement would stack all ranks on `cuda:0` and OOM). Labels are baked per row during tokenization so `unmask` can switch masking policy per row. Gradient checkpointing is always on. `--loss_weight 1.0` uses Unsloth's fused CE (bs8 OK at 4200 ctx); `>1.0` uses a custom `compute_loss` that materializes logits — drop to bs4 at 4200 ctx to avoid OOM.
- **GRPO EOS fix:** the multimodal processor can report an EOS that is not the turn-end marker, so rollouts never stop; the script points the tokenizer at `--turn_end_token` before training.
- Launch with `torchrun`/`accelerate` for multi-GPU, never plain `python` (except 1-GPU or `--device_map balanced`).
