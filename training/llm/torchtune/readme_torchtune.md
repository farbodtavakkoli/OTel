# Setup & usage — `train_llm_torchtune.py`

## Overview & when to use

Continued pre-training (domain-adaptive next-token prediction) with
[torchtune](https://github.com/pytorch/torchtune). This is the **CPT example**
for this repo: unstructured text, no chat template, loss on every token.

> **Status: legacy — upstream development wound down in 2025.** The torchtune README now
> carries a "no longer actively maintained" banner (see
> [The future of torchtune](https://github.com/pytorch/torchtune/issues/2883)). This folder
> targets the last stable API (`torchtune==0.6.1`) so you can still run a CPT job. Prefer
> `training/llm/torchtitan` or `training/llm/megatron` for anything you intend to keep.

Files in this folder:
- `train_llm_torchtune.py` — flattens the shipped sample, then execs `tune run`.
- `prepare_data_torchtune.py` — chat JSONL (`messages`) → `{text: ...}` JSONL.
- `config_cpt_lora.yaml` — LoRA CPT config using `text_completion_dataset`.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample (see Data below).
- `requirements_torchtune.txt` — pinned 0.6.x environment.
- `readme_torchtune.md` — this file.

> **Tested topology:** **AMD MI355X (gfx950) / ROCm 7.2.4 — tested August 2026, works.**
> 1× GPU: `lora_finetune_single_device`, Qwen2.5-0.5B LoRA CPT, 5 steps, finite loss.
> **8× GPU: also tested and works** — `lora_finetune_distributed` (FSDP2) with the new
> `config_cpt_lora_8gpu.yaml`, world size 8, 80 steps, all 8 GPUs busy, per-GPU VRAM
> lower than the same-geometry 1-GPU run. See [§7a](#7a-mi355x-rocm-72--tested) and
> [§7b the 8-GPU run](#8-gpu-run-8x-mi355x-rocm-724--tested-august-2026).
>
> **NVIDIA H100 80GB / CUDA 13.0 — also tested August 2026, works with changes.**
> 1× GPU (physical GPU 7): `lora_finetune_single_device`, Qwen2.5-1.5B LoRA CPT, `torch
> 2.13.0+cu130` + `torchtune 0.6.1` + **`torchao==0.10.0`**, 27 steps, per-epoch loss
> decreasing, adapters saved, 8.9 GB peak by-PID on GPU 7. See [§7c](#7c-h100-cuda-130--tested-august-2026).
> Multi-GPU on H100 is **deferred** (GPUs 0–3 were a production job).
> The shipped **Qwen2.5-7B default is still untested** (we test the 0.5B/1.5B of the same family).
> The data converter (`prepare_data_torchtune.py`) is stdlib-only and produces 10 rows.

## 1. Install

### NVIDIA (CUDA)

```bash
cd training/llm/torchtune
python3.12 -m venv .venv && source .venv/bin/activate

# torch FIRST: on CUDA 13 hosts the default stable wheel is native cu130 — no --index-url needed
pip install torch numpy                 # -> torch 2.13.0+cu130 (see H100 §7c)
# then the rest of the pins; torchtune 0.6.1 imports torchao at import time and needs the
# 0.6-era torchao API, so pin torchao==0.10.0 (0.18.x renamed torchao.dtypes.nf4tensor and breaks the import)
pip install "torchtune==0.6.1" torchvision "python-dotenv>=1.0.1"
pip install "torchao==0.10.0" --no-deps  # --no-deps so it can't drag torch off cu130

# one-time weight download (needs HF_TOKEN for gated repos)
tune download Qwen/Qwen2.5-7B --output-dir ./assets/Qwen2.5-7B
```

Verify (and re-check torch after installing torchtune — pip can silently clobber it):

```bash
python -c "import torchtune; print(torchtune.__version__)"
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 / 13.0
```

Edit `checkpointer.checkpoint_files` in `config_cpt_lora.yaml` if the downloaded
shard names differ (Qwen 7B is usually four `model-0000N-of-00004.safetensors`).

### AMD (ROCm)

There is still **no official AMD/ROCm support statement** from torchtune upstream (no ROCm
CI, benchmark, or vendor image — and since the project is unmaintained, none is coming).
**But "no support statement" turned out not to mean "does not work."** We tested it:
`torchtune==0.6.1` installs and trains fine on top of a **stable** ROCm torch wheel. It is
pure Python on top of torch, so it neither builds nor loads any CUDA kernels of its own.

Two corrections to what this folder used to claim:

- You do **not** need a torch *nightly*. The stable `torch 2.11.0+rocm7.2` wheel satisfies
  torchtune's `torch>=2.5` dependency.
- You do **not** need `--no-deps`. `pip install torchtune==0.6.1` resolves cleanly against
  the already-installed ROCm torch; pip does not try to pull a CUDA torch over it.

```bash
cd training/llm/torchtune
python3.12 -m venv .env_train_llm_torchtune && source .env_train_llm_torchtune/bin/activate

# 1) ROCm torch FIRST, from the ROCm index
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision

# 2) then torchtune — normal resolution, no --no-deps
pip install torchtune==0.6.1 python-dotenv

# 3) a small base model (7B default is overkill for a smoke test)
tune download Qwen/Qwen2.5-0.5B --output-dir ./assets/Qwen2.5-0.5B
```

Do **not** `pip install flash-attn` on ROCm — torchtune uses torch SDPA / flex attention,
which is the supported ROCm path and works as-is.

`training/llm/torchtitan` (AMD supports torchtitan via Primus) and `training/llm/megatron`
(`rocm/megatron-lm` images) remain the better long-term AMD targets, since this project is
unmaintained — but that is a maintenance argument, not a "does not run on AMD" one.

## 2. Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_torchtune.py` loads it with `load_dotenv("dev.env")`; the token is needed for
gated weight downloads. `dev.env` is git-ignored at the repo root. Never commit a token.

## 3. Data

### The shipped sample

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

`source_spec_id` / `source_version` are null in most (not all) rows.

### Conversion

torchtune's CPT path is `torchtune.datasets.text_completion_dataset`. It wants
one unstructured string per row:

```json
{"text": "Full document body, no chat structure, no special tokens."}
```

`prepare_data_torchtune.py` concatenates every turn's `content` into `text` and
drops the extras, writing `data/otel_cpt.jsonl` (10 rows from the shipped sample).
The launcher runs that conversion unless you pass `--skip-prepare`. Rows that already
have a `text` field are copied through unchanged.

```bash
python3 prepare_data_torchtune.py          # data/OTel_LLM_sample_10.jsonl -> data/otel_cpt.jsonl
```

The embedding and reranker samples elsewhere in this repo (`anchor` / `positive` /
`negative_*`) are the wrong modality — do not point this trainer at them.

## 4. Run

Run from inside `training/llm/torchtune/` (the YAML's `./assets` and `./data` paths and
`dev.env` resolve against the current directory).

```bash
# inspect first
python3 train_llm_torchtune.py --dry-run

# smoke: flatten the 10-row shipped sample and start LoRA CPT on 8 GPUs
nohup python3 train_llm_torchtune.py --nproc-per-node 8 \
  > train_llm_torchtune.log 2>&1 &

tail -f train_llm_torchtune.log
```

Single GPU:

```bash
python3 train_llm_torchtune.py \
  --recipe lora_finetune_single_device \
  --nproc-per-node 1
```

Your own corpus (already `{text}` JSONL):

```bash
python3 train_llm_torchtune.py --skip-prepare --flat-file /data/domain.jsonl
```

## 5. Arguments

### `train_llm_torchtune.py`

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config_cpt_lora.yaml` | torchtune YAML |
| `--recipe` | `lora_finetune_distributed` | `tune run` recipe; use `lora_finetune_single_device` on one GPU |
| `--nproc-per-node` | `8` | GPUs on this node |
| `--input` | `data/OTel_LLM_sample_10.jsonl` | chat JSONL to flatten |
| `--flat-file` | `data/otel_cpt.jsonl` | `{text}` JSONL the config reads |
| `--skip-prepare` | off | do not flatten; `--flat-file` must exist |
| `--dry-run` | off | print the command and exit |
| `--extra` | — | forwarded to `tune run` as `key=value` overrides |

### `prepare_data_torchtune.py`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/OTel_LLM_sample_10.jsonl` | source chat JSONL |
| `--output` | `data/otel_cpt.jsonl` | destination `{text}` JSONL |

### Key config fields (`config_cpt_lora.yaml`)

| Field | Value shipped | Notes |
|---|---|---|
| `model` | `lora_qwen2_5_7b`, rank 16, alpha 32 | LoRA on q/v/output projections + MLP. |
| `tokenizer.path` / `merges_file` | `./assets/Qwen2.5-7B/vocab.json` / `merges.txt` | From `tune download`. |
| `checkpointer.checkpoint_dir` | `./assets/Qwen2.5-7B` | Four-shard safetensors listed in `checkpoint_files`. |
| `dataset` | `text_completion_dataset`, `data_files: ./data/otel_cpt.jsonl`, `packed: True` | `column: text` matches the converter; the launcher overrides `data_files` with the absolute `--flat-file` path. |
| `optimizer.lr` | `1.0e-4` | LoRA CPT LR (full FT would want much lower). |
| `epochs` / `batch_size` / `gradient_accumulation_steps` | `1` / `1` / `8` | Effective batch 8 per GPU step cycle. |
| `dtype` / `device` | `bf16` / `cuda` | |
| `output_dir` | `./outputs/torchtune_cpt_lora` | See Output. |

## 6. Output

`output_dir` (`./outputs/torchtune_cpt_lora` by default) holds:

- adapter (or full) weights in HF-compatible shards
- `logs/` from `DiskLogger`
- a recipe checkpoint if you left `save_adapter_weights_only: False`

Merge adapters with torchtune's `tune run eleuther_eval` / checkpointer helpers,
or load them with PEFT (`training/llm/peft/merge_adapter.py` after converting).

## 7. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** legacy project; follows torch device support (incl. Apple MPS for some recipes) with no vendor certification.


| Platform | Status | Evidence (checked August 2026) |
|---|---|---|
| NVIDIA CUDA | **Works — verified here** (last stable 0.6.x) | Smoke-tested on 1× **H100 80GB, CUDA 13.0**, `torch 2.13.0+cu130` + `torchtune 0.6.1` + **`torchao==0.10.0`**: 27 LoRA CPT steps, per-epoch loss decreasing, adapters saved. See §7c. Upstream benchmark tables ([torchtune README](https://github.com/pytorch/torchtune)) are all NVIDIA (4090/A6000/A100); CUDA-first install lines. |
| AMD ROCm | **Works — verified here** (nothing official upstream) | Smoke-tested on 1× MI355X (gfx950), ROCm 7.2.4, `torch 2.11.0+rocm7.2` + `torchtune 0.6.1` (stable wheel, no nightly, no `--no-deps`): 5 LoRA CPT steps, finite loss. See §7a. Upstream still has no ROCm CI/doc/image and is unmaintained ([issue #2883](https://github.com/pytorch/torchtune/issues/2883)), so torchtitan/Megatron are still the better long-term AMD targets. |
| Intel XPU | Mentioned in torch wheel options only | Same install line lists `xpu`; no torchtune-level support statement. |

### 7a. MI355X (ROCm 7.2) — tested

**Verdict: WORKS WITH ONE CONFIG FIX.** The fix is *not* AMD-specific — the shipped config
named a loss class that does not exist in torchtune 0.6.1 and would fail identically on
NVIDIA. Nothing had to change for ROCm itself.

**Host:** 8× AMD Instinct MI355X (gfx950, 288 GB VRAM), ROCm 7.2.4, Python 3.12.3, no NVIDIA GPUs.
**Tested:** single device (physical GPU 2 via `HIP_VISIBLE_DEVICES=2`), August 2026.

**Versions that resolved:**

| Package | Version |
|---|---|
| torch | `2.11.0+rocm7.2` (stable ROCm index, **not** a nightly) |
| torchvision | `0.26.0+rocm7.2` |
| torchtune | `0.6.1` (plain `pip install`, **no** `--no-deps`) |
| torchao / torchdata / datasets | `0.10.0` / `0.11.0` / `5.0.1` |

**Model choice.** The shipped default is Qwen2.5-7B; we smoked on **Qwen/Qwen2.5-0.5B**
instead. Same `qwen2_5` model family and tokenizer layout (`vocab.json` + `merges.txt`) and
the same `QWEN2` checkpointer path, so it exercises every code path the 7B would — but it
downloads as one 954 MB `model.safetensors` shard instead of four ~4 GB shards, which
matters on a host where disk was the binding constraint. Torchtune 0.6.1 ships
`lora_qwen2_5_0_5b` as a first-class builder, so this needed only CLI overrides.

**Exact smoke command** (run from inside `training/llm/torchtune/`):

```bash
source .env_train_llm_torchtune/bin/activate   # exports HIP/CUDA_VISIBLE_DEVICES=2, HF_HOME
export $(grep -v '^#' ../dev.env | xargs)      # HF_TOKEN

python3 train_llm_torchtune.py \
  --recipe lora_finetune_single_device --nproc-per-node 1 --extra \
  model._component_=torchtune.models.qwen2_5.lora_qwen2_5_0_5b \
  tokenizer.path=./assets/Qwen2.5-0.5B/vocab.json \
  tokenizer.merges_file=./assets/Qwen2.5-0.5B/merges.txt \
  tokenizer.max_seq_len=1024 \
  checkpointer.checkpoint_dir=./assets/Qwen2.5-0.5B \
  checkpointer.checkpoint_files=[model.safetensors] \
  gradient_accumulation_steps=1 epochs=1 max_steps_per_epoch=5 \
  save_adapter_weights_only=True \
  output_dir=/mnt/data_1.5t/outputs/train_llm_torchtune/smoke_final
```

**Evidence:**

```
INFO:torchtune.utils._logging:Model is initialized with precision torch.bfloat16.
	GPU peak memory allocation: 1.32 GiB
DEBUG:torchtune.utils._logging:Using flex attention for attention computation since a BlockMask was passed in.
1|1|Loss: 2.629014253616333    1|2|Loss: 3.023068428039551    1|3|Loss: 2.063380479812622
1|4|Loss: 1.401292085647583    1|5|Loss: 2.685070514678955
INFO:torchtune.utils._logging:Adapter checkpoint of size 0.02 GiB saved to .../epoch_0/adapter_model.safetensors
```

`rocm-smi` sampled mid-run on the training GPU:

```
GPU[2] : GPU use (%): 65      GPU[2] : GPU Memory Allocated (VRAM%): 3
GPU[2] : Temperature (Sensor junction) (C): 42.0   Current Socket Graphics Package Power (W): 259.0
```

All 5 losses finite, dataset packed (10 rows → packed sequences), adapter written. Also ran
3 epochs × 5 steps in an earlier pass with the same finite-loss behaviour.

**Quirks found on ROCm:**

1. **`LinearCrossEntropyLoss` does not exist in 0.6.1** — it is a post-0.6 main-branch API.
   The shipped config referenced it and died with an `InstantiationError`. `config_cpt_lora.yaml`
   now uses `torchtune.modules.loss.CEWithChunkedOutputLoss`, the 0.6.1 CE loss.
   **This is a version bug, not an AMD bug** — it fails the same way on CUDA.
2. **Benign dtype warning.** `UserWarning: Mismatch dtype between input and weight ... Cannot
   dispatch to fused implementation` from `torch.rms_norm` in `layer_norm.cpp`. Cosmetic —
   it just means the fused RMSNorm kernel is skipped for that call. Loss is unaffected.
3. **Attention.** torchtune selects **flex attention** (BlockMask path) automatically. Do not
   install `flash-attn` on ROCm; SDPA/flex is the supported path and it worked untouched.
4. **`device: cuda` is correct on ROCm.** Leave the YAML as `cuda` — ROCm torch presents the
   HIP device through the `torch.cuda` API. Select a GPU with `HIP_VISIBLE_DEVICES`, and never
   set `CUDA_VISIBLE_DEVICES=""` on ROCm.
5. **Checkpoint size.** With the default `save_adapter_weights_only: False`, the recipe writes
   a **full 0.92 GiB model copy plus a recipe_state every epoch**. Pass
   `save_adapter_weights_only=True` for smoke runs — it drops each epoch's write to 0.02 GiB.
6. **Outputs off the repo disk.** Override `output_dir` to
   `/mnt/data_1.5t/outputs/train_llm_torchtune/` rather than the in-folder `./outputs`.

**Not tested in this section:** the Qwen2.5-7B default. The 8-GPU distributed path *was*
subsequently tested — see the next subsection.

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

**Verdict: WORKS WITH CHANGES.** The distributed path runs clean on 8× MI355X: FSDP2
shards, all 8 ranks train, loss decreases, exit code 0, no teardown hang. The "changes" are
**not ROCm fixes** — they are (a) a *separate config* for the distributed recipe, and (b) a
*bigger dataset*, because the shipped 10-row sample cannot feed 8 ranks. Nothing AMD-specific
had to change.

**Why a new config.** torchtune recipes come in pairs and each half owns its config surface.
`lora_finetune_distributed` reads `fsdp_cpu_offload` / `fsdp_reshard_after_forward` /
`custom_sharded_layers`, and rejects single-device-only keys (`optimizer_in_bwd`, low-bit
optimizers). So the proven 1-GPU `config_cpt_lora.yaml` was left untouched and a new
**`config_cpt_lora_8gpu.yaml`** was added, carrying forward the `CEWithChunkedOutputLoss`
fix from §7a. Same model as the 1-GPU run (**Qwen2.5-0.5B**), *not* the untested 7B default.

**Exact launch command** (from inside `training/llm/torchtune/`, everything under `flock`):

```bash
source .env_train_llm_torchtune/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # MUST override: activate pins these to "2"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/mnt/data_1.5t/hf_cache

tune run --nnodes 1 --nproc_per_node 8 --master_port 29690 \
  lora_finetune_distributed --config config_cpt_lora_8gpu.yaml \
  dataset.data_files=/mnt/data_1.5t/outputs/train_llm_torchtune/gpu8/otel_cpt_x256.jsonl \
  tokenizer.max_seq_len=2048 batch_size=2 gradient_accumulation_steps=1 \
  epochs=1 max_steps_per_epoch=80 save_adapter_weights_only=True \
  output_dir=/mnt/data_1.5t/outputs/train_llm_torchtune/gpu8/run_long
```

**Parallelism / geometry.** FSDP2 (`fully_shard`) data parallel, world size 8, 1 node.
`fsdp_reshard_after_forward: True` (full shard), `fsdp_cpu_offload: False`, activation
checkpointing on, bf16. Per-rank batch 2 × seq 2048 → **global batch 16 seqs = 32,768
tokens/step**, no gradient accumulation. `custom_sharded_layers` is deliberately unset:
Qwen2.5-0.5B ties its embedding/output weights, so `output` is not a separately shardable
module. LoRA r=16 on `q_proj,v_proj,output_proj` + MLP; only adapters are trainable, but
FSDP still shards the frozen base weights.

**Real log lines** (80-step run, `run_long.log`):

```
ASSERT device_count=8 torch=2.11.0+rocm7.2
devices: ['AMD Instinct MI355X', ... x8]        === device_count==8 assertion PASSED ===
INFO:torchtune.utils._logging:FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
INFO:torchtune.utils._logging:Memory stats after model init:  GPU peak memory allocation: 0.66 GiB
Step 1  | loss:2.2638285160064697 ... peak_memory_alloc:4.512502193450928
Step 80 | loss:0.42030152678489685 ... tokens_per_second_per_gpu:15001.66796875
=== PHASE B exit=0 ===   === 2026-08-19T19:32:41+00:00 DONE rc=0 ===
```

**8-GPU `rocm-smi` evidence** — sampler ran *inside* the lock, in-band with training
(`gpu8/rocm_smi_inband*.txt`), never sampled from a separate shell after the fact:

```
SAMPLE 2026-08-19T19:27:51  GPU use (%): 99 99 99 99 99 99 99 99   VRAM%: 4 4 4 4 4 4 4 4
SAMPLE 2026-08-19T19:32:26  GPU use (%): 25 26 25 25 24 24 25 25   VRAM%: 4 4 4 4 4 4 4 4
SAMPLE 2026-08-19T19:32:39  GPU use (%): 78 77 78 77 77 76 76 75   VRAM%: 4 4 4 4 4 4 4 4
```

**PID cross-check** (same sample: `rocm-smi --showpids` vs `pgrep -af`) — this box is shared,
so the GPUs being busy proves nothing on its own; the VRAM must be held by *our* pids:

```
681271 python3 1 13581639680      # rocm-smi --showpids: 8 pids, one GPU each, ~13.6-14.0 GB
681272 ... 681278                 # (rank workers) + 681036 pt_elastic (the launcher)
pgrep -af, same sample: 681271..681278 = .../recipes/lora_finetune_distributed.py --config config_cpt_lora_8gpu.yaml
```

The second run repeated it with pids `722415-722422`. Other agents' pids appeared in the same
`--showpids` table with `VRAM USED = 0` and were excluded.

**Per-GPU VRAM delta vs single device** (identical per-GPU geometry — bs 2, seq 2048, 80 steps;
the 1-GPU baseline ran in the same locked script immediately before the 8-GPU run):

| Metric (per GPU) | 1× GPU `single_device` | 8× GPU FSDP2 | Delta |
|---|---|---|---|
| peak alloc after model init | 1.32 GiB | **0.66 GiB** | **−50 %** |
| steady-state `peak_memory_alloc` (step 80) | 5.4437 GiB | **4.5908 GiB** | **−0.853 GiB (−15.7 %)** |
| `peak_memory_reserved` | 6.2773 GiB | 5.6484 GiB | −0.629 GiB |
| `tokens_per_second_per_gpu` | 44,585 | 15,002 | aggregate ≈ 120k (≈2.7× total) |

Model-state memory halves as expected from sharding; the steady-state saving is only ~16 %
because at 0.5B the activations (which do *not* shard) dominate, and per-GPU throughput drops
~3× because all-gather/reduce-scatter traffic dwarfs the compute for a model this small.
**8 GPUs are the wrong tool for a 0.5B LoRA** — this run proves the *path*, not a speedup.

**What differed from the 1-GPU run:**

1. **Different recipe + new config** — `lora_finetune_distributed` + `config_cpt_lora_8gpu.yaml`
   (new file; the 1-GPU config was not mutated).
2. **The stale venv pin bites here.** `.env_train_llm_torchtune/bin/activate` exports
   `HIP_VISIBLE_DEVICES=2` / `CUDA_VISIBLE_DEVICES=2` (lines 71-72). Sourcing it and launching
   with `--nproc_per_node 8` gives you 8 processes fighting over **one** GPU. Always re-export
   both to `0,1,...,7` *after* sourcing, and assert `torch.cuda.device_count() == 8`.
3. **Bigger dataset required.** The shipped sample is 10 rows / 12,492 tokens — it cannot fill
   8 ranks. We replicated it ×64 (640 rows) and ×256 (2560 rows) into
   `/mnt/data_1.5t/outputs/train_llm_torchtune/gpu8/`, outside the repo. **Row survival: all
   2560/2560 rows survived** — `text_completion_dataset` packs with `split_across_pack=True`,
   so nothing is silently dropped for exceeding `max_seq_len` (unlike SFT datasets, which do
   drop long rows). Treat the result as a **pipeline proof, not a learning result**: the loss
   fall 2.26 → 0.42 is memorisation of 256 copies of the same 10 documents.
4. **Rank-0-only checkpointing.** Only rank 0 writes; with `save_adapter_weights_only=True`
   each epoch costs 0.02 GiB. Leave it `False` and you get a full model copy per epoch.
   There is no "don't checkpoint at all" switch in 0.6.1 — the end-of-epoch save is
   unconditional, so cap epochs and delete the output dir afterwards.
5. **Benign new warning:** `DTensor is synchronizing RNG states of every rank with the state
   from rank 0. This behavior is deprecated.` (one line per rank, from
   `torch/distributed/tensor/_random.py`). Cosmetic; training is unaffected.
6. Same ROCm quirks as §7a otherwise: flex attention selected automatically, the cosmetic
   fused-RMSNorm dtype warning, `device: cuda` correct on ROCm. Fused AdamW worked on all 8.
7. **Clean teardown** — `PHASE B exit=0`, no `torchrun` restart/abort messages, no NCCL/RCCL
   timeout, no orphaned rank processes (VRAM back to 0 in the sample after the run).

Reproducing: run everything through the machine-wide lock (`flock /tmp/mi355x_gpu8.lock`) if
you share this box, and keep outputs off the repo disk. Raw logs, in-band `rocm-smi` samples
and a condensed `EVIDENCE.md` are under `/mnt/data_1.5t/outputs/train_llm_torchtune/gpu8/`
(weights deleted after capture).

**Weights on disk:** `assets/Qwen2.5-0.5B/` is **954 MB and deliberately left untracked** —
do not commit it. Re-download it with the `tune download` line in §1, or point
`checkpointer.checkpoint_dir` at a copy under `/mnt/data_1.5t`.

### 7c. H100 (CUDA 13.0) — tested August 2026

**Verdict: WORKS WITH CHANGES.** torchtune 0.6.1 runs clean on H100/CUDA-13 once two
things are pinned: **`torchao==0.10.0`** (the 0.6.1 import path) and the **`CEWithChunkedOutputLoss`**
fix already carried in `config_cpt_lora.yaml` from §7a. Neither change is NVIDIA-specific.
The only *hardware*-driven note is that we assembled the base model from a locally-cached
Qwen2.5-1.5B checkpoint because outbound HuggingFace was proxy-blocked on this node (see below) —
that is an environment quirk, not a torchtune limitation. Single-device only this wave; GPUs
0–3 were running someone else's production job, so we used **physical GPU 7** exclusively.

**Host:** 8× NVIDIA H100 80GB HBM3, driver **580.173.02**, **CUDA 13.0**, Hopper cc `(9, 0)`,
Python 3.12.3. Tested single device on **GPU 7** (`CUDA_VISIBLE_DEVICES=7`), master port 29644.

**Versions that resolved:**

| Package | Version |
|---|---|
| torch | `2.13.0+cu130` (default stable wheel — **no `--index-url` needed** on a CUDA-13 host) |
| torchvision | `0.28.0` |
| torchtune | `0.6.1` |
| torchao | **`0.10.0`** (pinned + `--no-deps`; see quirk 1) |
| torchdata / datasets / safetensors | `0.11.0` / `5.0.1` / `0.8.0` |
| numpy / omegaconf / tokenizers | `2.5.2` / `2.3.1` / `0.23.1` |

**Install that worked** (exact commands):

```bash
python3 -m venv .env_torchtune && source .env_torchtune/bin/activate
pip install torch numpy                          # -> torch 2.13.0+cu130, numpy 2.5.2
pip install "torchtune==0.6.1" torchvision "python-dotenv>=1.0.1"
pip install "torchao==0.10.0" --no-deps          # 0.6.1 imports torchao.dtypes.nf4tensor
python -c "import torch;print(torch.__version__,torch.version.cuda)"  # 2.13.0+cu130 13.0 (re-check: not clobbered)
```

**Model choice.** The proxy on this node returns `403 Forbidden` for huggingface.co, so
`tune download Qwen/Qwen2.5-0.5B` fails (`httpx.ProxyError: 403`). The shared HF cache
(`HF_HOME=/mnt/gsma/gsma/gsma/models`) had no Qwen2.5 *causal* base, so we assembled one from
the cached **`qwen2.5_1.5b_telelog_classification/checkpoint-2200`** checkpoint: it is a
`Qwen2ForSequenceClassification` fine-tune whose backbone *is* Qwen2.5-1.5B (28 layers, hidden
1536, `tie_word_embeddings: true`, vocab 151936). We wrote a `Qwen2ForCausalLM` `config.json`,
copied its `vocab.json`+`merges.txt`, symlinked the intact shard 1, and rewrote shard 2 with
the classifier head `score.weight` **stripped** (torchtune's `qwen2_hf_to_tune` converter
raises `Found unexpected key: "score.weight"` otherwise). Result: a clean 338-tensor causal
LM. On a normal (un-proxied) host just use the §7a `Qwen2.5-0.5B` download instead — the code
path is identical. For 1.5B the torchtune builder is **`lora_qwen2_5_1_5b_base`** (note the
`_base` suffix; only 0.5B is bare `lora_qwen2_5_0_5b`).

**Exact smoke command** (run from inside `training/llm/torchtune/`, `$DEST` = the assembled base dir):

```bash
export CUDA_VISIBLE_DEVICES=7 MASTER_PORT=29644 HF_HOME=/mnt/gsma/gsma/gsma/models
python train_llm_torchtune.py \
  --recipe lora_finetune_single_device --nproc-per-node 1 --extra \
  model._component_=torchtune.models.qwen2_5.lora_qwen2_5_1_5b_base \
  tokenizer.path=$DEST/vocab.json \
  tokenizer.merges_file=$DEST/merges.txt \
  tokenizer.max_seq_len=1024 \
  checkpointer.checkpoint_dir=$DEST \
  checkpointer.checkpoint_files=[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors] \
  gradient_accumulation_steps=1 epochs=3 \
  save_adapter_weights_only=True \
  output_dir=/dev/shm/h100/out/torchtune/smoke
```

**Step count:** 10 rows packed at `max_seq_len=1024` → **9 packed sequences/epoch**; with
`batch_size=1`, `gradient_accumulation_steps=1`, world=1 that is **9 optimizer steps/epoch ×
3 epochs = 27 real steps** (the §7a MI355X run used `max_steps_per_epoch=5`; we lifted the cap
and ran 3 epochs so the loss trend is visible). Exit code 0.

**Evidence** (real log lines):

```
INFO:torchtune.utils._logging:Model is initialized with precision torch.bfloat16.
Packing dataset: 100%|██████████| 10/10 [00:00<00:00, 152.49it/s]
1|1|Loss: 2.4011404514312744   1|4|Loss: 1.045843243598938    1|9|Loss: 1.8566558361053467
2|10|Loss: 2.344449996948242   2|17|Loss: 0.9861388802528381   2|18|Loss: 1.7228327989578247
3|19|Loss: 2.7127492427825928  3|22|Loss: 0.909440815448761    3|27|Loss: 2.100238561630249
        GPU peak memory allocation: 6.41 GiB
INFO:torchtune.utils._logging:Adapter checkpoint of size 0.03 GiB saved to .../epoch_2/adapter_model.safetensors
PHASE_DONE rc=0
```

Per-step loss is noisy (only 10 packed docs, LoRA rank 16), but the **per-epoch trend
decreases monotonically** — mean `2.0621 → 2.0282 → 1.8938`, min `1.0458 → 0.9861 → 0.9094`.
Treat this as a **pipeline proof, not a learning result** (same caveat as §7a/§7b). Three
adapters written (`epoch_0/1/2`, 35 MB `.safetensors` each).

**GPU-7 residency** — `nvidia-smi` sampled *inside* the run, filtered to GPU 7's UUID and to
our training PID (`nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so you must filter by
`gpu_uuid` + pid, not assume index 0 is yours):

```
GPU-9eb34eec-449b-7839-d362-d1e995e95239, 1342848, /dev/shm/h100/venvs/torchtune/bin/python, 7586 MiB
7, GPU-9eb34eec-449b-7839-d362-d1e995e95239, 2 %, 6243 MiB     # index 7, util, mem.used
```

Our PID climbed 518 MiB (init) → **8899 MiB peak** on GPU 7; GPUs 0–3 (the production job,
~64 GB/rank) were never touched. Low util % samples reflect the tiny/fast workload (~5 it/s),
not idleness — the 6–8.9 GB VRAM held by our exact PID is the residency proof.

**Quirks / what changed vs. the MI355X recipe:**

1. **`torchao==0.10.0` is mandatory and must be pinned.** `pip install torchtune==0.6.1`
   does *not* pull torchao, and a bare `pip install torchao` grabs **0.18.0**, which renamed
   `torchao.dtypes.nf4tensor` → torchtune 0.6.1's `import` dies with
   `ModuleNotFoundError: No module named 'torchao.dtypes.nf4tensor'`. Pin `0.10.0` (the §7a
   version) with `--no-deps` so it cannot drag torch off cu130. **Not H100-specific** — this
   bites any fresh 0.6.1 install; the ROCm venv happened to already have 0.10.0.
2. **tf32 engages automatically.** torchtune's `training/precision.py` calls
   `torch.set_float32_matmul_precision("high")` and sets `torch.backends.cudnn.allow_tf32=True`
   on `device: cuda`. Nothing to add — the guard is on by default on the H100.
3. **Attention = flex, no flash-attn to reverse.** torchtune selects the flex-attention
   (BlockMask) path automatically and exposes no `attn_implementation` knob in these configs,
   so the brief's "swap sdpa→flash_attention_2" does not apply here — there is nothing to swap.
   (flash-attn *is* installable on H100, but torchtune never calls it.) `device: cuda` is
   already correct; no `HIP_VISIBLE_DEVICES` to drop from the config.
4. **VRAM is a non-issue at this size.** 1.5B LoRA peaked at 6.41 GiB / ~8.9 GB by-PID — no
   OOM, no offload, no batch/seq reduction needed on the 80 GB card (vs 288 GB on MI355X).
5. **Env-driven, not torchtune:** outbound HF is proxy-blocked (`403`), and the `/mnt/gsma`
   share rejects the symlink/replace ops PyTorch's CUDA libs perform during install
   (`OSError: [Errno 1] Operation not permitted` on `libcusparseLt.so.0`). We built the venv
   on tmpfs (`/dev/shm/h100/venvs/torchtune`, symlinked back as `.env_torchtune`) and kept
   weights+outputs under `/dev/shm/h100/out/torchtune`. On a normal CUDA host the plain §1
   install into an in-folder `.venv` works.

**Multi-GPU (2, then 8) — DEFERRED, not run.** GPUs 0–3 held a production job this wave, so
only the single-device path was exercised. An 8-GPU pass would mirror §7b exactly: switch to
`--recipe lora_finetune_distributed` + `config_cpt_lora_8gpu.yaml` (carrying the
`CEWithChunkedOutputLoss` fix and the `torchao==0.10.0` pin), replicate the 10-row sample so it
can fill 8 ranks, launch with `tune run --nnodes 1 --nproc_per_node 8 --master_port 29644`, and
assert `torch.cuda.device_count()==<free GPUs>`. Expect FSDP2 sharding to work on H100 as it did
on MI355X; the lead coordinates that once GPUs 0–3 free up.

**Weights on disk:** the assembled base and all adapters lived under `/dev/shm/h100/` and were
deleted after evidence capture — nothing added to the repo.

## 8. Notes

- **CPT, not SFT.** `text_completion_dataset` + `packed: True` concatenates
  documents and trains next-token prediction on every token. There is no chat
  template and no prompt masking. That is the point of this folder.
- **LoRA, not full FT.** Full-parameter CPT of a 7B on 8×H100 is possible but
  needs a different recipe (`full_finetune_distributed`) and a much lower LR.
  LoRA is the cheaper, safer first run.
- **YAML is the program.** The launcher only flattens data and shells out.
  Change rank, LR, seq length, or the checkpoint dir in the YAML (or via
  `--extra lora_rank=32`).
- **Qwen2.5 tokenizer paths.** The 0.6 Qwen config wants `vocab.json` +
  `merges.txt` from the HF snapshot, not a `tokenizer.model`. Llama configs
  want the SentencePiece file instead — swap the whole `tokenizer:` block if
  you change families.
- **Relative paths in the YAML** (`./assets`, `./data`, `./outputs`) resolve against the
  directory you launch from — run from inside this folder. The launcher passes the
  flat-file path as an absolute override, so the dataset path is safe either way.
- **Why this still exists.** You asked for a torchtune CPT example. The
  maintained replacements in this repo are `training/llm/torchtitan` (PyTorch
  native, FSDP2) and `training/llm/megatron` (3D parallel, indexed data).
