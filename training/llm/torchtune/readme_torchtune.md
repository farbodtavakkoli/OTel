# Setup & usage — `train_llm_torchtune.py`

## Overview & when to use

Continued pre-training (domain-adaptive next-token prediction) with
[torchtune](https://github.com/pytorch/torchtune). This is the **CPT example**
for this repo: unstructured text, no chat template, loss on every token.

This folder targets `torchtune==0.6.1`, the last stable API.

Files in this folder:
- `train_llm_torchtune.py` — flattens the shipped sample, then execs `tune run`.
- `prepare_data_torchtune.py` — chat JSONL (`messages`) → `{text: ...}` JSONL.
- `config_cpt_lora.yaml` — LoRA CPT config using `text_completion_dataset`.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample (see Data below).
- `requirements_torchtune.txt` — pinned 0.6.x environment.
- `readme_torchtune.md` — this file.

> **Hardware coverage:** **AMD MI355X (gfx950, ROCm 7.2.4)** at 1 GPU
> (`lora_finetune_single_device`) and 8 GPUs (`lora_finetune_distributed`, FSDP2, with
> `config_cpt_lora_8gpu.yaml`), and **NVIDIA H100 80GB (CUDA 13.0)** at 1 GPU with
> `torch 2.13.0+cu130` and **`torchao==0.10.0`**. No ROCm- or CUDA-specific code change is
> required — see [§7a](#7a-amd-mi355x-rocm-72), [§7b](#7b-8-gpu-distributed-mi355x-rocm-72)
> and [§7c](#7c-nvidia-h100-cuda-130). Multi-GPU on NVIDIA is not covered, and the shipped
> **Qwen2.5-7B default is untested** (the 0.5B/1.5B of the same family are).

## 1. Install

### NVIDIA (CUDA)

```bash
cd training/llm/torchtune
python3.12 -m venv .venv && source .venv/bin/activate

# torch FIRST: on CUDA 13 hosts the default stable wheel is native cu130 — no --index-url needed
pip install torch numpy                 # -> the current CUDA 13 build (see H100 §7c)
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

`torchtune==0.6.1` is pure Python on top of torch and installs on a **stable** ROCm torch
wheel — no torch nightly, and no `--no-deps` on the torchtune install (pip does not pull a
CUDA torch over the ROCm one).

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts / adapters
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

```bash
cd training/llm/torchtune
python3.12 -m venv .env_torchtune && source .env_torchtune/bin/activate

# 1) ROCm torch FIRST, from the ROCm index
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision

# 2) then torchtune — normal resolution, no --no-deps
pip install torchtune==0.6.1 python-dotenv

# 3) torchao: pin 0.10.0 if it is not already present. torchtune 0.6.1 imports
#    torchao.dtypes.nf4tensor, which 0.18.x renamed; --no-deps keeps torch on ROCm.
python -c "import torchao;print(torchao.__version__)" || pip install "torchao==0.10.0" --no-deps

# 4) a small base model (7B default is overkill for a smoke test)
tune download Qwen/Qwen2.5-0.5B --output-dir ./assets/Qwen2.5-0.5B
```

Do **not** `pip install flash-attn` on ROCm — torchtune uses torch SDPA / flex attention,
which is the supported ROCm path and works as-is.

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
  --config config_cpt_lora_8gpu.yaml \
  > train_llm_torchtune.log 2>&1 &

tail -f train_llm_torchtune.log
```

**The two recipes have different config surfaces and each needs its own YAML.**
`lora_finetune_distributed` reads `fsdp_cpu_offload` / `fsdp_reshard_after_forward` /
`custom_sharded_layers` and rejects single-device-only keys (`optimizer_in_bwd`, low-bit
optimizers). `config_cpt_lora.yaml` is the single-device config;
`config_cpt_lora_8gpu.yaml` is the distributed one. Do not cross them.

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

Two things to set before any multi-GPU launch:

```bash
# If a previous single-GPU session left a pin in the venv's activate script, override it
# AFTER sourcing — otherwise 8 processes fight over one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ROCm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"
```

If the box is shared, use a non-default rendezvous port. The launcher does not expose one, so
either `export MASTER_PORT=<port>` or call `tune run` directly with `--master_port <port>`
(as in [§7b](#7b-8-gpu-distributed-mi355x-rocm-72)).

On a shared machine, run the whole job under a machine-wide lock (e.g.
`flock /tmp/gpu8.lock`) so nothing else claims the cards mid-run.

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

`output_dir` (`./outputs/torchtune_cpt_lora` by default, and
`./outputs/torchtune_cpt_lora_8gpu` for `config_cpt_lora_8gpu.yaml`) holds:

- adapter (or full) weights in HF-compatible shards
- `logs/` from `DiskLogger`
- a recipe checkpoint if you left `save_adapter_weights_only: False`

Merge adapters with torchtune's `tune run eleuther_eval` / checkpointer helpers,
or load them with PEFT (`training/llm/peft/merge_adapter.py` after converting).

## 7. Hardware support

| Platform | Status |
|---|---|
| NVIDIA CUDA | **Works** — H100 80GB, CUDA 13.0, `torch 2.13.0+cu130` + `torchao==0.10.0` (§7c) |
| AMD ROCm | **Works** — MI355X (gfx950), ROCm 7.2.4, `torch 2.11.0+rocm7.2`, 1 and 8 GPUs (§7a/§7b) |

### 7a. AMD MI355X (ROCm 7.2)

**This path works with one config fix** — the `CEWithChunkedOutputLoss` loss class (quirk 1),
which is a torchtune-version fix, not an AMD one. Nothing had to change for ROCm itself.
Covered: single device, `lora_finetune_single_device`, Qwen2.5-0.5B LoRA CPT. The Qwen2.5-7B
default is not covered.

**Model choice.** The smoke uses **Qwen/Qwen2.5-0.5B** instead of the shipped 7B default —
same `qwen2_5` family, tokenizer layout (`vocab.json` + `merges.txt`) and `QWEN2` checkpointer
path, one shard instead of four. `lora_qwen2_5_0_5b` is a first-class 0.6.1 builder, so this
needs only CLI overrides.

**Smoke command** (run from inside `training/llm/torchtune/`):

```bash
source .env_torchtune/bin/activate             # pin the GPU with HIP/CUDA_VISIBLE_DEVICES
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
  output_dir=$OUTPUT_DIR/train_llm_torchtune/smoke_final
```

**Expected output:**

```
INFO:torchtune.utils._logging:Model is initialized with precision torch.bfloat16.
	GPU peak memory allocation: 1.32 GiB
DEBUG:torchtune.utils._logging:Using flex attention for attention computation since a BlockMask was passed in.
1|1|Loss: 2.629014253616333    1|2|Loss: 3.023068428039551    1|3|Loss: 2.063380479812622
1|4|Loss: 1.401292085647583    1|5|Loss: 2.685070514678955
INFO:torchtune.utils._logging:Adapter checkpoint of size 0.02 GiB saved to .../epoch_0/adapter_model.safetensors
```

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
   a full model copy plus a recipe_state every epoch. Pass `save_adapter_weights_only=True`
   for smoke runs — it writes only the adapter.
6. **Outputs off the repo disk.** Override `output_dir` to
   `$OUTPUT_DIR/train_llm_torchtune/` rather than the in-folder `./outputs`.

### 7b. 8-GPU distributed (MI355X, ROCm 7.2)

**This path works with changes.** The changes are **not ROCm fixes** — they are (a) the
separate **`config_cpt_lora_8gpu.yaml`**, because the distributed recipe has its own config
surface (see §4), and (b) a *bigger dataset*, because the shipped 10-row sample cannot feed
8 ranks.

**Launch command** (from inside `training/llm/torchtune/`; run it under a machine-wide
`flock` if the box is shared, so the job owns all 8 GPUs):

```bash
source .env_torchtune/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # MUST override any single-GPU pin in activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# HF_HOME as exported in the install section

tune run --nnodes 1 --nproc_per_node 8 --master_port 29690 \
  lora_finetune_distributed --config config_cpt_lora_8gpu.yaml \
  dataset.data_files=$OUTPUT_DIR/train_llm_torchtune/gpu8/otel_cpt_x256.jsonl \
  tokenizer.max_seq_len=2048 batch_size=2 gradient_accumulation_steps=1 \
  epochs=1 max_steps_per_epoch=80 save_adapter_weights_only=True \
  output_dir=./outputs/torchtune_cpt_lora_8gpu
```

`output_dir` above is the value `config_cpt_lora_8gpu.yaml` already ships (it is spelled out
here only to make the destination explicit); `${output_dir}` feeds `checkpointer.output_dir`,
`metric_logger.log_dir` and the profiler, so overriding it moves every artifact at once.

`custom_sharded_layers` is deliberately unset: Qwen2.5-0.5B ties its embedding/output
weights, so `output` is not a separately shardable module.

**Expected output** (80-step run, exit code 0):

```
INFO:torchtune.utils._logging:FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
INFO:torchtune.utils._logging:Memory stats after model init:  GPU peak memory allocation: 0.66 GiB
Step 1  | loss:2.2638285160064697 ... peak_memory_alloc:4.512502193450928
Step 80 | loss:0.42030152678489685
```

**What differs from the 1-GPU run:**

1. **Different recipe + separate config** — `lora_finetune_distributed` +
   `config_cpt_lora_8gpu.yaml` (the 1-GPU config is not mutated).
2. **A stale venv pin.** If the venv's `activate` script exports a single-GPU
   `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` from an earlier run, sourcing it and
   launching with `--nproc_per_node 8` gives you 8 processes contending for **one** GPU. Always
   re-export both to the full list *after* sourcing, and assert
   `torch.cuda.device_count() == 8`.
3. **Bigger dataset required.** The shipped 10-row sample cannot fill 8 ranks. Replicate it
   (e.g. ×256 → 2560 rows) into a directory outside the repo and point `dataset.data_files`
   there. Nothing is silently dropped for exceeding `max_seq_len` —
   `text_completion_dataset` packs with `split_across_pack=True`.
4. **Rank-0-only checkpointing.** There is no "don't checkpoint at all" switch in 0.6.1 — the
   end-of-epoch save is unconditional, so pass `save_adapter_weights_only=True`, cap epochs
   and delete the output dir afterwards.
5. **Benign warning:** `DTensor is synchronizing RNG states of every rank with the state
   from rank 0. This behavior is deprecated.` (one line per rank). Training is unaffected.
6. Same ROCm quirks as §7a otherwise: flex attention selected automatically, the cosmetic
   fused-RMSNorm dtype warning, `device: cuda` correct on ROCm.

The run writes to `./outputs/torchtune_cpt_lora_8gpu` (the config's `output_dir`); override it
to a path outside the repo if the repo disk is small, and delete the artifacts afterwards.
Keep `assets/Qwen2.5-0.5B/` untracked — re-download it with the `tune download` line in §1, or
point `checkpointer.checkpoint_dir` at a copy on a larger disk.

### 7c. NVIDIA H100 (CUDA 13.0)

**This path works with changes.** torchtune 0.6.1 runs clean on H100 / CUDA 13 once two
things are pinned: **`torchao==0.10.0`** (the 0.6.1 import path) and the
**`CEWithChunkedOutputLoss`** fix already carried in `config_cpt_lora.yaml` from §7a. Neither
change is NVIDIA-specific. Coverage is single-device Qwen2.5-1.5B LoRA CPT.

**Install:**

```bash
python3 -m venv .env_torchtune && source .env_torchtune/bin/activate
pip install torch numpy                          # -> the current CUDA 13 build, numpy 2.5.2
pip install "torchtune==0.6.1" torchvision "python-dotenv>=1.0.1"
pip install "torchao==0.10.0" --no-deps          # 0.6.1 imports torchao.dtypes.nf4tensor
python -c "import torch;print(torch.__version__,torch.version.cuda)"  # 2.13.0+cu130 13.0 (re-check: not clobbered)
```

**Model choice.** On a normal host, use the §7a `Qwen2.5-0.5B` download — the code path is
identical. Behind a proxy that returns `403 Forbidden` for huggingface.co,
`tune download` fails with `httpx.ProxyError: 403` and you must supply a local base instead.
Two things to know if you build one from an existing checkpoint:

- For 1.5B the torchtune builder is **`lora_qwen2_5_1_5b_base`** (note the `_base` suffix;
  only 0.5B is the bare `lora_qwen2_5_0_5b`).
- If your source is a `Qwen2ForSequenceClassification` fine-tune, strip the classifier head
  `score.weight` and write a `Qwen2ForCausalLM` `config.json` beside the shards; torchtune's
  `qwen2_hf_to_tune` converter raises `Found unexpected key: "score.weight"` otherwise.

**Smoke command** (run from inside `training/llm/torchtune/`, `$DEST` = the base-model dir):

```bash
export CUDA_VISIBLE_DEVICES=7 MASTER_PORT=29644   # HF_HOME as exported in the install section
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
  output_dir=/dev/shm/torchtune_out/smoke
```

(The config's own `output_dir` is `./outputs/torchtune_cpt_lora`; override it as above when
the repo disk is small or on a network mount — see quirk 3 below.)

**Expected output:**

```
INFO:torchtune.utils._logging:Model is initialized with precision torch.bfloat16.
Packing dataset: 100%|██████████| 10/10 [00:00<00:00, 152.49it/s]
1|1|Loss: 2.4011404514312744   1|4|Loss: 1.045843243598938    1|9|Loss: 1.8566558361053467
2|10|Loss: 2.344449996948242   2|17|Loss: 0.9861388802528381   2|18|Loss: 1.7228327989578247
3|19|Loss: 2.7127492427825928  3|22|Loss: 0.909440815448761    3|27|Loss: 2.100238561630249
        GPU peak memory allocation: 6.41 GiB
INFO:torchtune.utils._logging:Adapter checkpoint of size 0.03 GiB saved to .../epoch_2/adapter_model.safetensors
```

Per-step loss is noisy on so few packed documents; read the trend per epoch. One adapter is
written per epoch (`epoch_0/1/2`). To confirm residency on a shared node, filter `nvidia-smi`
by `gpu_uuid` **and** your training PID — `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so
index 0 in its output is not necessarily your card.

**Quirks / what changed vs. the MI355X recipe:**

1. **`torchao==0.10.0` is mandatory and must be pinned.** `pip install torchtune==0.6.1`
   does *not* pull torchao, and a bare `pip install torchao` grabs **0.18.0**, which renamed
   `torchao.dtypes.nf4tensor` → torchtune 0.6.1's `import` dies with
   `ModuleNotFoundError: No module named 'torchao.dtypes.nf4tensor'`. Pin `0.10.0` with
   `--no-deps` so it cannot drag torch off cu130. **Not H100-specific** — this affects any
   fresh 0.6.1 install on either vendor.
2. **Attention = flex; there is no flash-attn to install.** torchtune selects the
   flex-attention (BlockMask) path automatically and exposes no `attn_implementation` knob in
   these configs. `device: cuda` is already correct; no `HIP_VISIBLE_DEVICES` to drop.
3. **Environment, not torchtune:** a shared network mount can reject the symlink/replace ops
   PyTorch's CUDA libs perform during install (`OSError: [Errno 1] Operation not permitted`
   on `libcusparseLt.so.0`). Build the venv on tmpfs (e.g. `/dev/shm/torchtune_venv`,
   symlinked back as `.env_torchtune`) and keep weights and outputs there too. On a normal
   CUDA host the plain §1 install into an in-folder `.venv` works.

**Multi-GPU on NVIDIA — not covered here.** An 8-GPU pass mirrors §7b: switch to
`--recipe lora_finetune_distributed` + `config_cpt_lora_8gpu.yaml` (which carries the
`CEWithChunkedOutputLoss` fix), keep the `torchao==0.10.0` pin, replicate the 10-row sample so
it can fill 8 ranks, launch with `tune run --nnodes 1 --nproc_per_node 8 --master_port <port>`,
and assert `torch.cuda.device_count()` matches. Expect FSDP2 sharding to work on H100 as it
does on MI355X.

Keep the base model and all adapters outside the repo and delete them afterwards — nothing
generated here should be committed.

## 8. Notes

- **CPT, not SFT.** `text_completion_dataset` + `packed: True` concatenates
  documents and trains next-token prediction on every token. There is no chat
  template and no prompt masking.
- **LoRA, not full FT.** Full-parameter CPT needs a different recipe
  (`full_finetune_distributed`) and a much lower LR.
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
