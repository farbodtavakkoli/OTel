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
- `config_sft_lora_smoke_mi355x.yaml`, `config_sft_lora_smoke_mi355x_8gpu.yaml`,
  `config_sft_lora_smoke_h100.yaml` (flash-attn), `config_sft_lora_smoke_h100_sdpa.yaml` —
  small single-/8-GPU smoke variants of the LoRA SFT recipe (section 2).
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample the SFT configs point at (self-contained smoke data).
- `data/dpo_sample.jsonl` — 10-row **synthetic** preference sample the DPO config points at (see section 4).
- `requirements_axolotl.txt` — dependency list plus the required install order, with a commented ROCm variant.

> **Hardware coverage.** LoRA SFT runs on **AMD MI355X (ROCm 7.2)** at 1 and 8 GPUs and on
> **NVIDIA H100 80GB (CUDA 13.0)** with both `sdpa` and `flash_attention_2` — see section 2.
> QLoRA, ZeRO-3 full fine-tuning, DPO and multi-node are **not covered**. The shipped
> configs target a single node of 8 GPUs; treat every batch-size and learning-rate number as
> a starting point. Running them on MI355X needs three overrides — `sdpa`, `tf32: false`,
> and the ROCm torch wheel (section 2).

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

Plain `pip` in a venv also works and is simpler where PyPI already serves native cu130
wheels — no `--index-url` and no `UV_TORCH_BACKEND` needed:

```bash
python3 -m venv .env_axolotl && source .env_axolotl/bin/activate
pip install torch numpy                      # step 1: torch FIRST (cu130 wheel from PyPI)
pip install packaging ninja
pip install --no-build-isolation axolotl     # step 2: base axolotl, no extras
pip install -r requirements_axolotl.txt      # step 3: this folder's launcher deps
```

`pip install axolotl` **replaces** torch (it pins `torch==2.12.1`). On a CUDA host that pin
also resolves to a `+cu130` wheel, so no recovery reinstall is needed — but re-verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # -> 2.12.1+cu130 13.0
axolotl --help
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

The `[deepspeed]`, `[flash-attn]` and `[vllm]` extras are optional: base axolotl is enough
for single-GPU LoRA/QLoRA and for DDP multi-GPU LoRA. `[deepspeed]` is only needed for
`config_sft_full.yaml`.

**flash-attn on CUDA.** There is no prebuilt flash-attn wheel on the default PyPI index
(`pip install --only-binary=:all: flash-attn` finds no distribution), so
`attn_implementation: flash_attention_2` requires a source build against the on-box `nvcc`:

```bash
export CUDA_HOME=/usr/local/cuda            # nvcc 13.0
MAX_JOBS=16 pip install --no-build-isolation flash-attn==2.8.3
```

The build does not clobber torch (re-verify anyway). `sdpa` is the zero-build fallback and
trains the same recipe unchanged; on Hopper, axolotl also offers Flash Attention 4
(`pip install flash-attn-4`).

Smoke run (single GPU, `config_sft_lora_smoke_h100_sdpa.yaml` — a copy of
`config_sft_lora.yaml` with a small base model, `sdpa`, `tf32: true`,
`sample_packing: false`, seq 2048, batch 1, `max_steps: 20`, and the LoRA-kernel autopatch
disabled; `config_sft_lora_smoke_h100.yaml` is the identical recipe with
`attn_implementation: flash_attention_2`):

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=/dev/shm/dscache_axolotl AXOLOTL_DO_NOT_TRACK=1 \
python3 train_llm_axolotl.py --config config_sft_lora_smoke_h100_sdpa.yaml \
    --num-processes 1 --main-process-port 29645
```

**Expected output** (20/20 steps, finite loss trending down, clean save, exit code 0):

```
{'loss': '1.325', 'grad_norm': '99.88', 'learning_rate': '0.0002',    'ppl': '3.761', 'epoch': '0.1111'}
{'loss': '0.1121','grad_norm': '7.018', 'learning_rate': '1.231e-06', 'ppl': '1.119', 'epoch': '2.222'}
{'train_runtime': '15.86', 'train_samples_per_second': '1.261', 'train_loss': '0.9872'}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-h100-sdpa
```

Nine of the ten sample rows survive `sequence_len: 2048` (one long row is dropped). The
adapter, `checkpoint-20/` and the resolved chat template land in the output dir.

### AMD / ROCm

Upstream's only AMD walkthrough is a manual source build; **it is not required.** On MI355X
(gfx950, ROCm 7.2) a plain `pip install axolotl` — no source checkout, no extras, no ROCm
flash-attn fork — trains LoRA SFT once the torch wheel is put back and two config keys are
overridden:

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

**ROCm rules for the configs in this folder:**

- **Put the ROCm torch wheel back after installing axolotl** (step 3 above). Without the
  force-reinstall you train nothing: the CUDA torch cannot see the AMD GPUs.
- **`attn_implementation: flash_attention_2` → `sdpa`** in every YAML. Do not
  `pip install flash-attn` on ROCm — the PyPI package is a CUDA-only source build. The ROCm
  fork (`github.com/ROCm/flash-attention`, built with `GPU_ARCHS` for your card) is a
  separate manual build that nothing here needs.
- **`tf32: true` → `false`** in every YAML — tf32 raises on ROCm.
- **bitsandbytes >= 0.50.0** (step 4) or QLoRA fails at import with
  `libbitsandbytes_rocm72.so not found`.
- **Skip the extras** (`[deepspeed]`, `[flash-attn]`, `[vllm]`) — they pull CUDA-only bits.
  Base axolotl still drags in CUDA-only `xformers` and `nvidia-*-cu13` wheels as transitive
  deps; leave them alone and do not enable xformers attention, which is incompatible with
  ROCm.
- **DeepSpeed is reported broken with Axolotl on ROCm.** For sharded AMD runs replace the
  `deepspeed:` key in `config_sft_full.yaml` with the commented `fsdp_version: 2` block.
  LoRA needs neither — DDP is enough.
- When several trainings share a machine, pass a unique `--main-process-port` (e.g. 29660)
  to avoid the default 29500 collision.

Smoke run (single GPU, `config_sft_lora_smoke_mi355x.yaml` — a copy of `config_sft_lora.yaml`
with Qwen3-0.6B instead of Qwen3-8B, `sdpa`, `tf32: false`, `sample_packing: false`, seq 512,
batch 1, `max_steps: 5`):

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python3 train_llm_axolotl.py --config config_sft_lora_smoke_mi355x.yaml --num-processes 1
```

**Expected output** (5/5 steps, finite loss dropping, LoRA kernels auto-patched on ROCm):

```
[axolotl.monkeypatch.lora_kernels] Patched attention class with LoRA optims: Qwen3Attention
{'loss': '1.692', 'grad_norm': '7.348', 'learning_rate': '0.0002', 'ppl': '5.432', ...}
{'loss': '0.08245', 'grad_norm': '1.043', 'learning_rate': '1.91e-05', 'ppl': '1.086', ...}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-mi355x
```

The adapter (`adapter_model.safetensors`) and checkpoints land in the output dir. Multi-GPU
LoRA is covered in section 5.

### Platform quirks (both vendors)

- **LoRA-kernel autopatch vs non-Qwen/Llama architectures.** Axolotl auto-enables its fused
  LoRA kernels and monkeypatches the attention QKV forward. On a model whose attention class
  does not match its regex — e.g. the hybrid conv+attention `lfm2` arch of
  `LiquidAI/LFM2.5-350M` — the patch aborts at load with
  `AssertionError: Original QKV code not found` (`axolotl/monkeypatch/lora_kernels.py`). Fix
  per axolotl's own warning: set `lora_mlp_kernel: false`, `lora_qkv_kernel: false`,
  `lora_o_kernel: false` in the YAML; plain PEFT LoRA then trains fine. Qwen bases
  (`Qwen3Attention`) match the patch and need no override.
- **Telemetry noise on restricted networks:** axolotl's posthog/HF telemetry emits
  `403 Forbidden` proxy warnings — harmless. `AXOLOTL_DO_NOT_TRACK=1
  HF_HUB_DISABLE_TELEMETRY=1` quiets it.
- **Datasets cache on a shared/network mount** can fail the `.arrow` write path — point
  `HF_DATASETS_CACHE` at tmpfs (`/dev/shm/...`) if you hit it.
- **Offline nodes:** with a populated `HF_HOME`, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  and use a locally cached `base_model`. `chat_template: tokenizer_default` works for any base
  that ships a template in `tokenizer_config.json` / `chat_template.jinja`.
- **Vendor deltas at a glance:** `tf32: true` is safe on CUDA and raises on ROCm;
  `flash_attention_2` needs a ~15 min nvcc source build on CUDA and is unavailable on ROCm
  (use `sdpa`); the torch force-reinstall and the bitsandbytes >= 0.50.0 bump are ROCm-only.

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
role in prompt construction. The extra columns in the sample are therefore ignored.

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

Notes:
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

### Multi-GPU

LoRA scales from 1 to N GPUs with **no new packages and no source build** —
`requirements_axolotl.txt` is unchanged, and the wrapper forwards `--num-processes` and the
passthrough `--main-process-port` to the axolotl CLI, which expands to
`accelerate launch --num-processes N --main-process-port <port> -m axolotl.cli.train <cfg>`.
Parallelism is **plain torch DDP** (accelerate multi-GPU, NCCL/RCCL) — no DeepSpeed and no
FSDP block in the config — so the upstream "DeepSpeed is broken on ROCm" caveat never
applies to LoRA. `config_sft_lora_smoke_mi355x_8gpu.yaml` is the worked 8-GPU example:

```bash
cd training/llm/axolotl
source .env_axolotl/bin/activate
# CRITICAL: if a 1-GPU run left a device pin at the end of .env_axolotl/bin/activate,
# sourcing the venv silently puts all N ranks on ONE GPU. Re-export after sourcing:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch;assert torch.cuda.device_count()==8"   # assert BEFORE training

python3 train_llm_axolotl.py \
    --config config_sft_lora_smoke_mi355x_8gpu.yaml \
    --num-processes 8 --main-process-port 29700
```

On a shared box, run the whole thing under a machine-wide mutex (`flock`) so the job owns
every GPU it claims, and give each concurrent training a unique `--main-process-port`.

**The dataset must be big enough to feed the global batch.** The shipped 10-row sample
cannot feed `micro_batch_size: 2` x 8 ranks = global batch 16. The 8-GPU config points at a
640-row replica; generate it once from this folder:

```bash
mkdir -p outputs/data
# awk, NOT cat: the shipped sample has no trailing newline, so a `cat` loop glues the
# last row of one copy onto the first row of the next, producing invalid JSON lines.
# `awk '{print}'` terminates every record it emits.
for i in $(seq 64); do awk '{print}' data/OTel_LLM_sample_10.jsonl; done \
    > outputs/data/otel_sample_x64.jsonl
wc -l outputs/data/otel_sample_x64.jsonl    # -> 640 (10 rows x 64), all valid JSON
```

Axolotl then applies `Dropping Invalid Sequences (<None or >512)` and keeps only the rows
that fit `sequence_len: 512`, so the 8-GPU smoke is a *pipeline* proof and its near-zero
loss is memorisation, not learning. For a real multi-GPU run raise `sequence_len` (or set
`long_sequences_strategy: truncate`) and use real data.

**Expected output** — finite, decreasing loss and a clean teardown on every rank:

```
{'loss': '1.695', 'grad_norm': '7.532', 'ppl': '5.444', 'epoch': '0.25'}
{'loss': '0.0004104', 'grad_norm': '0.009908', 'ppl': '1', 'epoch': '5'}
{'train_runtime': '35.87', 'train_loss': '0.1716', 'memory/max_allocated (GiB)': '3.17'}
[axolotl.train] Model successfully saved to ./outputs/sft-lora-smoke-mi355x-8gpu
```

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

## 8. Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 8 GPUs |
| Recipes | LoRA SFT, `sdpa` and `flash_attention_2` | LoRA SFT, `sdpa` only |
| PyTorch | plain PyPI (`torch 2.12.1+cu130`; no `--index-url`) | `download.pytorch.org/whl/rocm7.2`, force-reinstalled after axolotl |
| Config deltas | none (`tf32: true` is fine) | `attn_implementation: sdpa`, `tf32: false`, bitsandbytes >= 0.50.0 |

QLoRA, ZeRO-3 full fine-tuning, DPO, FSDP and multi-node are not covered on either vendor.
Base axolotl (no extras, no source build) is sufficient for everything above. Requirements:
Python >= 3.11, PyTorch >= 2.11.0.

## 9. Notes

- **The YAML is the program.** Axolotl builds the model, tokenizer, dataset pipeline,
  trainer and distributed strategy from one config; the same file is reused for
  `preprocess`, `train`, `inference` and `merge-lora`.
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

**Check these before relying on them:**
- `deepspeed_configs/zero2.json` and `zero3_bf16.json` are the filenames `axolotl fetch
  deepspeed_configs` is documented to write; confirm what actually lands in your working
  directory before uncommenting those lines.
- `transformer_layer_cls_to_wrap: Qwen3DecoderLayer` in the commented FSDP2 blocks must
  match the decoder-layer class name of the model you actually train.
- GRPO is not configured here: it needs a separate `axolotl vllm-serve` process, GPU
  partitioning between generation and training, and a local `rewards.py`.
