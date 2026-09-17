# `training/llm/axolotl` — config-driven post-training with Axolotl

Config-driven post-training with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl):
one YAML describes the model, dataset, algorithm and distributed strategy, and the
`axolotl` CLI runs it. It covers full fine-tuning, LoRA, QLoRA, QAT, preference tuning
(DPO, IPO, KTO, ORPO, SimPO), online RL (GRPO, GDPO) and reward modelling over DDP /
DeepSpeed ZeRO / FSDP2.

Pick this over the hand-written trainers (`../deepspeed/`, `../unsloth/`) to sweep recipes
by editing YAML instead of code, and to get multipack sample packing and the newer
attention/LoRA kernels without wiring them yourself.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 1 and 8 GPUs, LoRA SFT with `sdpa` · NVIDIA
H100 80GB (CUDA 13.0), LoRA SFT with `sdpa` and `flash_attention_2`.

## Files

| File | Purpose |
|---|---|
| `train_llm_axolotl.py` | Thin launcher: loads `dev.env`, validates the YAML, shells out to `axolotl train` |
| `config_sft_lora.yaml` | LoRA SFT on chat JSONL (the default recipe) |
| `config_sft_qlora.yaml` | QLoRA SFT (4-bit frozen base + LoRA) |
| `config_sft_full.yaml` | Full-parameter SFT with DeepSpeed ZeRO-3 |
| `config_dpo_lora.yaml` | DPO preference tuning with LoRA |
| `config_sft_lora_smoke_mi355x.yaml`, `..._mi355x_8gpu.yaml` | Small ROCm smoke variants (1 and 8 GPU) |
| `config_sft_lora_smoke_h100.yaml`, `..._h100_sdpa.yaml` | Small CUDA smoke variants (flash-attn / sdpa) |
| `data/OTel_LLM_sample_10.jsonl` | 10-row chat sample the SFT configs point at |
| `data/dpo_sample.jsonl` | 10-row synthetic preference sample for `config_dpo_lora.yaml` |
| `requirements_axolotl.txt` | Launcher deps plus the required install order |

## Setup

Python 3.12 recommended (>= 3.11 required), PyTorch >= 2.11. The install is order-sensitive
in both paths: **torch first, then axolotl with `--no-build-isolation`, then this folder's
launcher deps.**

### NVIDIA (CUDA 13)

Axolotl is uv-first upstream:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv
export UV_TORCH_BACKEND=cu130                       # match your CUDA toolkit (cu128 also works)

uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.12.0 torchvision                 # step 1: torch BEFORE axolotl
uv pip install --no-build-isolation axolotl[deepspeed]   # step 2: builds against that torch
uv pip install -r requirements_axolotl.txt               # step 3: launcher deps

axolotl fetch deepspeed_configs   # writes deepspeed_configs/*.json into the cwd
axolotl fetch examples            # optional: upstream reference configs
```

`config_sft_full.yaml` references `deepspeed_configs/zero3_bf16.json`, so run
`axolotl fetch deepspeed_configs` from this folder (or fix the path in the YAML).

Plain `pip` in a venv also works where PyPI already serves native cu130 wheels:

```bash
python3 -m venv .env_axolotl && source .env_axolotl/bin/activate
pip install torch numpy                      # step 1: torch FIRST (cu130 wheel from PyPI)
pip install packaging ninja
pip install --no-build-isolation axolotl     # step 2: base axolotl, no extras
pip install -r requirements_axolotl.txt      # step 3: launcher deps

python -c "import torch; print(torch.__version__, torch.version.cuda)"   # -> 2.12.1+cu130 13.0
axolotl --help
```

`pip install axolotl` replaces torch (it pins `torch==2.12.1`); on a CUDA host that pin also
resolves to a `+cu130` wheel, so no recovery reinstall is needed - re-verify as above.

The `[deepspeed]`, `[flash-attn]` and `[vllm]` extras are optional: base axolotl is enough
for single-GPU LoRA/QLoRA and for DDP multi-GPU LoRA. `[deepspeed]` is only needed for
`config_sft_full.yaml`.

**flash-attn on CUDA** has no prebuilt wheel on the default PyPI index, so
`attn_implementation: flash_attention_2` needs a source build against the on-box `nvcc`:

```bash
export CUDA_HOME=/usr/local/cuda            # nvcc 13.0
MAX_JOBS=16 pip install --no-build-isolation flash-attn==2.8.3
```

`sdpa` is the zero-build fallback and trains the same recipe unchanged. On Hopper, axolotl
also offers Flash Attention 4 (`pip install flash-attn-4`). Docker ships flash-attn
prebuilt: `docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest`.

### AMD / ROCm 7.2

Upstream's only AMD walkthrough is a manual source build; it is not required. A plain
`pip install axolotl` trains LoRA SFT once the torch wheel is put back and two config keys
are overridden.

```bash
cd training/llm/axolotl
python3 -m venv .env_axolotl && source .env_axolotl/bin/activate

# step 1: ROCm torch (axolotl will re-pin it in step 2)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# step 2: base axolotl - NO extras: [deepspeed]/[flash-attn]/[vllm] pull CUDA-only bits.
pip install packaging ninja
pip install --no-build-isolation axolotl

# step 3: CRITICAL - step 2 silently REPLACES torch with the CUDA wheel (axolotl pins
# torch==2.12.1 and pip resolves it from PyPI). Put the ROCm build back, or you train
# nothing: CUDA torch cannot see the AMD GPUs.
pip install --force-reinstall --no-deps torch==2.12.1 --index-url https://download.pytorch.org/whl/rocm7.2

# step 4: axolotl pins bitsandbytes==0.49.1, whose wheel has no ROCm-7.2 binary
# (libbitsandbytes_rocm72.so not found; fatal for QLoRA). 0.50.0 ships it.
pip install -U bitsandbytes==0.50.0

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.12.1+rocm7.2  AMD Instinct MI355X
```

Config overrides required on ROCm:

- **`attn_implementation: flash_attention_2` -> `sdpa`** in every YAML. Do not
  `pip install flash-attn` on ROCm - the PyPI package is a CUDA-only source build.
- **`tf32: true` -> `false`** in every YAML; tf32 raises on ROCm.
- For sharded runs replace the `deepspeed:` key in `config_sft_full.yaml` with the
  commented `fsdp_version: 2` block - DeepSpeed is reported broken with Axolotl on ROCm.
  LoRA needs neither; DDP is enough.

Base axolotl still drags in CUDA-only `xformers` and `nvidia-*-cu13` wheels as transitive
deps; leave them alone and do not enable xformers attention, which is incompatible with
ROCm.

### Secrets

`HF_TOKEN` for gated models comes from `dev.env` in this folder, loaded by
`load_dotenv("dev.env")` before anything else so the `axolotl` child process inherits it.
The launcher warns and continues when it is unset.

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

## Data

Chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The SFT configs point at `./data/OTel_LLM_sample_10.jsonl`. Its extra columns (`unmask`,
`flow`, `source_id`, `source_repo`, `source_spec_id`, `source_version`) play no role: the
`chat_template` strategy reads only the column named by `field_messages` and the keys in
`message_property_mappings`.

DPO rows are one preference pair per line, with `chosen` / `rejected` as single message
**objects**, not lists - that is what the `chat_template.default` DPO strategy expects:

```json
{"messages": [{"role": "user", "content": "..."}],
 "chosen": {"role": "assistant", "content": "..."},
 "rejected": {"role": "assistant", "content": "..."}}
```

`./data/dpo_sample.jsonl` is synthetic - `rejected` is a truncated variant of the real
answer, labeled as such. It exists so `config_dpo_lora.yaml` runs end to end; replace it
before any meaningful DPO run.

To use your own data, point `datasets[].path` at your file (a local path, an HF repo id, or
`s3://` / `gs://`). The launcher refuses to start when a local dataset path does not exist.
The dataset block that does the wiring:

```yaml
datasets:
  - path: ./data/OTel_LLM_sample_10.jsonl   # local file, HF repo id, s3:// or gs://
    ds_type: json                  # required when `path` is a file
    split: train
    type: chat_template            # the OpenAI-style messages strategy
    field_messages: messages       # column holding the turn list
    message_property_mappings:
      role: role
      content: content
    roles_to_train: ["assistant"]  # loss only on assistant turns
    train_on_eos: turn             # train on the terminator of each trainable turn
chat_template: tokenizer_default   # use the template in tokenizer_config.json
```

- `chat_template: tokenizer_default` errors if the tokenizer has no template; use
  `tokenizer_default_fallback_chatml`, or a named template (`gemma`, `llama3`, `qwen3`).
- If the chat template ends a turn with a token that is not `tokenizer.eos_token`, declare
  it under `eot_tokens:` - otherwise the terminator is never trained and the model will not
  learn to stop.
- For large datasets, tokenize once up front so later runs skip it:
  `python3 train_llm_axolotl.py --config config_sft_lora.yaml --task preprocess`.

## Run

```bash
python3 train_llm_axolotl.py --config config_sft_lora.yaml --validate-only   # no GPU needed
python3 train_llm_axolotl.py --config config_sft_lora.yaml --num-processes 1 # tiny real run
```

Full run - the axolotl CLI spawns the workers itself (accelerate under the hood), so there
is no `accelerate launch` or `torchrun` in the command line.

```bash
source .env_axolotl/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # AMD; on NVIDIA set only CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch;assert torch.cuda.device_count()==8"   # assert BEFORE training

nohup python3 train_llm_axolotl.py \
  --config config_sft_lora.yaml \
  --num-processes 8 --main-process-port 29700 \
  > train_llm_axolotl.log 2>&1 &

tail -f train_llm_axolotl.log
```

Swap the config for `config_sft_qlora.yaml`, `config_sft_full.yaml` or
`config_dpo_lora.yaml`. Unrecognized flags are forwarded verbatim to the `axolotl` CLI, so
config fields can be overridden on the command line (e.g. `--learning-rate 1e-5`).

LoRA scales from 1 to N GPUs with no new packages and no source build; parallelism is plain
torch DDP (NCCL/RCCL). `config_sft_lora_smoke_mi355x_8gpu.yaml` is the worked 8-GPU example.

- Re-export both GPU masks after activating the venv and assert the device count; a device
  pin left in `.env_axolotl/bin/activate` by a 1-GPU run silently puts all N ranks on one
  GPU.
- Give each concurrent training a unique `--main-process-port` (the default 29500 collides).
- **The dataset must be big enough to feed the global batch.** The shipped 10-row sample
  cannot feed `micro_batch_size: 2` x 8 ranks. Replicate it with `awk`, not `cat` - the
  sample has no trailing newline, so a `cat` loop glues the last row of one copy onto the
  first of the next and produces invalid JSON:

```bash
mkdir -p outputs/data
for i in $(seq 64); do awk '{print}' data/OTel_LLM_sample_10.jsonl; done \
    > outputs/data/otel_sample_x64.jsonl
wc -l outputs/data/otel_sample_x64.jsonl    # -> 640
```

Axolotl then drops rows that do not fit `sequence_len`, so raise `sequence_len` (or set
`long_sequences_strategy: truncate`) for a real multi-GPU run. A `CUDA out of memory`
traceback means lower `micro_batch_size` or `sequence_len`, or move to a higher ZeRO stage.

Merge a LoRA adapter back into the base model:

```bash
python3 train_llm_axolotl.py --config config_sft_lora.yaml \
  --task merge-lora --lora-model-dir ./outputs/sft-lora
```

The merged weights land in `<output_dir>/merged`.

## Arguments

Launcher flags:

| Arg | Meaning |
|---|---|
| `--config` | YAML recipe to run (required) |
| `--task` | `train` (default), `preprocess` (tokenize only), `merge-lora` |
| `--num-processes` | Training processes = GPUs on this node (default 8) |
| `--deepspeed` | Override the config's DeepSpeed json for this run |
| `--lora-model-dir` | Adapter directory, required for `--task merge-lora` |
| `--validate-only` | Check the config, print the command, exit without training |

Anything else is passed through to the `axolotl` CLI unchanged.

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

## Output

Everything lands under the config's `output_dir`, always relative to
`training/llm/axolotl` (e.g. `./outputs/sft-lora`):

- `checkpoint-<step>/` - periodic checkpoints (`saves_per_epoch` / `save_total_limit`).
- Final weights at the top level of `output_dir`: a PEFT adapter
  (`adapter_model.safetensors` + `adapter_config.json`) for LoRA/QLoRA, or full
  `model-*.safetensors` shards for full fine-tuning, plus tokenizer files and the resolved
  chat template.
- `merged/` - only after `--task merge-lora`.
- `dataset_prepared_path` (`./last_run_prepared`, or a per-recipe variant in the smoke
  configs) holds the tokenized arrow cache. **Delete it if you change the data or the
  template**, otherwise the stale cache is reused.
- Add `use_tensorboard: true` or the `wandb_*` fields to the YAML for a metrics backend.

## Notes

- **LoRA-kernel autopatch vs non-Qwen/Llama architectures.** Axolotl auto-enables its fused
  LoRA kernels and monkeypatches the attention QKV forward. On a model whose attention class
  does not match its regex - e.g. the hybrid conv+attention `lfm2` arch of
  `LiquidAI/LFM2.5-350M` - load aborts with `AssertionError: Original QKV code not found`.
  Set `lora_mlp_kernel: false`, `lora_qkv_kernel: false`, `lora_o_kernel: false` in the
  YAML. Qwen bases (`Qwen3Attention`) match the patch and need no override.
- **Masking.** `roles_to_train: ["assistant"]` plus `train_on_eos: turn` is standard
  completion-only SFT. Set `train_on_inputs: true` to supervise the whole sequence.
- **Sharding choices.** LoRA/QLoRA fit per-GPU, so DDP or ZeRO-2 is usually fastest; full
  fine-tuning uses `deepspeed: deepspeed_configs/zero3_bf16.json`. DeepSpeed, FSDP and DDP
  are mutually exclusive - set one. Upstream uses `use_reentrant: true` with ZeRO-3 and
  `false` elsewhere, which is why the configs differ on that key.
- If you uncomment an FSDP2 block, `transformer_layer_cls_to_wrap` must match the
  decoder-layer class name of the model you actually train.
- **DPO reference model.** With an adapter, TRL uses the adapter-disabled base as the
  implicit reference, so no second copy of the weights is loaded - that is why the DPO
  config uses LoRA. Set `rl_adapter_ref_model: true` to force a real reference model.
- Quiet telemetry `403 Forbidden` proxy warnings on restricted networks with
  `AXOLOTL_DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1`.
- Point `HF_DATASETS_CACHE` at tmpfs (`/dev/shm/...`) when the model cache is on a shared or
  network mount that rejects the `.arrow` write.
- For offline nodes with a populated `HF_HOME`, set `HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1` and use a locally cached `base_model`.
