# `training/llm/llamafactory` — config-driven post-training with LLaMA-Factory

One YAML per recipe, run by `llamafactory-cli`.
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) covers continued pre-training, SFT,
reward modelling, PPO, DPO, KTO, ORPO and SimPO, each with full-parameter, freeze, LoRA or QLoRA
tuning (2/3/4/8-bit via bitsandbytes / HQQ / EETQ / GPTQ / AWQ), plus GaLore, APOLLO, BAdam, Muon,
LoRA+, DoRA and PiSSA. Pick it when you want the widest menu of algorithms behind one flag
(`stage:` + `finetuning_type:`), a large library of built-in chat templates, and a dataset
registry that keeps data wiring out of the code.

**Hardware:** NVIDIA H100 (CUDA 13.0) · AMD MI355X (gfx950, ROCm 7.2.4), 1 and 8 GPUs. On both,
set `flash_attn: sdpa` rather than the shipped `fa2` unless you build flash-attn yourself — see
Setup. The shipped non-smoke configs target a single node of 8x H100 80GB.

## Files

- `train_llm_llamafactory.py` — launcher: loads `dev.env`, validates the YAML and the dataset
  registry, shells out to `llamafactory-cli`.
- `config_sft_lora.yaml` — LoRA SFT on chat JSONL (the default recipe, Qwen3-8B).
- `config_sft_lora_smoke_mi355x.yaml` / `config_sft_lora_smoke_h100.yaml` — single-GPU smoke
  variants (Qwen3-0.6B, `flash_attn: sdpa`).
- `config_sft_lora_smoke_mi355x_8gpu.yaml` — 8-rank DDP variant.
- `config_sft_qlora.yaml` — QLoRA SFT (4-bit bitsandbytes base + LoRA).
- `config_sft_full.yaml` — full-parameter SFT with DeepSpeed ZeRO-3.
- `config_dpo_lora.yaml` — DPO preference tuning with LoRA.
- `data/dataset_info.json` — the dataset registry. A custom file is invisible until declared here.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample registered as `otel_chat_sft`.
- `data/dpo_sample.jsonl` — 10-row **synthetic** preference sample registered as `otel_chat_dpo`.
- `requirements_llamafactory.txt` — dependency list plus install notes.

## Setup

Install from a **source checkout**, not the PyPI wheel: `config_sft_full.yaml` references
`examples/deepspeed/ds_z3_config.json`, which exists only in the git tree. One venv per recipe
folder.

```bash
cd training/llm/llamafactory
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded by train_llm_llamafactory.py
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

`dev.env` holds `HF_TOKEN=hf_...` and is git-ignored at the repo root. Never commit tokens.
Upstream also accepts `hf_hub_token:` in the YAML — do not use it, that puts the secret in a
tracked file.

### NVIDIA (CUDA)

```bash
python3 -m venv .env_llamafactory && source .env_llamafactory/bin/activate
pip install torch numpy            # -> the current CUDA 13 build; NO --index-url
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .env_llamafactory/LLaMA-Factory
pip install -e .env_llamafactory/LLaMA-Factory
pip install python-dotenv                         # launcher dep, not pulled by upstream
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130
```

The editable install adds matching `torchvision 0.28.0+cu130` / `torchaudio 2.11.0+cu130` and
leaves torch intact; re-verify anyway. Optional upstream extras from inside the checkout:
`pip install -r requirements/metrics.txt` (BLEU/ROUGE eval) and
`pip install -r requirements/deepspeed.txt` (needed by `config_sft_full.yaml`).

Docker ships flash-attn prebuilt and is the lower-friction alternative:

```bash
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

**Set `flash_attn: sdpa` unless you build flash-attn from source.** The prebuilt wheel installs
but fails to import against torch 2.13/cu130 (`undefined symbol:
_ZN3c104impl3cow23materialize_cow_storage...` — built against an older libc10 ABI). SDPA trains
fine; `fa2` works if you rebuild from source or use the upstream image.

### AMD (ROCm)

Install torch, torchvision and torchaudio from the ROCm index **before** `pip install -e .` —
upstream depends on all three, and pip will otherwise pull CUDA wheels over your ROCm torch.

```bash
cd training/llm/llamafactory
python3 -m venv .env_llamafactory && source .env_llamafactory/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .env_llamafactory/LLaMA-Factory
pip install -e .env_llamafactory/LLaMA-Factory
pip install python-dotenv
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.13.0+rocm7.2 AMD Instinct MI355X
```

Upstream also ships a ROCm compose file:

```bash
cd LLaMA-Factory/docker/docker-rocm/ && docker compose up -d
docker compose exec llamafactory bash
```

**Never `pip install flash-attn` on ROCm** — set `flash_attn: sdpa` in the YAML. Do not set a
`tf32:` key in any config; the ROCm guard raises on it, and these recipes are `bf16: true`
regardless. Treat `quantization_method: bnb` as CUDA-first, though the stock bitsandbytes 0.50.0
wheel does load its ROCm binary on gfx950.

The checkout lives inside the git-ignored venv dir to keep `git status` clean. For
`config_sft_full.yaml`, `deepspeed:` resolves relative to the **current working directory** — run
that recipe from inside the checkout (passing this folder's config by absolute path), or make that
key absolute. Everything else runs fine from this folder.

Verify either install with:

```bash
llamafactory-cli help
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

## Data

Chat JSONL, one conversation per line — what LLaMA-Factory calls the **OpenAI format**, treated
upstream as a special case of `sharegpt`:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

A custom file is invisible until registered in `dataset_info.json` inside `dataset_dir`:

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

The config then says `dataset: otel_chat_sft` and `dataset_dir: ./data`, and `file_name` resolves
relative to `dataset_dir`, so this folder is self-contained. Parsing is driven entirely by the
registry's `columns` mapping and the per-message `tags`, so the sample's extra columns (`unmask`,
`flow`, `source_id`, `source_repo`, `source_spec_id`, `source_version`) are ignored.

`data/dpo_sample.jsonl` is **synthetic**: `chosen` is the real assistant answer and `rejected` a
truncated variant labeled `[TRUNCATED — synthetic rejected sample for smoke testing only]`. It
makes `config_dpo_lora.yaml` runnable end to end but encodes no real human preference — replace it
before any meaningful DPO run. It is registered as `otel_chat_dpo` with `"ranking": true`, one row
per pair:

```json
{"messages": [{"role": "user", "content": "..."}],
 "chosen": {"role": "assistant", "content": "..."},
 "rejected": {"role": "assistant", "content": "..."}}
```

`chosen` / `rejected` are single message **objects**, not lists. The launcher refuses to start a
`stage: dpo` or `stage: rm` run against a dataset not marked `ranking: true`.

To use your own data: drop the JSONL into `data/`, add or change `file_name` in
`data/dataset_info.json`, and reference the registered name from the config's `dataset:` key.

- Allowed types: json, jsonl, csv, parquet, arrow. Use `hf_hub_url` instead of `file_name` for a
  Hub dataset.
- An optional `system` message may be the first element of `messages` — that is what
  `system_tag: system` enables. `user` and `assistant` turns must alternate.
- `template:` must match the model family (`qwen3_nothink`, `llama3`, `gemma3`, `deepseek3`, ...)
  and must be the **same** template you use at inference, or output quality degrades.
- For very large datasets set `tokenized_path: <dir>`: the first run writes the tokenized dataset
  there and later runs load it instead of re-tokenizing.

## Run

Validate without touching a GPU, then smoke-run:

```bash
python3 train_llm_llamafactory.py --config config_sft_lora.yaml --validate-only

export CUDA_VISIBLE_DEVICES=0                      # plus HIP_VISIBLE_DEVICES=0 on ROCm
export MASTER_PORT=29663
export HF_DATASETS_CACHE=/dev/shm/dscache_llamafactory
python3 train_llm_llamafactory.py --config config_sft_lora_smoke_mi355x.yaml --num-gpus 1
```

Full run:

```bash
nohup python3 train_llm_llamafactory.py \
  --config config_sft_lora.yaml \
  --num-gpus 8 \
  > train_llm_llamafactory.log 2>&1 &

tail -f train_llm_llamafactory.log
```

Swap the config for `config_sft_qlora.yaml`, `config_sft_full.yaml` or `config_dpo_lora.yaml`.
Unrecognized arguments are forwarded verbatim, and upstream accepts `key=value` overrides:
`... --config config_sft_lora.yaml learning_rate=1e-5 logging_steps=1`.

Quote any `key=value` override whose value YAML would coerce: `eval_strategy=no` parses as boolean
`False` and crashes transformers' `IntervalStrategy`. The smoke configs quote it as
`eval_strategy: "no"`.

Loss should be finite and drifting down. For DPO you also get `rewards/chosen`,
`rewards/rejected` and `rewards/accuracies` — accuracy above 0.5 is the signal that preference
training is doing something. `CUDA out of memory` means lower `per_device_train_batch_size` or
`cutoff_len`, or move to ZeRO-3.

### Multi-GPU

`llamafactory-cli` auto-detects `get_device_count() > 1` and re-execs itself under `torchrun`, so
LoRA gets plain DDP (no DeepSpeed, no FSDP) for free. There is no `--nproc_per_node` to pass —
upstream's `src/llamafactory/launcher.py` builds the torchrun command itself from
`NPROC_PER_NODE` (default: device count) and `MASTER_PORT` (default: a random free port).
`train_llm_llamafactory.py` only forces `FORCE_TORCHRUN=1` for multi-GPU **full** fine-tuning,
which LoRA does not need.

```bash
cd training/llm/llamafactory
source .env_llamafactory/bin/activate
# CRITICAL: if bin/activate ends with a HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES pin left
# over from a 1-GPU smoke, sourcing it silently pins you to ONE GPU. Override after:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29710 NPROC_PER_NODE=8
python3 -c "import torch; assert torch.cuda.device_count()==8"   # assert before training
python3 train_llm_llamafactory.py \
  --config config_sft_lora_smoke_mi355x_8gpu.yaml --num-gpus 8
```

Ten rows cannot feed 8 ranks: 8 ranks x batch 4 x accum 2 = 64 samples per optimizer step.
Replicate the sample x256 into `data/` — already registered as `otel_chat_sft_x256`:

```bash
# awk normalises the trailing newline the shipped sample lacks - a plain `cat` loop would
# glue the last row of each copy onto the first of the next and emit invalid JSON.
awk '{print}' data/OTel_LLM_sample_10.jsonl > /tmp/one.jsonl
for i in $(seq 256); do cat /tmp/one.jsonl; done > data/otel_chat_sft_x256.jsonl
wc -l data/otel_chat_sft_x256.jsonl        # -> 2560
```

The generated file is a build artifact — do not commit it. The 8-GPU config differs from the
1-GPU one only in `max_steps: 20`, `per_device_train_batch_size: 4`,
`gradient_accumulation_steps: 2`, `preprocessing_num_workers: 8`, `dataloader_num_workers: 2`,
`save_strategy: "no"` and the x256 dataset. `save_strategy: "no"` still leaves a final
`adapter_model.safetensors` — HF `Trainer` saves at the end of `do_train` regardless.

Do not combine QLoRA with ZeRO-3: a quantized base cannot be ZeRO-3 sharded. Upstream ships a
separate FSDP+QLoRA recipe under `examples/extras/fsdp_qlora/`.

### Merging a LoRA adapter

Write a small export config (`model_name_or_path`, `adapter_name_or_path`, `template`,
`finetuning_type: lora`, `export_dir`), then:

```bash
python3 train_llm_llamafactory.py --config config_export.yaml --task export
```

Do **not** merge against a quantized base or with `quantization_bit` set.

## Arguments

| Arg | Meaning |
|---|---|
| `--config` | YAML recipe to run (required) |
| `--task` | `train` (default), `export` (merge LoRA), `chat`, `api` |
| `--gpus` | Sets `CUDA_VISIBLE_DEVICES`, e.g. `0,1,2,3`. Default: all visible GPUs |
| `--num-gpus` | GPU count for the summary and the torchrun decision (default 8) |
| `--force-torchrun` | Always set `FORCE_TORCHRUN=1` (automatic for multi-GPU full FT) |
| `--validate-only` | Check config + dataset registry, print the command, exit |

Anything else on the command line passes through to `llamafactory-cli` unchanged.

Key config fields (source of truth: `src/llamafactory/hparams/` upstream):

| Field | Meaning |
|---|---|
| `model_name_or_path` | Hub repo id or local path of the model to train |
| `stage` | `pt`, `sft`, `rm`, `ppo`, `dpo`, `kto` |
| `finetuning_type` | `lora`, `oft`, `freeze`, `full` |
| `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target` | LoRA shape (`lora_target: all` = all linear modules; `lora_alpha` defaults to `lora_rank * 2`) |
| `quantization_bit`, `quantization_method`, `quantization_type` | QLoRA: 4/8-bit, `bnb`/`hqq`/`eetq`, `nf4`/`fp4` |
| `pref_beta`, `pref_loss`, `pref_ftx` | DPO family: beta, loss variant (`sigmoid`/`ipo`/`orpo`/`simpo`/...), optional SFT-loss mix |
| `dataset`, `dataset_dir`, `eval_dataset` | Registered dataset name(s); the folder holding `dataset_info.json` |
| `template` | Chat template — must match the model |
| `cutoff_len`, `packing`, `neat_packing` | Max tokens per sample; sequence packing (`neat_packing` also blocks cross-attention). Upstream decrements `cutoff_len` by 1 when packing is on |
| `train_on_prompt`, `mask_history` | Unmask the prompt / train only on the last turn. Mutually exclusive |
| `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs` | Batch schedule (effective batch = per-device x accum x GPUs) |
| `learning_rate`, `lr_scheduler_type`, `warmup_ratio` | Optimization |
| `bf16`, `pure_bf16`, `flash_attn`, `enable_liger_kernel` | Precision and kernels (`flash_attn`: `auto`/`disabled`/`sdpa`/`fa2`/`fa3`) |
| `deepspeed` | Path to a ZeRO json (`examples/deepspeed/ds_z{0,2,3}_config.json`), resolved against the CWD |
| `val_size`, `eval_strategy`, `eval_steps` | Held-out split and eval cadence (mutually exclusive with `eval_dataset`) |
| `output_dir`, `save_steps`, `save_total_limit`, `plot_loss` | Checkpointing and the loss plot |
| `report_to` | `none`, `wandb`, `tensorboard`, `swanlab`, `mlflow` |

## Output

Everything lands under the config's `output_dir` (`saves/<model>/<method>/<stage>`):

- `checkpoint-<step>/` — periodic checkpoints (`save_steps` / `save_total_limit`).
- Final artifacts at the top level: a PEFT adapter (`adapter_model.safetensors` +
  `adapter_config.json`) for LoRA/QLoRA, or full `model-*.safetensors` shards for full
  fine-tuning, plus tokenizer files.
- `trainer_log.jsonl` — per-logging-step metrics.
- `training_args.yaml` / `all_results.json` / `train_results.json` — resolved config and final
  metrics.
- `training_loss.png` — written when `plot_loss: true`.
- `running_log.txt` — the framework's own log.
- Merged weights land in the `export_dir` of your export config, not in `output_dir`.

## Notes

- **The YAML is the program.** `stage` picks the algorithm, `finetuning_type` picks how many
  parameters move, and the rest are HF `Seq2SeqTrainingArguments` plus LLaMA-Factory's own
  `ModelArguments` / `DataArguments` / `FinetuningArguments`.
- **Masking.** Default SFT supervises assistant turns only. `train_on_prompt: true` supervises the
  prompt too; `mask_history: true` trains only on the final turn.
- **DPO reference model.** With LoRA the adapter-disabled base acts as the reference, so no second
  copy of the weights is loaded — which is why the DPO recipe here uses LoRA. Set `ref_model:` for
  an explicit reference; `pref_loss: orpo` / `simpo` need no reference at all.
- PPO, KTO and reward modelling are supported upstream but are not configured here — they need
  extra pieces (a reward model, `kto_tag` data). See `examples/train_lora/` upstream.
- Multi-node is supported upstream (`FORCE_TORCHRUN=1 NNODES=... NODE_RANK=... MASTER_ADDR=...`)
  but is not configured here.
