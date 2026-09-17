# `training/llm/torchtune` — LoRA continued pre-training with torchtune

Continued pre-training (domain-adaptive next-token prediction) with
[torchtune](https://github.com/pytorch/torchtune) `0.6.1`, the last stable API. This is the
**CPT example** for this repo: unstructured text, no chat template, loss on every token. Pick
a sibling recipe if you want SFT with prompt masking, DPO, or full-parameter training.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) at 1 and 8 GPUs · NVIDIA H100 80GB (CUDA 13.0).
No vendor-specific code change; keep `device: cuda` in the YAML on both, since ROCm torch
exposes HIP through the `torch.cuda` API.

## Files

- `train_llm_torchtune.py` — flattens the shipped sample, then execs `tune run`.
- `prepare_data_torchtune.py` — chat JSONL (`messages`) to `{text: ...}` JSONL.
- `config_cpt_lora.yaml` — single-device LoRA CPT config (`text_completion_dataset`).
- `config_cpt_lora_8gpu.yaml` — the distributed config; separate surface, see Run.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample.
- `requirements_torchtune.txt` — pinned 0.6.x environment.

## Setup

One venv for this folder. Install torch first, matched to the accelerator, then the rest.

`torchao==0.10.0` is mandatory on both vendors: torchtune 0.6.1 does
`import torchao.dtypes.nf4tensor` at import time and does not pull torchao itself, and a bare
`pip install torchao` grabs 0.18.x, which moved that module. Install it with `--no-deps` so
the pin cannot drag torch off its accelerator build.

Do not install `flash-attn` on either platform.

### NVIDIA (CUDA 13)

```bash
cd training/llm/torchtune
python3.12 -m venv .env_torchtune && source .env_torchtune/bin/activate

pip install torch numpy                 # the default stable wheel is already native cu130
pip install "torchtune==0.6.1" torchvision "python-dotenv>=1.0.1"
pip install "torchao==0.10.0" --no-deps
```

### AMD (ROCm 7.2)

`torchtune==0.6.1` is pure Python on top of torch and installs on a stable ROCm wheel — no
nightly, and no `--no-deps` on the torchtune install.

```bash
cd training/llm/torchtune
python3.12 -m venv .env_torchtune && source .env_torchtune/bin/activate

pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision
pip install torchtune==0.6.1 python-dotenv
python -c "import torchao;print(torchao.__version__)" || pip install "torchao==0.10.0" --no-deps
```

### Verify, weights, and secrets

pip can silently clobber torch during the torchtune install — re-check it:

```bash
python -c "import torchtune; print(torchtune.__version__)"
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 / 13.0
```

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN, needed for gated weight downloads
export OUTPUT_DIR=/path/to/outputs     # training artifacts / adapters
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache

tune download Qwen/Qwen2.5-7B --output-dir ./assets/Qwen2.5-7B     # the shipped default
tune download Qwen/Qwen2.5-0.5B --output-dir ./assets/Qwen2.5-0.5B # lighter smoke model
```

Edit `checkpointer.checkpoint_files` in `config_cpt_lora.yaml` if the downloaded shard names
differ (Qwen 7B is usually four `model-0000N-of-00004.safetensors`). Keep `assets/` and all
adapters out of the repo. Never commit a token.

## Data

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

torchtune's CPT path is `torchtune.datasets.text_completion_dataset`, which wants one
unstructured string per row:

```json
{"text": "Full document body, no chat structure, no special tokens."}
```

`prepare_data_torchtune.py` concatenates every turn's `content` into `text`, drops the extras,
and writes `data/otel_cpt.jsonl`; rows that already have `text` pass through unchanged. The
launcher runs it unless you pass `--skip-prepare`.

```bash
python3 prepare_data_torchtune.py          # data/OTel_LLM_sample_10.jsonl -> data/otel_cpt.jsonl
```

The embedding and reranker samples elsewhere in this repo (`anchor` / `positive` /
`negative_*`) are the wrong modality — do not point this trainer at them.

## Run

Run from inside `training/llm/torchtune/`: the YAML's `./assets`, `./data`, and `./outputs`
paths and `dev.env` resolve against the launch directory.

**The two recipes have different config surfaces and each needs its own YAML.**
`lora_finetune_distributed` reads `fsdp_cpu_offload` / `fsdp_reshard_after_forward` /
`custom_sharded_layers` and rejects single-device-only keys (`optimizer_in_bwd`, low-bit
optimizers). `config_cpt_lora.yaml` is single-device; `config_cpt_lora_8gpu.yaml` is
distributed. Do not cross them.

```bash
# inspect the command first
python3 train_llm_torchtune.py --dry-run

# single GPU
python3 train_llm_torchtune.py --recipe lora_finetune_single_device --nproc-per-node 1

# your own corpus (already {text} JSONL)
python3 train_llm_torchtune.py --skip-prepare --flat-file /data/domain.jsonl
```

Before any multi-GPU launch, re-export the device list *after* sourcing the venv — a stale
single-GPU pin in `activate` gives you 8 processes fighting over one card — and assert it took:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ROCm; never set either to ""
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"

nohup python3 train_llm_torchtune.py --nproc-per-node 8 \
  --config config_cpt_lora_8gpu.yaml \
  > train_llm_torchtune.log 2>&1 &

tail -f train_llm_torchtune.log
```

The 10-row sample cannot fill 8 ranks — replicate it (e.g. x256 to 2560 rows) outside the repo
and point `dataset.data_files` at it. Nothing is dropped for exceeding `max_seq_len`;
`text_completion_dataset` packs with `split_across_pack=True`.

On a shared box, pass a non-default rendezvous port (the launcher does not expose one, so
either `export MASTER_PORT=<port>` or call `tune run` directly), and hold a machine-wide lock
such as `flock /tmp/gpu8.lock` so nothing claims the cards mid-run:

```bash
tune run --nnodes 1 --nproc_per_node 8 --master_port 29690 \
  lora_finetune_distributed --config config_cpt_lora_8gpu.yaml \
  dataset.data_files=$OUTPUT_DIR/train_llm_torchtune/gpu8/otel_cpt_x256.jsonl \
  tokenizer.max_seq_len=2048 batch_size=2 gradient_accumulation_steps=1 \
  epochs=1 max_steps_per_epoch=80 save_adapter_weights_only=True \
  output_dir=./outputs/torchtune_cpt_lora_8gpu
```

To smoke-test on a smaller model than the 7B default, override the builder and tokenizer paths
rather than editing the YAML (`lora_qwen2_5_0_5b` is a first-class 0.6.1 builder; for 1.5B the
name is `lora_qwen2_5_1_5b_base`):

```bash
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

```
INFO:torchtune.utils._logging:Model is initialized with precision torch.bfloat16.
	GPU peak memory allocation: 1.32 GiB
1|1|Loss: 2.629014253616333    1|5|Loss: 2.685070514678955
INFO:torchtune.utils._logging:Adapter checkpoint of size 0.02 GiB saved to .../epoch_0/adapter_model.safetensors
```

Per-step loss is noisy on so few packed documents; read the trend per epoch.

## Arguments

### `train_llm_torchtune.py`

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config_cpt_lora.yaml` | torchtune YAML |
| `--recipe` | `lora_finetune_distributed` | `tune run` recipe; `lora_finetune_single_device` on one GPU |
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

### Key config fields

| Field | `config_cpt_lora.yaml` | Notes |
|---|---|---|
| `model` | `lora_qwen2_5_7b`, rank 16, alpha 32 | LoRA on q/v/output projections + MLP; the 8gpu config ships `lora_qwen2_5_0_5b` |
| `tokenizer.path` / `merges_file` | `./assets/Qwen2.5-7B/vocab.json` / `merges.txt` | From `tune download` |
| `tokenizer.max_seq_len` | `4096` | |
| `checkpointer.checkpoint_dir` | `./assets/Qwen2.5-7B` | Shards listed in `checkpoint_files` |
| `dataset` | `text_completion_dataset`, `data_files: ./data/otel_cpt.jsonl`, `packed: True` | `column: text`; the launcher overrides `data_files` with the absolute `--flat-file` path |
| `loss` | `torchtune.modules.loss.CEWithChunkedOutputLoss` | The 0.6.1 CE loss |
| `optimizer.lr` | `1.0e-4` | LoRA CPT LR; full FT would want much lower |
| `epochs` / `batch_size` / `gradient_accumulation_steps` | `1` / `1` / `8` | Effective batch 8 per GPU step cycle (8gpu config uses `batch_size: 2`) |
| `dtype` / `device` | `bf16` / `cuda` | |
| `output_dir` | `./outputs/torchtune_cpt_lora` | `./outputs/torchtune_cpt_lora_8gpu` in the 8gpu config |

## Output

`output_dir` feeds `checkpointer.output_dir`, `metric_logger.log_dir`, and the profiler, so
overriding it moves every artifact at once. It holds:

- adapter (or full) weights in HF-compatible shards, one per epoch (`epoch_0/1/2/...`)
- `logs/` from `DiskLogger`
- a recipe checkpoint if you left `save_adapter_weights_only: False`

Merge adapters with torchtune's checkpointer helpers, or load them with PEFT
(`../peft/merge_adapter.py` after converting).

## Notes

- Pass `save_adapter_weights_only=True` for smoke runs. With the default `False` the recipe
  writes a full model copy plus a recipe_state every epoch, and 0.6.1 has no switch to skip the
  end-of-epoch save entirely.
- Override `output_dir` to a path outside the repo (`$OUTPUT_DIR/...`) when the repo disk is
  small, and delete the artifacts afterwards.
- **CPT, not SFT.** `text_completion_dataset` + `packed: True` trains next-token prediction on
  every token — no chat template, no prompt masking.
- **LoRA, not full FT.** Full-parameter CPT needs `full_finetune_distributed` and a much lower
  LR.
- **Qwen2.5 tokenizer paths** are `vocab.json` + `merges.txt` from the HF snapshot, not a
  `tokenizer.model`. Llama configs want the SentencePiece file — swap the whole `tokenizer:`
  block if you change families.
- `custom_sharded_layers` is deliberately unset in the 8gpu config: Qwen2.5-0.5B ties its
  embedding/output weights, so `output` is not a separately shardable module.
