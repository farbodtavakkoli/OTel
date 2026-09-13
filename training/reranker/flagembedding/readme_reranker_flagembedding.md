# `train_reranker_flagembedding.py`

## Overview & when to use

Reranker fine-tuner that drives **FlagEmbedding** (BAAI) — the training code the BGE rerankers
are built with. One script covers three reranker families through `--reranker_type`:

| `--reranker_type` | Architecture | Example base model |
|---|---|---|
| `encoder` | cross-encoder, classification head over `[CLS]` | `BAAI/bge-reranker-base`, `bge-reranker-v2-m3` |
| `llm` | **decoder-only LLM reranker** — scores a (query, passage) pair from the LM head's yes/no logit, trained with LoRA | `BAAI/bge-reranker-v2-gemma`, `Qwen/Qwen3-0.6B` |
| `llm_layerwise` | LLM reranker with per-layer scoring heads (early exit) | `BAAI/bge-reranker-v2-minicpm-layerwise` |

The script is a thin, repo-style wrapper: `argparse` exposes every tunable, `utils.build_argv`
translates them into the `HfArgumentParser` dataclasses FlagEmbedding expects, and the matching
`Runner` does the training.

This folder covers the reranker side only; for embedding training see
[`../../embedding/sentence_transformers`](../../embedding/sentence_transformers).

FlagEmbedding's finetune runner emits no metrics and selects no best checkpoint — use
`../sentence_transformers` when you want an nDCG number out of the box, and this folder when
you want the LLM-reranker architecture or the BGE group-wise recipe.

## Install

Python 3.12, in its own venv. The commands below refer to a few machine-specific
locations through environment variables — set them to suit your machine:

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

If the root filesystem is tight, put the venv outside the repo on a larger volume:

```bash
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
python3 -m venv $DATA_DIR/envs/.env_flagembedding
source $DATA_DIR/envs/.env_flagembedding/bin/activate
```

### AMD (ROCm)

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> 2.11.0+rocm7.2
pip install -r requirements_reranker_flagembedding.txt
ln -sf ../../../dev.env dev.env
```

Never install `flash-attn` here — the pinned build is CUDA-only. `utils.resolve_attn()` keeps
FlagEmbedding's `--use_flash_attn` at `False` whenever `torch.version.hip` is set, and the model
falls back to PyTorch SDPA.

### NVIDIA (CUDA)

Skip the ROCm index line — the default PyPI torch wheels are CUDA builds. On a CUDA 13 host
`torch==2.11.0` resolves to a native `+cu130` wheel on plain PyPI, so no `--index-url` is
needed. Install torch first, then the requirements file:

```bash
python3 -m venv $DATA_DIR/envs/.env_flagembedding
source $DATA_DIR/envs/.env_flagembedding/bin/activate
pip install torch==2.11.0 numpy    # -> 2.11.0+cu130
pip install -r requirements_reranker_flagembedding.txt   # pins transformers==4.57.1 (<5, on purpose)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-verify: FlagEmbedding must not have clobbered torch
ln -sf ../../../dev.env dev.env
```

`flash-attn==2.8.3` is optional (needs the CUDA toolkit) and only matters for the `llm` /
`llm_layerwise` decoder paths. `utils.resolve_attn()` returns `flash_attention_2` on CUDA
only if `import flash_attn` succeeds, else `sdpa`.

Verify:

```bash
python -c "import FlagEmbedding, transformers, torch; print(transformers.__version__, torch.__version__)"
```

If Hub egress is blocked, run fully offline with a populated cache
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`). `BAAI/bge-reranker-v2-m3`
(`XLMRobertaForSequenceClassification`, 568M) is a valid stand-in for the script default
`BAAI/bge-reranker-base` if the latter is not cached.


## Environment & secrets

`dev.env` is symlinked to the repo root (`ln -sf ../../../dev.env dev.env`) and loaded with
`load_dotenv("dev.env")`. It supplies `HF_TOKEN` for gated models. The BGE and Qwen defaults
used here are ungated. Never print or commit the token.

Point `HF_HOME` at a shared cache so the 5 GB gemma weights are not re-downloaded:

```bash
export HF_HOME=/path/to/hf_cache
```

or pass `--hf_home`.

## Data + converter

FlagEmbedding does not read the repo's `anchor` / `positive` / `negative_N` schema. It wants:

```json
{"query": "...", "pos": ["..."], "neg": ["...", "..."], "prompt": "..."}
```

`convert_data.py` does the mapping and is committed with its output, per repo convention:

```bash
python convert_data.py     # ../sentence_transformers/OTel_reranker_sample_100.jsonl
                           # -> OTel_reranker_flagembedding_100.jsonl (100 rows)
```

| Flag | Default | Meaning |
|---|---|---|
| `--src` | `../sentence_transformers/OTel_reranker_sample_100.jsonl` | Source JSONL |
| `--dst` | `OTel_reranker_flagembedding_100.jsonl` | Converted output |
| `--n_neg` | `5` | Negatives to carry over |
| `--prompt` | `Predict whether passage B contains an answer to query A.` | Value stored in the `prompt` field (used by the LLM rerankers) |
| `--max_chars` | `2000` | Per-text truncation |

`OTel_reranker_flagembedding_100.jsonl` is the shipped sample and the default `--train_data`,
so the folder runs with no arguments. The source file's extra `answer` column is dropped.

## Run

Smoke test (encoder, single GPU, shipped sample):

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=1 --master_port=29811 train_reranker_flagembedding.py --epochs 1
```

Smoke test (LLM reranker, 2 GPUs):

```bash
torchrun --nproc_per_node=2 --master_port=29811 train_reranker_flagembedding.py \
  --reranker_type llm --model_name_or_path BAAI/bge-reranker-v2-gemma \
  --epochs 1 --batch_size 1 --train_group_size 4 --gradient_checkpointing
```

Full run:

```bash
nohup torchrun --nproc_per_node=2 --master_port=29811 train_reranker_flagembedding.py \
  --reranker_type llm --model_name_or_path BAAI/bge-reranker-v2-gemma \
  --train_data /path/to/your_train.jsonl \
  --output_dir $OUTPUT_DIR/train_reranker_flagembedding/run1 \
  --batch_size 4 --train_group_size 8 --epochs 2 --lr 1e-5 --max_len 1024 \
  --gradient_checkpointing --save_merged_lora_model True \
  > train_reranker_flagembedding.log 2>&1 &
```

**What "working" looks like:** `trainable params: ...` once per rank (LLM types), then
`{'loss': ...}` every `--logging_steps`, a `{'train_runtime': ...}` summary, and a model or
adapter written under `--output_dir`. There is no metric line — this trainer does not evaluate.

### Expected output

Single GPU, encoder reranker, shipped 100-row sample, `--batch_size 2 --train_group_size 4`.
`floor(100 / (2 x 1 x 1)) = 50` steps/epoch — `train_group_size` sets passages per query and
does not change the step count. Six epochs gives 300 steps and a clean convergence curve:

```
[flagembedding] 1x <device> (torch <version>)
[flagembedding] type=encoder model=BAAI/bge-reranker-v2-m3
{'loss': 0.7468, 'grad_norm': 0.1779,   'learning_rate': 1.67e-05, 'epoch': 1.0}
{'loss': 0.1038, 'grad_norm': 0.000145, 'learning_rate': 1.34e-05, 'epoch': 2.0}
{'loss': 0.0130, 'grad_norm': 0.000441, 'learning_rate': 1.01e-05, 'epoch': 3.0}
{'loss': 0.0013, 'grad_norm': 0.00187,  'learning_rate': 6.67e-08, 'epoch': 6.0}
{'train_runtime': 65.91, 'train_samples_per_second': 9.10, 'train_loss': 0.1074, 'epoch': 6.0}
Training complete. Model saved to .../smoke1gpu_enc_6ep
```

Per-step loss is noisy because negatives are resampled each epoch — read the epoch-boundary
trend. Saved output: `config.json`, `model.safetensors`, `sentencepiece.bpe.model`,
tokenizer files, `checkpoint-<steps>/`, `training_args.bin`, `runs/`.

For the LLM reranker path, `trainable params: 7,372,800` (LoRA only) is printed once per
rank and the saved artifacts are `adapter_config.json` + `adapter_model.safetensors` rather
than a full model.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name_or_path` | `BAAI/bge-reranker-base` | Base reranker model |
| `--reranker_type` | `encoder` | `encoder`, `llm`, or `llm_layerwise` |
| `--train_data` | `OTel_reranker_flagembedding_100.jsonl` | Training JSONL (query/pos/neg) |
| `--output_dir` | `output` | Checkpoints and final model/adapter |
| `--batch_size` | `2` | Per-device batch (queries per step) |
| `--train_group_size` | `4` | Passages per query: 1 positive + N-1 negatives |
| `--query_max_len` | `128` | Max query tokens |
| `--passage_max_len` | `256` | Max passage tokens |
| `--max_len` | `512` | Max combined length (LLM rerankers) |
| `--epochs` | `1.0` | Training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--warmup_steps` | `0` | LR warmup steps (transformers dropped `warmup_ratio` in 5.x) |
| `--grad_accum` | `1` | Gradient accumulation steps |
| `--knowledge_distillation` | `False` | Use `pos_scores`/`neg_scores` teacher columns |
| `--query_instruction` | `None` | Query instruction prefix (gemma wants `"A: "`) |
| `--passage_instruction` | `None` | Passage instruction prefix (gemma wants `"B: "`) |
| `--use_lora` | `True` | LoRA training (LLM types) |
| `--lora_rank` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--target_modules` | `q_proj k_proj v_proj o_proj` | LoRA target modules |
| `--save_merged_lora_model` | `False` | Merge the adapter into the base model on save |
| `--start_layer` | `None` | First scored layer (`llm_layerwise`) |
| `--head_multi` | `None` | One head per layer (`llm_layerwise`) |
| `--head_type` | `None` | Head type (`llm_layerwise`) |
| `--attn_implementation` | `auto` | `auto` (sdpa on ROCm), `sdpa`, `eager`, `flash_attention_2` |
| `--bf16` | on | bf16 training |
| `--gradient_checkpointing` | off | Gradient checkpointing |
| `--gc_use_reentrant` | `False` | Reentrant checkpointing — must stay `False` under DDP |
| `--save_strategy` | `epoch` | `no`, `epoch`, `steps` |
| `--save_total_limit` | `1` | Checkpoints kept |
| `--logging_steps` | `5` | Log every N steps |
| `--dataloader_drop_last` | `True` | Drop the last partial batch |
| `--seed` | `42` | Random seed |
| `--report_to` | `tensorboard` | Trainer reporting backend |
| `--deepspeed` | `None` | DeepSpeed config JSON |
| `--cache_dir` | `None` | Model cache override |
| `--cache_path` | `None` | Tokenized-dataset cache dir |
| `--trust_remote_code` | off | Trust remote code on load |
| `--hf_home` | `None` | `HF_HOME` override |
| `--extra_args` | `None` | Raw flags forwarded verbatim to FlagEmbedding |

## Output

Under `--output_dir`:

- `--reranker_type encoder`: `config.json`, `model.safetensors`, tokenizer files — a plain HF
  sequence-classification folder.
- `--reranker_type llm` / `llm_layerwise`: `adapter_config.json`, `adapter_model.safetensors`
  (a PEFT adapter over the base model) plus tokenizer files. Pass
  `--save_merged_lora_model True` for a standalone merged model.
- `checkpoint-<step>/` per `--save_strategy`, capped by `--save_total_limit`.
- `training_args.bin`, `runs/` (TensorBoard).

**No evaluation output.** For nDCG@10 and best-checkpoint selection use
`../sentence_transformers/`, or FlagEmbedding's separate `FlagEmbedding.evaluation`
entrypoints.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Verified | H100 80GB (Hopper cc 9.0), CUDA 13.0, 1 GPU | MI355X 288GB (gfx950), ROCm 7.2.4, 1 and 2 GPUs |
| PyTorch | `torch==2.11.0` resolves to a `+cu130` wheel on plain PyPI — no `--index-url` | `torch 2.11.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2` |
| Reranker types | `encoder` | `encoder`, and `llm` with LoRA at 1 and 2 GPUs |
| Attention | `flash_attention_2` when `flash_attn` imports, else `sdpa` | `sdpa` (`--use_flash_attn` forced off) |

No hardware-specific code change is needed on either vendor.

- `--reranker_type llm_layerwise` requires a **MiniCPM base**. On anything else it fails with
  `AttributeError: 'Qwen3Config' object has no attribute 'scale_depth'` — FlagEmbedding's
  layerwise modeling code is hardwired to that architecture. Use
  `BAAI/bge-reranker-v2-minicpm-layerwise`.

## Notes

- **`transformers<5` is a hard requirement for this folder.** FlagEmbedding 1.4.0's reranker
  dataset (`AbsDataset.create_one_example`) calls `tokenizer.prepare_for_model()`, which
  transformers 5.x removed from the standard tokenizers; the run dies on the **first batch**
  with `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model`. Setting
  `use_fast=False` does not help — 5.x drops it from the slow tokenizer path too. Keep the
  `transformers==4.57.1` pin.
- **`Trainer.tokenizer` shim.** `utils.patch_transformers5_compat()` does two things:
  it wraps `Trainer.__init__` so the `tokenizer=` keyword FlagEmbedding's reranker runners
  still pass is remapped to `processing_class`, and it aliases the `Trainer.tokenizer`
  attribute back onto `processing_class` so FlagEmbedding's `_save()` keeps working. Leave
  it in place if you raise the `transformers` pin.
- **Gradient checkpointing + DDP.** With `--gradient_checkpointing` on 2 ranks, reentrant
  checkpointing raises `RuntimeError: Expected to mark a variable ready only once`. The script
  therefore always passes `gradient_checkpointing_kwargs={"use_reentrant": false}`
  (`--gc_use_reentrant False`, the default). Flip it only on a single GPU.
- `tf32` is forced to `None` when `torch.version.cuda is None` — setting it on ROCm raises.
- No `flash-attn`: `--use_flash_attn` stays `False` on ROCm and the model uses SDPA.
- Default master port is **29811** (29500 often collides on a shared box).
- **Batch geometry.** `steps/epoch = floor(rows / (--batch_size × world_size × --grad_accum))`.
  The 100-row sample at `--batch_size 2` gives 50 steps on 1 GPU. Push `--batch_size` too high
  on a small dataset and the step count silently collapses toward zero — the same constraint
  documented in the sentence-transformers siblings.
- Harmless warnings: `destroy_process_group() was not called before program exit`,
  `expandable_segments not supported on this platform`, and the `Trainer.tokenizer is now
  deprecated` line from the shim.
- Point `--output_dir` at a volume with room for checkpoints (`$OUTPUT_DIR` above); the LLM
  reranker checkpoints are multi-GB.
- The shipped OTel sample is for pipeline validation only — 100 rows will not produce a useful
  reranker.
