# `training/reranker/flagembedding` — reranker fine-tuner on FlagEmbedding (BGE)

`train_reranker_flagembedding.py` drives **FlagEmbedding**, the training code the BGE
rerankers are built with. `--reranker_type` selects one of three families:

| `--reranker_type` | Architecture | Example base model |
|---|---|---|
| `encoder` | cross-encoder, classification head over `[CLS]` | `BAAI/bge-reranker-base`, `bge-reranker-v2-m3` |
| `llm` | decoder-only LLM reranker — scores a (query, passage) pair from the LM head's yes/no logit, trained with LoRA | `BAAI/bge-reranker-v2-gemma`, `Qwen/Qwen3-0.6B` |
| `llm_layerwise` | LLM reranker with per-layer scoring heads (early exit) | `BAAI/bge-reranker-v2-minicpm-layerwise` |

This runner emits no metrics and selects no best checkpoint. Use
[`../sentence_transformers`](../sentence_transformers) when you want an nDCG@10 / MRR@10
number out of the box; use this folder for the LLM-reranker architecture or the BGE
group-wise recipe.

**Hardware:** NVIDIA H100 80GB (CUDA 13.0, 1 GPU, `encoder`) and AMD MI355X 288GB
(ROCm 7.2.4, 1 and 2 GPUs, `encoder` plus `llm` with LoRA). No vendor-specific code change.

## Files

- `train_reranker_flagembedding.py` — trainer entry point; `utils.py` builds the argv for
  FlagEmbedding's dataclasses and holds the transformers-5 shim.
- `convert_data.py` — repo triplet JSONL -> FlagEmbedding JSONL.
- `requirements_reranker_flagembedding.txt` — pinned deps.
- `OTel_reranker_flagembedding_100.jsonl` — 100-row sample, the default `--train_data`.

## Setup

Python 3.12, one venv for this recipe. Put it on a larger volume if the root filesystem is
tight:

```bash
export DATA_DIR=/path/to/data          # venv + pip cache location
export OUTPUT_DIR=/path/to/outputs     # training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache; keeps the 5GB gemma weights shared
export PIP_CACHE_DIR=$DATA_DIR/pip_cache

cd training/reranker/flagembedding
python3 -m venv $DATA_DIR/envs/.env_flagembedding
source $DATA_DIR/envs/.env_flagembedding/bin/activate
```

Keep the `transformers==4.57.1` pin: FlagEmbedding 1.4.0's reranker dataset calls
`tokenizer.prepare_for_model()`, which transformers 5.x removed, so under 5.x the run dies
on the first batch.

### AMD (ROCm 7.2)

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> 2.11.0+rocm7.2
pip install -r requirements_reranker_flagembedding.txt
ln -sf ../../../dev.env dev.env
```

Do **not** install `flash-attn` on ROCm.

### NVIDIA (CUDA 13.0)

Skip the ROCm index — `torch==2.11.0` resolves to a native `+cu130` wheel on plain PyPI:

```bash
pip install torch==2.11.0 numpy    # -> 2.11.0+cu130
pip install -r requirements_reranker_flagembedding.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-verify: not clobbered
ln -sf ../../../dev.env dev.env
```

`flash-attn==2.8.3` is optional (needs the CUDA toolkit) and only affects the `llm` /
`llm_layerwise` paths.

Verify either install:

```bash
python -c "import FlagEmbedding, transformers, torch; print(transformers.__version__, torch.__version__)"
```

`dev.env` supplies `HF_TOKEN` for gated models; the BGE and Qwen defaults here are ungated.
With a populated cache, run fully offline with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`;
`BAAI/bge-reranker-v2-m3` (568M) is a valid stand-in for the default
`BAAI/bge-reranker-base` when that one is not cached.

## Data

FlagEmbedding does not read the repo's `anchor` / `positive` / `negative_N` schema. It
wants:

```json
{"query": "...", "pos": ["..."], "neg": ["...", "..."], "prompt": "..."}
```

`convert_data.py` does the mapping and ships its output:

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

`OTel_reranker_flagembedding_100.jsonl` is the default `--train_data`, so the folder runs
with no arguments. The source file's extra `answer` column is dropped. 100 rows validate
the pipeline; they will not produce a useful reranker.

## Run

Default master port is 29811 (29500 often collides on a shared box).

Smoke test — encoder, single GPU, shipped sample:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=1 --master_port=29811 train_reranker_flagembedding.py --epochs 1
```

Smoke test — LLM reranker, 2 GPUs:

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

Expected shape — encoder, single GPU, 100-row sample at `--batch_size 2`, 6 epochs. There
is no metric line; this trainer does not evaluate. Per-step loss is noisy because negatives
are resampled each epoch, so read the epoch-boundary trend:

```
[flagembedding] 1x <device> (torch <version>)
{'loss': 0.7468, 'grad_norm': 0.1779,  'learning_rate': 1.67e-05, 'epoch': 1.0}
{'loss': 0.0013, 'grad_norm': 0.00187, 'learning_rate': 6.67e-08, 'epoch': 6.0}
{'train_runtime': 65.91, 'train_samples_per_second': 9.10, 'train_loss': 0.1074, 'epoch': 6.0}
```

On the `llm` types, `trainable params: 7,372,800` (LoRA only) prints once per rank.

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
| `--warmup_steps` | `0` | LR warmup steps |
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
| `--gc_use_reentrant` | `False` | Reentrant checkpointing — keep `False` under DDP |
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

Under `--output_dir`, plus `checkpoint-<step>/` per `--save_strategy` (capped by
`--save_total_limit`), `training_args.bin` and `runs/` (TensorBoard). Point it at a volume
with room — LLM reranker checkpoints are multi-GB.

- `encoder`: `config.json`, `model.safetensors`, tokenizer files — a plain HF
  sequence-classification folder.
- `llm` / `llm_layerwise`: `adapter_config.json`, `adapter_model.safetensors` (a PEFT
  adapter over the base model) plus tokenizer files. Pass `--save_merged_lora_model True`
  for a standalone merged model.

**No evaluation output.** For nDCG@10 and best-checkpoint selection use
`../sentence_transformers/` or FlagEmbedding's `FlagEmbedding.evaluation` entrypoints.

## Notes

- `--reranker_type llm_layerwise` requires a **MiniCPM base** such as
  `BAAI/bge-reranker-v2-minicpm-layerwise`. On anything else FlagEmbedding's layerwise
  modeling code fails with `AttributeError: 'Qwen3Config' object has no attribute
  'scale_depth'`.
- Keep `--gc_use_reentrant False` (the default) whenever you run `--gradient_checkpointing`
  on more than one rank; reentrant checkpointing raises `RuntimeError: Expected to mark a
  variable ready only once` under DDP.
- Batch geometry: `steps/epoch = floor(rows / (--batch_size x world_size x --grad_accum))`;
  `--train_group_size` sets passages per query and does not change the step count. The
  100-row sample at `--batch_size 2` gives 50 steps on 1 GPU. Push `--batch_size` too high
  on a small dataset and the step count silently collapses toward zero.
- If you raise the `transformers` pin, leave `utils.patch_transformers5_compat()` in place —
  it remaps the `tokenizer=` keyword FlagEmbedding passes to `Trainer`.
