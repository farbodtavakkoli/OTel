# `training/llm/peft` — chat SFT with standalone Hugging Face PEFT

Chat-model SFT with Hugging Face [PEFT](https://github.com/huggingface/peft) used directly
- no Unsloth wrapper, no DeepSpeed. Covers LoRA, QLoRA (bitsandbytes 4-bit nf4 base), DoRA
(`--use_dora`) and rsLoRA (`--use_rslora`), built from a plain `LoraConfig` +
`get_peft_model` and trained by the stock HF `Trainer`.

Pick this folder when the model fits on one GPU: QLoRA on a single card is the intended
small-scale path, and 4-bit weights cannot be sharded by ZeRO-3 or FSDP2 anyway. The
multi-GPU route here is plain DDP (one full replica per GPU), not sharding - for sharded
training use `../fsdp/` or `../deepspeed/`.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4), 1 GPU and 8 GPUs under DDP · NVIDIA H100 80GB
(CUDA 13.0), 1 GPU. bf16 LoRA, QLoRA, DoRA and `merge_adapter.py` all work on both.

## Files

| File | Purpose |
|---|---|
| `train_llm_peft.py` | Training entrypoint (LoRA / QLoRA / DoRA / rsLoRA) |
| `merge_adapter.py` | Merges a trained adapter into base weights via `merge_and_unload()` |
| `requirements_peft.txt` | Pinned dependencies (Python 3.12) |
| `data/OTel_LLM_sample_10.jsonl` | 10-row chat sample; the default `--train_file` |

## Setup

Python 3.12, one venv per recipe folder. Install `torch` first, matched to your
accelerator, then the requirements - a later `bitsandbytes`/`peft` install can silently
downgrade torch, so re-verify afterwards.

### NVIDIA (CUDA 13)

```bash
python3.12 -m venv .env_peft && source .env_peft/bin/activate
pip install torch==2.11.0 numpy          # the PyPI wheel is the CUDA 13 build
pip install -r requirements_peft.txt     # every pin installs unchanged
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

To pin the `+cu130` local version tag explicitly, use
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.

### AMD (ROCm 7.2)

torch 2.11.0 ROCm wheels live on the `rocm7.2` index (the older `rocm6.4` index stops at
torch 2.9.1).

```bash
python3.12 -m venv .env_peft && source .env_peft/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_peft.txt     # bitsandbytes 0.50.0 from plain PyPI
```

The one plain-PyPI `bitsandbytes==0.50.0` wheel covers both vendors and auto-selects its
backend (`libbitsandbytes_rocm72.so` on ROCm 7.2 / gfx950,
`libbitsandbytes_cuda130.so` on CUDA 13), so QLoRA needs no build step on either. The ROCm
binaries target CDNA gfx90a/gfx942/gfx950 and current RDNA; on an unlisted card either
build from source with `-DCOMPUTE_BACKEND=hip` or drop `--load_in_4bit` and train plain
bf16 LoRA - nothing else in this folder needs bitsandbytes.

Do not install `flash-attn` on ROCm. On CUDA it is optional and not in the requirements: it
is a from-source nvcc build (`pip install flash-attn --no-build-isolation`, `CUDA_HOME`
exported) and only then can you pass `--attn_implementation flash_attention_2`. The default
`sdpa` needs nothing.

### Verify

```bash
python -c "import torch, peft, transformers, bitsandbytes; \
print('peft', peft.__version__, '| transformers', transformers.__version__, '| bnb', bitsandbytes.__version__)"
```

### Secrets

`HF_TOKEN` for gated models comes from `dev.env` in this folder, loaded by
`load_dotenv("dev.env")` - so run both scripts from inside `training/llm/peft/`:

```bash
ln -sf ../../../dev.env dev.env   # HF_TOKEN, for gated checkpoints
```

`dev.env` is git-ignored at the repo root; never commit a token.

## Data

`--train_file` takes a chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Only the `messages` key is read, so the shipped sample's extra columns (`unmask`, `flow`,
`source_id`, ...) are ignored and your own data may include or omit them. A `system` turn
and extra turns are fine. Each row is rendered with the tokenizer's own
`apply_chat_template`, so a model whose tokenizer has no chat template is rejected up front.

Loss is completion-only by default (`--mask_prompt`): every token outside an assistant turn
is set to `-100`, and in a multi-turn row all assistant turns are supervised. Pass
`--no_mask_prompt` to train on the full rendered sequence. Rows longer than `--max_seq_len`
and rows with no supervised token are dropped, never truncated; both counts are logged.

## Run

There is no `--max_steps`: step count is `rows / (batch_size x grad_acc_steps)` per epoch.

Smoke test - one GPU, the shipped sample (10 rows, 9 usable at `--max_seq_len 2048`):

```bash
python train_llm_peft.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --num_train_epochs 1 --logging_steps 1 \
  --output_dir ./peft_smoke
```

QLoRA on a single GPU (4-bit nf4 base + bf16 adapter), from inside `training/llm/peft/`:

```bash
nohup python train_llm_peft.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./qlora_run \
  --load_in_4bit --optim paged_adamw_8bit \
  --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear \
  --batch_size 2 --grad_acc_steps 8 --num_train_epochs 3 --learning_rate 2e-4 \
  --gradient_checkpointing \
  > train_llm_peft.log 2>&1 &

tail -f train_llm_peft.log
```

Pin a single card on a shared node with `export CUDA_VISIBLE_DEVICES=<n>` (plus
`HIP_VISIBLE_DEVICES=<n>` on ROCm); torch then sees it as device 0.

bf16 LoRA across 8 GPUs (DDP - one replica per GPU, so the model must fit on one card):

```bash
source .env_peft/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch; assert torch.cuda.device_count()==8"

nohup torchrun --nproc_per_node=8 --master_port 29690 train_llm_peft.py \
  --train_file /path/to/train.jsonl \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./lora_run \
  --lora_r 32 --lora_alpha 64 --use_rslora \
  --batch_size 2 --grad_acc_steps 4 --num_train_epochs 3 --learning_rate 2e-4 \
  --gradient_checkpointing --ddp_find_unused_parameters \
  > train_llm_peft.log 2>&1 &
```

`accelerate launch --num_processes 8 --mixed_precision bf16` is equivalent to the
`torchrun` line above. Multi-GPU specifics:

- **Pass `--ddp_find_unused_parameters` whenever `all-linear` meets a multimodal base**
  (e.g. gemma-4): its vision/audio tower adapters get no gradient from a text-only batch
  and DDP fails the first backward with `Expected to have finished reduction in the prior
  iteration...`. The alternative is an explicit text-only `--lora_target_modules` list.
  Single-GPU runs have no DDP wrapper and never hit this.
- Re-export both GPU masks after activating the venv and assert the device count; a stale
  `export CUDA_VISIBLE_DEVICES=3` in `.env_peft/bin/activate` silently trains on one card.
- Under DDP prefer `--optim adamw_torch`; the paged optimizer's unified-memory paging costs
  more with 8 ranks on one node than it saves for a small adapter. Keep
  `paged_adamw_8bit` for the 1-GPU memory-tight case.
- **Watch GPU 0 when scaling QLoRA.** With `--load_in_4bit` every rank also allocates on
  GPU 0, so rank 0 carries a much larger allocation - the pattern that OOMs rank 0 while
  the other GPUs look idle. It comes from the bitsandbytes 4-bit path, not the optimizer.
  bf16 LoRA under DDP is flat across ranks, so it is the safe first multi-GPU step.
- Size the dataset to the world size: the shipped 10-row sample gives ~1 step/epoch at
  world size 8. Replicate it (e.g. x64 -> 640 rows) outside the repo and point
  `--train_file` there.

Merge the adapter into base weights when training finishes:

```bash
python merge_adapter.py \
  --adapter ./qlora_run/final_adapter \
  --output_dir ./qlora_run/merged_model
```

## Arguments

`train_llm_peft.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat JSONL, one `{"messages": [...]}` per line |
| `--model_name` | (required) | HF repo id or local path; tokenizer must have a chat template |
| `--output_dir` | `./peft_run` | Checkpoints and `final_adapter/` land here |
| `--resume_from_checkpoint` | `""` | Checkpoint dir to resume from (used only if it exists) |
| `--max_seq_len` | `4096` | Token cap per row; longer rows are dropped |
| `--max_samples` | `None` | Hard cap on rows loaded |
| `--eval_samples` | `0` | Rows held out for a per-epoch eval (0 = off) |
| `--mask_prompt` / `--no_mask_prompt` | on | Completion-only loss vs full-sequence loss |
| `--lora_r` | `32` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha; 2x rank is a good default |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_target_modules` | `all-linear` | `all-linear` (QLoRA-style) or a comma-separated module list |
| `--modules_to_save` | `None` | Extra modules trained in full and saved with the adapter (e.g. `embed_tokens,lm_head`) |
| `--use_dora` | off | DoRA: decompose the update into magnitude + direction (better at low rank, slower) |
| `--use_rslora` | off | rsLoRA: scale by `alpha/sqrt(r)` instead of `alpha/r` (stabler at high rank) |
| `--load_in_4bit` | off | QLoRA: load the frozen base in 4-bit nf4 via bitsandbytes |
| `--bnb_4bit_quant_type` | `nf4` | 4-bit data type (`nf4` or `fp4`) |
| `--no_double_quant` | (double quant on) | Disable nested quantization of the quantization constants |
| `--batch_size` | `2` | Per-device train/eval batch size |
| `--grad_acc_steps` | `8` | Gradient accumulation steps |
| `--num_train_epochs` | `3.0` | Epochs |
| `--learning_rate` | `2e-4` | Peak LR (LoRA/QLoRA typically 1e-4..2e-4) |
| `--lr_scheduler_type` | `cosine` | LR scheduler |
| `--warmup_ratio` | `0.03` | Fraction of total steps spent warming up |
| `--weight_decay` | `0.0` | Weight decay |
| `--optim` | `adamw_torch` | HF optimizer id; `paged_adamw_8bit` cuts optimizer memory with QLoRA |
| `--logging_steps` | `10` | Log metrics every N steps |
| `--save_total_limit` | `2` | Max checkpoints kept |
| `--seed` | `42` | Random seed |
| `--gradient_checkpointing` | off | Recompute activations; needed for long sequences on one GPU |
| `--ddp_find_unused_parameters` | off | Multi-GPU: tolerate adapter params that get no gradient in a step |
| `--attn_implementation` | `sdpa` | `sdpa`, `flash_attention_2` (CUDA only), `eager` |

`merge_adapter.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--adapter` | (required) | Trained adapter dir (e.g. `<output_dir>/final_adapter`) |
| `--base_model` | `None` | Base model id/path; defaults to `base_model_name_or_path` in `adapter_config.json` |
| `--output_dir` | (required) | Where the merged, HF-ready model is written |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` - the precision the merge happens in |
| `--device_map` | `cpu` | `cpu` (safest) or `auto` if the full model fits in HBM |
| `--max_shard_size` | `5GB` | Shard size for the saved safetensors files |

## Output

Under `--output_dir`:

- `checkpoint-<step>/` - per-epoch Trainer checkpoints (adapter + optimizer + scheduler),
  capped by `--save_total_limit`; pass one back via `--resume_from_checkpoint`.
- `final_adapter/` - `adapter_model.safetensors`, `adapter_config.json` (which records the
  base model), plus tokenizer files. Load with
  `PeftModel.from_pretrained(base_model, "<path>/final_adapter")`. Adapter-sized, not
  model-sized: tens to hundreds of MB.
- `merged_model/` - only from `merge_adapter.py`: a standalone directory loadable with
  `AutoModelForCausalLM.from_pretrained(...)`.
- `runs/` - TensorBoard event files; view with `tensorboard --logdir <output_dir>/runs`.

## Notes

- `target_modules="all-linear"` is the default because that is the QLoRA recipe - adapt
  every linear layer rather than only `q_proj`/`v_proj`, with no per-architecture module
  list to maintain.
- `--load_in_4bit` keeps the base frozen and quantized and trains only the adapter; pair it
  with `--optim paged_adamw_8bit` for the full memory win on one GPU.
- DoRA adds runtime overhead, so merge the weights before serving. rsLoRA is worth reaching
  for when raising rank stops helping. The two are independent and can be combined.
- Chat-template masking assumes prefix-stable chat templates (true for mainstream instruct
  models); labels outside assistant spans are `-100`.
- **Merging:** `merge_adapter.py` reloads the base in full precision - never 4-bit, since
  bitsandbytes layers cannot absorb a LoRA delta (quantize the merged model afterwards if
  you need a quantized artifact). Keep the original `final_adapter/`: after merging you no
  longer have a swappable adapter. aLoRA-style adapters cannot be merged at all.
- A base that does not fit on one card needs `../fsdp/` or `../deepspeed/`, and neither
  shards 4-bit weights - so QLoRA belongs here.
