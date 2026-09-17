# `training/llm/redhat` — OSFT continual fine-tuning via Red Hat's training_hub

Continual fine-tuning with **OSFT** (Orthogonal Subspace Fine-Tuning) through Red Hat's
[`training_hub`](https://github.com/Red-Hat-AI-Innovation-Team/training_hub), single node,
multi-GPU. OSFT adapts a base or instruct model to a new domain — e.g. observability / OTel
telemetry — **without catastrophic forgetting**, so pick it over the SFT recipes in this repo
when you need to add domain knowledge without degrading general capability and without replay
or data mixing. Works with any HF causal LM that training_hub's dependencies support.

`train_llm_redhat.py` wraps `training_hub.osft` with argument logging, an optional live
speed/ETA monitor, and an EOS-override staging step.

**Hardware:** AMD MI355X (ROCm 7.2.4, `torch==2.11.0+rocm7.2`) at 2 and 8 GPUs · NVIDIA H100
80GB (CUDA 13.0, `torch==2.11.0+cu130`). Same code on both; the install differs and ROCm needs
two extra steps.

## Files

- `train_llm_redhat.py` — the OSFT training entrypoint.
- `speed_monitor.py` — live speed/ETA reporter (`--speed-steps N`).
- `check_memory.py` — post-hoc run summary (peak memory, step, duration, loss plot).
- `data/OTel_LLM_sample_10.jsonl` — 10-row sample, the default `--data-path`.
- `requirements_redhat.txt` — this folder's direct deps (training_hub manages the rest).

Python 3.12+, one venv for this folder.

## Setup

### NVIDIA (CUDA 13)

Install sequentially — the `[grpo]` extras constrain torch/vllm/transformers and can conflict
with `[cuda]` if solved together. Expect `training-hub[grpo,lora]` to pull torch back to
`2.11.0+cu130`; keep it, that respects training_hub's pins.

```bash
cd training/llm/redhat
python3.12 -m venv .env_redhat && source .env_redhat/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch torchvision
pip install training-hub[grpo,lora]
pip install training-hub[cuda] --no-build-isolation   # builds flash-attn via nvcc; the slow step
pip install "kernels>=0.12,<0.13"                     # REQUIRED, see below
pip install -r requirements_redhat.txt
```

**The `kernels` pin is not optional.** `training-hub[cuda]` transitively pulls `kernels 0.16.0`,
but `transformers 5.5.0` pins `kernels<0.13,>=0.12.0`, and 0.16 rejects the
`LayerRepository(...)` call transformers makes with
`ValueError: Either a revision or a version must be specified` — crashing *every* transformers
import. `0.12.3` restores it.

flash-attn builds and engages on CUDA, so **do not** set `TESTING=true` here; mini-trainer
selects `flash_attention_2` on its own. To skip the nvcc build entirely, use the SDPA route
instead: `pip install "training-hub[grpo,lora]"` + `pip install "liger-kernel>=0.5.10"` +
`export TESTING=true`, exactly as on ROCm.

### AMD / ROCm 7.2

Skip the `[cuda]` extra — it exists to build CUDA flash-attn. Never pip-install flash-attn on
ROCm.

```bash
cd training/llm/redhat
python3 -m venv .env_redhat && source .env_redhat/bin/activate
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install "training-hub[grpo,lora]"      # keeps the ROCm torch (it requires only torch>=2.6)
pip install "liger-kernel>=0.5.10"         # REQUIRED: the script sets use_liger=True, and
                                           # upstream ships liger only in the [cuda] extra
pip install -r requirements_redhat.txt
```

**Export `TESTING=true` before every run on ROCm.** mini-trainer 0.8.1 hard-requires
`import flash_attn` for standard causal LMs and only falls back to SDPA when `TESTING=true`.
PyTorch ROCm's SDPA uses AOTriton flash attention on gfx950.

### Both platforms

```bash
ln -sf ../../../dev.env dev.env                 # HF_TOKEN, for gated models
export OUTPUT_DIR=/path/to/outputs              # checkpoints / training artifacts
export HF_HOME=/path/to/hf_cache                # Hugging Face model cache
export HF_DATASETS_CACHE=/dev/shm/hf_datasets   # keep .arrow writes off a network mount
export RDZV_ENDPOINT=127.0.0.1:29647            # move torchrun off the default :29500

python -c "from training_hub import osft; print('training_hub OK')"
```

`HF_DATASETS_CACHE` is not optional when `HF_HOME` is on a shared network mount: `datasets`
writes `cache-*.arrow` under `$HF_HOME/datasets/...` and dies with
`OSError: [Errno 1] Operation not permitted`. `HF_HOME` itself can stay on the mount for
read-only model loads.

`train_llm_redhat.py` loads `dev.env` before importing training_hub, since
transformers/datasets resolve tokens and cache locations at import time. Never commit a token.

## Data

`--data-path` is a JSONL file of chat conversations — one `{"messages": [...]}` per line, the
format training_hub/instructlab expects. The bundled sample carries extra metadata columns
(`flow`, `source_repo`, `source_id`, `source_spec_id`, `source_version`, `unmask`);
training_hub keys on `messages` and tolerates the extras.

`--unmask-messages` (default on) trains on all turns; `--no-unmask-messages` gives standard SFT
on assistant turns only.

**EOS caveat.** instructlab keys label-unmasking on the tokenizer's `eos_token`. If the chat
template closes the assistant turn with a *different* token, the terminator is never unmasked
and the model never learns to stop. Gemma 4 closes turns with `<turn|>`, so pass
`--eos-token "<turn|>"`; the script stages a copy of the model with the corrected EOS (weights
symlinked, only tokenizer/config regenerated) and trains from that, leaving the base model
untouched. Models whose template terminator already equals `eos_token` — e.g. LFM2's
`<|im_end|>` — need no `--eos-token`, and the staging path (which downloads from the Hub) is
skipped.

## Run

Smoke test on the bundled sample — small batch so 10 rows produce steps:

```bash
export TESTING=true          # ROCm only
python3 train_llm_redhat.py \
  --model-path google/gemma-4-E4B-it \
  --ckpt-output-dir checkpoints_smoke \
  --num-epochs 1 --effective-batch-size 2 --nproc-per-node 2 \
  --eos-token "<turn|>"
```

```
Epoch 1:  20% | 1/5 | loss: 10.7442 | lr: 5.00e-06 | 112 tok/s
Epoch 1: 100% | 5/5 | loss: 6.0593  | lr: 4.77e-07 | 2385 tok/s
Saved model at 10.0 samples in 162.99 seconds
```

Full run:

```bash
nohup python3 train_llm_redhat.py \
  --model-path google/gemma-4-31b-it \
  --data-path path/to/train.jsonl \
  --ckpt-output-dir checkpoints \
  --num-epochs 4 --effective-batch-size 512 \
  --learning-rate 2e-5 --unfreeze-rank-ratio 0.35 \
  --max-seq-len 5056 --max-tokens-per-gpu 37312 \
  --nproc-per-node 8 --eos-token "<turn|>" --seed 42 \
  --speed-steps 20 \
  > train_llm_redhat.log 2>&1 &

tail -f train_llm_redhat.log
```

### Scaling to 8 GPUs

No code or requirements change. Three things must be right:

```bash
# AFTER `source .env_redhat/bin/activate` — a stale pin in bin/activate silently
# runs you on fewer GPUs or fails rendezvous
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # ROCm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TESTING=true                              # ROCm
python3 -c "import torch; assert torch.cuda.device_count()==8"

python3 train_llm_redhat.py \
  --model-path google/gemma-4-E4B-it \
  --data-path  $OUTPUT_DIR/data_rep_640.jsonl \
  --ckpt-output-dir $OUTPUT_DIR/ckpt8 --data-output-dir $OUTPUT_DIR/data_output \
  --num-epochs 1 --effective-batch-size 64 --nproc-per-node 8 \
  --max-seq-len 4096 --max-tokens-per-gpu 8192 \
  --eos-token "<turn|>" --seed 42 --speed-steps 2
```

1. Re-export the device lists after sourcing the venv, and assert the count.
2. The bundled 10-row sample cannot feed 8 ranks — replicate it (e.g. x64 to 640 rows) outside
   the repo.
3. Raise `--effective-batch-size` so the global batch divides across the world size.

On a shared machine hold a machine-wide GPU mutex (`flock`) so the job owns all the cards.

mini-trainer wraps the model with **FSDP2** (`torch.distributed.fsdp.fully_shard`), launched by
`torchrun` from the `osft_params` single-node block. It batches by token budget
(`max_tokens_per_gpu`) and derives gradient accumulation itself from `effective_batch_size` and
the world size.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` | *(required)* | Base/instruct model — HF id or local path |
| `--data-path` | `data/OTel_LLM_sample_10.jsonl` | Training JSONL |
| `--ckpt-output-dir` | *(required)* | Where checkpoints (`hf_format/samples_*`) are written |
| `--num-epochs` | 4 | Epochs |
| `--unfreeze-rank-ratio` | 0.3 | OSFT adaptation vs preservation (0.2-0.35 typical) |
| `--effective-batch-size` | 128 | Global batch size; must be >= world size |
| `--learning-rate` | 5e-6 | Learning rate |
| `--max-seq-len` | 4096 | Max sequence length in tokens |
| `--max-tokens-per-gpu` | 8192 | Lower on OOM; raise with headroom |
| `--nproc-per-node` | 8 | Number of GPUs |
| `--data-output-dir` | `data_output` | Processed-data / EOS-staging dir; a RAM disk is faster |
| `--unmask-messages` / `--no-unmask-messages` | on | All turns vs assistant-only |
| `--eos-token` | None | EOS override for data processing + checkpoint (see Data) |
| `--seed` | 42 | Random seed |
| `--speed-steps` | 0 | Print a live speed/ETA report every N steps (0 = off) |
| `--validation-split` | 0.0 | Held-out fraction, [0.0, 1.0); 0.0 disables validation |
| `--validation-frequency` | None | Validate every N steps — required when split > 0 |
| `--save-best-val-loss` | off | Checkpoint whenever validation loss improves |

## Output

Checkpoints go to `<ckpt-output-dir>/hf_format/samples_*` (plus `samples_*_best_val_loss` with
`--save-best-val-loss`); the script prints the most recent path on success. Run arguments land
in `logs/<timestamp>/run_args.json`, and the backend writes step metrics to
`<ckpt-output-dir>/training_metrics_0.jsonl`.

Monitoring:

- **Live** (`--speed-steps N`) — `speed_monitor.py` polls `training_metrics_0.jsonl` and prints
  progress, per-step rate, ETA, peak memory, peak tokens/sec, and last validation loss.
  Steps-per-epoch is `ceil(num_samples / effective_batch_size)`; the per-step rate spans the
  first to last logged step, so it excludes load/warmup time.
- **After the fact** — `python check_memory.py <ckpt_output_dir>` prints peak memory, current
  step, and duration, and renders a loss plot via `training_hub.plot_loss`.

## Notes

- **Checkpointing cannot be disabled from the CLI.** `osft_params` hardcodes
  `checkpoint_at_epoch=True` and `save_final_checkpoint=True`, so even a 1-epoch smoke run
  writes a full HF checkpoint (an 8B model is ~16 GB). Point `--ckpt-output-dir` at a large disk
  and delete afterwards.
- **OOM:** reduce `--max-tokens-per-gpu` first (large-vocabulary models are memory-hungry), then
  `--effective-batch-size`. `use_liger` and `osft_memory_efficient_init` are already on.
- For domain adaptation, start with `--unfreeze-rank-ratio` between 0.2 and 0.3.
- On ROCm, `torchao` prints `Failed to load ..._C_mxfp8...so` warnings at import; they are
  CUDA-only kernels and harmless.
- The script pins `nnodes=1` in `osft_params`. Multi-node needs `nnodes`/`rdzv_*` edited there.
