# `train_reranker_standalone.py`

## Overview & when to use

Cross-encoder **reranker** fine-tuner built on sentence-transformers `CrossEncoder`.
Trains a reranker (default `Qwen/Qwen3-Reranker-0.6B`) to score (query, document) pairs,
using one positive and N hard negatives per query with `BinaryCrossEntropyLoss`, and
evaluates with a reranking nDCG@10 evaluator. Single-node multi-GPU via `accelerate`.

Use it as the second stage of a retrieval pipeline — a cross-encoder scores a (query,
document) pair jointly (`num_labels=1`), which is more accurate than embedding
similarity but too slow for first-stage retrieval, so it reranks a candidate set
produced by an embedding model (see `../../embedding/sentence_transformers/`).

Design notes:

- **Pair expansion** — `load_and_split` turns each query into `1 + n_neg` labeled pairs
  (positive = 1.0, negatives = 0.0) for `BinaryCrossEntropyLoss`.
- **Deterministic split** — a fixed `random.seed(42)` shuffle + slice gives every rank
  the same train/eval split.
- **Qwen pad token** — Qwen has no dedicated pad token, so the tokenizer's `eos` is used
  and synced to `model.config.pad_token_id`; otherwise the trainer raises a
  "pad_token_id not set" ValueError.
- **Evaluation** — `CrossEncoderRerankingEvaluator` at nDCG@10; a baseline eval runs on
  rank 0 before training (barriers keep ranks in sync) and `load_best_model_at_end`
  selects on `reranking_ndcg@10`.

## Install

Python 3.12, in its own venv — deps differ from the generative trainers:

```bash
python3.12 -m venv ~/.venv-rerank && source ~/.venv-rerank/bin/activate
pip install -r requirements_reranker.txt
```

Verify:

```bash
python -c "from sentence_transformers.cross_encoder import CrossEncoder; print('imports OK')"
```

### NVIDIA (CUDA)

The default `torch` wheels from PyPI ship with CUDA support. The model loads with
`attn_implementation="flash_attention_2"`; building `flash-attn` needs the CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3
```

If you can't build `flash-attn`, change it to `"sdpa"` in `train_reranker_standalone.py`.

### AMD (ROCm)

The `CrossEncoder` stack is plain PyTorch + HF Transformers, so it runs on AMD GPUs via
the official ROCm torch wheels — PyTorch publishes pip wheels for ROCm-capable Linux
systems (see [pytorch.org — Get Started, "With ROCm"](https://pytorch.org/get-started/locally/)).
Install torch from the ROCm index **before** the rest of the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_reranker.txt
```

The ROCm build reuses the CUDA semantics at the Python API level — `cuda:{rank}`
devices and the `nccl` backend (mapped to RCCL) work unchanged. Skip `flash-attn`
(CUDA-only build here); the script auto-detects its absence and falls back to
`attn_implementation="sdpa"`.

#### Tested on MI355X (ROCm 7.2) — verdict: works with changes

Verified 2026-08-19 on 1× AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4,
Python 3.12.3 — default model `Qwen/Qwen3-Reranker-0.6B`, shipped OTel sample.

Exact install that worked (venv inside this folder):

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_train_reranker_standalone
source .env_train_reranker_standalone/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> torch 2.11.0+rocm7.2
pip install -r requirements_reranker.txt                                         # all pins install unchanged
ln -sf ../../../dev.env dev.env                                                  # HF_TOKEN for the Hub
```

> Note: the campaign venvs (including `.env_train_reranker_standalone`) were removed in
> the 2026-08 repo reorg — to reproduce, recreate one as `python3 -m venv
> .env_sentence_transformers` and run the same installs.

Exact smoke command that worked (single GPU — see the launcher quirk below):

```bash
torchrun --nproc_per_node=1 train_reranker_standalone.py --test_mode --epochs 1 --batch 16
```

Observed log lines (single MI355X, ~2.3 steps/s, GPU at 99% use, ~12.7GB VRAM):

```
Baseline Results: {'reranking_map': 1.0, 'reranking_mrr@10': 1.0, 'reranking_ndcg@10': 1.0}
{'loss': '1.435', 'grad_norm': '6.281', 'learning_rate': '5.882e-06', 'epoch': '0.5405'}
{'train_runtime': '10.72', 'train_samples_per_second': '55.41', 'train_steps_per_second': '1.772', 'train_loss': '0.9063', 'epoch': '1'}
Training complete. Best model loaded and saved to .../final
```

(An 8-epoch run on the same sample decayed loss 3.387 → 0.10 — training is
genuinely converging, not just running.)

Changes needed on ROCm (all applied here):

- **flash-attn → sdpa**: `flash-attn` is a CUDA-only build; the script now tries
  `import flash_attn` and falls back to `"sdpa"` automatically, so no manual edit
  is needed anymore (NVIDIA behavior unchanged).
- **ROCm 7.2 wheel index**: `--index-url .../whl/rocm7.2` (same `torch==2.11.0`
  pin; the rocm7.1 index also carries it, but match your installed ROCm).
- **Single-GPU launch quirk** (not ROCm-specific): the script calls
  `dist.init_process_group` unconditionally, and
  `accelerate launch --num_processes=1` does not set `RANK`/`WORLD_SIZE`, so it
  dies with "environment variable RANK expected, but not set". Use
  `torchrun --nproc_per_node=1 ...` for a single GPU; the multi-GPU
  `accelerate launch --num_processes=8` command sets the env vars and is fine.
- Harmless warning: `expandable_segments not supported on this platform` — the
  `PYTORCH_CUDA_ALLOC_CONF` setting is ignored by the HIP allocator.

#### Tested on H100 (CUDA 13.0) — verdict: WORKS

Verified 2026-08-22 on 1× NVIDIA H100 80GB HBM3 (Hopper cc 9.0), driver
**580.173.02**, **CUDA 13.0**, Python 3.12.3 — default model
`Qwen/Qwen3-Reranker-0.6B` (cached), shipped OTel sample. Single-GPU smoke only
(multi-GPU deferred — see note below). **No code change was needed.**

Key versions: `torch 2.13.0+cu130`, sentence-transformers 5.7.0, transformers
5.5.0, accelerate 1.14.0, datasets 4.3.0, numpy 2.5.1.

Exact install that worked (venv inside this folder):

```bash
cd training/reranker/sentence_transformers
python3 -m venv .env_sentence_transformers && source .env_sentence_transformers/bin/activate
pip install torch numpy                       # -> torch 2.13.0+cu130 (native CUDA 13; NO --index-url)
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
# Install the rest WITHOUT the torch==2.11.0 pin (no cu130 wheel exists for 2.11.0):
grep -v '^torch==' requirements_reranker.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt
python -c "import torch; print(torch.__version__)"   # re-verify: still 2.13.0+cu130 (NOT clobbered)
ln -sf ../../../dev.env dev.env                # HF_TOKEN for the Hub
```

> **torch pin deviation (documented):** `requirements_reranker.txt` pins
> `torch==2.11.0` (the ROCm-verified pin). There is no `+cu130` wheel for 2.11.0,
> so on H100 use the default stable `torch 2.13.0+cu130` and install the rest of
> the file with the `torch==` line dropped (otherwise pip drags in a CPU/older
> torch on top). Everything else installs at its pinned version unchanged.

Exact smoke command that worked (single GPU — same launcher quirk as ROCm):

```bash
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/path/to/hf_cache   # plain CUDA_VISIBLE_DEVICES; no HIP_* vars
torchrun --nproc_per_node=1 --master_port 29648 train_reranker_standalone.py \
  --test_mode --epochs 3 --batch 16 --eval_frac 0.05 --max_len 512 --out <out>
```

Real log lines (single H100; loss decays 2.279 → ~0.17; 54 optimizer steps over 3 epochs):

```
Baseline Results: {'reranking_map': 1.0, 'reranking_mrr@10': 1.0, 'reranking_ndcg@10': 1.0}
{'loss': '2.279', 'grad_norm': '7.281', 'learning_rate': '9.375e-06', 'epoch': '0.5714'}
{'loss': '0.1893', 'grad_norm': '6.938', 'learning_rate': '5.208e-06', 'epoch': '1.686'}
{'loss': '0.1686', 'grad_norm': '7.25', 'learning_rate': '3.125e-06', 'epoch': '2.229'}
{'train_runtime': '47.91', 'train_samples_per_second': '35.7', 'train_steps_per_second': '1.127', 'train_loss': '0.5844', 'epoch': '3'}
Training complete. Best model loaded and saved to <out>/final
```

GPU-7 residency (`nvidia-smi --id=7 --query-compute-apps` sampled from inside the
run — my venv's python was the *only* compute-app on GPU 7):

```
1556132, /dev/shm/.../venv_rerank_st/bin/python, 7724 MiB     # my PID, sole process on GPU 7
92 %, 7729 MiB      # utilization.gpu, memory.used  (peaks 86–92%; ~7.7 GB VRAM)
```

**Step count:** 99-row sample → `--test_mode` (cap 200, no-op here), `--eval_frac
0.05` leaves 94 train rows → `94×(1+5)=564` pairs; `dataloader_drop_last=True` +
`gradient_accumulation_steps=2` → `floor(564/16)/2 = 18` optimizer steps/epoch ×3
= **54 steps** (progress bar `.../54`, checkpoint-18 + checkpoint-54 written).
Non-trivial, and loss decreased across them. Post-train sanity `predict()`: the
relevant OTel doc scored -0.625 vs -9.375/-10.75 for irrelevant docs — correct
ranking.

Differences vs the MI355X recipe:

- **torch:** `pip install torch numpy` → `2.13.0+cu130` (native CUDA 13, no index
  URL), vs `torch==2.11.0+rocm7.2 --index-url .../whl/rocm7.2` on MI355X. Drop the
  `torch==2.11.0` pin from the requirements install (deviation noted above).
- **Devices:** plain `CUDA_VISIBLE_DEVICES`; dropped `HIP_VISIBLE_DEVICES` /
  `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`.
- **flash-attn:** `flash_attention_2` was attempted but **no prebuilt wheel** is
  published for torch 2.13/cu130/py3.12 on PyPI (`pip install --only-binary :all:
  flash-attn` → "No matching distribution"). A source build (nvcc 13.0 *is*
  present at `/usr/local/cuda`) was skipped to stay in the time-box. The script's
  `try: import flash_attn / except ImportError: attn_impl = "sdpa"` fallback
  engages automatically, so **sdpa was used, no edit needed** (identical to ROCm).
  To use FA2 on H100, build flash-attn from source and it will be picked up.
- **VRAM:** ~7.7 GB peak on `--batch 16 --max_len 512` (of 80 GB) — same as MI355X,
  not memory-bound. The shipped `--batch 64` default would also fit easily; batch
  is dictated by dataset size, not VRAM.
- **`expandable_segments`:** the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  set by the script IS honored by the CUDA allocator here (the ROCm "not supported"
  warning does not appear).
- **tf32:** this script never sets tf32, so the tf32 concern that hits the other
  trainers in this repo does not apply (matmuls run in bf16 via `bf16=True`).

**The `_verify_pair_roles_supported` trap (ST 5.7.0) — checked, does NOT fire on
this path.** The cached `Qwen3-Reranker-0.6B` ships a *generic generation*
chat_template (system/user/assistant + tools, architecture `Qwen3ForCausalLM`),
which the ST 5.7.0 pair-role check rejects when a reranker is loaded via the raw
transformers path. But `CrossEncoder(...)` here installs ST's own
**Qwen3-Reranker-specific** chat_template at load time — it branches on
`query`/`document`/`system` roles (`selectattr("role","eq","query")` …), exactly
what the check requires — so `predict()` and training both succeed with no
intervention. (The saved `<out>/final/chat_template.jinja` shows this
`<Instruct>/<Query>/<Document>` template.) *If* you hit the ValueError on a
different code path or model, the fix is to install a Query/Document template via
`model.processor.chat_template = ...`, `processor_kwargs={"chat_template": ...}`,
or a `chat_template.jinja` beside the model.

**Downstream note — score SCALE:** the saved model uses ST's `LogitScore` module
with an `Identity` activation, so `predict()` returns **raw logits** (e.g.
-0.625 / -9.375), not 0–1 probabilities. If you compare against a reference
implementation (e.g. the generative yes/no scorer in `inference/*/reranker/`),
**diff the RANKINGS, not the absolute scores** — the scale is template/head
dependent.

**Multi-GPU (deferred):** not run this wave (GPUs 0–3 held by a co-tenant
production job). A 2- or 8-GPU pass would use the same
`torchrun --nproc_per_node=N --master_port <free>` shape (single-node DDP, model
replicated not sharded), but see the "8-GPU run (8x MI355X)" section below: the
shipped `--batch 64` silently degenerates to a **single optimizer step** on this
tiny sample (`global_batch = batch × N × 2` overruns the 564–594 pairs). Lower
`--batch` (or enlarge the dataset) so `floor(pairs/(batch×N))/2` is a meaningful
step count before launching multi-GPU.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

Verified 2026-08-19 on 8× AMD Instinct MI355X (gfx950, 288GB each), ROCm 7.2.4,
`torch 2.11.0+rocm7.2` (HIP 7.2.26015), Python 3.12.3 — verdict: **works, no code
change needed; the shipped `--batch` default must be lowered.**

Exact working command (from inside the folder, venv active):

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port 29632 train_reranker_standalone.py \
  --data /path/to/train.jsonl --out /path/to/out --batch 16 --epochs 3
```

**Parallelism** — plain single-node DDP: `torchrun` spawns 8 ranks, the script's own
`dist.init_process_group(backend="nccl", device_id=cuda:LOCAL_RANK)` binds one rank
per GPU (NCCL maps to RCCL on ROCm), and `CrossEncoderTrainer` wraps the model in
DDP with `ddp_find_unused_parameters=False`. The model is replicated, not sharded —
a 0.6B cross-encoder fits in one GPU many times over, so no FSDP/DeepSpeed is
needed. `accelerate launch --num_processes=8` also works; `torchrun` is used here
so the single- and multi-GPU commands are identical in shape.

**Batch geometry (the thing that actually breaks)** — `gradient_accumulation_steps`
is hardcoded to 2 and `dataloader_drop_last=True`, so
`global_batch = --batch × num_GPUs × 2`. With the shipped defaults at 8 ranks that
is `64 × 8 × 2 = 1024` against a dataset that expands to only `99 × (1+5) = 594`
pairs — the loader yields **one** micro-batch per rank and the whole "run" performs
**a single optimizer step**. It does not crash and it exits 0:

```
{'train_runtime': '7.867', 'train_samples_per_second': '75.51',
 'train_steps_per_second': '0.127', 'train_loss': '3.974', 'epoch': '1'}   # 1 step, untrained
```

That is a silent no-op, not a pass. Fix it by lowering `--batch` until the step
count is meaningful: `steps/epoch = floor(pairs / (batch × 8)) / 2`. The evidence
run below used `--batch 16` (global batch 256) over a 4000-row dataset → 93
optimizer steps per epoch, 279 steps over 3 epochs.

The shipped 100-row sample cannot fill 8 GPUs at any sane batch size (at `--batch 2`
it manages 18 steps/epoch with a global batch of 32). The measurements below
therefore use the shipped sample replicated ×40 (4000 rows, generated **outside**
the repo, not committed) — this is a **pipeline proof, not a learning result**: the
eval split is drawn from the same replicated rows, so it is contaminated by
construction and its metrics are saturated.

Also note `--eval_frac 0.003` on a small dataset: 100 rows leaves **1** eval query,
which makes `reranking_ndcg@10` a coin-flip metric (it reports exactly `1.0`) and
`metric_for_best_model` meaningless. Raise `--eval_frac` for any real run.

**Measured evidence** (`rocm-smi` sampled every 5s from inside the same locked job,
PIDs cross-checked against that job's own 8 workers):

| | 8× MI355X (`--batch 16`) | 1× MI355X (`--batch 128`) |
|---|---|---|
| global batch | 256 (16×8×2) | 256 (128×1×2) |
| optimizer steps / epoch | 93 | 93 |
| throughput | **497.2 samples/s**, 1.93 steps/s | 77.2 samples/s, 0.30 steps/s |
| epoch wall time | ~48 s | 309.9 s |
| GPU utilization | **96.2% mean** across all 8 (min 86, max 100) | 81–100% on GPU 0 |
| VRAM / GPU | **15.7 GB** mean, 16.1 GB peak (of 288 GB) | 32.4 GB peak |
| loss at end of epoch 1 | 0.0011 | 0.0096 |
| `reranking_ndcg@10` | 1.0 (baseline 0.9692) | 1.0 (baseline 0.9692) |

Three-epoch 8-GPU run: `{'train_runtime': '144.4', 'train_samples_per_second':
'497.2', 'train_steps_per_second': '1.932', 'train_loss': '0.2037', 'epoch': '3'}`,
loss decaying 3.748 → 5.9e-05.

**Scaling:** 6.4× throughput on 8 GPUs at matched global batch (497.2 vs 77.2
samples/s) — ~80% efficiency, which is what a 0.6B model with
`gradient_checkpointing=True` and a 256-sample global batch gives you; the per-step
work is small enough that the DDP all-reduce and the recompute overhead are visible.

**Quality vs the 1-GPU baseline: held.** Both world sizes produce the identical
pre-training baseline (`ndcg@10 0.9692`, `map 0.9583`) — the seeded shuffle in
`load_and_split` gives every rank the same split, so the data pipeline is
world-size invariant. Both converge on the same trajectory and reach `ndcg@10 1.0`.
The 8-GPU epoch-1 loss is lower (0.0011 vs 0.0096) only because its LR schedule is
stretched over 3 epochs, so `warmup_ratio=0.1` puts it at a higher LR during epoch 1
— it is not a quality difference. The nDCG comparison is weak evidence on its own:
the metric is saturated on this sample (see caveat above).

**Checkpointing at 8 ranks works.** A separate run of the **unmodified** script with
the shipped `--batch 64` (`save_strategy="epoch"` + `load_best_model_at_end=True`)
completed cleanly at 8 ranks: 23 steps, 456.5 samples/s, 97.2% mean utilization,
24.3 GB/GPU, checkpoint written, best model reloaded and saved to `<out>/final`. Be
aware it wrote **4.5 GB for a single epoch** (checkpoint + optimizer state + final
model); budget disk accordingly, or the run dies on a full filesystem rather than on
anything GPU-related.

Caveats and things that will cost you an hour:

- **No code change was required** — `attn_implementation` already falls back to
  `sdpa`, the rank-0-only baseline eval is fenced by `dist.barrier()` on both sides,
  and the in-training evaluator runs on *all* ranks inside
  `evaluation_loop` (sentence-transformers 5.7.0), so no collective is called on a
  subset of ranks. Nothing hangs at 8 ranks.
- All 8 ranks write the evaluator CSV to `<out>/eval/` concurrently. It is benign
  here (identical content, same split) but it is an unsynchronized write.
- `tf32` is never set by this script, so the ROCm `tf32` crash that hits the other
  trainers in this repo does not apply.
- Only `--batch` was tuned. `--max_len 1024`, `--n_neg 5`, `--lr`, the hardcoded
  `gradient_accumulation_steps=2` and `gradient_checkpointing=True` were left at
  their shipped values.
- With 288 GB per GPU and 16 GB used at `--batch 16`, this workload is nowhere near
  memory-bound — `--batch` here is dictated entirely by dataset size, not by VRAM.
- Warnings seen on all ranks and safe to ignore: `expandable_segments not supported`,
  the `warmup_ratio`/`logging_dir` v5 deprecations, and
  `destroy_process_group() was not called before program exit`.

## Environment & secrets

Optional `dev.env` next to the script, loaded via `load_dotenv("dev.env")` — only needed
for gated models on the Hub:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`--hf_home` optionally points the HF model cache somewhere other than the default
`~/.cache/huggingface`.

## Data

`--data` is a JSONL, one row per query — an `anchor` (query), one `positive` (relevant
doc), and hard negatives `negative_1 … negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

Each row is expanded into `1 + n_neg` labeled pairs — the positive with label `1.0` and
each negative with `0.0` — as `{sentence_0=query, sentence_1=doc, label}`. Extra columns
(e.g. an `answer` field) are ignored.

Shipped sample: `OTel_reranker_sample_100.jsonl` (100 rows, includes the `answer`
column) — the default `--data`, there to smoke-test end to end. To train for real, point
`--data` at your own JSONL with the same columns.

## Run

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_reranker_standalone.py --test_mode
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_reranker_standalone.py --data /path/to/your_train.jsonl --out reranker_run1 \
  > train_reranker_standalone.log 2>&1 &

tail -f train_reranker_standalone.log
```

**What "working" looks like:** a baseline eval prints before training, then per-epoch
`{'loss': ...}` and reranking metrics; the best checkpoint (by `reranking_ndcg@10`) is
loaded at the end and saved to `<out>/final`.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder base model |
| `--data` | `OTel_reranker_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--out` | `output` | Output dir for checkpoints, logs, and the final model |
| `--max_len` | `1024` | Max sequence length (query + document) |
| `--batch` | `64` | Per-GPU train/eval batch size — tuned for 80GB GPUs |
| `--epochs` | `2` | Number of training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--n_neg` | `5` | Hard negatives per query (`negative_1..negative_N`) |
| `--eval_frac` | `0.003` | Fraction of rows held out for evaluation |
| `--test_mode` | off | Cap the dataset at 200 rows for a quick test |
| `--hf_home` | `None` | Optional `HF_HOME` model-cache override |

## Output

- Checkpoints and TensorBoard logs under `--out` (`<out>/logs`).
- Baseline evaluation results written to `--out` before training.
- The best model (by `reranking_ndcg@10`) saved to `<out>/final`.

## Hardware support & evidence

- **Tested:** Azure cluster, 8×H100 80GB — single-node DDP. Batch sizes are tuned for
  80GB GPUs.
- **NVIDIA:** default PyPI torch wheels; flash-attention 2 by default (fallback: `"sdpa"`).
- **AMD:** **tested** — 1× and 8× MI355X (gfx950, 288GB), ROCm 7.2.4,
  `torch==2.11.0+rocm7.2` from
  [download.pytorch.org/whl/rocm7.2](https://download.pytorch.org/whl/rocm7.2/torch/),
  Python 3.12. Smoke training on the shipped sample converges (see the
  "Tested on MI355X" section above). Only the flash-attn → sdpa switch is needed
  (now automatic), plus `torchrun` instead of `accelerate launch` for
  single-GPU runs. 8-GPU DDP works unmodified at 96% utilization, but the default
  `--batch 64` silently degenerates to a single optimizer step on a small dataset —
  see "8-GPU run (8x MI355X, ROCm 7.2.4)" above before you launch one.
- **Other hardware (upstream claims — not verified here):** sentence-transformers (the
  `CrossEncoder` stack) runs on any PyTorch backend and auto-selects `cuda`, `mps`, or
  `cpu` — so Apple Silicon (MPS) and CPU training are claimed upstream via the HF
  Trainer (single-process; no distributed backend on MPS). Its ONNX and OpenVINO
  backends are inference-only.

## Notes

- `transformers 5.x` development builds can miss `save_safetensors` — the script patches
  it onto the training args if absent.
- Gradient accumulation is fixed at 2 in the training args — effective batch =
  `batch × 2 × num_GPUs`.
- The original training data for this work was an internal telco retrieval set; the
  shipped OTel sample is only for pipeline validation, not for producing a useful model.
