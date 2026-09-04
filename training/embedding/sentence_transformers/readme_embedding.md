# `train_embedding_standalone.py`

## Overview & when to use

Unified **embedding fine-tuner** built on sentence-transformers. One script trains any of
four model families for retrieval with hard negatives:

- sentence-transformers (`all-mpnet-base-v2`, `all-MiniLM-*`)
- BGE (`BAAI/bge-small/large/m3`)
- EmbeddingGemma (`google/embeddinggemma-300m`)
- static model2vec (`minishlab/potion-*`)

Use it when you need a retrieval embedding model adapted to your domain corpus — the loss
(MultipleNegativesRankingLoss, dot-product, cross-device gather) treats each query's
positive as the target and the negatives as in-batch hard negatives, and is wrapped in
MatryoshkaLoss so one trained model serves multiple embedding dimensions.

Per-model defaults live in the `MODELS` registry in the script and can be overridden on
the CLI. Registry fields:

| Field | Meaning |
|---|---|
| `loader` | `"transformer"` (SentenceTransformer) or `"static"` (model2vec) |
| `query_prefix` / `doc_prefix` | Instruction prompts prepended to queries/documents at encode time |
| `train_batch` / `eval_batch` | Per-device batch sizes — tuned for H100 80GB |
| `epochs` | Number of training epochs |
| `learning_rate` | Optimizer LR — static lookup tables want a much higher LR |
| `use_checkpointing` | Gradient checkpointing on/off (n/a for static) |
| `max_seq_length` | Optional; `None` keeps the model default. Gemma uses 1024 — covers >99.9% of the original Tele-Eval data |
| `matryoshka_dims` | Optional; `None` derives dims dynamically from the hidden size |
| `eval_corpus_size` | `None` — corpus = all positives; int — distractor pool of that size |
| `eval_on_start` | Run a baseline evaluation before training |

The Gemma train batch (96) is downscaled relative to smaller models because each row
expands to 7 inputs — 1 anchor + 1 positive + 5 negatives.

Design notes:

- **Registry-driven** — `resolve_cfg` merges `DEFAULT_CFG` ← `MODELS[model_name]` ← CLI
  overrides, so one code path serves every family.
- **Loader switch** — `transformer` ⇒ `SentenceTransformer(name)` with bf16 + flash-attn;
  `static` ⇒ `StaticEmbedding.from_model2vec(name)` — a lookup table, so no
  dtype/attention/tokenizer kwargs apply.
- **Prompts** — query/document instruction prefixes are set per family (BGE's query
  prefix, Gemma's `search_query:` / `search_document:`).
- **MTEB compatibility** — `SafeSTWrapper` coerces MTEB's varied encode inputs (dicts,
  ndarrays, nested lists) and injects a model-meta fallback so off-hub / static models
  still evaluate; MTEB 2.7.x otherwise crashes on models without a strict
  `organization/model` name.
- **Tokenizer quirk** — a problematic `fix_mistral_regex` tokenizer flag (from Gemma) is
  popped before serialization; a harmless no-op for other families.
- **Best model** — `load_best_model_at_end` on `telco_unseen` nDCG@10; the final model is
  saved to `<output_dir>/final_model`.

## Install

Python 3.12, in its own venv — these deps differ from the other trainers:

```bash
python3.12 -m venv ~/.venv-embed && source ~/.venv-embed/bin/activate
pip install -r requirements_embedding.txt
```

`model2vec` is only needed for the static `potion-*` models. Verify:

```bash
python -c "import sentence_transformers, mteb; print('imports OK')"
```

### NVIDIA (CUDA)

The default `torch` wheels from PyPI ship with CUDA support — `pip install -r
requirements_embedding.txt` is all you need. Transformer-backed models load with
`attn_implementation="flash_attention_2"`; building `flash-attn` needs the CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3
```

If you can't build `flash-attn`, change the implementation to `"sdpa"` in `load_model()`.

### AMD (ROCm)

sentence-transformers sits on top of PyTorch, so it runs on AMD GPUs via the official
ROCm torch wheels — PyTorch publishes pip wheels for ROCm-capable Linux systems (see
[pytorch.org — Get Started, "With ROCm"](https://pytorch.org/get-started/locally/)).
Install torch from the ROCm wheel index **before** the rest of the requirements:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding.txt
```

Notes for ROCm:
- The ROCm build reuses the CUDA semantics at the Python API level — `cuda:{rank}`
  devices, `torch.cuda.set_device`, and the `nccl` backend (mapped to RCCL) work as-is,
  so the script needs no code changes.
- `flash-attn` as pinned here is a CUDA build — skip it. `load_model()` now detects a
  ROCm torch build (`torch.version.hip`) and selects `attn_implementation="sdpa"`
  automatically; NVIDIA keeps `flash_attention_2`.

### Tested on AMD Instinct MI355X — ROCm 7.2 (verified 2026-08-19)

**Verdict: works with changes.** Single-GPU smoke training of
`google/embeddinggemma-300m` on the shipped OTel sample ran end to end on one MI355X
(gfx950, 288GB) — finite decreasing loss, all three evaluators (telco IR + MTEB), best
checkpoint reload, and final-model save. The changes vs. the NVIDIA path:

1. torch from the ROCm wheel index (the `torch==2.11.0` pin exists there —
   installs as `2.11.0+rocm7.2`); every other pin in
   `requirements_embedding.txt` installed unmodified.
2. `flash-attn` skipped; sdpa auto-selected by the ROCm switch in `load_model()`.

Exact install that worked (Python 3.12.3, ROCm 7.2.4):

```bash
python3 -m venv .env_train_embedding_standalone
source .env_train_embedding_standalone/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_embedding.txt
```

> Note: the campaign venvs (including `.env_train_embedding_standalone`) were removed in
> the 2026-08 repo reorg — to reproduce, recreate one as `python3 -m venv
> .env_sentence_transformers` and run the same installs.

Exact smoke command (single GPU). Note it is `torchrun`, not `accelerate launch`:
with `--num_processes=1` accelerate uses its simple launcher and does not set `RANK`,
which the script's explicit `dist.init_process_group` requires — a launcher quirk,
not a ROCm one. Add `--master_port` if 29500 is taken by another job:

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=29610 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 8 --epochs 1
```

Observed log lines (11 steps; `--batch_size 8` because the 100-row sample with
`dataloader_drop_last=True` yields zero steps at the registry batch of 96):

```
{'loss': '5.162', 'grad_norm': '380', 'learning_rate': '0', 'epoch': '0.09091'}
{'loss': '1.436', 'grad_norm': '241', 'learning_rate': '1e-05', 'epoch': '0.2727'}
{'train_runtime': '35.12', 'train_samples_per_second': '2.563', 'train_loss': '2.264', 'epoch': '1'}
[rank=0] INFO: MTEB Success. Avg Score: 0.5605
```

`rocm-smi` sampled during the run showed 84–99% GPU utilization and ~28GB peak VRAM.
The RCCL (`nccl`) backend, bf16, gradient checkpointing, MatryoshkaLoss, and the Gemma
tokenizer workaround all behaved identically to CUDA. The 8-GPU verification is below.

### Tested on NVIDIA H100 80GB — CUDA 13.0 (verified 2026-08-22)

**Verdict: works (with a one-line attn-fallback fix).** Single-GPU smoke training of
`google/embeddinggemma-300m` on the shipped OTel sample ran end to end on one H100 80GB
(Hopper cc 9.0, driver 580.173.02) — finite decreasing loss, all three evaluators
(telco IR + MTEB), best-checkpoint reload, and final-model save. MTEB parity with the
MI355X baseline (**0.5609** vs. 0.5605/0.5613). Differences vs. the ROCm path:

1. torch from the default PyPI index — the `torch==2.11.0` pin resolves to
   `2.11.0+cu130` (native CUDA 13, real bf16 matmul confirmed on-GPU); no `--index-url`
   needed. Every other pin in `requirements_embedding.txt` installed unmodified.
2. `flash-attn==2.8.3` **built from source** against torch 2.11+cu130 (no prebuilt wheel
   exists for this torch/CUDA combo) and runs on H100 — so `flash_attention_2` was used,
   not sdpa.
3. `load_model()`'s attn switch was hardened: on CUDA it now picks `flash_attention_2`
   only if `import flash_attn` succeeds, else falls back to `sdpa`. The shipped code
   unconditionally requested `flash_attention_2` on any non-ROCm build, which crashes on
   a CUDA box without flash-attn installed. ROCm behavior (`torch.version.hip` → sdpa) is
   unchanged.

Exact install that worked (Python 3.12.3, CUDA 13.0, driver 580.173.02):

```bash
python3 -m venv .env_sentence_transformers
source .env_sentence_transformers/bin/activate
pip install torch==2.11.0 numpy==2.5.1          # -> torch 2.11.0+cu130 (verify torch.version.cuda == '13.0')
pip install -r requirements_embedding.txt        # torch/numpy already satisfied
# Optional but used here — flash-attn built from source (needs the CUDA toolkit):
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.8.3 --no-build-isolation   # ~2 min on this box; skip -> sdpa
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # re-check: several deps clobber torch
```

Exact smoke command (single GPU). Note it is `torchrun`, not `accelerate launch` (same
launcher quirk as ROCm: `accelerate --num_processes=1` does not set `RANK`, which the
script's `dist.init_process_group` requires). On CUDA use plain `CUDA_VISIBLE_DEVICES`
(no `HIP_VISIBLE_DEVICES`). `--experiment_root` must point at the model cache because the
script sets `HF_HOME = --experiment_root` internally; `--output_dir` keeps checkpoints off
that cache dir:

```bash
# The proxy on this box 403s huggingface.co (model resolution + MTEB SciFact/NFCorpus
# dataset download), so unset it first; the gemma model itself is cache-resident.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=4          # one assigned GPU on a shared node
export HF_HOME=/path/to/model/cache HF_DATASETS_CACHE=/tmp/dscache_emb_st
torchrun --nproc_per_node=1 --master_port=29644 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 8 --epochs 1 \
  --experiment_root /path/to/model/cache --output_dir /path/to/out
```

Observed log lines (11 steps; `--batch_size 8` because the 100-row sample with
`dataloader_drop_last=True` yields 0 steps at the registry batch of 96 — identical batch
geometry to the MI355X smoke):

```
{'loss': '5.097', 'grad_norm': '392', 'learning_rate': '0', 'epoch': '0.09091'}
{'loss': '1.44', 'grad_norm': '223', 'learning_rate': '1e-05', 'epoch': '0.2727'}
{'train_runtime': '70.47', 'train_samples_per_second': '1.277', 'train_steps_per_second': '0.156', 'train_loss': '2.286', 'epoch': '1'}
[rank=0] INFO: [rank=0] MTEB Success. Avg Score: 0.5609
```

Loss is noisy step-to-step (8-row global batch with in-batch negatives) but decreasing:
mean of the first 3 steps 3.285 → mean of the last 3 steps 1.880, final `train_loss`
2.286. `telco_unseen`/`telco_seen` nDCG@10 were 1.0 (saturated on the ~10-doc eval
corpus — uninformative, same as MI355X). MTEB **0.5609** (SciFact 0.7471, NFCorpus
0.3747), within ±0.001 of the MI355X 1-GPU baseline.

GPU-4 residency proof — `nvidia-smi` filtered to this job's PID + GPU-4 UUID, sampled
from inside the run:

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader | grep <GPU4-UUID>
1546102, /dev/shm/.../venv_emb_st/bin/python3, 5336 MiB, GPU-e13d18b6-ccfb-6676-668a-cd489ad01b55
# per-GPU snapshot: GPU4 mem=5345 MiB util=62%  (peak ~5.3 GiB — see VRAM note)
```

**VRAM.** Peak was only ~5.3 GiB (batch 8, seq 1024, grad-checkpointing on) — no OOM
concerns on the 80 GB card; the registry `train_batch=96` would fit comfortably (it is
disabled here only by the drop-last step-count trap, not memory). Contrast the MI355X
~28 GiB reading, which was at the default larger batch.

**Multi-GPU (deferred).** Not run in this wave (GPUs 0–3 were a co-tenant production
job). To reproduce the ROCm 8-GPU result on H100 it would be the same recipe:
`torchrun --nproc_per_node=8 --master_port=<free> ... --batch_size 1` (global batch 8 →
11 steps, matching this baseline), NCCL backend, `gather_across_devices=True` gathering
the negative pool across all ranks. Assert `torch.cuda.device_count()==8` after setting
`CUDA_VISIBLE_DEVICES` before launching.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**Verdict: works, no code change.** The script scales from 1 to 8 MI355X GPUs as shipped —
the only thing that has to change is `--batch_size`, because the 100-row sample and
`dataloader_drop_last=True` leave too few optimizer steps at 8 ranks (see *Batch geometry*).
Verified 2026-08-19 on 8×MI355X (gfx950, 288GB), ROCm 7.2.4, `torch==2.11.0+rocm7.2`.

Exact working command (shipped dataset, one process per GPU):

```bash
source .env_train_embedding_standalone/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29631 train_embedding_standalone.py \
  --model_name google/embeddinggemma-300m --batch_size 1 --epochs 1
```

**Parallelism.** Plain single-node DDP via `torchrun` — 8 ranks, one rank per GPU, RCCL
(the ROCm `nccl` backend) for gradient all-reduce. The script's own
`dist.init_process_group` picks `LOCAL_RANK`/`RANK` straight out of the torchrun
environment, so no launcher config is involved; `accelerate launch --num_processes=8`
works too, but `torchrun` is what was verified (and `--num_processes=1` is broken for this
script — see the single-GPU section). `MultipleNegativesRankingLoss` is constructed with
`gather_across_devices=True`, so DDP is not merely data parallel here: the in-batch
negative pool is gathered across all 8 ranks. A probe at `--batch_size 8` measured the
candidate pool growing 48 → 384 vectors (8×) with gradients still flowing through the
gather, which is the reason multi-GPU is worth doing for this trainer at all.

**Batch geometry (the one thing you must get right).** Steps per epoch are
`floor(train_rows / (per_device_batch × 8))` because `dataloader_drop_last=True`. The
shipped `OTel_embedding_sample_100.jsonl` splits 100 rows into 90 train / 10 eval, so at
8 ranks:

| `--batch_size` | global batch | optimizer steps/epoch |
|---|---|---|
| 96 (registry default) | 768 | **0 — silently trains nothing** |
| 8 (the 1-GPU smoke value) | 64 | 1 |
| **1 (used here)** | **8** | **11** |

`--batch_size 1` was chosen so the global batch (1×8 = 8) equals the 1-GPU baseline's
global batch (8×1 = 8) — same optimizer trajectory, so the MTEB score is directly
comparable. A second run used a 32× replicated copy of the sample (3200 rows → 2880 train)
at `--batch_size 8`, giving a global batch of 64 and 45 steps/epoch × 2 epochs = 90 steps,
purely to exercise the pipeline for long enough to get clean utilization evidence. That
replicated file was written outside the repo and is **a pipeline proof, not a learning
result** — see the caveats.

**Measured evidence** (`rocm-smi` sampled every 10s from inside the run, with
`--showpids`/`pgrep` in the same sample to confirm the PIDs were this job's):

| Run | steps | wall | throughput | final train loss | MTEB avg |
|---|---|---|---|---|---|
| shipped, `--batch_size 1`, 1 epoch | 11 | 35.0s | 2.57 samples/s | 2.274 | **0.5613** |
| replicated ×32, `--batch_size 8`, 2 epochs | 90 | 125.8s | 45.8 samples/s (0.716 steps/s) | 1.761 | 0.5208 |

Per-GPU utilization and VRAM during the training phase (all 8 GPUs, steady state):

```
21:44:51 use% [97, 98, 98, 98, 98, 97, 97, 98]   VRAM 11.6 12.1 11.8 11.5 11.8 12.1 11.6 11.6 GiB
21:45:02 use% [97, 98, 98, 95, 98, 96, 94, 96]   VRAM 11.6 12.1 11.8 12.3 11.8 12.1 12.0 11.6 GiB
21:45:13 use% [98, 98, 99, 97, 98, 98, 97, 98]   VRAM 11.7 12.1 11.8 12.3 11.8 12.1 12.0 11.7 GiB
```

94–99% busy on every rank, ~11.5–12.3 GiB per GPU at `--batch_size 8` (~10.5–10.9 GiB at
`--batch_size 1`) — roughly 4% of each card's 288GB, so there is a lot of headroom for
larger per-device batches. Rank 0 peaks at **29.3 GiB** during evaluation, because the
MTEB check runs rank-0-only; in those samples ranks 1–7 read 100% busy while spinning in
the RCCL barrier, not doing useful work. Both runs finished with `rc=0`, decreasing loss,
all three evaluators, and the final-model save.

**Quality vs. the 1-GPU baseline.** The comparable run (same global batch of 8, same 11
steps) scored **MTEB 0.5613** (SciFact 0.7475, NFCorpus 0.3751) against the 1-GPU
baseline's **0.5605** — a +0.0008 difference, i.e. quality held; nothing about DDP or the
cross-device gather degrades the result. `telco_unseen`/`telco_seen` nDCG@10 were 1.0 in
both, but that metric is saturated on a ~100-document corpus and says nothing useful.

**Caveats.**

- **No speedup at this size.** 8 GPUs at global batch 8 took 35.0s versus 35.1s on one GPU
  — splitting an 8-row global batch across 8 GPUs is launch- and sync-bound, so the extra
  cards buy nothing. Multi-GPU pays off only when you keep the per-device batch up
  (45.8 samples/s at global batch 64); do not read the two rows of the table above as a
  scaling measurement, since the short run's fixed warmup is amortized very differently.
- **The replicated dataset is not a learning result.** Its MTEB fell to 0.5208 after 90
  steps over only 100 unique rows repeated 32×, which is over-fitting a tiny domain sample.
  Duplicated rows also land in the same global batch, so some of MNRL's "negatives" are
  actually copies of the positive. It exists to prove the 8-rank pipeline, nothing more; it
  was written outside the repo and is deliberately not committed.
- **Checkpoint saving was disabled** for this verification (disk-constrained host) by
  patching `save_strategy="no"` / `load_best_model_at_end=False` onto the training args
  from an external wrapper. The repo script was not modified, and checkpointing is not
  related to whether 8 GPUs work — the default `save_strategy="epoch"` path was exercised
  by an earlier 8-GPU run on the same host, which wrote `checkpoint-45` and `final_model`
  normally.
- **Watch the venv's device pin.** `.env_train_embedding_standalone/bin/activate` carries a
  leftover `export HIP_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0` from the single-GPU
  session. Sourcing it and launching 8 ranks silently trains on one GPU; override both
  after activating, and assert with
  `python -c "import torch; assert torch.cuda.device_count()==8"` before training.

## Environment & secrets

Optional `dev.env` next to the script — only needed for gated models on the Hub
(e.g. `google/embeddinggemma-300m`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`. `HF_HOME` (the model/dataset cache) is pointed at
`--experiment_root`.

## Data

`--train_file` is a JSONL with one triplet-style row per line — an `anchor` (query), one
`positive` (relevant passage), and hard negatives `negative_1 … negative_5`:

```json
{"anchor": "...", "positive": "...", "negative_1": "...", "negative_2": "...",
 "negative_3": "...", "negative_4": "...", "negative_5": "..."}
```

Shipped sample: `OTel_embedding_sample_100.jsonl` (100 rows) — the default
`--train_file`, there to smoke-test the pipeline end to end. To train for real, point
`--train_file` at your own JSONL with the same columns.

## Run

Launch with `accelerate` (single-node DDP). The model family is selected by
`--model_name`; everything else defaults from the registry.

Smoke test against the shipped sample:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m
```

Full run:

```bash
nohup accelerate launch --num_processes=8 --mixed_precision=bf16 \
  train_embedding_standalone.py --model_name google/embeddinggemma-300m \
  --train_file /path/to/your_train.jsonl \
  > train_embedding_standalone.log 2>&1 &

tail -f train_embedding_standalone.log
```

Any family works the same way:

```bash
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name sentence-transformers/all-mpnet-base-v2
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name minishlab/potion-retrieval-32M
```

Override registry defaults on the CLI:

```bash
accelerate launch --num_processes=8 train_embedding_standalone.py --model_name BAAI/bge-m3 \
  --batch_size 24 --epochs 3 --lr 2e-5 --eval_corpus_size 100000
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `google/embeddinggemma-300m` | Model id/path; selects the registry entry |
| `--train_file` | `OTel_embedding_sample_100.jsonl` | Training JSONL (anchor/positive/negatives) |
| `--output_dir` | `None` | Output dir; defaults to `<experiment_root>/<RUN_ID>/trained_<model>` |
| `--experiment_root` | `experiments` | Root for `HF_HOME` and the default output dir |
| `--sample_fraction` | `1.0` | Fraction of the dataset to use |
| `--seed` | `42` | Seed for data splits and eval sampling — fixed so all ranks split identically |
| `--loader` | `None` | `transformer` or `static` — overrides registry |
| `--batch_size` | `None` | Per-device train batch size — overrides registry (eval batch = 2×) |
| `--epochs` | `None` | Training epochs — overrides registry |
| `--lr` | `None` | Learning rate — overrides registry |
| `--max_seq_length` | `None` | Max sequence length — overrides registry |
| `--eval_corpus_size` | `None` | Distractor corpus size for eval — overrides registry; omit for all-positives corpus |

## Output

- Checkpoints, TensorBoard logs (`logs/`), and `run.log` under the resolved output dir.
- A backup of the training script (`train_script_backup.py`) alongside the checkpoints.
- The final (best) model under `<output_dir>/final_model`.

Each epoch runs a `SequentialEvaluator`:

- **`telco_unseen`** — IR evaluator over held-out eval queries; its
  `eval_telco_unseen_dot_ndcg@10` picks the best checkpoint.
- **`telco_seen`** — IR evaluator over a sample of training queries (memorization signal).
- **MTEB** — `SciFact` + `NFCorpus` as a general-capability check (rank-0 only).

`--eval_corpus_size` controls the retrieval corpus: unset ⇒ all known positives; an int ⇒
required answers plus that many sampled distractors (used for the larger Gemma/model2vec runs).

## Hardware support & evidence

- **Tested:** Azure cluster, 8×H100 80GB — single-node DDP via `accelerate`. The registry
  batch sizes are tuned for 80GB GPUs.
- **NVIDIA:** default PyPI torch wheels; flash-attention 2 optional (fallback: `"sdpa"`).
- **AMD:** **tested** — 1× and 8× MI355X (gfx950, 288GB), ROCm 7.2.4,
  `torch==2.11.0+rocm7.2` from
  [download.pytorch.org/whl/rocm7.2](https://download.pytorch.org/whl/rocm7.2/torch/).
  The ROCm build keeps the `torch.cuda` API semantics, and the script runs unchanged
  apart from the (now automatic) flash-attn → sdpa switch. Single-node DDP across 8
  MI355X via `torchrun` is verified too (94–99% utilization on all 8 ranks, MTEB parity
  with the 1-GPU run) — only `--batch_size` has to be lowered so the 100-row sample still
  yields optimizer steps at 8 ranks. See "Tested on AMD Instinct MI355X" and
  "8-GPU run" above for the exact commands and evidence.
- **Other hardware (upstream claims — not verified here):** sentence-transformers runs on
  any PyTorch backend and auto-selects `cuda`, `mps`, or `cpu` — so Apple Silicon (MPS)
  and CPU training are claimed upstream via the HF Trainer (single-process; no
  distributed backend on MPS). Its ONNX and OpenVINO backends are inference-only.

## Notes

- The evaluators and the MTEB wrapper only run their heavy work on rank 0; barriers keep
  the other ranks in sync.
- `transformers 5.0.0.dev0` misses `save_safetensors` — the script patches it onto the
  training args if absent.
- The original training data for this work was an internal telco retrieval set; the
  shipped OTel sample is only for pipeline validation, not for producing a useful model.
