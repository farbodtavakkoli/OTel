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

### Compared to `../sentence_transformers/` (sentence-transformers `CrossEncoder`)

| | `../sentence_transformers` (ST) | this folder (FlagEmbedding) |
|---|---|---|
| Architecture | `CrossEncoder`, `num_labels=1` + BCE over flattened pairs | group-wise cross-entropy over 1 positive vs N-1 negatives |
| **LLM rerankers** | **no** — `CrossEncoder` wraps a `*ForSequenceClassification` head | **yes** — `llm` and `llm_layerwise`, LoRA, LM-head scoring |
| Loss shape | each (q, d) pair scored independently | softmax **within** the group, so negatives compete with the positive |
| Negative sampling | fixed `negative_1..5` columns | `--train_group_size` resampled per epoch from a `neg` list of any length |
| Teacher distillation | no | yes (`--knowledge_distillation` with `pos_scores`/`neg_scores`) |
| **Built-in evaluation** | **yes** — `CrossEncoderRerankingEvaluator`, nDCG@10, best-checkpoint selection | **no** — trains and saves only |
| Output | ST CrossEncoder folder | HF model folder, or a PEFT adapter for the LLM types |

**Verdict on usefulness: keep — this folder earns its place.** It is not a second way to do what
`../sentence_transformers` already does; it trains a **different architecture**. A
`CrossEncoder` cannot express `bge-reranker-v2-gemma`: that model has no classification head, it
scores by reading a token logit off a causal LM, and it is trained with LoRA against a
group-wise softmax. That path was **verified working here on 2× MI355X** (see below). The
group-wise loss is also a real modelling difference from the sentence-transformers folder's
per-pair BCE — negatives compete with the positive inside one softmax rather than being
independent binary targets.

Where the sentence-transformers folder still wins: it evaluates. FlagEmbedding's finetune
runner emits no metrics and selects no best checkpoint, so use `../sentence_transformers` when
you want an nDCG number out of the box, and this folder when you want the LLM-reranker
architecture or the BGE group-wise recipe.

Note that FlagEmbedding's **hard-negative mining** script (`scripts/hn_mine.py`) is *not* in the
`FlagEmbedding==1.4.0` wheel — it ships only in the GitHub repo. Don't plan around it being
importable from the venv.

## Install

Python 3.12. On this host `/` is at 94% (~97 GB free) and the existing per-folder venvs
already total 404 GB, so the venv lives **outside the repo** on the roomy volume:

```bash
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache
python3 -m venv /mnt/data_450g/envs/.env_train_flagembedding
source /mnt/data_450g/envs/.env_train_flagembedding/bin/activate
```

> This venv survived the 2026-08 repo reorg (it lives outside the repo); the in-repo campaign
> venvs were removed in that reorg. It originally also served a FlagEmbedding *embedding*
> trainer folder, which was removed in the reorg as redundant — that embedding-side
> capability is covered by `../../embedding/sentence_transformers`.

### AMD (ROCm) — the exact route that worked

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2   # -> 2.11.0+rocm7.2
pip install -r requirements_reranker_flagembedding.txt
ln -sf ../../../dev.env dev.env
```

Never install `flash-attn` here — the pinned build is CUDA-only. `utils.resolve_attn()` keeps
FlagEmbedding's `--use_flash_attn` at `False` whenever `torch.version.hip` is set, and the model
falls back to PyTorch SDPA.

### NVIDIA (CUDA)

Skip the ROCm index line; the default PyPI torch wheels are CUDA builds. `flash-attn==2.8.3` is
optional (needs the CUDA toolkit) and would be picked up by `--attn_implementation auto`.

Verify:

```bash
python -c "import FlagEmbedding, transformers, torch; print(transformers.__version__, torch.__version__)"
```

### Tested on NVIDIA H100 80GB — CUDA 13.0 (verified 2026-08-22)

**Verdict: WORKS on H100.** Single-GPU `encoder` reranker training completed with `rc=0`,
converging loss, and a saved model. Same `transformers<5` pin and `Trainer.tokenizer` shim as
MI355X — those are library constraints, not hardware ones, so they carry over unchanged. No
CUDA-specific code edit was needed; the ROCm-only guards in `utils.py` correctly no-op on CUDA
(`tf32` is left at its HF default instead of being forced to `None`, and `resolve_attn()` no
longer forces sdpa — see below).

Versions: Python 3.12.3, driver **580.173.02**, **`torch 2.13.0+cu130`** (native CUDA 13,
Hopper cc 9.0), `FlagEmbedding 1.4.0`, `transformers 4.57.1`, `accelerate 1.14.0`,
`datasets 5.0.1`, `peft 0.20.0`, `numpy 2.5.2`.

**Install that worked (venv on tmpfs, CUDA):**

```bash
python3 -m venv /dev/shm/h100/venv_flagemb && source /dev/shm/h100/venv_flagemb/bin/activate
pip install torch numpy            # -> torch 2.13.0+cu130, CUDA 13.0 (verify BEFORE the next step)
# The requirements pins torch==2.11.0, which has NO cu130 wheel. Install requirements WITHOUT
# the torch line so it does not downgrade you off the CUDA-13 build:
grep -v -E '^\s*torch==' requirements_reranker_flagembedding.txt > /tmp/reqs_notorch.txt
pip install -r /tmp/reqs_notorch.txt   # pins transformers==4.57.1 (<5, on purpose)
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # RE-VERIFY: FlagEmbedding must NOT have clobbered torch -> still 2.13.0+cu130
ln -sf ../../../dev.env dev.env
```

`transformers==4.57.1` (the `<5` pin) still ships `tokenizer.prepare_for_model()`, so the
reranker data pipeline that dies under 5.x runs fine here — verified by
`hasattr(PreTrainedTokenizerBase, 'prepare_for_model') == True`.

**Model choice.** `BAAI/bge-reranker-base` (the script default) is NOT in the shared cache and
the box's egress proxy 403s huggingface.co, so the encoder smoke used the cached, valid encoder
reranker **`BAAI/bge-reranker-v2-m3`** (`XLMRobertaForSequenceClassification`, 568M). To use the
tiny `bge-reranker-base` instead, `unset HTTP_PROXY HTTPS_PROXY ...` and let it download. Because
everything needed was cached, the run was forced fully offline
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) to skip the proxy-blocked HEAD freshness probes that
otherwise burn ~30s in retries before falling back to cache.

**Exact smoke command (single GPU, encoder, shipped 100-row sample):**

```bash
export CUDA_VISIBLE_DEVICES=7            # plain CUDA var; NO HIP_VISIBLE_DEVICES on NVIDIA
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --nproc_per_node=1 --master_port=29658 train_reranker_flagembedding.py \
  --reranker_type encoder --model_name_or_path BAAI/bge-reranker-v2-m3 \
  --output_dir /dev/shm/h100/out/flagembedding/smoke1gpu_enc \
  --batch_size 2 --train_group_size 4 --epochs 2 --logging_steps 10
```

That is `floor(100 / (2 × 1 × 1)) = 50` steps/epoch × 2 epochs = **100 optimizer steps**
(`train_group_size` only sets passages-per-query, it does not change the step count). A 6-epoch
repeat (`--epochs 6 --logging_steps 25`, lr 2e-5) ran **300 steps** to show clean convergence —
real log lines from that run:

```
[flagembedding] 1x NVIDIA H100 80GB HBM3 (torch 2.13.0+cu130, cuda 13.0)
[flagembedding] type=encoder model=BAAI/bge-reranker-v2-m3
{'loss': 0.7468, 'grad_norm': 0.1779,     'learning_rate': 1.67e-05, 'epoch': 1.0}
{'loss': 0.1038, 'grad_norm': 0.000145,   'learning_rate': 1.34e-05, 'epoch': 2.0}
{'loss': 0.0130, 'grad_norm': 0.000441,   'learning_rate': 1.01e-05, 'epoch': 3.0}
{'loss': 0.0013, 'grad_norm': 0.00187,    'learning_rate': 6.67e-08, 'epoch': 6.0}
{'train_runtime': 65.91, 'train_samples_per_second': 9.10, 'train_steps_per_second': 4.55, 'train_loss': 0.1074, 'epoch': 6.0}
Training complete. Model saved to .../smoke1gpu_enc_6ep
```

Loss falls 0.75 → 0.10 → 0.013 → 0.0013 with `grad_norm` collapsing to ~6e-8 — the model
memorizes the 100-row sample (expected on a toy set; per-step loss is noisy because negatives
are resampled each epoch, so read the epoch-boundary trend, not adjacent steps). The 100-step
2-epoch run finished identically (`train_runtime 25.2s`, `train_loss 0.263`).

**GPU-7 residency proof** — `nvidia-smi` filtered to GPU 7's UUID, sampled from inside the run
(the training PID is the only compute app on that GPU):

```
-- 04:21:15 GPU7[12343 MiB, 56%] :: PID 1589951, GPU-9eb34eec-...-e995e95239, 12334 MiB
-- 04:21:26 GPU7[12915 MiB,  5%] :: PID 1589951, GPU-9eb34eec-...-e995e95239, 12906 MiB
```

Peak ~13 GB VRAM for this encoder at `batch_size 2` / `group 4` — trivial against the 80 GB
card (vs 288 GB on MI355X); no OOM pressure and lots of headroom to raise `--batch_size`.
Saved output: `config.json`, `model.safetensors` (2.27 GB), `sentencepiece.bpe.model`,
tokenizer files, `checkpoint-100/` (or `-300/`), `training_args.bin`, `runs/`. Large
checkpoints were deleted after evidence capture (outputs lived on `/dev/shm`).

**flash-attn on the encoder path is moot:** `bge-reranker-v2-m3` is XLM-RoBERTa, whose HF
attention is SDPA regardless. `utils.resolve_attn()` returns `flash_attention_2` on CUDA only if
`import flash_attn` succeeds (not installed here → sdpa); it matters for the `llm`/`llm_layerwise`
decoder paths. To exercise flash-attn, `pip install flash-attn` (prebuilt CUDA wheel) and run an
`llm` smoke with `Qwen/Qwen3-0.6B` (cached) — not needed for the encoder verdict.

**Multi-GPU (deferred).** Only single-GPU was in scope for this wave (GPUs 0–3 were running a
co-tenant production job). A 2- or 8-GPU pass would reuse the MI355X recipe verbatim — bump
`--nproc_per_node`, keep `--gc_use_reentrant False` under DDP if `--gradient_checkpointing` is
on, and pick a free `--master_port` (29500 and the low-2965x ports collide on this shared host;
29658/29659 were free). DDP scales samples-per-step, not wall-clock, on a 100-row toy set.

### Tested on AMD Instinct MI355X — ROCm 7.2 (verified 2026-08-20)

**Verdict: works, with a pinned `transformers<5` and one compat shim.** All three of
single-GPU encoder, single-GPU LLM, and 2-GPU LLM training completed with `rc=0`, decreasing
loss, and a saved model/adapter. Versions: Python 3.12.3, ROCm 7.2.4, `torch 2.11.0+rocm7.2`
(HIP 7.2.26015), `FlagEmbedding 1.4.0`, `transformers 4.57.1`, `accelerate 1.14.0`,
`datasets 5.0.1`, `peft 0.20.0`, `sentence-transformers 5.7.0`.

Nothing failed for a ROCm reason. The blockers were library-version issues (see *Notes*).

**1. Encoder cross-encoder, single GPU** (`BAAI/bge-reranker-base`, shipped sample, 50 steps):

```bash
source /mnt/data_450g/envs/.env_train_flagembedding/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/mnt/data_1.5t/hf_cache
torchrun --nproc_per_node=1 --master_port=29811 train_reranker_flagembedding.py \
  --output_dir /mnt/data_450g/outputs/train_reranker_flagembedding/smoke1gpu_enc --epochs 1
```

```
[flagembedding] 2x AMD Instinct MI355X (torch 2.11.0+rocm7.2, hip 7.2.26015)
{'loss': 0.4515, 'grad_norm': 146.05, 'learning_rate': 9.2e-06, 'epoch': 0.1}
{'loss': 0.1979, 'grad_norm': 0.654, 'learning_rate': 5.2e-06, 'epoch': 0.5}
{'loss': 0.5061, 'grad_norm': 0.0184, 'learning_rate': 2.0e-07, 'epoch': 1.0}
Training complete. Model saved to .../smoke1gpu_enc
```

Output dir: `config.json model.safetensors tokenizer.json sentencepiece.bpe.model
checkpoint-50/ training_args.bin runs/`.

**2. LLM reranker — the thing sentence-transformers cannot do.** Verified on 2× MI355X with
the real BAAI model, `BAAI/bge-reranker-v2-gemma` (2.5B gemma, LoRA):

```bash
torchrun --nproc_per_node=2 --master_port=29811 train_reranker_flagembedding.py \
  --reranker_type llm --model_name_or_path BAAI/bge-reranker-v2-gemma \
  --output_dir /mnt/data_450g/outputs/train_reranker_flagembedding/smoke2gpu_gemma \
  --epochs 1 --batch_size 1 --train_group_size 4 --max_len 512 \
  --gradient_checkpointing --logging_steps 10 \
  --query_instruction "A: " --passage_instruction "B: "
```

```
trainable params: 7,372,800          # LoRA only, on both ranks
{'loss': 0.2845, 'grad_norm': 6.151, 'learning_rate': 8.2e-06, 'epoch': 0.2}
{'loss': 0.5750, 'grad_norm': 1.667, 'learning_rate': 2.2e-06, 'epoch': 0.8}
{'loss': 0.1966, 'grad_norm': 0.555, 'learning_rate': 2.0e-07, 'epoch': 1.0}
{'train_runtime': 13.05, 'train_samples_per_second': 7.663, 'train_steps_per_second': 3.832, 'train_loss': 0.3652, 'epoch': 1.0}
```

`rocm-smi` during that run: `07:40:43 use% 91 90` — both assigned GPUs busy together.
Saved: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, `checkpoint-50/`.

**3. LLM reranker, longer 2-GPU run** (`Qwen/Qwen3-0.6B`, 6 epochs, 300 steps) — loss falls
monotonically, which is the real convergence evidence:

```bash
torchrun --nproc_per_node=2 --master_port=29811 train_reranker_flagembedding.py \
  --reranker_type llm --model_name_or_path Qwen/Qwen3-0.6B \
  --output_dir /mnt/data_450g/outputs/train_reranker_flagembedding/smoke2gpu_llm \
  --epochs 6 --batch_size 1 --train_group_size 4 --max_len 512 \
  --gradient_checkpointing --logging_steps 25
```

```
{'loss': 1.734,  'grad_norm': 37.12, 'epoch': 0.5}
{'loss': 1.1877, 'grad_norm': 29.71, 'epoch': 2.0}
{'loss': 0.9152, 'grad_norm': 27.23, 'epoch': 4.5}
{'loss': 0.7336, 'grad_norm': 75.48, 'epoch': 6.0}
{'train_runtime': 61.04, 'train_samples_per_second': 9.829, 'train_steps_per_second': 4.915, 'train_loss': 1.1157, 'epoch': 6.0}
```

`rocm-smi` sampled every 4 s across the run, both GPUs loaded together:

```
-- 07:38:13 use% 94 94
-- 07:38:25 use% 93 97
-- 07:38:34 use% 94 96
-- 07:38:46 use% 86 92
```

Single vs 2 GPU on the same config: 1 GPU did 100 steps in 21.2 s (4.72 steps/s), 2 GPUs did
300 steps in 61.0 s (4.92 steps/s) — i.e. **2× the samples per step at the same step rate**, so
DDP is scaling the work, not the wall clock. Do not read a speedup number off a 100-row sample.

**Not verified: `llm_layerwise`.** It fails on any non-MiniCPM base with
`AttributeError: 'Qwen3Config' object has no attribute 'scale_depth'` — FlagEmbedding's
layerwise modeling code is hardwired to the MiniCPM architecture. It needs
`BAAI/bge-reranker-v2-minicpm-layerwise` (or a MiniCPM base); that model was not downloaded
here. The flag is wired up and the failure is a base-model requirement, not a ROCm problem.

## Environment & secrets

`dev.env` is symlinked to the repo root (`ln -sf ../../../dev.env dev.env`) and loaded with
`load_dotenv("dev.env")`. It supplies `HF_TOKEN` for gated models. The BGE and Qwen defaults
used here are ungated. Never print or commit the token.

Point `HF_HOME` at the shared cache so the 5 GB gemma weights are not re-downloaded:

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache
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
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/mnt/data_1.5t/hf_cache
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
  --output_dir /mnt/data_450g/outputs/train_reranker_flagembedding/run1 \
  --batch_size 4 --train_group_size 8 --epochs 2 --lr 1e-5 --max_len 1024 \
  --gradient_checkpointing --save_merged_lora_model True \
  > train_reranker_flagembedding.log 2>&1 &
```

**What "working" looks like:** `trainable params: ...` once per rank (LLM types), then
`{'loss': ...}` every `--logging_steps`, a `{'train_runtime': ...}` summary, and a model or
adapter written under `--output_dir`. There is no metric line — this trainer does not evaluate.

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

## Hardware support & evidence

- **AMD: tested** — 1× and 2× MI355X (gfx950, 288 GB), ROCm 7.2.4, `torch 2.11.0+rocm7.2`,
  Python 3.12.3, verified 2026-08-20. Encoder on 1 GPU (50 steps, `rc=0`, model saved); LLM
  reranker `bge-reranker-v2-gemma` on 2 GPUs (50 steps, `rc=0`, adapter saved, GPUs at 90–91%);
  LLM reranker `Qwen3-0.6B` on 2 GPUs (300 steps, loss 1.73 → 0.73, GPUs at 86–97%). No
  ROCm-specific code change was needed.
- **NVIDIA:** untested here. `--attn_implementation auto` picks `flash_attention_2` when a CUDA
  torch and `flash_attn` are present; the `tf32` guard is a no-op on CUDA.
- **Other hardware (upstream claims — not verified here):** none. FlagEmbedding's own docs
  claim nothing beyond CUDA-class GPUs (its trainer is plain PyTorch + HF Transformers, so
  other torch backends are possible in principle, but no MPS/TPU/NPU support is claimed
  upstream).
- Only GPUs 0 and 1 were assigned on this host, so 4/8-GPU scaling is unverified.
- `llm_layerwise` unverified — needs a MiniCPM-architecture base (see above).

## Notes

- **`transformers<5` is a hard requirement for this folder.** FlagEmbedding 1.4.0's reranker
  dataset (`AbsDataset.create_one_example`) calls `tokenizer.prepare_for_model()`, which
  transformers 5.x removed from the standard tokenizers; the run dies on the **first batch**
  with `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model`. Setting
  `use_fast=False` does not help — 5.x drops it from the slow tokenizer path too. 4.57.1 is
  pinned and verified. (The same call appears throughout `FlagEmbedding/inference/reranker/`,
  so inference has the same constraint.)
- **`Trainer.tokenizer` shim.** FlagEmbedding's `_save()` and the reranker runners still use the
  pre-4.46 `tokenizer` name. `utils.patch_transformers5_compat()` aliases
  `Trainer.tokenizer` → `processing_class` and rewrites the `tokenizer=` constructor kwarg. On
  4.57.1 this only silences a deprecation; on 5.x it is what keeps the save step alive. It is
  kept so the folder still runs if you raise the pin.
- **Gradient checkpointing + DDP.** With `--gradient_checkpointing` on 2 ranks, reentrant
  checkpointing raises `RuntimeError: Expected to mark a variable ready only once`. The script
  therefore always passes `gradient_checkpointing_kwargs={"use_reentrant": false}`
  (`--gc_use_reentrant False`, the default). Flip it only on a single GPU.
- `tf32` is forced to `None` when `torch.version.cuda is None` — setting it on ROCm raises.
- No `flash-attn`: `--use_flash_attn` stays `False` on ROCm and the model uses SDPA.
- Default master port is **29811** here (29500 collides on this host). Port 29810 was used by
  the FlagEmbedding embedding folder removed in the 2026-08 reorg, so it is free again.
- **Batch geometry.** `steps/epoch = floor(rows / (--batch_size × world_size × --grad_accum))`.
  The 100-row sample at `--batch_size 2` gives 50 steps on 1 GPU. Push `--batch_size` too high
  on a small dataset and the step count silently collapses toward zero — the same trap
  documented in the sentence-transformers siblings.
- The `VRAM%` reported by `rocm-smi` on this host sits at 46–48% before any of these jobs start
  (shared machine), so per-GPU VRAM from `rocm-smi` is not attributable to a single run.
- Harmless warnings: `destroy_process_group() was not called before program exit`,
  `expandable_segments not supported on this platform`, and the `Trainer.tokenizer is now
  deprecated` line from the shim.
- Outputs went to `/mnt/data_450g/outputs/...` because `/` is at 94%; large checkpoints were
  deleted after the evidence was captured.
- The shipped OTel sample is for pipeline validation only — 100 rows will not produce a useful
  reranker.
