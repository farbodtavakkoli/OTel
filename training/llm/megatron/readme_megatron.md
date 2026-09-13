# Setup & usage — `train_llm_megatron.py`

## Overview & when to use

Large-scale training on [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) —
Megatron Core's TP / PP / DP / CP / EP parallelism, the distributed optimizer, FP8 and FP4
support, and the fused kernels NVIDIA ships in TransformerEngine. Megatron-LM is a
**pre-training** codebase, so the realistic job for it here is **continued pre-training /
domain-adaptive pretraining** of an existing model that has been converted from Hugging Face
into Megatron format. Pick it over the other trainers in this repo when you need maximum
throughput at 8B+ scale and are willing to pay for it with a heavy install
(TransformerEngine, Apex) and an offline data-preprocessing step. It is a poor fit for
quick SFT/DPO/GRPO iteration — see "What this is not good for" below.

Files in this folder:
- `train_llm_megatron.py` — thin launcher/validator. Reads the TOML config, turns every key
  into the matching `pretrain_gpt.py` flag, checks the common misconfigurations, and execs
  `torchrun ... pretrain_gpt.py` from your Megatron-LM clone.
- `megatron_cpt_config.toml` — the run configuration: model geometry, parallelism, optimizer,
  data paths, checkpointing, logging. This is where you actually make changes.
- `preprocess_megatron_data.py` — wraps upstream `tools/preprocess_data.py` to turn a JSONL
  corpus into the indexed `.bin`/`.idx` format Megatron requires, and prints the resulting
  `data_path` prefix. Chat JSONL (`messages`) is flattened to `{text}` automatically.
- `utils.py` — the stdlib flatten helpers (`flatten_chat_record`,
  `_maybe_flatten_chat_jsonl`) used by the preprocessor; kept separate so they can be
  imported and tested without any Megatron or dotenv dependency.
- `data/OTel_LLM_sample_10.jsonl` — the shipped 10-row chat sample (see Data below).
- `requirements_megatron.txt` — dependencies, with the NGC container option, the ROCm
  container option, the source installs, and the `--no-build-isolation` cases called out.
- `readme_megatron.md` — this file.

> **Hardware coverage.** The preprocessing → training path runs end to end on **AMD Instinct
> MI355X (gfx950, ROCm 7.2, torch 2.11.0+rocm7.2)** at 2 and 8 GPUs — DP=2 and TP=2 on two,
> then TP2/PP2/DP2, DP=8 and TP2/DP4 on eight — with a tiny custom GPT config. It **works
> with changes**: four flags must be flipped because TransformerEngine and Apex are CUDA-only,
> and those same four flags are all that is needed at 8 GPUs. See
> ["AMD MI355X (ROCm 7.2)"](#amd-mi355x-rocm-72) and the
> ["8-GPU run"](#8-gpu-run-8x-mi355x-rocm-724) subsection. On **NVIDIA H100 in the NGC
> PyTorch container** the same path runs with all four flags at their Megatron defaults (ON),
> single-GPU and TP=2 across two GPUs — see
> ["NVIDIA H100 (NGC PyTorch container)"](#nvidia-h100-ngc-pytorch-container). Not covered on
> either vendor: the Llama-3.1-8B geometry, Megatron-Bridge checkpoint conversion, and
> checkpoint saving.
>
> Megatron has hundreds of flags that move between releases; run
> `python pretrain_gpt.py --help` in your checkout before a long run.

### What this is not good for

Megatron-LM proper has no SFT/DPO/GRPO dataloader for GPT models: `pretrain_gpt.py` consumes
packed pre-training token streams from an indexed dataset, with no chat template and no
prompt masking. For post-training on this stack use
**[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** or NeMo (on ROCm,
AMD's [Primus](https://github.com/AMD-AGI/Primus)). This folder covers continued
pre-training only.

## 1. Install

### NVIDIA (CUDA)

Use the NGC PyTorch container unless you have a strong reason not to. TransformerEngine,
Apex and cuDNN fused attention are pre-built and version-matched there:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace -v /data:/data -it \
  nvcr.io/nvidia/pytorch:<YY.MM>-py3 bash
```

Then, inside the container (or inside a venv):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && pip install -e . && cd ..     # MAX_JOBS=4 pip install -e . if the build OOMs
pip install -r requirements_megatron.txt
```

If you are **not** in the container, TransformerEngine must be built against your installed
torch, which means disabling build isolation (otherwise pip resolves a second torch inside
the isolated build environment and the extension links against the wrong one):

```bash
pip install --no-build-isolation transformer-engine[pytorch]
```

### AMD (ROCm)

Megatron-LM is NVIDIA-first, but AMD ships a maintained ROCm path — container-first, like
the NVIDIA one:

- **`rocm/megatron-lm` Docker images.** AMD publishes a ROCm-enabled Megatron-LM image on
  Docker Hub; current tags include `v26.1` (January 2026), `v25.11`, `v25.10`, and
  architecture-specific tags such as `v25.9_gfx942` (MI300X) and `v25.9_gfx950` (MI350X):
  https://hub.docker.com/r/rocm/megatron-lm/tags

  ```bash
  docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
    --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v $PWD:/workspace -v /data:/data -it rocm/megatron-lm:v26.1 bash
  ```

- **AMD Primus.** AMD's [Primus](https://github.com/AMD-AGI/Primus) training framework uses
  Megatron-LM as one of its backends (alongside TorchTitan and JAX MaxText) and is AMD's
  recommended entry point on Instinct GPUs, shipped as `rocm/primus` Docker images. The
  ROCm AI-Ecosystem docs carry a "Training with Primus and Megatron" recipe:
  https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/training/recipes/primus-megatron.html

Inside the ROCm container the TransformerEngine equivalent is pre-built; do not
`pip install transformer-engine` on top of it. This folder's launcher and TOML are
hardware-agnostic (they just build a `torchrun` command line), but the shipped flags were
read from NVIDIA upstream — re-check attention/fusion flags against the ROCm fork before a
long run. See "Hardware support" below for sources.

#### AMD MI355X (ROCm 7.2)

Upstream NVIDIA/Megatron-LM trains on MI355X with a plain ROCm PyTorch wheel and **no
TransformerEngine and no Apex**, but four flags must be flipped (all of them TE/Apex fused
kernels that default to on). No source patch is needed — every change goes through the
launcher's `--set`, so `megatron_cpt_config.toml` is untouched.

Stack:

| | |
|---|---|
| Hardware | AMD Instinct MI355X (gfx950, 288GB) |
| Stack | ROCm 7.2.4, Python 3.12 |
| torch | `2.11.0+rocm7.2` (`torch.version.hip` = `7.2.26015`), `triton-rocm` 3.6.0 |
| Megatron-LM | commit `f481e6361520dcac1554891a6ae83b353eb1d91b`, `megatron-core 0.20.0+f481e63` |
| TransformerEngine / Apex / flash-attn | **not installed** (CUDA-only; do not try) |

**Route: venv + source clone, not the container.** `rocm/megatron-lm:v26.1` is AMD's
documented route and the better one for a production run (it ships AMD's TE/hipBLASLt/CK
stack pre-built), but it is a ~116 GB image. The pip route below is what the steps that
follow cover.

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # preprocessed data + training artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache

cd training/llm/megatron
python3 -m venv .env_megatron                      # git-ignored via .env_*/
source .env_megatron/bin/activate                  # needed: torchrun must be on PATH

pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0
pip install pybind11 "packaging>=24.2" numpy       # build deps for the helpers_cpp extension
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git .env_megatron/Megatron-LM
pip install --no-build-isolation -e .env_megatron/Megatron-LM
pip install -r requirements_megatron.txt
```

Notes on the install itself:
- The clone lives **inside** the git-ignored venv directory so `git status` stays clean.
- `pip install --no-build-isolation -e Megatron-LM` fails at metadata generation unless
  `pybind11` and `packaging` are already in the venv — install them first, in that order.
- `megatron/core/datasets/helpers_cpp` (the pybind11 C++ dataset indexer) **builds fine** on
  the ROCm host; it is plain C++ with no CUDA in it.
- `import megatron.core` then prints
  `UserWarning: Transformer Engine and Apex are not installed. Falling back to Torch
  optimizers` (and the same for LayerNorm / `multi_tensor_applier`). These warnings are
  expected and harmless — they are the fallbacks that make the ROCm run possible.

**Preprocessing** runs unmodified:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
python preprocess_megatron_data.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --tokenizer-model Qwen/Qwen2.5-0.5B \
  --output-prefix $OUTPUT_DIR/train_llm_megatron/otel \
  --workers 8
# -> .../otel_text_document.bin (49968 B) + .idx; chat `messages` flattening works as documented
```

The tokenizer is the one deviation from the shipped defaults: `meta-llama/Llama-3.1-8B` is
gated and returns `403 Cannot access gated repo` without access, so substitute an ungated
tokenizer. Nothing else about the preprocessing path changes.

**Training** — the smoke below uses a tiny custom GPT (8 layers / hidden 1024) instead of the
shipped Llama-3.1-8B geometry, with **saving off** and **no checkpoint load** (there is no
converted Megatron checkpoint to load). `--set key=false` makes the launcher omit the flag
entirely, which is how `save`, `load` and `tensorboard_dir` are switched off without editing
the TOML:

```bash
source .env_megatron/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
export $(grep -v '^#' ../dev.env | xargs)          # HF_TOKEN, never echoed

python train_llm_megatron.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --nproc-per-node 2 --train-iters 200 \
  --data-path $OUTPUT_DIR/train_llm_megatron/otel_text_document \
  --set master_port=29750 \
  --set transformer_impl='"local"' \
  --set tokenizer_model='"Qwen/Qwen2.5-0.5B"' \
  --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
  --set num_attention_heads=16 --set num_query_groups=4 \
  --set seq_length=1024 --set max_position_embeddings=1024 \
  --set use_rope_scaling=false \
  --set no_rope_fusion=true \
  --set no_persist_layer_norm=true \
  --set no_gradient_accumulation_fusion=true \
  --set global_batch_size=16 --set micro_batch_size=2 \
  --set lr_warmup_iters=10 --set lr_decay_iters=200 \
  --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=10000 \
  --set log_interval=10 --set log_throughput=true \
  --set load=false --set save=false --set save_interval=100000 \
  --set no_load_optim=false --set no_load_rng=false \
  --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
  --set tensorboard_dir=false
```

**What a healthy run looks like** (DP=2, TP=1): each `iteration N/200` line reports a finite
`lm loss` falling monotonically and `number of nan iterations: 0`. With this tokenizer, loss
starts at 11.97 — `ln(151936)`, exactly right for a randomly-initialised model — which is the
quickest check that the data and tokenizer line up.

TP=2 works too: same command plus `--set tensor_model_parallel_size=2`.

**Quirks — the four flags you must flip on ROCm.** Each surfaces as a hard failure, in this
order, and each is a TE/Apex fused kernel that Megatron enables by default:

| Config change | Failure without it |
|---|---|
| `transformer_impl = "local"` (TOML ships `"transformer_engine"`) | TE is CUDA-only and is not installed; the `transformer_engine` spec cannot be built. |
| `no_rope_fusion = true` | `ValueError: apply_rope_fusion is not available. Please install TE >= 1.4.` |
| `no_persist_layer_norm = true` | `AssertionError: persist_layer_norm not supported by torch LayerNorm` (Apex fused persistent LN). |
| `no_gradient_accumulation_fusion = true` | `RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found` (Apex `--cuda_ext`). |

And one capability you lose:

- **`sequence_parallel` cannot be used without TE.** With `transformer_impl = "local"` it
  fails at model build with `AssertionError: sequence parallel not supported by torch
  LayerNorm`, so TP > 1 runs without sequence parallelism. AMD's `rocm/megatron-lm` image
  ships a ROCm TE build, which restores it.

Other ROCm observations:

- **Never `pip install flash-attn`.** `attention_backend = "auto"` resolves to Megatron's
  unfused torch attention with `transformer_impl = "local"` (the argument dump shows
  `use_flash_attn ... False`), which is the correct ROCm path here.
- `recompute_granularity = "selective"` works and only emits a cosmetic warning about
  `core_attn` recompute being unnecessary under TE.
- `use_distributed_optimizer` + `overlap_grad_reduce` + `overlap_param_gather` all work over
  RCCL, and are on in every command here.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  which the launcher exports, are accepted by the ROCm build (HIP maps both).
- **You must `source` the venv**, not just call `.env_megatron/bin/python train_llm_megatron.py`:
  the launcher shells out to bare `torchrun`, which fails with
  `FileNotFoundError: ... 'torchrun'` if the venv's `bin/` is not on `PATH`.
- Not covered on ROCm from this folder: the Llama-3.1-8B geometry, Megatron-Bridge checkpoint
  conversion, checkpoint save/resume, CP > 1, and FP8. (TP=2, PP=2 and DP=8 all work — see
  the 8-GPU subsection below.)

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**Nothing beyond the 2-GPU recipe changes.** The four-flag recipe above scales from 2 to all
8 MI355X unmodified: no source patch, no extra package, no RCCL/NCCL tuning variable, no
batch-size rescue. `megatron_cpt_config.toml` stays untouched — every change goes through the
launcher's `--set`. Three layouts train cleanly:

| Config | TP | PP | CP | DP | global batch |
|---|---|---|---|---|---|
| **A** `tp2_pp2_dp2` | 2 | 2 | 1 | 2 | 32 |
| **B** `dp8` | 1 | 1 | 1 | 8 | 64 |
| **C** `tp2_pp1_dp4` | 2 | 1 | 1 | 4 | 32 |

All three use the same toy geometry as the 2-GPU pass (8 layers / hidden 1024 / 16 heads /
seq 1024), the same preprocessed dataset, 200 iterations, and a distinct `--set master_port`
per layout.

Launch (config A; B and C differ only in the two parallel sizes, `global_batch_size`
and `master_port`):

```bash
cd training/llm/megatron
source .env_megatron/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -a; . ./dev.env; set +a                        # HF_TOKEN, never echoed
python -c "import torch;assert torch.cuda.device_count()==8"   # assert before training

python train_llm_megatron.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --nproc-per-node 8 --train-iters 200 \
  --data-path $OUTPUT_DIR/train_llm_megatron/otel_text_document \
  --set master_port=29630 \
  --set tensor_model_parallel_size=2 \
  --set pipeline_model_parallel_size=2 \
  --set context_parallel_size=1 \
  --set sequence_parallel=false \
  --set transformer_impl='"local"' \
  --set tokenizer_model='"Qwen/Qwen2.5-0.5B"' \
  --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
  --set num_attention_heads=16 --set num_query_groups=4 \
  --set seq_length=1024 --set max_position_embeddings=1024 \
  --set use_rope_scaling=false \
  --set no_rope_fusion=true \
  --set no_persist_layer_norm=true \
  --set no_gradient_accumulation_fusion=true \
  --set global_batch_size=32 --set micro_batch_size=2 \
  --set lr_warmup_iters=10 --set lr_decay_iters=200 \
  --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=10000 \
  --set log_interval=10 --set log_throughput=true \
  --set load=false --set save=false --set save_interval=100000 \
  --set no_load_optim=false --set no_load_rng=false \
  --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
  --set tensorboard_dir=false
```

Check Megatron's own argument dump to confirm the process groups were really built and not
silently collapsed: for config A it should report `inprocess_active_world_size 8` with
`tensor_model_parallel_size 2 / pipeline_model_parallel_size 2 / context_parallel_size 1 /
data_parallel_size 2`.

**What changes from the 2-GPU setup:** only `--nproc-per-node 8`, the three parallel sizes,
`global_batch_size`, and the port. No new package, no RCCL/NCCL environment tuning, and the
same four fused-kernel flags — they are a TE/Apex-absence issue, not a GPU-count issue.

**Notes that still apply at 8 GPUs:**

- `sequence_parallel` must stay `false`. It is unusable without TE (`AssertionError: sequence
  parallel not supported by torch LayerNorm`), and note the *other* direction of the same
  constraint: turning it on with `tensor_model_parallel_size = 1` makes Megatron abort with
  "Cannot use sequence parallelism without tensor parallelism". Config B (TP=1) would hit
  that, so SP is off everywhere here.
- Check `.env_megatron/bin/activate` for a stale `export CUDA_VISIBLE_DEVICES=...` left by an
  earlier session. Always re-export both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`
  after `source`, and assert `torch.cuda.device_count() == 8` before launching.
- Keep `--set save=false --set save_interval=100000` for smoke runs. Nothing is then written
  beyond logs.

Not covered at 8 GPUs: `context_parallel_size > 1`, TP > 2, FP8, the real Llama-3.1-8B
geometry, and checkpoint save/load.

### NVIDIA H100 (NGC PyTorch container)

This is the primary, intended route. Where the MI355X venv has to flip **four** TE/Apex
fused-kernel flags OFF, the NGC PyTorch container ships them pre-built and version-matched,
so all four stay at their **Megatron defaults (ON)** — no `--set no_*` overrides and
`transformer_impl` stays `"transformer_engine"`. `megatron_cpt_config.toml` is untouched —
every change is a launcher `--set`, same as MI355X.

Stack:

| | |
|---|---|
| Hardware | NVIDIA H100 80GB HBM3 (Hopper, cc 9.0) |
| Route | **NGC PyTorch container** `nvcr.io/nvidia/pytorch:25.06-py3` |
| torch | `2.8.0a0+5228986c39.nv25.06`, **CUDA 12.9** (container-shipped; do not re-install torch) |
| Megatron-LM | commit `f481e6361520dcac1554891a6ae83b353eb1d91b`, `megatron-core 0.20.0+f481e63` |
| TransformerEngine / Apex / flash-attn | `transformer-engine 2.4.0`, Apex, `flash-attn 2.7.4.post1`, cuDNN 9.10.2 — all pre-built in the container |
| transformers / other extras | `transformers 5.15.1` from `requirements_megatron.txt` (the container ships torch/TE/Apex but **not** transformers/dotenv/nltk) |

**Route: the NGC container, pinned by GPU UUID.** This needs Docker with the NVIDIA runtime
(CDI). If an HTTP proxy sits in front of the host, `nvcr.io` may be 403-blocked — **unset the
proxy** to pull. Pin the container to specific physical GPUs **by UUID** so it cannot see
co-tenant GPUs; `nvidia-smi` inside the container should then show exactly the devices you
named.

```bash
# 1. Pull (proxy MUST be unset for nvcr.io; ~14 GB compressed). Pick a recent YY.MM-py3 tag.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull nvcr.io/nvidia/pytorch:25.06-py3

# 2. Launch pinned to specific GPUs by UUID (nvidia-smi -L to get yours), mounting the repo,
#    the HF model cache, and a scratch output dir. --ipc=host + the ulimits are the
#    NGC-recommended flags (see requirements_megatron.txt / section 1).
sudo docker run -d --name megatron_h100 \
  --gpus '"device=GPU-xxxxxxxx-...."' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/workspace:/work \
  -v /path/to/hf_cache:/models \
  -v /path/to/scratch_out:/out \
  -e HF_HOME=/models \
  -w /work/training/llm/megatron \
  nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity
sudo docker exec megatron_h100 nvidia-smi -L   # MUST show exactly the GPUs you named

# 3. Inside the container: clone Megatron-LM, install it, add the pure-Python extras.
sudo docker exec megatron_h100 bash -c '
  cd /opt && git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM && git fetch --depth 1 origin f481e63 && git checkout f481e63   # optional: pin
  pip install --no-build-isolation -e .            # reuses the container torch/TE
  pip install "nvidia-resiliency-ext>=0.6.0"       # see nvrx note below
  cd /work/training/llm/megatron && pip install -r requirements_megatron.txt'
```

> **nvrx import note.** With Megatron `f481e63`, `import
> megatron.core` in the stock `25.06` container dies with
> `AttributeError: module 'nvidia_resiliency_ext' has no attribute '__version__'`
> (in `dist_checkpointing/strategies/nvrx.py`). The container ships
> `nvidia-resiliency-ext 0.4.0`, which (a) is below Megatron's `NVRX_MIN_VERSION = "0.6.0"`
> and (b) does not expose `__version__` as a module attribute. **Fix:**
> `pip install "nvidia-resiliency-ext>=0.6.0"` (0.6.0 exposes `__version__` and clears the
> assert). A newer NGC tag may already ship a compatible nvrx.

`pip install -r requirements_megatron.txt` does **not** replace the container's torch —
re-check with `torch.__version__` and `torch.cuda.is_available()` after installing.

**Confirm TE/Apex are engaged.** `import
megatron.core` should print **no** "Transformer Engine and Apex are not installed. Falling
back to Torch optimizers" warning (that line is the ROCm route's signature). Megatron's
resolved argument dump should show every fused kernel the ROCm route disables turned **on**:

```
transformer_impl ................................ transformer_engine
apply_rope_fusion ............................... True
gradient_accumulation_fusion .................... True
no_persist_layer_norm ........................... False      # i.e. persistent LN is ON
```

**Preprocessing** (identical to the ROCm route, tokenizer substituted — see below):

```bash
# HF_HUB_OFFLINE forces the cached tokenizer; the sample's chat `messages` are auto-flattened.
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 megatron_h100 \
  python preprocess_megatron_data.py \
    --megatron-repo /opt/Megatron-LM \
    --tokenizer-model Qwen/Qwen3-0.6B \
    --output-prefix /out/otel \
    --workers 8
# -> /out/otel_text_document.bin (49968 B) + .idx  (same size as the ROCm route produces)
```

Tokenizer deviation (same class as on ROCm): `meta-llama/Llama-3.1-8B` is gated, so an
ungated tokenizer is substituted — `Qwen/Qwen3-0.6B` here, for both preprocessing and
training. Its vocab is 151643, so a randomly-initialised model starts at `lm loss ≈
ln(151643) ≈ 11.93`, which is the quickest check that data and tokenizer line up. Nothing
else about the preprocessing path changes.

**Training — short foreground smoke** (README §4), tiny custom GPT (8 layers / hidden 1024),
TE **on**, saving off. Note there are **no** `--set no_rope_fusion` / `no_persist_layer_norm`
/ `no_gradient_accumulation_fusion` / `transformer_impl="local"` overrides — those are
exactly the four the ROCm route adds and the NVIDIA route drops:

```bash
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_TOKEN="$HF_TOKEN" megatron_h100 \
  python train_llm_megatron.py \
    --megatron-repo /opt/Megatron-LM \
    --nproc-per-node 1 --train-iters 20 \
    --data-path /out/otel_text_document \
    --set master_port=29660 \
    --set transformer_impl='"transformer_engine"' \
    --set tokenizer_model='"Qwen/Qwen3-0.6B"' \
    --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
    --set num_attention_heads=16 --set num_query_groups=4 \
    --set seq_length=1024 --set max_position_embeddings=1024 \
    --set use_rope_scaling=false \
    --set global_batch_size=8 --set micro_batch_size=1 \
    --set lr_warmup_iters=2 --set lr_decay_iters=20 \
    --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=100000 \
    --set log_interval=1 --set log_throughput=true \
    --set load=false --set save=false --set save_interval=100000 \
    --set no_load_optim=false --set no_load_rng=false \
    --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
    --set tensorboard_dir=false
```

**What a healthy smoke looks like** (DP=1, TP=1): each `iteration N/20` line carries a finite
`lm loss` falling monotonically from ~11.93 and `number of nan iterations: 0`, ending with
`INFO - training/llm/megatron - command completed successfully`.

**Training — longer pass with checkpoint saving ON.** Same command as above but
`--train-iters 200 --save /out/ckpt --set global_batch_size=16 --set micro_batch_size=2 --set
lr_warmup_iters=10 --set lr_decay_iters=200 --set log_interval=10 --set save_interval=100`
(and drop `--set save=false`). Megatron then writes `torch_dist`-format checkpoints at
iterations 100 and 200, logging `successfully saved checkpoint from iteration 200 to
/out/ckpt`. The checkpoint for this toy geometry is ~11 GB (`iter_0000100/`,
`iter_0000200/`, `latest_checkpointed_iteration.txt`); write it to scratch, not into the repo.

**Scaling past two GPUs** needs nothing new on the software side: launch the same container
with `--gpus '"device=…,…"'` listing the GPUs, pass `--nproc-per-node N`, and set the
parallel sizes with `--set tensor_model_parallel_size` / `pipeline_model_parallel_size` /
`context_parallel_size` exactly as the MI355X 8-GPU section does. The **difference from ROCm at
multi-GPU is `sequence_parallel`**: on ROCm it must stay `false` (torch LayerNorm cannot do
SP without TE), but with TE present you can set `sequence_parallel=true` whenever
`tensor_model_parallel_size > 1` and reclaim the activation memory. FP8 (`--fp8-format`) also
becomes available on Hopper via TE. FP8, `sequence_parallel`, CP > 1 and the real
Llama-3.1-8B geometry are not covered here; TP=2 across two GPUs is, below.

### 2-GPU run (2× H100, TP=2)

`tensor_model_parallel_size=2` splits every attention/MLP weight matrix across the two ranks.
Same container, same `megatron_cpt_config.toml`, same toy geometry as the single-GPU pass —
only the launcher flags change (`--nproc-per-node 2` + the three parallel sizes), and the run
saves a TP-sharded `torch_dist` checkpoint (two shards, one per rank). `sequence_parallel` is
left `false` here; enabling it at TP>1 is documented upstream for NVIDIA.

**Container — pinned to exactly two GPUs.** On a shared node, launch the container with
`--gpus '"device=6,7"'` (substitute your free devices) so it sees only those two;
`nvidia-smi -L` inside then enumerates them as GPU 0 and GPU 1. Install is identical to the
single-GPU route (clone Megatron `f481e63`, `pip install "nvidia-resiliency-ext>=0.6.0"`,
`pip install -r requirements_megatron.txt`); make Megatron-core importable with
`PYTHONPATH=/opt/Megatron-LM` (the pure-Python package needs no build step — the fused kernels
come from the container's TE/Apex).

```bash
sudo docker run -d --name megatron_h100_2gpu \
  --gpus '"device=6,7"' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/workspace:/work -v /path/to/hf_cache:/models -v /path/to/scratch_out:/out \
  -e HF_HOME=/models -w /work/training/llm/megatron \
  nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity
sudo docker exec megatron_h100_2gpu nvidia-smi -L   # MUST show exactly 2 GPUs

# preprocess once (same as single-GPU; -> /out/otel_text_document.bin + .idx)
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-LM megatron_h100_2gpu \
  python preprocess_megatron_data.py --megatron-repo /opt/Megatron-LM \
    --tokenizer-model Qwen/Qwen3-0.6B --output-prefix /out/otel --workers 8

# TP=2 training — note --nproc-per-node 2 and --set tensor_model_parallel_size=2
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-LM -e HF_TOKEN="$HF_TOKEN" megatron_h100_2gpu \
  python train_llm_megatron.py \
    --megatron-repo /opt/Megatron-LM \
    --nproc-per-node 2 --train-iters 150 \
    --data-path /out/otel_text_document --save /out/ckpt \
    --set master_port=29673 \
    --set transformer_impl='"transformer_engine"' \
    --set tensor_model_parallel_size=2 \
    --set pipeline_model_parallel_size=1 --set context_parallel_size=1 \
    --set tokenizer_model='"Qwen/Qwen3-0.6B"' \
    --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
    --set num_attention_heads=16 --set num_query_groups=4 \
    --set seq_length=1024 --set max_position_embeddings=1024 --set use_rope_scaling=false \
    --set global_batch_size=16 --set micro_batch_size=2 \
    --set lr_warmup_iters=10 --set lr_decay_iters=150 \
    --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=100000 \
    --set log_interval=10 --set log_throughput=true \
    --set load=false --set save_interval=100 \
    --set no_load_optim=false --set no_load_rng=false \
    --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
    --set tensorboard_dir=false
```

**TP=2 config banner + parallel state built** (the launcher's own validator prints the geometry,
then Megatron confirms the 2-rank TP group and no TE fallback):

```
INFO - training/llm/megatron - world size 2, tp*pp*cp = 2, data-parallel degree = 1
using world size: 2, data-parallel size: 1, context-parallel size: 1, tensor-model-parallel size: 2, pipeline-model-parallel size: 1
  transformer_impl ................................ transformer_engine
  apply_rope_fusion ............................... True
  gradient_accumulation_fusion .................... True
  no_persist_layer_norm ........................... False
> initialized tensor model parallel with size 2
> number of parameters on (tensor, gtp_remat, pipeline) model parallel rank (0, 0, 0): 200557568
> number of parameters on (tensor, gtp_remat, pipeline) model parallel rank (1, 0, 0): 200557568
```

There should be **no** "Transformer Engine and Apex are not installed. Falling back to Torch
optimizers" line. Megatron reports the parameter count **per tensor-parallel rank**: each of
the two ranks owns its shard of the split matrices.

Both ranks log the same synchronized `lm loss`, which should fall monotonically from
`≈ ln(vocab_size)` with `number of nan iterations: 0` and `number of skipped iterations: 0`.

**TP-sharded checkpoint.** With `--save` and `--set save_interval=100`, Megatron writes
`torch_dist` checkpoints. The save banner's `t 1/2` and the **two shard files** are the
on-disk proof that the weights were split across the 2 TP ranks:

```
saving checkpoint at iteration 100 to /out/ckpt in torch_dist format
successfully saved checkpoint from iteration 150 to /out/ckpt [ t 1/2, gtp_remat 1/1, p 1/1 ]
/out/ckpt/iter_0000150/__0_0.distcp   /out/ckpt/iter_0000150/__1_0.distcp   (latest_checkpointed_iteration.txt = 150)
```

The checkpoint is ~11 GB — keep it in `/out` scratch, not in the repo.

**8 GPUs on NVIDIA is not covered here**; the MI355X 8-GPU section runs that geometry. The
software path is unchanged — same container, `--nproc-per-node 8`, `--set
tensor_model_parallel_size=2 pipeline_model_parallel_size=2 context_parallel_size=1`.

**NVIDIA quirks / what changes vs ROCm:**

- **The four fused-kernel flags are reversed.** ROCm adds
  `transformer_impl="local"`, `no_rope_fusion=true`, `no_persist_layer_norm=true`,
  `no_gradient_accumulation_fusion=true`. NVIDIA adds **none** of them — the TOML default
  `transformer_impl="transformer_engine"` and all three fusions-on defaults just work.
- **Pin the container by GPU UUID, not index**, on a shared node. `--gpus
  '"device=GPU-<uuid>"'` makes the container see exactly that GPU (it re-enumerates from
  index 0 inside), so there is no way to accidentally touch a co-tenant GPU. Verify with
  `nvidia-smi -L` inside the container before training.
- **`unset` the proxy to pull from `nvcr.io`** (403 otherwise). `HF_HUB_OFFLINE=1` + a cached
  tokenizer then means training needs no network.
- **nvrx `__version__` AttributeError** — see the boxed note above;
  `pip install "nvidia-resiliency-ext>=0.6.0"` fixes it.
- The container ships torch/TE/Apex/flash-attn/cuDNN but **not** transformers/dotenv/nltk —
  you still need `pip install -r requirements_megatron.txt` for the HuggingFaceTokenizer and
  the launcher's `load_dotenv`.
- Not covered on NVIDIA from this folder: the real Llama-3.1-8B geometry, a TE-vs-no-TE
  throughput number at that scale, FP8, `sequence_parallel`, CP > 1, more than 2 GPUs, and
  Megatron-Bridge checkpoint conversion.

### Checkpoint conversion (both vendors)

Convert the Hugging Face checkpoint to Megatron format. Upstream now routes all
HF <-> Megatron conversion through Megatron-Bridge:

```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Llama-3.1-8B \
  --megatron-path ./checkpoints/llama3_1_8b \
  --torch-dtype bfloat16 \
  --device-map auto
```

That directory is what `[checkpoint].load` in the TOML points at. Going back the other way
(Megatron -> HF, for serving or for the HF-based trainers in this repo) uses the same script
with `export` instead of `import`.

Sanity check:

```bash
python -c "import torch, megatron.core; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

## 2. Environment & secrets

Put a `dev.env` **in this folder**:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Both `train_llm_megatron.py` and `preprocess_megatron_data.py` load it with
`load_dotenv("dev.env")` and read `HF_TOKEN` from the environment — it is needed because
`--tokenizer-type HuggingFaceTokenizer` pulls the tokenizer from the Hub, and Llama
tokenizers are gated. `dev.env` is **git-ignored** at the repo root. **Never commit tokens** —
nothing in this folder hardcodes a secret, and nothing should.

The launcher exports two environment variables for the child process, matching upstream's
example scripts: `CUDA_DEVICE_MAX_CONNECTIONS=1` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 3. Data

### The shipped sample

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

`source_spec_id` / `source_version` are null in most (not all) rows. The preprocessor
flattens `messages` into a single `text` document per row and drops the extra columns, so
the sample works end to end without touching other folders.

### The .bin/.idx pipeline

Megatron does **not** read JSONL at training time. You must pre-tokenize into an indexed
dataset: a `<prefix>.bin` blob of token ids plus a `<prefix>.idx` offset table. This is a
one-time cost.

Start from a JSONL, one JSON object per line. A `{text: ...}` corpus is the native form;
chat JSONL is flattened automatically:

```json
{"text": "Full document body, no chat structure, no special tokens."}
```

Smoke run against the shipped sample (defaults point at it):

```bash
python preprocess_megatron_data.py --megatron-repo ../Megatron-LM --dry-run
python preprocess_megatron_data.py --megatron-repo ../Megatron-LM
```

Full run on your own corpus:

```bash
python preprocess_megatron_data.py \
  --megatron-repo ../Megatron-LM \
  --input /data/domain_corpus.jsonl \
  --output-prefix data/domain \
  --tokenizer-model meta-llama/Llama-3.1-8B \
  --workers 64 --partitions 4
```

Upstream's naming rule is `<output-prefix>_<json-key>_document.{bin,idx}`, so the default
run produces `data/otel_text_document.bin` and `.idx`. The **prefix without the suffix** is
what goes into the config — the helper prints it for you:

```toml
data_path = "./data/otel_text_document"
```

Notes:
- Relative paths in the TOML and on the preprocessor command line are resolved against
  **this folder** before the command is built, because the upstream tools run with the
  Megatron clone as their working directory.
- `--append-eod` is on by default (pass `--no-append-eod` to disable). Appending the
  end-of-document token is correct for pre-training; without it, documents bleed into each
  other across the packed sequence boundary.
- `--workers` must be divisible by `--partitions`; upstream silently drops non-divisible
  worker counts, so this helper errors instead.
- `[data].split = "990,8,2"` splits the *same* indexed dataset into train/valid/test by
  document. There is no separate validation file.
- Use the **same tokenizer** for preprocessing and for training. A mismatch fails silently.

## 4. Run

Run from inside `training/llm/megatron/`. Check the resolved command first, then launch.

```bash
# see exactly what would run, with no side effects
python train_llm_megatron.py --megatron-repo ../Megatron-LM --dry-run

# short smoke run in the foreground (10-row sample, 20 iterations)
python train_llm_megatron.py --megatron-repo ../Megatron-LM \
  --train-iters 20 --set log_interval=1 --set save_interval=20

# full continued-pretraining run
nohup python train_llm_megatron.py \
  --megatron-repo ../Megatron-LM \
  --config megatron_cpt_config.toml \
  --nproc-per-node 8 \
  > train_llm_megatron.log 2>&1 &

tail -f train_llm_megatron.log
```

Anything not in the TOML can be added without editing it: `--set key=value` (repeatable,
values parsed as TOML scalars) or `--extra <raw flags...>` appended verbatim.

## 5. Arguments

### `train_llm_megatron.py`

| Argument | Default | What it does |
|---|---|---|
| `--megatron-repo` | *(required)* | Path to your Megatron-LM clone. `pretrain_gpt.py` runs with this as CWD. |
| `--config` | `megatron_cpt_config.toml` | TOML translated into Megatron flags. |
| `--entrypoint` | `pretrain_gpt.py` | Repo-root training entrypoint. `pretrain_mamba.py` / `pretrain_hybrid.py` also exist. |
| `--nproc-per-node` | from TOML (`8`) | GPUs on this node. |
| `--data-path` | from TOML | Override the indexed-dataset prefix. |
| `--load` / `--save` | from TOML | Override the checkpoint to start from / write to. |
| `--train-iters` | from TOML | Override the iteration count. |
| `--set KEY=VALUE` | — | Override or add any config key. Repeatable. |
| `--extra ...` | — | Everything after this goes to `pretrain_gpt.py` verbatim. |
| `--dry-run` | off | Print the resolved command and exit. |

**How a config key becomes a flag.** Underscores turn into hyphens, so
`num_layers = 32` is passed as `--num-layers 32`. A boolean is a bare switch: `swiglu = true`
passes `--swiglu`, and `false` omits the flag entirely — that is what lets you switch `save`,
`load` and `tensorboard_dir` off from the command line. A list becomes the flag followed by
its items. A key may appear only once across the whole file; a second occurrence in another
table is rejected by name before anything launches, so you never get a silent last-one-wins.

**`--set` values are parsed as TOML**, which is how types stay honest: `--set num_layers=8` is
an integer and `--set log_throughput=true` a boolean, while a string needs TOML quotes *inside*
the shell quotes — `--set transformer_impl='"local"'`. A value TOML cannot parse is kept as a
plain string. Precedence runs file, then `--set`, then the dedicated flags
(`--nproc-per-node`, `--data-path`, `--load`, `--save`, `--train-iters`), so a dedicated flag
always wins over a `--set` of the same key.

### `preprocess_megatron_data.py`

| Argument | Default | What it does |
|---|---|---|
| `--megatron-repo` | *(required)* | Path to a Megatron-LM clone (or `/workspace/megatron` in the NGC container). |
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Input JSONL; a glob when `--partitions > 1`. Chat JSONL is flattened first. |
| `--output-prefix` | `data/otel` | Output path prefix, no suffix. |
| `--tokenizer-model` | `meta-llama/Llama-3.1-8B` | HF repo id or local tokenizer dir — must match training. |
| `--tokenizer-type` | `HuggingFaceTokenizer` | Megatron tokenizer type. |
| `--json-keys` | `text` | JSON field(s) holding the document text. |
| `--workers` | all cores | Worker processes; must be divisible by `--partitions`. |
| `--partitions` | `1` | Parallel input partitions, merged afterwards. |
| `--no-append-eod` | off | Do not append the end-of-document token. |
| `--dry-run` | off | Print the command and exit without running it. |

### Key config fields (`megatron_cpt_config.toml`)

| Field | Value shipped | Notes |
|---|---|---|
| `[data].data_path` | `./data/otel_text_document` | Indexed-dataset **prefix**, no `.bin`/`.idx`; matches the preprocessor defaults. |
| `[data].split` | `990,8,2` | train/valid/test document split of that one dataset. |
| `[data].tokenizer_type` | `HuggingFaceTokenizer` | Must match what preprocessing used. |
| `[training].micro_batch_size` | `1` | Per GPU, per pipeline microbatch. |
| `[training].global_batch_size` | `128` | Must be divisible by `micro_batch_size * data-parallel degree`. |
| `[training].train_iters` | `2000` | Iterations, not epochs — Megatron thinks in tokens/iterations. |
| `[training].lr` / `min_lr` | `1e-5` / `1e-6` | Continued pre-training wants a much lower LR than fresh pretraining. |
| `[training].lr_warmup_iters` | `100` | Warmup before the cosine decay. |
| `[training].bf16` | `true` | bf16 mixed precision. |
| `[parallelism].tensor_model_parallel_size` | `1` | Keep TP inside one node; raise before reaching for PP. |
| `[parallelism].pipeline_model_parallel_size` | `1` | Multi-node territory. |
| `[parallelism].context_parallel_size` | `1` | Raise for sequences at or beyond ~8K when memory-bound. |
| `[parallelism].sequence_parallel` | `false` | Only does something with TP >= 2. |
| `[parallelism].use_distributed_optimizer` | `true` | Shards optimizer state across DP ranks (ZeRO-1 equivalent). |
| `[parallelism].overlap_grad_reduce` / `overlap_param_gather` | `true` | Overlap DP collectives with compute. |
| `[memory].recompute_granularity` | `selective` | Selective activation recomputation. |
| `[checkpoint].load` | `./checkpoints/llama3_1_8b` | **Not** an HF directory — convert first with Megatron-Bridge. |
| `[checkpoint].save` | `./checkpoints/llama3_1_8b_domain` | Where checkpoints land. |
| `[checkpoint].no_load_optim` / `no_load_rng` | `true` | The converted checkpoint carries no optimizer or RNG state. |
| `[checkpoint].use_checkpoint_args` | `true` | Take architecture args from the checkpoint rather than trusting the TOML. |
| `[checkpoint].exit_on_missing_checkpoint` | `true` | Fail loudly instead of silently training from random init. |
| `[logging].log_interval` | `10` | Iterations between loss lines. |

**What "working" looks like:** the launcher prints the world size, `tp*pp*cp`, and the
derived data-parallel degree, then the full `torchrun` line. Megatron then prints its
resolved argument dump (hundreds of lines — read the top of it, that is where the parallel
sizes and batch sizes are confirmed), builds the datasets and reports the number of
train/valid/test samples, loads the checkpoint, and starts emitting
`iteration N/M | consumed samples | elapsed time per iteration (ms) | learning rate | global batch size | lm loss | grad norm`
lines. A healthy continued-pre-training run starts at a low `lm loss` (the model already
speaks the language) and drifts down; a loss near `ln(vocab_size)` means the checkpoint did
not actually load — which is precisely what `exit_on_missing_checkpoint` is there to prevent.

## 6. Output

Everything lands under `[checkpoint].save`:

```
./checkpoints/llama3_1_8b_domain/
  iter_0000500/           checkpoint at iteration 500
  iter_0001000/
  iter_0002000/
  latest_checkpointed_iteration.txt
  tensorboard/            TensorBoard event files ([logging].tensorboard_dir)
```

- **Resuming** is automatic: `[checkpoint].save` and `[checkpoint].load` in the shipped
  config point at *different* directories on purpose (start from the converted base, write
  somewhere new). Once the first checkpoint has been written, point `load` at the `save`
  directory to resume with optimizer state, and drop `no_load_optim` / `no_load_rng`.
- **The output is a Megatron checkpoint, not a Hugging Face one.** To serve it or hand it to
  the HF-based trainers in this repo, export it back with Megatron-Bridge
  (`convert_checkpoints.py export`).
- The launcher's own stdout/stderr go to `train_llm_megatron.log` (from the `nohup` line
  above); Megatron writes its per-iteration lines to the same stream.

## 7. Hardware support

| Platform | Status | Detail |
|---|---|---|
| NVIDIA CUDA | First-class; covered here in the NGC container at 1 and 2 GPUs (TP=2) | All four fused-kernel flags at their defaults. See ["NVIDIA H100 (NGC PyTorch container)"](#nvidia-h100-ngc-pytorch-container). |
| AMD ROCm | **Works with 4 flag changes** — MI355X (gfx950), ROCm 7.2, torch 2.11.0+rocm7.2, at 2 and 8 GPUs | Runs on upstream Megatron-LM `f481e63` with **no TE/Apex**; TP2/PP2/DP2, DP=8 and TP2/DP4 all train. See ["AMD MI355X (ROCm 7.2)"](#amd-mi355x-rocm-72) and ["8-GPU run"](#8-gpu-run-8x-mi355x-rocm-724). AMD also ships [`rocm/megatron-lm`](https://hub.docker.com/r/rocm/megatron-lm/tags) images. |
| Intel XPU / Apple | No supported path | |

## 8. Notes

- **The Python file is a launcher, not a trainer.** It validates `--megatron-repo`, the
  `.bin`/`.idx` pair at the data prefix, the checkpoint directory, and that the world size is
  divisible by `tp*pp*cp` (a mismatch otherwise produces an opaque process-group failure deep
  in initialization). Then it execs `torchrun ... pretrain_gpt.py`.
- **Every flag comes from the TOML, mechanically:** `key_name = value` -> `--key-name value`,
  `key = true` -> `--key`, `key = false` -> flag omitted, list -> space-separated values.
  Duplicate keys across tables are a hard error. Relative values of `data_path`, `load`,
  `save` and `tensorboard_dir` are resolved against this folder before launch.
- **Continued pre-training is "pretraining with a warm start and a small LR".** There is no
  separate CPT mode: point `--load` at the converted checkpoint, disable optimizer/RNG
  loading (`no_load_optim`, `no_load_rng`) because the converted checkpoint has none, set a
  low LR with a short warmup, and let `use_checkpoint_args` reconcile the architecture.
- **Parallelism ordering, for one 8x H100 node with an 8B model:** DP=8 with the distributed
  optimizer is the default and usually the fastest. Raise TP only when a single GPU cannot
  hold the layer, keep `TP*EP` inside the NVLink domain, add `sequence_parallel` whenever
  TP > 1, use CP for long sequences, and save PP for multi-node.

### Where this folder is uncertain

Known gaps in `megatron_cpt_config.toml`:

1. **`--finetune` is not in the shipped config.** It is the conventional Megatron flag for
   "start from a pretrained checkpoint at iteration 0", but its presence in the current
   upstream tree is unverified. The config achieves the same effect with
   `no_load_optim` + `no_load_rng` + `use_checkpoint_args`. If `--help` confirms it exists in
   your checkout, add it with `--set finetune=true`.
2. **`--ckpt-format`** (e.g. `torch_dist`) is likewise not shipped here; the default format
   is used. Check `--help` if you need distributed checkpointing explicitly.
3. **Model geometry is Llama 3.1 8B.** For a different base model, get the real numbers from
   the HF `config.json` and update `[model]`, or rely on `use_checkpoint_args = true` and
   trim the section.
4. **FP8 / MoE / CUDA-graph flags are not wired up.** They exist upstream
   (`--fp8-format`, `--fp8-recipe`, `--num-experts`, `--cuda-graph-impl`, ...) and can be
   added with `--set`, but none of them are exercised by this config.
