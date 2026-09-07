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
  into the matching `pretrain_gpt.py` flag, checks the obvious footguns, and execs
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

> **Tested topology:** **partially tested.** The preprocessing -> training path has now been
> run end to end on **2x and then all 8x AMD Instinct MI355X (gfx950), ROCm 7.2/7.2.4, torch
> 2.11.0+rocm7.2**, with a tiny custom GPT config — DP=2 and TP=2 on two GPUs, then
> TP2/PP2/DP2, DP=8 and TP2/DP4 on eight. See
> ["MI355X (ROCm 7.2) — tested"](#mi355x-rocm-72--tested) and the
> ["8-GPU run"](#8-gpu-run-8x-mi355x-rocm-724) subsection below. It
> **works with changes**: four flags must be flipped because TransformerEngine and Apex are
> CUDA-only — and those same four flags are all that is needed at 8 GPUs too, with per-GPU
> throughput flat from 2 to 8.
> The **NVIDIA/NGC path is still unrun**, as is the Llama-3.1-8B geometry, the Megatron-Bridge
> checkpoint conversion and checkpoint saving. The config was written against the Megatron-LM
> `main` branch README, `docs/llama_mistral.md`, the Megatron Core MoE README,
> `tools/preprocess_data.py`, and the `examples/gpt3` + `examples/mixtral` training scripts for
> the pinned versions below, and it targets the house default of a single node with 8x H100 80GB
> launched with `torchrun`. Megatron has hundreds of flags that move between releases; run
> `python pretrain_gpt.py --help` in your checkout before a long run.

### What this is not good for

Megatron-LM proper has no SFT/DPO/GRPO dataloader for GPT models: `pretrain_gpt.py` consumes
packed pre-training token streams from an indexed dataset, with no chat template and no
prompt masking. If you want post-training on this stack, the supported path is
**[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** (bidirectional
HF <-> Megatron conversion plus production recipes) or NeMo, both of which build on
**Megatron Core** — the same library, packaged for post-training. AMD's
[Primus](https://github.com/AMD-AGI/Primus) similarly builds a training stack on Megatron
Core for ROCm. This folder deliberately covers only the continued-pre-training case, which
is what the base repository actually supports out of the box.

## 1. Install

### NVIDIA (CUDA)

Use the NGC PyTorch container unless you have a strong reason not to. TransformerEngine,
Apex and cuDNN fused attention are pre-built and version-matched there:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace -v /data:/data -it \
  nvcr.io/nvidia/pytorch:<YY.MM>-py3 bash
```

Then, inside the container (or inside a venv, if you are doing it the hard way):

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
long run. See "Hardware support & evidence" below for sources.

#### MI355X (ROCm 7.2) — tested

**This path works on MI355X, with the changes below.** Upstream NVIDIA/Megatron-LM trains on MI355X with a plain
ROCm PyTorch wheel and **no TransformerEngine and no Apex**, but four flags must be flipped
(all of them TE/Apex fused kernels that default to on). No source patch was needed — every
change goes through the launcher's `--set`, so `megatron_cpt_config.toml` is untouched.

Environment validated:

| | |
|---|---|
| Hardware | 2x AMD Instinct MI355X (gfx950, 288GB), physical GPUs 4 and 5 of 8 |
| Stack | ROCm 7.2.4, Ubuntu, Python 3.12.3 |
| torch | `2.11.0+rocm7.2` (`torch.version.hip` = `7.2.26015`), `triton-rocm` 3.6.0 |
| Megatron-LM | commit `f481e6361520dcac1554891a6ae83b353eb1d91b`, `megatron-core 0.20.0+f481e63` |
| TransformerEngine / Apex / flash-attn | **not installed** (CUDA-only; do not try) |

**Route: venv + source clone, not the container.** `rocm/megatron-lm:v26.1` is the
documented AMD route and is almost certainly the better one for a production run (it ships
AMD's TE/hipBLASLt/CK stack pre-built), but it was **not pulled here**: sibling ROCm images
are ~116GB on disk and the image would have been a large, slow write to a
shared root filesystem. The pip route costs a ~3GB wheel, so it was tried first — and it
worked. Treat the container tag as documented-but-unverified from this folder's side.

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

**Preprocessing** (ran unmodified; `.bin`/`.idx` written in ~10s for the 10-row sample):

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
python preprocess_megatron_data.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --tokenizer-model Qwen/Qwen2.5-0.5B \
  --output-prefix $OUTPUT_DIR/train_llm_megatron/otel \
  --workers 8
# -> .../otel_text_document.bin (49968 B) + .idx; chat `messages` flattening worked as documented
```

The tokenizer is the one deviation from the shipped defaults: `meta-llama/Llama-3.1-8B` is
gated, and without access it returns `403 Cannot access gated repo`, so an ungated tokenizer
was substituted. Nothing else about the preprocessing path changed.

**Training** — the smoke used a tiny custom GPT (4 layers / hidden 512 for the first pass,
8 layers / hidden 1024 for the timed pass) instead of the shipped Llama-3.1-8B geometry,
with **saving off** and **no checkpoint load** (no converted Megatron checkpoint is
available). `--set key=false` makes the launcher omit the flag entirely, which is how `save`,
`load` and `tensorboard_dir` are switched off without editing the TOML:

```bash
source .env_megatron/bin/activate
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
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

**What a healthy run looks like** (DP=2, TP=1, exit 0):

```
iteration    1/  200 | elapsed time per iteration (ms): 7553.4 | throughput per GPU (TFLOP/s/GPU): 1.7   | lm loss: 1.196653E+01 | grad norm: 11.835
iteration   50/  200 | elapsed time per iteration (ms):  123.5 | throughput per GPU (TFLOP/s/GPU): 101.0 | lm loss: 1.028609E+01 | grad norm: 9.007
iteration  200/  200 | elapsed time per iteration (ms):  127.1 | throughput per GPU (TFLOP/s/GPU): 98.2  | lm loss: 7.938519E+00 | grad norm: 7.005
[Rank 0] (after 10 iterations) memory (MB) | allocated: 4806.14 | max allocated: 7116.91 | reserved: 7724.00
```

Loss starts at 11.97 — which is `ln(151936)`, exactly right for a randomly-initialised model
on this tokenizer — and falls monotonically. No NaN or skipped iterations.

**TP=2 bonus** (same command with `--set tensor_model_parallel_size=2`, 1000 iterations,
DP=1): also exit 0, `lm loss: 1.170070E+00` at iteration 1000 at 51.5 TFLOP/s/GPU. Tensor
parallelism works; the lower per-GPU TFLOP/s is the expected TP collective overhead at this
toy model size.

`rocm-smi` sampled mid-run (only the two assigned GPUs are busy):

```
Device  Temp     Power    SCLK     PwrCap   VRAM%  GPU%
4       49.0°C   548.0W   2382Mhz  1400.0W  3%     97%
5       50.0°C   547.0W   2378Mhz  1400.0W  3%     96%
```

**Quirks — the four flags you must flip on ROCm.** Each was hit as a hard failure, in this
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
  LayerNorm`, so TP > 1 runs without sequence parallelism (more activation memory). This is
  the main reason to prefer `rocm/megatron-lm` for a real run: AMD's image ships a ROCm TE
  build, which should restore the fused-norm/SP path.

Other ROCm observations:

- **Never `pip install flash-attn`.** `attention_backend = "auto"` resolves to Megatron's
  unfused torch attention with `transformer_impl = "local"` (the argument dump shows
  `use_flash_attn ... False`), which is the correct ROCm path here and was fast enough.
- `recompute_granularity = "selective"` works and only emits a cosmetic warning about
  `core_attn` recompute being unnecessary under TE.
- `use_distributed_optimizer` + `overlap_grad_reduce` + `overlap_param_gather` all work over
  RCCL — these were on for every run above.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  which the launcher exports, are accepted by the ROCm build (HIP maps both).
- **You must `source` the venv**, not just call `.env_megatron/bin/python train_llm_megatron.py`:
  the launcher shells out to bare `torchrun`, which fails with
  `FileNotFoundError: ... 'torchrun'` if the venv's `bin/` is not on `PATH`.
- Untested on ROCm from this folder: the Llama-3.1-8B geometry, Megatron-Bridge checkpoint
  conversion, checkpoint save/resume, CP > 1, and FP8. (TP=2, PP=2 and DP=8 are all tested —
  see the 8-GPU subsection below.)

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This works with no changes beyond the 2-GPU recipe.** The exact four-flag recipe above
scales from 2 to all 8 MI355X unmodified. **Nothing new was needed**: no source patch, no
extra package, no RCCL/NCCL tuning variable, no batch-size rescue, no OOM, no hang. Three
parallelism layouts were run back to back and all three exited 0 with finite, monotonically
decreasing loss and **zero NaN and zero skipped iterations**. `megatron_cpt_config.toml` is
still untouched — every change goes through the launcher's `--set`.

| Config | TP | PP | CP | DP | global batch | TFLOP/s/GPU (steady) | ms/iter | lm loss 1 -> 200 | exit |
|---|---|---|---|---|---|---|---|---|---|
| **A** `tp2_pp2_dp2` | 2 | 2 | 1 | 2 | 32 | **41.6 - 42.2** | ~148 | 11.985 -> 7.688 | 0 |
| **B** `dp8` | 1 | 1 | 1 | 8 | 64 | **92.1 - 112.8** | ~111 | 11.955 -> 7.414 | 0 |
| **C** `tp2_pp1_dp4` | 2 | 1 | 1 | 4 | 32 | **53.1 - 56.1** | ~116 | 11.993 -> 7.733 | 0 |

Same toy geometry as the 2-GPU timed pass (8 layers / hidden 1024 / 16 heads / seq 1024,
0.09 B params in the transformer block), same preprocessed dataset
(`otel_text_document.bin/.idx` reused from the 2-GPU session), 200 iterations each,
`--master-port` 29630 / 29631 / 29632.

**Why these three.** Config **A** is the headline: it is the only one that exercises tensor
**and** pipeline parallelism at once, so `2 x 2 x 1 x 2 = 8` covers every process-group type
Megatron builds (TP all-reduce, PP point-to-point, DP all-reduce/reduce-scatter) on one
box. 8 layers / PP=2 gives 4 layers per stage, and gbs 32 / (mbs 2 x DP 2) gives 8
micro-batches per stage, deep enough that the pipeline bubble stays bounded. Config **B**
is pure data parallel sized for an apples-to-apples scaling comparison — gbs 64 over DP=8
puts **exactly the same 8 samples per DP rank** as the 2-GPU baseline's gbs 16 over DP=2,
so per-GPU TFLOP/s is directly comparable. Config **C** isolates tensor-parallel cost from
pipeline cost. Larger TP (4 or 8) was not used: at hidden 1024 / 16 heads the per-GPU shard
becomes too thin for the collective to be worth measuring.

Launch (config A; B and C differ only in the two parallel sizes, `global_batch_size`
and `master_port`). Run everything under the machine-wide GPU mutex:

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

**Expected output** — config A (TP=2, PP=2, DP=2), the model-parallel one:

```
iteration    1/  200 | elapsed time per iteration (ms): 26515.0 | lm loss: 1.198454E+01 | number of nan iterations:   0
iteration  100/  200 | elapsed time per iteration (ms):  147.8 | throughput per GPU (TFLOP/s/GPU): 42.2 | lm loss: 8.826759E+00 | grad norm: 7.225
iteration  190/  200 | elapsed time per iteration (ms):  149.4 | throughput per GPU (TFLOP/s/GPU): 41.8 | lm loss: 7.692151E+00 | grad norm: 5.836
iteration  200/  200 | elapsed time per iteration (ms):  149.9 | throughput per GPU (TFLOP/s/GPU): 41.6 | lm loss: 7.688103E+00 | grad norm: 4.983
[Rank 1] (after 1 iterations) memory (MB) | allocated: 1336.86 | max allocated: 1530.72 | reserved: 1994.00
INFO - training/llm/megatron - command completed successfully
```

Config B (DP=8) — the scaling datapoint:

```
iteration    1/  200 | elapsed time per iteration (ms): 20399.3 | lm loss: 1.195488E+01 | grad norm: 9.707
iteration   50/  200 | elapsed time per iteration (ms):  112.6 | throughput per GPU (TFLOP/s/GPU): 110.8 | lm loss: 1.003584E+01
iteration  150/  200 | elapsed time per iteration (ms):  115.0 | throughput per GPU (TFLOP/s/GPU): 108.5 | lm loss: 7.686722E+00
iteration  200/  200 | elapsed time per iteration (ms):  110.6 | throughput per GPU (TFLOP/s/GPU): 112.8 | lm loss: 7.414137E+00 | grad norm: 4.324
[Rank 0] (after 10 iterations) memory (MB) | allocated: 3086.19 | max allocated: 5395.33 | reserved: 6518.00
```

Megatron's own argument dump confirms the groups were really built (not silently collapsed):
`inprocess_active_world_size 8`, and `tensor_model_parallel_size 2 / pipeline_model_parallel_size 2 /
context_parallel_size 1 / data_parallel_size 2` for config A.

**`rocm-smi` proof — all 8 GPUs busy** (sampled every 5 s during the runs; full capture
written under `$OUTPUT_DIR/train_llm_megatron/gpu8/rocm_smi_8gpu.txt`). Peak mid-run sample:

```
Device  Temp     Power    SCLK     PwrCap   VRAM%  GPU%
0       49.0°C   739.0W   2311Mhz  1400.0W  6%     100%
1       49.0°C   727.0W   2323Mhz  1400.0W  6%     100%
2       47.0°C   747.0W   2316Mhz  1400.0W  6%     97%
3       52.0°C   746.0W   2303Mhz  1400.0W  6%     99%
4       50.0°C   715.0W   2335Mhz  1400.0W  6%     99%
5       51.0°C   720.0W   2335Mhz  1400.0W  6%     96%
6       50.0°C   719.0W   2328Mhz  1400.0W  6%     98%
7       49.0°C   739.0W   2315Mhz  1400.0W  6%     96%
```

8/8 GPUs at 96-100 % utilisation, **5852 W total socket power**, 2.3 GHz SCLK. The
model-parallel run A sampled the same way shows 8/8 at 98-100 % and 3919 W. The
sample taken 5 s after run A ended reads 2-3 % util / 2474 W, so the busy samples are the
training itself and not background load. VRAM is only 6 % of 288 GB per GPU because the
model is deliberately tiny — the authoritative per-rank figures are torch's own: 6518 MB
reserved on DP=8, 1994 MB on TP2/PP2.

**Scaling vs the 2-GPU run.** Per-GPU work held constant (8 samples per DP rank both times):

| | 2 GPU (DP=2, gbs 16) | 8 GPU (DP=8, gbs 64) |
|---|---|---|
| TFLOP/s/GPU | 98.2 - 101.0 | 92.1 - 112.8 (median ~110) |
| ms/iteration | 123.5 - 127.1 | 110.6 - 135.5 |

**Per-GPU throughput does not degrade at all going 2 -> 8 — weak-scaling efficiency is
~100 % (measured 105-110 %, i.e. flat within run-to-run clock/noise variance).** Aggregate
throughput therefore goes from ~200 TFLOP/s on 2 GPUs to ~880 TFLOP/s on 8. Do not read the
>100 % as genuinely super-linear; read it as "RCCL all-reduce over 8 ranks on a single node is
effectively free at this model size, and `overlap_grad_reduce` hides what is left."

Tensor parallelism scales just as cleanly, which was the real risk here: **TP=2 costs the
same on 8 GPUs as it did on 2.** The 2-GPU TP=2/DP=1 pass measured 51.5 TFLOP/s/GPU; config
C (TP=2, DP=4) measures 53.1-56.1. Adding a 4-wide data-parallel dimension on top of the
TP collective did not degrade the TP collective at all — no hang, no corruption, no
mismatched-rank stall. The ~50 % gap between TP=2 and pure DP is the expected TP all-reduce
overhead at this toy hidden size (1024), not a ROCm problem; it is the price of splitting a
tiny GEMM eight ways.

Config A is slower again (41.6 vs 53.8 TFLOP/s/GPU for TP=2 alone) purely because of the
pipeline bubble: 8 micro-batches over 2 stages wastes roughly `(PP-1)/micro_batches` = 12.5 %
of every step, plus PP point-to-point sends. Expected, not a defect — raise
`global_batch_size` to deepen the pipeline if you care.

**What differed from the 2-GPU run:** only `--nproc-per-node 8`, the three parallel sizes,
`global_batch_size`, and the port. Explicitly *not* needed:

- No new pip package and no new pin — `requirements_megatron.txt` is unchanged.
- No RCCL/NCCL environment tuning. `NCCL_DEBUG=WARN` was exported for diagnostics only and
  emitted nothing; `NCCL_P2P_DISABLE`, `NCCL_IB_*`, `NCCL_SOCKET_IFNAME` were never touched.
- No OOM and no batch-size reduction; peak reserved VRAM was 6.5 GB of 288 GB.
- No hang or deadlock in any of the three layouts.
- The four fused-kernel flags from the 2-GPU section are still all four required — they are
  a TE/Apex-absence issue, not a GPU-count issue.

**Gotchas that still apply at 8 GPUs:**

- `sequence_parallel` must stay `false`. It is unusable without TE (`AssertionError: sequence
  parallel not supported by torch LayerNorm`), and note the *other* direction of the same
  trap: turning it on with `tensor_model_parallel_size = 1` makes Megatron abort with
  "Cannot use sequence parallelism without tensor parallelism". Config B (TP=1) would hit
  that, so SP is off everywhere here.
- Check `.env_megatron/bin/activate` for a stale `export CUDA_VISIBLE_DEVICES=...`
  appended by an earlier 2-GPU session. Always re-export both `HIP_VISIBLE_DEVICES` and
  `CUDA_VISIBLE_DEVICES` after `source`, and assert `torch.cuda.device_count() == 8` before
  launching.
- Keep `--set save=false --set save_interval=100000` for smoke runs. Nothing is written
  beyond logs: the whole 8-GPU output directory is 464 KB.
- First iteration takes 17-27 s in every layout (kernel autotune + RCCL group setup); steady
  state arrives by roughly iteration 20. Do not judge throughput before iteration 50.

Still untested at 8 GPUs: `context_parallel_size > 1`, TP > 2, FP8, the real Llama-3.1-8B
geometry, and checkpoint save/load.

### NVIDIA H100 (NGC PyTorch container) — tested

**This path works, and it is the primary, intended route — the one the MI355X section
could not take.** The MI355X run had to flip **four** TE/Apex fused-kernel flags OFF because
TransformerEngine and Apex are CUDA-only and were not installed. In the NGC PyTorch
container they are **pre-built and version-matched**, so all four flags go **back to their
Megatron defaults (ON)** — no `--set no_*` overrides, `transformer_impl` stays
`"transformer_engine"` — and Megatron trains with its fused kernels. Preprocessing, a
20-iteration smoke, and a 200-iteration timed pass **with checkpoint saving** all ran on a
**single H100**, exit 0, finite monotonically-decreasing loss, zero NaN/skipped iterations.
`megatron_cpt_config.toml` is untouched — every change is a launcher `--set`, same as MI355X.

Environment validated:

| | |
|---|---|
| Hardware | 1x NVIDIA H100 80GB HBM3 (Hopper, cc 9.0), physical GPU 6 of 8 |
| Route | **NGC PyTorch container** `nvcr.io/nvidia/pytorch:25.06-py3` (13.7 GB compressed / 41 GB on disk) |
| Host | driver 580.173.02, Docker 29.1.3, nvidia-container-toolkit 1.20.0 (CDI), Python 3.12 (in container) |
| torch | `2.8.0a0+5228986c39.nv25.06`, **CUDA 12.9** (container-shipped; do not re-install torch) |
| Megatron-LM | commit `f481e6361520dcac1554891a6ae83b353eb1d91b`, `megatron-core 0.20.0+f481e63` (same commit as the MI355X run, for apples-to-apples) |
| TransformerEngine / Apex / flash-attn | **`transformer-engine 2.4.0`, Apex (with `fused_weight_gradient_mlp_cuda` + `amp_C`), `flash-attn 2.7.4.post1`, cuDNN 9.10.2 — ALL pre-built in the container** |
| transformers / other extras | `transformers 5.15.1`, installed from `requirements_megatron.txt` (the container ships torch/TE/Apex but **not** transformers/dotenv/nltk) |

**Route: the NGC container, pinned to a single GPU.** This needs Docker with the NVIDIA
runtime (CDI). If an HTTP proxy sits in front of the host, `nvcr.io` may be 403-blocked —
**unset the proxy** to pull; `pypi.org` normally stays reachable with the proxy
on. Pin the container to exactly one physical GPU **by UUID** so it can never see
co-tenant GPUs; `nvidia-smi` inside the container should then show exactly one device.

```bash
# 1. Pull (proxy MUST be unset for nvcr.io; ~14 GB compressed). Pick a recent YY.MM-py3 tag.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull nvcr.io/nvidia/pytorch:25.06-py3

# 2. Launch pinned to ONE GPU by UUID (nvidia-smi -L to get yours), mounting the repo,
#    the HF model cache, and a scratch output dir. --ipc=host + the ulimits are the
#    NGC-recommended flags (see requirements_megatron.txt / section 1).
sudo docker run -d --name megatron_h100 \
  --gpus '"device=GPU-xxxxxxxx-...."' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/training_junk:/work \
  -v /path/to/hf_cache:/models \
  -v /path/to/scratch_out:/out \
  -e HF_HOME=/models \
  -w /work/training/llm/megatron \
  nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity
sudo docker exec megatron_h100 nvidia-smi -L   # MUST show exactly your one GPU

# 3. Inside the container: clone Megatron-LM, install it, add the pure-Python extras.
sudo docker exec megatron_h100 bash -c '
  cd /opt && git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM && git fetch --depth 1 origin f481e63 && git checkout f481e63   # optional: pin
  pip install --no-build-isolation -e .            # reuses the container torch/TE
  pip install "nvidia-resiliency-ext>=0.6.0"       # see nvrx note below
  cd /work/training/llm/megatron && pip install -r requirements_megatron.txt'
```

> **nvrx import gotcha (real, cost ~10 min).** With Megatron `f481e63`, `import
> megatron.core` in the stock `25.06` container dies with
> `AttributeError: module 'nvidia_resiliency_ext' has no attribute '__version__'`
> (in `dist_checkpointing/strategies/nvrx.py`). The container ships
> `nvidia-resiliency-ext 0.4.0`, which (a) is below Megatron's `NVRX_MIN_VERSION = "0.6.0"`
> and (b) does not expose `__version__` as a module attribute. **Fix:**
> `pip install "nvidia-resiliency-ext>=0.6.0"` (0.6.0 exposes `__version__` and clears the
> assert). This only affects the *async* dist-checkpoint path; training and normal
> checkpoint saving work fine once the import succeeds. A newer NGC tag may already ship a
> compatible nvrx.

**No torch clobber.** `pip install -r requirements_megatron.txt` (transformers 5.15.1 etc.)
did **not** replace the container's torch — re-checked: `torch 2.8.0a0+...nv25.06`, CUDA 12.9,
`torch.cuda.is_available() == True` after every install. The only pip complaint was a
cosmetic `nvidia-dali` vs `packaging` pin, harmless here.

**TE/Apex are actually engaged — the whole point of this route.** `import megatron.core`
prints **no** "Transformer Engine and Apex are not installed. Falling back to Torch
optimizers" warning (the MI355X run's signature line). Megatron's own resolved argument dump
confirms every fused kernel the MI355X section had to disable is now **on**:

```
transformer_impl ................................ transformer_engine
apply_rope_fusion ............................... True
gradient_accumulation_fusion .................... True
no_persist_layer_norm ........................... False      # i.e. persistent LN is ON
```

**Preprocessing** (identical to MI355X, tokenizer substituted — see below; `.bin`/`.idx` in
~10 s for the 9-row sample):

```bash
# HF_HUB_OFFLINE forces the cached tokenizer; the sample's chat `messages` are auto-flattened.
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 megatron_h100 \
  python preprocess_megatron_data.py \
    --megatron-repo /opt/Megatron-LM \
    --tokenizer-model Qwen/Qwen3-0.6B \
    --output-prefix /out/otel \
    --workers 8
# -> /out/otel_text_document.bin (49968 B) + .idx  (byte-identical size to the MI355X run)
```

Tokenizer deviation (same class of deviation as MI355X): `meta-llama/Llama-3.1-8B` is gated
and was not cached on the test node, and `Qwen/Qwen2.5-0.5B` (the MI355X substitute) was not
cached either. **`Qwen/Qwen3-0.6B` was fully cached**, so it was used for both preprocessing and
training. Its vocab is 151643, so a randomly-initialised model starts at `lm loss ≈
ln(151643) ≈ 11.93` — which is exactly what both runs below show. Nothing else about the
preprocessing path changed.

**Training — 20-iteration foreground smoke** (README §4), tiny custom GPT (8 layers /
hidden 1024), TE **on**, saving off. Note there are **no** `--set no_rope_fusion` /
`no_persist_layer_norm` / `no_gradient_accumulation_fusion` / `transformer_impl="local"`
overrides — those are exactly the four the MI355X section adds and the H100 route drops:

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

**What a healthy smoke looks like** (DP=1, TP=1, exit 0):

```
iteration    1/  20 | elapsed time per iteration (ms): 4613.4 | throughput per GPU (TFLOP/s/GPU): 2.7 | lm loss: 1.194369E+01 | grad norm: 13.477 | number of nan iterations: 0
iteration   10/  20 | elapsed time per iteration (ms):  404.1 | throughput per GPU (TFLOP/s/GPU): 30.9 | lm loss: 1.128362E+01 | grad norm: 6.455
iteration   20/  20 | elapsed time per iteration (ms):  393.8 | throughput per GPU (TFLOP/s/GPU): 31.7 | lm loss: 1.106648E+01 | grad norm: 4.822
INFO - training/llm/megatron - command completed successfully
```

**Training — 200-iteration timed pass, `global_batch_size = 16`, checkpoint saving ON.**
Same command as above but `--train-iters 200 --save /out/ckpt --set global_batch_size=16
--set micro_batch_size=2 --set lr_warmup_iters=10 --set lr_decay_iters=200 --set
log_interval=10 --set save_interval=100` (drop `--set save=false`). This also closes the
**checkpoint-save gap the MI355X run left open** — Megatron wrote checkpoints at iterations
100 and 200 in `torch_dist` format:

```
iteration    1/  200 | elapsed time per iteration (ms): 4546.2 | throughput per GPU (TFLOP/s/GPU): 5.5  | lm loss: 1.193240E+01 | grad norm: 11.079 | number of nan iterations: 0
iteration   50/  200 | elapsed time per iteration (ms):  372.0 | throughput per GPU (TFLOP/s/GPU): 67.1 | lm loss: 1.028933E+01 | grad norm: 7.135
iteration  200/  200 | elapsed time per iteration (ms):  390.6 | throughput per GPU (TFLOP/s/GPU): 63.9 | lm loss: 7.945034E+00 | grad norm: 6.831
[Rank 0] (after 10 iterations) memory (MB) | allocated: 7294.80 | max allocated: 9594.57 | reserved: 9636.00
successfully saved checkpoint from iteration 200 to /out/ckpt [ t 1/1, gtp_remat 1/1, p 1/1 ]
INFO - training/llm/megatron - command completed successfully
```

Loss starts at 11.93 (`ln(151643)`, correct for random init on this tokenizer) and falls
monotonically to 7.95 over 200 iterations; zero NaN, zero skipped. The saved checkpoint is
~11 GB (`iter_0000100/`, `iter_0000200/`, `latest_checkpointed_iteration.txt` = `200`);
write it to scratch, not into the repo.

**GPU residency proof.** `nvidia-smi` sampled mid-run — inside the container only the single
pinned GPU exists (it enumerates as index 0 but is physical GPU 6), and the host view
confirms the co-tenant GPUs 0–3 are never touched:

```
# inside container (query-compute-apps): the training PID owns memory on the one GPU
pid <pid> | used_gpu_memory 10442 MiB
0, GPU-<uuid>, NVIDIA H100 80GB HBM3, 31 %, 8878 MiB, 181.47 W

# host nvidia-smi at the same moment: GPU 6 is the training run and busy; 0-3 are the co-tenant job
6,  33 %, 12526 MiB      <- this run
0, 100 %, 67124 MiB      <- co-tenant production job, untouched
1, 100 %, 67070 MiB
2, 100 %, 66048 MiB
3, 100 %, 67630 MiB
```

**Throughput vs the MI355X "no TE/Apex" run — read this carefully.** A clean *per-GPU* delta
between the two boxes cannot be claimed honestly from these runs, because the geometries
differ: the MI355X figures are all **multi-GPU** (its DP=1 datapoint was TP=2, and its
98–101 TFLOP/s/GPU headline is DP=2 with 8 samples/GPU), whereas the H100 numbers here are a
**single GPU** with 16 samples/GPU at `global_batch_size=16`. On this **deliberately tiny toy
model** (0.09 B in the transformer block, hidden 1024), the H100 single-GPU pass steadies at
**~64–69 TFLOP/s/GPU** — a model this small is launch-/overhead-bound, not FLOP-bound, so it
does **not** exercise what TE fusion buys you (fused attention, fused RoPE, persistent
LayerNorm, fused wgrad all pay off at 8B scale / long sequences, not at hidden 1024). The
load-bearing finding here is **qualitative and decisive**: on H100 the four fused kernels are
**present and engaged** (argument dump above), the run needs **zero** ROCm workaround flags,
and it trains and checkpoints clean. A meaningful TE-vs-no-TE throughput number needs the real
Llama-3.1-8B geometry (see below), which is deferred with the multi-GPU work.

**What a wider multi-GPU pass would need (not run here — GPUs 0–3 were a co-tenant production job).**
Nothing new on the software side: launch the same container with `--gpus '"device=…,…"'`
listing your free GPUs (or all 8 once they free up), pass `--nproc-per-node N`, and set the
parallel sizes with `--set tensor_model_parallel_size` / `pipeline_model_parallel_size` /
`context_parallel_size` exactly as the MI355X 8-GPU section does. The **big win over MI355X
at multi-GPU is `sequence_parallel`**: MI355X had to keep it `false` (torch LayerNorm can't
do SP without TE), but with TE present on H100 you can set `sequence_parallel=true` whenever
`tensor_model_parallel_size > 1` and reclaim the activation memory. FP8 (`--fp8-format`) also
becomes available on Hopper via TE and is worth a datapoint. FP8, `sequence_parallel`, CP > 1,
and the real Llama-3.1-8B geometry remain untested from this folder, but the **2-GPU
tensor-parallel pass below is measured**.

### 2-GPU run (2× H100, TP=2) — tested

**This works — real 2-rank tensor parallelism, TransformerEngine on, no source patch.**
This adds the parallelism the single-GPU section could not: `tensor_model_parallel_size=2`
splits every attention/MLP weight matrix across the two ranks (each rank holds half of each
parallelized layer — distinct from data- or optimizer-sharding, which replicate the layer and
split the batch/optimizer state). A 150-iteration pass with **TE fused kernels still on** ran
to exit 0 on **two H100s**, finite monotonically-decreasing loss, zero NaN/skipped iterations,
and saved a **TP-sharded `torch_dist` checkpoint (two shards, one per rank)**. Same container,
same `megatron_cpt_config.toml`, same toy geometry as the single-GPU timed pass — only the
launcher flags changed (`--nproc-per-node 2` + the three parallel sizes).

| | |
|---|---|
| Hardware | 2× NVIDIA H100 80GB HBM3 (Hopper cc 9.0), physical GPUs **6 and 7** of 8 (GPUs 0–3 were a co-tenant production job, never touched; see safety note) |
| Route | identical NGC container `nvcr.io/nvidia/pytorch:25.06-py3`, torch `2.8.0a0+…nv25.06` / CUDA 12.9, `megatron-core 0.20.0+f481e63`, `transformer-engine 2.4.0`, `nvidia-resiliency-ext 0.6.0` (nvrx fix), `transformers 5.15.1` |
| Parallelism | **TP=2, PP=1, CP=1, DP=1** (world size 2). `sequence_parallel` left `false` (optional; it is the documented H100 win to enable at TP>1) |

**Container — pinned to exactly GPUs 6 and 7.** On a shared node, launch the container
with `--gpus '"device=6,7"'` so it sees only those two devices (`nvidia-smi -L` inside then
enumerates GPU 0 = physical 6, GPU 1 = physical 7).
Install is identical to the single-GPU route (clone Megatron `f481e63`, `pip install
"nvidia-resiliency-ext>=0.6.0"`, `pip install -r requirements_megatron.txt`); Megatron-core
was made importable via `PYTHONPATH=/opt/Megatron-LM` (the pure-Python package needs no build
step — the fused kernels come from the container's TE/Apex).

```bash
sudo docker run -d --name megatron_h100_2gpu \
  --gpus '"device=6,7"' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/training_junk:/work -v /path/to/hf_cache:/models -v /path/to/scratch_out:/out \
  -e HF_HOME=/models -w /work/training/llm/megatron \
  nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity
sudo docker exec megatron_h100_2gpu nvidia-smi -L   # MUST show exactly 2 GPUs

# preprocess once (same as single-GPU; -> /out/otel_text_document.bin 49968 B + .idx)
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

(No "Transformer Engine and Apex are not installed. Falling back to Torch optimizers" line —
grep count 0. Megatron reports the parameter count **per tensor-parallel rank**: each of the
two ranks owns its shard of the split matrices.)

**Loss curve** (both ranks log the same synchronized loss; DP=1, TP=2, `global_batch_size=16`):

```
iteration    1/  150 | lm loss: 1.197467E+01 | grad norm: 10.996 | number of skipped iterations: 0 | number of nan iterations: 0
iteration   20/  150 | throughput per GPU (TFLOP/s/GPU): 33.9 | lm loss: 1.118220E+01 | grad norm: 3.402
iteration  100/  150 | throughput per GPU (TFLOP/s/GPU): 33.5 | lm loss: 9.202365E+00 | grad norm: 7.814
iteration  150/  150 | throughput per GPU (TFLOP/s/GPU): 33.6 | lm loss: 8.801434E+00 | grad norm: 6.472
```

Loss starts at 11.97 (`≈ ln(151643)`, correct random init for the Qwen3 vocab, same as the
single-GPU run) and falls monotonically to 8.80 over 150 iterations; zero NaN, zero skipped.
Steady ~33–34 TFLOP/s/GPU — this toy model (0.09 B transformer block, hidden 1024) is
launch-/comm-bound at TP=2, so throughput is not the headline; the point is that TP splits the
matrices and stays numerically clean.

**TP-sharded checkpoint saved** (`torch_dist`, at iterations 100 and 150). The save banner's
`t 1/2` and the **two shard files** are the on-disk proof that the weights were split across
the 2 TP ranks:

```
saving checkpoint at iteration 100 to /out/ckpt in torch_dist format
successfully saved checkpoint from iteration 150 to /out/ckpt [ t 1/2, gtp_remat 1/1, p 1/1 ]
/out/ckpt/iter_0000150/__0_0.distcp   /out/ckpt/iter_0000150/__1_0.distcp   (latest_checkpointed_iteration.txt = 150)
```

The checkpoint is ~11 GB — keep it in `/out` scratch, not in the repo, and delete it when
you are done.

**GPU residency proof — BOTH GPUs 6 and 7 busy, one training PID each.** `nvidia-smi
--query-compute-apps` mid-run shows a distinct rank PID owning memory on each of the two GPUs
(this is what TP=2 requires — two live ranks, not one):

```
# inside container (query-compute-apps): one rank PID per GPU
GPU-<uuid-a>  pid <pid0>  used_gpu_memory 8786 MiB   # physical GPU 6, rank 0
GPU-<uuid-b>  pid <pid1>  used_gpu_memory 8530 MiB   # physical GPU 7, rank 1

# host view: GPUs 6 and 7 are the training run and both busy …
6, 27 %,  8802 MiB
7, 70 %,  8546 MiB
# … and the co-tenant GPUs 0-3 show ONLY their production PIDs, untouched:
0, 100 %, 65984 MiB   1, 100 %, 66948 MiB   2, 100 %, 67148 MiB   3, 100 %, 63888 MiB
```

**Shared-node safety.** GPUs 0–3 were sampled from the host repeatedly during the run and never
showed any PID other than the co-tenant production job's; `--gpus
'"device=6,7"'` guarantees the container cannot address them.

**8-GPU (TP2/PP2/DP2) is projected, NOT measured here.** The MI355X 8-GPU section ran that
geometry; on the H100 node only GPUs 6+7 were free.
The software path is unchanged — same container, `--nproc-per-node 8`, `--set
tensor_model_parallel_size=2 pipeline_model_parallel_size=2 context_parallel_size=1` — and TP=2
is demonstrated real; the 8-GPU scaling number on H100 remains unmeasured.

**H100 quirks / what changed vs MI355X:**

- **The four fused-kernel flags are the whole story, reversed.** MI355X adds
  `transformer_impl="local"`, `no_rope_fusion=true`, `no_persist_layer_norm=true`,
  `no_gradient_accumulation_fusion=true`. H100 adds **none** of them — the TOML default
  `transformer_impl="transformer_engine"` and all three fusions-on defaults just work.
- **Pin the container by GPU UUID, not index**, on a shared node. `--gpus
  '"device=GPU-<uuid>"'` makes the container see exactly that one GPU (it re-enumerates as
  index 0 inside), so there is no way to accidentally touch a co-tenant GPU. Verify with
  `nvidia-smi -L` inside the container before training.
- **`unset` the proxy to pull from `nvcr.io`** (403 otherwise). Once pulled, the container
  runs offline; `HF_HUB_OFFLINE=1` + a cached tokenizer means training needs no network.
- **nvrx `__version__` AttributeError** — see the boxed note above;
  `pip install "nvidia-resiliency-ext>=0.6.0"` fixes it.
- The container ships torch/TE/Apex/flash-attn/cuDNN but **not** transformers/dotenv/nltk —
  you still need `pip install -r requirements_megatron.txt` for the HuggingFaceTokenizer and
  the launcher's `load_dotenv`.
- Still untested on H100 from this folder: the real Llama-3.1-8B geometry, a TE-vs-no-TE
  throughput number at that scale, FP8, `sequence_parallel`, CP > 1, multi-GPU, and
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
one-time cost and it is the step people forget.

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
- Use the **same tokenizer** for preprocessing and for training. A mismatch produces a
  perfectly quiet garbage run.

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

## 7. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** none claimed beyond NVIDIA (NGC-first) and AMD (rocm/megatron-lm builds).


| Platform | Status | Evidence |
|---|---|---|
| NVIDIA CUDA | First-class upstream | [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM) — NGC container recommended; TransformerEngine/Apex are NVIDIA components. |
| AMD ROCm | **Tested — 2x and 8x MI355X (gfx950), ROCm 7.2/7.2.4, torch 2.11.0+rocm7.2; works with 4 flag changes** (see ["MI355X (ROCm 7.2) — tested"](#mi355x-rocm-72--tested) and ["8-GPU run"](#8-gpu-run-8x-mi355x-rocm-724)) | Run in this repo on upstream Megatron-LM `f481e63` with **no TE/Apex**. 8-GPU: TP2/PP2/DP2, DP=8 and TP2/DP4 all exit 0, per-GPU TFLOP/s flat from 2 to 8. Vendor sources: [`rocm/megatron-lm` Docker Hub tags](https://hub.docker.com/r/rocm/megatron-lm/tags) (`v26.1`, `v25.11`, `v25.10`, `v25.9_gfx942`/`gfx950`, `latest` — verified via the Docker Hub API); [ROCm AI-Ecosystem docs, "Training with Primus and Megatron"](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/training/recipes/primus-megatron.html); [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus) README lists Megatron-LM as a supported backend with `rocm/primus` images. |
| Intel XPU / Apple | No supported path | Not mentioned by upstream or vendor docs. |

## 8. Notes

- **The Python file is a launcher, not a trainer.** Megatron's real entrypoint is
  `pretrain_gpt.py` under `torchrun`. `train_llm_megatron.py` validates that
  `--megatron-repo` is a real Megatron checkout, that the `.bin`/`.idx` pair actually exists
  at the data prefix, that the checkpoint directory exists, and that the world size is
  divisible by `tp*pp*cp` (a mismatch there produces an opaque process-group failure deep in
  initialization). Then it execs.
- **Every flag comes from the TOML, mechanically.** The translation is
  `key_name = value` -> `--key-name value`, `key = true` -> `--key`, `key = false` -> flag
  omitted, list -> space-separated values. There is no hardcoded flag table in the Python,
  so nothing can silently drift from what you wrote; conversely, a typo in the TOML surfaces
  as an argparse error from Megatron rather than being swallowed. Duplicate keys across
  tables are a hard error. Relative values of `data_path`, `load`, `save`, and
  `tensorboard_dir` are resolved against this folder before launch.
- **Continued pre-training is "pretraining with a warm start and a small LR".** There is no
  separate CPT mode: you point `--load` at the converted checkpoint, disable optimizer/RNG
  loading (`no_load_optim`, `no_load_rng`) because the converted checkpoint has none, set a
  low LR with a short warmup, and let `use_checkpoint_args` reconcile the architecture.
- **Megatron Core is the reusable part.** `megatron/core/` is a library — the transformer
  blocks, the parallelism, the distributed optimizer, the dist-checkpointing. Megatron-LM is
  the reference training harness around it. NeMo, Megatron-Bridge and AMD's Primus all build
  on Megatron Core rather than on `pretrain_gpt.py`, which is why post-training features
  land there and not here.
- **Parallelism ordering, for one 8x H100 node with an 8B model:** DP=8 with the distributed
  optimizer is the default and usually the fastest. Raise TP only when a single GPU cannot
  hold the layer, keep `TP*EP` inside the NVLink domain, add `sequence_parallel` whenever
  TP > 1, use CP for long sequences, and save PP for multi-node.

### Where this folder is uncertain

Every flag in `megatron_cpt_config.toml` was read from upstream source or docs:
`examples/gpt3/train_gpt3_175b_distributed.sh` and `examples/mixtral/train_mixtral_8x7b_distributed.sh`
(model/training/parallelism/logging flags), `docs/llama_mistral.md` (the Llama-3.x launch
argument set and the Megatron-Bridge conversion commands),
`megatron/core/transformer/moe/README.md` (`use_distributed_optimizer`, `overlap_grad_reduce`,
`overlap_param_gather`, `tp_comm_overlap`, `recompute_granularity selective`,
`context_parallel_size`, `manual_gc`), and `tools/preprocess_data.py` (every preprocessing
flag). Known gaps:

1. **`--finetune` is not in the shipped config.** It is the conventional Megatron flag for
   "start from a pretrained checkpoint at iteration 0", but it was not verified in the
   current upstream tree during this write-up. The config achieves the same effect with
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
