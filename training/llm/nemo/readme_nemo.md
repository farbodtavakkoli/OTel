# `training/llm/nemo` — NVIDIA NeMo AutoModel (SFT + LoRA)

## 1. Overview & when to use

Post-training (**SFT** and **PEFT/LoRA**) of an existing Hugging Face checkpoint with
NVIDIA's NeMo Framework, via [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
— the LLM post-training library that the original [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
monorepo was split into (see Notes). Pick this over the other trainers in this repo when you
specifically want NVIDIA's stack: Transformer Engine kernels, FP8, MoE expert parallelism,
and a documented path onto Megatron-Core parallelism for models too large for FSDP alone.
For an ordinary 8xH100 LoRA run, `training/llm/unsloth/` or `training/llm/deepspeed/` are far
lighter to stand up — **NeMo is heavy and effectively container-bound**.

NeMo AutoModel is YAML-recipe driven and is launched by its own `automodel` CLI, which owns
the torchrun/SPMD launch. `train_llm_nemo.py` is therefore a thin wrapper: it turns argparse
knobs into a valid AutoModel recipe YAML, preflights the data, writes the recipe next to the
run, and execs `automodel <config.yaml> --nproc-per-node N`, propagating its exit code.
Reimplementing the training loop here would work against the framework's design.

Files in this folder:
- `train_llm_nemo.py` — entrypoint. Generates a NeMo AutoModel recipe YAML from CLI flags and launches the `automodel` CLI.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample shipped with this folder (the default `--train_file`).
- `requirements_nemo.txt` — dependencies, with the NGC container called out as the recommended path.
- `readme_nemo.md` — this document.

> **Hardware coverage.** The full `train_llm_nemo.py` → `automodel` → training path runs end
> to end on **AMD Instinct MI355X (gfx950, ROCm 7.2)** — full SFT and LoRA on one GPU against
> the shipped sample, producing loadable checkpoints, plus SFT across **8x MI355X**
> (`dp_size 8`, world size 8, FSDP2 + RCCL). On **NVIDIA H100 (CUDA 13)** both routes work:
> the NGC container and a bare-host pip venv, SFT and LoRA, single GPU. See "AMD MI355X
> (ROCm 7.2)", "8-GPU run (8x MI355X, ROCm 7.2.4)" and "NVIDIA H100 80GB (CUDA 13)" in
> section 2 for the commands, including the one real limitation found at 8 GPUs (the
> `automodel` CLI cannot change torchrun's rendezvous port).
>
> Not covered: tensor/pipeline/context parallel (`tp_size`/`pp_size`/`cp_size` > 1), LoRA at
> 8 GPUs, FP8, and multi-node. See Notes for the API-churn caveat — NeMo's LLM API changed
> substantially and the version you install may not match.

## 2. Install

### NVIDIA (the supported path)

**Recommended — the NGC container.** NeMo pulls Transformer Engine, Megatron-FSDP, and
custom CUDA/Triton kernels that compile against the exact CUDA and torch in the image.
A from-scratch install on a bare host is the most likely thing to fail.

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.06.00

docker run --rm -it --gpus all --runtime=nvidia \
  --shm-size=64g --ulimit memlock=-1 \
  -v $PWD:/workspace -w /workspace \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/hf_home \
  -e HF_HOME=/hf_home \
  nvcr.io/nvidia/nemo-automodel:26.06.00 bash
```

The `26.06.00` tag is the container that accompanies NeMo AutoModel v0.5.0 — upstream's
README pairs "v0.5.0 / 26.06 container" explicitly. Check
[the NGC tag list](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags)
for a newer one before pinning.

**Native install.** Upstream uses `uv`, not plain pip:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel
uv venv
uv sync --frozen                 # LLM recipes
# uv sync --frozen --extra cuda  # + Transformer Engine, Mamba SSM (slow build)
```

Or approximate it with pip (`pip install -r requirements_nemo.txt`). The `cuda` extra
builds `transformer-engine`, `mamba-ssm`, `causal-conv1d`, and `nv-grouped-gemm`, all of
which upstream marks **no-build-isolation** — under pip they need
`--no-build-isolation` and a matching toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
python -c "import nemo_automodel; print('NeMo AutoModel ready')"
automodel --help
```

### NVIDIA H100 80GB (CUDA 13)

> **Both install routes work** on H100 (Hopper cc 9.0, CUDA 13, Python 3.12): the **NGC
> container** — the native, supported path that is *unavailable* on AMD — and the bare-host
> pip route. Full **SFT** and **LoRA** paths of `train_llm_nemo.py` train against the shipped
> sample on a single GPU, writing consolidated HF safetensors and a reloadable LoRA adapter.

**Route A — NGC container (recommended).** Behind a proxy the `nvcr.io` pull is 403'd; unset
it first (pypi.org stays reachable either way):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker pull nvcr.io/nvidia/nemo-automodel:26.06.00        # ~20GB

# NOTE: recent Docker may register only the `runc` runtime, so `--runtime=nvidia`
# errors ("unknown or invalid runtime name: nvidia"). The `--gpus` flag alone is the
# working GPU-passthrough mechanism (nvidia-container-toolkit hook). Pin the GPU:
docker run --rm --gpus '"device=5"' --ipc host \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 \
  -v $HF_HOME:/models -v <repo>:/work \
  -w /work/training/llm/nemo \
  nvcr.io/nvidia/nemo-automodel:26.06.00 bash
```

Inside the container — exactly one GPU, and it is the one pinned above:

```
$ nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
0, GPU-<uuid>, NVIDIA H100 80GB HBM3   # = the pinned physical GPU
$ nvidia-smi -L | wc -l
1
$ python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
2.12.0a0+0291f960b6.nv26.04.48445190 13.2 True 1
$ python -c "import nemo_automodel,transformers;print(nemo_automodel.__version__, transformers.__version__)"
0.5.0+d02f49cb 5.8.1
$ python -c "import transformer_engine as te;print('transformer_engine', te.__version__)"
transformer_engine 2.14.1                          # <-- present; this is what fails on ROCm
```

The container may report `CUDA Forward Compatibility mode ENABLED` — its userspace CUDA 13.2
runs forward-compat over an older host driver, which is fine. In-container SFT trains and
writes consolidated HF safetensors to the mounted output dir.

**Route B — bare-host pip (fallback; mirrors the AMD recipe but with CUDA wheels).**
Unlike AMD's `--index-url .../rocm7.2`, the plain PyPI wheel is CUDA-native here:

```bash
cd training/llm/nemo
ln -sf ../../../dev.env dev.env               # HF_TOKEN; loaded by train_llm_nemo.py
python3 -m venv .env_nemo && source .env_nemo/bin/activate
pip install torch numpy                       # -> the current CUDA 13 build (CUDA 13.0)
pip install nemo-automodel==0.5.0 transformers==5.8.1 \
            datasets torchdata megatron-fsdp==0.5.0 pyyaml python-dotenv
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
#  -> 2.13.0+cu130 True        # torch NOT clobbered by the nemo-automodel install
automodel --help                              # exit 0
```

Two operational notes:

- **The HF metadata check hits the network even for a cached model.** Behind a proxy this
  surfaces as `httpx.ProxyError: 403 Forbidden` while `Qwen/Qwen3-0.6B` is being *resolved*
  — HF is listing the repo tree online. Fix: run with the proxy unset **and** `export
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (the weights are cached, so no download is
  needed). Same trap as the NGC pull.
- **SDPA uses flash-attention kernels on H100 out of the box.** The default
  `attn_implementation: sdpa` logs `Patched model with SDPA method=[CUDNN_ATTENTION,
  FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH]` — i.e. you already get FlashAttention via
  PyTorch SDPA without building `flash-attn`. (`flash_attention_2` remains selectable if
  you install the wheel, but is unnecessary.)

**Full SFT, pip route.** 9 train / 1 val split, seq 512, gbs 2 / lbs 1 → grad_accum 2,
capped at 12 optimizer steps over 4 epochs:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=5 MASTER_PORT=29661 MASTER_ADDR=127.0.0.1
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
  --checkpoint_dir ./checkpoints_sft --config_out generated_recipe.yaml \
  --nproc_per_node 1 --dp_size 1 --seq_length 512 \
  --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 4 --max_steps 12 --warmup_steps 2 \
  --val_every_steps 6 --ckpt_every_steps 12
```
```
- torch: 2.13.0+cu130 CUDA 13.0                          # Backend: nccl, World size 1
Trainable parameters: 596,049,920 | Trainable parameters percentage: 100.00%
step 0  | epoch 0 | loss 1.8393 | grad_norm 31.3682 | mem 5.63 GiB | num_label_tokens 265
step 10 | epoch 2 | loss 1.7023 | grad_norm 27.5500 | mem 6.54 GiB | num_label_tokens 265
Successfully exported consolidated HF safetensors to .../epoch_2_step_11/model/consolidated
Training: 100%|##########| 12/12
```

The consolidated checkpoint reloads as a plain HF model, so the section 7 contract holds:
`AutoModelForCausalLM.from_pretrained(".../epoch_2_step_11/model/consolidated")` returns a
596,049,920-parameter model.

**LoRA, pip route (`use_triton: true`).** `--seq_length 2048` keeps the long OTel assistant
turns instead of truncating them, and with 8 epochs / 24 steps gives a cleanly decreasing
curve. It is a longer variant than the shipped `generated_recipe_lora.yaml` (seq 512 /
4 epochs / 12 steps), so give it its own `--config_out` rather than overwriting that file:

```bash
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B --use_lora \
  --checkpoint_dir ./checkpoints_lora --config_out generated_recipe_lora_h100.yaml \
  --lora_dim 16 --lora_alpha 32 --nproc_per_node 1 --dp_size 1 \
  --seq_length 2048 --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 8 --max_steps 24 --warmup_steps 3 \
  --val_every_steps 12 --ckpt_every_steps 24
```
```
Trainable parameters: 10,092,544 | Trainable parameters percentage: 1.67%
step 5  | epoch 1 | loss 1.0672 | grad_norm 3.68 | mem 4.14 GiB | num_label_tokens 645
step 10 | epoch 2 | loss 0.8804 | grad_norm 3.62 | mem 5.16 GiB | num_label_tokens 423
step 16 | epoch 3 | loss 0.5780 | grad_norm 5.13 | mem 7.92 GiB | num_label_tokens 158
step 18 | epoch 3 | loss 0.3516 | grad_norm 3.46 | mem 3.98 GiB | num_label_tokens 424
[val] step 9  | loss 0.9033   ->   [val] step 11 | loss 0.8679   ->   [val] step 14 | loss 0.8330
Saving checkpoint to .../checkpoints_lora/epoch_4_step_23
#  -> model/adapter_model.safetensors (392 tensors), adapter_config.json, automodel_peft_config.json
```

The held-out validation loss should fall alongside the train loss. The adapter reloads with
`safe_open(...)`, e.g. `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`.

**Quirks on NVIDIA:**

- **`--runtime=nvidia` may not be registered** (only `runc`); use `--gpus '"device=N"'`
  alone, as the container command above does.
- **Unset the proxy and set `HF_HUB_OFFLINE=1`** even for a cached model (403 otherwise),
  for both the NGC pull and the pip-route run. See the note above.
- **`--warmup_steps` must stay `< --max_steps`** — the same platform-neutral scheduler assert
  the AMD section documents.
- **Data/seq-len artifact, not a bug:** at seq 512 several micro-batches truncate the
  assistant span entirely and log `num_label_tokens 0 | loss 0.0000`. Raising to seq 2048
  largely removes it. Not NVIDIA-specific.

**Multi-GPU.** A 2- or 8-GPU pass reuses the exact recipe with `--nproc_per_node N
--dp_size N`; on NVIDIA the natural launch is `automodel <cfg> --nproc-per-node N` (or the
`torchrun --master_port <free>` escape hatch the AMD 8-GPU section documents, since
`automodel` cannot change torchrun's default 29500). NCCL (not RCCL) collectives apply.
Not covered here.

| Component | On H100 / CUDA 13 |
|---|---|
| NGC container `nemo-automodel:26.06.00` | **works** (native path; `--gpus`, not `--runtime=nvidia`) |
| `transformer-engine` 2.14.1 (in container) | **present** (the exact TE that fails to build on ROCm) |
| Bare-host pip `torch 2.13.0+cu130` | works; not clobbered by the nemo install |
| `nemo-automodel` 0.5.0 + `automodel` CLI | works (both routes) |
| SFT (full FT) → consolidated HF safetensors | works; reloads with `from_pretrained` |
| PEFT/LoRA (`use_triton: true`) | works; decreasing val loss, adapter reloads |
| `attn_implementation: sdpa` | works — uses FLASH_ATTENTION/CUDNN backends natively |
| Multi-GPU (`dp_size` > 1), TP/PP/CP | not covered |

### AMD / ROCm

Upstream is NVIDIA-first — there is no ROCm image and no ROCm install documentation. **But
the SFT/LoRA path this folder generates does run on AMD Instinct**, via the pip route below.

```bash
python3.12 -m venv .env_nemo && source .env_nemo/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
# NOTE: transformers must be pinned to 5.8.1, not 5.12.1 (see requirements file)
pip install -r requirements_nemo.txt
```

Do **not** install the `[cuda]` extra on ROCm — it is Transformer Engine and friends, and
it cannot build without a CUDA toolkit. You lose FP8/MXFP8, MoE/DeepEP and the Mamba
kernels; you keep HF-checkpoint SFT, PEFT/LoRA, FSDP2 and safetensors checkpointing.

`training/llm/primus/` remains this repo's *first-party* AMD path (AMD-authored, ROCm images,
Megatron-Bridge). Prefer it for large-scale AMD work; this folder is a viable AMD option
when you specifically want AutoModel's HF-native recipe/checkpoint contract.

### AMD MI355X (ROCm 7.2)

> **Works, with one pin change.** On AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, 288 GB
> HBM), ROCm 7.2, Python 3.12, `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0`, both the
> **full SFT** and the **LoRA (`use_triton: true`)** paths of `train_llm_nemo.py` train
> against the shipped `data/OTel_LLM_sample_10.jsonl` sample and write loadable checkpoints.

**Environment.** The NGC container is unavailable here: it is CUDA-only
(`nvcr.io/nvidia/nemo-automodel:26.06.00` needs `--runtime=nvidia` and CUDA devices, neither
of which exists on a ROCm host). Everything below is the native pip route.

```bash
cd training/llm/nemo
python3 -m venv .env_nemo && source .env_nemo/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
#  -> torch-2.13.0+rocm7.2, triton-rocm-3.7.1
python -c "import torch; print(torch.version.hip, torch.cuda.get_device_name(0))"
#  -> 7.2.53211 AMD Instinct MI355X
```

**1. The folder's pip route fails to resolve — and this is not an AMD problem.**

```bash
pip install -r requirements_nemo.txt
```
```
ERROR: Cannot install -r requirements_nemo.txt (line 34) and transformers==5.12.1
because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested transformers==5.12.1
    nemo-automodel 0.5.0 depends on transformers==5.8.1

ERROR: ResolutionImpossible
```

`nemo-automodel==0.5.0` pins `transformers==5.8.1` **exactly**, so any other pin is a hard
resolver failure on **any** platform, NVIDIA included. `requirements_nemo.txt` carries 5.8.1.

**2. With that pin right, the whole stack installs on ROCm.**

```bash
pip install -r requirements_nemo.txt      # transformers==5.8.1
#  -> nemo-automodel-0.5.0 megatron-fsdp-0.5.0 transformers-5.8.1 torchao-0.18.0
#     flashoptim-0.1.4 datasets-5.0.1 ... (torch 2.13.0+rocm7.2 left untouched)
python -c "import nemo_automodel; print(nemo_automodel.__version__)"   # 0.5.0+761b6fe
automodel --help                                                       # exit 0
```

`nemo-automodel` ships a pure-Python wheel and none of its *required* dependencies is
CUDA-bound; every NVIDIA-only component lives behind the optional `[cuda]`, `[fa]` and
`[moe]` extras, which this folder does not install.

**3. Full SFT.**

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HOME=/path/to/hf_cache
python train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
  --checkpoint_dir ./checkpoints_sft --config_out generated_recipe.yaml \
  --nproc_per_node 1 --dp_size 1 --seq_length 512 \
  --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 4 --max_steps 12 --warmup_steps 2 \
  --val_every_steps 6 --ckpt_every_steps 12
```
```
> initializing torch distributed with 1 workers.
Backend: nccl
- torch: 2.13.0+rocm7.2 CUDA None
Trainable parameters: 596,049,920 | Trainable parameters percentage: 100.00%
step 0  | epoch 0 | loss 1.8380 | grad_norm 31.4733 | mem 5.72 GiB
step 10 | epoch 2 | loss 1.7038 | grad_norm 27.5190 | mem 6.62 GiB
Successfully exported consolidated HF safetensors to .../epoch_2_step_11/model/consolidated
Training: 100%|##########| 12/12
```

The consolidated checkpoint reloads as an ordinary HF model, so the section 7 output
contract holds unchanged on AMD:
`AutoModelForCausalLM.from_pretrained(".../epoch_2_step_11/model/consolidated")` returns a
596,049,920-parameter model.

**4. LoRA, including NVIDIA's Triton LoRA kernel.**

`build_recipe()` sets `peft.use_triton: true`, which is the most AMD-suspect thing the
folder emits. It works — Triton's ROCm backend JIT-compiles for `gfx950`:

```bash
python train_llm_nemo.py --model_name Qwen/Qwen3-0.6B --use_lora \
  --checkpoint_dir ./checkpoints_lora --config_out generated_recipe_lora.yaml \
  --lora_dim 16 --lora_alpha 32 --nproc_per_node 1 --dp_size 1 \
  --seq_length 512 --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 4 --max_steps 12 --warmup_steps 2 \
  --val_every_steps 6 --ckpt_every_steps 12
```
```
use_triton: true
Trainable parameters: 10,092,544 | Trainable parameters percentage: 1.67%
step 0  | loss 1.8339 | grad_norm 9.2990 | mem 2.88 GiB     # 37.7 s -- Triton JIT for gfx950
step 5  | loss 1.1134 | grad_norm 5.3182 | mem 3.00 GiB
step 11 | loss 0.0767 | grad_norm 8.4692 | mem 3.00 GiB
Saving checkpoint to .../checkpoints_lora/epoch_2_step_11
#  -> model/adapter_model.safetensors, adapter_config.json, automodel_peft_config.json
```

Note the slow first step: that is one-time Triton kernel compilation for `gfx950`, not a
hang.

**5. What does not work on AMD — the `[cuda]` extra.**

```bash
pip install --no-build-isolation "transformer-engine[pytorch]>=2.14.1"
```
```
Collecting transformer_engine_torch==2.18.0
  Downloading transformer_engine_torch-2.18.0.tar.gz (412 kB)
  Preparing metadata (pyproject.toml): finished with status 'error'
  ...
  File ".../transformer-engine-torch_.../build_tools/pytorch.py", line 60, in setup_pytorch_extension
    include_dirs = get_cuda_include_dirs()
  File ".../transformer-engine-torch_.../build_tools/utils.py", line 241, in get_cuda_include_dirs
    raise RuntimeError("CUDA not found.")
error: metadata-generation-failed
```

Transformer Engine's build backend refuses even to generate metadata without a CUDA toolkit.
The same applies to the rest of `extra == "cuda"` (`causal-conv1d`, `mamba-ssm`,
`nv-grouped-gemm`, `tilelang`, `tile-kernels`, `apache-tvm-ffi`), to `extra == "fa"`
(`flash-attn`) and to `extra == "moe"` (`deep_ep`). Do not install them on ROCm.

**Summary on AMD MI355X:**

| Component | On MI355X / ROCm 7.2 |
|---|---|
| `nemo-automodel` 0.5.0 wheel (pure Python) | works |
| `megatron-fsdp` 0.5.0 (pure Python) | works |
| `automodel` CLI + `InteractiveLauncher` | works |
| `NeMoAutoModelForCausalLM` on an HF checkpoint | works |
| `FSDP2Manager`, `dist_env.backend: nccl` | works (torch ROCm maps NCCL calls onto RCCL) |
| `ChatDataset` / `MaskedCrossEntropy` / StatefulDataLoader | works |
| safetensors + `save_consolidated` checkpointing | works, reloads with `from_pretrained` |
| PEFT/LoRA incl. `use_triton: true` | works (Triton ROCm backend, gfx950) |
| `attn_implementation: sdpa` (the default) | works |
| **Transformer Engine / `[cuda]` extra** | **fails — `RuntimeError: CUDA not found.`** |
| **FP8 / MXFP8, MoE + DeepEP, mamba-ssm, flash-attn** | **unavailable** |
| NGC container `nemo-automodel:26.06.00` | **unavailable — no NVIDIA runtime, no CUDA devices** |

**Caveats:**

- **`triton` overwrites `triton-rocm`.** `flashoptim` (a *required* nemo-automodel dep)
  requires `triton>=3.0.0; sys_platform == "linux"`, so pip installs the NVIDIA-published
  `triton` wheel on top of the `triton-rocm` that the ROCm torch wheel brought in. This is
  benign at `triton` 3.7.1, which ships **both** `amd` and `nvidia` backends and leaves
  torch fully functional — but check
  `python -c "import torch; print(torch.cuda.is_available())"` after installing, and
  reinstall `pytorch-triton-rocm` if a future version drops the AMD backend.
- **`--warmup_steps` must be < `--max_steps`** (platform-neutral bug). The defaults
  (`--warmup_steps 10`) assert in AutoModel's scheduler when you shrink `--max_steps` for a
  smoke run: `assert self.lr_warmup_steps < self.lr_decay_steps` → bare `AssertionError` in
  `nemo_automodel/components/optim/scheduler.py:100`.
- **Run `automodel` from an activated venv.** `train_llm_nemo.py` shells out to a bare
  `automodel`, so invoking `.env_nemo/bin/python train_llm_nemo.py` without
  activating gives `FileNotFoundError: [Errno 2] No such file or directory: 'automodel'`.
- **Not covered by the single-GPU route:** `tp_size` / `pp_size` / `cp_size` meshes,
  throughput, and larger models. Multi-GPU data parallel *is* covered — see the 8-GPU
  section immediately below.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

> **The full SFT path runs across 8x AMD Instinct MI355X** (`gfx950`, 288 GB HBM each,
> ROCm 7.2.4, `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0+761b6fe`, `transformers 5.8.1`,
> Python 3.12), writing a consolidated HF safetensors checkpoint that reloads. **No install
> changes are needed** — the single-GPU venv works unmodified, so `requirements_nemo.txt` is
> unchanged. FSDP2 sharding and RCCL collectives across 8 ranks work. TP/PP/CP meshes remain
> uncovered.

**One real blocker, and it is in the launch path, not in NeMo.** The folder's own
command — `train_llm_nemo.py --nproc_per_node 8` → `automodel <cfg> --nproc-per-node 8` —
goes through `InteractiveLauncher`, which calls `torch.distributed.run` with
`get_args_parser().parse_known_args()` and therefore inherits **torchrun's default
`--master_port 29500`**. `automodel` exposes no port flag, and extra flags are forwarded to
its config-override parser rather than to torchrun, so **the rendezvous port cannot be
changed through the folder's CLI**. On a shared box where port 29500 is already bound, that
path collides. The supported escape hatch is upstream's own documented invocation —
`InteractiveLauncher` detects an existing torchrun worker (`LOCAL_RANK` + torchelastic env)
and runs the recipe in-process instead of re-launching:

```bash
cd training/llm/nemo && source .env_nemo/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/path/to/hf_cache

# 1. generate the 8-GPU recipe (no GPU touched)
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
  --train_file  $OUT/train_8gpu.jsonl \
  --checkpoint_dir $OUT/checkpoints_8gpu \
  --config_out  $OUT/generated_recipe_8gpu.yaml \
  --nproc_per_node 8 --dp_size 8 --tp_size 1 --pp_size 1 --cp_size 1 \
  --seq_length 512 --global_batch_size 16 --local_batch_size 1 \
  --num_epochs 1 --max_steps 20 --warmup_steps 2 \
  --val_every_steps 10 --ckpt_every_steps 20 --dry_run

# 2. launch it with an explicit, non-colliding rendezvous port
torchrun --nproc-per-node 8 --nnodes 1 \
  --master_addr 127.0.0.1 --master_port 29770 \
  -m nemo_automodel.cli.app $OUT/generated_recipe_8gpu.yaml
```

Use `automodel <cfg> --nproc-per-node 8` directly only when port 29500 is free.

**Geometry.** `dp_size 8`, `tp_size 1`, `pp_size 1`, `cp_size 1` → **world size 8**, pure
FSDP2 data parallel, one model replica sharded across the 8 ranks.
`global_batch_size 16` / `local_batch_size 1` over 8 ranks = **2 gradient-accumulation
steps** per optimizer step; `seq_length 512`; 20 optimizer steps; `attn_implementation:
sdpa`; `dist_env.backend: nccl` (ROCm maps this onto RCCL).

**Data note.** The shipped `data/OTel_LLM_sample_10.jsonl` splits to 9 train rows — fewer
than a single global batch across 8 ranks — so duplicate it (x64 gives a 640-row JSONL,
628 train / 12 val) purely so every rank has real data. The schema is unchanged.

**Expected output:**

```
PREFLIGHT OK device_count 8 AMD Instinct MI355X
> initializing torch distributed with 8 workers
step 0  | epoch 0 | loss 1.8957 | grad_norm 30.6374 | mem 4.81 GiB | tps 150.43(18.80/gpu)
step 5  | epoch 0 | loss 1.5928 | grad_norm 24.2663 | mem 5.10 GiB | tps 2807.56(350.95/gpu)
[val] name "default" | step 9  | epoch 0 | loss 1.7634
step 19 | epoch 0 | loss 1.5724 | grad_norm 27.1872 | mem 4.58 GiB | tps 9444.95(1180.62/gpu)
[val] name "default" | step 19 | epoch 0 | loss 1.6892
Saving checkpoint to .../checkpoints_8gpu/epoch_0_step_19
Successfully exported consolidated HF safetensors to .../epoch_0_step_19/model/consolidated
Updated LOWEST_VAL checkpoint symlink to epoch_0_step_19 (val_loss=1.6892)
Training: 100%|##########| 20/20
```

The per-GPU throughput reporting (`tps N(M/gpu)`) divides by the world size, which confirms
all 8 ranks are in the group.

**Output contract holds at 8 GPUs.** The FSDP2-sharded checkpoint consolidates to plain HF
safetensors and reloads on CPU with
`AutoModelForCausalLM.from_pretrained(".../epoch_0_step_19/model/consolidated")`.

**What changes from the single-GPU setup:** nothing in the environment — only the launch
invocation (explicit `torchrun --master_port`, because of the hardcoded-29500 limitation
above), `dp_size 8` in the generated recipe, and a larger input file. Not covered on AMD:
`tp_size`/`pp_size`/`cp_size` > 1, LoRA at 8 GPUs, and anything behind the `[cuda]` extra.

## 3. Environment & secrets

Put a `dev.env` **in this folder** with your Hub token (needed for gated models such as
Llama):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_nemo.py` loads it via `load_dotenv("dev.env")`, so run the script from inside
`training/llm/nemo/`. AutoModel reads `HF_TOKEN` / `HF_HOME` from the environment. `dev.env`
is **git-ignored** at the repo root. **Never commit a token** — no token belongs in the
source or in a recipe YAML. If one was ever committed, rotate it on the Hub. Point
`HF_HOME` at a large disk if your model cache is not on `/`.

## 4. Data

This folder ships a working sample: `data/OTel_LLM_sample_10.jsonl` — 10 rows of this
repo's canonical chat JSONL, one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version` — the last of these null in most rows); AutoModel's
`ChatDataset` ignores them. To train on your own data, pass `--train_file
/path/to/your.jsonl` in the same schema — the sample is the default so a smoke run works
out of the box.

**No conversion is needed.** Earlier NeMo SFT required its own
`{"input": ..., "output": ...}` JSONL schema. The current AutoModel `ChatDataset` reads OpenAI-style
`messages` rows natively — it is the exact contract above — and renders them through the
tokenizer's own chat template to produce `input_ids` / `labels` / `attention_mask`.

Two constraints:

- **A chat template is required.** `ChatDataset` raises if the tokenizer has none. Base
  (non-instruct) checkpoints often ship without one; use the `-Instruct` variant, or pass
  an explicit Jinja template via the recipe's `dataset.chat_template` field.
- **Validation split.** The recipe wants a `dataset` and a `validation_dataset`, each with
  its own file. If you do not pass `--val_file`, the script holds out `--val_fraction`
  (default 2%) from `--train_file` and writes `*_split_train.jsonl` / `*_split_val.jsonl`
  next to it. (On the 10-row sample the 2% floor still holds out 1 row.)

If you instead have prompt/completion columns, AutoModel ships
`ColumnMappedTextInstructionDataset`, which maps arbitrary column names onto
`prompt`/`completion` — swap the dataset `_target_` in the generated YAML.

## 5. Run

**Smoke test** — writes the recipe YAML and prints the launch command without touching a
GPU; read the generated config before burning GPU time:

```bash
python3 train_llm_nemo.py --dry_run
```

**Full run:**

```bash
nohup python3 train_llm_nemo.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_file data/train.jsonl \
  --checkpoint_dir ./checkpoints \
  --use_lora --lora_dim 16 --lora_alpha 32 \
  --seq_length 2048 --global_batch_size 64 --local_batch_size 1 \
  --num_epochs 3 --max_steps 1000 \
  --nproc_per_node 8 \
  > train_llm_nemo.log 2>&1 &

tail -f train_llm_nemo.log
```

**What "working" looks like:** the script logs a preflight line confirming your JSONL
parses as chat `messages`, the train/val split sizes, the recipe path, and the exact
launch command. Control then passes to the `automodel` CLI, which logs the resolved
`Config:` and `Recipe:` lines, spawns one worker per GPU, loads the base checkpoint, and
begins emitting per-step loss. A falling loss with a finite grad norm and a checkpoint
appearing under `--checkpoint_dir` at the first `--ckpt_every_steps` boundary means the
run is healthy. If the job dies immediately after "Launching job interactively", the
failure is almost always the model load (gated repo / missing `HF_TOKEN`) or OOM — lower
`--local_batch_size` or `--seq_length` first.

## 6. Arguments

Every flag `train_llm_nemo.py` accepts:

| Arg | Default | Meaning |
|---|---|---|
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Base HF checkpoint to post-train (repo id or local path) |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat `messages` JSONL; the shipped sample by default |
| `--val_file` | `None` | Optional held-out chat JSONL; split off `--train_file` if omitted |
| `--val_fraction` | `0.02` | Held-out fraction when `--val_file` is not given |
| `--checkpoint_dir` | `./checkpoints` | Where AutoModel writes checkpoints |
| `--config_out` | `generated_recipe.yaml` | Path for the generated recipe YAML |
| `--seq_length` | `2048` | Max tokens per sequence |
| `--global_batch_size` | `64` | Global batch size across all ranks |
| `--local_batch_size` | `1` | Per-GPU micro batch size |
| `--num_epochs` | `3` | Number of epochs |
| `--max_steps` | `1000` | Hard cap on optimizer steps |
| `--learning_rate` | `None` | Peak LR; defaults to 1e-4 with `--use_lora`, else 5e-6 |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `10` | LR warmup steps |
| `--val_every_steps` | `100` | Run validation every N steps |
| `--ckpt_every_steps` | `200` | Write a checkpoint every N steps |
| `--seed` | `42` | Random seed (also seeds the train/val split) |
| `--attn_implementation` | `sdpa` | `sdpa`, `eager`, or `flash_attention_2` (FA2 needs flash-attn built) |
| `--use_lora` | off | Train a LoRA adapter instead of full fine-tuning |
| `--lora_dim` | `16` | LoRA rank (AutoModel calls this `dim`) |
| `--lora_alpha` | `32` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `*_proj` | Glob or comma-separated module names to adapt |
| `--nproc_per_node` | `8` | GPUs per node, passed to the `automodel` CLI |
| `--dp_size` | `None` | Data-parallel size (defaults to `nproc_per_node`) |
| `--tp_size` | `1` | Tensor-parallel size |
| `--pp_size` | `1` | Pipeline-parallel size |
| `--cp_size` | `1` | Context-parallel size |
| `--dry_run` | off | Write the YAML and print the command; do not launch |

## 7. Output

Checkpoints are written under `--checkpoint_dir` (default `./checkpoints`) at every
`--ckpt_every_steps` boundary. With `model_save_format: safetensors` and
`save_consolidated: true`, the consolidated result is a standard HF-format checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("checkpoints/<step-dir>")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

With `--use_lora` the saved artifact is the **adapter**, so inference needs the base model
plus the adapter — the same contract as the other LoRA trainers in this repo. The
generated recipe YAML (`--config_out`, default `generated_recipe.yaml`) is written next to
the run and is the exact, reproducible description of what was trained; keep it with the
checkpoint.

That default is a **fixed filename**, so every invocation — SFT or LoRA — rewrites the same
`generated_recipe.yaml` unless you pass `--config_out` explicitly. Pair it with a matching
`--checkpoint_dir`. The two recipes shipped in this folder are regenerated by the section 2
commands exactly as documented there: `generated_recipe.yaml` (full SFT,
`checkpoint.checkpoint_dir: ./checkpoints_sft`) and `generated_recipe_lora.yaml`
(LoRA, `./checkpoints_lora`).

## 8. Hardware support

- **NVIDIA — supported.** Both routes work on H100 80GB: the
  `nvcr.io/nvidia/nemo-automodel:26.06.00` NGC container (with `transformer-engine 2.14.1`)
  and the bare-host pip route (`torch 2.13.0+cu130`), for SFT and LoRA — commands in the
  "NVIDIA H100 80GB (CUDA 13)" subsection of section 2. Single-GPU; multi-GPU not covered.
- **AMD/ROCm — undocumented upstream, but working.** There is no ROCm container or install
  path upstream, so expect no vendor support. This folder's SFT and LoRA paths still train on
  **AMD Instinct MI355X (gfx950), ROCm 7.2, `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0`**
  — see "AMD MI355X (ROCm 7.2)" — and SFT also trains across **8x MI355X (`dp_size 8`)**, see
  the "8-GPU run" subsection. The NVIDIA-only surface (Transformer Engine, FP8/MXFP8,
  MoE/DeepEP, flash-attn, mamba-ssm) sits entirely in optional extras this folder does not
  install.
- **AMD, what to use anyway.** `training/llm/primus/` is this repo's first-party AMD path and
  `training/llm/verl/` is the AMD RL path. Pick this folder on AMD only when you want
  AutoModel's HF-native recipe and checkpoint contract.

## 9. Notes

- **Check which library you have before debugging.** This folder targets **NeMo AutoModel**;
  `automodel --help` succeeding means you are on it. On an **older** NeMo (2.x,
  `nemo_toolkit[llm]`) the `nemo.collections.llm` / `llm.import_ckpt` API applies instead and
  the recipe YAML here will **not** work.

- **HF <-> NeMo checkpoint conversion is no longer a step.** AutoModel trains the HF
  checkpoint in place and writes safetensors back out
  (`checkpoint.model_save_format: safetensors`, `save_consolidated: true`), so the output
  loads with plain `AutoModelForCausalLM.from_pretrained`. When you *do* need Megatron format
  (very large models, 6D parallelism), use **Megatron-Bridge**:
  ```bash
  ./scripts/conversion/convert.sh import \
    --executor local --device cpu \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b

  ./scripts/conversion/convert.sh export \
    --executor local --device cpu \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b/iter_0000000 \
    --hf-path ./exports/llama3_2_1b_hf
  ```
  Megatron-Bridge also exports Megatron LoRA/DoRA adapters to HF PEFT format
  (`examples/conversion/adapter/export_adapter.py`).

- **RLHF is a different repo.** NeMo AutoModel does SFT/PEFT/distillation only; GRPO, GSPO,
  DAPO, DPO and reward modelling live in
  **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)**, which can start from an AutoModel
  checkpoint directly.

- **Parallelism is configuration, not code.** `FSDP2Manager` takes a device mesh
  (`dp_size`/`tp_size`/`pp_size`/`cp_size`), so scaling is a YAML change rather than a
  rewrite. The default here is pure data parallel across 8 GPUs (`dp_size = nproc_per_node`,
  everything else 1), which is the right starting point for a LoRA or small-model SFT run.
  Raise `tp_size` only when a single model replica will not fit on one GPU.

- **LoRA naming differs from PEFT.** AutoModel's `PeftConfig` calls the rank `dim` (not
  `r`) and defaults `target_modules` to the glob `*_proj`. It is NVIDIA's own LoRA
  implementation with an optional Triton path (`use_triton: true`), not huggingface/peft.

- **Loss and masking.** The recipe uses `MaskedCrossEntropy`, and `ChatDataset` builds the
  loss mask from the chat template so only assistant turns are supervised. For multi-turn
  data every assistant turn is supervised by default; set `mask_history: true` on the
  dataset block to supervise only the final turn.

- **The recipe schema moves.** If AutoModel rejects a key in the generated YAML, diff it
  against a current recipe in
  [`examples/llm_finetune/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune)
  rather than guessing.
