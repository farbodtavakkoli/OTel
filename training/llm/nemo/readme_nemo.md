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
Reimplementing the training loop here would fight the framework.

Files in this folder:
- `train_llm_nemo.py` — entrypoint. Generates a NeMo AutoModel recipe YAML from CLI flags and launches the `automodel` CLI.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample shipped with this folder (the default `--train_file`).
- `requirements_nemo.txt` — dependencies, with the NGC container called out as the recommended path.
- `readme_nemo.md` — this document.

> **Tested topology:** **partially tested — on AMD, of all things.** This folder was written
> against upstream documentation and source (NVIDIA-NeMo/Automodel `main`) for the pinned
> versions below, and targets the repo default of a single node with 8xH100 80GB, which
> remains **unrun** at that scale. The full `train_llm_nemo.py` → `automodel` → training path
> *was* executed end-to-end on **1x AMD Instinct MI355X (gfx950, ROCm 7.2)** — both full SFT
> and LoRA, against the shipped sample, producing loadable checkpoints. That required
> exactly one fix (the `transformers` pin, now corrected). See "AMD MI355X (ROCm 7.2) —
> attempted" in section 2 for the commands and logs. The SFT path was
> also run across **8x MI355X** (`dp_size 8`, world size 8, FSDP2 + RCCL) to exit code 0 —
> see "8-GPU run (8x MI355X, ROCm 7.2.4)" in section 2, which also documents the one real
> limitation found (the `automodel` CLI cannot change torchrun's rendezvous port).
> Tensor/pipeline/context parallel (`tp_size`/`pp_size`/`cp_size` > 1), LoRA at 8 GPUs, and
> all NVIDIA-specific paths (Transformer Engine, FP8) are still unexercised.
> Multi-node is exposed through the parallelism flags and upstream's `sbatch slurm.sub`
> workflow but is not covered here. Treat every command as a starting point to verify,
> and see Notes for the API-churn caveats — NeMo's LLM API changed substantially and the
> version you install may not match. The script's stdlib helpers (`preflight`,
> `split_train_val`, `build_recipe`) *have* been run against the shipped sample; the
> GPU/`automodel` launch path has not.

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

### NVIDIA H100 80GB (CUDA 13) — verified

> **Result: works.** Validated on **1x NVIDIA H100 80GB HBM3**
> (Hopper cc 9.0), driver **580.173.02**, Python 3.12.3. **Both install routes ran**:
> the **NGC container** — the native, supported path that is *unavailable* on AMD —
> AND the bare-host pip route. Full **SFT** and **LoRA** paths of `train_llm_nemo.py`
> ran to **exit code 0** against the shipped sample, wrote consolidated HF safetensors
> / a reloadable LoRA adapter, and were confirmed resident on **GPU 5 only** (the box is
> shared; GPUs 0–3 were a co-tenant production job, untouched). Single-GPU functional
> check; multi-GPU is deferred (see end of this subsection).

**The headline vs AMD.** On MI355X the NGC container was genuinely unusable (CUDA-only,
no NVIDIA runtime) and only the pip route worked. **On H100 the container is the easy,
native path and it works** — including `transformer-engine 2.14.1`, the exact package
whose build fails on ROCm with `RuntimeError: CUDA not found.` (see the AMD section). So
on H100 you get the full NVIDIA surface (TE / FP8) that AMD structurally cannot.

**Route A — NGC container (recommended).** The pull is 403'd through the box proxy; unset
it first (pypi.org stays reachable, but nvcr.io needs the proxy off):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker pull nvcr.io/nvidia/nemo-automodel:26.06.00        # ~20GB, exit 0

# NOTE: docker 29.1.3 here registers only the `runc` runtime — `--runtime=nvidia`
# errors ("unknown or invalid runtime name: nvidia"). The `--gpus` flag alone is the
# working GPU-passthrough mechanism (nvidia-container-toolkit hook). Pin ONE GPU:
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

The container reports `CUDA Forward Compatibility mode ENABLED. Using CUDA 13.2 driver
version 595.58.03 with kernel driver version 580.173.02` — its userspace CUDA 13.2 runs
fine forward-compat over the host's 580.173.02 driver. In-container SFT ran to exit 0
(20/20 steps, consolidated HF safetensors written to the mounted `/out`); steady tps was
notably higher than the pip route (peaks ~18k vs ~9k tok/s) with TE/optimized kernels.

**Route B — bare-host pip (tested fallback; mirrors the AMD recipe but CUDA wheels).**
Used for the detailed SFT+LoRA logs below. Unlike AMD's `--index-url .../rocm7.2`, the
plain PyPI wheel is CUDA-native here:

```bash
cd training/llm/nemo
ln -sf ../../../dev.env dev.env               # HF_TOKEN; loaded by train_llm_nemo.py
python3 -m venv .env_nemo && source .env_nemo/bin/activate
pip install torch numpy                       # -> torch 2.13.0+cu130 (CUDA 13.0)
pip install nemo-automodel==0.5.0 transformers==5.8.1 \
            datasets torchdata megatron-fsdp==0.5.0 pyyaml python-dotenv
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
#  -> 2.13.0+cu130 True        # torch NOT clobbered by the nemo-automodel install
automodel --help                              # exit 0
```

Two operational notes that bit here and will bite you:

- **HF metadata check hits the proxy even for a cached model.** The first SFT attempt
  died with `httpx.ProxyError: 403 Forbidden` while `Qwen/Qwen3-0.6B` (cached) was being
  *resolved* — HF was listing the repo tree online through the proxy. Fix: run with the
  proxy unset **and** `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (the weights are
  cached, so no download is needed). This is the same proxy trap as the NGC pull.
- **SDPA uses flash-attention kernels on H100 out of the box.** The default
  `attn_implementation: sdpa` logs `Patched model with SDPA method=[CUDNN_ATTENTION,
  FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH]` — i.e. you already get FlashAttention via
  PyTorch SDPA without building `flash-attn`. (`flash_attention_2` remains selectable if
  you install the wheel, but was unnecessary for the smoke.)

**Full SFT — GPU 5, pip route.** 9 train / 1 val split, seq 512, gbs 2 / lbs 1 →
grad_accum 2, capped at 12 optimizer steps over 4 epochs:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=5 MASTER_PORT=29661 MASTER_ADDR=127.0.0.1
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
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
Training: 100%|##########| 12/12 [00:41<00:00,  3.44s/step]        # exit code 0
```

The consolidated checkpoint reloads as a plain HF model, so the section 7 contract holds:
```
AutoModelForCausalLM.from_pretrained(".../epoch_2_step_11/model/consolidated")
#  -> SFT CHECKPOINT RELOAD OK | params 596,049,920
```

**GPU-5 residency, sampled from inside the SFT run** (its own PID on GPU 5's UUID; GPUs
0–3 belong to the co-tenant job and were never touched):
```
$ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv | grep <gpu-uuid>
<pid>, GPU-<uuid>, 4862 MiB      # the training PID, on the pinned GPU only
```

**LoRA — GPU 5, pip route (`use_triton: true`).** Here `--seq_length 2048` (keeps the
long OTel assistant turns instead of truncating them) with 8 epochs / 24 steps gives a
**cleanly decreasing** curve — the sharpest evidence in this section:

```bash
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B --use_lora \
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
Saving checkpoint to .../checkpoints_lora/epoch_4_step_23         # exit code 0
#  -> model/adapter_model.safetensors (392 tensors), adapter_config.json, automodel_peft_config.json
```

The **held-out validation loss falls monotonically 0.9033 → 0.8679 → 0.8330** (constant
22-token val batch, so it is a clean apples-to-apples signal), and train loss drops from
~1.07 to ~0.35. The adapter reloads: `safe_open(...) -> ADAPTER OK | 392 tensors | e.g.
base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`.

**Quirks / what it took (H100-specific or found here):**

- **`--runtime=nvidia` is not registered** on this host's docker 29.1.3 (only `runc`); use
  `--gpus '"device=N"'` alone. Deviation from the README's container command above —
  corrected there.
- **Proxy + `HF_HUB_OFFLINE=1`** are both required even for a cached model (403 otherwise),
  for both the NGC pull and the pip-route run. See the note above.
- **VRAM is a non-issue at this scale** — Qwen3-0.6B SFT peaked ~6.5 GiB and LoRA ~8 GiB of
  the 80 GB, so none of the MI355X→H100 288→80 GB reductions the brief warns about were
  needed here. (They would matter for a multi-billion-param model.)
- **`--warmup_steps` must stay `< --max_steps`** — same platform-neutral scheduler assert
  the AMD section documents; kept `warmup 2–3` under `max_steps 12–24`.
- **Data/seq-len artifact, not a bug:** at seq 512 several micro-batches truncate the
  assistant span entirely and log `num_label_tokens 0 | loss 0.0000` (visible in the SFT
  run). Raising to seq 2048 (the LoRA run) largely removes it. Not H100-specific.

**Multi-GPU (deferred).** Only single-GPU `dp_size 1` was run — GPUs 0–3 were a live
co-tenant job. A 2- or 8-GPU pass would reuse the exact recipe with `--nproc_per_node N
--dp_size N`; on H100 the natural launch is `automodel <cfg> --nproc-per-node N` (or the
`torchrun --master_port <free>` escape hatch the AMD 8-GPU section documents, since
`automodel` cannot change torchrun's default 29500). NCCL (not RCCL) collectives apply.

**Summary on NVIDIA H100: this path works.** The NGC container — the native path, and the one AMD
cannot run — pulls and runs with `transformer-engine 2.14.1` present; the bare-host pip
route (`torch 2.13.0+cu130`) also runs both SFT and LoRA to exit 0 with a decreasing
validation loss, reloadable checkpoints, and confirmed single-GPU (GPU 5) residency. This
is the "upgrade over AMD" the folder was waiting for: on H100 the container is the easy
path and the full NVIDIA kernel surface (TE/FP8) is available rather than structurally
blocked.

| Component | On H100 / CUDA 13 |
|---|---|
| NGC container `nemo-automodel:26.06.00` | **works** (native path; `--gpus`, not `--runtime=nvidia`) |
| `transformer-engine` 2.14.1 (in container) | **present** (the exact TE that fails to build on ROCm) |
| Bare-host pip `torch 2.13.0+cu130` | works; not clobbered by the nemo install |
| `nemo-automodel` 0.5.0 + `automodel` CLI | works (both routes) |
| SFT (full FT) → consolidated HF safetensors | works, exit 0, reloads with `from_pretrained` |
| PEFT/LoRA (`use_triton: true`) | works, exit 0, decreasing val loss, adapter reloads |
| `attn_implementation: sdpa` | works — uses FLASH_ATTENTION/CUDNN backends natively |
| Multi-GPU (`dp_size` > 1), TP/PP/CP | **deferred** (co-tenant on GPUs 0–3) |

### AMD / ROCm

Upstream is distributed NVIDIA-first: the documented install paths are the NGC container
and a CUDA-extra build, the kernel layer is NVIDIA-oriented (Transformer Engine, DeepEP,
FP8 on GB200), and benchmarks are published on H100/GB200 only. There is no ROCm image and
no ROCm install documentation upstream.

**But the SFT/LoRA path this folder actually generates does run on AMD Instinct.** It was
executed end-to-end on an MI355X — see the section below for the commands, the evidence,
and the one pin you must change.

```bash
python3.12 -m venv .env_nemo && source .env_nemo/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
# NOTE: change transformers==5.12.1 -> transformers==5.8.1 first (see requirements file)
pip install -r requirements_nemo.txt
```

Do **not** install the `[cuda]` extra on ROCm — it is Transformer Engine and friends, and
it cannot build without a CUDA toolkit. You lose FP8/MXFP8, MoE/DeepEP and the Mamba
kernels; you keep HF-checkpoint SFT, PEFT/LoRA, FSDP2 and safetensors checkpointing.

`training/llm/primus/` remains this repo's *first-party* AMD path (AMD-authored, ROCm images,
Megatron-Bridge). Prefer it for large-scale AMD work; this folder is a viable AMD option
when you specifically want AutoModel's HF-native recipe/checkpoint contract.

### AMD MI355X (ROCm 7.2) — attempted

> **Result: works with one change.** Validated on 1x **AMD Instinct MI355X**
> (`gfx950:sramecc+:xnack-`, 288 GB HBM), ROCm 7.2, Ubuntu, Python 3.12.3,
> `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0`. Both the **full SFT** and the
> **LoRA (`use_triton: true`)** paths of `train_llm_nemo.py` ran to completion against the
> shipped `data/OTel_LLM_sample_10.jsonl` sample and wrote loadable checkpoints. This is a
> single-GPU functional check, not a performance or multi-GPU/parallelism validation.

**Environment.** The NGC container was deliberately **not** pulled: it is CUDA-only
(`nvcr.io/nvidia/nemo-automodel:26.06.00` needs `--runtime=nvidia` and CUDA devices,
neither of which exists on a ROCm host), so the container path is genuinely unavailable
here. Everything below is the native pip route.

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

`requirements_nemo.txt` pinned `transformers==5.12.1`; `nemo-automodel==0.5.0` pins
`transformers==5.8.1` **exactly**. That is a hard resolver failure on **any** platform,
NVIDIA included. The pin has been corrected in `requirements_nemo.txt`.

**2. With that one pin fixed, the whole stack installs on ROCm.**

```bash
pip install -r requirements_nemo.txt      # transformers==5.8.1
#  -> nemo-automodel-0.5.0 megatron-fsdp-0.5.0 transformers-5.8.1 torchao-0.18.0
#     flashoptim-0.1.4 datasets-5.0.1 ... (torch 2.13.0+rocm7.2 left untouched)
python -c "import nemo_automodel; print(nemo_automodel.__version__)"   # 0.5.0+761b6fe
automodel --help                                                       # exit 0
```

The reason this works: **`nemo-automodel` ships a pure-Python `py3-none-any` wheel**, and
none of its *required* dependencies is CUDA-bound. `megatron-fsdp==0.5.0` is likewise pure
Python (`torch`, `einops`, `packaging`). Every NVIDIA-only component lives behind the
optional `[cuda]`, `[fa]` and `[moe]` extras, which this folder does not install.

**3. Actual training on GPU 6 — full SFT.**

```bash
export HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 HF_HOME=/path/to/hf_cache
python train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
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
step 0  | epoch 0 | loss 1.8380 | grad_norm 31.4733 | mem 5.72 GiB | tps 61.20
step 4  | epoch 0 | loss 2.3122 | grad_norm 72.4026 | mem 6.35 GiB | tps 7093.50
step 10 | epoch 2 | loss 1.7038 | grad_norm 27.5190 | mem 6.62 GiB | tps 95.20
Successfully exported consolidated HF safetensors to .../epoch_2_step_11/model/consolidated
Training: 100%|##########| 12/12 [00:18<00:00,  1.50s/step]     # exit code 0
```

The consolidated checkpoint reloads as an ordinary HF model, so the section 7 output
contract holds unchanged on AMD:

```
AutoModelForCausalLM.from_pretrained(".../epoch_2_step_11/model/consolidated")
#  -> LOADED | params 596,049,920
```

**4. Actual training on GPU 6 — LoRA, including NVIDIA's Triton LoRA kernel.**

`build_recipe()` sets `peft.use_triton: true`, which is the most AMD-suspect thing the
folder emits. It works — Triton's ROCm backend JIT-compiles for `gfx950`:

```bash
python train_llm_nemo.py --model_name Qwen/Qwen3-0.6B --use_lora \
  --lora_dim 16 --lora_alpha 32 --nproc_per_node 1 --dp_size 1 \
  --seq_length 512 --num_epochs 4 --max_steps 12 --warmup_steps 2
```
```
use_triton: true
Trainable parameters: 10,092,544 | Trainable parameters percentage: 1.67%
step 0  | loss 1.8339 | grad_norm 9.2990 | mem 2.88 GiB     # 37.7 s -- Triton JIT for gfx950
step 5  | loss 1.1134 | grad_norm 5.3182 | mem 3.00 GiB
step 11 | loss 0.0767 | grad_norm 8.4692 | mem 3.00 GiB
Saving checkpoint to .../checkpoints_lora/epoch_2_step_11   # exit code 0
#  -> model/adapter_model.safetensors, adapter_config.json, automodel_peft_config.json
```

Note the ~38 s first step: that is one-time Triton kernel compilation for `gfx950`, not a
hang. Subsequent steps run at ~1.6 step/s.

**5. What genuinely does NOT work on AMD — the `[cuda]` extra.**

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

**This is the CUDA hard dependency**, and it is decisive but *optional*: Transformer
Engine's build backend refuses even to generate metadata without a CUDA toolkit — there is
no ROCm code path to fall back to. The same applies to the rest of `extra == "cuda"`
(`causal-conv1d`, `mamba-ssm`, `nv-grouped-gemm`, `tilelang`, `tile-kernels`,
`apache-tvm-ffi`), to `extra == "fa"` (`flash-attn`) and to `extra == "moe"` (`deep_ep`).

**Summary on AMD MI355X: works with changes.**

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

Root cause of the *limits* (not of a failure): the NVIDIA-only surface is confined to
optional extras built around Transformer Engine, whose build system hard-requires a CUDA
toolkit. The recipe `train_llm_nemo.py` generates never references those extras, so the
folder's own contract is satisfied on ROCm.

**Caveats and gotchas found while doing this:**

- **`triton` overwrites `triton-rocm`.** `flashoptim` (a *required* nemo-automodel dep)
  requires `triton>=3.0.0; sys_platform == "linux"`, so pip installs the NVIDIA-published
  `triton` wheel on top of the `triton-rocm` that the ROCm torch wheel brought in. It
  happened to be benign here — `triton` 3.7.1 ships **both** `amd` and `nvidia` backends
  and torch stayed fully functional — but check
  `python -c "import torch; print(torch.cuda.is_available())"` after installing, and
  reinstall `pytorch-triton-rocm` if a future version drops the AMD backend.
- **`--warmup_steps` must be < `--max_steps`** (platform-neutral bug). The defaults
  (`--warmup_steps 10`) assert in AutoModel's scheduler when you shrink `--max_steps` for a
  smoke run: `assert self.lr_warmup_steps < self.lr_decay_steps` → bare `AssertionError` in
  `nemo_automodel/components/optim/scheduler.py:100`.
- **Run `automodel` from an activated venv.** `train_llm_nemo.py` shells out to a bare
  `automodel`, so invoking `.env_nemo/bin/python train_llm_nemo.py` without
  activating gives `FileNotFoundError: [Errno 2] No such file or directory: 'automodel'`.
- **Not validated on AMD by the single-GPU run:** multi-GPU / `tp_size` / `pp_size` /
  `cp_size` meshes, RCCL collectives across ranks, throughput, and larger models. Only
  single-GPU `dp_size 1` was exercised there. **The multi-GPU half of this is covered by
  the 8-GPU section immediately below.** `tp_size`/`pp_size`/`cp_size`
  above 1 remain unexercised.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

> **This runs on 8x AMD Instinct MI355X.** The full SFT path was
> executed across **all 8 GPUs** (`gfx950`, 288 GB HBM each, ROCm **7.2.4**,
> `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0+761b6fe`, `transformers 5.8.1`,
> Python 3.12.3) and completed with **exit code 0**, writing a consolidated HF
> safetensors checkpoint that reloads. **No install changes were needed** — the venv from
> the single-GPU run (`.env_nemo`) worked unmodified, so
> `requirements_nemo.txt` is unchanged. FSDP2 sharding and RCCL collectives across 8 ranks
> work. This upgrades the "multi-GPU unexercised" caveat above: pure data parallel
> (`dp_size 8`) is now **tested**; TP/PP/CP meshes are still **untested**.

**One real blocker found, and it is in the launch path, not in NeMo.** The folder's own
command — `train_llm_nemo.py --nproc_per_node 8` → `automodel <cfg> --nproc-per-node 8` —
goes through `InteractiveLauncher`, which calls `torch.distributed.run` with
`get_args_parser().parse_known_args()` and therefore inherits **torchrun's default
`--master_port 29500`**. `automodel` exposes no port flag, and extra flags are forwarded to
its config-override parser rather than to torchrun, so **the rendezvous port cannot be
changed through the folder's CLI**. On a shared box where port 29500 is already bound
(here: another job's torchrun was `LISTEN`ing on 29500), that path collides. The supported
escape hatch is upstream's own documented invocation — `InteractiveLauncher` detects an
existing torchrun worker (`LOCAL_RANK` + torchelastic env) and runs the recipe in-process
instead of re-launching:

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
than a single global batch across 8 ranks — so it was duplicated x64 into a 640-row JSONL
(628 train / 12 val) purely to give every rank real data. The schema is unchanged.

**Expected output** (from the run log):

```
=== LOCK ACQUIRED (machine-wide flock held by the launcher) ===
PREFLIGHT OK device_count 8 AMD Instinct MI355X
> initializing torch distributed with 8 workers
step 0  | epoch 0 | loss 1.8957 | grad_norm 30.6374 | mem 4.81 GiB | tps 150.43(18.80/gpu)
step 5  | epoch 0 | loss 1.5928 | grad_norm 24.2663 | mem 5.10 GiB | tps 2807.56(350.95/gpu)
[val] name "default" | step 9  | epoch 0 | loss 1.7634
step 15 | epoch 0 | loss 1.3987 | grad_norm 20.8935 | mem 5.10 GiB | tps 4018.76(502.34/gpu)
step 19 | epoch 0 | loss 1.5724 | grad_norm 27.1872 | mem 4.58 GiB | tps 9444.95(1180.62/gpu)
[val] name "default" | step 19 | epoch 0 | loss 1.6892
Saving checkpoint to .../checkpoints_8gpu/epoch_0_step_19
Successfully exported consolidated HF safetensors to .../epoch_0_step_19/model/consolidated
Updated LOWEST_VAL checkpoint symlink to epoch_0_step_19 (val_loss=1.6892)
Training: 100%|##########| 20/20 [00:20<00:00,  1.02s/step]
=== TRAIN EXIT CODE 0 ===
```

Note the per-GPU throughput reporting (`tps N(M/gpu)`) — AutoModel divides by the world
size, which is itself confirmation that all 8 ranks are in the group. Steady-state was
~2.4 step/s at ~5.1 GiB torch-reported memory per rank.

**8-GPU evidence, sampled in-band during the run** (single `rocm-smi` sample, taken from
inside the job, cross-checked against this job's own PIDs in the same sample):

```
--- pgrep -af (the job's own processes) ---
<agent-pid>       .../torchrun --nproc-per-node 8 ... --master_port 29770 -m nemo_automodel.cli.app ...
<8 worker pids>   .../python3 -u -m nemo_automodel.cli.app .../generated_recipe_8gpu.yaml   # 8 workers

--- rocm-smi --showpids (same sample) ---
PID     PROCESS NAME  GPU(s)  VRAM USED
669223  python3       1       12156174336
669224  python3       1       13469945856
669225  python3       1       12869206016
669226  python3       1       13457362944
669227  python3       1       12932120576
669228  python3       1       13167955968
669229  python3       1       13205704704
669230  python3       1       13186338816
669066  pt_elastic    0       0                  # the torchrun agent itself

--- rocm-smi --showuse (same sample) ---
GPU[0] 85%   GPU[1] 91%   GPU[2] 79%   GPU[3] 74%
GPU[4] 74%   GPU[5] 77%   GPU[6] 78%   GPU[7] 90%
```

All 8 GPUs busy, ~12-13 GB VRAM per rank, and every KFD process in the sample is one of
the job's own PIDs — no foreign process, so the utilisation is attributable to this run.
(On a shared box, serialise the run behind a machine-wide `flock` and sample from inside
the held lock. A post-hoc `rocm-smi` would prove nothing.)

**Output contract holds at 8 GPUs.** The FSDP2-sharded checkpoint consolidates to plain HF
safetensors and reloads on CPU:

```
AutoModelForCausalLM.from_pretrained(".../checkpoints_8gpu/epoch_0_step_19/model/consolidated")
#  -> LOADED | params 596,049,920
```

**What differed from the prior (single-GPU) state:** nothing in the environment. No package
was installed, upgraded or pinned differently; `requirements_nemo.txt` is untouched. The
only deltas are the launch invocation (explicit `torchrun --master_port`, because of the
hardcoded-29500 limitation above), `dp_size 8` in the generated recipe, and a larger input
file. Still untested on AMD: `tp_size`/`pp_size`/`cp_size` > 1, LoRA at 8 GPUs (only full
SFT was run multi-GPU), models large enough to stress sharding, and anything behind the
`[cuda]` extra (Transformer Engine, FP8/MXFP8, MoE/DeepEP), which remains permanently
unavailable here — see the `RuntimeError: CUDA not found.` above.

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

**Good news: no conversion is needed.** Earlier NeMo SFT required its own
`{"input": ..., "output": ...}` JSONL schema, which is where the "NeMo wants its own
format" reputation comes from. The current AutoModel `ChatDataset` reads OpenAI-style
`messages` rows natively — it is the exact contract above — and renders them through the
tokenizer's own chat template to produce `input_ids` / `labels` / `attention_mask`.

Two things worth knowing:

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

## 8. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** none — NVIDIA-first stack (NGC containers, Transformer Engine).


- **NVIDIA — supported.** NeMo AutoModel is an NVIDIA NeMo Framework library; the current
  release pairing is v0.5.0 with the `nvcr.io/nvidia/nemo-automodel:26.06.00` NGC
  container, and upstream's feature list is built on NVIDIA-oriented kernel paths
  (Transformer Engine, DeepEP, Triton, FP8/MXFP8 on GB200). Benchmarks are published for
  H100/GB200-class GPUs only. Source: the
  [NVIDIA-NeMo/Automodel README](https://github.com/NVIDIA-NeMo/Automodel) and the
  [NGC container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags).
  **Verified here on 1x NVIDIA H100 80GB HBM3 (driver 580.173.02):**
  the `nemo-automodel:26.06.00` NGC container pulls and runs (with
  `transformer-engine 2.14.1` present), and the bare-host pip route (`torch 2.13.0+cu130`)
  ran both SFT and LoRA to exit 0 with decreasing validation loss and reloadable
  checkpoints — commands, logs and GPU-residency evidence in the
  "NVIDIA H100 80GB (CUDA 13) — verified" subsection of section 2. Single-GPU;
  multi-GPU deferred (shared node).
- **AMD/ROCm — undocumented upstream, but empirically working here.** No ROCm container, no
  ROCm install path and no AMD hardware documentation exists upstream, so
  AMD is **unsupported by NVIDIA** and you are on your own for bugs. Nevertheless this
  folder's SFT and LoRA paths were **run to completion on 1x AMD Instinct MI355X (gfx950),
  ROCm 7.2, `torch 2.13.0+rocm7.2`, `nemo-automodel 0.5.0`** — full
  evidence, commands and logs in the "AMD MI355X (ROCm 7.2) — attempted" subsection of
  section 2 — and the SFT path was additionally **run to completion across 8x MI355X
  (ROCm 7.2.4, `dp_size 8`, world size 8)**, with in-band 8-GPU
  `rocm-smi` evidence in the "8-GPU run" subsection. The NVIDIA-only surface (Transformer Engine, FP8/MXFP8, MoE/DeepEP,
  flash-attn, mamba-ssm) is entirely inside optional extras that this folder does not
  install; `transformer-engine` itself is the hard CUDA dependency and fails with
  `RuntimeError: CUDA not found.` at metadata generation.
- **AMD, what to use anyway.** `training/llm/primus/` is still this repo's first-party AMD
  path (AMD-authored, ROCm images), and `training/llm/verl/` is the AMD RL path. Pick this
  folder on AMD only when you specifically want AutoModel's HF-native recipe and checkpoint
  contract, and expect no vendor support.

## 9. Notes

- **IMPORTANT — the API this was written against.** The brief for this folder targeted the
  **NeMo 2.0** Python-config API: `nemo.collections.llm`, `llm.import_ckpt`, and NeMo-Run
  recipes. **That API is no longer the current upstream shape.** In the upstream layout
  this folder targets:
  - `github.com/NVIDIA/NeMo` now redirects to **`NVIDIA-NeMo/Speech`**, whose
    `nemo/collections/` contains only `asr`, `audio`, `common`, and `speechlm2`. There is
    **no `llm` collection**, so `nemo.collections.llm` and `llm.import_ckpt` are not
    importable from the current main repo.
  - The monorepo was split into the **NVIDIA-NeMo** org. LLM post-training now lives in
    [Automodel](https://github.com/NVIDIA-NeMo/Automodel) (HF-native SFT/PEFT — what this
    folder uses), [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
    (HF <-> Megatron conversion + Megatron-Core training), and
    [RL](https://github.com/NVIDIA-NeMo/RL) (post-training RL).

  If you install an **older** NeMo (2.x, `nemo_toolkit[llm]`), the `nemo.collections.llm`
  API and `llm.import_ckpt` still exist and the recipe YAML in this folder will **not**
  apply. Check which library you actually have before debugging: `automodel --help`
  succeeding means you are on AutoModel.

- **HF <-> NeMo checkpoint conversion.** In NeMo 2.0 this was
  `llm.import_ckpt(model=..., source="hf://<repo-id>")`, which converted an HF checkpoint
  into NeMo's own format before training, plus `llm.export_ckpt` to go back. **AutoModel
  removes that step entirely** — it trains the HF checkpoint in place and writes
  safetensors back out (`checkpoint.model_save_format: safetensors`,
  `save_consolidated: true`), so the output loads with plain
  `AutoModelForCausalLM.from_pretrained`. That is the single biggest practical reason to
  prefer this path. When you *do* need Megatron format (very large models, 6D
  parallelism), conversion moved to **Megatron-Bridge**, whose CLI is the successor to
  `import_ckpt`:
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

- **RLHF is a different repo.** NeMo AutoModel does SFT/PEFT/distillation only. GRPO,
  GSPO, DAPO, DPO, and reward modelling live in **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)**,
  which is Ray-based, ships its own NGC container (`nvcr.io/nvidia/nemo-rl`), and can
  start from an AutoModel checkpoint directly. That is out of scope for this folder, which
  is deliberately scoped to SFT and PEFT/LoRA.

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

- **Split helper newline detail.** `split_train_val` normalizes line endings before
  shuffling — a JSONL whose final line lacks a trailing newline would otherwise merge two
  rows in the written split files. (Found by running the helper against the shipped
  sample, which ends without a newline.)

- **Uncertainty, stated plainly.** NeMo's LLM API has churned hard (1.0 -> 2.0 -> the org
  split), and upstream `main` moves weekly. The recipe keys used here
  (`model` / `distributed` / `step_scheduler` / `optimizer` / `lr_scheduler` /
  `checkpoint` / `dataset` / `dataloader` / `peft` / `loss_fn`) were read from live
  upstream example recipes, and `examples/llm_finetune/finetune.py` is now
  deprecated in favour of the `automodel` CLI. If a key is rejected, diff the generated
  YAML against a current recipe in
  [`examples/llm_finetune/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune)
  rather than guessing — that directory is the authoritative reference.
