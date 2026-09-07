# Setup & usage — `train_llm_openrlhf.py`

## Overview & when to use

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) is a production-grade RLHF framework built
on **Ray + vLLM + DeepSpeed**: Ray places the actor / critic / reference / reward models and
the vLLM generation engines across GPUs, vLLM handles rollout generation (which is ~80% of
RL wall-clock), and DeepSpeed ZeRO-3 shards the training-side models. It implements **PPO,
GRPO, REINFORCE++, REINFORCE++-baseline, RLOO and Dr. GRPO**, plus **SFT**, **DPO** and
reward-model training. Pick this folder over `../deepspeed/` when you need *real*
online RL at scale — a separate critic, a served reward model, fast vLLM rollouts, and
multi-node Ray placement — rather than TRL's single-process GRPO; pick
`../rapidfire/` instead when the goal is comparing many configs cheaply.

Files in this folder:
- `train_llm_openrlhf.py` — launcher; builds and execs the right `deepspeed` / `ray job submit` command.
- `prepare_data_openrlhf.py` — stdlib-only converter: chat JSONL → flat prompt/answer JSONL for the PPO/GRPO path.
- `requirements_openrlhf.txt` — dependencies, including the source-only and `--no-build-isolation` ones.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.
- `readme_openrlhf.md` — this document.
- `dev.env` — **you create this**; holds `HF_TOKEN`. Git-ignored, never committed.

> **Tested topology:** **the 8xH100 target is untested; partially tested on AMD.** The 8xH100 target
> below has not been executed (a single-H100 run is documented under Install). What *has* been run is 1x AMD Instinct MI355X (gfx950,
> ROCm 7.2): **SFT and DPO train** in a plain ROCm venv, and **PPO and GRPO
> train too — but only inside a container that already ships a ROCm build of vLLM**
> (verified in `rocm/verl:...vllm0.20.2` on 4x MI355X). See
> "AMD MI355X (ROCm 7.2) — attempted" under Install for the exact route per tier.
> This folder was written against the upstream OpenRLHF
> README and `examples/scripts/` for the pinned versions below (OpenRLHF 0.10.x).
> It targets the repo default of a single node with 8xH100 80GB, using the Hybrid
> Engine (`--train.colocate_all`) so the actor, reference model and vLLM engines share those
> 8 GPUs. Multi-node via Ray is a documented upstream path (and the reason to use this
> framework at all) but is not exercised here. Verify every flag against your installed
> version before a real run — see "Upstream API uncertainty" in Notes.

## Install

Set these once per shell; the commands below reference them:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # training artifacts and run logs
```

### NVIDIA (CUDA)

Upstream treats **Docker as the supported path**, because Ray + vLLM + DeepSpeed +
flash-attn pins are version-sensitive:

```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:26.03-py3 bash

# Inside the container: remove packages that collide with OpenRLHF's own versions.
pip uninstall xgboost transformer_engine flash_attn pynvml -y

pip install openrlhf[vllm]
```

For an existing CUDA environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_openrlhf.txt

# flash-attn must be built AFTER torch is present, without build isolation.
pip install flash-attn==2.8.3 --no-build-isolation
```

The `[vllm]` extra pins a vLLM version the release was tested against — prefer it over
installing vLLM separately. If you need an unreleased fix (the namespaced CLI is recent and
moves quickly), install from source with `git clone ... && pip install -e .`.

#### NVIDIA H100 (CUDA 13.0) — TESTED, works in a plain venv (no container)

**This was actually run** on 1x NVIDIA H100 80GB HBM3 (Hopper, cc 9.0), driver 580.173.02,
CUDA 13.0, Ubuntu, Python 3.12.3 — a single free GPU (physical GPU 6) on a
shared 8-GPU node. **The headline result is the mirror image of the MI355X story: the RL
tiers are NOT blocked here.** On AMD, PPO/GRPO needed the `rocm/verl` container purely
because `pip install vllm` ships a CUDA-only wheel; on NVIDIA that wheel *is* the native
build, so `pip install openrlhf[vllm]` resolves a working vLLM into a plain venv and the
whole framework (SFT + the Ray/vLLM RL path) installs without Docker and without `--no-deps`
surgery. Per-tier verdict:

| Tier | Launcher route | Status on H100 (plain venv) | Note |
|---|---|---|---|
| **SFT** | `deepspeed --module openrlhf.cli.train_sft` | **WORKS** — loss 2.04 → 3.6e-6, 148 steps, HF checkpoint written, 11.5 GB resident on GPU 6 | needs the flash-attn fix below (or `sdpa`) |
| **DPO** | `deepspeed --module openrlhf.cli.train_dpo` | **expected to work** (same DeepSpeed path as SFT; not separately re-run on H100) | preference-triple data, same install |
| **PPO / GRPO** (+ RLOO / REINFORCE++ / Dr. GRPO) | `train_ppo_ray` (Ray + vLLM) | **UNBLOCKED — vLLM 0.27.1 installs natively** (see the RL note below) | the AMD blocker (CUDA-only vLLM wheel) does not exist here |

##### The exact install that worked

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf && source .env_openrlhf/bin/activate

# 1. Base torch FIRST (proxy on; pypi.org is allowlisted). Resolves the native CUDA 13 build.
pip install torch numpy
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130

# 2. OpenRLHF + the tested vLLM. On NVIDIA this is the whole install — no --no-deps needed.
#    Pulls: openrlhf 0.11.0, vllm 0.27.1, flash-attn 2.8.3 (wheel), deepspeed 0.19.5,
#    transformers 5.15.0, ray 2.55.0, accelerate 1.14.0, datasets 5.0.1. ~90 s, exit 0.
pip install 'openrlhf[vllm]'
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130
pip check   # -> "No broken requirements found."
```

**torch is NOT clobbered.** vLLM 0.27.1 pins `torch==2.13.0`, which is exactly the version
`pip install torch` resolves on a CUDA 13 host, so the `[vllm]` install is a no-op for torch and
`pip check` stays clean. (On AMD this same pin is what *would* overwrite a ROCm torch with a
CUDA one — the pin is identical; only the starting torch differs.)

##### The one real snag — the prebuilt flash-attn wheel is ABI-broken against torch 2.13.0

`openrlhf[vllm]` installs `flash-attn==2.8.3` from a **prebuilt wheel**, but that wheel's
compiled extension is built against a different torch ABI and fails to load:

```
import flash_attn_2_cuda as flash_attn_gpu
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
  _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

This is fatal for *every* trainer, not just the flash path: `openrlhf/models/actor.py`
imports `openrlhf/models/ring_attn_utils.py` at module level, which does
`from flash_attn.bert_padding import ...` — so `import openrlhf.cli.train_sft` itself dies
before any GPU code runs. (Same top-level-import mechanism the MI355X section documents,
different root cause — there it was a missing package, here it is an ABI mismatch.) Two
fixes, either works:

- **Rebuild flash-attn from source against the installed torch** — **this is what was done,
  and it works.** Also enables `--ds.attn_implementation flash_attention_2`:
  ```bash
  pip uninstall -y flash-attn
  MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
  ```
  Needs `nvcc` on `PATH` (system CUDA 13.0 at `/usr/local/cuda`, matches torch's cu130). The
  `nvcc` compile takes **~15–25 min** on an H100 (MAX_JOBS=32).
  After it lands, `import flash_attn_2_cuda` succeeds, a real
  `flash_attn_func` kernel runs on-GPU, both OpenRLHF trainers import with no shim, and SFT
  was re-run with the launcher's true defaults (`--ds.packing_samples` +
  `--ds.attn_implementation flash_attention_2`) — loss 2.39 → 3.2e-4, exit 0, 12.2 GB on
  GPU 6. (On AMD, `--ds.packing_samples` must be dropped because it silently forces
  `flash_attention_2`, which was unbuildable there in-budget; on H100 that default is fine.)
- **Or, to get SFT/DPO going immediately without the build, shim the eager import.**
  `flash_attn.bert_padding` and `flash_attn.utils.distributed` are pure torch/einops (no
  kernels) — only `flash_attn/__init__.py`'s eager import of `flash_attn_interface` (the
  broken `.so`) is the problem. Wrap it in `try/except ImportError: pass`, then run with
  `--ds.attn_implementation sdpa`. That is exactly what the SFT evidence below used.

##### Shared-node / DeepSpeed device-pinning trap (cost real time here)

`deepspeed --num_gpus 1` **ignores `CUDA_VISIBLE_DEVICES=6` and forces
`CUDA_VISIBLE_DEVICES=0`** — it prints `Detected VISIBLE_DEVICES=6 but ignoring it because
--num_gpus was used ... Setting CUDA_VISIBLE_DEVICES=0`, which silently lands your job on
**physical GPU 0** (someone else's production GPU on a shared box). The run still exits 0 and
"trains", so this is easy to miss. On a shared node you must pin by index instead:

```bash
unset CUDA_VISIBLE_DEVICES        # let DeepSpeed do the pinning
deepspeed --include localhost:6 --master_port 29646 --module openrlhf.cli.train_sft ...
# -> [launch.py] WORLD INFO DICT: {'localhost': [6]} ; Setting CUDA_VISIBLE_DEVICES=6
```

Also note: `nvidia-smi -i 6` is unreliable while `CUDA_VISIBLE_DEVICES=6` is exported (nvidia-smi
then sees only one remapped device); sample residency with `env -u CUDA_VISIBLE_DEVICES
nvidia-smi -i <GPU-UUID> ...` against the physical UUID.

##### The exact SFT smoke command (verified on physical GPU 6)

`data/OTel_LLM_sample_10.jsonl` is 10 rows; it was replicated ×8 into a 80-row file so the
run takes a non-trivial number of optimizer steps (this is a **pipeline proof, not a
learning result** — the model memorizes 10 distinct rows seen repeatedly). The 10-row file
also trains and checkpoints, at fewer steps.

```bash
export HF_HUB_OFFLINE=1
export PATH=<venv>/bin:/usr/local/cuda/bin:$PATH        # ninja (from the venv) + nvcc on PATH
export CUDA_HOME=/usr/local/cuda                        # DeepSpeed JITs FusedAdam at first run
unset CUDA_VISIBLE_DEVICES
deepspeed --include localhost:6 --master_port 29646 --module openrlhf.cli.train_sft \
  --model.model_name_or_path LiquidAI/LFM2.5-350M \
  --data.dataset sft_80.jsonl --data.max_len 1024 --data.max_samples 80 \
  --train.batch_size 4 --train.micro_batch_size 2 --train.max_epochs 12 \
  --adam.lr 5e-06 --ds.zero_stage 2 --ds.param_dtype bf16 \
  --ckpt.output_dir ./sft-ckpt --ckpt.save_steps -1 --logger.logging_steps 1 --eval.steps -1 \
  --model.gradient_checkpointing_enable \
  --data.input_key messages --data.apply_chat_template \
  --ds.attn_implementation sdpa
```

`LiquidAI/LFM2.5-350M` (arch `Lfm2ForCausalLM`, a hybrid conv+attention model) was used
because it was fully cached on the test node and the default `Qwen/Qwen2.5-7B-Instruct` was not.
Its chat template renders correctly (`<|im_start|>...`); OpenRLHF's `sft_dataset.py` calls
`apply_chat_template(tokenize=False)` (string render, then a separate tokenize), so it does
**not** hit the transformers-5.x `BatchEncoding` mask trap.

##### SFT — what a healthy run looks like (physical GPU 6, exit 0)

```
Train step of epoch 0:   0%|          | 0/17 [00:02<?, gpt_loss=2.04, lr=0, grad_norm=0]
Train step of epoch 0:   5%|▌         | 1/17 [00:04<.., gpt_loss=0.793, lr=1.25e-6, grad_norm=38.9]
...
Train step of epoch 11: 100%|██████████| 17/17 [00:02<00:00, gpt_loss=3.61e-6, lr=5.01e-7, grad_norm=0.000894]
[launch.py:367:main] Process <pid> exits successfully.        <- SFT exit code 0
```

`gpt_loss` falls 2.04 → 3.6e-6 and `grad_norm` 38.9 → 0.0009 over 148 optimizer steps
(12 epochs). `--ckpt.save_steps -1` still writes an HF checkpoint at fit-end:
`sft-ckpt/model.safetensors` = **843 MB** + `config.json`, `tokenizer.json`,
`chat_template.jinja`.

`nvidia-smi` sampled *in-band* against physical GPU 6's UUID (the training PID
holds VRAM on the assigned card — not on GPUs 0-3):

```
GPU6 compute-apps: PID <pid>, 518 MiB      <- model loading
GPU6 compute-apps: PID <pid>, 1174 MiB     <- optimizer + activations
GPU6 memory.used peak during run: 11493 MiB (~11.5 GB)
```

##### RL tiers (PPO / GRPO) on H100 — the AMD blocker is gone

The reason this folder exists is online RL with vLLM rollouts, and **the thing that forced a
container on AMD does not apply on NVIDIA.** `pip install openrlhf[vllm]` installed
**vllm 0.27.1 natively** (the CUDA build; verified `torch 2.13.0+cu130` survives, `pip check`
clean) alongside Ray 2.55.0 — no `rocm/verl` image, no `--no-deps`, no ROCm-vLLM source
build. So `openrlhf.cli.train_ppo_ray` (which PPO, GRPO, RLOO, REINFORCE++ and Dr. GRPO all
share) has its one hard dependency satisfied in the venv. The single-GPU RL launch differs
from this folder's 8xH100 defaults exactly as the MI355X container run did — colocate_all
wants `vllm.num_engines × vllm.tensor_parallel_size == actor GPUs`, so on one GPU:
`--vllm.num_engines 1 --vllm.tensor_parallel_size 1`, `--rollout.batch_size` a few,
`--max_samples 10`, and a lowered `--vllm.gpu_memory_utilization`. Start the Ray head first
(`ray start --head --num-gpus 1`, `ulimit -n` raised) and submit the `train_ppo_ray` command
(or run the module directly against the cluster).

##### GRPO actually ran end-to-end on 1x H100 (physical GPU 6) — plain venv, no container

This is the concrete proof of the upgrade. Ray head pinned to GPU 6 (`CUDA_VISIBLE_DEVICES=6`
so Ray reports `1.0 GPU`; `RAY_EXPERIMENTAL_NOSET_*` is **not**
needed on CUDA — only `CUDA_VISIBLE_DEVICES=6` was set, no HIP/NOSET vars), then
`train_ppo_ray` run directly against the cluster:

```bash
unset PYTHONPATH   # (real flash_attn was built; before the build, a pure-python flash_attn
                   #  shim on PYTHONPATH is what let train_ppo_ray import while running sdpa)
ray start --head --num-gpus 1 --num-cpus 16 --port 29656 --dashboard-port 29657
python3 -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path LiquidAI/LFM2.5-350M \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
  --data.prompt_dataset data/otel_rl.jsonl --data.input_key prompt --data.label_key answer \
  --data.max_len 1024 --data.max_samples 8 \
  --rollout.batch_size 4 --rollout.n_samples_per_prompt 4 --rollout.max_new_tokens 64 \
  --train.batch_size 4 --train.micro_batch_size 1 --train.max_epochs 1 \
  --actor.adam.lr 5e-06 --algo.kl.init_coef 0.01 --ds.zero_stage 2 --ds.param_dtype bf16 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 --vllm.gpu_memory_utilization 0.35 \
  --vllm.sync_backend nccl --vllm.enforce_eager \
  --ckpt.output_dir ./grpo-ckpt --ckpt.save_hf --logger.logging_steps 1 --eval.steps -1 \
  --actor.gradient_checkpointing_enable --algo.advantage.estimator group_norm \
  --reward.remote_url ./reward_func.py --data.apply_chat_template --ds.attn_implementation sdpa \
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep
```

Notes vs the 8-GPU defaults: single GPU means `--vllm.num_engines 1
--vllm.tensor_parallel_size 1` (colocate_all requires `num_engines × tp == actor GPUs`),
`--vllm.gpu_memory_utilization 0.35` so the vLLM engine, actor, and reference model share the
one 80 GB card, and tiny rollout/batch (`--max_samples 8`, `--rollout.batch_size 4`,
`--max_new_tokens 64`). `data/otel_rl.jsonl` (prompt/answer) comes from
`prepare_data_openrlhf.py`. `reward_func.py` is a rule-based `1.0 if answer in generation
else 0.0` stub matching the documented signature. vLLM natively supports `Lfm2ForCausalLM`
(confirmed in its arch registry), so the cached LFM2.5-350M drives the rollout side too.

**Expected output (exit 0):**

```
(EngineCore) [core.py:355] init engine (profile, create kv cache, warmup model) took 81.75 s
(EngineCore) [kv_cache_utils.py:2235] GPU KV cache size: 2,252,970 tokens
(EngineCore) [topk_topp_sampler.py:62] Using FlashInfer for top-p & top-k sampling.
(RolloutRayActor) (EngineCore) update weight: model.layers.15.feed_forward.w2.weight, dtype: torch.bfloat16, shape: [1024, 4608]   <- actor->vLLM NCCL weight sync
(EngineCore) [cumem.py:271] CuMemAllocator: sleep freed 27.29 GiB memory ...               <- colocate_all sleep/wake works
(PPOTrainer) ppo_trainer.py:547] ✨ Global step 1: {'response_length': 40.375, 'rollout/num_samples': 16.0, 'group_reward_std': 0.0, 'policy_loss': 0.0, 'timing/generation': 2.95, 'timing/ppo_train': 5.04, 'actor_lr': 3.25e-06, ...}
(PPOTrainer) ✨ Global step 2: {'response_length': 49.5625, 'rollout/num_samples': 16.0, 'timing/generation': 1.63, 'timing/ppo_train': 3.39, 'actor_lr': 5.0e-07, ...}
(PolicyModelActor) Writing model shards: 100%|██████████| 1/1                              <- HF checkpoint written
```

- **2 GRPO optimizer steps**, each a full make-experience → generate (vLLM) → score
  (`reward_func`) → group-normalized advantage → PPO actor update cycle, `rollout/num_samples
  16` per step. `accuracy`/`reward` are `0.0` because the rule-based reward found no
  answer-string match in the short 64-token completions (expected for a smoke; the loop and
  the metric plumbing are real — `group_reward_std`, `policy_loss`, `actor_grad_norm`,
  `ppo_kl` all computed and logged).
- **GPU-6 residency:** peak **33972 MiB (~34 GB)** on physical GPU 6 during rollout
  (vLLM KV cache + actor + ref colocated), sampled in-band against the GPU UUID. GPUs 0-3
  (the co-tenant production job) stayed at their own ~67 GB / 100 % util, untouched.
- **HF checkpoint:** `grpo-ckpt/model.safetensors` = 843 MB.
- **`--ds.packing_samples` was dropped** for this GRPO smoke to keep it on the `sdpa`
  code path during the window when flash-attn was still compiling; with the source-built
  flash-attn present, the launcher's default `flash_attention_2` + packing is available for
  RL too (verified working for SFT).

**RL tiers on H100:** GRPO ran to completion in a plain venv with native vLLM
generation, actor→engine NCCL weight sync, and colocate_all sleep/wake — **no `rocm/verl`
container, no `--no-deps` install surgery, no ROCm-vLLM source build.** The single dependency
that forced the container on MI355X (a CUDA-only vLLM wheel) is the *native* wheel here. PPO
is the same `train_ppo_ray` entrypoint plus a critic (`--critic.num_gpus_per_node 1`,
`--critic.adam.lr`); it was not separately re-run but has no
NVIDIA-specific blocker. RLOO / REINFORCE++ / Dr. GRPO are the same path with a different
`--algo.advantage.estimator`.

##### Multi-GPU (not run here)

The H100 validation was single-GPU only (one free GPU on a shared node). A 2- or 8-GPU pass
needs the other GPUs free and would set `--include localhost:0,1,...` (SFT/DPO) or
`--actor.num_gpus_per_node N` with `--vllm.num_engines × --vllm.tensor_parallel_size`
matching the actor GPU count (PPO/GRPO). Not launched here.

### AMD / ROCm

**No first-party AMD/ROCm support was found upstream.** The
[OpenRLHF README](https://github.com/OpenRLHF/OpenRLHF) documents only the NVIDIA path:
its quick start is `docker run --runtime=nvidia ... nvcr.io/nvidia/pytorch:26.03-py3`, the
repo's `dockerfile/` directory contains a single NVIDIA-based Dockerfile, weight sync uses
`--vllm.sync_backend nccl`, and the recommended attention path is `flash-attn` built
against CUDA. There is no ROCm Dockerfile, no AMD tutorial, and no ROCm install
documentation in the repository. Treat OpenRLHF as **NVIDIA-first**: running it on ROCm
would mean assembling ROCm builds of torch/vLLM/DeepSpeed/flash-attn yourself, entirely
unsupported by upstream. If AMD hardware is a requirement, use `../verl/` instead —
verl has first-party ROCm support (see `../verl/readme_verl.md`; that folder is
self-contained).

#### AMD MI355X (ROCm 7.2) — attempted, partial success

**This was actually run** on 1x AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4,
Ubuntu, Python 3.12.3. The outcome differs per tier, so read the table:

| Tier | Launcher route | Status on MI355X | Root cause |
|---|---|---|---|
| **SFT** | `deepspeed --module openrlhf.cli.train_sft` | **WORKS** (trained, loss decreasing, checkpoint written) | needs only torch + DeepSpeed, both ROCm-capable |
| **DPO** | `deepspeed --module openrlhf.cli.train_dpo` | **WORKS** (trained, `acc=1`, reward margin opening) | same |
| **PPO** | `train_ppo_ray` (Ray) | **BLOCKED in this venv** / **WORKS in the `rocm/verl` container** | the venv has no ROCm `vllm`; the container ships one — see [RL tiers unblocked](#rl-tiers-unblocked-on-mi355x-via-the-rocmverl-container) |
| **GRPO** (and RLOO / REINFORCE++ / Dr. GRPO) | `train_ppo_ray` (Ray) | **BLOCKED in this venv** / **WORKS in the `rocm/verl` container** | same — all RL modes share `train_ppo_ray` |
| `--ds.packing_samples` | any tier | **BLOCKED in this venv** / **WORKS in the container** | forces `flash_attention_2`, which the container already ships (ROCm flash-attn 2.8.4) |
| `--ds.ring_attn_size > 1` | any tier | **BLOCKED** | needs real `ring_flash_attn` kernels; not retested |

So the honest summary is: **the pip/venv route on AMD covers SFT and DPO only. The
RL-with-rollouts tiers — the entire reason to pick this folder — do run on AMD, but only
inside a container that already carries a ROCm build of vLLM.** GRPO and PPO were both
trained end-to-end on 4x MI355X this way; see
[RL tiers unblocked](#rl-tiers-unblocked-on-mi355x-via-the-rocmverl-container).

> **Scaled to 8 GPUs.** SFT and DPO were also run on all 8x MI355X and both pass
> (ZeRO-2, world size 8) — one batch-geometry change is required. PPO/GRPO remain blocked
> for the same vLLM reason. See
> [8-GPU run (8x MI355X, ROCm 7.2.4)](#8-gpu-run-8x-mi355x-rocm-724).

##### Blocker 1 — `flash-attn` is an unconditional dependency

The folder's `openrlhf[vllm]` line cannot install on ROCm. `flash-attn==2.8.3` is in
OpenRLHF's `install_requires`, **not** in an extra (true for 0.9.x through 0.11.0), so
every `pip install openrlhf` triggers a source build of it:

```
Collecting flash-attn==2.8.3 (from openrlhf[vllm])
  Getting requirements to build wheel: finished with status 'error'
  ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
```

That first error is only the documented build-isolation one. Two further facts matter:

- flash-attn 2.8.3 *does* have a ROCm branch and **explicitly allows gfx950**
  (`setup.py: allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]`), with
  `composable_kernel` bundled in the sdist. With `--no-build-isolation` its metadata
  generates fine against ROCm torch. So it is not *impossible* — it is a multi-hour
  `hipcc` CK build with no prebuilt ROCm wheel anywhere. Not attempted here.
- OpenRLHF barely uses it. The only import-time need is
  `openrlhf/models/ring_attn_utils.py`, which pulls `flash_attn.bert_padding`
  (`index_first_axis`, `pad_input`, `unpad_input`) and `flash_attn.utils.distributed.all_gather`
  — all **pure PyTorch/einops, no kernels**. But `openrlhf/models/actor.py` imports that
  module at top level, so *every* trainer including SFT dies on
  `ModuleNotFoundError: No module named 'flash_attn'` before it reaches any GPU code.

##### Blocker 2 — the vLLM **PyPI wheel** is CUDA-only, and would delete your ROCm torch

> **Scope correction.** This blocker is about *PyPI wheels only*, not about
> AMD hardware. A ROCm build of vLLM exists and runs fine on MI355X — it just does not
> come from `pip install vllm`. Ship OpenRLHF into a container that already has one and
> the RL tiers train; see
> [RL tiers unblocked](#rl-tiers-unblocked-on-mi355x-via-the-rocmverl-container).
> The claim below stands as written *for the pip route*.

`vllm==0.27.1` (the version OpenRLHF's `[vllm]` extra pins) declares:

```
torch==2.13.0                     <- plain PyPI CUDA build, NOT the rocm7.2 wheel
flashinfer-python==0.6.16.post3
nvidia-cudnn-frontend>=1.19.1
nvidia-cutlass-dsl[cu13]==4.6.0
```

`pip install 'vllm==0.27.1'` on a ROCm host resolves to `Collecting torch==2.13.0` plus
`cuda-python`, `nccl4py`, `nvidia-cuda-nvcc`, `nvidia-cuda-runtime` — i.e. it **silently
replaces the working ROCm torch with a CUDA one**, leaving an environment that imports but
cannot see the GPUs. ROCm vLLM ships only via source builds (`PYTORCH_ROCM_ARCH=gfx950`)
or AMD's first-party images (`rocm/vllm`, `rocm/verl`). Neither is *wired into* OpenRLHF —
but nothing in OpenRLHF requires CUDA either: `--vllm.sync_backend nccl` resolves to
**RCCL** on ROCm and the actor→engine weight broadcast works unmodified (proven below).

##### Blocker 3 — `--ds.packing_samples` silently re-enables flash-attn

A launcher default worth knowing about. Passing `--ds.attn_implementation sdpa` is **not
respected** while packing is on — `openrlhf/cli/train_sft.py:324` (and the identical block
in `train_dpo.py:358`, `train_rm.py:335`, `train_ppo_ray.py:656`) overrides your choice:

```python
if args.ds.packing_samples and "flash_attention" not in args.ds.attn_implementation:
    args.ds.attn_implementation = "flash_attention_2"
```

which surfaces as a confusing error long after you thought you had turned it off:

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the
following error: the package for FlashAttention2 doesn't seem to be installed.
```

`train_llm_openrlhf.py` sets `--packing_samples` **and** `--flash_attn` on by default, and
neither can be turned off from the command line (both are `action="store_true",
default=True`). In the venv, on ROCm, you must edit those two defaults in the script, or
bypass the launcher and call `deepspeed --module` yourself as below. **Inside the container
route this blocker disappears** — the image ships ROCm flash-attn 2.8.4, so both defaults
can stay on (the GRPO/PPO runs below ran with `--ds.packing_samples
--ds.attn_implementation flash_attention_2` untouched).

#### RL tiers unblocked on MI355X via the `rocm/verl` container

**GRPO and PPO both train on AMD.** Not in the venv — in a container that already
carries a ROCm build of vLLM. Ray placed the actor, reference, critic and four vLLM rollout
engines across **4x MI355X (physical GPUs 0-3)**, generation ran on the GPUs, RCCL synced
the updated actor weights into the engines every step, and the optimizer took finite steps
against a rule-based reward. An HF checkpoint was written.

##### Route chosen, and why

Reuse the **already-present** `rocm/verl` image (the same one `../verl/` used) as
nothing more than a ROCm runtime — ROCm torch + ROCm vLLM + ROCm flash-attn + Ray, all
prebuilt — and drop OpenRLHF into it with `--no-deps`. That sidesteps every blocker above
at once: no flash-attn source build, no CUDA torch clobbering, no multi-hour vLLM build.
`rocm/vllm` would work the same way; `rocm/verl` was chosen only because it was on disk.

```bash
# 1. Container (GPUs pinned by render node, NOT by env var: 0,1,2,3 -> renderD128/136/144/152)
docker run -d --name orlhf_mi355x \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --group-add 44 --group-add 993 \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=64g --network=host \
  -e HF_HOME=/hf_cache \
  -v "$PWD":/workspace/train_llm_openrlhf \
  -v "$HF_HOME":/hf_cache \
  -v "$OUTPUT_DIR/train_llm_openrlhf":/outputs \
  -w /workspace/train_llm_openrlhf \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity

# 2. OpenRLHF + the four things the image lacks. NOTHING ELSE — --no-deps is load-bearing,
#    a plain `pip install openrlhf` drags in flash-attn 2.8.3 (source build) and CUDA torch.
docker exec orlhf_mi355x pip install --no-deps \
  openrlhf==0.11.0 deepspeed==0.19.5 hjson py-cpuinfo nvidia-ml-py pynvml python-dotenv

# 3. Ray head. `ulimit -n` is load-bearing — see quirk 1.
docker exec orlhf_mi355x bash -lc \
  'ulimit -n 262144; ray start --head --num-gpus 4 --num-cpus 64 --port 6379 --dashboard-port 8265'
```

Versions actually exercised (all from the image except OpenRLHF/DeepSpeed):

| Component | Version |
|---|---|
| torch | `2.9.1.dev20251204+rocm7.0.2` |
| **vLLM** | **`0.20.2rc1.dev253+g1ff9d3353.rocm702`** (ROCm build — the point of this route) |
| ray | 2.55.1 |
| openrlhf | 0.11.0 (`--no-deps`, from PyPI) |
| deepspeed | 0.19.5 (`--no-deps`) |
| transformers | 5.14.1 |
| flash_attn | 2.8.4 (ROCm/CK, preinstalled — never pip-install it) |

##### The GRPO command that trained (4x MI355X)

`--reward_func` needs a file the Ray workers can see; it was written to the mounted
`/outputs` rather than into this folder. Same signature the readme documents above.

```bash
docker exec -e HF_TOKEN="$HF_TOKEN" -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 -e RAY_ADDRESS=127.0.0.1:6379 \
  orlhf_mi355x bash -lc 'ulimit -n 262144; cd /outputs/rt && python3 -m openrlhf.cli.train_ppo_ray \
 --actor.model_name_or_path Qwen/Qwen3-0.6B --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
 --ref.num_nodes 1 --ref.num_gpus_per_node 4 \
 --data.prompt_dataset data/otel_rl.jsonl --data.input_key prompt --data.label_key answer \
 --data.max_len 3072 --data.max_samples 10 \
 --rollout.batch_size 4 --rollout.n_samples_per_prompt 4 --rollout.max_new_tokens 128 \
 --train.batch_size 16 --train.micro_batch_size 1 --train.max_epochs 2 \
 --actor.adam.lr 5e-06 --algo.kl.init_coef 0.01 --ds.zero_stage 3 --ds.param_dtype bf16 \
 --vllm.num_engines 4 --vllm.tensor_parallel_size 1 --vllm.gpu_memory_utilization 0.35 \
 --vllm.sync_backend nccl --vllm.enforce_eager \
 --ckpt.output_dir /outputs/grpo_smoke --ckpt.save_hf --logger.logging_steps 1 --eval.steps -1 \
 --actor.gradient_checkpointing_enable --algo.advantage.estimator group_norm \
 --reward.remote_url /outputs/reward_func.py --data.apply_chat_template \
 --ds.packing_samples --ds.attn_implementation flash_attention_2 \
 --train.colocate_all --vllm.enable_sleep --ds.enable_sleep'
```

This is exactly what `train_llm_openrlhf.py --mode grpo ... --dry_run` prints, minus the
`ray job submit` wrapper (see quirk 2). Flag changes vs. the folder's 8xH100 defaults:
`--num_gpus 4`, `--vllm_num_engines 4 --vllm_tensor_parallel_size 1` (colocate_all wants
`engines x TP == actor GPUs`), `--rollout_batch_size 4` and `--max_samples 10` (the sample
has 10 rows), `--max_new_tokens 128`, `--max_len 3072` (prompts reach ~2.2k tokens),
`--vllm_gpu_memory_utilization 0.35`.

##### What a healthy run looks like

```
INFO [gpu_worker.py:462] Available KV cache memory: 99.0 GiB [repeated 3x across cluster]
INFO [kv_cache_utils.py:1710] GPU KV cache size: 926,896 tokens [repeated 3x across cluster]
        <- 4 vLLM engines, one per MI355X, KV cache allocated on device

(RolloutRayActor pid=<pid>) update weight: model.layers.25.self_attn.q_proj.weight, dtype:
  torch.bfloat16, shape: [2048, 1024] [repeated 1100x across cluster]
        <- actor -> engine weight sync over RCCL (--vllm.sync_backend nccl), every step

✨ Global step 1: {'reward': 0.4499, 'policy_loss': -2.37e-07, 'actor_grad_norm': 5.4978,
   'kl': 0.0, 'group_reward_std': 0.0330, 'rollout/num_samples': 16.0,
   'timing/generation': 2.97, 'timing/step_total': 12.19}
✨ Global step 2: {'reward': 0.6182, 'policy_loss': -0.00938, 'actor_grad_norm': 4.2731,
   'ppo_kl': 0.00104, 'ppo_clip_ratio': 0.0305, 'rollout/num_samples': 16.0}
✨ Global step 3: {'reward': 0.4096, 'policy_loss': -0.00390, 'actor_grad_norm': 18.08,
   'rollout/num_samples': 8.0, 'timing/step_total': 5.11}
        <- 3 GRPO steps, all metrics finite, group_reward_std > 0 so the groups differ
```

`rocm-smi` sampled mid-run (physical GPUs 0-3 only; 4-7 were in use elsewhere):

```
use=[14, 3, 9,12]  vram%=[31,31,31,31]   <- weights loading into the 4 engines
use=[ 3,11,62,71]  vram%=[38,38,38,38]   <- rollout generation
use=[48,74,19,53]  vram%=[ 6, 6, 8, 8]   <- engines asleep (--vllm.enable_sleep), actor training
```

Exit code 0, and `--ckpt.save_hf` wrote a real checkpoint:
`/outputs/grpo_smoke/model.safetensors` = **1.19 GB** + `config.json`, `tokenizer.json`,
`chat_template.jinja`.

##### PPO too — the critic path also runs

The same container, same data, plus a critic (`--critic.num_nodes 1
--critic.num_gpus_per_node 4 --critic.adam.lr 9e-6`, and **no**
`--algo.advantage.estimator`). 3 global steps, exit 0:

```
✨ Global step 1: {'critic_loss': 0.2135, 'values': 0.0491, 'critic_grad_norm': 10.559,
   'reward': 0.5894, 'actor_grad_norm': 7.4539, 'rollout/num_samples': 16.0}
✨ Global step 2: {'critic_loss': 0.0877, 'values': 0.0161, 'critic_grad_norm': 38.214,
   'reward': 0.3921, 'rollout/num_samples': 16.0}
        <- critic_loss falls 0.213 -> 0.088; a value head is genuinely being fit on ROCm
```

So **all `train_ppo_ray` modes are unblocked** by this route, not just GRPO: PPO (critic),
GRPO (`group_norm`) and by extension RLOO / REINFORCE++ / Dr. GRPO, which are the same
entry point with a different `--algo.advantage.estimator`. Only PPO and GRPO were executed.

##### Quirks that cost time on this route

1. **`ulimit -n` inside the container kills Ray.** The image's soft limit is **1024** (hard
   524288). With 256 CPUs visible, the raylet prestarts hundreds of workers, blows through
   the fd limit and **SIGABRTs the moment a driver connects**:
   ```
   (raylet) logging.cc:118: Unhandled exception: ... what(): epoll: Too many open files [system:24]
   core_worker_process.cc:216: Failed to register worker to Raylet: IOError: Failed to read data
     from the socket: End of file
   ```
   Fix: `ulimit -n 262144` in every shell that starts Ray *or* submits to it (or
   `--ulimit nofile=1048576:1048576` on `docker run`). `--num-cpus 64` on `ray start` also
   helps by cutting the prestart worker count.
2. **`ray job submit` never registered the job**; the module was run directly against the
   cluster (`RAY_ADDRESS=127.0.0.1:6379 python3 -m openrlhf.cli.train_ppo_ray ...`), which
   is the identical code path. The submit client hung after
   `Uploading package gcs://_ray_pkg_*.zip` with `ray job list` staying `[]`. Not
   investigated further — it is a Ray job-server issue, not an OpenRLHF or ROCm one.
3. **Never point `--working_dir` at this folder.** The launcher defaults to
   `{"working_dir": "."}`, and Ray then tries to upload the whole tree — including
   `.env_openrlhf/`, i.e. `libmagma.so` (1.29 GB), `libMIOpen.so` (1.02 GB),
   `librocrand.so` (766 MB)... Point it at a small dir holding just `data/`
   (`--working_dir /outputs/rt`), or add an `excludes` list.
4. **`--reward_func` must live on a path every Ray worker can open.** It is loaded inside
   the *vLLM engine actor* (`openrlhf/utils/agent.py:194`,
   `SingleTurnAgentExecutor.__init__`, via `importlib` when the path ends in `.py`), not in
   the driver. A mounted absolute path (`/outputs/reward_func.py`) is the simple answer.
5. Harmless noise that is **not** a failure: `quark ... ImportError: cannot import name
   'RoutedExperts'` (the image's quantization plugin vs. this vLLM), `NCCL WARN Could not
   read node # 9` from RCCL topology detection, and `Unknown vLLM environment variable
   detected: VLLM_USE_V1`. All three appear in runs that go on to train cleanly.

##### What actually worked — reproducible commands

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf          # git-ignored via .env_*/
source .env_openrlhf/bin/activate
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7   # never leave these empty on ROCm

# 1. ROCm torch FIRST, so nothing else drags in a CUDA build.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2

# 2. OpenRLHF WITHOUT its dependency list (this is what skips flash-attn).
pip install openrlhf==0.11.0 --no-deps

# 3. Its dependencies by hand, minus flash-attn and minus the vllm extra.
pip install accelerate aiohttp bitsandbytes datasets deepspeed==0.19.5 einops \
  'grpcio>=1.74.0' 'huggingface-hub>=1.0.0' isort jsonlines loralib optimum \
  'optree>=0.15.0' packaging peft pylatexenc 'pynvml>=12.0.0' 'ray[default]==2.55.0' \
  sympy tensorboard torchdata torchmetrics tqdm transformers==5.15.0 \
  transformers-stream-generator wandb wheel python-dotenv

# 4. Shim the pure-python half of flash-attn so actor.py can import.
#    bert_padding.py and utils/distributed.py are plain torch+einops - no kernels.
SP=$(python -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/flash_attn/utils" && touch "$SP/flash_attn/utils/__init__.py"
# copy bert_padding.py -> $SP/flash_attn/ and utils/distributed.py -> $SP/flash_attn/utils/
# from the flash_attn-2.8.3 sdist, then an __init__.py with __version__ = "2.8.3+rocm-shim"
```

Verify the ROCm torch survived step 3 — this is the step where pip can quietly swap it:

```bash
python -c "import torch; print(torch.__version__)"   # must still end in +rocm7.2
# observed: 2.11.0+rocm7.2 ... gfx950:sramecc+:xnack-
```

Then run SFT directly (no `--ds.packing_samples`, explicit `sdpa`):

```bash
deepspeed --num_gpus 1 --master_port 29760 --module openrlhf.cli.train_sft \
  --model.model_name_or_path Qwen/Qwen3-0.6B \
  --data.dataset data/OTel_LLM_sample_10.jsonl \
  --data.max_len 1024 --data.max_samples 8 \
  --train.batch_size 2 --train.micro_batch_size 1 --train.max_epochs 1 \
  --adam.lr 5e-06 --ds.zero_stage 2 --ds.param_dtype bf16 \
  --ckpt.output_dir ./checkpoint/rocm-sft --ckpt.save_steps -1 \
  --logger.logging_steps 1 --eval.steps -1 \
  --model.gradient_checkpointing_enable \
  --data.input_key messages --data.apply_chat_template \
  --ds.attn_implementation sdpa
```

Observed on MI355X (exit 0, checkpoint written):

```
Train step of epoch 0: 60%|██████  | 3/5 [00:06<00:03, gpt_loss=3.17, lr=5e-7, grad_norm=43.3]
Train step of epoch 0: 100%|██████████| 5/5 [00:06<00:00, gpt_loss=2.22, lr=5e-7, grad_norm=43.3]
Writing model shards: 100%|██████████| 1/1 [00:00<00:00, 2.47it/s]
```

DPO on the same env (preference triples as full chat lists in `chosen`/`rejected`, no
`--data.prompt_key` — passing a chat-list `prompt` alongside string `chosen` makes
`datasets` fail feature alignment, which is a schema trap, not a ROCm one):

```
Train step of epoch 1: 100%|██████████| 10/10 [00:01<00:00, loss=0.000357, acc=1,
  chosen_reward=3.71, reject_reward=-4.23, lr=5e-7, grad_norm=0.094]
```

Ray itself is fine on this hardware — in this venv it is only the missing ROCm vLLM that
blocks the RL tiers (supply one via a container and they run, see
[RL tiers unblocked](#rl-tiers-unblocked-on-mi355x-via-the-rocmverl-container)):

```python
>>> ray.cluster_resources()
{'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, 'GPU': 1.0}
>>> # a @ray.remote(num_gpus=1) task reports: gfx950:sramecc+:xnack-
```

Set `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` (the ROCm analogue of the
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` mentioned above) for correct per-worker
device binding.

##### Versions used

`torch 2.11.0+rocm7.2` (HIP 7.2.26015, triton-rocm 3.6.0) · `deepspeed 0.19.5` (pure-python
wheel, JIT ops) · `transformers 5.15.0` · `ray 2.55.0` · `peft 0.20.0` · `accelerate 1.14.0`
· `openrlhf 0.11.0` (`--no-deps`) · `vllm` not installed · `flash-attn` not installed.
(Container versions for the RL tiers are listed in the RL section above.)

##### Recommendation

For **RL on AMD**, the cheapest working setup is OpenRLHF `--no-deps` inside a ROCm image
that already has vLLM (`rocm/verl` or `rocm/vllm`) — proven above. `../verl/`
remains the lower-friction option if you do not specifically need OpenRLHF, since verl
ships first-party `rocm/verl` images with vLLM and SGLang already built for MI300X/MI355X
and needs no `--no-deps` surgery. For **SFT/DPO on AMD**, prefer
`../deepspeed/` or `../fsdp/`: they reach the same result without
`--no-deps`, without a flash-attn shim, and without upstream pins that assume CUDA.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

The section above was a **1-GPU** result. Both working tiers were also run on **all 8**
MI355X cards in a single locked session (~2.5 minutes wall clock), same
`Qwen/Qwen3-0.6B`, same venv, same `--no-deps` install. Results:

| Tier | 1-GPU result | 8-GPU result | Note |
|---|---|---|---|
| **SFT** | WORKS | **WORKS WITH CHANGES** | only change is batch geometry (see below); exit 0 |
| **DPO** | WORKS | **WORKS WITH CHANGES** | same batch-geometry change; exit 0, `acc=1` |
| **PPO** | BLOCKED in venv | **still BLOCKED in venv** | `vllm` still not installed here — but **WORKS in the container**, 4 GPUs, see the RL section |
| **GRPO** / RLOO / REINFORCE++ | BLOCKED in venv | **still BLOCKED in venv** | same `train_ppo_ray` + vLLM path; **WORKS in the container** |
| `--ds.packing_samples` | BLOCKED in venv | **still BLOCKED in venv** | forces `flash_attention_2`, absent here; present in the container |

`pip show vllm` → `WARNING: Package(s) not found: vllm`, `pip show flash-attn` → likewise.
The RL blocker is a dependency problem, not a scale problem, so 8 GPUs does not move it —
and supplying the dependency from a ROCm image *does* move it, at 4 GPUs. See
[RL tiers unblocked](#rl-tiers-unblocked-on-mi355x-via-the-rocmverl-container).

##### The 8-GPU gotcha: batch geometry tightens with world size

**This is the one thing that breaks when you go from 1-2 GPUs to 8.** DeepSpeed requires

```
train_batch_size == micro_train_batch_size * gradient_accumulation_steps * world_size
```

and OpenRLHF derives the accumulation count by integer division in
`openrlhf/utils/deepspeed/deepspeed.py:110`:

```python
self.accumulated_gradient = (
    self.train_batch_size * self.ring_attn_size * self.ds_tensor_parallel_size
    // self.micro_train_batch_size // self.world_size
)
```

The 1-GPU settings used above were `--train.batch_size 2 --train.micro_batch_size 1`. At
`world_size=8` that yields `2 // 1 // 8 == 0` accumulation steps, and DeepSpeed rejects the
config before any GPU work happens. There is **no OpenRLHF-side assert** with a helpful
message — the failure surfaces from DeepSpeed's own batch-parameter check. Note also that
`--train.batch_size` is the **GLOBAL** batch and `--train.micro_batch_size` is **per-GPU**.

Geometry actually used at 8 ranks: **global 64 = micro 2 × grad-accum 4 × world 8.**

##### Exact launch commands

Run with `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
exported *after* sourcing the venv, and assert `torch.cuda.device_count() == 8` first.
(Check `.env_openrlhf/bin/activate` for a stale `VISIBLE_DEVICES` pin — grep the whole file
before trusting it.)

```bash
deepspeed --num_gpus 8 --master_port 29730 --module openrlhf.cli.train_sft \
  --model.model_name_or_path Qwen/Qwen3-0.6B \
  --data.dataset $OUT/sft_640.jsonl \
  --data.input_key messages --data.apply_chat_template \
  --data.max_len 4096 --data.max_samples 640 \
  --train.batch_size 64 --train.micro_batch_size 2 --train.max_epochs 2 \
  --adam.lr 5e-06 \
  --ds.zero_stage 2 --ds.param_dtype bf16 --ds.attn_implementation sdpa \
  --model.gradient_checkpointing_enable \
  --ckpt.output_dir $OUT/hf_sft \
  --ckpt.path $OUT/ds_ckpt_sft --ckpt.save_steps 10 --ckpt.max_num 1 \
  --logger.logging_steps 1 --eval.steps -1

deepspeed --num_gpus 8 --master_port 29731 --module openrlhf.cli.train_dpo \
  --model.model_name_or_path Qwen/Qwen3-0.6B \
  --data.dataset $OUT/dpo_648.jsonl \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --data.max_len 1024 --data.max_samples 648 \
  --train.batch_size 64 --train.micro_batch_size 2 --train.max_epochs 2 \
  --adam.lr 5e-06 \
  --ds.zero_stage 2 --ds.param_dtype bf16 --ds.attn_implementation sdpa \
  --model.gradient_checkpointing_enable \
  --ckpt.output_dir $OUT/hf_dpo \
  --ckpt.path $OUT/ds_ckpt_dpo --ckpt.save_steps 10 --ckpt.max_num 1 \
  --logger.logging_steps 1 --eval.steps -1
```

`deepspeed --num_gpus 8` prints `Detected VISIBLE_DEVICES=0,...,7 but ignoring it because
one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used` — that
warning is benign here, and the launcher then re-sets the full list itself:

```
[launch.py:162] WORLD INFO DICT: {'localhost': [0, 1, 2, 3, 4, 5, 6, 7]}
[launch.py:180] dist_world_size=8
[launch.py:184] Setting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

##### Parallelism: ZeRO stage 2 confirmed active across 8 ranks

Not taken on faith from the flag. `--ckpt.save_steps 10` was set so DeepSpeed would write a
real ZeRO checkpoint, which produced **exactly 8 per-rank optimizer shards** per tier:

```
ds_ckpt_sft/global_step20/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt  894081221 bytes
ds_ckpt_sft/global_step20/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt  894081221 bytes
...                                          (ranks 2-6, ~894 MB each) ...
ds_ckpt_sft/global_step20/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt  894083525 bytes
```

The arithmetic confirms real sharding rather than 8 replicas: Qwen3-0.6B is 596M
parameters, and fp32 Adam state is 12 bytes/param (fp32 master weights + momentum +
variance) = **7.15 GB total**. `7.15 GB / 8 = 894 MB`, which is exactly the observed shard
size. So each rank holds 1/8 of the optimizer state. `ds_ckpt_dpo/global_step20/` is
identical. Parallelism is therefore **pure data parallel × 8 with ZeRO-2 optimizer/gradient
sharding** — no tensor, pipeline, or sequence parallelism (`ring_attn_size=1`,
`ds_tensor_parallel_size=1`; both remain unavailable, see the ring-attention blocker above).

##### What a healthy run looks like

SFT (progress bar counts **micro-steps per rank**: 640 samples ÷ 8 ranks ÷ micro 2 = 40):

```
Train step of epoch 0:   0%|    | 0/40 [00:07<?, ?it/s, gpt_loss=1.37, lr=0, grad_norm=0]
Train step of epoch 0:  50%|██▌ | 20/40 [00:14<00:04, 4.74it/s, gpt_loss=1.5, lr=4.53e-6, grad_norm=12.6]
Train step of epoch 1: 100%|████| 40/40 [00:09<00:00, 4.13it/s, gpt_loss=0.658, lr=5e-7, grad_norm=7.3]
=== SFT@8 exit code 0 ===
```

DPO — accuracy climbs off chance and the reward margin opens from ~0 to ~16:

```
Train step of epoch 0:   0%|    | 0/40 [00:07<?, ?it/s, loss=0.692, acc=0.125, chosen_reward=0.000241, reject_reward=-0.00159]
Train step of epoch 0:  25%|█▌  | 10/40 [00:09<00:08, 3.63it/s, loss=0.373, acc=1, chosen_reward=0.543, reject_reward=-0.287]
Train step of epoch 1: 100%|████| 40/40 [00:08<00:00, 4.72it/s, loss=1.93e-6, acc=1, chosen_reward=4.88, reject_reward=-11.4]
=== DPO@8 exit code 0 ===
```

Both tiers exited 0 with no teardown hang. Step time at 8 ranks was **~4.1-4.7 it/s** of
micro-steps; wall clock was 80 s for SFT and 76 s for DPO including model load and the
checkpoint write. The only teardown noise is a benign
`destroy_process_group() was not called before program exit` warning from each rank.

##### 8-GPU evidence (rocm-smi sampled *in-band*, cross-checked against the run's own PIDs)

Sample from inside the training script while it runs, not afterwards from a separate shell
— on a shared box an after-the-fact sample can capture a different job. Mid-SFT:

```
########## TIER=sft ##########
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,28,309220868096,2726113280      card4,27,309220868096,3028103168
card1,29,309220868096,3032297472      card5,27,309220868096,2975678464
card2,27,309220868096,3036495872      card6,27,309220868096,3009241088
card3,27,309220868096,2994556928      card7,26,309220868096,2965192704
```

All 8 cards busy and holding VRAM simultaneously. The PID cross-check in the same sample
shows the VRAM holders are the run's own ranks — `rocm-smi --showpids` lists 8 `python3`
PIDs, and `pgrep -af` in the same sample resolves those to
`openrlhf.cli.train_sft --local_rank=0` … `--local_rank=7` under one
`deepspeed.launcher.launch` parent. DPO repeats this pattern.

##### Per-GPU VRAM, 8 ranks vs 1 rank

| Phase | Per-GPU VRAM at 8 ranks |
|---|---|
| after model load (SFT) | ~2.7-3.0 GB |
| steady-state SFT training | ~9.6-10.5 GB |
| steady-state DPO training (ref model resident) | ~12.0-13.0 GB |
| peak, during ZeRO checkpoint write | 24-50 GB (allocator high-water, transient) |

The clean, measurable sharding delta is the optimizer state: the **full 7.15 GB** fp32 Adam
state sits on one card in the 1-GPU run, versus **894 MB per card** at 8 ranks — an 8x
reduction, confirmed by the shard sizes above. A direct 1-GPU VRAM byte comparison is *not*
offered here: the 1-GPU `rocm-smi` capture recorded only `VRAM%` (which read 0
for a model this small) and no byte counts, so there is no honest baseline number to
subtract. DPO sits ~2.5 GB/GPU above SFT because it keeps a frozen reference model resident
alongside the policy.

##### Everything that differed from the 1-2 GPU run

1. **Batch geometry had to change** — `batch 2 / micro 1` is invalid at world size 8
   (accum → 0). Replaced with global 64 / micro 2 / accum 4. This is the only *code-level*
   change required; no new package, no new pin, so `requirements_openrlhf.txt` is unchanged.
2. **The dataset had to grow.** `data/OTel_LLM_sample_10.jsonl` is 10 rows — at a global
   batch of 64 that is not even one optimizer step, and 8 ranks cannot each be fed. The 10
   rows were replicated ×64 into `sft_640.jsonl` (640 rows) and the earlier 10-row
   preference file ×72 into `dpo_648.jsonl`, both written **outside the repo** under
   `$OUTPUT_DIR/train_llm_openrlhf/gpu8/`. **This is a pipeline proof, not a
   learning result** — the loss curve is fitting 10 distinct rows seen 64 times over, and
   DPO's `acc=1` after ~10 steps means the model memorized a 10-row preference set, exactly
   as it did at 1 GPU.
3. **Sample survival was verified**, because OpenRLHF silently drops rows longer than the
   limit (`sft_dataset.py:156` and `reward_dataset.py:124` skip when
   `prompt_ids_len >= max_length - 2`). With `--data.max_len 4096`, all **640/640** SFT rows
   survived — confirmed arithmetically, since 40 micro-steps × 8 ranks × micro 2 = 640. For
   DPO at `max_len 1024`, 640 of the 648 rows were consumed, the remainder dropped by the
   sampler's `drop_last`, not by the length filter.
4. **`--master_port` must differ per tier** when running tiers back-to-back (29730 / 29731);
   reusing a port immediately after a teardown can hit a lingering bind.
5. Unchanged: ZeRO stage 2, `--ds.attn_implementation sdpa` (mandatory — `sdpa` is still the
   only viable attention path without flash-attn), bf16, gradient checkpointing, and the
   `--no-deps` + flash-attn-shim install recipe. Nothing about the ROCm workaround set
   needed revisiting at 8 GPUs.

##### Reproduction note

On a shared box, serialize against other GPU users and capture evidence in-band:

```bash
nohup flock -w 25200 /tmp/mi355x_gpu8.lock bash /tmp/run8_openrlhf.sh \
    > $OUTPUT_DIR/train_llm_openrlhf/gpu8/run.log 2>&1 &
```

Logs and the distilled evidence file land under
`$OUTPUT_DIR/train_llm_openrlhf/gpu8/` (`run.log`, `rocm_smi_inband.txt`,
`evidence_8gpu.txt`). The HF weights and ZeRO checkpoints these runs produce are ~19 GB and
carry no value beyond the proof — delete them once the shard listing is captured.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hugging Face token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The launcher calls `load_dotenv('dev.env')` and reads `HF_TOKEN` from the environment,
warning (not failing) when it is absent, since ungated models do not need one. `dev.env` is
**git-ignored** — never commit tokens, and never inline one into a script or a `ray job
submit` line. Weights-and-Biases keys are passed the same way: read `WANDB_API_KEY` from
`dev.env` and append `--logger.wandb.key "$WANDB_API_KEY"` yourself rather than hardcoding.

If Ray workers need the token too, export it before submitting, or add it to the runtime env:
`--runtime-env-json '{"working_dir": ".", "env_vars": {"HF_TOKEN": "..."}}'` — note that this
puts the token in your shell history, so prefer exporting it on the nodes.

## Data

This folder ships a 10-row sample, `data/OTel_LLM_sample_10.jsonl` — telecom-spec chat
records in the repo's usual format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version`). **Null-column trap:** `source_spec_id` and
`source_version` are null in most rows. OpenRLHF only reads the fields you name via
`--data.*_key`, so the extras are ignored — but any other loader that infers a schema from
mostly-null columns may mistype or reject them.

OpenRLHF reads HF dataset ids **or** local JSONL, and you tell it which JSON field plays
which role via `--data.*_key` flags. That indirection is why the launcher exposes
`--input_key`, `--output_key` and `--label_key`.

**SFT** — the shipped sample works as-is: `--input_key messages` with
`--apply_chat_template` (both launcher defaults).

**DPO** — preference triples (not derivable from the shipped sample):
```json
{"prompt": "Define entropy.", "chosen": "A measure of disorder.", "rejected": "I don't know."}
```
The launcher passes `--data.chosen_key chosen --data.rejected_key rejected`.

**PPO / GRPO** — needs a prompt field plus a ground-truth field for the reward function.
The chat sample has no `answer` field, so convert first — the last assistant turn becomes
`answer`, the preceding turns become `prompt`:

```bash
python prepare_data_openrlhf.py
# equivalent to:
# python prepare_data_openrlhf.py --input data/OTel_LLM_sample_10.jsonl --output data/otel_rl.jsonl
```

Output rows look like:
```json
{"prompt": [{"role": "user", "content": "..."}], "answer": "..."}
```

Launch RL with `--dataset data/otel_rl.jsonl --input_key prompt` (the launcher's
`--label_key` already defaults to `answer` to line up with this file). `--data.label_key`
is what makes the `answer` field arrive as `labels` in your reward function.

**Reward function** (`--reward_func /abs/path/reward_func.py`) — a module exposing
`reward_func`. Verified upstream signature:

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    """queries = prompt+response text, prompts = prompts only, labels = --data.label_key values."""
    rewards = torch.tensor([1.0 if str(l) in q else 0.0 for q, l in zip(queries, labels)])
    return {
        "rewards": rewards,      # used for advantage calculation
        "scores": rewards,       # 0-1, used by dynamic filtering
        "extra_logs": {"accuracy": rewards.mean().item()},
    }
```

Alternatively serve a trained reward model with `--reward_model <repo-or-path>`, or point
`--reward.remote_url` at an HTTP endpoint.

## Run

### Smoke test against the shipped sample

`--dry_run` prints the exact command without executing anything — no GPU, cluster, or even
OpenRLHF install needed:

```bash
# SFT on the chat sample (all data defaults line up already):
python train_llm_openrlhf.py --mode sft --dry_run

# GRPO on the converted sample:
python prepare_data_openrlhf.py
python train_llm_openrlhf.py --mode grpo --dataset data/otel_rl.jsonl \
  --input_key prompt --reward_func $PWD/reward_func.py --dry_run
```

### Start the Ray cluster first (PPO/GRPO only)

This is the real-world usage path and the step most easily missed — `ray job submit` fails
immediately if no head node is listening.

```bash
# On the head node:
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# On each additional node (multi-node only):
ray start --address <MASTER-NODE-IP>:6379 --num-gpus 8

# Confirm the cluster sees every GPU before submitting:
ray status
```

The Ray dashboard listens on `http://127.0.0.1:8265`; that is the `--ray_address` the
launcher submits to. Set `export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` if you hit
DeepSpeed GPU-index errors.

### Full examples

**GRPO** (8xH100, Hybrid Engine, rule-based reward):

```bash
nohup python train_llm_openrlhf.py \
  --mode grpo \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset data/otel_rl.jsonl \
  --input_key prompt --label_key answer \
  --reward_func $PWD/reward_func.py \
  --n_samples_per_prompt 8 \
  --rollout_batch_size 1024 --batch_size 128 \
  --output_dir ./checkpoint/qwen7b-grpo \
  > train_llm_openrlhf.log 2>&1 &

tail -f train_llm_openrlhf.log
```

**PPO** (adds a critic and a served reward model):

```bash
nohup python train_llm_openrlhf.py \
  --mode ppo \
  --model_name OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_model OpenRLHF/Llama-3-8b-rm-700k \
  --dataset OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --output_dir ./checkpoint/llama3-8b-ppo \
  > train_llm_openrlhf.log 2>&1 &

tail -f train_llm_openrlhf.log
```

**SFT / DPO** (DeepSpeed, no Ray cluster needed):

```bash
nohup python train_llm_openrlhf.py \
  --mode sft --model_name meta-llama/Meta-Llama-3-8B \
  --dataset data/OTel_LLM_sample_10.jsonl --input_key messages \
  --output_dir ./checkpoint/llama3-8b-sft \
  > train_llm_openrlhf.log 2>&1 &

tail -f train_llm_openrlhf.log
```

**What "working" looks like:** the launcher logs the mode and the full command, then hands
off. For RL runs, `ray job submit` streams the job's output back: you should see Ray placing
actor/reference (and critic) workers, vLLM engines loading the model and reporting KV-cache
size, then repeating generate -> make_experience -> train cycles. The meaningful signals per
step are a rising mean `reward`, a `kl` that stays small and bounded rather than climbing,
and `response_length` that does not collapse to near-zero or run straight to
`--max_new_tokens`. Per-phase timings (`timing/generation`, `timing/make_experience`,
`timing/ppo_train`) tell you where wall-clock is going; generation normally dominates. For
SFT/DPO you get ordinary DeepSpeed loss lines instead.

## Arguments

`train_llm_openrlhf.py` (every flag):

| Argument | Default | What it does |
|---|---|---|
| `--mode` | `grpo` | `sft`/`dpo` route to `deepspeed --module`; `ppo`/`grpo` route to `ray job submit`. |
| `--model_name` | `Qwen/Qwen2.5-7B-Instruct` | Actor/policy model (`--actor.model_name_or_path`, or `--model.model_name_or_path` for SFT/DPO). |
| `--dataset` | `data/OTel_LLM_sample_10.jsonl` | HF dataset id or local JSONL. Default is the shipped chat sample (SFT-ready). |
| `--output_dir` | `./checkpoint/openrlhf-run` | Checkpoint directory (`--ckpt.output_dir`). |
| `--num_gpus` | `8` | GPUs on this node, given to every Ray role. |
| `--input_key` | `messages` | SFT/PPO prompt field (`--data.input_key`). Use `prompt` with the converted RL file. |
| `--output_key` | `None` | SFT target field; leave unset with chat templates. |
| `--label_key` | `answer` | PPO/GRPO ground-truth field (`--data.label_key`); matches `data/otel_rl.jsonl`. |
| `--apply_chat_template` | on | Adds `--data.apply_chat_template`. |
| `--max_len` | `4096` | Max sequence length (`--data.max_len`). |
| `--max_samples` | `100000` | Dataset row cap (`--data.max_samples`). |
| `--batch_size` / `--micro_batch_size` | `128` / `2` | Global and per-GPU training batch. |
| `--max_epochs` | `1` | Training epochs (`--train.max_epochs`). |
| `--learning_rate` | `5e-6` | SFT/DPO: `--adam.lr`; RL: `--actor.adam.lr`. |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage for the training-side models. |
| `--packing_samples` | on | Adds `--ds.packing_samples`. |
| `--flash_attn` | on | Adds `--ds.attn_implementation flash_attention_2`. |
| `--reward_model` | `None` | PPO: served reward model (`--reward.model_name_or_path`). |
| `--reward_func` | `None` | Path to `reward_func.py` (`--reward.remote_url`). One of these two is required for RL. |
| `--critic_lr` | `9e-6` | PPO critic LR (`--critic.adam.lr`). |
| `--kl_coef` | `0.01` | `--algo.kl.init_coef`. `0` drops the reference model entirely. |
| `--rollout_batch_size` | `1024` | Prompts per generation phase (`--rollout.batch_size`). |
| `--n_samples_per_prompt` | `8` | Completions per prompt. **Must be >1 for GRPO** — the group is the baseline. |
| `--max_new_tokens` | `1024` | Rollout generation length (`--rollout.max_new_tokens`). |
| `--vllm_num_engines` / `--vllm_tensor_parallel_size` | `4` / `2` | vLLM generation engines and TP width. |
| `--vllm_gpu_memory_utilization` | `0.5` | Lower it when the Hybrid Engine OOMs; 0.5 is upstream's 8xA100/H100 starting point. |
| `--colocate_all` | on | Hybrid Engine: actor/ref/critic/vLLM share GPUs, adding `--vllm.enable_sleep --ds.enable_sleep`. |
| `--ray_address` | `http://127.0.0.1:8265` | Ray dashboard to submit the job to. |
| `--working_dir` | `.` | Shipped to Ray workers as the runtime env. |
| `--dry_run` | off | Print the assembled command and exit. Use this first, every time. |

`prepare_data_openrlhf.py`:

| Argument | Default | What it does |
|---|---|---|
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Source chat JSONL. |
| `--output` | `data/otel_rl.jsonl` | Destination JSONL with `prompt` (chat list) + `answer` (string). |

## Output

- **Checkpoints** land in `--output_dir` (`--ckpt.output_dir`). The launcher passes
  `--ckpt.save_hf` for RL runs, so the actor is written in HuggingFace format and loads with
  `AutoModelForCausalLM.from_pretrained` — no conversion step. Resume with
  `--ckpt.load_enable`; `--ckpt.save_steps` and `--ckpt.max_num` control retention.
- **Launcher stdout** goes to `train_llm_openrlhf.log` via the `nohup` line above. For RL
  runs this contains the streamed Ray job output.
- **Ray logs** live under `/tmp/ray/session_latest/logs/` on each node — that is where the
  real traceback goes when a worker dies, since the submitting process often only sees a
  generic job failure. Check there first when a run dies without an obvious cause.
- **Metrics** go to wandb (`--logger.wandb.key`) or TensorBoard (`--logger.tensorboard_dir`);
  neither is enabled by this launcher by default.
- **LoRA runs** save adapters only. Merge before serving:
  `python -m openrlhf.cli.lora_combiner --model_path <base> --lora_path <adapter> --output_path <merged> --ds.param_dtype bf16`.

## Hardware support & evidence

| Hardware | Status | Evidence |
|---|---|---|
| NVIDIA (A100/H100 class) | Primary and only documented target | Upstream README quick start uses `--runtime=nvidia` with `nvcr.io/nvidia/pytorch:26.03-py3`; NCCL weight sync; CUDA flash-attn — <https://github.com/OpenRLHF/OpenRLHF> |
| AMD / ROCm | **Not supported upstream** | No ROCm docs, Dockerfile, or install path anywhere in the repository; `dockerfile/` contains a single NVIDIA image definition |
| AMD MI355X (gfx950, ROCm 7.2) — SFT / DPO | **tested here — works**, off the supported path | 1x MI355X, torch 2.11.0+rocm7.2 + deepspeed 0.19.5; needs `openrlhf --no-deps`, a pure-python `flash_attn` shim, `--ds.attn_implementation sdpa`, and no `--ds.packing_samples`. See "AMD MI355X (ROCm 7.2) — attempted" |
| AMD MI355X (gfx950, ROCm 7.2) — PPO / GRPO | **tested here — blocked in a plain venv; works in the `rocm/verl` container** | Plain venv: `openrlhf.cli.train_ppo_ray` needs `vllm`; `vllm==0.27.1` pins `torch==2.13.0` (CUDA) and would overwrite the ROCm torch. Inside `rocm/verl:...vllm0.20.2`, PPO and GRPO trained on 4x MI355X — see "RL tiers unblocked on MI355X via the `rocm/verl` container" |

The upstream rows record OpenRLHF's documented support. The MI355X rows are **local
verification** from an actual run — the install route, error text and
per-tier evidence are in the "AMD MI355X (ROCm 7.2) — attempted" section under Install.
For AMD **RL**, use `../verl/`; for AMD **SFT/DPO**, `../deepspeed/` or
`../fsdp/` are far less work than making this folder go.

**Other hardware (upstream claims — not verified here):** OpenRLHF documents no
supported hardware beyond NVIDIA CUDA. Huawei Ascend NPU support exists only as an
unmerged work-in-progress PR (OpenRLHF/OpenRLHF#605) and an open feature request
(#852) — nothing official. No Intel Gaudi/XPU, Apple MPS, TPU, Trainium, or CPU
training path is claimed upstream.

## Notes

- **Why a launcher, not a trainer.** OpenRLHF's entire interface is its CLI modules
  (`openrlhf.cli.train_sft`, `train_dpo`, `train_ppo_ray`), each with a large namespaced
  argument surface. Reimplementing the training loop would guarantee drift, so this script
  only assembles the argument list, prints it, and execs it. `--dry_run` makes the whole
  thing auditable, and you can copy the printed command into a shell or SLURM script.
- **Two different argument namespaces.** This trips people up: the DeepSpeed path (SFT/DPO)
  uses `--model.model_name_or_path` and a single `--adam.lr`, while the Ray path (PPO/GRPO)
  uses `--actor.model_name_or_path` with per-role optimizers (`--actor.adam.lr`,
  `--critic.adam.lr`) because there are several models in play. The launcher keeps them
  straight so you pass one `--model_name` / `--learning_rate` either way.
- **GRPO is PPO with a different advantage estimator.** There is no `train_grpo` module.
  GRPO is `train_ppo_ray` plus `--algo.advantage.estimator group_norm`, and the launcher
  omits the critic in that mode because the group of `n_samples_per_prompt` completions
  *is* the baseline. The same switch selects `reinforce`, `reinforce_baseline`, `rloo` and
  `dr_grpo` — see the `ESTIMATORS` dict in the script. Upstream recommends
  `reinforce_baseline` for reasoning/RLVR tasks because it is robust to reward scale.
- **Hybrid Engine (`--train.colocate_all`).** All models and vLLM engines share the same
  GPUs, with `--vllm.enable_sleep` / `--ds.enable_sleep` letting each side release memory
  while the other runs. It serializes generate and train, which makes it the *most stable*
  and strictly on-policy mode — every rollout uses current weights. The alternative,
  `--train.async_enable`, overlaps generation with training for throughput at the cost of
  off-policyness; this launcher does not enable it, deliberately.
- **Ray is the placement layer.** `--{actor,critic,ref,reward}.num_nodes` and
  `.num_gpus_per_node` tell Ray how many GPUs each role gets. With `colocate_all` on a single
  8-GPU node, every role is given all 8 and they time-share. Multi-node is the same flags
  with more nodes joined via `ray start --address`.
- **Agent-based execution.** Every rollout, even plain PPO, flows through OpenRLHF's
  token-in-token-out agent pipeline (single-turn by default). Multi-turn environments plug in
  via `--train.agent_func_path` with `reset()`/`step()` methods. Not wired up here.
- **Memory tuning.** On OOM: drop `--vllm_gpu_memory_utilization`, reduce
  `--micro_batch_size`, or turn off colocation. When you have headroom, upstream suggests
  disabling `--ds.adam_offload` and enabling `--ds.overlap_comm`.

### Upstream API uncertainty

Recorded honestly rather than guessed:

- **The CLI module names are verified** against the contents of `openrlhf/cli/` upstream:
  `train_sft.py`, `train_dpo.py`, `train_rm.py`, `train_ppo_ray.py`, `lora_combiner.py`,
  `serve_rm.py`. There is **no** `train_grpo` module, confirming GRPO is an estimator flag.
- **The README and `examples/scripts/` disagree on the SFT namespace.** The root README's SFT
  snippet uses `--actor.model_name_or_path` and `--actor.gradient_checkpointing_enable`,
  while `examples/scripts/train_sft.sh` uses `--model.model_name_or_path` and
  `--model.gradient_checkpointing_enable`. This launcher follows **`examples/scripts/`** for
  SFT/DPO, on the assumption that runnable scripts track the code more closely than prose.
  **If SFT or DPO fails with an unrecognized-argument error, this is the first thing to
  check** — swap `--model.` for `--actor.` in `build_deepspeed_cmd`.
- **The namespaced (dotted) argument style is recent.** Older OpenRLHF used flat flags
  (`--pretrain`, `--learning_rate`, `--prompt_data`). If you install an older release than
  0.10.x, essentially every flag here will be rejected. Check `python -m openrlhf.cli.train_sft --help`
  against your installed version before debugging anything else.
- **`--rollout.max_new_tokens` vs `--generate_max_len`.** The README's PPO block uses
  un-namespaced `--prompt_max_len` / `--generate_max_len`, while the newer example scripts
  use `--data.max_len` and `--rollout.max_new_tokens`. This launcher uses the latter. Same
  caveat as above.
- **`--reward.remote_url` is overloaded.** Upstream passes both a local filesystem path to a
  `reward_func.py` and an HTTP endpoint through the same flag. The launcher passes whatever
  you give `--reward_func` through verbatim; use an absolute path, since Ray workers resolve
  it relative to the shipped `working_dir`.
- **Not covered here:** reward-model training (`openrlhf.cli.train_rm`), multi-turn agents
  (`--train.agent_func_path`), async training (`--train.async_enable`), DAPO dynamic
  filtering, VLM RLHF, and SLURM multi-node launch. All are documented upstream; none are
  wired into this launcher.
