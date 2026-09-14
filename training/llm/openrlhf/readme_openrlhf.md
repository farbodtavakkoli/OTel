# Setup & usage — `train_llm_openrlhf.py`

## Overview & when to use

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) is an RLHF framework built on
**Ray + vLLM + DeepSpeed**: Ray places the actor / critic / reference / reward models and
the vLLM generation engines across GPUs, vLLM handles rollout generation, and DeepSpeed
ZeRO shards the training-side models. It implements **PPO, GRPO, REINFORCE++,
REINFORCE++-baseline, RLOO and Dr. GRPO**, plus **SFT**, **DPO** and reward-model training.

Files in this folder:
- `train_llm_openrlhf.py` — launcher; builds and execs the right `deepspeed` / `ray job submit` command.
- `prepare_data_openrlhf.py` — stdlib-only converter: chat JSONL → flat prompt/answer JSONL for the PPO/GRPO path.
- `requirements_openrlhf.txt` — dependencies, including the source-only and `--no-build-isolation` ones.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.
- `readme_openrlhf.md` — this document.
- `dev.env` — **you create this**; holds `HF_TOKEN`. Git-ignored, never committed.

> **Hardware coverage.** On AMD Instinct MI355X (gfx950, ROCm 7.2), SFT and DPO train in a
> plain ROCm venv at 1 and 8 GPUs, and PPO and GRPO train inside a container that already
> ships a ROCm build of vLLM (`rocm/verl:...vllm0.20.2`, 4 GPUs) — the pip route cannot
> reach the RL tiers on AMD. On NVIDIA (CUDA 13.0), SFT and GRPO both train in a plain venv
> with no container and no `--no-deps` surgery. The per-tier routes are under
> [Install](#install).
>
> The flags here target OpenRLHF 0.10.x–0.11.0 on a single node with 8xH100 80GB, using the
> Hybrid Engine (`--train.colocate_all`). Check them against your installed version — see
> "Upstream API uncertainty" in Notes.

## Install

Set these once per shell; the commands below reference them:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # training artifacts and run logs
```

### NVIDIA (CUDA)

Docker is the supported path — the Ray + vLLM + DeepSpeed + flash-attn pins are
version-sensitive:

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

Install the `[vllm]` extra rather than vLLM separately — it pins a matching vLLM version.

#### NVIDIA H100 (CUDA 13.0) — plain venv, no container

On NVIDIA every tier installs from PyPI: `pip install openrlhf[vllm]` resolves a working
vLLM into a plain venv, so SFT/DPO via DeepSpeed and the Ray/vLLM RL path both work without
Docker and without `--no-deps` surgery. Per-tier:

| Tier | Launcher route | Status on H100 (plain venv) | Note |
|---|---|---|---|
| **SFT** | `deepspeed --module openrlhf.cli.train_sft` | **works** | needs the flash-attn fix below (or `sdpa`) |
| **DPO** | `deepspeed --module openrlhf.cli.train_dpo` | **works** | preference-triple data, same install |
| **PPO / GRPO** (+ RLOO / REINFORCE++ / Dr. GRPO) | `train_ppo_ray` (Ray + vLLM) | **works — vLLM 0.27.1 installs natively** | |

##### Install

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf && source .env_openrlhf/bin/activate

# 1. Base torch FIRST. Resolves the native CUDA 13 build.
pip install torch numpy
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130

# 2. OpenRLHF + the pinned vLLM. On NVIDIA this is the whole install — no --no-deps needed.
#    Pulls: openrlhf 0.11.0, vllm 0.27.1, flash-attn 2.8.3 (wheel), deepspeed 0.19.5,
#    transformers 5.15.0, ray 2.55.0, accelerate 1.14.0, datasets 5.0.1.
pip install 'openrlhf[vllm]'
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130
pip check   # -> "No broken requirements found."
```

##### The prebuilt flash-attn wheel is ABI-broken against torch 2.13.0

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
before any GPU code runs. Two fixes, either works:

- **Rebuild flash-attn from source against the installed torch** (recommended — it also
  enables `--ds.attn_implementation flash_attention_2`):
  ```bash
  pip uninstall -y flash-attn
  MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
  ```
  Needs `nvcc` on `PATH` (system CUDA 13.0 at `/usr/local/cuda`, matches torch's cu130).
  Afterwards both trainers import with no shim and the launcher's defaults
  (`--ds.packing_samples` + `--ds.attn_implementation flash_attention_2`) work as-is.
- **Or, to get SFT/DPO going immediately without the build, shim the eager import.**
  `flash_attn.bert_padding` and `flash_attn.utils.distributed` are pure torch/einops (no
  kernels) — only `flash_attn/__init__.py`'s eager import of `flash_attn_interface` (the
  broken `.so`) is the problem. Wrap it in `try/except ImportError: pass`, then run with
  `--ds.attn_implementation sdpa`.

##### DeepSpeed device-pinning trap

`deepspeed --num_gpus 1` **ignores `CUDA_VISIBLE_DEVICES` and forces
`CUDA_VISIBLE_DEVICES=0`** — it prints `Detected VISIBLE_DEVICES=<n> but ignoring it because
--num_gpus was used ... Setting CUDA_VISIBLE_DEVICES=0`, which silently lands your job on
GPU 0 no matter which card you selected, and the run still exits 0. To target a specific
GPU, pin by index instead:

```bash
unset CUDA_VISIBLE_DEVICES        # let DeepSpeed do the pinning
deepspeed --include localhost:6 --master_port 29646 --module openrlhf.cli.train_sft ...
# -> [launch.py] WORLD INFO DICT: {'localhost': [6]} ; Setting CUDA_VISIBLE_DEVICES=6
```

##### SFT smoke command

`data/OTel_LLM_sample_10.jsonl` is 10 rows; replicate it ×8 into an 80-row `sft_80.jsonl`
so the run takes a few optimizer steps. The 10-row file also trains and checkpoints.

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

A healthy SFT run shows `gpt_loss` and `grad_norm` both falling monotonically and ends with
`[launch.py] Process <pid> exits successfully`. `--ckpt.save_steps -1` still writes an HF
checkpoint at fit-end: `sft-ckpt/model.safetensors` plus `config.json`, `tokenizer.json`
and `chat_template.jinja`.

##### RL tiers (PPO / GRPO) on NVIDIA

`openrlhf.cli.train_ppo_ray` (shared by PPO, GRPO, RLOO, REINFORCE++ and Dr. GRPO) needs
only vLLM, which installs natively here. Start the Ray head first, then run the module
against the cluster. `--train.colocate_all` requires
`vllm.num_engines × vllm.tensor_parallel_size == actor GPUs`.

##### Single-GPU GRPO command

Pin the Ray head to the target GPU with `CUDA_VISIBLE_DEVICES` so Ray reports `1.0 GPU`;
`RAY_EXPERIMENTAL_NOSET_*` is **not** needed on CUDA (it is the ROCm-side equivalent). Then
run `train_ppo_ray` directly against the cluster:

```bash
unset PYTHONPATH   # only needed once real flash_attn is built; without it, a pure-python
                   # flash_attn shim on PYTHONPATH lets train_ppo_ray import and run sdpa
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

`data/otel_rl.jsonl` (prompt/answer) comes from `prepare_data_openrlhf.py`;
`reward_func.py` is the stub under [Data](#data).

A healthy run logs vLLM engine init and KV-cache allocation, then per-step
`update weight: model.layers.N...` lines (the actor→vLLM NCCL weight sync), a
`CuMemAllocator: sleep freed ...` line (colocate_all sleep/wake), one
`Global step N: {...}` line per optimizer step with finite `policy_loss`,
`actor_grad_norm`, `ppo_kl` and `group_reward_std`, and finally `Writing model shards` for
the HF checkpoint. With the rule-based stub reward and short completions, `reward` and
`accuracy` legitimately read `0.0` on a smoke.

##### Scaling past one GPU

A 2- or 8-GPU pass sets `--include localhost:0,1,...` (SFT/DPO) or
`--actor.num_gpus_per_node N` with `--vllm.num_engines × --vllm.tensor_parallel_size`
matching the actor GPU count (PPO/GRPO).

### AMD / ROCm

**OpenRLHF is NVIDIA-first — there is no upstream ROCm support.** It still runs on MI355X
via the routes below. If AMD hardware is a requirement, `../verl/` has first-party ROCm
support and is less work.

#### AMD MI355X (ROCm 7.2)

On MI355X (gfx950), the outcome differs per tier, so read the table:

| Tier | Launcher route | Status on MI355X | Root cause |
|---|---|---|---|
| **SFT** | `deepspeed --module openrlhf.cli.train_sft` | **works** in a plain ROCm venv | needs only torch + DeepSpeed, both ROCm-capable |
| **DPO** | `deepspeed --module openrlhf.cli.train_dpo` | **works** in a plain ROCm venv | same |
| **PPO** | `train_ppo_ray` (Ray) | **blocked in a venv** / **works in the `rocm/verl` container** | the venv has no ROCm `vllm`; the container ships one — see [RL tiers on MI355X](#rl-tiers-on-mi355x-via-the-rocmverl-container) |
| **GRPO** (and RLOO / REINFORCE++ / Dr. GRPO) | `train_ppo_ray` (Ray) | **blocked in a venv** / **works in the `rocm/verl` container** | same — all RL modes share `train_ppo_ray` |
| `--ds.packing_samples` | any tier | **blocked in a venv** / **works in the container** | forces `flash_attention_2`, which the container already ships (ROCm flash-attn 2.8.4) |
| `--ds.ring_attn_size > 1` | any tier | **blocked** | needs real `ring_flash_attn` kernels |

In short: the pip/venv route on AMD covers SFT and DPO only; the RL tiers need a container
that already carries a ROCm build of vLLM. See
[RL tiers on MI355X](#rl-tiers-on-mi355x-via-the-rocmverl-container). SFT and DPO also run
on 8x MI355X with one batch-geometry change — see
[8-GPU run (8x MI355X, ROCm 7.2.4)](#8-gpu-run-8x-mi355x-rocm-724).

##### Blocker 1 — `flash-attn` is an unconditional dependency

`flash-attn==2.8.3` is in OpenRLHF's `install_requires`, **not** in an extra, so every
`pip install openrlhf` triggers a source build of it (multi-hour `hipcc` on ROCm, no
prebuilt wheel):

```
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
  ModuleNotFoundError: No module named 'torch'
```

Install with `--no-deps` instead (see [ROCm venv install](#rocm-venv-install-sft--dpo)).
`openrlhf/models/actor.py` imports `ring_attn_utils.py` at top level, so *every* trainer
including SFT dies on `ModuleNotFoundError: No module named 'flash_attn'` before reaching
any GPU code — hence the pure-python shim in that install.

##### Blocker 2 — the vLLM **PyPI wheel** is CUDA-only, and would delete your ROCm torch

> **Scope:** this blocker is about *PyPI wheels only*, not about AMD hardware. A ROCm build
> of vLLM exists and runs fine on MI355X — it just does not come from `pip install vllm`.
> Ship OpenRLHF into a container that already has one and the RL tiers train; see
> [RL tiers on MI355X](#rl-tiers-on-mi355x-via-the-rocmverl-container).

`vllm==0.27.1` (the version the `[vllm]` extra pins) declares `torch==2.13.0` — the plain
PyPI CUDA build. `pip install 'vllm==0.27.1'` on a ROCm host therefore **silently replaces
your working ROCm torch with a CUDA one**, leaving an environment that imports but cannot
see the GPUs. Get ROCm vLLM from a source build (`PYTORCH_ROCM_ARCH=gfx950`) or from AMD's
images (`rocm/vllm`, `rocm/verl`). No OpenRLHF change is needed:
`--vllm.sync_backend nccl` resolves to RCCL on ROCm.

##### Blocker 3 — `--ds.packing_samples` silently re-enables flash-attn

`--ds.attn_implementation sdpa` is **not respected** while packing is on — every trainer
overrides it back to `flash_attention_2`, which surfaces as:

```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the
following error: the package for FlashAttention2 doesn't seem to be installed.
```

`train_llm_openrlhf.py` sets `--packing_samples` **and** `--flash_attn` on by default and
neither can be turned off from the command line (both are `action="store_true",
default=True`). In a ROCm venv, edit those two defaults in the script or bypass the
launcher and call `deepspeed --module` yourself as below. Inside the container route the
blocker disappears — the image ships ROCm flash-attn 2.8.4, so both defaults can stay on.

#### RL tiers on MI355X via the `rocm/verl` container

##### Route

Use the `rocm/verl` image as a ROCm runtime — ROCm torch + ROCm vLLM + ROCm flash-attn +
Ray, all prebuilt — and drop OpenRLHF into it with `--no-deps`. `rocm/vllm` works the same
way.

```bash
# 1. Container. GPUs are pinned by render node, NOT by env var — one
#    --device /dev/dri/renderD<N> per GPU (see `ls /dev/dri`); --group-add takes the host's
#    video and render group IDs.
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

The image supplies ROCm torch, ROCm vLLM and `flash_attn` 2.8.4 — **never pip-install
those**, it is what `--no-deps` protects.

##### GRPO command (4x MI355X)

`--reward_func` needs a file the Ray workers can see; write it to the mounted `/outputs`
rather than into this folder. Same signature the readme documents above.

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

This is what `train_llm_openrlhf.py --mode grpo ... --dry_run` prints, minus the
`ray job submit` wrapper (see quirk 2).

##### What a healthy run looks like

Expect one `GPU KV cache size: ...` line per vLLM engine, then per-step
`update weight: model.layers.N...` lines (the actor→engine sync) and one
`Global step N: {...}` line per optimizer step with finite `reward`, `policy_loss`,
`actor_grad_norm` and `group_reward_std > 0`. `--ckpt.save_hf` then writes
`<output_dir>/model.safetensors` plus `config.json`, `tokenizer.json` and
`chat_template.jinja`.

##### PPO — the critic path

Same container, same data, plus a critic (`--critic.num_nodes 1
--critic.num_gpus_per_node 4 --critic.adam.lr 9e-6`, and **no**
`--algo.advantage.estimator`). The step log then also carries `critic_loss`, `values` and
`critic_grad_norm`.

##### Quirks on this route

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
2. **`ray job submit` may never register the job** — the submit client hangs after
   `Uploading package gcs://_ray_pkg_*.zip` with `ray job list` staying `[]`. Run the module
   directly against the cluster instead (`RAY_ADDRESS=127.0.0.1:6379 python3 -m
   openrlhf.cli.train_ppo_ray ...`); it is the identical code path. This is a Ray
   job-server issue, not an OpenRLHF or ROCm one.
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

#### ROCm venv install (SFT / DPO)

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf          # git-ignored via .env_*/
source .env_openrlhf/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # never leave these empty on ROCm

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
# e.g. 2.11.0+rocm7.2, device gfx950:sramecc+:xnack-
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

A healthy run ends with a falling `gpt_loss` and `Writing model shards: 100%`.

DPO runs on the same env. Feed preference triples as full chat lists in `chosen`/`rejected`
with **no** `--data.prompt_key` — passing a chat-list `prompt` alongside a string `chosen`
makes `datasets` fail feature alignment. A healthy DPO run shows `acc` climbing and the
`chosen_reward` / `reject_reward` margin opening.

Ray works in this venv; only the missing ROCm vLLM blocks the RL tiers (supply one via a
container, see [RL tiers on MI355X](#rl-tiers-on-mi355x-via-the-rocmverl-container)). Set
`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` (the ROCm analogue of
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`) for correct per-worker device binding.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

SFT and DPO run on all 8 MI355X cards from the same venv and the same `--no-deps` install.
PPO/GRPO stay blocked in a venv at any GPU count (no ROCm `vllm`) — use the container route.

##### 8-GPU batch geometry tightens with world size

DeepSpeed requires

```
train_batch_size == micro_train_batch_size * gradient_accumulation_steps * world_size
```

and OpenRLHF derives the accumulation count by integer division, so the 1-GPU settings
(`--train.batch_size 2 --train.micro_batch_size 1`) yield `2 // 1 // 8 == 0` accumulation
steps at `world_size=8` and DeepSpeed rejects the config before any GPU work happens.
`--train.batch_size` is the **global** batch; `--train.micro_batch_size` is **per-GPU**.

A working geometry at 8 ranks: **global 64 = micro 2 × grad-accum 4 × world 8.**

##### Launch commands

Export `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
*after* sourcing the venv, and assert `torch.cuda.device_count() == 8` first. (Check
`.env_openrlhf/bin/activate` for a stale `VISIBLE_DEVICES` pin — grep the whole file before
trusting it.)

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
one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used`. That
warning is benign — the launcher re-sets the full device list itself.

##### What a healthy run looks like

The progress bar counts **micro-steps per rank** (samples ÷ ranks ÷ micro batch), so at 640
samples / 8 ranks / micro 2 it shows 40 per epoch. SFT is healthy when `gpt_loss` falls
steadily; DPO when `acc` climbs and the `chosen_reward` / `reject_reward` margin opens. A
`destroy_process_group() was not called before program exit` warning per rank at teardown
is benign.

##### Other differences from the 1–2 GPU setup

1. **The dataset must be large enough.** `data/OTel_LLM_sample_10.jsonl` is 10 rows — at a
   global batch of 64 that is not even one optimizer step. Replicate it ×64 into
   `sft_640.jsonl` and the 10-row preference file ×72 into `dpo_648.jsonl`, written
   **outside the repo**.
2. **Check sample survival** — OpenRLHF silently drops rows longer than the limit (it skips
   when `prompt_ids_len >= max_length - 2`). At `--data.max_len 4096` all 640 SFT rows
   survive; micro-steps × ranks × micro batch must equal the row count.
3. **`--master_port` must differ per tier** when running tiers back-to-back (e.g. 29730 /
   29731); reusing a port immediately after a teardown can hit a lingering bind.

## Environment & secrets

Put a `dev.env` **in this folder** containing your Hugging Face token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

The launcher calls `load_dotenv('dev.env')` and reads `HF_TOKEN` from the environment,
warning (not failing) when it is absent. `dev.env` is **git-ignored** — never commit
tokens. Weights-and-Biases keys go the same way: read `WANDB_API_KEY` from `dev.env` and
append `--logger.wandb.key "$WANDB_API_KEY"`.

If Ray workers need the token too, export it before submitting, or add it to the runtime env:
`--runtime-env-json '{"working_dir": ".", "env_vars": {"HF_TOKEN": "..."}}'` — note that this
puts the token in your shell history, so prefer exporting it on the nodes.

## Data

This folder ships a 10-row sample, `data/OTel_LLM_sample_10.jsonl` — telecom-spec chat
records in the repo's usual format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

OpenRLHF reads HF dataset ids **or** local JSONL, and you tell it which JSON field plays
which role via `--data.*_key` flags (`--input_key`, `--output_key`, `--label_key` on the
launcher).

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
`reward_func` with this signature:

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

`ray job submit` fails immediately if no head node is listening.

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
`--max_new_tokens`. For SFT/DPO you get ordinary DeepSpeed loss lines instead.

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

## Hardware support

| Hardware | Status | Detail |
|---|---|---|
| NVIDIA (A100/H100 class) | Primary, only upstream-documented target | Every tier installs from PyPI into a plain venv |
| AMD / ROCm | **Not supported upstream** | No ROCm docs, Dockerfile, or install path upstream |
| AMD MI355X (gfx950, ROCm 7.2) — SFT / DPO | **works**, off the supported path | Needs `openrlhf --no-deps`, a pure-python `flash_attn` shim, `--ds.attn_implementation sdpa`, and no `--ds.packing_samples`. See [AMD MI355X (ROCm 7.2)](#amd-mi355x-rocm-72) |
| AMD MI355X (gfx950, ROCm 7.2) — PPO / GRPO | **blocked in a plain venv; works in the `rocm/verl` container** | The venv has no ROCm `vllm`; the image ships one — see [RL tiers on MI355X](#rl-tiers-on-mi355x-via-the-rocmverl-container) |

For AMD **RL**, `../verl/` is less work; for AMD **SFT/DPO**, `../deepspeed/` or
`../fsdp/`.

## Notes

- **Two argument namespaces.** The DeepSpeed path (SFT/DPO) uses
  `--model.model_name_or_path` and a single `--adam.lr`; the Ray path (PPO/GRPO) uses
  `--actor.model_name_or_path` with per-role optimizers (`--actor.adam.lr`,
  `--critic.adam.lr`). The launcher keeps them straight, so you pass one `--model_name` /
  `--learning_rate` either way.
- **There is no `train_grpo` module.** GRPO is `train_ppo_ray` plus
  `--algo.advantage.estimator group_norm` (and no critic). The same switch selects
  `reinforce`, `reinforce_baseline`, `rloo` and `dr_grpo` — see the `ESTIMATORS` dict in the
  script.
- **Multi-turn agents** plug in via `--train.agent_func_path` with `reset()`/`step()`
  methods. Not wired up here.
- **Memory tuning.** On OOM: drop `--vllm_gpu_memory_utilization`, reduce
  `--micro_batch_size`, or turn off colocation. With headroom, disable `--ds.adam_offload`
  and enable `--ds.overlap_comm`.

### Upstream API uncertainty

- **If SFT or DPO fails with an unrecognized-argument error**, swap `--model.` for
  `--actor.` in `build_deepspeed_cmd` — upstream's README and `examples/scripts/` disagree
  on that namespace, and this launcher follows `examples/scripts/`.
- **The namespaced (dotted) argument style is recent.** On a release older than 0.10.x
  essentially every flag here is rejected (older OpenRLHF used `--pretrain`,
  `--learning_rate`, `--prompt_data`). Check
  `python -m openrlhf.cli.train_sft --help` against your installed version before debugging
  anything else.
- **`--reward.remote_url` takes either a `reward_func.py` path or an HTTP endpoint.** The
  launcher passes `--reward_func` through verbatim; use an absolute path, since Ray workers
  resolve it relative to the shipped `working_dir`.
