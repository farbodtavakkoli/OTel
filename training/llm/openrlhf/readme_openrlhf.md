# `training/llm/openrlhf` — RLHF with Ray + vLLM + DeepSpeed

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) is an RLHF framework built on **Ray + vLLM +
DeepSpeed**: Ray places the actor / critic / reference / reward models and the vLLM generation
engines across GPUs, vLLM runs the rollouts, and DeepSpeed ZeRO shards the training-side models.
It implements **PPO, GRPO, REINFORCE++, REINFORCE++-baseline, RLOO and Dr. GRPO**, plus **SFT**,
**DPO** and reward-model training. Pick it when you want a full RL stack on NVIDIA; for AMD RL,
`../verl/` is first-party and less work.

The flags here target OpenRLHF 0.10.x-0.11.0 on a single node, using the Hybrid Engine
(`--train.colocate_all`). The namespaced (dotted) argument style is recent — on a release older
than 0.10.x essentially every flag here is rejected, so check
`python -m openrlhf.cli.train_sft --help` against your install before debugging anything else.

**Hardware:** NVIDIA H100 (CUDA 13.0, plain venv) · AMD MI355X (gfx950, ROCm 7.2.4). Support on
AMD differs per tier:

| Tier | Launcher route | NVIDIA venv | AMD MI355X |
|---|---|---|---|
| **SFT** | `deepspeed --module openrlhf.cli.train_sft` | works | works in a ROCm venv |
| **DPO** | `deepspeed --module openrlhf.cli.train_dpo` | works | works in a ROCm venv |
| **PPO / GRPO** (+ RLOO / REINFORCE++ / Dr. GRPO) | `train_ppo_ray` (Ray + vLLM) | works | needs a container with a ROCm vLLM |
| `--ds.ring_attn_size > 1` | any | needs the `ring` extra | unavailable (no ROCm `ring_flash_attn`) |

## Files

- `train_llm_openrlhf.py` — launcher; builds and execs the right `deepspeed` /
  `ray job submit` command.
- `prepare_data_openrlhf.py` — stdlib-only converter: chat JSONL -> flat prompt/answer JSONL for
  the PPO/GRPO path.
- `requirements_openrlhf.txt` — dependencies, including the source-only and
  `--no-build-isolation` ones.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample.

## Setup

One venv (or container) per recipe folder. Set these once per shell; the commands below use them:

```bash
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # training artifacts and run logs

cd training/llm/openrlhf
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded by train_llm_openrlhf.py
```

`dev.env` holds `HF_TOKEN=hf_...` and is git-ignored at the repo root. Never commit tokens. A
wandb key goes the same way — read `WANDB_API_KEY` from `dev.env` and append
`--logger.wandb.key "$WANDB_API_KEY"`. Ray workers do not inherit your shell automatically;
export `HF_TOKEN` on the nodes rather than passing it through `--runtime-env-json`, which puts it
in your shell history.

### NVIDIA (CUDA)

Every tier installs from PyPI into a plain venv — `pip install openrlhf[vllm]` resolves a working
vLLM, so no container and no `--no-deps` surgery is needed. Install the `[vllm]` extra rather
than vLLM separately; it pins a matching version.

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf && source .env_openrlhf/bin/activate

# 1. Base torch FIRST - resolves the native CUDA 13 build.
pip install torch numpy
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130

# 2. OpenRLHF + the pinned vLLM. Pulls openrlhf 0.11.0, vllm 0.27.1, flash-attn 2.8.3,
#    deepspeed 0.19.5, transformers 5.15.0, ray 2.55.0, accelerate 1.14.0, datasets 5.0.1.
pip install 'openrlhf[vllm]'
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # STILL 2.13.0+cu130
pip check                                                               # no broken requirements

# 3. Rebuild flash-attn from source: the prebuilt 2.8.3 wheel is ABI-broken against
#    torch 2.13.0, and openrlhf/models/actor.py imports flash_attn at module level, so
#    EVERY trainer (SFT included) fails to import until this is fixed. Needs nvcc on PATH.
pip uninstall -y flash-attn
MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTENTION_FORCE_BUILD=TRUE \
  pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
```

After the rebuild the launcher's defaults (`--ds.packing_samples` +
`--ds.attn_implementation flash_attention_2`) work as-is.

DeepSpeed JIT-compiles FusedAdam on the first run, so `ninja` (from the venv) and `nvcc` must be
on `PATH` with `CUDA_HOME` set:

```bash
export PATH=<venv>/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
```

Inside an NGC PyTorch container, remove the packages that collide with OpenRLHF's own versions
first:

```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:26.03-py3 bash
pip uninstall xgboost transformer_engine flash_attn pynvml -y
pip install openrlhf[vllm]
```

### AMD (ROCm) — venv, SFT and DPO

OpenRLHF has no upstream ROCm support. Two things must be worked around: `flash-attn==2.8.3` is
in `install_requires` (not an extra) and has no ROCm wheel, and `vllm==0.27.1` declares
`torch==2.13.0`, the CUDA build — installing it silently replaces your ROCm torch.

```bash
cd training/llm/openrlhf
python3 -m venv .env_openrlhf          # git-ignored via .env_*/
source .env_openrlhf/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0   # never leave these empty on ROCm

# 1. ROCm torch FIRST, so nothing else drags in a CUDA build.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2

# 2. OpenRLHF WITHOUT its dependency list - this is what skips flash-attn.
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

# 5. Verify the ROCm torch survived step 3 - this is where pip can quietly swap it.
python -c "import torch; print(torch.__version__)"   # must still end in +rocm7.2
```

Run with `--ds.attn_implementation sdpa` and **without** `--ds.packing_samples`: packing
overrides the attention choice back to `flash_attention_2`, which the shim cannot serve. Both are
`action="store_true", default=True` in `train_llm_openrlhf.py` and cannot be turned off from its
CLI — edit those two defaults, or call `deepspeed --module` directly as the Run section does.

Export `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` (the ROCm analogue of
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`) for correct per-worker device binding.

### AMD (ROCm) — container, PPO and GRPO

The RL tiers need a ROCm build of vLLM, which has no pip line. Use the `rocm/verl` image as a
ROCm runtime — ROCm torch + ROCm vLLM + ROCm flash-attn 2.8.4 + Ray, all prebuilt — and drop
OpenRLHF into it with `--no-deps`. `rocm/vllm` works the same way. No OpenRLHF change is needed:
`--vllm.sync_backend nccl` resolves to RCCL on ROCm, and with flash-attn present both launcher
defaults can stay on.

```bash
# 1. Container. GPUs are pinned by render node, NOT by env var - one
#    --device /dev/dri/renderD<N> per GPU (see `ls /dev/dri`); --group-add takes the host's
#    video and render group IDs. --ulimit nofile is load-bearing (see Notes).
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

# 2. OpenRLHF + the four things the image lacks. NOTHING ELSE - --no-deps is load-bearing;
#    a plain `pip install openrlhf` drags in flash-attn 2.8.3 (source build) and CUDA torch.
docker exec orlhf_mi355x pip install --no-deps \
  openrlhf==0.11.0 deepspeed==0.19.5 hjson py-cpuinfo nvidia-ml-py pynvml python-dotenv

# 3. Ray head.
docker exec orlhf_mi355x bash -lc \
  'ulimit -n 262144; ray start --head --num-gpus 4 --num-cpus 64 --port 6379 --dashboard-port 8265'
```

Never pip-install torch, vLLM or flash-attn inside this image — that is what `--no-deps` protects.

## Data

`data/OTel_LLM_sample_10.jsonl` — 10 telecom-spec chat records in the repo's usual format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

OpenRLHF reads HF dataset ids **or** local JSONL; `--data.*_key` flags tell it which JSON field
plays which role (`--input_key`, `--output_key`, `--label_key` on the launcher).

**SFT** — the shipped sample works as-is: `--input_key messages` with `--apply_chat_template`
(both launcher defaults).

**DPO** — preference triples, which the shipped sample does not contain:

```json
{"prompt": "Define entropy.", "chosen": "A measure of disorder.", "rejected": "I don't know."}
```

The launcher passes `--data.chosen_key chosen --data.rejected_key rejected`. If you supply chat
lists in `chosen`/`rejected`, do **not** also pass `--data.prompt_key` — a chat-list `prompt`
alongside a string `chosen` makes `datasets` fail feature alignment.

**PPO / GRPO** — needs a prompt field plus a ground-truth field for the reward function. Convert
the chat sample first; the last assistant turn becomes `answer`, the preceding turns `prompt`:

```bash
python prepare_data_openrlhf.py
# equivalent to:
# python prepare_data_openrlhf.py --input data/OTel_LLM_sample_10.jsonl --output data/otel_rl.jsonl
```

```json
{"prompt": [{"role": "user", "content": "..."}], "answer": "..."}
```

Launch RL with `--dataset data/otel_rl.jsonl --input_key prompt`; `--label_key` already defaults
to `answer`, and it is what makes that field arrive as `labels` in the reward function.

**Reward function** (`--reward_func /abs/path/reward_func.py`) — a module exposing `reward_func`:

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

Use an **absolute path on a filesystem every Ray worker can open** (a mounted `/outputs/...` in
the container route): the file is loaded inside the vLLM engine actor, not in the driver.
Alternatively serve a trained reward model with `--reward_model <repo-or-path>`, or point
`--reward.remote_url` at an HTTP endpoint.

The 10-row sample is too small for a multi-GPU batch. Replicate it (x64 -> `sft_640.jsonl`, x72
-> `dpo_648.jsonl`) **outside the repo**. OpenRLHF silently drops rows longer than the limit
(it skips when `prompt_ids_len >= max_length - 2`), so check that micro-steps x ranks x micro
batch equals your row count.

## Run

### Smoke test

`--dry_run` prints the exact command without executing anything — no GPU, cluster, or even an
OpenRLHF install needed:

```bash
python train_llm_openrlhf.py --mode sft --dry_run

python prepare_data_openrlhf.py
python train_llm_openrlhf.py --mode grpo --dataset data/otel_rl.jsonl \
  --input_key prompt --reward_func $PWD/reward_func.py --dry_run
```

### Start the Ray cluster (PPO/GRPO only)

`ray job submit` fails immediately if no head node is listening.

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8          # head node
ray start --address <MASTER-NODE-IP>:6379 --num-gpus 8           # each extra node
ray status                                                       # confirm every GPU is seen
```

The dashboard listens on `http://127.0.0.1:8265` — that is the default `--ray_address`. Export
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` if you hit DeepSpeed GPU-index errors.

### SFT / DPO (DeepSpeed, no Ray cluster)

```bash
nohup python train_llm_openrlhf.py \
  --mode sft --model_name meta-llama/Meta-Llama-3-8B \
  --dataset data/OTel_LLM_sample_10.jsonl --input_key messages \
  --output_dir ./checkpoint/llama3-8b-sft \
  > train_llm_openrlhf.log 2>&1 &

tail -f train_llm_openrlhf.log
```

`deepspeed --num_gpus N` **ignores `CUDA_VISIBLE_DEVICES`** and forces `CUDA_VISIBLE_DEVICES=0`,
so a job you meant for GPU 6 silently lands on GPU 0 and still exits 0. To target specific
devices, `unset CUDA_VISIBLE_DEVICES` and pin by index with `--include localhost:6`.

Calling the module directly is the way to override the launcher's always-on `--packing_samples` /
`--flash_attn` defaults — required on the ROCm venv route, which needs
`--ds.attn_implementation sdpa` and no packing. `--ckpt.save_steps -1` still writes an HF
checkpoint at fit-end. SFT is healthy when `gpt_loss` and `grad_norm` fall steadily; DPO when
`acc` climbs and the `chosen_reward` / `reject_reward` margin opens.

At 8 GPUs the batch geometry tightens: DeepSpeed requires
`train_batch_size == micro_train_batch_size * gradient_accumulation_steps * world_size`, and
OpenRLHF derives the accumulation count by integer division — so `--train.batch_size 2
--train.micro_batch_size 1` gives `2 // 1 // 8 == 0` steps and DeepSpeed rejects the config before
any GPU work. `--train.batch_size` is the **global** batch, `--train.micro_batch_size` is
**per-GPU**; a working 8-rank geometry is global 64 = micro 2 x grad-accum 4 x world 8:

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
```

DPO is the same command with `--module openrlhf.cli.train_dpo`, `--data.chosen_key chosen
--data.rejected_key rejected`, `--data.max_len 1024` and a different `--master_port` (reusing a
port straight after a teardown can hit a lingering bind). The progress bar counts **micro-steps
per rank** — 640 samples / 8 ranks / micro 2 = 40 per epoch.

### GRPO

`--train.colocate_all` requires `vllm.num_engines x vllm.tensor_parallel_size == actor GPUs`.

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
```

The equivalent direct invocation, which is what `--dry_run` prints minus the `ray job submit`
wrapper — use it when `ray job submit` hangs (see Notes). On ROCm this is the container route,
with flash-attn and packing left on:

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

On a single NVIDIA GPU the same shape works with `--actor.num_gpus_per_node 1`,
`--ref.num_gpus_per_node 1`, `--vllm.num_engines 1` and `--ds.attn_implementation sdpa`: pin the
Ray head with `CUDA_VISIBLE_DEVICES` so Ray reports `1.0 GPU` (`RAY_EXPERIMENTAL_NOSET_*` is the
ROCm-side equivalent and is not needed on CUDA), then run the module against the cluster.

A healthy RL run logs one `GPU KV cache size:` line per vLLM engine, `update weight:
model.layers.N...` per step (the actor -> vLLM weight sync), and one `Global step N: {...}` line
with finite `policy_loss`, `actor_grad_norm`, `ppo_kl` and `group_reward_std > 0`. Watch for a
rising mean `reward`, a bounded `kl`, and a `response_length` that neither collapses nor pins to
`--max_new_tokens`. With the rule-based stub reward and short completions, `reward` and `accuracy`
legitimately read `0.0` on a smoke run.

### PPO

PPO adds a critic and (usually) a served reward model. The launcher adds
`--critic.num_nodes/--critic.num_gpus_per_node/--critic.adam.lr` and omits
`--algo.advantage.estimator` automatically for `--mode ppo`; the step log then also carries
`critic_loss`, `values` and `critic_grad_norm`.

```bash
nohup python train_llm_openrlhf.py \
  --mode ppo \
  --model_name OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_model OpenRLHF/Llama-3-8b-rm-700k \
  --dataset OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --output_dir ./checkpoint/llama3-8b-ppo \
  > train_llm_openrlhf.log 2>&1 &
```

## Arguments

`train_llm_openrlhf.py`:

| Argument | Default | What it does |
|---|---|---|
| `--mode` | `grpo` | `sft`/`dpo` route to `deepspeed --module`; `ppo`/`grpo` to `ray job submit`. |
| `--model_name` | `Qwen/Qwen2.5-7B-Instruct` | Actor/policy model. |
| `--dataset` | `data/OTel_LLM_sample_10.jsonl` | HF dataset id or local JSONL. |
| `--output_dir` | `./checkpoint/openrlhf-run` | Checkpoint directory (`--ckpt.output_dir`). |
| `--num_gpus` | `8` | GPUs on this node, given to every Ray role. |
| `--input_key` | `messages` | SFT/PPO prompt field. Use `prompt` with the converted RL file. |
| `--output_key` | `None` | SFT target field; leave unset with chat templates. |
| `--label_key` | `answer` | PPO/GRPO ground-truth field; matches `data/otel_rl.jsonl`. |
| `--apply_chat_template` | on | Adds `--data.apply_chat_template`. |
| `--max_len` | `4096` | Max sequence length. |
| `--max_samples` | `100000` | Dataset row cap. |
| `--batch_size` / `--micro_batch_size` | `128` / `2` | Global and per-GPU training batch. |
| `--max_epochs` | `1` | Training epochs. |
| `--learning_rate` | `5e-6` | SFT/DPO: `--adam.lr`; RL: `--actor.adam.lr`. |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage for the training-side models. |
| `--packing_samples` | on | Adds `--ds.packing_samples`. Cannot be disabled from the CLI. |
| `--flash_attn` | on | Adds `--ds.attn_implementation flash_attention_2`. Cannot be disabled from the CLI. |
| `--reward_model` | `None` | PPO: served reward model (`--reward.model_name_or_path`). |
| `--reward_func` | `None` | Path to `reward_func.py` (`--reward.remote_url`). One of these two is required for RL. |
| `--critic_lr` | `9e-6` | PPO critic LR. |
| `--kl_coef` | `0.01` | `--algo.kl.init_coef`. `0` drops the reference model entirely. |
| `--rollout_batch_size` | `1024` | Prompts per generation phase. |
| `--n_samples_per_prompt` | `8` | Completions per prompt. **Must be >1 for GRPO** — the group is the baseline. |
| `--max_new_tokens` | `1024` | Rollout generation length. |
| `--vllm_num_engines` / `--vllm_tensor_parallel_size` | `4` / `2` | vLLM engines and TP width. |
| `--vllm_gpu_memory_utilization` | `0.5` | Lower it when the Hybrid Engine OOMs. |
| `--colocate_all` | on | Hybrid Engine; also adds `--vllm.enable_sleep --ds.enable_sleep`. |
| `--ray_address` | `http://127.0.0.1:8265` | Ray dashboard to submit the job to. |
| `--working_dir` | `.` | Shipped to Ray workers as the runtime env. See Notes. |
| `--dry_run` | off | Print the assembled command and exit. Use this first, every time. |

`prepare_data_openrlhf.py`:

| Argument | Default | What it does |
|---|---|---|
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Source chat JSONL. |
| `--output` | `data/otel_rl.jsonl` | Destination JSONL with `prompt` (chat list) + `answer` (string). |

## Output

- **Checkpoints** land in `--output_dir`. The launcher passes `--ckpt.save_hf` for RL runs, so
  the actor is written in HuggingFace format and loads with `AutoModelForCausalLM.from_pretrained`
  — no conversion step. Resume with `--ckpt.load_enable`; `--ckpt.save_steps` and `--ckpt.max_num`
  control retention.
- **Launcher stdout** goes to `train_llm_openrlhf.log`; for RL runs it carries the streamed Ray
  job output.
- **Ray logs** live under `/tmp/ray/session_latest/logs/` on each node. That is where the real
  traceback goes when a worker dies — check there first, since the submitting process usually only
  reports a generic job failure.
- **Metrics** go to wandb (`--logger.wandb.key`) or TensorBoard (`--logger.tensorboard_dir`);
  neither is enabled by default.
- **LoRA runs** save adapters only. Merge before serving:
  `python -m openrlhf.cli.lora_combiner --model_path <base> --lora_path <adapter> --output_path <merged> --ds.param_dtype bf16`.

## Notes

- **Raise the fd limit in every shell that starts Ray or submits to it:** `ulimit -n 262144` (or
  `--ulimit nofile=1048576:1048576` on `docker run`). With hundreds of CPUs visible the raylet
  prestarts enough workers to exhaust a 1024-fd soft limit and SIGABRTs the moment a driver
  connects. `--num-cpus 64` on `ray start` also cuts the prestart count.
- **Never point `--working_dir` at this folder.** Ray uploads the whole tree as the runtime env,
  including `.env_openrlhf/` (multi-GB ROCm/CUDA shared objects). Point it at a small directory
  holding just `data/`, or add an `excludes` list.
- **If `ray job submit` hangs** after `Uploading package gcs://_ray_pkg_*.zip` with `ray job list`
  still `[]`, run the module directly against the cluster instead
  (`RAY_ADDRESS=127.0.0.1:6379 python3 -m openrlhf.cli.train_ppo_ray ...`) — it is the identical
  code path.
- **Two argument namespaces.** The DeepSpeed path (SFT/DPO) uses `--model.model_name_or_path` and
  a single `--adam.lr`; the Ray path (PPO/GRPO) uses `--actor.model_name_or_path` with per-role
  optimizers. If SFT or DPO fails with an unrecognized-argument error, swap `--model.` for
  `--actor.` in `build_deepspeed_cmd` — upstream's README and `examples/scripts/` disagree, and
  this launcher follows `examples/scripts/`.
- **There is no `train_grpo` module.** GRPO is `train_ppo_ray` plus
  `--algo.advantage.estimator group_norm` and no critic. The same switch selects `reinforce`,
  `reinforce_baseline`, `rloo` and `dr_grpo` — see the `ESTIMATORS` dict in the launcher.
- **On OOM:** drop `--vllm_gpu_memory_utilization`, reduce `--micro_batch_size`, or turn off
  colocation. With headroom, disable `--ds.adam_offload` and enable `--ds.overlap_comm`.
- Harmless noise in the `rocm/verl` image: `quark ... ImportError: cannot import name
  'RoutedExperts'`, `NCCL WARN Could not read node #` from RCCL topology detection, and
  `Unknown vLLM environment variable detected: VLLM_USE_V1`.
- **Multi-turn agents** plug in via `--train.agent_func_path` with `reset()`/`step()` methods.
  Not wired up here.
