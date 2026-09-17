# `training/llm/verl` — online RL post-training (GRPO / PPO) with verl

RL post-training with [verl](https://github.com/volcengine/verl) (Volcano Engine RL), the
open-source HybridFlow implementation. This is the folder for **online RL** — GRPO, PPO, DAPO,
RLOO, REINFORCE++, GSPO — where the model generates rollouts, a reward function scores them,
and the policy is updated from that signal. Pick it over `../unsloth` (single-node GRPO, small
models) or `../deepspeed` (SFT/DPO) when you need rollout generation to scale: verl separates
the training engine (FSDP2 or Megatron) from the rollout engine (vLLM or SGLang) and schedules
both with Ray.

**Hardware:** AMD MI355X / MI350X (gfx950) and MI300X / MI325X (gfx942) · NVIDIA H100 80GB
(CUDA 13.0), both with the vLLM rollout engine and the FSDP2 trainer. verl ships as a ~42 GB
container on both; the scripts in this folder run unmodified.

## Files

- `train_llm_verl.py` — launcher; turns a few flags into the Hydra override list for `verl.trainer.main_ppo`.
- `prepare_data_verl.py` — converts this repo's chat JSONL into the parquet schema verl requires.
- `reward_verl.py` — rule-based reward functions (exact-match and a format-aware variant).
- `data/OTel_LLM_sample_10.jsonl` — 10-row telecom-spec chat sample.
- `requirements_verl.txt` — dependency notes; the container path is strongly preferred.

## Setup

**Use the container.** verl pins a training engine, an inference engine, and Ray against each
other; a hand-built venv is the most common source of "it imports but rollouts hang". On ROCm
there is no container-free route at all — no ROCm vLLM wheel is published on PyPI, and modern
verl has no vLLM-free rollout engine.

```bash
export OUTPUT_DIR=/path/to/outputs     # run artifacts and logs (mounted into the container)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache (mounted into the container)
```

verl's config surface moves fast; check the current tag at
<https://verl.readthedocs.io/en/latest/start/install.html> and run with `--dry-run` first.

### NVIDIA (CUDA 13)

The CUDA-13 image is `verlai/verl:uv.cu130` (12.8 GB over the wire, 43.3 GB on disk). Older
`verlai/verl:app-verl0.5-*` tags 404 on Docker Hub; list live tags at
<https://hub.docker.com/r/verlai/verl/tags>.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # if a proxy 403s Docker Hub
docker pull verlai/verl:uv.cu130

docker run -d --name verl_h100 \
  --gpus '"device=4"' --ipc host --shm-size 16g \
  -e HF_HOME=/models -e HF_TOKEN="$HF_TOKEN" \
  -v "$HF_HOME":/models -v "$PWD":/work -v "$OUTPUT_DIR/verl":/out \
  verlai/verl:uv.cu130 sleep infinity

docker exec verl_h100 nvidia-smi -L      # confirm exactly the GPUs you intend
```

**The `uv.*` images ship no ready `.venv` — sync one.** They bake only the uv package cache;
`/workspace/verl/.venv` does not exist until you create it. The sync is offline and takes
seconds:

```bash
docker exec verl_h100 bash -lc 'cd /workspace/verl && python3 manage_envs.py sync vllm -- --frozen'
```

`/workspace/verl/.venv/bin` is already on `PATH`, so `python3` then resolves to it.

Two Hydra overrides the launcher does not expose are required on this image:

```
+actor_rollout_ref.model.override_config.attn_implementation=sdpa
reward.custom_reward_function.path=/work/reward_verl.py
reward.custom_reward_function.name=compute_score_format
```

The first is needed because the vllm slice ships no flash-attn while verl 0.9 + transformers
5.x default to it; do not try to install flash-attn instead, there is no matching prebuilt
wheel. The second is verl 0.9 reward-API drift: `custom_reward_function.*` alone is silently
ignored and the run fails late with
`NotImplementedError: Reward function is not implemented for data_source='otel_local'`. Use an
absolute path — Ray reward actors do not run in your CWD. Because the launcher hardcodes
`use_remove_padding=True` and cannot set either key, a CUDA run usually calls
`python3 -m verl.trainer.main_ppo` directly with the full override list.

### AMD / ROCm

verl has first-party AMD support documented in
[`docs/amd_tutorial/amd_quick_start.rst`](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst).
The image is `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` (42.5 GB on
disk; the tag understates the contents — it ships verl 0.8.0.dev0 with a ROCm/CK flash_attn
build). Its ROCm 7.0.2 userspace runs against a ROCm 7.2.4 host driver. A source build exists
via `docker/rocm/Dockerfile.rocm`.

Pin GPUs by exposing only their render nodes — GPU *i* is usually
`/dev/dri/renderD$((128+8*i))`, confirmed with `rocm-smi --showbus` and `/dev/dri/by-path`.
Pass numeric GIDs: `--group-add render` fails because the image has no `render` group.

```bash
docker run -d --name verl_mi355x \
  --device /dev/kfd \
  --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --device /dev/dri/renderD160 --device /dev/dri/renderD168 \
  --device /dev/dri/renderD176 --device /dev/dri/renderD184 \
  --group-add 44 --group-add 993 \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64G --network=host \
  -e HF_HOME=/hf_cache \
  -v "$PWD":/workspace/train_llm_verl \
  -v "$HF_HOME":/hf_cache \
  -v "$OUTPUT_DIR/verl":/outputs \
  -w /outputs \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

**Set neither `HIP_VISIBLE_DEVICES` nor `CUDA_VISIBLE_DEVICES` in the container.** vLLM's ROCm
platform raises `ValueError: Inconsistent GPU visibility env vars` if both are set and differ,
and Ray and vLLM each rewrite one of the two per actor. The per-device flags above are the
pinning mechanism.

Run every ROCm command with:

```bash
docker exec -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  verl_mi355x bash -lc '...'
```

`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` is required: Ray blanks the accelerator visibility var
for `num_gpus=0` actors, verl's `TaskRunner` is one, and AITER's Triton kernels import there
(`RuntimeError: 0 active drivers ([])`). `VLLM_ROCM_USE_AITER_MOE=0` because AITER MoE kernels
corrupt output on gfx950 — harmless for a dense model. AITER JIT-compiles on first launch,
adding 2-4 minutes of startup. Do not pip-install flash-attn; the image ships a ROCm build.

If you use the SGLang rollout engine on ROCm, upstream requires `attention_backend=triton`.

### Secrets

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded by train_llm_verl.py
```

Containers do not inherit your host `dev.env` — pass the token with `docker exec -e HF_TOKEN=...`.
Optionally `export WANDB_API_KEY=...` when you pass `--logger console,wandb`. Never commit a
token; rotate it on the Hub if one ever lands in a commit.

## Data

`data/OTel_LLM_sample_10.jsonl` — telecom-spec chat records in the repo's usual format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

verl reads **parquet**, not JSONL. Convert (defaults point at the shipped sample):

```bash
python prepare_data_verl.py
# equivalent to:
# python prepare_data_verl.py --input data/OTel_LLM_sample_10.jsonl \
#   --output data/otel_train.parquet --split train
```

The converter splits each record at the final assistant turn: everything before it becomes
`prompt`, the assistant content becomes `reward_model.ground_truth`. Rows with no `messages`,
a non-assistant final turn, or an empty prompt/answer are skipped with a warning.

| Column | Type | Meaning |
|---|---|---|
| `data_source` | str | Tag your reward function switches on |
| `prompt` | list | Chat messages **as a list** — verl applies the chat template itself, so do not pre-render to a string |
| `ability` | str | Free-form task tag |
| `reward_model` | dict | `{"style": "rule", "ground_truth": ...}` |
| `extra_info` | dict | `{"split": ..., "index": ...}` |

For open-ended tasks with no verifiable gold answer, keep the column but ignore it in the
reward function and score with a model-based reward instead.

### Reward function contract

verl loads the scorer via `custom_reward_function.path` / `custom_reward_function.name` (the
launcher's `--reward-file` / `--reward-fn`) and calls:

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
```

`solution_str` is the decoded completion; `data_source`, `ground_truth`, and `extra_info` come
from the row's columns. `reward_verl.py` ships `compute_score` (exact match: 1.0 normalized
match, 0.5 gold answer contained verbatim, 0.1 any non-empty completion, 0.0 empty, reading the
last `\boxed{...}` or last non-empty line) and `compute_score_format`
(`0.8 * correctness + 0.2` bonus for a `<think>...</think>` block, capped at 1.0).

**Use `compute_score_format` on the shipped sample.** With `compute_score` every rollout scores
exactly `0.1`, so GRPO's group-relative advantage is identically 0 and `actor/grad_norm` stays
`0.0` — the trainer steps but the update is a numerical no-op. Keep rewards roughly in
`[0, 1]`, and treat a flat reward curve as a non-discriminating scorer rather than an RL
problem.

## Run

```bash
python prepare_data_verl.py
python3 train_llm_verl.py --dry-run --algo grpo
```

The 10-row sample has no held-out split; the launcher reuses it as `--val-file` for a smoke
eval.

```bash
nohup python3 train_llm_verl.py \
  --algo grpo \
  --model-path Qwen/Qwen3-4B \
  --train-file data/otel_train.parquet \
  --val-file data/otel_test.parquet \
  --gpus-per-node 8 --strategy fsdp2 --rollout-backend vllm \
  --rollout-n 5 --rollout-tp 2 \
  --train-batch-size 512 --mini-batch-size 256 \
  --max-prompt-length 1024 --max-response-length 1024 \
  --actor-lr 1e-6 --epochs 15 \
  --experiment-name grpo_qwen3_4b \
  > train_llm_verl.log 2>&1 &

tail -f train_llm_verl.log
```

```
step:5 pg_loss:-0.0033790 kl_loss:0.0008408 grad_norm:0.5375 entropy:0.39564 rewards/mean:0.30050 resp_len:820.5
step:5 ... critic/advantages/max:2.4748 critic/advantages/min:-2.4748
=== TRAINING EXIT rc=0 ===
```

Watch for mean reward trending up while response length stays sane, with non-zero `grad_norm`
and non-zero advantages.

**`--train-batch-size x --rollout-n` must be divisible by the world size.** verl asserts this
in `_validate_config` before any GPU work. The default `--rollout-n 5` against a 10-row dataset
gives 50, which is not divisible by 8 and aborts immediately; `--rollout-n 8` (80) is the
smallest fix that keeps all 10 rows. This is the most likely thing to break when moving a
working 2-GPU configuration to 8.

For a small sample also raise `--max-prompt-length` above your longest prompt (the shipped
sample reaches 2189 tokens, and at the default 1024 `data.filter_overlong_prompts=True`
silently drops half the rows), and pass `--save-freq -1 --test-freq -1` to skip checkpointing
and eval.

verl's default hybrid engine **colocates** the trainer and the rollout engine on the same GPUs
rather than splitting them: `n_gpus / rollout_tp` vLLM engines share the GPUs with the FSDP2
ranks, with weights resynced every step. On a real dataset raise `--train-batch-size` so each
rank gets a full micro-batch, and raise `--rollout-tp` only if the model no longer fits.

## Arguments

`train_llm_verl.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--algo` | `grpo` | `grpo` (no critic, group-relative advantage) or `ppo` (trains a value network) |
| `--model-path` | `Qwen/Qwen3-4B` | Policy model — HF id or local path |
| `--train-file` | `data/otel_train.parquet` | Training parquet from `prepare_data_verl.py` |
| `--val-file` | `data/otel_train.parquet` | Held-out parquet; the sample is reused for smoke eval |
| `--reward-file` | `reward_verl.py` | Python file holding the reward function |
| `--reward-fn` | `compute_score` | Function name inside `--reward-file` |
| `--nnodes` | `1` | Ray nodes |
| `--gpus-per-node` | `8` | GPUs per node |
| `--strategy` | `fsdp2` | Training engine: `fsdp2`, `fsdp`, or `megatron` |
| `--rollout-backend` | `vllm` | Generation engine: `vllm` or `sglang`. `hf` is still in the launcher's choices but was dropped from verl's rollout registry in 0.6.0 and asserts at worker init |
| `--rollout-tp` | `2` | Tensor-parallel size for the rollout engine only — independent of training sharding |
| `--train-batch-size` | `512` | Prompts per training step |
| `--mini-batch-size` | `256` | Prompts per policy update within a step |
| `--micro-batch-size-per-gpu` | `2` | Per-GPU forward/backward chunk — first knob to lower on OOM |
| `--max-prompt-length` | `1024` | Prompt token cap (overlong prompts filtered) |
| `--max-response-length` | `1024` | Response token cap |
| `--rollout-n` | `5` | Rollouts per prompt; GRPO forms its baseline from this group, so >= 2 (5-8 typical) |
| `--actor-lr` | `1e-6` | Policy LR; RL wants far lower LRs than SFT |
| `--kl-loss-coef` | `0.001` | KL penalty against the reference policy; raise it if the model drifts |
| `--entropy-coeff` | `0.0` | Entropy bonus coefficient |
| `--gpu-mem-util` | `0.6` | Fraction of HBM the rollout engine may reserve; lower it when the actor OOMs |
| `--epochs` | `15` | Total training epochs |
| `--save-freq` | `20` | Checkpoint every N steps |
| `--test-freq` | `5` | Eval every N steps |
| `--project-name` | `verl_local` | Checkpoint path component |
| `--experiment-name` | `grpo_run` | Checkpoint path component |
| `--logger` | `console` | Comma-separated: `console,wandb,tensorboard,mlflow` |
| `--dry-run` | off | Print the command without running it |

`prepare_data_verl.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Source chat JSONL |
| `--output` | `data/otel_train.parquet` | Destination parquet |
| `--split` | `train` | `train` or `test` — stored in `extra_info` |
| `--data-source` | `otel_local` | Tag your reward fn switches on |
| `--ability` | `general` | Free-form task tag |

## Output

Checkpoints land under `checkpoints/<project_name>/<experiment_name>/global_step_<n>/`, every
`--save-freq` steps. They are sharded FSDP/Megatron checkpoints, **not** loadable by
`AutoModelForCausalLM` — merge them first:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/verl_local/grpo_qwen3_4b/global_step_100/actor \
  --target_dir ./merged_hf_model
```

Check the subcommand against your installed version (`python -m verl.model_merger --help`); it
has been renamed upstream at least once. The merged directory is a standard HF model you can
serve with vLLM or push to the Hub.

## Notes

- verl writes a Hydra `outputs/` dir into the CWD, root-owned when run in a container. Run from
  a mounted scratch dir (`-w /outputs`) and `rm -rf` it from inside a container, not the host.
  The CWD is then not this folder, so `load_dotenv("dev.env")` finds nothing — pass `HF_TOKEN`
  via `docker exec -e` and give `--train-file` / `--val-file` / `--reward-file` as absolute
  paths.
- `RuntimeError: DataLoader worker ... is killed by signal: Killed.` after training completes
  is benign; the process still exits 0.
- **GRPO vs PPO.** PPO trains a separate critic, roughly doubling memory. GRPO drops it and
  samples `--rollout-n` completions per prompt, using the group mean as the baseline — which is
  why `--rollout-n 1` is meaningless under GRPO.
- **Two engines, one set of GPUs.** `--gpu-mem-util` splits HBM between the actor and the
  rollout engine; lowering it is the most common OOM fix, ahead of lowering batch size.
- **KL anchoring.** Too low a `--kl-loss-coef` and the model reward-hacks into gibberish; too
  high and it never moves.
