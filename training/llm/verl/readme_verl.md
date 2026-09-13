# Setup & usage — `train_llm_verl.py`

## Overview & when to use

RL post-training with [verl](https://github.com/volcengine/verl) (Volcano Engine RL), the
open-source implementation of the HybridFlow RLHF architecture. This is the folder for
**online RL** — GRPO, PPO, DAPO, RLOO, REINFORCE++, GSPO — where the model generates
rollouts, a reward function scores them, and the policy is updated from that signal.

Pick this over the other trainers in this repo when you need *real* RL: `training/llm/unsloth`
does single-node GRPO for small models, and `training/llm/deepspeed` covers SFT/DPO, but
neither scales rollout generation. verl separates the **training engine** (FSDP2 or
Megatron) from the **rollout engine** (vLLM or SGLang) and schedules both with Ray. It also
has first-party AMD ROCm support, though it is distributed as a ~42 GB container rather
than as a pip install.

Files in this folder:
- `train_llm_verl.py` — launcher; turns a few flags into the Hydra override list for `verl.trainer.main_ppo`.
- `prepare_data_verl.py` — converts this repo's chat JSONL into the parquet schema verl requires.
- `reward_verl.py` — rule-based reward functions (exact-match + a format-aware variant).
- `requirements_verl.txt` — dependency notes; the container path is strongly preferred.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.

> **Coverage:** GRPO steps end-to-end on **AMD MI355X** (gfx950, ROCm 7.2.4 host) in the
> first-party `rocm/verl` container, and on **NVIDIA H100 80GB** (CUDA 13.0) in the
> `verlai/verl:uv.cu130` container — both with the **vLLM** rollout engine and the FSDP2
> trainer. This folder's scripts run unmodified on both; on CUDA two Hydra overrides the
> launcher does not expose are needed (see [Platform notes](#platform-notes--quirks)).
> Multi-GPU on NVIDIA and the `sglang` rollout engine are untested. verl's config surface
> moves fast — run with `--dry-run` first.
>
> **`--rollout-backend hf` no longer exists upstream** (removed in verl 0.6.0). Use `vllm`
> (verified) or `sglang`.

## Install

**Use the container.** verl pins a training engine, an inference engine, and Ray against
each other; a hand-built venv is the single most common source of "it imports but rollouts
hang" failures.

The container commands below use two placeholders — set them once to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # run artifacts and logs (mounted into the container)
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache (mounted into the container)
```

### NVIDIA (CUDA)

Use a current tag: verl retagged its images and the older
`verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0` name **404s on Docker Hub**. The verified
CUDA-13 image is `verlai/verl:uv.cu130` (12.8 GB over the wire, **43.3 GB on disk**). Query
live tags with:

```bash
curl -s "https://hub.docker.com/v2/repositories/verlai/verl/tags/?page_size=100&ordering=last_updated" \
  | python3 -c "import sys,json;[print(t['name']) for t in json.load(sys.stdin)['results']]"
```

If a corporate proxy 403s Docker Hub, unset the proxy variables before pulling:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker pull verlai/verl:uv.cu130

docker run -d --name verl_h100 \
  --gpus '"device=4"' --ipc host --shm-size 16g \
  -e HF_HOME=/models -e HF_TOKEN="$HF_TOKEN" \
  -v "$HF_HOME":/models -v "$PWD":/work -v "$OUTPUT_DIR/verl":/out \
  verlai/verl:uv.cu130 sleep infinity

docker exec verl_h100 nvidia-smi -L      # confirm exactly the GPUs you intend
```

**The `uv.*` images ship NO ready `.venv` — you must sync one.** Unlike the old "app"
images (verl + vLLM pre-installed), they bake only the **uv package cache** for every
backend; `/workspace/verl/.venv` does not exist until you create it. The sync is fully
offline from the baked cache and takes seconds:

```bash
docker exec verl_h100 bash -lc 'cd /workspace/verl && python3 manage_envs.py sync vllm -- --frozen'
```

`/workspace/verl/.venv/bin` is already on `PATH`, so afterwards `python3` resolves to it.
The vllm slice does **not** include flash-attn, which is why the
`attn_implementation=sdpa` override in [Platform notes](#platform-notes--quirks) is needed.

Check the current tag at <https://verl.readthedocs.io/en/latest/start/install.html> — the
image name encodes the vllm/megatron versions and is bumped often.

If you must build a venv:

```bash
python3.12 -m venv ~/.venv-verl && source ~/.venv-verl/bin/activate
git clone https://github.com/volcengine/verl.git && cd verl
pip install --no-deps -e .
pip install -r /path/to/this/folder/requirements_verl.txt
```

Install verl **from source, not the PyPI wheel** — the `examples/`, `recipe/` and default
Hydra config trees only exist in the git checkout, and the launcher's override names are
validated against those defaults.

Verify:
```bash
python -c "import verl, ray, vllm; print('verl OK')"
ray status
```

### AMD / ROCm

verl has **first-party AMD support** for MI300X / MI325X (`gfx942`) and MI350X / MI355X
(`gfx950`), with FSDP/FSDP2/Megatron trainers and vLLM/SGLang rollout, documented in
[`docs/amd_tutorial/amd_quick_start.rst`](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst).

The verified image is `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`
(**42.5 GB on disk**; the tag understates the contents — it ships verl **0.8.0.dev0**). Its
ROCm 7.0.2 userspace runs fine against a ROCm 7.2.4 kernel driver on the host. A source
build path exists via `docker/rocm/Dockerfile.rocm` in the verl tree.

`--group-add render` fails (`unable to find group render`) because the image has no
`render` group — pass **numeric GIDs**. Pin GPUs by exposing only their render nodes, never
by environment variable (see [Platform notes](#platform-notes--quirks)); GPU *i* is usually
`/dev/dri/renderD$((128+8*i))`, confirmed via `rocm-smi --showbus` and `/dev/dri/by-path`.

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

ROCm-specific caveats recorded upstream: SGLang requires `attention_backend=triton`, and
the ROCm image sets `vllm.disable_custom_all_reduce=True` by default because
`PYTORCH_ALLOC_CONF=expandable_segments:True` conflicts with vLLM custom all-reduce.
Follow the AMD tutorial rather than the CUDA instructions above.

**Why not a venv on ROCm.** A pip install reaches "imports and converts data" but not
"trains": **there is no ROCm vLLM wheel on PyPI**, and vLLM is the only rollout engine
modern verl can build (`--rollout-backend hf` was removed in 0.6.0). **There is no
lightweight, container-free way to run this folder on ROCm.**

### GPU layout — what verl does with N GPUs

verl's default hybrid engine **colocates** the trainer and the rollout engine on the same
GPUs; it does not split them into a "rollout half" and a "trainer half". Rollout data
parallelism is implicit: `n_gpus / rollout_tp` vLLM engines, each tensor-parallel, sharing
the same GPUs as the FSDP2 ranks, with weights resynced trainer → rollout every step. On a
real dataset raise `--train-batch-size` so each rank gets a full micro-batch, and raise
`--rollout-tp` only if the model no longer fits.

## Platform notes & quirks

### AMD / ROCm

1. **Never set both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.** vLLM's ROCm platform
   runs `_sync_hip_cuda_env_vars()` at import and **raises** if both are set and differ:
   `ValueError: Inconsistent GPU visibility env vars: HIP_VISIBLE_DEVICES='0' vs
   CUDA_VISIBLE_DEVICES='0,1'`. Ray and vLLM each rewrite *one* of the two per actor, so
   setting both is a guaranteed conflict. Set **neither** in the container and pin GPUs with
   per-device `--device /dev/dri/renderDxxx` instead.
2. **`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` is required.** Ray blanks the accelerator
   visibility var for `num_gpus=0` actors; verl's `TaskRunner` is one, and AITER's Triton
   kernels import there, giving `RuntimeError: 0 active drivers ([]). There should only be
   one.` Ray itself prints the fix as a `FutureWarning`.
3. **Ray-on-ROCm otherwise just works** — local cluster, dashboard, GPU actor placement and
   the vLLM HTTP server actors all come up unmodified.
4. **AITER JIT-compiles on first launch** (`[aiter] start build [module_rmsnorm]` ...),
   adding ~2-4 min of startup. `VLLM_ROCM_USE_AITER=1` is the image default;
   `VLLM_ROCM_USE_AITER_MOE=0` is a safe precaution because AITER MoE kernels are known to
   corrupt output on gfx950 (irrelevant for a dense model, harmless either way). If you swap
   in an MoE model and rollouts come out as nonsense, set `VLLM_ROCM_USE_AITER=0` and retry.
   `USE_ROCM_AITER_ROPE_BACKEND=0` disables the lower-precision AITER fused RoPE if you need
   bit-comparable results.
5. **Do not pip-install flash-attn.** The image already ships a ROCm build, and the sdpa/CK
   path is the ROCm attention path. `use_remove_padding=True` works as-is.

The full ROCm environment for a run:

```bash
docker exec -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  verl_mi355x bash -lc '...'
```

### NVIDIA / CUDA

1. **flash-attn is not in the `uv.cu130` image's vllm slice, and verl 0.9 + transformers 5.x
   demand it by default** (`ImportError: FlashAttention2 has been toggled on, but ... the
   package ... doesn't seem to be installed`). `use_remove_padding=False` is **not enough**;
   force **`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`**. Do not try
   to install flash-attn instead — there is no matching prebuilt wheel and the source build
   OOM-kills the container.
2. **verl 0.9 reward-API drift — `custom_reward_function.*` alone is silently ignored**, and
   the run fails late with `NotImplementedError: Reward function is not implemented for
   data_source='otel_local'`. Pass the reward on the **new** keys too, with an **absolute**
   path (Ray reward actors do not run in your CWD):
   `reward.custom_reward_function.path=/work/reward_verl.py
   reward.custom_reward_function.name=compute_score_format`. Keeping the legacy keys is
   harmless. verl 0.8 (the ROCm image) does not have this drift.
3. The ROCm quirks above do **not** apply: no `HIP_*`/`CUDA_VISIBLE_DEVICES` conflict (pin
   the container with `--gpus '"device=N"'`), no `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO`
   needed, no AITER JIT warmup.

Because the launcher hardcodes `use_remove_padding=True` and cannot set
`attn_implementation` or the verl-0.9 `reward.*` keys, a CUDA run may need to call
`python3 -m verl.trainer.main_ppo` directly with the Hydra overrides rather than going
through `train_llm_verl.py`.

### Both platforms

- **Reward collapse is easy to hit.** With the shipped `compute_score`, every rollout on the
  shipped telecom sample scores exactly `0.1` (the "non-empty completion" floor), so GRPO's
  group-relative advantage is identically 0 and `actor/grad_norm` stays `0.0` — the trainer
  steps, but the update is numerically a no-op. `compute_score_format` discriminates (some
  rollouts close `<think>`, some are truncated) and produces non-zero gradients. See
  [Reward function contract](#reward-function-contract).
- **verl writes a Hydra `outputs/` dir into the CWD**, root-owned when run in a container.
  Run from a mounted scratch dir (`-w /outputs`) to keep it out of this folder, and
  `rm -rf` it *from inside a container*, not the host. Because the CWD is then not this
  folder, `load_dotenv("dev.env")` finds nothing — pass `HF_TOKEN` via `docker exec -e` and
  give `--train-file` / `--val-file` / `--reward-file` as absolute paths.
- **Benign teardown noise:** `RuntimeError: DataLoader worker ... is killed by signal:
  Killed.` prints after training completes; the process still exits 0.
- **Verifying the GPUs are yours on a shared box:** sample `rocm-smi` / `nvidia-smi` from a
  loop running *for the lifetime of the training process*, not after it, and resolve every
  reported PID to its owning container via `/proc/<pid>/cgroup`. A sample taken seconds
  after your run exits can show someone else's job at 100%.

### Rollout backends

| Backend | Status |
|---|---|
| `vllm` | **Verified** on both platforms |
| `sglang` | **Not exercised** — shipped in the ROCm image; upstream says it needs `attention_backend=triton` |
| `hf` | **Removed upstream — cannot work.** `--rollout-backend hf` is no longer accepted |

## Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_verl.py` loads it with `load_dotenv("dev.env")`. `dev.env` is git-ignored at the
repo root — **never commit a token**, and rotate it on the Hub if one ever lands in a commit.

Optional:
```bash
export WANDB_API_KEY=...        # only if you pass --logger console,wandb
export VLLM_ATTENTION_BACKEND=XFORMERS   # workaround if FlashAttention misbehaves in rollout
```

When running in Docker, pass the token through explicitly (`--env HF_TOKEN=...`); the
container does not inherit your host `dev.env`.

## Data

This folder ships a 10-row sample, `data/OTel_LLM_sample_10.jsonl` — telecom-spec chat
records in the repo's usual format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each row also carries extra columns (`unmask`, `flow`, `source_id`, ...), which the
converter drops — but if you feed the raw JSONL to a different loader, note that some are
null in most rows and can be inferred with the wrong type by strict schema readers.

verl reads **parquet**, not JSONL. Convert (defaults point at the shipped sample):

```bash
python prepare_data_verl.py
# equivalent to:
# python prepare_data_verl.py --input data/OTel_LLM_sample_10.jsonl \
#   --output data/otel_train.parquet --split train
```

The converter splits each record at the final assistant turn — everything before it becomes
`prompt`, the assistant content becomes `reward_model.ground_truth`. Rows with no
`messages`, a non-assistant final turn, or an empty prompt/answer are skipped with a warning.

Output column contract:

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

verl loads the scorer via `custom_reward_function.path` / `custom_reward_function.name`
(the launcher's `--reward-file` / `--reward-fn`). The signature verl calls is:

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
```

- `data_source` — the row's `data_source` column; route different datasets to different scoring logic.
- `solution_str` — the model's generated completion, already decoded to text.
- `ground_truth` — the row's `reward_model.ground_truth` value.
- `extra_info` — the row's `extra_info` dict.

`reward_verl.py` ships two scorers:
- `compute_score` — exact match. 1.0 for a normalized exact match; 0.5 when the gold answer
  appears verbatim inside the completion; 0.1 for any non-empty completion (keeps a weak
  gradient toward answering); 0.0 for empty. It extracts the answer from the last
  `\boxed{...}` if present, else the last non-empty line — reasoning models emit a lot of
  preamble, and scoring the raw completion against gold almost always under-rewards.
- `compute_score_format` — `0.8 * correctness + 0.2` bonus when the completion contains a
  `<think>...</think>` block, capped at 1.0. Point `--reward-fn` at it when training a
  reasoning format rather than raw accuracy.

Keep rewards roughly in `[0, 1]`. Returning a constant makes GRPO's advantage collapse to
zero (all rollouts in a group score identically), so a run that produces flat reward curves
usually means the scorer is not discriminating.

## Run

Smoke test against the shipped sample (convert first, then dry-run):

```bash
python prepare_data_verl.py
python3 train_llm_verl.py --dry-run --algo grpo
```

The 10-row sample has no held-out split; the launcher defaults reuse it as `--val-file`
for a smoke eval. Full example:

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

**What "working" looks like:** Ray starts and prints a resource table, vLLM loads the policy
into the rollout engine, then each step logs reward, KL and timing:

```
step:5 pg_loss:-0.0033790 kl_loss:0.0008408 grad_norm:0.5375 entropy:0.39564 rewards/mean:0.30050 resp_len:820.5
step:5 ... critic/advantages/max:2.4748 critic/advantages/min:-2.4748
=== TRAINING EXIT rc=0 ===
```

Loss, KL and reward finite, `grad_norm` non-zero, advantages non-zero, reward spread rather
than collapsed. The signal to watch is **mean reward trending up while response length stays
sane**. Flat reward from step 1 almost always means the reward function is not
discriminating between rollouts — check `reward_verl.py` against a few real completions
before blaming the RL.

### Batch divisibility as the GPU count changes

verl asserts that **`--train-batch-size × --rollout-n` is divisible by the world size**, in
`_validate_config`, before any GPU work happens. The default `--rollout-n 5` against a
10-row dataset gives `50`, which is not divisible by 8 and aborts immediately;
`--rollout-n 8` (→ `80`) is the smallest fix that keeps all 10 rows. This constraint
tightens as you add GPUs and is the single most likely thing to break when moving a working
2-GPU configuration to 8.

Other flags worth adjusting for a small sample: `--max-prompt-length` must exceed your
longest prompt (the shipped sample reaches 2189 tokens, so the default 1024 combined with
`data.filter_overlong_prompts=True` silently drops half the rows — with
`data.truncation=error` it aborts instead), and `--save-freq -1 --test-freq -1` turns off
checkpointing and eval for a smoke run.

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
| `--strategy` | `fsdp2` | Training engine: `fsdp2` (recommended), `fsdp`, or `megatron` |
| `--rollout-backend` | `vllm` | Generation engine. `vllm` (**verified on MI355X**) or `sglang`. **`hf` is removed** — dropped from verl's rollout registry in 0.6.0; it asserts at worker init |
| `--rollout-tp` | `2` | Tensor-parallel size for the **rollout engine only** — independent of training sharding |
| `--train-batch-size` | `512` | Prompts per training step |
| `--mini-batch-size` | `256` | Prompts per policy update within a step |
| `--micro-batch-size-per-gpu` | `2` | Per-GPU forward/backward chunk — the first knob to lower on OOM |
| `--max-prompt-length` | `1024` | Prompt token cap (overlong prompts filtered) |
| `--max-response-length` | `1024` | Response token cap |
| `--rollout-n` | `5` | Rollouts per prompt. GRPO forms its baseline from this group, so it must be >= 2 (5-8 typical) |
| `--actor-lr` | `1e-6` | Policy LR. RL wants far lower LRs than SFT: `1e-6` is a normal starting point |
| `--kl-loss-coef` | `0.001` | KL penalty against the reference policy; raise it if the model drifts or collapses |
| `--entropy-coeff` | `0.0` | Entropy bonus coefficient |
| `--gpu-mem-util` | `0.6` | Fraction of HBM the rollout engine may reserve. Lower it when the actor OOMs |
| `--epochs` | `15` | Total training epochs |
| `--save-freq` | `20` | Checkpoint every N steps |
| `--test-freq` | `5` | Eval every N steps |
| `--project-name` | `verl_local` | Trainer project name (checkpoint path component) |
| `--experiment-name` | `grpo_run` | Experiment name (checkpoint path component) |
| `--logger` | `console` | Comma-separated: `console,wandb,tensorboard,mlflow` |
| `--dry-run` | off | Print the command without running it. Use this first, every time |

`prepare_data_verl.py`:

| Arg | Default | Meaning |
|---|---|---|
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Source chat JSONL |
| `--output` | `data/otel_train.parquet` | Destination parquet |
| `--split` | `train` | `train` or `test` — stored in `extra_info` |
| `--data-source` | `otel_local` | Tag your reward fn switches on |
| `--ability` | `general` | Free-form task tag |

## Output

Checkpoints land under `checkpoints/<project_name>/<experiment_name>/global_step_<n>/`,
written every `--save-freq` steps. They are sharded FSDP/Megatron checkpoints, **not**
directly loadable by `AutoModelForCausalLM` — convert with verl's model merger before
serving:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/verl_local/grpo_qwen3_4b/global_step_100/actor \
  --target_dir ./merged_hf_model
```

Verify the merger subcommand against your installed verl version (`python -m verl.model_merger --help`);
this entrypoint has been renamed at least once upstream. The merged directory is a standard
HF model you can serve with vLLM or push to the Hub.

## Hardware support

| Hardware | Status |
|---|---|
| NVIDIA (H100/A100 class) | Primary target — **verified**; use the `verlai/verl` images |
| AMD MI300X / MI325X (`gfx942`) | First-party supported |
| AMD MI350X / MI355X (`gfx950`) | **First-party supported — verified** in the `rocm/verl` image |
| Ascend NPU | Documented upstream; unverified |

## Notes

- **GRPO vs PPO.** PPO trains a separate critic to estimate the value baseline, which
  roughly doubles memory. GRPO drops the critic and instead samples `--rollout-n` completions
  per prompt, using the group's mean reward as the baseline. That is why `--rollout-n 1`
  is meaningless under GRPO — the advantage would be identically zero.
- **Two engines, one set of GPUs.** The actor (training) and the rollout engine (generation)
  both want HBM. `--gpu-mem-util` is how you split the budget; the most common OOM fix is
  lowering it, not lowering batch size.
- **Rollout TP is separate from training sharding.** `--rollout-tp` only sizes the vLLM/SGLang
  engine. You can shard training across 8 GPUs while generating with TP=2.
- **Reward functions are the actual work.** The RL machinery is generic; almost all run
  quality comes from `reward_verl.py`. Keep scores roughly in `[0, 1]` and make sure a
  better answer really does score higher — see the contract section above.
- **KL anchoring.** `--kl-loss-coef` keeps the policy near the frozen reference model.
  Too low and the model reward-hacks into gibberish; too high and it never moves.
- **Dynamic batching.** `use_dynamic_bsz=True` packs by token count instead of sequence
  count, which matters a lot when response lengths vary. The launcher enables it by default.
