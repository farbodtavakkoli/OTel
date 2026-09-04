# Setup & usage — `train_llm_verl.py`

## Overview & when to use

RL post-training with [verl](https://github.com/volcengine/verl) (Volcano Engine RL), the
open-source implementation of the HybridFlow RLHF architecture. This is the folder for
**online RL** — GRPO, PPO, DAPO, RLOO, REINFORCE++, GSPO — where the model generates
rollouts, a reward function scores them, and the policy is updated from that signal.

Pick this over the other trainers in this repo when you need *real* RL: `training/llm/unsloth`
does single-node GRPO for small models, and `training/llm/deepspeed` covers SFT/DPO, but
neither scales rollout generation. verl separates the **training engine** (FSDP2 or
Megatron) from the **rollout engine** (vLLM or SGLang) and schedules both with Ray, which
is what makes multi-node RL practical. It also has first-party AMD ROCm support — **verified
here on MI355X** (see Hardware support below) — so it doubles as a hardware-portability path.
The catch: that portability arrives as a ~42 GB container, not as a pip install.

Files in this folder:
- `train_llm_verl.py` — launcher; turns a few flags into the Hydra override list for `verl.trainer.main_ppo`.
- `prepare_data_verl.py` — converts this repo's chat JSONL into the parquet schema verl requires.
- `reward_verl.py` — rule-based reward functions (exact-match + a format-aware variant).
- `requirements_verl.txt` — dependency notes; the container path is strongly preferred.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row telecom-spec chat sample for smoke tests.

> **Tested topology:** **AMD MI355X (gfx950) / ROCm 7.2.4 — TESTED, works.** GRPO stepped
> end-to-end on 2x MI355X — and then on **all 8x MI355X at the folder's default topology**
> (`--gpus-per-node 8 --rollout-tp 2`, Qwen3-4B) — inside the first-party `rocm/verl`
> container with the **vLLM** rollout engine; see
> [§ MI355X (ROCm 7.2) tested](#mi355x-rocm-72--tested-2026-08-19) for the
> exact route, commands, quirks and logs, and the 8-GPU subsection at its end.
> NVIDIA remains UNTESTED here — the flags were
> written against upstream docs and the `examples/grpo_trainer/run_qwen3_4b_fsdp.sh`
> reference script as of August 2026, targeting a single node with 8xH100 80GB. verl's
> config surface moves fast; run with `--dry-run` first and diff the printed overrides
> against the current upstream example before trusting them.
>
> **`--rollout-backend hf` no longer exists upstream** (removed in verl 0.6.0) — see the
> tested section. Use `vllm` (verified) or `sglang`.

## Install

**Use the container.** verl pins a training engine, an inference engine, and Ray against
each other; a hand-built venv is the single most common source of "it imports but rollouts
hang" failures.

### NVIDIA (CUDA)

```bash
docker pull verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0
docker run --gpus all -it --shm-size=32g \
  -v "$PWD":/workspace -v ~/data:/data \
  verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0 bash
```

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

verl has **first-party AMD support**, documented upstream in
[`docs/amd_tutorial/amd_quick_start.rst`](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst)
(rendered at [verl.readthedocs.io](https://verl.readthedocs.io/en/latest/amd_tutorial/amd_quick_start.html),
last updated 2026-07-24). Verified scope from that document:

- **GPUs:** MI300X / MI325X (`gfx942`) and MI350X / MI355X (`gfx950`).
- **Trainer backends:** FSDP, FSDP2 and Megatron.
- **Rollout engines:** vLLM and SGLang both fully supported, in Colocate and Fully Async modes.
- **Images:** the tutorial's validation image is `amdagi/verl-dev:rocm7.14_torch2.12_release_0724`;
  AMD also publishes ROCm verl images on Docker Hub under `rocm/verl` — current tag
  `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` (pushed 2026-07-31).
  A source build path exists via `docker/rocm/Dockerfile.rocm` in the verl tree.

```bash
docker run -it --device /dev/kfd --device /dev/dri --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g -w /workspace \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 /bin/bash
```

ROCm-specific caveats recorded upstream: SGLang requires `attention_backend=triton`, and
the ROCm image sets `vllm.disable_custom_all_reduce=True` by default because
`PYTORCH_ALLOC_CONF=expandable_segments:True` conflicts with vLLM custom all-reduce.
Follow the AMD tutorial rather than the CUDA instructions above.

The first-party claim above **held up locally** — see the next section for what actually ran.

## MI355X (ROCm 7.2) — TESTED (2026-08-19)

**Verdict: WORKS — as a container. The pip/venv route does not reach a trainable state.**
GRPO ran end-to-end on **2x AMD Instinct MI355X (gfx950, 288GB, ROCm 7.2.4 host)** using the
first-party `rocm/verl` image, with the **vLLM** rollout engine and the FSDP2 trainer. This
folder's own launcher, converter and reward file were used unmodified; only *run flags*
changed (the defaults assume 8xH100 and a large dataset — see "Flag changes" below).

### Route chosen: the `rocm/verl` container (not pip)

```bash
docker pull rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2
# 12 GB compressed -> 42.5 GB on disk. Only 3 tags exist under rocm/verl; this is the newest.
```

The container's ROCm 7.0.2 userspace ran fine against this host's **ROCm 7.2.4 kernel driver**.

Contents actually shipped in that image (the tag understates the verl version):

| Component | Version in image |
|---|---|
| verl | **0.8.0.dev0** (tag says 0.7.1) |
| torch | `2.9.1.dev20251204+rocm7.0.2` |
| vLLM | `0.20.2rc1.dev253+g1ff9d3353.rocm702` |
| ray | 2.55.1 |
| transformers | 5.14.1 |
| flash_attn | 2.8.4 (ROCm/CK build — **already present, never pip-install it**) |
| amd-aiter | 0.1.12.post2.dev217 |

**Why not pip.** A venv on this host reaches "imports and converts data" but not "trains":

- `torch==2.11.0+rocm7.2` (`--index-url https://download.pytorch.org/whl/rocm7.2`) installs
  clean and sees both MI355X as `gfx950:sramecc+:xnack-`. `prepare_data_verl.py` and
  `train_llm_verl.py --dry-run` work in that venv.
- But **there is no ROCm vLLM wheel on PyPI**, and vLLM is the only rollout engine modern
  verl can build (see the `hf` note below). Building vLLM-for-ROCm from source is a
  multi-hour job the container already did for you.
- verl **0.5.0** would have allowed a vLLM-free `rollout.name=hf` smoke, but it top-level
  imports `flash_attn.bert_padding` in `verl/workers/actor/dp_actor.py` whenever
  `torch.cuda.is_available()` — which is True on ROCm — so it cannot even import without
  flash-attn. (verl >= 0.6 fixed this: `verl/utils/attention_utils.py` imports flash-attn
  **lazily**, so `use_remove_padding=True` is fine on ROCm as long as a ROCm flash-attn
  exists, which it does inside the image.)

### Exact commands that worked

`--group-add render` fails (`unable to find group render`) because the image has no `render`
group — pass **numeric GIDs**. GPUs were pinned to physical 0,1 by exposing only their render
nodes (`/dev/dri/by-path/pci-<bus>-render`), **not** by env var — see the quirk below.

```bash
docker run -d --name verl_mi355x_smoke \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add 44 --group-add 993 \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=64g --network=host \
  -e HF_HOME=/hf_cache \
  -v "$PWD":/workspace/train_llm_verl -v /path/to/hf_cache:/hf_cache \
  -w /workspace/train_llm_verl \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity

docker exec -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  verl_mi355x_smoke bash -lc '
cd /workspace/train_llm_verl && python3 prepare_data_verl.py && \
python3 train_llm_verl.py \
  --algo grpo --model-path Qwen/Qwen3-0.6B \
  --reward-fn compute_score_format \
  --gpus-per-node 2 --strategy fsdp2 --rollout-backend vllm \
  --rollout-n 4 --rollout-tp 1 \
  --train-batch-size 4 --mini-batch-size 4 --micro-batch-size-per-gpu 1 \
  --max-prompt-length 2560 --max-response-length 640 \
  --gpu-mem-util 0.5 --epochs 2 --save-freq -1 --test-freq -1 \
  --experiment-name grpo_mi355x_smoke'
```

**Flag changes vs. this folder's defaults, and why:**

| Default | Used | Reason |
|---|---|---|
| `--train-batch-size 512` / `--mini-batch-size 256` | `4` / `4` | the shipped sample has **10 rows** |
| `--max-prompt-length 1024` | `2560` | sample prompts run to **2189 tokens**; with `data.filter_overlong_prompts=True` the default silently drops **5 of the 10 rows** |
| `--gpus-per-node 8` | `2` | only 2 GPUs allotted |
| `--rollout-tp 2` | `1` | Qwen3-0.6B needs no rollout TP |
| `--save-freq 20` / `--test-freq 5` | `-1` / `-1` | checkpointing + eval off for the smoke |
| `--reward-fn compute_score` | `compute_score_format` | see "reward collapse" below |

### Evidence

4/4 GRPO steps, exit 0, on `Qwen/Qwen3-0.6B`:

```
Training Progress: 100%|██████████| 4/4 [00:25<00:00,  6.42s/it]
step:2 - actor/pg_loss:1.490116e-08 - actor/grad_norm:1.727918 - critic/rewards/mean:0.267499 - critic/rewards/min:0.079999 - response_length/mean:536.0 - perf/throughput:2340.383
step:4 - actor/pg_loss:0.048292 - actor/kl_loss:0.001024 - actor/grad_norm:2.226501 - critic/rewards/mean:0.327499 - critic/rewards/max:0.600000 - perf/throughput:2152.627
```

`rocm-smi` sampled mid-run — only the two allotted GPUs are busy:

```
0  2  0x75a3, 49875  45.0°C  325.0W  NPS1, SPX, 0  2392Mhz  2000Mhz  auto  1400.0W  28%  0%
1  3  0x75a3, 58501  45.0°C  319.0W  NPS1, SPX, 0  2406Mhz  2000Mhz  auto  1400.0W  28%  0%
2  4  0x75a3, 56879  42.0°C  259.0W  NPS1, SPX, 0    94Mhz  2000Mhz  auto  1400.0W   0%  0%
```

Loss, KL and reward are all finite, `grad_norm` is non-zero, and mean reward rose
0.267 -> 0.327 over the four steps. That is the RL loop genuinely stepping on gfx950.

### Rollout backends: what was and was not exercised

| Backend | Status on MI355X |
|---|---|
| `vllm` | **Exercised and working** (vLLM 0.20.2rc1+rocm702, async server mode) |
| `sglang` | **Not exercised** — shipped in the image, upstream says it needs `attention_backend=triton` |
| `hf` | **Removed upstream — cannot work.** `--rollout-backend hf` is a dead option |

**The `hf` rollout is gone.** `verl/workers/rollout/base.py::_ROLLOUT_REGISTRY` contains only
`vllm`/`sglang` (0.6.x) and `vllm`/`sglang`/`trtllm` (0.7.1+), and `fsdp_workers.py` calls
`get_rollout_class(name, mode)` unconditionally, so `hf` trips
`assert (rollout_name, mode) in _ROLLOUT_REGISTRY`. `HFRollout` is still exported but
unreachable. The last version where `--rollout-backend hf` actually builds is **0.5.0**.
There is therefore **no vLLM-free rollout path** on current verl.

### Quirks worth knowing (ROCm-specific)

1. **Never set both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.** vLLM's ROCm platform
   runs `_sync_hip_cuda_env_vars()` at import and **raises** if both are set and differ:
   `ValueError: Inconsistent GPU visibility env vars: HIP_VISIBLE_DEVICES='0' vs CUDA_VISIBLE_DEVICES='0,1'`.
   Ray and vLLM each rewrite *one* of the two per actor, so setting both is a guaranteed
   conflict. It treats empty string as unset and syncs whichever one is set — so set
   **neither** in the container and pin GPUs with per-device `--device /dev/dri/renderDxxx`
   instead. Map GPU index -> render node via `rocm-smi --showbus` + `ls -l /dev/dri/by-path`.
2. **`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` is required.** Ray blanks the accelerator
   visibility var for `num_gpus=0` actors; verl's `TaskRunner` is one, and AITER's Triton
   kernels import there, giving `RuntimeError: 0 active drivers ([]). There should only be
   one.` Ray itself prints the fix as a `FutureWarning`.
3. **Ray-on-ROCm otherwise just works** — local cluster, dashboard, GPU actor placement, and
   the vLLM HTTP server actors all came up unmodified.
4. **AITER JIT-compiles on first launch** (`[aiter] start build [module_rmsnorm]` ...),
   adding ~2-4 min of startup. `VLLM_ROCM_USE_AITER=1` is the image default and was left on;
   `VLLM_ROCM_USE_AITER_MOE=0` was set defensively (AITER MoE kernels are known to corrupt
   output on gfx950 — irrelevant for dense Qwen3-0.6B, but harmless and cheap insurance).
   `USE_ROCM_AITER_ROPE_BACKEND=0` disables the lower-precision AITER fused RoPE if you need
   bit-comparable results.
5. **Do not pip-install flash-attn.** The image already ships a ROCm build; the sdpa/CK path
   is the ROCm attention path. `use_remove_padding=True` worked as-is.
6. **verl writes a Hydra `outputs/` dir into the CWD** — root-owned when run in the
   container, so `rm -rf` it from a container, not the host.
7. **Reward collapse is easy to hit.** With the shipped `compute_score`, every rollout scored
   exactly `0.1` (the "non-empty completion" floor) on this telecom sample, so GRPO's
   group-relative advantage was identically 0 and `actor/grad_norm` stayed `0.0` for all four
   steps — the trainer stepped, but the update was numerically a no-op. This is precisely the
   failure the "Reward function contract" section warns about, reproduced on real hardware.
   `compute_score_format` discriminates (some rollouts close `<think>`, some are truncated),
   which is what produced the non-zero gradients above.
8. **Benign teardown noise:** `RuntimeError: DataLoader worker ... is killed by signal:
   Killed.` prints after training completes; the process still exits 0.

### 8-GPU run (8x MI355X, ROCm 7.2.4) — tested August 2026

**Verdict: WORKS — scales cleanly from 2 to 8 GPUs with no code change and no new flags.**
The 2-GPU smoke above was re-run on **all 8 MI355X** with this folder's *default* topology
(`--gpus-per-node 8 --rollout-tp 2`) and the folder's *default model* (`Qwen/Qwen3-4B`),
5 GRPO steps, exit 0, both passes. Nothing in `train_llm_verl.py`, `prepare_data_verl.py`,
`reward_verl.py` or `requirements_verl.txt` needed to change — only batch/length flags, and
those are forced by the 10-row sample, not by the GPU count.

#### GPU layout (what verl actually did with 8 GPUs)

verl's default hybrid engine **colocates** the trainer and the rollout engine on the same
8 GPUs — it does not partition them into a "rollout half" and a "trainer half":

| Role | Sharding | Processes seen holding VRAM |
|---|---|---|
| FSDP2 actor + ref (`trainer.n_gpus_per_node=8`, `nnodes=1`) | world size **8**, 1 rank/GPU | 8x `ray::WorkerDict` |
| vLLM rollout (`rollout.tensor_model_parallel_size=2`) | **4 engine replicas x TP2 = 8 GPUs** | 4x `ray::vLLMHttpServer` -> 8x `VLLM::Worker_TP` |
| Reward / agent loop (CPU) | — | `ray::RewardLoop`, `ray::AgentLoopWorker` (0 VRAM) |

So rollout DP is implicit: `n_gpus / rollout_tp = 8 / 2 = 4` vLLM engines, each 2-way
tensor-parallel, sharing the *same* 8 GPUs as the FSDP2 ranks. Weights are resynced
trainer -> rollout every step (`timing_s/update_weights` ~1.8-2.3 s). The classic 8-GPU RL
failure points did **not** materialise: no RCCL timeout on weight resync, no Ray
placement-group error claiming all 8 GPUs, no vLLM engine-init failure at TP2.

#### Exact commands

```bash
# 1. container — all 8 GPUs (numeric GIDs; the image has no `render` group)
docker run -d --name verl_mi355x_gpu8 \
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
  -v /mnt/data_1.5t/hf_cache:/hf_cache \
  -v /mnt/data_1.5t/outputs/train_llm_verl/gpu8:/outputs \
  -w /outputs \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity

# 2. data prep (unchanged)
docker exec verl_mi355x_gpu8 bash -lc 'cd /workspace/train_llm_verl && python3 prepare_data_verl.py'

# 3. GRPO on 8 GPUs
docker exec -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  verl_mi355x_gpu8 bash -lc '
cd /outputs && python3 /workspace/train_llm_verl/train_llm_verl.py \
  --algo grpo --model-path Qwen/Qwen3-4B \
  --train-file /workspace/train_llm_verl/data/otel_train.parquet \
  --val-file /workspace/train_llm_verl/data/otel_train.parquet \
  --reward-file /workspace/train_llm_verl/reward_verl.py \
  --reward-fn compute_score_format \
  --gpus-per-node 8 --strategy fsdp2 --rollout-backend vllm \
  --rollout-n 8 --rollout-tp 2 \
  --train-batch-size 10 --mini-batch-size 10 --micro-batch-size-per-gpu 1 \
  --max-prompt-length 2560 --max-response-length 1024 \
  --gpu-mem-util 0.6 --epochs 5 --save-freq -1 --test-freq -1 \
  --experiment-name grpo_mi355x_gpu8'
```

`-w /outputs` matters: verl drops a Hydra `outputs/` tree into the **CWD**, root-owned
(quirk 6). Running from a mounted scratch dir keeps it out of this folder. Because the
CWD is no longer the folder, `load_dotenv("dev.env")` finds nothing — pass `HF_TOKEN`
through `docker exec -e` instead, and give `--train-file` / `--val-file` / `--reward-file`
as absolute paths.

#### What differed from the 2-GPU run

| Knob | 2-GPU smoke | 8-GPU run | Why |
|---|---|---|---|
| `--gpus-per-node` | 2 | **8** (folder default) | all 8 GPUs |
| `--rollout-tp` | 1 | **2** (folder default) | 4 vLLM replicas x TP2 = 8 |
| `--model-path` | `Qwen/Qwen3-0.6B` | **`Qwen/Qwen3-4B`** (folder default) | 8 GPUs can hold the default model |
| `--rollout-n` | 4 | **8** | divisibility, see below |
| `--train-batch-size` / `--mini-batch-size` | 4 / 4 | **10 / 10** | uses all 10 sample rows |
| `--max-response-length` | 640 | **1024** (folder default) | headroom available |
| `--gpu-mem-util` | 0.5 | **0.6** (folder default) | 288 GB HBM/GPU |
| GPU pinning | 2 render nodes | 8 render nodes (`renderD128+8i`) | GPU *i* -> `/dev/dri/renderD$((128+8*i))`, confirmed via `rocm-smi --showbus` + `/dev/dri/by-path` |
| container flags | same | same | `--group-add 44 --group-add 993`, `--ipc=host`, `--shm-size 64G` unchanged |

**The one 8-GPU-specific gotcha: batch divisibility.** verl asserts
`train_batch_size * rollout.n` is divisible by the world size. The folder default
`--rollout-n 5` with a 10-row dataset gives `50`, which is **not** divisible by 8 and aborts
in `_validate_config` before any GPU work. `--rollout-n 8` (-> `80`) is the smallest fix
that keeps all 10 rows. This constraint tightens as you add GPUs and is the single most
likely thing to bite when moving a working 2-GPU config to 8.

Everything else carried over verbatim: `VLLM_USE_TRITON_FLASH_ATTN=0`,
`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=0`,
`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` (quirk 2), no `HIP_VISIBLE_DEVICES`/
`CUDA_VISIBLE_DEVICES` (quirk 1), `compute_score_format` (quirk 7).

#### Evidence — training

5/5 steps, `rc=0`, on `Qwen/Qwen3-4B`, 8 GPUs (second pass; the first pass reproduced it):

```
Training Progress: 100%|██████████| 5/5 [01:00<00:00, 12.08s/it]
step:1 pg_loss:-0.0022836 kl_loss:0.0       grad_norm:0.7018 entropy:0.38875 rewards/mean:0.29400 resp_len:822.8 step_s:13.58 throughput:1404.2
step:3 pg_loss:-0.0021772 kl_loss:0.0007757 grad_norm:0.5642 entropy:0.38440 rewards/mean:0.29100 resp_len:816.8 step_s:10.74 throughput:1769.4
step:5 pg_loss:-0.0033790 kl_loss:0.0008408 grad_norm:0.5375 entropy:0.39564 rewards/mean:0.30050 resp_len:820.5 step_s:11.10 throughput:1715.2
step:5 ... critic/advantages/max:2.4748 critic/advantages/min:-2.4748 timing_s/update_weights:2.131 timing_s/gen:6.543 perf/total_num_tokens:152367
'Final validation metrics: None'
=== TRAINING EXIT rc=0 ===
```

Rollouts generated (80/step = 10 prompts x `rollout-n 8`), reward computed
(`timing_s/reward`, scores spread `0.08 .. 0.60` — **not** collapsed), advantages non-zero
(+/-2.47), policy updated (`timing_s/update_actor` ~1.6 s, `grad_norm` 0.54-0.70, never 0),
KL finite and rising off zero as the policy moves, mean reward `0.2940 -> 0.3005` over
5 steps. Honest caveat: **5 steps on 10 rows is not a learning curve** — reward wobbles
(0.294 / 0.290 / 0.291 / 0.292 / 0.301) rather than climbing monotonically. What is proven
is that the full RL loop executes correctly at 8-GPU scale, not that the model learned.

Rollout text was **coherent, not garbage**: the `0.60` scores require both a closed
`<think>...</think>` block *and* the gold answer appearing verbatim in the completion
(`0.8*0.5 + 0.2`). So `VLLM_ROCM_USE_AITER=1` was safe here — Qwen3-4B is dense, and the
known-bad AITER **fused-MoE** path is never taken. The `VLLM_ROCM_USE_AITER_MOE=0`
insurance from the 2-GPU run was kept. If you swap in an MoE model and rollouts come out as
nonsense, set `VLLM_ROCM_USE_AITER=0` and retry.

#### Evidence — all 8 GPUs, sampled *inside* the run, PID-verified

`rocm-smi` was sampled from a background loop running for the lifetime of the training
process (not after it), every ~20 s. Per-sample summary, one column per GPU:

```
17:29:47  use=[ 0,0,0,0,0,0,0,0]      vram%=[ 0 x8]   maxW=261   <- lock acquired, idle
17:30:50  use=[ 0,0,0,0,0,0,0,0]      vram%=[ 2 x8]   maxW=312   <- 8 FSDP2 ranks loading
17:31:39  use=[ 0,0,0,0,0,0,0,0]      vram%=[74 x8]   maxW=326   <- vLLM KV caches allocated on all 8
17:32:07  use=[100 x8]                vram%=[64 x8]   maxW=548   <- generation + update, ALL 8 BUSY
17:32:37  use=[ 9 x8]                 vram%=[30 x8]   maxW=358   <- teardown
```

(The first pass, with a cold AITER JIT cache, peaked harder still: all 8 at 76-77 % use and
**830-868 W** each during the actor update.)

At the 17:32:07 sample, 16 processes held VRAM — exactly the predicted layout:

```
3803376..3803383  ray::WorkerDict   x8   ~13.0-13.3 GB each   <- FSDP2 trainer, 1 rank per GPU
3814220..3814374  VLLM::Worker_TP   x8   ~186 GB each         <- 4 engines x TP2, gpu_mem_util 0.6 of 288 GB
```

**PID cross-check (the part that matters on a shared box).** Every PID reported by
`rocm-smi --showpids` was resolved *live, in the same sample* to its owning container via
`/proc/<pid>/cgroup`:

```
my container id=3eef378ca0f7
pid=3803376 comm=ray::WorkerDict   owner=3eef378ca0f7 MINE
pid=3814220 comm=VLLM::Worker_TP   owner=3eef378ca0f7 MINE
pid=3810095 comm=ray::vLLMHttpSe   owner=3eef378ca0f7 MINE
...
sample 17:32:07 -> MINE=51  OTHER=0
```

51/51 GPU-touching processes belonged to this run's container, and zero belonged to anything
else, at the same instant all 8 GPUs read 100 % busy. Three lines earlier in the run showed
`OTHER` with an *empty* owner — those are PIDs that exited between the `--showpids` call and
the `/proc` read (transient AITER JIT workers), not another tenant.

> **Do not sample `rocm-smi` from a separate shell after the fact.** This box runs several
> agents' 8-GPU jobs back to back; a sample taken seconds after your run exits will happily
> show *someone else's* job at 100 %. Put the sampler inside the same locked script as the
> training, and resolve PIDs to your own container while they are still alive.

#### Scaling observation

| | 2x MI355X | 8x MI355X |
|---|---|---|
| Model | Qwen3-0.6B | Qwen3-4B (**6.7x params**) |
| Rollouts per step | 4 x 4 = 16 | 10 x 8 = **80** (5x) |
| Response cap | 640 tok | 1024 tok |
| Time per step | ~6.4 s | ~11 s |
| `perf/throughput` | ~2 150-2 340 tok/s | ~1 715-1 774 tok/s |

Throughput per step is *lower* in absolute tokens/s while doing far more work per step, which
is expected and not a scaling failure: the run is **launch-bound, not compute-bound**. With
10 prompts spread over 8 GPUs each rank sees ~1 sequence, so per-step cost is dominated by
fixed overheads — `timing_s/gen` 6.5 s and `timing_s/update_weights` ~1.8-2.3 s together are
~80 % of an 11 s step, while `update_actor` is only ~1.6 s. `global_seqlen/minmax_diff` of
~11 000-16 000 tokens shows the token-balancing across 8 ranks is working but the batch is
simply too small to fill them. **The honest read: 8 GPUs buy capacity (bigger model, 5x the
rollouts, longer responses at the same wall-clock per step), not speed, on a 10-row sample.**
To convert 8 GPUs into throughput you need a real dataset — raise `--train-batch-size`
toward the folder default of 512 so each rank gets a full micro-batch, and consider raising
`--rollout-tp` only if the model no longer fits at TP2.

Startup cost is the other 8-GPU tax: **~6 min** cold (AITER JIT builds + 4 vLLM engines +
CUDA-graph capture) vs **~3 min** warm on the second pass in the same container. Ray, RCCL
and the vLLM engines all came up clean at 8 GPUs — no placement-group contention, no
`update_weights` timeout.

#### Housekeeping

`--save-freq -1` was used, so **no checkpoints were written** (verified: no
`.safetensors`/`.pt`/`checkpoints/` anywhere under the output dir). The only artefacts are
logs plus a 104 KB root-owned Hydra `outputs/` config dump. Note the `rocm/verl` image is
**42.5 GB on disk** — if it has been pruned since the last run you must re-pull it (12 GB
over the wire, ~2 min here), which is the single largest disk cost of this folder.

## NVIDIA H100 (CUDA 13.0) — TESTED (2026-08-22)

**Verdict: WORKS-WITH-CHANGES — as a container. GRPO stepped end-to-end (4/4 steps, exit 0)
on 1x H100 80GB** using the upstream `verlai/verl` CUDA image, vLLM rollout + FSDP2 trainer,
`Qwen/Qwen3-0.6B`. Three run-flag/override changes were needed vs. the MI355X recipe — none
touched this folder's `train_llm_verl.py`, `prepare_data_verl.py` or `reward_verl.py` code,
but two of them were **Hydra overrides the launcher does not expose** (see "Changes" below),
so they were passed by calling `verl.trainer.main_ppo` directly. Single-GPU smoke only;
multi-GPU is deferred (see end).

### The image tag in the README/requirements is dead — use `uv.cu130`

`verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0` (the tag written into this README's
"NVIDIA (CUDA)" install block and `requirements_verl.txt`) **404s on Docker Hub** — verl
retagged everything. The current CUDA-13 image is `verlai/verl:uv.cu130` (12.8 GB over the
wire, **43.3 GB on disk**, pushed 2026-07-30). Query live tags with:

```bash
curl -s "https://hub.docker.com/v2/repositories/verlai/verl/tags/?page_size=100&ordering=last_updated" \
  | python3 -c "import sys,json;[print(t['name']) for t in json.load(sys.stdin)['results']]"
```

Host: 8x H100 80GB HBM3, driver **580.173.02**, CUDA 13.0, Python 3.12.3. Docker 29.1.3,
nvidia runtime configured. The proxy (`proxy.conexus.svc.local:3128`) 403s Docker Hub — you
must `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy` before pulling.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull verlai/verl:uv.cu130
```

### The `uv.cu130` image ships NO ready `.venv` — you must `manage_envs.py sync`

Unlike the old "app" images (verl + vllm pre-installed), the `uv.*` images bake only the
**uv package cache** for every backend; `/workspace/verl/.venv` does not exist until you
sync one. `/workspace/verl/.venv/bin` is already on `PATH`, so after syncing, `python3`
resolves to it. The sync is fully offline from the baked cache (~9 s):

```bash
# GPU 4 ONLY — verify exactly 1 GPU is visible inside before doing anything (see quirk 1)
sudo docker run -d --name verl_h100_smoke \
  --gpus '"device=4"' --ipc host --shm-size 16g \
  -e HF_HOME=/models -e HF_TOKEN="$HF_TOKEN" \
  -v /mnt/gsma/gsma/gsma/models:/models \
  -v "$PWD":/work -v /dev/shm/h100/out/verl:/out \
  verlai/verl:uv.cu130 sleep infinity

sudo docker exec verl_h100_smoke nvidia-smi -L      # MUST show exactly ONE H100
sudo docker exec verl_h100_smoke bash -lc 'cd /workspace/verl && python3 manage_envs.py sync vllm -- --frozen'
```

What the vllm slice materialised (re-verified with `python -c "import torch,verl,vllm"`):

| Component | Version in `.venv` (vllm slice) |
|---|---|
| torch | **`2.11.0+cu130`** (native CUDA 13, `torch.cuda.is_available()`→True, "NVIDIA H100 80GB HBM3") |
| verl | **0.9.0.dev0** (newer than MI355X's 0.8.0.dev0 — reward API drifted, see below) |
| vLLM | **0.24.0** |
| ray | 2.55.1 |
| transformers | **5.5.3** (5.x) |
| triton | 3.6.0 · tensordict 0.10.0 |
| flash_attn | **NOT installed in the vllm slice** — present only in the uv cache (2.8.3). This is the pivotal H100 change; see below. |

### Exact command that worked

The folder launcher hardcodes `use_remove_padding=True` and cannot set `attn_implementation`
or the verl-0.9 `reward.*` keys, so the smoke calls `verl.trainer.main_ppo` directly. Data
prep is unchanged (`python3 prepare_data_verl.py` → 10 rows, none dropped at
`--max-prompt-length 2560`).

```bash
sudo docker exec verl_h100_smoke bash -lc '
source /workspace/verl/.venv/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /work && python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/otel_train.parquet data.val_files=data/otel_train.parquet \
  data.train_batch_size=4 data.max_prompt_length=2560 data.max_response_length=640 \
  data.filter_overlong_prompts=True data.truncation=error algorithm.use_kl_in_reward=False \
  actor_rollout_ref.model.path=/models/hub/models--Qwen--Qwen3-0.6B/snapshots/<snap> \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-06 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path=/work/reward_verl.py custom_reward_function.name=compute_score_format \
  reward.custom_reward_function.path=/work/reward_verl.py reward.custom_reward_function.name=compute_score_format \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=2 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.logger=[console] trainer.critic_warmup=0'
```

**Changes vs. this folder's defaults / the MI355X recipe, and why:**

| Knob | MI355X (2-GPU) | H100 (1-GPU) | Reason |
|---|---|---|---|
| `--gpus-per-node` / `n_gpus_per_node` | 2 | **1** | one free GPU allotted (GPUs 0–3 were a production job) |
| `--rollout-tp` | 1 | 1 | Qwen3-0.6B needs no rollout TP |
| model | Qwen3-0.6B | Qwen3-0.6B | small model; FSDP2 actor + colocated vLLM fit on one 80 GB card |
| `use_remove_padding` | True | **False** | avoids verl's remove-padding path (which pulls in flash-attn) |
| `model.override_config.attn_implementation` | (default fa2) | **`sdpa`** | **the key H100 change** — see below |
| `reward.custom_reward_function.{path,name}` | n/a (verl 0.8) | **set explicitly, abs path** | verl-0.9 reward API drift — see below |
| `--train-batch-size` / `--mini-batch-size` | 4 / 4 | 4 / 4 | 10-row sample; `4×rollout_n(4)=16` divisible by world size 1 |
| `--max-prompt-length` | 2560 | 2560 | sample prompts reach 2189 tok; default 1024 + `truncation=error` would abort |

#### The two H100-specific gotchas

1. **flash-attn is not in the image's vllm slice, and verl 0.9 + transformers 5.5.3 demand
   it by default.** With `use_remove_padding=True` (the launcher default) the actor init
   raises `ImportError: FlashAttention2 has been toggled on, but ... the package ... doesn't
   seem to be installed` from `transformers/modeling_utils.py::_flash_attn_can_dispatch`.
   Setting `use_remove_padding=False` is **not enough** — verl's automodel engine
   (`verl/workers/config/model.py:185`) still defaults `attn_implementation="flash_attention_2"`
   when building the HF actor/ref, so `AutoModelForCausalLM.from_pretrained` re-raises the same
   error. The fix is to force **`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`**.
   H100 supports flash-attn (a prebuilt wheel exists), and per the brief the reversal of the
   ROCm workaround is to *use* fa2 — but installing it here is a **compile**:
   `uv pip install flash-attn==2.8.3 --no-build-isolation` for torch 2.11+cu130/py3.12 found no
   matching prebuilt wheel and started an nvcc source build that OOM-killed the container
   (exit 137) inside the 25-min box. `sdpa` is the documented fallback and cost nothing —
   the vLLM **rollout** still uses its own baked `vllm_flash_attn`, so only the trainer/ref
   forward runs on SDPA. A follow-up could pre-bake flash-attn into the image or sync a slice
   that includes it; for a smoke, SDPA is correct and complete.
2. **verl 0.9 reward-API drift — the launcher's `custom_reward_function.*` is silently
   ignored.** verl 0.9 introduced an experimental `RewardLoopWorker`. The legacy→new config
   migration (`migrate_legacy_reward_impl`, which copies `custom_reward_function` →
   `reward.custom_reward_function`) is **only called from the fully-async entrypoint**
   (`experimental/fully_async_policy/fully_async_main.py`), **not** from the standard
   `main_ppo`/`trainer_base.py` path this folder uses. So the worker reads
   `config.reward.custom_reward_function.path`, finds it `None`, and falls through to the
   default score registry → `NotImplementedError: Reward function is not implemented for
   data_source='otel_local'` — *after* rollouts generate and CUDA graphs capture, i.e. late.
   Fix: pass the reward on the **new** keys too, with an **absolute** path (Ray reward actors
   don't run in `/work`): `reward.custom_reward_function.path=/work/reward_verl.py
   reward.custom_reward_function.name=compute_score_format`. (Keeping the legacy keys is
   harmless.) The MI355X run used verl 0.8.0.dev0 and never hit this.

Other MI355X quirks that **did not apply on CUDA**: no `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`
conflict (that is ROCm-vLLM-specific — here the container is pinned with `--gpus '"device=4"'`
and sees exactly one GPU as index 0); no `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` needed to
*run* (Ray prints it as a `FutureWarning` but the CPU reward actors ran fine without it, since
no AITER/Triton import happens in the num_gpus=0 TaskRunner on CUDA); no AITER JIT warmup.

### Evidence — training

4/4 GRPO steps, `rc=0`, on `Qwen/Qwen3-0.6B`, 1×H100. Real log lines (from
`/dev/shm/h100/out/verl/train_h100.log`):

```
step:0 - val-core/otel_local/acc/mean@1:0.3120   <- custom reward IS routing on otel_local
Training Progress: 100%|██████████| 4/4 [01:02<00:00, 15.72s/it]
step:1 - actor/pg_loss:0.044740 - actor/grad_norm:2.85926 - actor/kl_loss:0.0 - critic/rewards/mean:0.3150 (max 0.60 min 0.08) - critic/advantages/min:-1.49999 max:0.49999 - response_length/mean:541.3 - perf/throughput:1547.9
step:3 - actor/pg_loss:0.014787 - actor/grad_norm:2.14228 - actor/kl_loss:0.000926 - critic/rewards/mean:0.3200 (max 1.00 min 0.08) - critic/advantages/max:1.34597 - perf/throughput:1652.3
step:4 - actor/pg_loss:0.033660 - actor/grad_norm:2.19386 - actor/kl_loss:0.001082 - critic/rewards/mean:0.3075 - actor/perf/max_memory_allocated_gb:57.93 - perf/throughput:1770.6
=== TRAINING EXIT rc=0 ===
```

Loss/KL/reward all finite; `grad_norm` non-zero on steps 1/3/4; reward spread `0.08..1.00`
(a step-3 rollout hit an exact match → 1.0), advantages non-zero (±1.5). vLLM↔actor
consistency was excellent: `training/rollout_actor_probs_pearson_corr:0.9993`. **Honest
caveat — same as the MI355X note:** step 2 hit the documented reward-collapse (all four
rollouts scored exactly 0.28 → `critic/advantages:0.0`, `actor/grad_norm:0.0004`, a
numerical no-op); and 4 steps on 10 rows is not a learning curve. What is proven is that the
full RL loop — vLLM rollout generation, custom-reward scoring, GRPO advantage, FSDP2 actor
update, weight resync — executes correctly on Hopper.

### Evidence — GPU 4 residency, sampled live mid-run, PID-verified

`nvidia-smi --id=<GPU-4 UUID>` at 05:19:15 while `Training Progress` read 1/4, with every
compute PID resolved to its owning container via `/proc/<pid>/cgroup`:

```
GPU4 (UUID GPU-e13d18b6-...-cd489ad01b55): util=100%  mem=70897 MiB  power=330 W
  pid=1681003 comm=ray::WorkerDict  mem=69474 MiB  owner=d7c7db7e0dd2  MINE   <- FSDP2 actor
  pid=1683720 comm=VLLM::Worker     mem= 1400 MiB  owner=d7c7db7e0dd2  MINE   <- colocated rollout
my container id=d7c7db7e0dd2
```

Both GPU-4 PIDs belong to this run's container; the classic verl colocated layout (FSDP2
actor + vLLM engine sharing one GPU). Peak trainer memory was
`actor/perf/max_memory_allocated_gb:57.9` (reserved 70.8 GB) — comfortable on 80 GB at
`gpu_memory_utilization=0.5`. GPUs 0–3 (the production job) stayed at their own 100%/~65 GB
and were **never visible to this container** (`nvidia-smi -L` inside showed exactly one H100).

> **Shared-node hygiene:** the container was pinned with `--gpus '"device=4"'`, so it can
> only ever touch GPU 4. Do not sample the whole node and claim a GPU — resolve PIDs to your
> own container id (`docker inspect -f '{{.Id}}'`) while they are alive, as above.

### Multi-GPU (deferred)

Single-GPU only this wave. A 2- or 8-GPU pass on H100 would need: `trainer.n_gpus_per_node`
raised to the GPU count; the **batch-divisibility** rule from the MI355X 8-GPU run
(`train_batch_size × rollout.n` must be divisible by world size — e.g. `rollout.n=8` for 8
GPUs on the 10-row sample); and, if a larger model (Qwen3-4B) is used, `rollout-tp 2` and a
higher `gpu-mem-util`. The `sdpa` and `reward.custom_reward_function` overrides above carry
over unchanged. Do not launch it until GPUs 0–3 are free and the lead coordinates.

### Housekeeping

`save_freq=-1` → **no checkpoints written** (verified: no `.safetensors`/`.pt`/`checkpoints/`).
verl still drops a ~163 KB **root-owned** Hydra `outputs/` dir into the CWD (`/work`, i.e. this
folder) — `rm -rf /work/outputs` it **from inside the container** (root-owned on the host).
The `verlai/verl:uv.cu130` image is **43.3 GB on disk**; the 12.8 GB pull is the largest cost.

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

Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version`). **Null-column trap:** `source_spec_id` and
`source_version` are null in most rows. The converter here only reads `messages`, so the
extra columns are dropped — but if you feed the raw JSONL to a different loader, mostly-null
columns can be inferred with the wrong type or crash strict schema readers.

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
into the rollout engine, then each step logs `critic/rewards/mean`, `actor/kl_loss`,
`response_length/mean`, and timing breakdowns. The signal to watch is **mean reward trending
up while response length stays sane**. Flat reward from step 1 almost always means the reward
function is not discriminating between rollouts — check `reward_verl.py` against a few real
completions before blaming the RL.

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
| `--rollout-backend` | `vllm` | Generation engine. `vllm` (**verified on MI355X**) or `sglang`. **`hf` is dead** — removed from verl's rollout registry in 0.6.0; it asserts at worker init |
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

## Hardware support & evidence

**Other hardware (upstream claims — not verified here):** Ascend NPU support is documented upstream (vllm-ascend path), in addition to NVIDIA/AMD.


| Hardware | Status | Evidence |
|---|---|---|
| NVIDIA (H100/A100 class) | Primary target | Upstream `verlai/verl` Docker images and install docs — <https://verl.readthedocs.io/en/latest/start/install.html> |
| AMD MI300X / MI325X (`gfx942`) | First-party supported | [`docs/amd_tutorial/amd_quick_start.rst`](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_quick_start.rst) (updated 2026-07-24): FSDP/FSDP2/Megatron, vLLM + SGLang, Colocate + Fully Async |
| AMD MI350X / MI355X (`gfx950`) | **First-party supported — VERIFIED HERE, 2 AND 8 GPUs** | 2026-08-19: 4 GRPO steps on 2x MI355X, then 5 GRPO steps on **8x MI355X** (Qwen3-4B, FSDP2 world 8 + 4x vLLM TP2, all 8 GPUs at 100% with PID-verified ownership), ROCm 7.2.4 host, `rocm/verl` image, non-zero `grad_norm`, finite loss/KL/reward, exit 0. See [§ MI355X tested](#mi355x-rocm-72--tested-2026-08-19) above |
| AMD ROCm images | Published — **pulled and run** | Docker Hub `rocm/verl` — tag `verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` (2026-07-31, 12 GB compressed / 42.5 GB on disk). Only 3 tags exist. Ships verl **0.8.0.dev0**, not 0.7.1. Plus `docker/rocm/Dockerfile.rocm` for source builds |
| Ascend NPU | Documented upstream | verl docs carry an `ascend_tutorial` section; not exercised here |

The AMD row is **locally verified**; the NVIDIA and Ascend rows still record upstream's
support claims only. The first-party AMD support is real: the published image worked on
gfx950 with no patches to this folder's code — but note that the *only* rollout engine
available on current verl is vLLM/SGLang (`--rollout-backend hf` was removed in verl 0.6.0),
so there is no lightweight, container-free way to run this folder on ROCm.

## Notes

- **GRPO vs PPO.** PPO trains a separate critic to estimate the value baseline, which
  roughly doubles memory. GRPO drops the critic and instead samples `--rollout-n` completions
  per prompt, using the group's mean reward as the baseline. That is why `--rollout-n 1`
  is meaningless under GRPO — the advantage would be identically zero.
- **Two engines, one set of GPUs.** The actor (training) and the rollout engine (generation)
  both want HBM. verl's 3D-HybridEngine reshards the actor between the training and
  generation phases rather than keeping two copies resident. `--gpu-mem-util` is how you
  split the budget; the most common OOM fix is lowering it, not lowering batch size.
- **Rollout TP is separate from training sharding.** `--rollout-tp` only sizes the vLLM/SGLang
  engine. You can shard training across 8 GPUs while generating with TP=2.
- **Reward functions are the actual work.** The RL machinery is generic; almost all run
  quality comes from `reward_verl.py`. Keep scores roughly in `[0, 1]` and make sure a
  better answer really does score higher — see the contract section above.
- **KL anchoring.** `--kl-loss-coef` keeps the policy near the frozen reference model.
  Too low and the model reward-hacks into gibberish; too high and it never moves.
- **Dynamic batching.** `use_dynamic_bsz=True` packs by token count instead of sequence
  count, which matters a lot when response lengths vary. The launcher enables it by default.
