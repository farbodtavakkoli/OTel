# `training/llm/primus` — AMD Primus post-training (SFT + LoRA on ROCm)

## 1. Overview & when to use

Post-training (SFT and LoRA) on **AMD Instinct GPUs** with
[Primus](https://github.com/AMD-AGI/Primus) (Primus-LM), AMD's training framework for
large-scale foundation models on ROCm. Primus unifies several backends —
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[TorchTitan](https://github.com/pytorch/torchtitan), JAX MaxText, and Megatron-Bridge —
under one configuration system, with ROCm-optimized kernels underneath.

This is the **AMD hardware path** for this repo. Every other trainer here assumes NVIDIA;
Primus is the one that is designed for MI300X/MI355X first, and it is production-proven —
AMD used it for the Llama 2 70B LoRA fine-tune in MLPerf Training v6.0. Post-training
specifically runs through the **Megatron-Bridge** backend via the `train posttrain`
subcommand.

Primus is config-driven: all real settings live in the YAML, and training is started with
`primus-cli <mode> -- train posttrain --config <yaml>`. `train_llm_primus.py` is a thin
wrapper that picks the right config for your GPU arch, applies common overrides, checks
the ROCm environment, and execs the CLI — so a run is one command and the arch/config
pairing is validated before you burn a cluster allocation.

Files in this folder:
- `train_llm_primus.py` — launcher; resolves the right config for your GPU arch, checks ROCm, and execs `primus-cli`.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample shipped with this folder (the default `--data-path`).
- `configs/MI300X/qwen3_32b_sft_posttrain.yaml` — full fine-tune, gfx942 (MI300X / MI325X).
- `configs/MI300X/qwen3_32b_lora_posttrain.yaml` — LoRA, gfx942.
- `configs/MI355X/qwen3_32b_sft_posttrain.yaml` — full fine-tune, gfx950 (MI350X / MI355X).
- `configs/MI355X/qwen3_32b_lora_posttrain.yaml` — LoRA, gfx950.
- `requirements_primus.txt` — install paths (container / wheel / bare metal).
- `readme_primus.md` — this document.

> **Tested topology:** **Single node, 2x AMD Instinct MI355X (gfx950), ROCm 7.2.4
> — the FRAMEWORK WORKS; the SHIPPED CONFIGS DO NOT RUN AS WRITTEN.** Primus +
> Megatron-Bridge SFT trained end-to-end in `rocm/primus:v26.5` with finite, decreasing
> loss (600/600 iterations, ~250-280 ms/iter, zero NaN/skipped). But **all four configs in
> `configs/` fail before training starts** — two independent bugs, both one-line fixes, both
> now corrected in this folder. The AMD-only claim in §8 is **confirmed by execution**.
> LoRA (`peft: lora`) applies its adapters but then **fails to load the base checkpoint**
> in this image — see [§8.1](#81-mi355x-rocm-724). The launcher's
> `--data-path` plumbing is **dead code** on the Megatron-Bridge post-training path (§4).
> Not tested: Qwen3-32B at the shipped scale (weights were not downloaded — the 0.6B
> flavor of the same `qwen.qwen3` recipe was used), multi-node, `slurm` mode, MI300X/gfx942.

## 2. Install

### AMD / ROCm (the only supported path)

**Verified on 2x MI355X / ROCm 7.2.4 — the container route works.** Image
`rocm/primus:v26.5`, digest
`sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1`.
It is **14.1 GB compressed / 54.8 GB on disk** — check free space before
pulling. The image ships the whole stack pre-built at `/workspace/Primus` (Primus 0.2.0,
checkout `b511d1b`, with `third_party/Megatron-Bridge` @ `9577b12` and
`third_party/Megatron-LM` @ `d3528a2`), so **you do not need to clone anything** — the
`git clone` below is only for reading sources on the host.

Primus is built around AMD's published ROCm images. Upstream recommends using them, and
that is by far the least painful route — a bare-metal ROCm + Megatron build is a project in
itself.

**Prerequisites:** ROCm drivers >= 7.0, Docker >= 24.0 with ROCm support, and an Instinct
GPU — MI300X / MI325X (gfx942) or MI350X / MI355X (gfx950). Quick check:

```bash
rocm-smi && docker --version
```

**Option 1 — clone and run in the container (recommended):**

```bash
docker pull rocm/primus:v26.5

git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init --recursive
```

The `v26.5` tag was re-verified against
[Docker Hub](https://hub.docker.com/r/rocm/primus/tags) — it is live, alongside the more
explicit alias `v26.5-pytorch2.12-te2.15` which points at the same image digest.

**Starting the container (flags for MI355X):** note the GPU selection — see the
`HIP_VISIBLE_DEVICES` warning in [§8.1](#81-mi355x-rocm-724). You must
scope GPUs with `--device`, **not** with `-e HIP_VISIBLE_DEVICES`.

The `docker run` lines below refer to two host directories by environment variable — set
them to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # training artifacts, mounted at /work
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

```bash
# Map the physical GPUs you want to their DRM nodes first:
#   for d in /sys/class/drm/renderD*; do echo "$d -> $(basename $(readlink -f $d/device))"; done
# e.g. physical GPU2 = 0000:a5:00.0 -> renderD144 + card17
#      physical GPU3 = 0000:dc:00.0 -> renderD152 + card25

docker run -d --name primus_mi355x \
  --device /dev/kfd \
  --device /dev/dri/renderD144 --device /dev/dri/card17 \
  --device /dev/dri/renderD152 --device /dev/dri/card25 \
  --group-add video --group-add render \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64G \
  -e HF_HOME=/root/.cache/huggingface \
  -v $HF_HOME:/root/.cache/huggingface \
  -v $OUTPUT_DIR:/work \
  -v $OUTPUT_DIR/data_root:/data \
  rocm/primus:v26.5 sleep infinity

# one required fix inside the container (see §8.1 "datasets too old"):
docker exec primus_mi355x pip install "datasets==4.3.0"
```

In-container versions as shipped: torch **2.12.0+rocm7.15.0a20260720**, transformers 4.55.0
(the launcher upgrades it to 4.57.6 at runtime), `datasets` 3.6.0 (**too old — must be
upgraded**). `/data` must be mounted: the checkpoint-conversion hook writes the converted
Megatron checkpoint to `/data/megatron_checkpoints/<model>`.

**Option 2 — pip wheel (ships `primus-cli` only):**

```bash
python -m venv primus-env && source primus-env/bin/activate
pip install "primus==26.5.0" --no-deps \
  --extra-index-url https://amd-agi.github.io/Primus/simple/

# fetch the pinned backend sources
primus-cli deps sync --dir ~/.cache/Primus/third_party
```

The wheel bundles the launcher, not the backends — it still starts training inside the
container. See `requirements_primus.txt` for all three paths including bare metal.

> **Arch note:** MI325X uses the MI300X configs (both `gfx942`); MI350X uses the MI355X
> configs (both `gfx950`). `--arch` in the launcher handles this mapping for you.

### NVIDIA

**Not supported — stated plainly, and now confirmed by execution on H100.** Primus targets
AMD GPUs on ROCm; there is no CUDA image, no CUDA install path, and no NVIDIA hardware
support upstream. If you are on NVIDIA GPUs, use `../nemo/`, `../deepspeed/`, or any of the
other trainers in this repo instead.

**Evidenced negative — 8×NVIDIA H100 80GB, driver 580.173.02, CUDA 13.0.**
This is the documented AMD-only outcome; see [§8.3](#83-nvidia-h100-cuda-130--not-supported-evidenced)
for the full transcript. In short, on an NVIDIA box:

- **No ROCm stack exists** (as required): `rocm-smi` not on PATH, `/dev/kfd` absent,
  `/opt/rocm` absent — only `NVIDIA H100 80GB HBM3` GPUs are present.
- **The launcher offers no NVIDIA target:** `train_llm_primus.py --arch` accepts only
  `{MI300X, MI325X, MI350X, MI355X}` (gfx942/gfx950); `configs/` ships only `MI300X/` and
  `MI355X/` YAMLs. There is no H100/NVIDIA config to select.
- **Running it anyway** warns `rocm-smi not found - Primus targets AMD Instinct GPUs on
  ROCm >= 7.0`, then tries to exec `./primus-cli container --image rocm/primus:v26.5 ...` —
  i.e. it routes into AMD's **ROCm** image, which contains ROCm-compiled PyTorch and
  gfx-targeted kernels that cannot execute on an H100. This is architectural, not a
  packaging gap: there is no NVIDIA path to fix.

## 3. Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_primus.py` loads it with `load_dotenv("dev.env")` and forwards it into the
container as `--env HF_TOKEN=...`. This matters more than usual here: the container does
**not** inherit your host environment, so a config that downloads weights or a tokenizer
from the Hub will fail without the explicit pass-through.

`dev.env` is git-ignored at the repo root. **Never commit a token**; rotate it on the Hub if
one ever lands in a commit.

You may also need `PRIMUS_WORKSPACE` set if your team's config templates reference it.

## 4. Data

This folder ships a working sample: `data/OTel_LLM_sample_10.jsonl` — 10 rows of this
repo's canonical chat JSONL, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

Each row also carries extra columns (`unmask`, `flow`, `source_id`, `source_repo`,
`source_spec_id`, `source_version` — the last of these null in most rows). `--data-path`
defaults to the shipped sample, so the launcher's mount-and-inject path is exercised out of
the box: in container mode the sample's directory is mounted at `/data` and the
in-container path is forwarded to the config. Swap in your own data with
`--data-path /path/to/train.jsonl`.

**The dataset-key caveat — RESOLVED, and the answer is "you cannot".** This was flagged as
an open question; running it on MI355X settled it. **`--data-path` is dead code on the
Megatron-Bridge post-training path, and the shipped `OTel_LLM_sample_10.jsonl` cannot be
used by it.** Three findings, all verified in `rocm/primus:v26.5`:

1. **The dataset is hard-coded to SQuAD.** The `qwen.qwen3` recipe builds its config in
   `_qwen3_finetune_common()`
   (`third_party/Megatron-Bridge/src/megatron/bridge/recipes/qwen/qwen3.py`), which ends in
   `dataset=default_squad_config(seq_length, packed_sequence)`. That function takes **no**
   dataset/path argument — there is no key you can set to change it. The
   `dataset: {dataset_name: ...}` block in the model YAML is *informational only*; upstream
   says so in a comment in `qwen3_30b_a3b.yaml`.
2. **Bad keys are silently swallowed, not rejected.** Primus calls the recipe through
   `auto_filter_and_call()` (`primus/backends/megatron_bridge/config_utils.py`), which
   catches `TypeError`, regexes the offending kwarg out of the error message, and retries up
   to 50 times. So `data_path=` / `dataset=` do not fail loudly — they are **dropped**, and
   training proceeds on SQuAD as if you had said nothing. Observed live:
   `⚠️  Retry 20: Removing invalid parameter 'dataset'`.
3. **The only real path keys are Megatron indexed-dataset prefixes**, not chat JSONL:
   `DATASET_PATH_KEYS = ("data_paths", "train_data_path", "valid_data_path",
   "test_data_path")` plus `data_args_path` / `per_split_data_args_path` — note `data_paths`
   is **plural**; the launcher's singular `data_path` is not among them.

**What this means in practice:** to post-train on your own data here you must either write a
custom recipe exposing a dataset argument, or pre-convert your JSONL to the format the
Bridge `HFDatasetConfig` pipeline expects. The same `⚠️ Retry` mechanism also silently drops
`recompute_granularity`, `recompute_method` and `recompute_num_layers` from the shipped
configs — so **the activation-recompute half of the OOM playbook in §9 does not take effect**
on the `qwen.qwen3` post-training recipe.

## 5. Run

**The command that trains on MI355X.** This runs `primus-cli`
directly *inside* the already-started container (`--mode direct` semantics), which is the
route that works — see [§8.1](#81-mi355x-rocm-724) for why the wrapper's
`--mode container` path is not the tested one.

```bash
docker exec -w /workspace/Primus \
  -e GPUS_PER_NODE=2 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=localhost -e MASTER_PORT=29740 \
  -e HF_HOME=/root/.cache/huggingface \
  primus_mi355x bash -lc \
  './runner/primus-cli direct --config /work/smoke/qwen3_600m_sft_smoke.yaml \
     -- train posttrain --config /work/smoke/qwen3_600m_sft_smoke.yaml'
```

`GPUS_PER_NODE` **defaults to 8** — set it to your GPU count or the launcher will try to
start 8 ranks. It also determines `HIP_VISIBLE_DEVICES`, which primus-cli overwrites
(§8.1).

**Smoke test** — prints the fully assembled `primus-cli` command without launching
anything:

```bash
python3 train_llm_primus.py --method sft --arch MI300X --dry-run
```

**Full run:**

```bash
# LoRA on MI355X, inside the ROCm container
nohup python3 train_llm_primus.py \
  --method lora --arch MI355X \
  --mode container --image rocm/primus:v26.5 \
  --data-path /path/to/train.jsonl \
  --train-iters 500 --global-batch-size 32 --micro-batch-size 4 \
  --seq-length 8192 --finetune-lr 1e-4 \
  > train_llm_primus.log 2>&1 &

tail -f train_llm_primus.log
```

Equivalent raw CLI, if you prefer to skip the wrapper:

```bash
./runner/primus-cli direct -- train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

**What "working" looks like:** `rocm-smi` prints your GPUs, the container starts, Primus
echoes the resolved config, Megatron-Bridge builds the model and reports the parallel layout
(TP/PP/CP sizes), then per-iteration lines with loss, learning rate, and throughput. LoRA
runs should show a far smaller trainable-parameter count than SFT — if it reports the full
parameter count, `peft: lora` did not take effect.

## 6. Arguments

Every flag `train_llm_primus.py` accepts:

| Arg | Default | Meaning |
|---|---|---|
| `--config` | `None` | Explicit Primus YAML path; overrides `--method`/`--arch` selection |
| `--method` | `sft` | `sft` (full fine-tune) or `lora` (adapters only) |
| `--arch` | `MI300X` | `MI300X` / `MI325X` / `MI350X` / `MI355X` — selects the tuned reference config |
| `--mode` | `container` | `container` (recommended), `direct` (bare metal / already inside a container), or `slurm` |
| `--image` | `rocm/primus:v26.5` | ROCm training image for container mode |
| `--cli` | `./primus-cli` | Path to `primus-cli`; a repo clone usually exposes `./runner/primus-cli` |
| `--data-path` | `data/OTel_LLM_sample_10.jsonl` | Training data; mounted into the container and forwarded to the config (see the dataset-key caveat) |
| `--train-iters` | `None` | Override `train_iters` in the YAML |
| `--global-batch-size` | `None` | Override `global_batch_size` |
| `--micro-batch-size` | `None` | Override `micro_batch_size` |
| `--seq-length` | `None` | Override `seq_length` |
| `--finetune-lr` | `None` | Override `finetune_lr` |
| `--log-file` | `None` | Write the training log here (important in container mode — see Output) |
| `--extra` | `[]` | Everything after this flag is forwarded verbatim to `primus-cli` |
| `--dry-run` | off | Print the command and exit |

Key config fields (in the YAML, under `modules.post_trainer.overrides`):

| Field | Meaning |
|---|---|
| `peft` | `"none"` for SFT, `lora` for LoRA |
| `finetune_lr`, `min_lr`, `lr_warmup_iters`, `lr_decay_iters` | LR schedule |
| `precision_config` | Typically `bf16_mixed` |
| `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `sequence_parallel` | Parallelism; raise TP/PP when the model does not fit |
| `recompute_granularity`, `recompute_method`, `recompute_num_layers` | Activation recompute — trade compute for memory |
| `train_iters`, `global_batch_size`, `micro_batch_size`, `seq_length` | Batching and sequence length |

## 7. Output

Checkpoints and logs go to the output directory configured in the trainer YAML. In container
mode, pass `--log-file` so the log is written to a mounted volume — otherwise it lands inside
the container at the Primus install directory (`site-packages/primus/logs` by default) and is
lost when the container exits.

> **Where the loss lines actually are (MI355X, tested):** *not* on stdout. Primus patches
> Megatron's `training_log` through `print_rank_last` + its own logger, so per-iteration
> lines land in the **last rank's `debug.log`**:
> `<workspace>/<work_group>/<user_name>/<exp_name>/logs/post_trainer/rank-<N-1>/debug.log`.
> `rank-0/info.log` does **not** contain them, and neither does the launcher's stdout. Tail
> this instead:
> ```bash
> tail -f output/local/local/<exp_name>/logs/post_trainer/rank-1/debug.log
> ```

Mount your data and output directories explicitly:

```bash
primus-cli container --image rocm/primus:v26.5 \
  --env HF_TOKEN="hf_xxx" \
  --volume /path/to/your/data:/data \
  -- --log_file /data/run.log \
  -- train posttrain --config /data/your/config.yaml
```

Monitoring integrates with MLflow and TraceLens if configured in the experiment YAML.

## 8. Hardware support & evidence

- **AMD-only.** Primus is AMD's training framework for AMD Instinct GPUs on ROCm.
  Upstream's prerequisites: ROCm drivers >= 7.0, Docker >= 24.0 with ROCm support, and
  ROCm-compatible Instinct GPUs (MI300 series and up). Source: the
  [AMD-AGI/Primus README](https://github.com/AMD-AGI/Primus), "Prerequisites".
- **Supported arches in this folder's configs:** gfx942 (MI300X / MI325X) and gfx950
  (MI350X / MI355X); the launcher maps `--arch` to the right config directory. AMD's
  MLPerf Training 6.0 examples run on MI355X.
- **Images:** `rocm/primus` on Docker Hub is the published training image for the
  Megatron-LM / TorchTitan / Megatron-Bridge backends; `rocm/jax-training:maxtext-v26.5`
  covers the MaxText backend. The `v26.5` tag was verified live on
  [Docker Hub](https://hub.docker.com/r/rocm/primus/tags) (alias
  `v26.5-pytorch2.12-te2.15`, same digest).
- **NVIDIA is NOT supported.** No CUDA image or install path exists upstream. The
  launcher's `rocm-smi` preflight will warn (not block) on non-ROCm hosts so `--dry-run`
  still works anywhere. **Confirmed, not just documented:** the whole stack that trained
  below is ROCm-native (`torch 2.12.0+rocm7.15`, hipBLASLt, Primus-Turbo/Triton kernel
  patches, RCCL). There is no CUDA fallback to accidentally rely on.
- **Other hardware (upstream claims — not verified here):** none. Primus is AMD-only by
  design — upstream claims nothing beyond AMD Instinct GPUs on ROCm. No NVIDIA CUDA,
  Intel Gaudi/XPU, Apple MPS, Ascend NPU, TPU, Trainium, or CPU training path exists
  or is claimed.

### 8.1 MI355X (ROCm 7.2.4)

**This path works with changes.** The Primus + Megatron-Bridge stack trains correctly on
gfx950. The folder's *own configs* were broken and are now fixed; LoRA remains broken
inside this image for a reason upstream of this folder.

**Host / route.** Single node, 8x MI355X (gfx950, 288 GB), ROCm 7.2.4, Ubuntu, Python
3.12.3, Docker 29.7.2 — **2 GPUs used** (physical 2 and 3). Route chosen: **the
vendor container**, `rocm/primus:v26.5`, digest
`sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1` (14.1 GB
compressed, 54.8 GB on disk). This is the upstream-recommended path and the only one that
carries the built backends; the pip wheel ships the launcher only. Versions in-image:
torch **2.12.0+rocm7.15.0a20260720**, **primus 0.2.0** (checkout `b511d1b`),
**Megatron-Bridge `9577b12`**, **Megatron-LM `d3528a2`**, transformers 4.55.0→4.57.6.

**Run command:** see [§5](#5-run). Model: **Qwen3-0.6B** via the `qwen.qwen3` recipe's
`qwen3_600m_finetune_config` flavor — the *same recipe* as the shipped Qwen3-32B configs,
smallest flavor. 32B was not used because its weights are a ~65 GB download and the point
was to exercise the framework, not the parameter count. Dataset: SQuAD (forced by the
recipe — see §4). TP 1 / PP 1 / CP 1, bf16_mixed, GBS 8, MBS 1, seq 1024, 2 ranks.

**Expected output — SFT, 600/600 iterations, exit code 0:**

```
iteration        1/     600 | consumed samples:    8 | elapsed time per iteration (ms): 10059.3 | lm loss: 9.745878E+00 | grad norm: 458.126 | number of nan iterations:   0 |
iteration      100/     600 | consumed samples:  800 | elapsed time per iteration (ms):   268.6 | lm loss: 3.537978E-01 | grad norm:  64.572 | number of nan iterations:   0 |
iteration      400/     600 | consumed samples: 3200 | elapsed time per iteration (ms):   240.8 | lm loss: 3.696553E-01 | grad norm:  39.788 | number of nan iterations:   0 |
iteration      600/     600 | consumed samples: 4800 | elapsed time per iteration (ms):   276.0 | lm loss: 8.565324E-01 | grad norm: 123.243 | number of nan iterations:   0 |
validation loss at iteration 600 on validation set | lm loss value: 3.841884E-01 | lm loss PPL: 1.468422E+00 |
```

Loss is finite and falls from 9.75 to <1 with zero NaN and zero skipped iterations;
steady-state ~240–280 ms/iter after a ~10 s first step. `rocm-smi` mid-run, both assigned
GPUs saturated and every other GPU idle:

```
Device  Node  Temp    Power    SCLK      VRAM%  GPU%
2       4     43.0°C  428.0W   2391Mhz   5%     98%     <- 14.57 GiB
3       5     47.0°C  406.0W   2390Mhz   5%     99%     <- 15.66 GiB
```

**Bug 1 — every shipped config fails to parse (FIXED here).** `workspace` is a *required*
top-level key (`primus/core/launcher/parser.py::parse_meta_info` asserts on `work_group`,
`user_name`, `exp_name`, **`workspace`**). All four configs omitted it:

```
AssertionError: Failed to find key(workspace) in namespace(PrimusConfig)
```

Every upstream reference config carries `workspace: ${PRIMUS_WORKSPACE:./output}`. Added to
all four.

**Bug 2 — `sequence_parallel: true` with `tensor_model_parallel_size: 1` is fatal (FIXED
here).** Megatron rejects it outright:

```
Error during post-training: Cannot use sequence parallelism without tensor parallelism
```

Three of the four configs had TP 1 + SP true. **Every** upstream reference config
(MI300X and MI355X, SFT and LoRA) sets `sequence_parallel: false`. Corrected to `false` in
all four; the MI300X SFT config keeps TP 2, where SP is at least legal, but upstream still
sets it false, so it now matches upstream.

**Quirk — `primus-cli` overwrites `HIP_VISIBLE_DEVICES`; scope GPUs with `--device`.**
This one is a trap. `runner/helpers/envs/base_env.sh:130` does:

```bash
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES
```

It **unconditionally clobbers** whatever you set, so `docker run -e HIP_VISIBLE_DEVICES=2,3`
is silently ignored and you train on physical GPU 0,1. Verified by allocation probe (40 GB
on the container's `cuda:0` moved host `rocm-smi` GPU[2] from 0.28→38.01 GiB only *after*
switching to device scoping). The fix is to expose only the GPUs you want, as in §2. Also
note `rocm-smi --showpids`' `GPU(s)` column is *not* the physical index — use
`--showmeminfo vram` per index to attribute usage.

**Quirk — never set `CUDA_VISIBLE_DEVICES` alongside `HIP_VISIBLE_DEVICES`.** Setting both
in this container is an immediate hard abort, not a warning:

```
F0819 11:25:26.566322 agent.cpp:245] Conflicting visibility of agent-2 between
HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES ... Aborted (core dumped)
```

**Quirk — the image's `datasets` is too old for the Hub.** `datasets==3.6.0` cannot parse
SQuAD's current dataset card: `ValueError: Feature type 'List' not found` (the `List`
feature type arrived in `datasets` 4.x). Fix: `pip install datasets==4.3.0` in the
container.

**Quirk — the container stalls on the HF CDN.** The HF *API* is reachable from inside the
container, but blob downloads hang mid-transfer (`.incomplete` files). Pre-seed the cache
**from the host** and mount it (`-v <cache>:/root/.cache/huggingface`). Watch file
ownership: the container runs as root and will leave root-owned lock/blob files that then
block host-side writes.

**Quirk — the launcher pip-installs at runtime.** The `00_install_requirements.sh` framework
hook installs Megatron-Bridge deps on every run (it upgraded transformers 4.55.0→4.57.6 and
pulled timm/open-clip/qwen-vl-utils). The container therefore needs network egress, and the
first run is slower. `01_convert_checkpoints.sh` then converts HF→Megatron into
`/data/megatron_checkpoints/<model>`, so `/data` must be a mounted, writable volume.

**LoRA does NOT work in this image (`peft: lora`).** The adapters *are* built —
`Adding lora to: decoder.layers.N.self_attention.linear_qkv` for every layer, and the
optimizer is constructed over them — but loading the converted base checkpoint then fails.
The failure walks through three distinct errors as you work around each:

1. Default: `KeyError: "decoder.layers.self_attention.linear_proj.adapter.linear_in.weight
   from model not in state dict"` — raised by
   `megatron/core/dist_checkpointing/strategies/torch.py::_validate_global_shapes`. Primus
   ships a patch for exactly this class of problem
   (`megatron.patch.apply_factory_merges_tolerant`, "tolerate missing keys ... LoRA adapter
   factories not in base checkpoint") but it does not cover this code path.
2. With `checkpoint.dist_ckpt_strictness: ignore_all`: `CheckpointingException: Global shape
   mismatch for loaded (torch.Size([596049920])) and expected ((17432576,)) tensor for key
   optimizer.distributed...` — i.e. it tries to load the *full-model* optimizer state into
   the LoRA parameter set.
3. Additionally with `checkpoint.load_optim: false` / `load_rng: false`:
   `RuntimeError: Training execution failed: 'model'`.

Nested overrides *do* work (`↳ Set config_container.checkpoint.dist_ckpt_strictness =
ignore_all` is logged), so this is a genuine PEFT-checkpoint-loading defect in the
v26.5 image, not a config error. **SFT is the working post-training method here.**
Time-boxed after three attempts; not pursued further.

**Not tested:** Qwen3-32B at the shipped scale, multi-node, `slurm` mode, MI300X/gfx942
(no gfx942 hardware on this host), the MaxText/JAX image, and the wrapper's own
`--mode container` path (see §9).

### 8.2 8-GPU run (8x MI355X, ROCm 7.2.4)

**This scales cleanly to 8 GPUs as pure data parallel (DP 8), and only as
pure data parallel.** The identical §8.1 SFT recipe ran 600/600 iterations on all eight
MI355X at **~74 ms/iter** against the 2-GPU run's ~240–280 ms/iter — a **~3.5x** speedup
at fixed global batch, about **88% of the ideal 4x**. Zero NaN, zero skipped iterations,
exit code 0. **Tensor/pipeline parallelism does NOT work on this path** — not for lack of
hardware, but because the image's checkpoint-conversion hook only ever emits a TP1/PP1
checkpoint (details below). LoRA is still broken at 8 GPUs, exactly as at 2.

**What changed vs §8.1:** only `GPUS_PER_NODE` (2 → 8) and exposing all 8 GPUs to the
container. No config change, no env var, no batch-size change, no OOM, no hang. The model
preset had to be copied into the Primus tree because this was a *fresh* container (see the
trap below).

**Container (all 8 GPUs).** With every GPU in play there is no reason to hand-pick DRM
nodes as §2 does — pass the whole `/dev/dri` and let `GPUS_PER_NODE=8` line up with the
`HIP_VISIBLE_DEVICES` clobber described in §8.1, which now works *for* you rather than
against you:

```bash
docker run -d --name primus_mi355x_gpu8 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64G \
  -e HF_HOME=/root/.cache/huggingface \
  -v $HF_HOME:/root/.cache/huggingface \
  -v $OUTPUT_DIR:/work \
  -v $OUTPUT_DIR/data_root:/data \
  rocm/primus:v26.5 sleep infinity

docker exec primus_mi355x_gpu8 pip install "datasets==4.3.0"   # §8.1 "datasets too old"
```

`torch.cuda.device_count()` reports **8** inside this container.

**Train command** — identical to §5 except `GPUS_PER_NODE=8`:

```bash
docker exec -w /workspace/Primus \
  -e GPUS_PER_NODE=8 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=localhost -e MASTER_PORT=29752 \
  -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \
  primus_mi355x_gpu8 bash -lc \
  './runner/primus-cli direct --config /work/gpu8/qwen3_600m_sft_dp8.yaml \
     -- train posttrain --config /work/gpu8/qwen3_600m_sft_dp8.yaml'
```

The launcher confirms the scoping — `GPUS_PER_NODE=8` →
`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, and every rank logs as `rank-N/8`:

```
[Primus:Env] rank=0, world_size=8, local_rank=0, master=localhost:29752
```

**Parallelism actually used: TP 1 / PP 1 / CP 1 → DP 8**, bf16_mixed, GBS 8, MBS 1,
seq 1024, `qwen3_600m_finetune_config`, SQuAD. Because global batch stays at 8, each rank
drops from 4 micro-batches per iteration (DP 2) to 1 (DP 8) — that 4x reduction in
per-rank work is exactly what the measured 3.5x speedup is cashing in.

**Expected output — SFT, 600/600 iterations, exit code 0** (from
`output/local/local/qwen3_600m_sft_dp8/logs/post_trainer/rank-7/debug.log`; note there are
now eight `rank-*` dirs and the loss lines are on **rank-7**, the last rank):

```
iteration        1/     600 | consumed samples:    8 | elapsed time per iteration (ms): 11624.2 | lm loss: 9.745878E+00 | grad norm: 458.126 | number of skipped iterations: 0 | number of nan iterations: 0 |
iteration      100/     600 | consumed samples:  800 | elapsed time per iteration (ms):    72.5 | lm loss: 3.561614E-01 | grad norm:  67.602 | number of skipped iterations: 0 | number of nan iterations: 0 |
iteration      400/     600 | consumed samples: 3200 | elapsed time per iteration (ms):    74.1 | lm loss: 3.823107E-01 | grad norm:  39.308 | number of skipped iterations: 0 | number of nan iterations: 0 |
iteration      600/     600 | consumed samples: 4800 | elapsed time per iteration (ms):    76.0 | lm loss: 8.372381E-01 | grad norm: 123.824 | number of skipped iterations: 0 | number of nan iterations: 0 |
validation loss at iteration 600 on validation set | lm loss value: 3.852793E-01 | lm loss PPL: 1.470025E+00 |
```

All 600 iteration lines report `number of nan iterations: 0` and
`number of skipped iterations: 0`. Steady-state ms/iter over iterations 6–600:
**median 73.9, p10 72.3, p90 77.8** — a very tight distribution. Twelve outliers: the
11.6 s first iteration (kernel autotune / warmup) and ~11 periodic 2.5–5 s dataloader
stalls. Whole run, including startup and the runtime pip installs, took **4 m 27 s**.

**8-GPU proof — host `rocm-smi` mid-run.** All eight physical GPUs
are lit, each drawing ~370–405 W with 15.7–19.1 GiB resident — not a 2-GPU run wearing an
8-rank hat:

```
Device  Node  Temp    Power    SCLK     VRAM%  GPU%     VRAM used
0       2     45.0°C  375.0W   2392Mhz  5%     100%     15.75 GiB
1       3     44.0°C  369.0W   2400Mhz  5%     100%     16.87 GiB
2       4     43.0°C  378.0W   2406Mhz  6%     100%     17.37 GiB
3       5     47.0°C  405.0W   2206Mhz  6%      68%     19.09 GiB
4       6     47.0°C  373.0W   2400Mhz  5%     100%     16.65 GiB
5       7     47.0°C  372.0W   2397Mhz  5%     100%     16.95 GiB
6       8     46.0°C  375.0W   2398Mhz  6%     100%     17.42 GiB
7       9     46.0°C  378.0W   2400Mhz  6%     100%     17.55 GiB
```

VRAM% reads 5–6% only because each MI355X has 288 GB; the absolute footprint is ~16–19 GiB
per rank, up slightly from the ~14.6–15.7 GiB of the 2-GPU run (RCCL buffers for an 8-way
all-reduce).

**Scaling observation.** 2 GPUs → 8 GPUs at fixed GBS 8: **~262 ms/iter → 73.9 ms/iter,
≈3.5x on 4x the GPUs (≈88% efficiency)**. The lost 12% is the 8-way gradient all-reduce
plus a per-rank micro-batch count of 1, which leaves nothing to overlap the collective
against. Final quality is unchanged — validation lm loss 3.852793E-01 / PPL 1.470 at DP 8
versus 3.841884E-01 / PPL 1.468 at DP 2, i.e. the same run to within noise, which is what
you want from a pure-data-parallel scale-out at constant global batch.

**Bug 3 — TP/PP is unusable on this recipe: the conversion hook only emits TP1/PP1
(NOT fixed — upstream limitation).** The obvious use of a bigger world size is 3D
parallelism, so TP 2 x PP 2 (→ DP 2) was tried. The model *builds* correctly — PP 2 splits
Qwen3-0.6B's 28 layers into two 14-layer stages — and then checkpoint load dies:

```
Error during post-training: (TP, PP) mismatch after resume ((2, 2) vs (1, 1) from
checkpoint): not supported for DistributedOptimizer with sharding type dp_reshardable.
Please use `checkpoint_config.fully_parallel_save=True` for checkpoint saving.
```

The cause is structural, not a config mistake. `01_convert_checkpoints.sh` converts HF →
Megatron with

```bash
python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model "${HF_PATH}" --megatron-path "${MEGATRON_PATH}"
```

— **no parallelism arguments at all**, so the base checkpoint is *always* TP1/PP1, and
deleting the cache to force re-conversion cannot help. The guard is in
`Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1393`, and it is reached
only when `not cfg.checkpoint.finetune and cfg.checkpoint.load_optim` — so
`checkpoint.finetune: true` should escape it. **It does not, and the reason is worth
knowing (see below).** Net effect: on the `qwen.qwen3` post-training path in
`rocm/primus:v26.5`, the only usable layout is TP 1 / PP 1, i.e. **DP = GPUS_PER_NODE**.
To use TP/PP you would have to produce the base checkpoint yourself at the target
parallelism, or save it with `fully_parallel_save=True`.

**Quirk — nested overrides are echoed but then silently dropped on the recipe path.**
§8.1 records that nested overrides "do work". At 8 GPUs this turns out to be only half
true, and the distinction matters. `checkpoint.finetune: true` is logged as applied twice
during config assembly:

```
checkpoint.finetune : True (bool)
```

but the recipe call then strips it via the same `auto_filter_and_call()` retry loop that
already eats `dataset` and `recompute_*` (§4), and the value that actually reaches training
is the default:

```
⚠️  Retry 21: Removing invalid parameter 'checkpoint.finetune' (22 params remaining)
   error_msg: _qwen3_finetune_common() got an unexpected keyword argument 'checkpoint.finetune'
✅ Successfully called qwen3_600m_finetune_config() after removing 25 invalid parameters
checkpoint.finetune : False (bool)
```

So a nested override only survives if the *recipe function* accepts it as a keyword.
Anything else is echoed into the config dump, dropped, and reverted — which looks exactly
like a working override right up until behaviour disagrees with you. **Always grep the log
for `Removing invalid parameter '<your key>'` before trusting an override.**

**Trap — the model preset must live inside the Primus tree, not next to your config.**
A fresh container fails in ~2 seconds:

```
FileNotFoundError: [Primus] Preset 'qwen3_600m.yaml' not found for framework
'megatron_bridge' in 'models'.
Expected: /workspace/Primus/primus/configs/models/megatron_bridge/qwen3_600m.yaml
```

`modules.post_trainer.model:` is a **preset name** resolved against
`primus/configs/models/<framework>/`, not a path relative to the experiment YAML. Copy it
in before running:

```bash
docker exec primus_mi355x_gpu8 \
  cp /work/gpu8/qwen3_600m.yaml /workspace/Primus/primus/configs/models/megatron_bridge/
```

**LoRA is still broken at 8 GPUs — same failure, unchanged by world size.** `peft: lora`
at DP 8 builds all 112 adapters (`Adding lora to: decoder.layers.N...`), constructs the
optimizer over them, then dies at base-checkpoint load with byte-for-byte the §8.1
failure #1:

```
KeyError: "decoder.layers.self_attention.linear_proj.adapter.linear_in.weight from model
not in state dict: ['decoder.final_layernorm._extra_state/shard_0_1', ...]"
  at megatron/core/dist_checkpointing/strategies/torch.py:600 _validate_global_shapes
```

Confirmed and not pursued further — **SFT remains the working post-training method.**

**Disk.** Checkpointing was disabled (`save_interval: 100000`) and **no weight files were
written** — the entire 8-GPU run left 11 MB of logs under `$OUTPUT_DIR`, and free disk was
unchanged before and after. The `rocm/primus:v26.5` image was reused, not re-pulled.

### 8.3 NVIDIA H100 (CUDA 13.0) — NOT SUPPORTED, evidenced

Tested on a cross-vendor verification box: **8×NVIDIA H100 80GB HBM3, driver 580.173.02,
CUDA 13.0, Python 3.12.3**. This is the **correct, expected negative** — Primus is AMD-only
by design, and this result is backed by execution on real NVIDIA hardware. **No NVIDIA GPU
was consumed** beyond running the pure-Python launcher; the co-tenant GPUs 0–3 were never
touched.

**1. The ROCm prerequisites are absent (as they must be on NVIDIA):**

```
rocm-smi : NOT FOUND (not on PATH)
/dev/kfd : ABSENT (no AMD compute node)
/opt/rocm: ABSENT
GPUs     : NVIDIA H100 80GB HBM3   (nvidia-smi)
```

**2. The launcher exposes no NVIDIA target.** `train_llm_primus.py --help`:

```
--arch {MI300X,MI325X,MI350X,MI355X}     # gfx942 / gfx950 only — no NVIDIA option
--mode {container,direct,slurm}
--image IMAGE                            # default rocm/primus:v26.5 (a ROCm image)
```

`configs/` ships only `MI300X/` and `MI355X/` YAMLs — there is no H100/NVIDIA config.

**3. Running it on the H100 box routes straight into the ROCm image and stops:**

```
WARNING - rocm-smi not found - Primus targets AMD Instinct GPUs on ROCm >= 7.0
INFO - method=sft arch=MI355X mode=container
INFO - config=configs/MI355X/qwen3_32b_sft_posttrain.yaml
INFO - launching: ./primus-cli container --image rocm/primus:v26.5 ... train posttrain ...
```

The launcher hands off to AMD's `rocm/primus:v26.5` container — ROCm-compiled PyTorch and
gfx-targeted kernels that cannot execute on an H100. There is nothing to "port": the entire
training stack is the ROCm image. **Primus is AMD-only by design, confirmed on NVIDIA
hardware. Use `../nemo/`, `../megatron/`, `../deepspeed/`, or `../openrlhf/` on H100.**

> **Note (launcher hygiene, vendor-neutral):** in `--mode container` the launcher builds a
> `--env HF_TOKEN=<value>` argument and logs the full command at INFO, echoing the token in
> plaintext. Harmless on a private host but worth redacting in shared logs; unrelated to the
> NVIDIA result.

## 9. Notes

- **SFT vs LoRA is one config key.** `peft: "none"` trains everything; `peft: lora` trains
  adapters. The knock-on effects are what the separate reference configs encode: LoRA drops
  to TP 1 and runs a 4x larger global batch, and wants a much higher LR (`1e-4`–`5e-4`
  versus `5e-6`–`1e-5` for SFT). Copying an SFT learning rate into a LoRA run is the classic
  way to get a model that barely moves.
- **Arch-specific configs are not cosmetic.** gfx950 (MI355X) has more headroom than gfx942
  (MI300X), so the reference configs differ in TP and micro-batch size. Using the MI300X
  config on MI355X works but leaves throughput on the table; the reverse OOMs.
- **Megatron-Bridge is the post-training backend.** Primus's other backends (Megatron-LM,
  TorchTitan, MaxText) are aimed at pretraining. `train posttrain` specifically routes
  through Megatron-Bridge, which is also what `../megatron/` in this repo builds on —
  the mental model transfers.
- **Plan before you allocate.** Primus ships a projection tool that estimates parallelism and
  memory fit before you take a cluster, plus a tuning agent that searches configs
  automatically. On a large run these are worth using before burning an allocation.
- **The ecosystem is three layers.** Primus-LM (this framework) sits on Primus-Turbo
  (fused kernels: FlashAttention, GEMM, collectives) and optionally under Primus-SaFE
  (cluster orchestration, fault tolerance). You only interact with Primus-LM directly.
- **OOM playbook.** For SFT: raise `tensor_model_parallel_size`, then lower `micro_batch_size`
  or `seq_length`, then enable activation recompute. For LoRA: first confirm `peft: lora`
  actually applied, then lower `micro_batch_size`. **Caveat on this image:** on the
  `qwen.qwen3` post-training path neither of the first two levers is actually available —
  raising TP/PP fails at checkpoint load because the conversion hook only emits a TP1/PP1
  checkpoint, and `recompute_*` is silently dropped by the recipe's kwarg filter. See
  [§8.2](#82-8-gpu-run-8x-mi355x-rocm-724) and §4. That leaves
  `micro_batch_size` and `seq_length` as the only working knobs here.
