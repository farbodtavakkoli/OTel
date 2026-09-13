# `training/llm/primus` — AMD Primus post-training (SFT + LoRA on ROCm)

## 1. Overview & when to use

Post-training (SFT and LoRA) on **AMD Instinct GPUs** with
[Primus](https://github.com/AMD-AGI/Primus) (Primus-LM), AMD's training framework for
large-scale foundation models on ROCm. Primus unifies several backends —
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[TorchTitan](https://github.com/pytorch/torchtitan), JAX MaxText, and Megatron-Bridge —
under one configuration system, with ROCm-optimized kernels underneath.

This is the **AMD hardware path** for this repo. Post-training runs through the
**Megatron-Bridge** backend via the `train posttrain` subcommand.

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

> **Coverage:** SFT through Primus + Megatron-Bridge in `rocm/primus:v26.5` is verified on a
> single node of AMD Instinct MI355X (gfx950, ROCm 7.2.4). Three limitations of that image
> are live and documented below: **LoRA (`peft: lora`) cannot load the base checkpoint**,
> **TP/PP is unusable** (the only layout is DP = `GPUS_PER_NODE`), and the launcher's
> `--data-path` is **dead code** on the Megatron-Bridge post-training path (§4). Untested:
> multi-node, `slurm` mode, MI300X/gfx942.

## 2. Install

### AMD / ROCm (the only supported path)

**The container route is the verified one.** Image `rocm/primus:v26.5`, digest
`sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1` — **54.8 GB on
disk**, check free space before pulling. The image ships the whole stack pre-built at
`/workspace/Primus`, so **you do not need to clone anything** — the `git clone` below is
only for reading sources on the host.

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

**Starting the container (flags for MI355X):** you must scope GPUs with `--device`,
**not** with `-e HIP_VISIBLE_DEVICES`. `primus-cli` unconditionally clobbers
`HIP_VISIBLE_DEVICES` with `seq -s, 0 $((GPUS_PER_NODE - 1))`
(`runner/helpers/envs/base_env.sh`), so `-e HIP_VISIBLE_DEVICES=2,3` is silently ignored
and you train on physical GPUs 0,1 instead.

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

# one required fix inside the container (the image's datasets is too old — see Notes):
docker exec primus_mi355x pip install "datasets==4.3.0"
```

When you want **all** GPUs there is no reason to hand-pick DRM nodes — pass the whole
`/dev/dri` and let `GPUS_PER_NODE` line up with the `HIP_VISIBLE_DEVICES` rewrite:

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

docker exec primus_mi355x_gpu8 pip install "datasets==4.3.0"
```

Confirm with `torch.cuda.device_count()` inside the container before launching.

The image ships `datasets` 3.6.0, which is **too old and must be upgraded** (see Notes).
`/data` must be mounted: the checkpoint-conversion hook writes the converted Megatron
checkpoint to `/data/megatron_checkpoints/<model>`.

**Option 2 — pip wheel (ships `primus-cli` only):**

```bash
python -m venv primus-env && source primus-env/bin/activate
pip install "primus==26.5.0" --no-deps \
  --extra-index-url https://amd-agi.github.io/Primus/simple/

# fetch the pinned backend sources
primus-cli deps sync --dir ~/.cache/Primus/third_party
```

The wheel bundles the launcher, not the backends — it still starts training inside the
container. `requirements_primus.txt` has all three paths including bare metal.

> **Arch note:** MI325X uses the MI300X configs (both `gfx942`); MI350X uses the MI355X
> configs (both `gfx950`). `--arch` in the launcher handles this mapping for you.

### NVIDIA

**Not supported.** There is no CUDA image and no CUDA install path; `--arch` accepts only
`{MI300X, MI325X, MI350X, MI355X}`. On NVIDIA GPUs use `../nemo/`, `../megatron/`,
`../deepspeed/`, or `../openrlhf/` instead.

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

`--data-path` defaults to the shipped sample; swap in your own data with
`--data-path /path/to/train.jsonl`.

**The dataset-key caveat — you cannot point this recipe at your own JSONL.** On the
Megatron-Bridge post-training path `--data-path` is **dead code**: the `qwen.qwen3` recipe
hard-codes `default_squad_config(...)` and exposes no dataset argument, and Primus's
`auto_filter_and_call()` **silently drops** unknown keys (`⚠️  Retry 20: Removing invalid
parameter 'dataset'`) instead of failing. To train on your own data you must write a custom
recipe exposing a dataset argument, or pre-convert your JSONL to what the Bridge
`HFDatasetConfig` pipeline expects.

The same filter also drops `recompute_granularity` / `recompute_method` /
`recompute_num_layers`, so **the activation-recompute half of the OOM playbook in §9 does
not take effect** here. Always grep the log for `Removing invalid parameter '<your key>'`
before trusting an override.

## 5. Run

**The command that trains on MI355X.** This runs `primus-cli`
directly *inside* the already-started container (`--mode direct` semantics), which is the
verified route; the wrapper's own `--mode container` path is untested.

**Copy the model preset into the Primus tree first.** `modules.post_trainer.model:` is a
**preset name** resolved against `primus/configs/models/<framework>/`, not a path relative
to your experiment YAML, so a fresh container fails in ~2 seconds with
`FileNotFoundError: [Primus] Preset '<name>.yaml' not found for framework
'megatron_bridge' in 'models'`:

```bash
docker exec primus_mi355x \
  cp /work/<your>/qwen3_600m.yaml /workspace/Primus/primus/configs/models/megatron_bridge/
```

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
start 8 ranks. It also determines `HIP_VISIBLE_DEVICES`, which primus-cli overwrites (see
§2). Each rank logs as `[Primus:Env] rank=N, world_size=<GPUS_PER_NODE>, ...`, which is the
quickest confirmation that the scoping took. Scaling 2 → 8 GPUs needs nothing else: no
config change, no env var, no batch-size change.

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

**What "working" looks like:** the container starts, Primus echoes the resolved config,
Megatron-Bridge builds the model and reports the parallel layout, then per-iteration lines
(in the **last rank's** `debug.log` — see §7):

```
iteration        1/     600 | consumed samples:    8 | lm loss: 9.745878E+00 | grad norm: 458.126 | number of nan iterations:   0 |
iteration      600/     600 | consumed samples: 4800 | lm loss: 8.565324E-01 | grad norm: 123.243 | number of nan iterations:   0 |
validation loss at iteration 600 on validation set | lm loss value: 3.841884E-01 | lm loss PPL: 1.468422E+00 |
```

LoRA runs should show a far smaller trainable-parameter count than SFT — if it reports the
full parameter count, `peft: lora` did not take effect.

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

> **Where the loss lines actually are:** *not* on stdout. Primus patches
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

## 8. Hardware support

- **AMD-only.** Prerequisites: ROCm drivers >= 7.0, Docker >= 24.0 with ROCm support, and a
  ROCm-compatible Instinct GPU (MI300 series and up).
- **Supported arches in this folder's configs:** gfx942 (MI300X / MI325X) and gfx950
  (MI350X / MI355X); the launcher maps `--arch` to the right config directory.
- **Images:** `rocm/primus` is the training image for the Megatron-LM / TorchTitan /
  Megatron-Bridge backends; `rocm/jax-training:maxtext-v26.5` covers the MaxText backend.
- **NVIDIA is not supported.** The launcher's `rocm-smi` preflight warns (not blocks) on
  non-ROCm hosts, so `--dry-run` still works anywhere.

### 8.1 Known limitations of `rocm/primus:v26.5`

These are defects in the image/recipe, not configuration mistakes.

**LoRA (`peft: lora`) cannot load the base checkpoint.** The adapters are built, then the
load fails with `KeyError: "...adapter.linear_in.weight from model not in state dict"`;
`checkpoint.dist_ckpt_strictness: ignore_all` only moves it to a `CheckpointingException`
shape mismatch, and `checkpoint.load_optim: false` / `load_rng: false` to
`RuntimeError: Training execution failed: 'model'`. **SFT is the working post-training
method in this image.**

**TP/PP is unusable — the conversion hook only emits TP1/PP1.** TP 2 × PP 2 builds and then
dies at checkpoint load:

```
Error during post-training: (TP, PP) mismatch after resume ((2, 2) vs (1, 1) from
checkpoint): not supported for DistributedOptimizer with sharding type dp_reshardable.
Please use `checkpoint_config.fully_parallel_save=True` for checkpoint saving.
```

`01_convert_checkpoints.sh` passes no parallelism arguments, so the base checkpoint is
*always* TP1/PP1 and clearing the cache cannot help. The only usable layout here is
**TP 1 / PP 1, i.e. DP = `GPUS_PER_NODE`**; to use TP/PP, produce the base checkpoint
yourself at the target parallelism or save it with `fully_parallel_save=True`.

**Nested overrides are echoed and then silently dropped unless the recipe function accepts
them as keywords** — `checkpoint.finetune: true` is logged as applied, then stripped by the
same `auto_filter_and_call()` retry loop that eats `dataset` and `recompute_*` (§4).
**Always grep the log for `Removing invalid parameter '<your key>'` before trusting an
override.**

### 8.2 Writing a new config

Two mistakes break every run before training starts; both were present in this folder's
configs and are fixed here, but they recur in any hand-written config:

- **`workspace` is a required top-level key.** `primus/core/launcher/parser.py::parse_meta_info`
  asserts on `work_group`, `user_name`, `exp_name` **and `workspace`**; omitting it gives
  `AssertionError: Failed to find key(workspace) in namespace(PrimusConfig)`. Every upstream
  reference config carries `workspace: ${PRIMUS_WORKSPACE:./output}`.
- **`sequence_parallel: true` with `tensor_model_parallel_size: 1` is fatal** —
  `Cannot use sequence parallelism without tensor parallelism`. Every upstream reference
  config sets `sequence_parallel: false`, including where TP > 1.

## 9. Notes

- **SFT vs LoRA is one config key.** `peft: "none"` trains everything; `peft: lora` trains
  adapters — and wants a much higher LR (`1e-4`–`5e-4` versus `5e-6`–`1e-5` for SFT). The
  separate reference configs already encode this; do not copy an SFT LR into a LoRA run.
- **Use the config matching your arch.** The MI300X (gfx942) and MI355X (gfx950) configs
  differ in TP and micro-batch size; running the MI355X config on MI300X OOMs.
- **Never set `CUDA_VISIBLE_DEVICES` alongside `HIP_VISIBLE_DEVICES`.** In this container
  that is an immediate hard abort, not a warning:
  `agent.cpp:245] Conflicting visibility of agent-2 between HIP_VISIBLE_DEVICES and
  CUDA_VISIBLE_DEVICES ... Aborted (core dumped)`. Scope GPUs with `--device` (§2).
- **`rocm-smi --showpids`' `GPU(s)` column is not the physical index** — use
  `rocm-smi --showmeminfo vram` per index to attribute usage to a GPU.
- **The image's `datasets` is too old for the Hub.** `datasets==3.6.0` cannot parse SQuAD's
  current dataset card (`ValueError: Feature type 'List' not found`; the `List` feature type
  arrived in `datasets` 4.x). Fix: `pip install datasets==4.3.0` in the container (§2).
- **Pre-seed the Hub cache from the host.** The HF *API* is reachable from inside the
  container but blob downloads can hang mid-transfer, leaving `.incomplete` files. Download
  on the host and mount the cache (`-v <cache>:/root/.cache/huggingface`). The container
  runs as root, so watch for root-owned lock/blob files that then block host-side writes.
- **The launcher pip-installs at runtime.** `00_install_requirements.sh` installs
  Megatron-Bridge deps on every run, so the container needs network egress.
  `01_convert_checkpoints.sh` then converts HF→Megatron into
  `/data/megatron_checkpoints/<model>`, so `/data` must be a mounted, writable volume.
- **Token hygiene.** In `--mode container` the launcher builds a `--env HF_TOKEN=<value>`
  argument and logs the full command at INFO, echoing the token in plaintext. Harmless on a
  private host; redact it in shared logs.
- **OOM playbook.** For SFT: raise `tensor_model_parallel_size`, then lower `micro_batch_size`
  or `seq_length`, then enable activation recompute. For LoRA: first confirm `peft: lora`
  actually applied, then lower `micro_batch_size`. **Caveat on this image:** on the
  `qwen.qwen3` post-training path neither of the first two levers is actually available —
  raising TP/PP fails at checkpoint load because the conversion hook only emits a TP1/PP1
  checkpoint, and `recompute_*` is silently dropped by the recipe's kwarg filter. See
  [§8.1](#81-known-limitations-of-rocmprimusv265) and §4. That leaves
  `micro_batch_size` and `seq_length` as the only working knobs here.
