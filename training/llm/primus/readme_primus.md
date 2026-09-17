# `training/llm/primus` — AMD Primus post-training (SFT + LoRA on ROCm)

Post-training (SFT and LoRA) on AMD Instinct GPUs with
[Primus](https://github.com/AMD-AGI/Primus) (Primus-LM), AMD's training framework for
large-scale foundation models on ROCm. Primus unifies several backends —
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
[TorchTitan](https://github.com/pytorch/torchtitan), JAX MaxText, and Megatron-Bridge — under
one YAML configuration system; post-training here runs through **Megatron-Bridge** via the
`train posttrain` subcommand. `train_llm_primus.py` picks the config matching your GPU arch,
applies common overrides, checks the ROCm environment, and execs `primus-cli`.

**Hardware:** AMD Instinct only, by design — MI300X / MI325X (gfx942) and MI350X / MI355X
(gfx950), ROCm 7.2.4. There is no CUDA image and no CUDA install path; on NVIDIA use
`../nemo/`, `../megatron/`, `../deepspeed/`, or `../openrlhf/` instead.

## Files

- `train_llm_primus.py` — launcher; resolves the config for your GPU arch, checks ROCm, execs `primus-cli`.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample.
- `configs/MI300X/qwen3_32b_sft_posttrain.yaml` — full fine-tune, gfx942 (MI300X / MI325X).
- `configs/MI300X/qwen3_32b_lora_posttrain.yaml` — LoRA, gfx942.
- `configs/MI355X/qwen3_32b_sft_posttrain.yaml` — full fine-tune, gfx950 (MI350X / MI355X).
- `configs/MI355X/qwen3_32b_lora_posttrain.yaml` — LoRA, gfx950.
- `requirements_primus.txt` — install paths (container / wheel / bare metal).

## Setup

**Prerequisites:** ROCm drivers >= 7.0, Docker >= 24.0 with ROCm support, and an Instinct GPU.
Check with `rocm-smi && docker --version`.

Use the container. Image `rocm/primus:v26.5`, digest
`sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1` — **54.8 GB on
disk**, so check free space before pulling. The whole stack ships pre-built at
`/workspace/Primus`; the clone below is only for reading sources on the host.

```bash
export OUTPUT_DIR=/path/to/outputs     # training artifacts, mounted at /work
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache

docker pull rocm/primus:v26.5

git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init --recursive
```

**Scope GPUs with `--device`, never with `-e HIP_VISIBLE_DEVICES`.** `primus-cli`
unconditionally overwrites `HIP_VISIBLE_DEVICES` with `seq -s, 0 $((GPUS_PER_NODE - 1))` in
`runner/helpers/envs/base_env.sh`, so an env-var scope is silently ignored and you train on
physical GPUs 0,1. Map the GPUs you want to their DRM nodes first:

```bash
for d in /sys/class/drm/renderD*; do echo "$d -> $(basename $(readlink -f $d/device))"; done
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

# required fix: the image ships datasets 3.6.0, too old to parse current Hub dataset cards
docker exec primus_mi355x pip install "datasets==4.3.0"
```

For all GPUs, pass the whole `/dev/dri` (`--device /dev/kfd --device /dev/dri`) and let
`GPUS_PER_NODE` line up with the rewrite. Confirm `torch.cuda.device_count()` inside the
container before launching.

`/data` must be a mounted, writable volume: the conversion hook writes the Megatron checkpoint
to `/data/megatron_checkpoints/<model>`.

**Pip wheel (launcher only; still trains inside the container):**

```bash
python -m venv primus-env && source primus-env/bin/activate
pip install "primus==26.5.0" --no-deps \
  --extra-index-url https://amd-agi.github.io/Primus/simple/

primus-cli deps sync --dir ~/.cache/Primus/third_party   # fetch pinned backend sources
```

`requirements_primus.txt` has all three paths, including bare metal.

MI325X uses the MI300X configs (both gfx942); MI350X uses the MI355X configs (both gfx950).
`--arch` maps this for you. Running the MI355X config on MI300X OOMs — the two differ in TP and
micro-batch size.

### Secrets

```bash
ln -sf ../../../dev.env dev.env        # HF_TOKEN; loaded by train_llm_primus.py
```

The launcher forwards it into the container as `--env HF_TOKEN=...`, which matters here because
the container does not inherit your host environment. In `--mode container` the launcher logs
the full command at INFO, token included — redact it in shared logs. Never commit a token.

You may also need `PRIMUS_WORKSPACE` set if your team's config templates reference it.

## Data

`data/OTel_LLM_sample_10.jsonl` — 10 rows of this repo's canonical chat JSONL:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

**`--data-path` is dead code on the Megatron-Bridge post-training path.** The `qwen.qwen3`
recipe hard-codes `default_squad_config(...)` and exposes no dataset argument, and Primus's
`auto_filter_and_call()` silently drops the unknown key (`Retry 20: Removing invalid parameter
'dataset'`) instead of failing. To train on your own data, write a custom recipe that exposes a
dataset argument, or pre-convert your JSONL to what the Bridge `HFDatasetConfig` pipeline
expects.

The same filter drops `recompute_granularity` / `recompute_method` / `recompute_num_layers` and
nested keys such as `checkpoint.finetune`. **Grep the log for
`Removing invalid parameter '<your key>'` before trusting any override.**

## Run

Print the assembled `primus-cli` command without launching anything:

```bash
python3 train_llm_primus.py --method sft --arch MI300X --dry-run
```

Run `primus-cli` inside the already-started container. Copy the model preset into the Primus
tree first — `modules.post_trainer.model:` is a **preset name** resolved against
`primus/configs/models/<framework>/`, not a path relative to your experiment YAML, so a fresh
container otherwise fails in about two seconds with
`FileNotFoundError: [Primus] Preset '<name>.yaml' not found for framework 'megatron_bridge'`:

```bash
docker exec primus_mi355x \
  cp /work/<your>/qwen3_600m.yaml /workspace/Primus/primus/configs/models/megatron_bridge/

docker exec -w /workspace/Primus \
  -e GPUS_PER_NODE=2 -e NNODES=1 -e NODE_RANK=0 \
  -e MASTER_ADDR=localhost -e MASTER_PORT=29740 \
  -e HF_HOME=/root/.cache/huggingface \
  primus_mi355x bash -lc \
  './runner/primus-cli direct --config /work/smoke/qwen3_600m_sft_smoke.yaml \
     -- train posttrain --config /work/smoke/qwen3_600m_sft_smoke.yaml'
```

**Set `GPUS_PER_NODE`** — it defaults to 8, and it also determines the `HIP_VISIBLE_DEVICES`
rewrite. Each rank logs `[Primus:Env] rank=N, world_size=<GPUS_PER_NODE>, ...`, the quickest
confirmation that the scoping took. Scaling 2 to 8 GPUs needs nothing else.

Through the launcher:

```bash
nohup python3 train_llm_primus.py \
  --method lora --arch MI355X \
  --mode container --image rocm/primus:v26.5 \
  --data-path /path/to/train.jsonl \
  --train-iters 500 --global-batch-size 32 --micro-batch-size 4 \
  --seq-length 8192 --finetune-lr 1e-4 \
  > train_llm_primus.log 2>&1 &

tail -f train_llm_primus.log
```

Equivalent raw CLI:

```bash
./runner/primus-cli direct -- train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

Per-iteration lines land in the **last rank's** `debug.log`, not on stdout (see Output):

```
iteration        1/     600 | consumed samples:    8 | lm loss: 9.745878E+00 | grad norm: 458.126 | number of nan iterations:   0 |
iteration      600/     600 | consumed samples: 4800 | lm loss: 8.565324E-01 | grad norm: 123.243 | number of nan iterations:   0 |
validation loss at iteration 600 on validation set | lm loss value: 3.841884E-01 | lm loss PPL: 1.468422E+00 |
```

A LoRA run must report a far smaller trainable-parameter count than SFT; the full count means
`peft: lora` did not take effect.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--config` | `None` | Explicit Primus YAML path; overrides `--method`/`--arch` selection |
| `--method` | `sft` | `sft` (full fine-tune) or `lora` (adapters only) |
| `--arch` | `MI300X` | `MI300X` / `MI325X` / `MI350X` / `MI355X` — selects the tuned reference config |
| `--mode` | `container` | `container`, `direct` (bare metal or already inside a container), or `slurm` |
| `--image` | `rocm/primus:v26.5` | ROCm training image for container mode |
| `--cli` | `./primus-cli` | Path to `primus-cli`; a repo clone usually exposes `./runner/primus-cli` |
| `--data-path` | `data/OTel_LLM_sample_10.jsonl` | Training data (see Data — unused on this path) |
| `--train-iters` | `None` | Override `train_iters` in the YAML |
| `--global-batch-size` | `None` | Override `global_batch_size` |
| `--micro-batch-size` | `None` | Override `micro_batch_size` |
| `--seq-length` | `None` | Override `seq_length` |
| `--finetune-lr` | `None` | Override `finetune_lr` |
| `--log-file` | `None` | Write the training log here; required in container mode (see Output) |
| `--extra` | `[]` | Everything after this is forwarded verbatim to `primus-cli` |
| `--dry-run` | off | Print the command and exit |

Key config fields (in the YAML, under `modules.post_trainer.overrides`):

| Field | Meaning |
|---|---|
| `peft` | `"none"` for SFT, `lora` for LoRA |
| `finetune_lr`, `min_lr`, `lr_warmup_iters`, `lr_decay_iters` | LR schedule |
| `precision_config` | Typically `bf16_mixed` |
| `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `sequence_parallel` | Parallelism |
| `recompute_granularity`, `recompute_method`, `recompute_num_layers` | Activation recompute |
| `train_iters`, `global_batch_size`, `micro_batch_size`, `seq_length` | Batching and sequence length |

LoRA wants a much higher LR than SFT (`1e-4`-`5e-4` versus `5e-6`-`1e-5`). The reference
configs already encode this — do not copy an SFT LR into a LoRA run.

## Output

Checkpoints and logs go to the output directory configured in the trainer YAML. In container
mode pass `--log-file` so the log lands on a mounted volume; otherwise it is written inside the
container (`site-packages/primus/logs` by default) and lost when the container exits.

Per-iteration loss lines are **not** on stdout and **not** in `rank-0/info.log`. Primus routes
Megatron's `training_log` through `print_rank_last`, so they land in the last rank's
`debug.log`:

```bash
tail -f output/local/local/<exp_name>/logs/post_trainer/rank-1/debug.log
# full form: <workspace>/<work_group>/<user_name>/<exp_name>/logs/post_trainer/rank-<N-1>/debug.log
```

Mount data and output directories explicitly:

```bash
primus-cli container --image rocm/primus:v26.5 \
  --env HF_TOKEN="hf_xxx" \
  --volume /path/to/your/data:/data \
  -- --log_file /data/run.log \
  -- train posttrain --config /data/your/config.yaml
```

Monitoring integrates with MLflow and TraceLens if configured in the experiment YAML.

## Notes

- **Use `--method sft` on `rocm/primus:v26.5`.** LoRA builds the adapters and then fails to
  load the base checkpoint (`KeyError: "...adapter.linear_in.weight from model not in state
  dict"`).
- **Keep TP and PP at 1** (so DP = `GPUS_PER_NODE`). `01_convert_checkpoints.sh` passes no
  parallelism arguments, so the base checkpoint is always TP1/PP1 and any other layout dies at
  checkpoint load. To use TP/PP, produce the base checkpoint yourself at the target
  parallelism, or save it with `checkpoint_config.fully_parallel_save=True`.
- **Never set `CUDA_VISIBLE_DEVICES` alongside `HIP_VISIBLE_DEVICES`** in this container — it
  is an immediate abort (`Conflicting visibility of agent-N ... core dumped`), not a warning.
- When writing a new config, `workspace` is a **required** top-level key
  (`workspace: ${PRIMUS_WORKSPACE:./output}`), and `sequence_parallel: true` with
  `tensor_model_parallel_size: 1` is fatal (`Cannot use sequence parallelism without tensor
  parallelism`). Upstream reference configs set `sequence_parallel: false` even where TP > 1.
- **OOM playbook:** `micro_batch_size` then `seq_length` are the only working knobs on this
  image — raising TP/PP and the `recompute_*` keys are both unavailable for the reasons above.
- `rocm-smi --showpids`' `GPU(s)` column is not the physical index; use
  `rocm-smi --showmeminfo vram` per index to attribute usage.
- **Pre-seed the Hub cache from the host.** Blob downloads can hang mid-transfer inside the
  container, leaving `.incomplete` files. Download on the host and mount the cache. The
  container runs as root, so watch for root-owned files that block host-side writes afterwards.
- **The launcher pip-installs at runtime.** `00_install_requirements.sh` installs
  Megatron-Bridge deps on every run, so the container needs network egress.
