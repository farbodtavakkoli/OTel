# `training/llm/megatron` — continued pre-training with Megatron-LM

Large-scale continued pre-training (domain-adaptive pre-training) on
[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM): TP / PP / DP / CP parallelism, the
distributed optimizer, and the TransformerEngine fused kernels. Pick it over the other trainers
here when you need maximum throughput at 8B+ scale and can pay for it with a heavy install and an
offline data-preprocessing step.

`pretrain_gpt.py` consumes packed pre-training token streams from an indexed dataset — no chat
template, no prompt masking — so this folder is not an SFT/DPO/GRPO path. For post-training on
this stack use [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), NeMo, or AMD's
[Primus](https://github.com/AMD-AGI/Primus).

**Hardware:** NVIDIA H100 (NGC PyTorch `25.06-py3`, CUDA 12.9) · AMD MI355X (gfx950, ROCm 7.2.4,
torch `2.11.0+rocm7.2`). Megatron's flag set moves between releases — run
`python pretrain_gpt.py --help` in your checkout before a long run.

## Files

- `train_llm_megatron.py` — launcher. Translates the TOML config into `pretrain_gpt.py` flags,
  validates the geometry, and execs `torchrun` from your Megatron-LM clone.
- `megatron_cpt_config.toml` — the run configuration: model geometry, parallelism, optimizer,
  data paths, checkpointing, logging.
- `preprocess_megatron_data.py` — wraps upstream `tools/preprocess_data.py` to turn a JSONL
  corpus into the indexed `.bin`/`.idx` format Megatron requires.
- `utils.py` — chat-JSONL flatten helpers used by the preprocessor.
- `data/OTel_LLM_sample_10.jsonl` — shipped 10-row chat sample.
- `requirements_megatron.txt` — pure-Python extras (torch / TE / Apex come from the container or
  the ROCm wheel, never from this file).

## Setup

One venv (or one container) per recipe folder. Both routes need the repo-root `dev.env` linked
into this folder — `train_llm_megatron.py` and `preprocess_megatron_data.py` both call
`load_dotenv("dev.env")` relative to the working directory:

```bash
cd training/llm/megatron
ln -sf ../../../dev.env dev.env                    # HF_TOKEN, for gated tokenizers
```

`dev.env` holds `HF_TOKEN=hf_...` and is git-ignored at the repo root. Never commit tokens.

### NVIDIA (NGC PyTorch container)

The container ships TransformerEngine, Apex, flash-attn and cuDNN pre-built and version-matched;
do not re-install torch on top of it.

```bash
# nvcr.io 403s behind an HTTP proxy - unset it to pull (~14 GB compressed).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull nvcr.io/nvidia/pytorch:25.06-py3

# Pin to specific GPUs by UUID (nvidia-smi -L) so the container cannot see co-tenant GPUs.
sudo docker run -d --name megatron_h100 \
  --gpus '"device=GPU-xxxxxxxx-...."' \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/workspace:/work \
  -v /path/to/hf_cache:/models \
  -v /path/to/scratch_out:/out \
  -e HF_HOME=/models \
  -w /work/training/llm/megatron \
  nvcr.io/nvidia/pytorch:25.06-py3 sleep infinity
sudo docker exec megatron_h100 nvidia-smi -L        # must show exactly the GPUs you named

sudo docker exec megatron_h100 bash -c '
  cd /opt && git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM && git fetch --depth 1 origin f481e63 && git checkout f481e63   # optional pin
  pip install --no-build-isolation -e .              # reuses the container torch/TE/Apex
  pip install "nvidia-resiliency-ext>=0.6.0"         # image ships 0.4.0, below Megatron minimum
  cd /work/training/llm/megatron && pip install -r requirements_megatron.txt'
```

`nvidia-resiliency-ext>=0.6.0` is required: with the version the `25.06` image ships,
`import megatron.core` fails with `AttributeError: module 'nvidia_resiliency_ext' has no
attribute '__version__'`. The container does not ship transformers / dotenv / nltk, so
`requirements_megatron.txt` is not optional either.

Outside a container, TransformerEngine must be built against your installed torch:

```bash
pip install --no-build-isolation transformer-engine[pytorch]
```

Verify TE and Apex are engaged — `import megatron.core` must print **no** "Transformer Engine and
Apex are not installed. Falling back to Torch optimizers" warning:

```bash
python -c "import torch, megatron.core; print('imports OK', torch.cuda.device_count(), 'GPUs')"
```

### AMD (ROCm)

Upstream NVIDIA/Megatron-LM trains on MI355X with a plain ROCm PyTorch wheel and **no
TransformerEngine, no Apex, no flash-attn** (all three are CUDA-only). No source patch is needed;
the four fused-kernel flags are flipped through the launcher's `--set`, so
`megatron_cpt_config.toml` stays untouched.

```bash
export OUTPUT_DIR=/path/to/outputs                 # preprocessed data + training artifacts
export HF_HOME=/path/to/hf_cache

cd training/llm/megatron
python3 -m venv .env_megatron                      # git-ignored via .env_*/
source .env_megatron/bin/activate                  # required: torchrun must be on PATH

pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0
pip install pybind11 "packaging>=24.2" numpy       # install BEFORE the editable install below,
                                                   # or --no-build-isolation fails on metadata
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git .env_megatron/Megatron-LM
pip install --no-build-isolation -e .env_megatron/Megatron-LM
pip install -r requirements_megatron.txt
```

`source` the venv rather than calling `.env_megatron/bin/python` directly — the launcher shells
out to bare `torchrun`. Check `.env_megatron/bin/activate` for a stale
`export CUDA_VISIBLE_DEVICES=...` from an earlier session, and re-export both device masks after
activating.

AMD also publishes ROCm-enabled Megatron-LM images (`v26.1`, `v25.11`, `v25.10`, and
architecture-specific tags such as `v25.9_gfx950`) at
https://hub.docker.com/r/rocm/megatron-lm/tags — larger (~116 GB) but they ship AMD's
TE/hipBLASLt/CK stack pre-built, which also restores `sequence_parallel`:

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $PWD:/workspace -v /data:/data -it rocm/megatron-lm:v26.1 bash
```

Do not `pip install transformer-engine` inside the ROCm image — the equivalent is already built.

#### ROCm flags you must flip

Each of these is a TE/Apex fused kernel that Megatron enables by default; without TE installed,
each is a hard failure at model build.

| `--set` override | Failure without it |
|---|---|
| `transformer_impl='"local"'` (TOML ships `"transformer_engine"`) | the `transformer_engine` spec cannot be built |
| `no_rope_fusion=true` | `ValueError: apply_rope_fusion is not available. Please install TE >= 1.4.` |
| `no_persist_layer_norm=true` | `AssertionError: persist_layer_norm not supported by torch LayerNorm` |
| `no_gradient_accumulation_fusion=true` | `RuntimeError: ... fused_weight_gradient_mlp_cuda module is not found` |

`sequence_parallel` must stay `false` on this route: torch LayerNorm cannot do SP without TE.
Note the constraint in the other direction too — `sequence_parallel=true` with
`tensor_model_parallel_size=1` aborts with "Cannot use sequence parallelism without tensor
parallelism". Do not install `flash-attn`; with `transformer_impl="local"`,
`attention_backend = "auto"` resolves to Megatron's unfused torch attention, which is correct here.

## Data

`data/OTel_LLM_sample_10.jsonl` — 10 chat rows, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "unmask": true, "flow": "doc_direct", "source_id": "...", "source_repo": "...",
 "source_spec_id": null, "source_version": null}
```

Megatron does not read JSONL at training time. Pre-tokenize into an indexed dataset — a
`<prefix>.bin` blob of token ids plus a `<prefix>.idx` offset table. A `{"text": ...}` corpus is
the native form; chat `messages` are flattened to `text` automatically.

```bash
# ROCm venv route
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
python preprocess_megatron_data.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --tokenizer-model Qwen/Qwen2.5-0.5B \
  --output-prefix $OUTPUT_DIR/train_llm_megatron/otel \
  --workers 8

# NGC container route (HF_HUB_OFFLINE forces the cached tokenizer)
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 megatron_h100 \
  python preprocess_megatron_data.py \
    --megatron-repo /opt/Megatron-LM \
    --tokenizer-model Qwen/Qwen3-0.6B \
    --output-prefix /out/otel \
    --workers 8
```

Both write `otel_text_document.bin` (49968 B) + `.idx`.

The config default `meta-llama/Llama-3.1-8B` is gated and returns `403 Cannot access gated repo`
without granted access — substitute an ungated tokenizer as above, or supply a token with access.

Full run on your own corpus:

```bash
python preprocess_megatron_data.py \
  --megatron-repo ../Megatron-LM \
  --input /data/domain_corpus.jsonl \
  --output-prefix data/domain \
  --tokenizer-model meta-llama/Llama-3.1-8B \
  --workers 64 --partitions 4
```

Upstream's naming rule is `<output-prefix>_<json-key>_document.{bin,idx}`. The **prefix without
the suffix** is what goes into the config (the helper prints it):

```toml
data_path = "./data/otel_text_document"
```

- `--workers` must be divisible by `--partitions`.
- `--append-eod` is on by default; keep it for pre-training or documents bleed across the packed
  sequence boundary. Disable with `--no-append-eod`.
- `[data].split = "990,8,2"` splits that one indexed dataset into train/valid/test by document.
  There is no separate validation file.
- Use the **same tokenizer** for preprocessing and training. A mismatch fails silently.

## Run

Check the resolved command first with `--dry-run`, then launch:

```bash
python train_llm_megatron.py --megatron-repo ../Megatron-LM --dry-run
```

Loss should fall monotonically from `ln(vocab_size)` for a randomly-initialised model (11.97 for
Qwen2.5-0.5B, 11.93 for Qwen3-0.6B) with `number of nan iterations: 0`. For a real continued
pre-training run loss starts low instead; a loss near `ln(vocab_size)` means the checkpoint did
not load, which is what `exit_on_missing_checkpoint` exists to prevent.

### AMD MI355X

The command below is the 2-GPU smoke: a tiny custom GPT (8 layers / hidden 1024) rather than the
shipped Llama-3.1-8B geometry, saving off, no checkpoint load. `--set key=false` makes the
launcher omit the flag entirely, which is how `save`, `load` and `tensorboard_dir` are switched
off without editing the TOML.

```bash
cd training/llm/megatron
source .env_megatron/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

python train_llm_megatron.py \
  --megatron-repo .env_megatron/Megatron-LM \
  --nproc-per-node 2 --train-iters 200 \
  --data-path $OUTPUT_DIR/train_llm_megatron/otel_text_document \
  --set master_port=29750 \
  --set transformer_impl='"local"' \
  --set tokenizer_model='"Qwen/Qwen2.5-0.5B"' \
  --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
  --set num_attention_heads=16 --set num_query_groups=4 \
  --set seq_length=1024 --set max_position_embeddings=1024 \
  --set use_rope_scaling=false \
  --set no_rope_fusion=true \
  --set no_persist_layer_norm=true \
  --set no_gradient_accumulation_fusion=true \
  --set global_batch_size=16 --set micro_batch_size=2 \
  --set lr_warmup_iters=10 --set lr_decay_iters=200 \
  --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=10000 \
  --set log_interval=10 --set log_throughput=true \
  --set load=false --set save=false --set save_interval=100000 \
  --set no_load_optim=false --set no_load_rng=false \
  --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
  --set tensorboard_dir=false
```

To scale to 8 GPUs, change only `--nproc-per-node`, the device masks, the parallel sizes,
`global_batch_size`, and `master_port` — no extra package, no RCCL tuning variable, and the same
four fused-kernel flags:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -c "import torch;assert torch.cuda.device_count()==8"      # assert before training
```

| Layout | `--nproc-per-node` | TP | PP | CP | DP | `global_batch_size` |
|---|---|---|---|---|---|---|
| `tp2_pp2_dp2` | 8 | 2 | 2 | 1 | 2 | 32 |
| `dp8` | 8 | 1 | 1 | 1 | 8 | 64 |
| `tp2_pp1_dp4` | 8 | 2 | 1 | 1 | 4 | 32 |

Set them with `--set tensor_model_parallel_size=2 --set pipeline_model_parallel_size=2 --set
context_parallel_size=1 --set sequence_parallel=false`, plus a distinct `--set master_port` per
layout. Megatron's argument dump confirms the groups were really built — for `tp2_pp2_dp2` it
reports `inprocess_active_world_size 8` with `tensor_model_parallel_size 2 /
pipeline_model_parallel_size 2 / context_parallel_size 1 / data_parallel_size 2`.

### NVIDIA H100

All four fused-kernel flags stay at their Megatron defaults (ON) and `transformer_impl` stays
`"transformer_engine"` — none of the ROCm `--set no_*` overrides apply. Single-GPU smoke:

```bash
sudo docker exec -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_TOKEN="$HF_TOKEN" megatron_h100 \
  python train_llm_megatron.py \
    --megatron-repo /opt/Megatron-LM \
    --nproc-per-node 1 --train-iters 20 \
    --data-path /out/otel_text_document \
    --set master_port=29660 \
    --set transformer_impl='"transformer_engine"' \
    --set tokenizer_model='"Qwen/Qwen3-0.6B"' \
    --set num_layers=8 --set hidden_size=1024 --set ffn_hidden_size=2816 \
    --set num_attention_heads=16 --set num_query_groups=4 \
    --set seq_length=1024 --set max_position_embeddings=1024 \
    --set use_rope_scaling=false \
    --set global_batch_size=8 --set micro_batch_size=1 \
    --set lr_warmup_iters=2 --set lr_decay_iters=20 \
    --set split='"100,0,0"' --set eval_iters=0 --set eval_interval=100000 \
    --set log_interval=1 --set log_throughput=true \
    --set load=false --set save=false --set save_interval=100000 \
    --set no_load_optim=false --set no_load_rng=false \
    --set use_checkpoint_args=false --set exit_on_missing_checkpoint=false \
    --set tensorboard_dir=false
```

For TP=2 across two GPUs, writing a TP-sharded `torch_dist` checkpoint: launch the container with
`--gpus '"device=6,7"'` (substitute your free devices), add `-e PYTHONPATH=/opt/Megatron-LM`, and
change the command above to

```
--nproc-per-node 2 --train-iters 150 --save /out/ckpt \
--set master_port=29673 \
--set tensor_model_parallel_size=2 \
--set pipeline_model_parallel_size=1 --set context_parallel_size=1 \
--set global_batch_size=16 --set micro_batch_size=2 \
--set lr_warmup_iters=10 --set lr_decay_iters=150 --set log_interval=10 \
--set save_interval=100                  # and drop --set save=false
```

Megatron then reports the parameter count per TP rank and writes two shard files
(`iter_0000150/__0_0.distcp`, `__1_0.distcp`) — the on-disk proof the weights were split.

With TE present you can additionally set `sequence_parallel=true` whenever
`tensor_model_parallel_size > 1` to reclaim activation memory, and FP8 (`--fp8-format`) is
available on Hopper.

### Full run

```bash
nohup python train_llm_megatron.py \
  --megatron-repo ../Megatron-LM \
  --config megatron_cpt_config.toml \
  --nproc-per-node 8 \
  > train_llm_megatron.log 2>&1 &

tail -f train_llm_megatron.log
```

Anything not in the TOML can be added without editing it: `--set key=value` (repeatable) or
`--extra <raw flags...>` appended verbatim.

### Checkpoint conversion

`[checkpoint].load` must be a Megatron-format checkpoint, not an HF directory. Upstream routes
all HF <-> Megatron conversion through Megatron-Bridge:

```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
python Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model meta-llama/Llama-3.1-8B \
  --megatron-path ./checkpoints/llama3_1_8b \
  --torch-dtype bfloat16 \
  --device-map auto
```

Use `export` instead of `import` to go back to HF format for serving or for the HF-based trainers
in this repo.

## Arguments

### `train_llm_megatron.py`

| Argument | Default | What it does |
|---|---|---|
| `--megatron-repo` | *(required)* | Path to your Megatron-LM clone; `pretrain_gpt.py` runs with this as CWD. |
| `--config` | `megatron_cpt_config.toml` | TOML translated into Megatron flags. |
| `--entrypoint` | `pretrain_gpt.py` | Repo-root entrypoint; `pretrain_mamba.py` / `pretrain_hybrid.py` also exist. |
| `--nproc-per-node` | from TOML (`8`) | GPUs on this node. |
| `--data-path` | from TOML | Override the indexed-dataset prefix. |
| `--load` / `--save` | from TOML | Override the checkpoint to start from / write to. |
| `--train-iters` | from TOML | Override the iteration count. |
| `--set KEY=VALUE` | — | Override or add any config key. Repeatable. |
| `--extra ...` | — | Everything after this goes to `pretrain_gpt.py` verbatim. |
| `--dry-run` | off | Print the resolved command and exit. |

Config keys become flags mechanically: `num_layers = 32` -> `--num-layers 32`, `swiglu = true` ->
`--swiglu`, `false` -> flag omitted entirely, list -> flag followed by its items. A key may appear
only once across the whole file.

`--set` values are parsed as TOML, so a string needs TOML quotes *inside* the shell quotes:
`--set transformer_impl='"local"'`. A value TOML cannot parse is kept as a plain string.
Precedence is file, then `--set`, then the dedicated flags (`--nproc-per-node`, `--data-path`,
`--load`, `--save`, `--train-iters`) — a dedicated flag always wins.

### `preprocess_megatron_data.py`

| Argument | Default | What it does |
|---|---|---|
| `--megatron-repo` | *(required)* | Path to a Megatron-LM clone. |
| `--input` | `data/OTel_LLM_sample_10.jsonl` | Input JSONL; a glob when `--partitions > 1`. |
| `--output-prefix` | `data/otel` | Output path prefix, no suffix. |
| `--tokenizer-model` | `meta-llama/Llama-3.1-8B` | HF repo id or local tokenizer dir; must match training. |
| `--tokenizer-type` | `HuggingFaceTokenizer` | Megatron tokenizer type. |
| `--json-keys` | `text` | JSON field(s) holding the document text. |
| `--workers` | all cores | Worker processes; must be divisible by `--partitions`. |
| `--partitions` | `1` | Parallel input partitions, merged afterwards. |
| `--no-append-eod` | off | Do not append the end-of-document token. |
| `--dry-run` | off | Print the command and exit. |

### Key config fields (`megatron_cpt_config.toml`)

| Field | Shipped | Notes |
|---|---|---|
| `[data].data_path` | `./data/otel_text_document` | Indexed-dataset prefix, no `.bin`/`.idx`. |
| `[data].split` | `990,8,2` | train/valid/test document split. |
| `[data].tokenizer_type` | `HuggingFaceTokenizer` | Must match preprocessing. |
| `[training].micro_batch_size` | `1` | Per GPU, per pipeline microbatch. |
| `[training].global_batch_size` | `128` | Must be divisible by `micro_batch_size * DP degree`. |
| `[training].train_iters` | `2000` | Iterations, not epochs. |
| `[training].lr` / `min_lr` | `1e-5` / `1e-6` | Continued pre-training wants a low LR. |
| `[training].lr_warmup_iters` | `100` | Warmup before the cosine decay. |
| `[training].bf16` | `true` | bf16 mixed precision. |
| `[parallelism].tensor_model_parallel_size` | `1` | Keep TP inside one node. |
| `[parallelism].pipeline_model_parallel_size` | `1` | Multi-node territory. |
| `[parallelism].context_parallel_size` | `1` | Raise for sequences at or beyond ~8K. |
| `[parallelism].sequence_parallel` | `false` | Only does something with TP >= 2; needs TE. |
| `[parallelism].use_distributed_optimizer` | `true` | ZeRO-1 style optimizer-state sharding. |
| `[parallelism].overlap_grad_reduce` / `overlap_param_gather` | `true` | Overlap DP collectives with compute. |
| `[memory].recompute_granularity` | `selective` | Selective activation recomputation. |
| `[checkpoint].load` | `./checkpoints/llama3_1_8b` | Megatron format, not HF — convert first. |
| `[checkpoint].save` | `./checkpoints/llama3_1_8b_domain` | Where checkpoints land. |
| `[checkpoint].no_load_optim` / `no_load_rng` | `true` | The converted checkpoint carries neither. |
| `[checkpoint].use_checkpoint_args` | `true` | Take architecture args from the checkpoint. |
| `[checkpoint].exit_on_missing_checkpoint` | `true` | Fail loudly instead of training from random init. |
| `[logging].log_interval` | `10` | Iterations between loss lines. |

## Output

Everything lands under `[checkpoint].save`:

```
./checkpoints/llama3_1_8b_domain/
  iter_0000500/           checkpoint at iteration 500
  iter_0001000/
  iter_0002000/
  latest_checkpointed_iteration.txt
  tensorboard/            TensorBoard event files ([logging].tensorboard_dir)
```

Checkpoints are `torch_dist` format, one shard per TP rank. Even the toy 8-layer geometry writes
~11 GB — point `--save` at scratch, not into the repo.

The output is a Megatron checkpoint, not a Hugging Face one. Export it back with Megatron-Bridge
(`convert_checkpoints.py export`) to serve it or hand it to the HF-based trainers here.

## Notes

- Resuming: the shipped config points `load` and `save` at *different* directories on purpose
  (start from the converted base, write somewhere new). Once the first checkpoint exists, point
  `load` at the `save` directory and drop `no_load_optim` / `no_load_rng` to resume with optimizer
  state.
- Continued pre-training is pre-training with a warm start and a small LR; there is no separate
  CPT mode. That is what `no_load_optim` + `no_load_rng` + `use_checkpoint_args` + a low LR buy you.
- Parallelism ordering for one 8-GPU node with an 8B model: DP=8 with the distributed optimizer is
  usually fastest. Raise TP only when a single GPU cannot hold the layer, keep `TP*EP` inside the
  NVLink/xGMI domain, add `sequence_parallel` whenever TP > 1 (NVIDIA only), use CP for long
  sequences, and save PP for multi-node.
- FP8, MoE and CUDA-graph flags (`--fp8-format`, `--fp8-recipe`, `--num-experts`,
  `--cuda-graph-impl`) are not in the shipped config; add them with `--set`.
