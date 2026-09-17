# `training/llm/nemo` — NVIDIA NeMo AutoModel (SFT + LoRA)

Post-training (**SFT** and **PEFT/LoRA**) of a Hugging Face checkpoint with
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel), the LLM post-training library split
out of the original [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) monorepo. Pick it when you
specifically want NVIDIA's stack — Transformer Engine kernels, FP8, MoE expert parallelism, and a
documented path onto Megatron-Core parallelism. For an ordinary 8xH100 LoRA run,
`training/llm/unsloth/` or `training/llm/deepspeed/` are far lighter to stand up.

NeMo AutoModel is YAML-recipe driven and launched by its own `automodel` CLI, which owns the
torchrun/SPMD launch. `train_llm_nemo.py` turns CLI flags into a valid recipe YAML, preflights the
data, and execs `automodel <config.yaml> --nproc-per-node N`.

**Hardware:** NVIDIA H100 80GB (CUDA 13, NGC container or bare-host pip) · AMD MI355X (gfx950,
ROCm 7.2.4, `torch 2.13.0+rocm7.2`), pip route only.

## Files

- `train_llm_nemo.py` — entrypoint. Generates the AutoModel recipe YAML and launches the
  `automodel` CLI.
- `generated_recipe.yaml` / `generated_recipe_lora.yaml` — the recipes the SFT and LoRA commands
  below produce (`./checkpoints_sft` and `./checkpoints_lora` respectively).
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat-`messages` sample, the default `--train_file`.
- `requirements_nemo.txt` — pinned dependencies for the pip route.

## Setup

One venv per recipe folder. Both routes need the repo-root `dev.env` linked in —
`train_llm_nemo.py` calls `load_dotenv("dev.env")` relative to the working directory, so run it
from inside this folder:

```bash
cd training/llm/nemo
ln -sf ../../../dev.env dev.env               # HF_TOKEN, for gated models such as Llama
```

`dev.env` holds `HF_TOKEN=hf_...` and is git-ignored at the repo root. Never commit a token; none
belongs in a recipe YAML either. Point `HF_HOME` at a large disk if your model cache is not on `/`.

### NVIDIA — NGC container (recommended)

NeMo pulls Transformer Engine, Megatron-FSDP and custom CUDA/Triton kernels that compile against
the exact CUDA and torch in the image. `26.06.00` is the container upstream pairs with AutoModel
v0.5.0; check [the NGC tag list](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel/tags)
before pinning.

```bash
# nvcr.io 403s behind an HTTP proxy - unset it to pull (~20 GB).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker pull nvcr.io/nvidia/nemo-automodel:26.06.00

# Use --gpus, not --runtime=nvidia: recent Docker often registers only `runc`, and
# --runtime=nvidia then errors with "unknown or invalid runtime name: nvidia".
docker run --rm --gpus '"device=5"' --ipc host \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 \
  -v $HF_HOME:/models -v <repo>:/work \
  -w /work/training/llm/nemo \
  nvcr.io/nvidia/nemo-automodel:26.06.00 bash
```

The image carries `torch 2.12.0a0+...nv26.04`, CUDA 13.2, `nemo-automodel 0.5.0`,
`transformers 5.8.1` and `transformer-engine 2.14.1`. A `CUDA Forward Compatibility mode ENABLED`
banner is expected — the userspace CUDA 13.2 runs forward-compat over an older host driver.

### NVIDIA — bare-host pip

```bash
cd training/llm/nemo
python3 -m venv .env_nemo && source .env_nemo/bin/activate
pip install torch numpy                       # -> the current CUDA build (torch 2.13.0+cu130)
pip install nemo-automodel==0.5.0 transformers==5.8.1 \
            datasets torchdata megatron-fsdp==0.5.0 pyyaml python-dotenv
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
automodel --help
```

Upstream itself uses `uv` (`uv venv && uv sync --frozen`); `uv sync --frozen --extra cuda` adds
Transformer Engine and Mamba SSM, which is a slow build and needs `CUDA_HOME` / `PATH` /
`LD_LIBRARY_PATH` pointing at a matching toolkit.

`attn_implementation: sdpa` (the default) already routes to FlashAttention on H100 — it logs
`Patched model with SDPA method=[CUDNN_ATTENTION, FLASH_ATTENTION, ...]`. Building `flash-attn`
buys nothing.

### AMD (ROCm)

There is no ROCm container and no upstream ROCm install path, so expect no vendor support. The
SFT and LoRA paths this folder generates do run via the pip route:

```bash
cd training/llm/nemo
python3.12 -m venv .env_nemo && source .env_nemo/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch   # FIRST
pip install -r requirements_nemo.txt                                     # transformers==5.8.1
python -c "import torch; print(torch.version.hip, torch.cuda.get_device_name(0))"
python -c "import nemo_automodel; print(nemo_automodel.__version__)"
automodel --help
```

Never install the `[cuda]`, `[fa]` or `[moe]` extras on ROCm — `transformer-engine`, `mamba-ssm`,
`causal-conv1d`, `nv-grouped-gemm`, `flash-attn` and `deep_ep` are CUDA-only, and TE cannot even
generate metadata without a CUDA toolkit. You lose FP8/MXFP8, MoE/DeepEP and the Mamba kernels;
you keep HF-checkpoint SFT, PEFT/LoRA (including the Triton LoRA kernel, which JIT-compiles for
`gfx950`), FSDP2 and safetensors checkpointing. Keep `--attn_implementation sdpa`.

After installing, confirm the ROCm torch survived: `flashoptim` (a required `nemo-automodel`
dependency) pulls the NVIDIA-published `triton` wheel over `triton-rocm`. This is benign at
`triton` 3.7.1, which ships both backends, but verify and reinstall `pytorch-triton-rocm` if a
future release drops the AMD backend:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

`training/llm/primus/` is this repo's first-party AMD path and `training/llm/verl/` the AMD RL
path. Pick this folder on AMD only when you want AutoModel's HF-native recipe and checkpoint
contract.

## Data

`data/OTel_LLM_sample_10.jsonl` — this repo's canonical chat JSONL, one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Rows also carry `unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id` and
`source_version`; AutoModel's `ChatDataset` ignores them. No conversion step is needed —
`ChatDataset` reads OpenAI-style `messages` natively and renders them through the tokenizer's chat
template. Point `--train_file` at your own JSONL in the same schema.

Two constraints:

- **A chat template is required.** `ChatDataset` raises if the tokenizer has none. Base
  (non-instruct) checkpoints often ship without one — use the `-Instruct` variant, or set
  `dataset.chat_template` in the generated YAML.
- **Validation split.** If you do not pass `--val_file`, the script holds out `--val_fraction`
  (default 2%, minimum 1 row) from `--train_file` and writes `*_split_train.jsonl` /
  `*_split_val.jsonl` next to it.

At `--seq_length 512` the long OTel assistant turns get truncated and some micro-batches log
`num_label_tokens 0 | loss 0.0000`. Use `--seq_length 2048` to keep them.

For prompt/completion columns instead, swap the dataset `_target_` in the generated YAML to
AutoModel's `ColumnMappedTextInstructionDataset`.

## Run

Smoke test — writes the recipe YAML and prints the launch command without touching a GPU:

```bash
python3 train_llm_nemo.py --dry_run
```

Full SFT (single GPU, shipped sample):

```bash
export HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0            # plus HIP_VISIBLE_DEVICES=0 on ROCm
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
  --checkpoint_dir ./checkpoints_sft --config_out generated_recipe.yaml \
  --nproc_per_node 1 --dp_size 1 --seq_length 512 \
  --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 4 --max_steps 12 --warmup_steps 2 \
  --val_every_steps 6 --ckpt_every_steps 12
```

LoRA (`peft.use_triton: true` is always set by the generated recipe):

```bash
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B --use_lora \
  --checkpoint_dir ./checkpoints_lora --config_out generated_recipe_lora.yaml \
  --lora_dim 16 --lora_alpha 32 --nproc_per_node 1 --dp_size 1 \
  --seq_length 512 --global_batch_size 2 --local_batch_size 1 \
  --num_epochs 4 --max_steps 12 --warmup_steps 2 \
  --val_every_steps 6 --ckpt_every_steps 12
```

The first LoRA step is slow (~40 s) on ROCm: that is one-time Triton kernel compilation for
`gfx950`, not a hang.

Full production run:

```bash
nohup python3 train_llm_nemo.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --train_file data/train.jsonl \
  --checkpoint_dir ./checkpoints \
  --use_lora --lora_dim 16 --lora_alpha 32 \
  --seq_length 2048 --global_batch_size 64 --local_batch_size 1 \
  --num_epochs 3 --max_steps 1000 \
  --nproc_per_node 8 \
  > train_llm_nemo.log 2>&1 &

tail -f train_llm_nemo.log
```

A healthy run shows falling loss with a finite grad norm and a checkpoint under
`--checkpoint_dir` at the first `--ckpt_every_steps` boundary. A job that dies immediately after
"Launching job interactively" is almost always the model load (gated repo / missing `HF_TOKEN`) or
OOM — lower `--local_batch_size` or `--seq_length` first.

### Multi-GPU

`--nproc_per_node N --dp_size N` gives pure FSDP2 data parallel: one replica sharded across N
ranks, `global_batch_size / (local_batch_size * N)` gradient-accumulation steps per optimizer
step. NCCL on NVIDIA, RCCL on ROCm (the recipe's `dist_env.backend: nccl` maps onto it).

**The `automodel` CLI cannot change torchrun's rendezvous port.** `InteractiveLauncher` inherits
torchrun's default `--master_port 29500` and forwards extra flags to its config-override parser,
so on a box where 29500 is bound you must launch torchrun yourself. `InteractiveLauncher` detects
an existing torchrun worker and runs the recipe in-process rather than re-launching:

```bash
cd training/llm/nemo && source .env_nemo/bin/activate
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/path/to/hf_cache

# 1. generate the 8-GPU recipe (no GPU touched)
python3 train_llm_nemo.py --model_name Qwen/Qwen3-0.6B \
  --train_file  $OUT/train_8gpu.jsonl \
  --checkpoint_dir $OUT/checkpoints_8gpu \
  --config_out  $OUT/generated_recipe_8gpu.yaml \
  --nproc_per_node 8 --dp_size 8 --tp_size 1 --pp_size 1 --cp_size 1 \
  --seq_length 512 --global_batch_size 16 --local_batch_size 1 \
  --num_epochs 1 --max_steps 20 --warmup_steps 2 \
  --val_every_steps 10 --ckpt_every_steps 20 --dry_run

# 2. launch it with an explicit, non-colliding rendezvous port
torchrun --nproc-per-node 8 --nnodes 1 \
  --master_addr 127.0.0.1 --master_port 29770 \
  -m nemo_automodel.cli.app $OUT/generated_recipe_8gpu.yaml
```

Use `automodel <cfg> --nproc-per-node 8` directly only when port 29500 is free.

The shipped 10-row sample splits to 9 train rows — fewer than one global batch across 8 ranks.
Duplicate it (x64 gives 628 train / 12 val) so every rank has real data; the schema is unchanged.
`tps N(M/gpu)` in the log divides by the world size, which confirms all ranks are in the group.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Base HF checkpoint (repo id or local path) |
| `--train_file` | `data/OTel_LLM_sample_10.jsonl` | Chat `messages` JSONL |
| `--val_file` | `None` | Held-out chat JSONL; split off `--train_file` if omitted |
| `--val_fraction` | `0.02` | Held-out fraction when `--val_file` is not given |
| `--checkpoint_dir` | `./checkpoints` | Where AutoModel writes checkpoints |
| `--config_out` | `generated_recipe.yaml` | Path for the generated recipe YAML |
| `--seq_length` | `2048` | Max tokens per sequence |
| `--global_batch_size` | `64` | Global batch size across all ranks |
| `--local_batch_size` | `1` | Per-GPU micro batch size |
| `--num_epochs` | `3` | Number of epochs |
| `--max_steps` | `1000` | Hard cap on optimizer steps |
| `--learning_rate` | `None` | Peak LR; 1e-4 with `--use_lora`, else 5e-6 |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `10` | LR warmup steps; must be `< --max_steps` |
| `--val_every_steps` | `100` | Run validation every N steps |
| `--ckpt_every_steps` | `200` | Write a checkpoint every N steps |
| `--seed` | `42` | Random seed (also seeds the train/val split) |
| `--attn_implementation` | `sdpa` | `sdpa`, `eager`, or `flash_attention_2` (FA2 needs flash-attn built) |
| `--use_lora` | off | Train a LoRA adapter instead of full fine-tuning |
| `--lora_dim` | `16` | LoRA rank (AutoModel calls this `dim`, not `r`) |
| `--lora_alpha` | `32` | LoRA alpha (scaling) |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `*_proj` | Glob or comma-separated module names to adapt |
| `--nproc_per_node` | `8` | GPUs per node, passed to the `automodel` CLI |
| `--dp_size` | `None` | Data-parallel size (defaults to `nproc_per_node`) |
| `--tp_size` | `1` | Tensor-parallel size |
| `--pp_size` | `1` | Pipeline-parallel size |
| `--cp_size` | `1` | Context-parallel size |
| `--dry_run` | off | Write the YAML and print the command; do not launch |

## Output

Checkpoints land under `--checkpoint_dir` at every `--ckpt_every_steps` boundary. The recipe sets
`model_save_format: safetensors` and `save_consolidated: true`, so the consolidated result is a
standard HF checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("checkpoints/<step-dir>/model/consolidated")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

With `--use_lora` the artifact is the **adapter** (`model/adapter_model.safetensors`,
`adapter_config.json`, `automodel_peft_config.json`), so inference needs the base model plus the
adapter. AutoModel also maintains a `LOWEST_VAL` symlink to the best-validation checkpoint.

Keep the generated recipe YAML with the checkpoint — it is the reproducible description of what
was trained. `--config_out` defaults to the fixed name `generated_recipe.yaml`, so every
invocation overwrites it unless you pass a distinct path; pair each `--config_out` with its own
`--checkpoint_dir`.

## Notes

- **Run `automodel` from an activated venv.** The script shells out to a bare `automodel`, so
  `.env_nemo/bin/python train_llm_nemo.py` without activating fails with
  `FileNotFoundError: ... 'automodel'`.
- **Unset the proxy and export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`** even for a cached
  model. HF lists the repo tree online during resolution, which 403s behind a proxy.
- **`transformers` must be exactly `5.8.1`.** `nemo-automodel==0.5.0` pins it; any other version
  is a hard `ResolutionImpossible` on every platform.
- **This folder targets NeMo AutoModel, not NeMo 2.x.** `automodel --help` succeeding means you
  are on the right library; on `nemo_toolkit[llm]` the `nemo.collections.llm` / `llm.import_ckpt`
  API applies instead and this recipe YAML will not work. If AutoModel rejects a key in the
  generated YAML, diff it against a current recipe in
  [`examples/llm_finetune/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune).
- **RLHF is a different repo.** AutoModel does SFT/PEFT/distillation only; GRPO, GSPO, DAPO, DPO
  and reward modelling live in [NeMo RL](https://github.com/NVIDIA-NeMo/RL), which can start from
  an AutoModel checkpoint directly.
- **Megatron-format checkpoints** (very large models, 6D parallelism) go through Megatron-Bridge:
  ```bash
  ./scripts/conversion/convert.sh import \
    --executor local --device cpu \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b

  ./scripts/conversion/convert.sh export \
    --executor local --device cpu \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b/iter_0000000 \
    --hf-path ./exports/llama3_2_1b_hf
  ```
- **Loss masking:** the recipe uses `MaskedCrossEntropy` and `ChatDataset` builds the mask from
  the chat template, so only assistant turns are supervised. For multi-turn data every assistant
  turn is supervised; set `mask_history: true` on the dataset block to supervise only the last.
- Raise `tp_size` only when one model replica will not fit on a single GPU; the default pure data
  parallel is the right starting point for LoRA or small-model SFT.
