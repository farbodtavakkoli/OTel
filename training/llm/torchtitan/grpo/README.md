# TorchTitan RL GRPO

[TitanRL](https://github.com/pytorch/torchtitan) (`torchtitan/experiments/rl`) runs GRPO as a Monarch job: a trainer mesh plus vLLM generator meshes, with TorchStore publishing bf16 weights after every step. It is full parameter only and runs dense Qwen3 policies, so it is not comparable with the LoRA stacks. Both platforms use a host venv on local disk plus a torchtitan source checkout (`--titan-repo` or `TITAN_SRC`), no container. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

```bash
# MI355X (ROCm 7.2, gfx950, Python 3.12) host venv, in $GRPO_ENV_DIR
git clone https://github.com/pytorch/torchtitan torchtitan_src
git -C torchtitan_src checkout 57767a8ba37268bf4a2f8c7904ad13b4c950a958
git -C torchtitan_src apply $REPO_ROOT/training/llm/torchtitan/grpo/setup/torchtitan_rocm_patches.diff
python3 -m venv env_torchtitan_rl && source env_torchtitan_rl/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install -r torchtitan_src/requirements.txt -r torchtitan_src/torchtitan/experiments/rl/requirements.txt \
  -r torchtitan_src/torchtitan/experiments/rl/examples/dapo_math/requirements.txt torchmonarch
pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@8fe02264ae482c4a43cb466acbbb0e0b0221a5a1"
git clone https://github.com/vllm-project/vllm vllm_src && cd vllm_src && git checkout 252ed876214a0a01a6d0ce93bb5bbe77685a4160
pip install ninja cmake setuptools_scm setuptools-rust pybind11 numba scipy "setuptools>=77.0.3,<80"
python use_existing_torch.py && pip install -r requirements/common.txt
PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=96 python setup.py develop
pip install boto3 botocore "protobuf<7" narwhals colorama onnx-ir azure-identity \
  azure-storage-blob google-auth google-cloud-storage iniconfig pluggy   # replaces broken eggs

# H100 (CUDA) host venv, from REPO_ROOT
cd training/llm/torchtitan/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

## Run

```bash
cd $REPO_ROOT/training/llm/torchtitan/grpo
JUDGE_GPU=3 ../../grpo_common/judge_server.sh start      # judge above the --gpus range
source $GRPO_ENV_DIR/env_torchtitan_rl/bin/activate       # venv bin/ must be on PATH
python train_llm_grpo_torchtitan.py --model $GRPO_SHARED_DIR/models/Qwen--Qwen3-4B \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl --output-dir <fresh-run-dir> --gpus 1,2 \
  --steps 2 --prompt-batch-size 8 --group-size 4 --max-tokens 256 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--gpus` | required | physical GPUs, trainer first; length must equal trainer shards × tp + generators × tp |
| `--gpu-env-var` | `ROCR_VISIBLE_DEVICES` | variable `--gpus` is exported as |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--max-prompt-length` / `--max-tokens` | 4096 / 1024 | their sum is `seq_len` |
| `--learning-rate` | `1e-6` | full parameter; an adapter scale lr diverges |
| `--num-generators` / `--generator-tp` / `--trainer-dp-shard` / `--trainer-tp` | 1 / 1 / 1 / 1 | mesh layout |
| `--verify` / `--no-compile` / `--dry-run` | off | reload and generate; disable `torch.compile`; print and exit |

## Notes

1. ROCm needs nightly torch (stable torch fails in `fully_shard`), hence vLLM built from source; keep `vllm_src` next to the venv (`VLLM_SRC`).
2. Before the ROCm vLLM build install `miopen-hip-dev` (apt) and `pip install` the amd_smi bindings from `/opt/rocm/share/amd_smi`; the PyPI wheel breaks the cmake probe.
3. The ROCm patch switches the generator to `TRITON_ATTN`, adds a peak memory metric and fixes GPU visibility. Never apply it on CUDA.
4. CUDA pins: `apache-tvm-ffi==0.1.11`, `torchmonarch==0.6.0`; `pip check` reports one mismatch by design. torchstore comes from git `--no-deps` (the PyPI package is unrelated).
5. CUDA runs need `TITAN_ATTN_BACKEND=flex` and `--gpu-env-var CUDA_VISIBLE_DEVICES`; the default attention needs FlashAttention 3, which has no installable release.
6. `ninja` and `accelerate` are run time requirements.
7. The provisioner sets visibility from 0 upward, so give TitanRL low indices and the judge a higher one; the launcher refuses `--gpus` containing `JUDGE_GPU`.
8. Use a fresh `--output-dir` each run; a non empty one makes the trainer abort.

Benchmarked on MI355X with Qwen3, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
