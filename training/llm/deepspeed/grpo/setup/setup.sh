#!/usr/bin/env bash
# TRL + DeepSpeed GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

make_venv trl_deepspeed
# vLLM 0.27.1, not the shared default: trl 1.12 declares `vllm<=0.27.1` and imports a symbol
# from vLLM's nccl weight-transfer engine that 0.28 removed (GRPOTrainer fails to import).
install_vllm 0.27.1
"$PIP" install -q -r "$HERE/requirements.txt"

check_archs
"$PY" -c "import trl, deepspeed, vllm, peft; print('trl', trl.__version__, '| deepspeed', deepspeed.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_deepspeed.py --help"
