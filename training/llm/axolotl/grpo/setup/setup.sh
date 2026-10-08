#!/usr/bin/env bash
# Axolotl GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

make_venv axolotl
# vLLM 0.27.1, not the shared default: trl 1.12 declares `vllm<=0.27.1` and imports a symbol
# from vLLM's nccl weight-transfer engine that 0.28 removed (GRPOTrainer fails to import).
install_vllm 0.27.1
# --no-deps: axolotl pins transformers 5.14 and would replace what vLLM needs.
"$PIP" install -q --no-deps axolotl==0.18.0
"$PIP" install -q -r "$HERE/requirements.txt"
"$PIP" install -q --no-deps axolotl-contribs-lgpl==0.0.7 axolotl-contribs-mit==0.0.6

check_archs
"$PY" -c "import axolotl, trl, accelerate, vllm; import axolotl.cli.train; from axolotl.core.trainers.grpo import GRPOStrategy; print('axolotl', axolotl.__version__, '| trl', trl.__version__, '| accelerate', accelerate.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_axolotl.py --config grpo_smoke_llama31_8b.yaml"
