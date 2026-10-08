#!/usr/bin/env bash
# Unsloth GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

make_venv unsloth
# vLLM 0.27.1, not the shared default: trl 1.12 declares `vllm<=0.27.1` and imports a symbol
# from vLLM's nccl weight-transfer engine that 0.28 removed (GRPOTrainer fails to import).
install_vllm 0.27.1
# --no-deps: unsloth pins torch<2.12 and transformers<=5.5.0, which would replace what vLLM needs.
"$PIP" install -q --no-deps unsloth==2026.9.2 unsloth_zoo==2026.9.1
"$PIP" install -q -r "$HERE/requirements.txt"

# Gemma-4 only: unsloth_zoo's vLLM state-dict conversion assumes per_layer_input_gate/projection
# exist, but vLLM's Gemma-4 decoder sets them to None for the 31B. Inert for other architectures.
SP="$(site_packages)"
if [ -f "$SP/unsloth_zoo/vllm_utils.py" ]; then
    sed -i 's/\r$//' "$SP/unsloth_zoo/vllm_utils.py"
    apply_patch "$HERE/unsloth_zoo_vllm_utils_gemma4_31b.patch" "$SP/unsloth_zoo" 0
fi

check_archs
"$PY" -c "import trl, vllm; from importlib.metadata import version; print('unsloth', version('unsloth'), '| trl', trl.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_unsloth.py --help"
