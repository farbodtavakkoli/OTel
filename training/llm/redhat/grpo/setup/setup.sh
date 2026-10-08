#!/usr/bin/env bash
# Red Hat training-hub GRPO environment (verl backend). Usage: [GRPO_ENV_DIR=...] ./setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

VERL_COMMIT="${VERL_COMMIT:-d040717b21af2e23e8e789a3e354cff2394ae2de}"
TRAINING_HUB_COMMIT="${TRAINING_HUB_COMMIT:-70640575c328df6ec04e7a547b3405726f238538}"
VERL_SRC="${GRPO_ENV_DIR}/verl_src_traininghub"
TH_SRC="${GRPO_ENV_DIR}/training_hub_src"

make_venv redhat_traininghub
# vLLM 0.28.0, not the shared default: verl d040717b's LoRA hot-load path calls
# WeightsMapper.get_unstacked_mapper(), which 0.29 renamed to get_rename_mapper. On 0.29 every
# rollout worker fails with AttributeError after the first weight sync.
install_vllm 0.28.0

[ -d "$VERL_SRC" ] || git clone https://github.com/volcengine/verl.git "$VERL_SRC"
git -C "$VERL_SRC" checkout -q "$VERL_COMMIT"
"$PIP" install -q --no-deps -e "$VERL_SRC"

[ -d "$TH_SRC" ] || git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git "$TH_SRC"
git -C "$TH_SRC" checkout -q "$TRAINING_HUB_COMMIT"
"$PIP" install -q --no-deps -e "$TH_SRC"
"$PIP" install -q -r "$HERE/requirements.txt"

# Same LoRA hot-load fix as the verl stack (training-hub drives the same verl trainer).
apply_patch "$HERE/verl_scalarlm_vllm_fork_lora_redhat.patch" "$VERL_SRC"

check_archs
"$PY" -c "import training_hub, verl, vllm; from training_hub import lora_grpo; print('training_hub ok | verl', verl.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_redhat.py --help"
