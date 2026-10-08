#!/usr/bin/env bash
# verl GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

VERL_COMMIT="${VERL_COMMIT:-d040717b21af2e23e8e789a3e354cff2394ae2de}"
SRC="${GRPO_ENV_DIR}/verl_src"

make_venv verl
# vLLM 0.28.0, not the shared default: verl d040717b's LoRA hot-load path calls
# WeightsMapper.get_unstacked_mapper(), which 0.29 renamed to get_rename_mapper. On 0.29 every
# rollout worker dies with AttributeError after the first weight sync.
install_vllm 0.28.0

# verl from git at the benchmarked commit, --no-deps so it cannot replace the torch vLLM pinned.
[ -d "$SRC" ] || git clone https://github.com/volcengine/verl.git "$SRC"
git -C "$SRC" checkout -q "$VERL_COMMIT"
"$PIP" install -q --no-deps -e "$SRC"
"$PIP" install -q -r "$HERE/requirements.txt"

# LoRA hot-load fix. verl's VLLMHijack assumes vLLM's stock adapter manager; a vLLM whose
# adapter manager opens LoRA paths before the hijack silently drops the adapter, so the rollout
# policy never updates. Harmless where it does not apply (patch reports "already applied").
apply_patch "$HERE/verl_scalarlm_vllm_fork_lora.patch" "$SRC"

# Optional: DAPO-style dynamic sampling (drop zero-variance groups and refill). Inactive unless
# the run passes --override algorithm.filter_groups.enable=True.
apply_patch "$HERE/verl_classic_trainer_filter_groups.patch" "$SRC"

check_archs
"$PY" -c "import verl, vllm, transformers, peft, ray, tensordict; print('verl', verl.__version__, '| tensordict', tensordict.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_verl.py --help"
