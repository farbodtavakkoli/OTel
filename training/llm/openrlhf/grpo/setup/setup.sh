#!/usr/bin/env bash
# OpenRLHF GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

make_venv openrlhf
install_vllm
# --no-deps: openrlhf 0.11.0's own pins would replace the torch vLLM installed.
"$PIP" install -q --no-deps openrlhf==0.11.0
"$PIP" install -q -r "$HERE/requirements.txt"

# flash-attn last, and on its own line: it needs --no-build-isolation (its setup.py imports the
# already-installed torch), which cannot be expressed in requirements.txt. openrlhf.models imports
# flash_attn at module scope through ring_attn_utils, so it is required even for an sdpa run with no
# ring attention. Expect a long from-source compile if no wheel matches this torch build.
"$PIP" install -q flash-attn --no-build-isolation

# REQUIRED, not optional: stock openrlhf 0.11.0 never syncs LoRA weights to vLLM, so the rollout
# policy silently stays at the base model and the run looks successful while learning nothing.
# The patch also adds `re:<regex>` LoRA targeting so a vision tower stays frozen.
# The script locates the installed openrlhf itself, so run it on THIS venv's interpreter;
# --check re-runs it as a gate and exits non-zero if any edit did not land.
"$PY" "$HERE/patch_openrlhf_lora_sync.py"
"$PY" "$HERE/patch_openrlhf_lora_sync.py" --check

check_archs
"$PY" -c "import openrlhf, deepspeed, ray, vllm, peft; print('openrlhf ok | deepspeed', deepspeed.__version__, '| ray', ray.__version__)"
echo "[setup] done. Run with: $PY $HERE/../train_llm_grpo_openrlhf.py --help"
