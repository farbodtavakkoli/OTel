#!/usr/bin/env bash
# Shared helpers for the per-stack setup.sh scripts. Sourced, not run directly.

set -euo pipefail

GRPO_ENV_DIR="${GRPO_ENV_DIR:-./envs}"
GRPO_VLLM_VERSION="${GRPO_VLLM_VERSION:-0.29.0}"

# make_venv <stack> -> creates the venv and sets $PY / $PIP
make_venv() {
    local stack="$1" dir="$GRPO_ENV_DIR/env_$1"
    mkdir -p "$GRPO_ENV_DIR"
    [ -d "$dir" ] || python3 -m venv "$dir"
    PY="$dir/bin/python"; PIP="$dir/bin/pip"
    # ninja is a run-time dep: vLLM's FlashInfer backend JIT-compiles kernels on first use.
    "$PIP" install -q -U pip wheel ninja
    echo "[setup] venv $dir"
}

# install_vllm [version] -> vLLM first, so it pins torch and the accelerator runtime once.
install_vllm() {
    local want="${1:-$GRPO_VLLM_VERSION}"
    "$PY" -c "import vllm" 2>/dev/null && { echo "[setup] vllm present"; return; }
    "$PIP" install -q "vllm==$want"
    echo "[setup] vllm $want"
}

# apply_patch <patch> <target_dir> [strip] -> applies, or reports already-applied, or aborts.
# "already applied" and "does not apply" must not both count as success: for several of these
# patches a silently missing patch means the run completes while training nothing.
apply_patch() {
    local patch="$1" dir="$2" strip="${3:-1}" name
    name="$(basename "$patch")"
    if ( cd "$dir" && patch -p"$strip" --dry-run --silent --forward < "$patch" ) >/dev/null 2>&1; then
        ( cd "$dir" && patch -p"$strip" --forward --silent < "$patch" )
        echo "[setup] $name: applied"
    elif ( cd "$dir" && patch -p"$strip" --dry-run --reverse --silent < "$patch" ) >/dev/null 2>&1; then
        echo "[setup] $name: already applied"
    else
        echo "[setup] ERROR: $name does not apply in $dir and is not already applied." >&2
        echo "[setup] The pinned upstream version may have moved. Do not run without it." >&2
        return 1
    fi
}

# site_packages -> the venv's site-packages, for patches that edit installed packages
site_packages() { "$PY" -c "import site; print(site.getsitepackages()[0])"; }

# check_archs -> fail early if this vLLM cannot serve the policy and judge architectures
check_archs() {
    "$PY" - <<'EOF'
import vllm
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
missing = [a for a in ("Gemma4ForConditionalGeneration", "Qwen3_5ForConditionalGeneration")
           if a not in archs]
print("vllm", vllm.__version__, "| missing archs:", missing or "none")
raise SystemExit(1 if missing else 0)
EOF
}
