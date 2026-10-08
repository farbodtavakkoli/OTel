#!/usr/bin/env bash
# RapidFire AI GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

make_venv rapidfire
# vLLM 0.27.1, not the shared default: trl 1.12 declares `vllm<=0.27.1` and imports a symbol
# from vLLM's nccl weight-transfer engine that 0.28 removed (GRPOTrainer fails to import).
install_vllm 0.27.1
# --no-deps for rapidfireai: 0.16.1 pins datasets==3.6.0, which conflicts with the trl this stack
# needs (>=4.7.0) and would replace the vLLM torch. Its real dependencies are in requirements.txt.
"$PIP" install -q --no-deps rapidfireai==0.16.1
"$PIP" install -q -r "$HERE/requirements.txt"
# RapidFire expects this marker instead of running `rapidfireai init`, which would reinstall torch.
echo fit > "$(site_packages)/rapidfireai/rf_mode.txt" 2>/dev/null || true

# starlette 1.6 does a bare `import anyio` then calls anyio.from_thread.run(); anyio 4.15's lazy
# importer does not resolve submodules that way, so every mlflow request 500s. mlflow-skinny
# declares anyio!=4.15.0 for this reason. One-line fix, kept idempotent.
W="$(site_packages)/starlette/middleware/wsgi.py"
if [ -f "$W" ] && ! grep -q "^import anyio.from_thread" "$W"; then
    cp -n "$W" "$W.orig"
    sed -i '0,/^import anyio$/s//import anyio\nimport anyio.from_thread\nimport anyio.to_thread/' "$W"
fi

check_archs
"$PY" -c "import rapidfireai, trl, vllm, mlflow; print('rapidfireai ok | trl', trl.__version__, '| mlflow', mlflow.__version__)"
echo "[setup] done. Needs --shm-size 100g and a high nofile ulimit at run time; see ../README.md."
