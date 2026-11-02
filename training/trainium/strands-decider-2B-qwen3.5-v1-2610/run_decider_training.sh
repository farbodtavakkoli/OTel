#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DECIDER_WORK:?Set DECIDER_WORK to the directory created by setup_trainium_environment.py}"
DECIDER_WORK="$(cd -- "$DECIDER_WORK" && pwd)"
unset PYTHONPATH PYTHONHOME
PYTHON="$DECIDER_WORK/venv/bin/python"
BASE_BIN="$("$PYTHON" -c 'import json, pathlib, sys; print(pathlib.Path(json.load(open(sys.argv[1]))["base_python"]).parent)' "$DECIDER_WORK/environment.json")"
export PATH="$DECIDER_WORK/venv/bin:$BASE_BIN:/opt/aws/neuron/bin:$PATH"
export HF_HOME="$DECIDER_WORK/cache/huggingface"
export TORCH_NEURONX_NEFF_CACHE_DIR="$DECIDER_WORK/compiled"
export TORCH_NEURONX_HLO_CACHE_DIR="$DECIDER_WORK/compiled/hlo"
export TORCH_NEURONX_DEBUG_DIR="$DECIDER_WORK/compiled/debug"
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR="$DECIDER_WORK/compiled/local"
mkdir -p "$DECIDER_WORK/cache/tmp"
export TMPDIR="$DECIDER_WORK/cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$DECIDER_WORK/compiled/inductor"
export XDG_CACHE_HOME="$DECIDER_WORK/cache"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
cd -- "$DECIDER_WORK"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  preflight) exec "$PYTHON" "$PACKAGE/tests/probe_trainium_compilation.py" "$@" ;;
  gdn-probe) exec "$PYTHON" "$PACKAGE/tests/probe_trainium_gated_delta.py" "$@" ;;
  fsdp-probe) exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 "$PACKAGE/tests/probe_trainium_fsdp.py" "$@" ;;
  prepare) exec "$PYTHON" "$PACKAGE/prepare_decider_model.py" --upstream "$DECIDER_WORK/upstream" --cache "$HF_HOME" "$@" ;;
  train) export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
         exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 "$PACKAGE/train_decider_full.py" --upstream "$DECIDER_WORK/upstream" "$@" ;;
  verify) export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
          exec "$PYTHON" "$PACKAGE/verify_decider_training.py" --upstream "$DECIDER_WORK/upstream" "$@" ;;
  compare) exec "$PYTHON" "$PACKAGE/tests/compare_decider_checkpoints.py" "$@" ;;
  *) echo 'Usage: DECIDER_WORK=/path/to/work bash run_decider_training.sh {preflight|gdn-probe|fsdp-probe|prepare|train|verify|compare} [arguments]'; exit 2 ;;
esac
