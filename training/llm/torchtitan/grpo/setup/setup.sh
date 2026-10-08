#!/usr/bin/env bash
# TorchTitan RL (TitanRL) GRPO environment. Usage: [GRPO_ENV_DIR=...] ./setup/setup.sh
# The outlier stack: full-parameter (no LoRA), runs Qwen3-4B, and torchtitan is run from a
# source checkout rather than pip-installed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../grpo_common/setup_common.sh"

TT_COMMIT="${TT_COMMIT:-57767a8ba37268bf4a2f8c7904ad13b4c950a958}"
TS_COMMIT="${TS_COMMIT:-8fe02264ae482c4a43cb466acbbb0e0b0221a5a1}"
TT_SRC="${GRPO_ENV_DIR}/torchtitan_src"
TS_SRC="${GRPO_ENV_DIR}/torchstore_src"

make_venv torchtitan_rl
install_vllm

[ -d "$TT_SRC" ] || git clone https://github.com/pytorch/torchtitan "$TT_SRC"
git -C "$TT_SRC" checkout -q "$TT_COMMIT"
"$PIP" install -q -r "$TT_SRC/requirements.txt"
"$PIP" install -q -r "$TT_SRC/torchtitan/experiments/rl/requirements.txt"
"$PIP" install -q -r "$TT_SRC/torchtitan/experiments/rl/examples/dapo_math/requirements.txt"

# torchstore must come from this git commit. The PyPI package of the same name is unrelated and
# pulls an incompatible torch.
[ -d "$TS_SRC" ] || git clone https://github.com/meta-pytorch/torchstore.git "$TS_SRC"
git -C "$TS_SRC" checkout -q "$TS_COMMIT"
"$PIP" wheel -q --no-deps -w "$GRPO_ENV_DIR/ts_wheel" "$TS_SRC"
"$PIP" install -q --no-deps "$GRPO_ENV_DIR"/ts_wheel/torchstore-*.whl

# requirements.txt restores torchmonarch 0.6.0, which torchstore's pin downgrades, and adds the
# apache-tvm-ffi version all of vLLM/tilelang/tokenspeed-mla accept.
"$PIP" install -q -r "$HERE/requirements.txt"

"$PY" -c "import torch, vllm, monarch, torchstore; print('torch', torch.__version__, '| vllm', vllm.__version__, '| devices', torch.cuda.device_count())"
( cd "$TT_SRC" && "$PY" -c "import torchtitan, torchtitan.experiments.rl; print('torchtitan + experiments.rl import OK')" )
cat <<'NOTE'
[setup] done. Two run-time requirements for this stack:
  * TITAN_ATTN_BACKEND=flex   - the default `varlen` backend routes the generator to vLLM's CUSTOM
    backend, which on CUDA SM 9.0 (Hopper) demands FlashAttention-3 and fails without a
    from-source flash_attn_interface build. `flex` is torchtitan's own Qwen3 default.
  * the venv's bin/ on PATH    - FlexAttention compiles an inductor kernel at run time (needs ninja).
NOTE
