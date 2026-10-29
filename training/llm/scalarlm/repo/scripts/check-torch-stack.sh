#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-check}"
: "${EXPECT_TORCH_VERSION:?EXPECT_TORCH_VERSION not set}"
: "${TORCH_CUDA_INDEX:?TORCH_CUDA_INDEX not set}"

python - <<'EOF'
import importlib.metadata as md
import os
import sys

import torch

expect = os.environ["EXPECT_TORCH_VERSION"]
index = os.environ["TORCH_CUDA_INDEX"]
want_cuda = f"{index[2:-1]}.{index[-1]}"
errors = []
if torch.__version__ != expect:
    errors.append(f"torch is {torch.__version__}, expected {expect}")
if not (torch.version.cuda or "").startswith(want_cuda):
    errors.append(f"torch CUDA is {torch.version.cuda}, expected {want_cuda}")

major = want_cuda.split(".")[0]
other = "13" if major == "12" else "12"
runtime_prefixes = ("nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc",
                    "nvidia-cudnn", "nvidia-nccl", "nvidia-cusparse", "nvidia-cufft",
                    "nvidia-curand", "nvidia-cusolver", "nvidia-nvjitlink")
for dist in md.distributions():
    name = (dist.metadata["Name"] or "").lower()
    ver = dist.version
    if name == "cuda-toolkit" and not ver.startswith(major + "."):
        errors.append(f"cuda-toolkit {ver} installed (torch uses CUDA {want_cuda})")
    if name.startswith(runtime_prefixes) and name.endswith("-cu" + other):
        errors.append(f"{name} {ver} installed (CUDA {other} runtime wheel)")
    if name in ("cuda-bindings", "cuda-python") and not ver.startswith(major + "."):
        errors.append(f"{name} {ver} installed (torch uses CUDA {want_cuda})")

if errors:
    print("ERROR: torch stack check failed:\n  " + "\n  ".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"torch stack OK: torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"triton {md.version('triton')}")
EOF

set +e
PIP_CHECK="$(python -m pip check 2>&1)"
set -e
if [ -n "$PIP_CHECK" ] && ! echo "$PIP_CHECK" | grep -q '^No broken requirements'; then
    if echo "$PIP_CHECK" | grep -Ei '(^| )(torch|torchvision|torchaudio|torchcodec|triton|cuda-bindings|cuda-python|cuda-toolkit|nvidia-[a-z-]+|vllm|flashinfer-python) '; then
        echo "ERROR: pip check reports conflicts in the torch/CUDA stack (above)" >&2
        exit 1
    fi
    echo "WARNING: pip check (non-stack conflicts, not fatal):"
    echo "$PIP_CHECK"
fi
echo "[check-torch-stack] ${STAGE}: OK"
