#!/usr/bin/env bash
# Acceptance gate 1 — run the in-image unit suite correctly.
#
# Run it against a built image:
#   ./run_unit_tests.sh                     # defaults to the current tag for this host
#   ./run_unit_tests.sh mi355-v1.7
#   ./run_unit_tests.sh h100-v1.6
#   WITH_CMD=1 ./run_unit_tests.sh          # also run the two cmd/ tests (see below)
#
# Expected: 811 passed, 0 failed. With WITH_CMD=1: 813 passed.
#
# Two things must be set up or the suite reports 18 false failures. Both are
# harness requirements, not product defects:
#
#   1. Rendezvous environment. The FSDP wrap path calls get_rank(), which
#      initialises torch.distributed through the env:// rendezvous and needs
#      RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Those are
#      normally set by torchrun, so a bare `pytest` fails 16 fsdp and
#      pytorch_fsdp tests with
#        ValueError: environment variable RANK expected, but not set
#      ddp is unaffected because it does not initialise at wrap time.
#
#   2. cmd/ is not in the image. test_live_test_command.py reads
#      /app/cray/cmd/test_command.sh, and the Dockerfile copies infra, sdk,
#      test, ml and scripts but not cmd/ — that is a host-side developer CLI
#      which is deliberately not shipped. Two of its four tests need the file.
#      They are deselected by default so the gate needs no source mount. Set
#      WITH_CMD=1 to bind-mount repo/cmd and run them too.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script lives in docs/, while repo/ sits in the scalarlm folder above it.
# Resolve either layout so it keeps working if moved back alongside repo/.
if [ -d "$HERE/repo" ]; then
    SCALARLM_DIR="$HERE"
elif [ -d "$HERE/../repo" ]; then
    SCALARLM_DIR="$(cd "$HERE/.." && pwd)"
else
    SCALARLM_DIR="$HERE"   # only matters for WITH_CMD=1, which checks the path below
fi

IMAGE_REPO="${IMAGE_REPO:-farbodatdocker/scalarlm}"

# Pick the tag matching this host unless one was given.
if [ $# -ge 1 ]; then
    TAG="$1"
elif [ -d /opt/rocm ] || command -v rocm-smi >/dev/null 2>&1; then
    TAG="${IMAGE_TAG:-mi355-v1.7}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    TAG="${IMAGE_TAG:-h100-v1.6}"
else
    echo "ERROR: cannot detect GPU vendor; pass a tag, e.g. ./run_unit_tests.sh mi355-v1.7"
    exit 1
fi
IMAGE="$IMAGE_REPO:$TAG"

# GPU passthrough differs by vendor; the fsdp tests need a real device.
case "$TAG" in
    mi355-*) GPU_ARGS=(--device /dev/kfd --device /dev/dri --group-add video
                       --security-opt seccomp=unconfined) ;;
    h100-*)  GPU_ARGS=(--gpus all) ;;
    *)       echo "ERROR: cannot infer GPU flags from tag '$TAG'"; exit 1 ;;
esac

MOUNT_ARGS=()
DESELECT="--deselect test/unit/test_live_test_command.py::test_live_profile_stops_when_image_build_fails \
          --deselect test/unit/test_live_test_command.py::test_live_profile_forwards_pytest_filters"
if [ "${WITH_CMD:-0}" = "1" ]; then
    [ -d "$SCALARLM_DIR/repo/cmd" ] || { echo "ERROR: $SCALARLM_DIR/repo/cmd not found"; exit 1; }
    MOUNT_ARGS=(-v "$SCALARLM_DIR/repo/cmd:/app/cray/cmd:ro")
    DESELECT=""
    echo "==> WITH_CMD=1: bind-mounting repo/cmd so the two cmd/ tests can run"
fi

echo "==> Gate 1: unit suite in $IMAGE"

# pytest is a test-harness dependency and is not installed in the image, so
# install it in the throwaway container before collecting.
docker run --rm "${GPU_ARGS[@]}" "${MOUNT_ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /app/cray
    python3 -m pytest --version >/dev/null 2>&1 || \
        pip install --no-cache-dir -q -r test/requirements-pytest.txt
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29555} \
    python3 -m pytest test/unit -q $DESELECT
"
