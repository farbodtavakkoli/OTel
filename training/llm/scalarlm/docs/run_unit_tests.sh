#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$HERE/repo" ]; then
    SCALARLM_DIR="$HERE"
elif [ -d "$HERE/../repo" ]; then
    SCALARLM_DIR="$(cd "$HERE/.." && pwd)"
else
    SCALARLM_DIR="$HERE"
fi

IMAGE_REPO="${IMAGE_REPO:-farbodatdocker/scalarlm}"

if [ $# -ge 1 ]; then
    TAG="$1"
elif [ -d /opt/rocm ] || command -v rocm-smi >/dev/null 2>&1; then
    TAG="${IMAGE_TAG:-mi355-v1.8}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    TAG="${IMAGE_TAG:-h100-v1.9}"
else
    echo "ERROR: cannot detect GPU vendor; pass a tag, e.g. ./run_unit_tests.sh mi355-v1.8"
    exit 1
fi
IMAGE="$IMAGE_REPO:$TAG"

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

docker run --rm "${GPU_ARGS[@]}" "${MOUNT_ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /app/cray
    python3 -m pytest --version >/dev/null 2>&1 || \
        pip install --no-cache-dir -q -r test/requirements-pytest.txt
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29555} \
    python3 -m pytest test/unit -q $DESELECT
"
