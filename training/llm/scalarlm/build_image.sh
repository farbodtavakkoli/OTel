#!/usr/bin/env bash
# Build the ScalarLM server image from the vendored source in repo/.
# Copies ml/ into repo/ before building, so the image always bakes the current
# training recipe. repo/ml/ is a build artifact and is gitignored.
#
# Usage:
#   ./build_image.sh                              # auto-detect target, build + label
#   TARGET=amd    IMAGE_TAG=mi355-v1.6 ./build_image.sh
#   TARGET=nvidia IMAGE_TAG=h100-v1.5  ./build_image.sh
#
# Targets:
#   amd     MI355X (gfx950/ROCm). Runbook: DOCKER_IMAGE_MI355.md.
#   nvidia  H100 (sm_90/CUDA). Runbook: DOCKER_IMAGE_H100.md. This Dockerfile route
#           builds and is verified on an 8xH100 host (vLLM compiled sm_90, merged
#           server native); needs docker buildx + DOCKER_BUILDKIT=1 (the Dockerfile
#           uses BuildKit RUN --mount cache syntax).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/repo"
ML_DIR="$HERE/ml"

TARGET="${TARGET:-auto}"
if [ "$TARGET" = auto ]; then
    if [ -d /opt/rocm ] || command -v rocm-smi >/dev/null 2>&1; then TARGET=amd
    elif command -v nvidia-smi >/dev/null 2>&1; then TARGET=nvidia
    else echo "ERROR: cannot auto-detect GPU vendor; set TARGET=amd or TARGET=nvidia"; exit 1; fi
fi
case "$TARGET" in
    amd)
        BASE_NAME=amd;    ARCH_LIST=gfx942; VLLM_DEVICE=rocm
        DEFAULT_TAG=mi355-v1.6; DEFAULT_STAGE=cray:mi355-build
        TITLE="ScalarLM MI355X"
        DESCRIPTION="ScalarLM training+inference for AMD Instinct MI355X (gfx950/ROCm): NCCL/RCCL collectives, DDP+FSDP+FSDP2 verified on 8 GPUs" ;;
    nvidia)
        BASE_NAME=nvidia; ARCH_LIST=9.0;    VLLM_DEVICE=cuda
        DEFAULT_TAG=h100-v1.5;  DEFAULT_STAGE=cray:h100-build
        TITLE="ScalarLM H100"
        DESCRIPTION="ScalarLM training+inference for NVIDIA H100 (sm_90/CUDA): NCCL collectives, DDP+FSDP+FSDP2" ;;
    *)  echo "ERROR: TARGET must be amd or nvidia (got '$TARGET')"; exit 1 ;;
esac

IMAGE_REPO="${IMAGE_REPO:-farbodatdocker/scalarlm}"
IMAGE_TAG="${IMAGE_TAG:-$DEFAULT_TAG}"
IMAGE="$IMAGE_REPO:$IMAGE_TAG"
BUILD_STAGE="${BUILD_STAGE:-$DEFAULT_STAGE}"
echo "==> Target: $TARGET  ->  $IMAGE"

[ -d "$REPO_DIR" ] || { echo "ERROR: $REPO_DIR not found (vendored server source)"; exit 1; }
[ -d "$ML_DIR" ]   || { echo "ERROR: $ML_DIR not found (the training recipe)"; exit 1; }

# The Dockerfile bind-mounts ./vllm. It must exist or the build dies with
#   failed to compute cache key: "/vllm": not found
# It is intentionally empty: VLLM_SOURCE defaults to 'remote' and vLLM is cloned
# at the commit pinned in the Dockerfile.
mkdir -p "$REPO_DIR/vllm"

echo "==> Staging ml/ into repo/ (single source of truth: $ML_DIR)"
rm -rf "$REPO_DIR/ml"
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$ML_DIR/" "$REPO_DIR/ml/"
echo "    $(find "$REPO_DIR/ml" -name '*.py' | wc -l) python files staged"

# Record which source revision this image was built from. The vendored tree has
# no .git of its own, so this is the OUTER repository's HEAD -- which is the
# correct answer, because that is now where this source lives.
REVISION="$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY=""
if ! git -C "$HERE" diff --quiet HEAD -- "$HERE" 2>/dev/null; then
    DIRTY="-dirty"
    echo "    WARNING: working tree has uncommitted changes; labelling revision as ${REVISION}${DIRTY}"
fi

echo "==> Building $BUILD_STAGE from $REPO_DIR"
# --network=host: the in-build vLLM clone fails TLS without it.
# On amd, TORCH_CUDA_ARCH_LIST=gfx942 is an inherited build-arg NAME; inside the
# Dockerfile PYTORCH_ROCM_ARCH governs and covers gfx950. Do not "fix" it to
# gfx950 without re-running the arch check in DOCKER_IMAGE_MI355.md section 2.
docker build --network=host --platform linux/amd64 \
    --build-arg BASE_NAME="$BASE_NAME" \
    --build-arg TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
    --build-arg VLLM_TARGET_DEVICE="$VLLM_DEVICE" \
    -t "$BUILD_STAGE" \
    "$REPO_DIR"

echo "==> Adding entrypoint + provenance labels (metadata only, no layers rebuilt)"
BUILDDIR="$(mktemp -d)"
trap 'rm -rf "$BUILDDIR"' EXIT
MAINTAINER_NAME="${MAINTAINER_NAME:-Farbod Tavakkoli}"
MAINTAINER_EMAIL="${MAINTAINER_EMAIL:-farbodtavakoli@gmail.com}"

cat > "$BUILDDIR/Dockerfile" <<EOF
FROM $BUILD_STAGE
CMD ["/app/cray/scripts/start_one_server.sh"]
LABEL org.opencontainers.image.title="${TITLE}"
LABEL org.opencontainers.image.description="${DESCRIPTION}"
LABEL org.opencontainers.image.revision="${REVISION}${DIRTY}"
LABEL org.opencontainers.image.authors="${MAINTAINER_NAME} <${MAINTAINER_EMAIL}>"
LABEL maintainer="${MAINTAINER_NAME} <${MAINTAINER_EMAIL}>"
LABEL org.opencontainers.image.source="https://github.com/farbodtavakkoli/training_junk"
LABEL org.opencontainers.image.licenses="CC0-1.0"
LABEL org.opencontainers.image.version="${IMAGE_TAG}"
EOF
docker build -t "$IMAGE" "$BUILDDIR"

echo
echo "==> Built $IMAGE"
echo "    revision: $(docker inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
echo "    size:     $(docker images --format '{{.Size}}' "$IMAGE" | head -1)"
echo
RUNBOOK=DOCKER_IMAGE_MI355.md; [ "$TARGET" = nvidia ] && RUNBOOK=DOCKER_IMAGE_H100.md
echo "Next: run the acceptance gates in $RUNBOOK before publishing."
echo "At minimum, one FSDP and one DDP run whose final loss matches the fingerprints there."
