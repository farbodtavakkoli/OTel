#!/usr/bin/env bash
# Build grpo-base:<IMAGE_TAG> (ROCm torch + vLLM fork + training pins, no ScalarLM). Usage: [IMAGE_TAG=mi355] build_base_image.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/../scalarlm/repo"
IMAGE_TAG="${IMAGE_TAG:-mi355}"
IMAGE="${IMAGE_REPO:-grpo-base}:$IMAGE_TAG"
VLLM_STAGE="grpo-base-vllm-stage:$IMAGE_TAG"

[ -f "$REPO_DIR/Dockerfile" ] || { echo "ERROR: $REPO_DIR/Dockerfile not found"; exit 1; }
mkdir -p "$REPO_DIR/vllm"   # the Dockerfile bind-mounts ./vllm; empty means clone the pinned commit

echo "==> Phase 1: vllm stage -> $VLLM_STAGE"
DOCKER_BUILDKIT=1 docker build --network=host --platform linux/amd64 \
    --build-arg BASE_NAME=amd \
    --build-arg TORCH_CUDA_ARCH_LIST=gfx942 \
    --build-arg VLLM_TARGET_DEVICE=rocm \
    --target vllm -t "$VLLM_STAGE" "$REPO_DIR"

echo "==> Phase 2: $IMAGE"
REVISION="$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
git -C "$HERE" diff --quiet HEAD -- "$REPO_DIR" "$HERE/Dockerfile.base_mi355" 2>/dev/null || REVISION="${REVISION}-dirty"
DOCKER_BUILDKIT=1 docker build --network=host --platform linux/amd64 \
    --build-arg VLLM_STAGE="$VLLM_STAGE" \
    --label org.opencontainers.image.title="grpo-base MI355X" \
    --label org.opencontainers.image.description="ROCm torch + vLLM fork + transformers/peft pins for the GRPO benchmark stacks (no ScalarLM)" \
    --label org.opencontainers.image.revision="$REVISION" \
    --label org.opencontainers.image.version="$IMAGE_TAG" \
    -f "$HERE/Dockerfile.base_mi355" -t "$IMAGE" "$REPO_DIR"

echo "==> Built $IMAGE ($(docker images --format '{{.Size}}' "$IMAGE" | head -1)), revision $REVISION"
