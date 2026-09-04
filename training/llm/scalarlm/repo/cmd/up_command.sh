inspect_args

target=${args[target]}
sm_arch=${args[sm_arch]}

declare -a vllm_target_device
declare -a docker_compose_service
declare -a docker_platform

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_compose_service="cray"
    if [ "$(uname -m)" == "x86_64" ]; then
        docker_platform=("linux/amd64")
    else
        docker_platform=("linux/arm64/v8")
    fi
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_compose_service="cray-amd"
    docker_platform="linux/amd64"
    sm_arch="gfx942"
elif [ "$target" == "spark" ]; then
    # NVIDIA DGX Spark: aarch64 Grace CPU + Blackwell GPU (SM 12.0).
    vllm_target_device=("cuda")
    docker_compose_service="cray-spark"
    docker_platform="linux/arm64"
    if [ "$sm_arch" == "auto" ]; then
        sm_arch="12.0"
    fi
else
    vllm_target_device=("cuda")
    docker_compose_service="cray-nvidia"
    docker_platform="linux/amd64"
    if [ "$sm_arch" == "auto" ]; then
        echo "Autodetect sm_arch"
        # Auto-detect the architecture of the GPU using nvidia-smi
        sm_arch=($(nvidia-smi --query-gpu=compute_cap --format=csv,noheader))
    fi
fi

mkdir -p models
mkdir -p vllm
mkdir -p chat-ui

echo "SM arch is ${sm_arch}"

# Explicit -f (not Compose's native override auto-discovery) keeps
# compose-file selection deterministic: an inherited COMPOSE_FILE env var,
# or a stray .env in this directory, would otherwise silently redirect
# `docker compose` at an unrelated project when no -f is given at all.
# Check both override extensions since Compose itself accepts either -- but
# since both names are gitignored (so a stale one is invisible to `git
# status`), fail loudly if both exist instead of silently picking one and
# layering them, which would concatenate some config unexpectedly.
compose_files=(-f docker-compose.yaml)
if [ -f docker-compose.override.yaml ] && [ -f docker-compose.override.yml ]; then
    echo "Both docker-compose.override.yaml and docker-compose.override.yml exist; remove one." >&2
    exit 1
elif [ -f docker-compose.override.yaml ]; then
    compose_files+=(-f docker-compose.override.yaml)
elif [ -f docker-compose.override.yml ]; then
    compose_files+=(-f docker-compose.override.yml)
fi

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} \
    DOCKER_PLATFORM=${docker_platform} TORCH_CUDA_ARCH_LIST=${sm_arch} \
    docker compose "${compose_files[@]}" up ${docker_compose_service} --build --force-recreate
