ARG BASE_NAME=cpu

ARG CUDA_VERSION=12.9.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS nvidia

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        git ca-certificates curl pkg-config cmake ninja-build \
        openmpi-bin libopenmpi-dev libpmix-dev \
        slurm-wlm libslurm-dev \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
        libdw-dev libibverbs-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

ARG EFA_INSTALLER_VERSION=1.51.0
RUN cd /tmp && curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz | tar xz && \
    cd aws-efa-installer && apt-get update -y && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi --disable-ngc && \
    cd / && rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/* && ldconfig && \
    /opt/amazon/efa/bin/fi_info --version && \
    for f in /opt/amazon/ofi-nccl/lib/*.so; do patchelf --force-rpath --set-rpath /opt/amazon/efa/lib "$f"; done && \
    ldd /opt/amazon/ofi-nccl/lib/libnccl-net.so | grep -q "libfabric.so.1 => /opt/amazon/efa/lib/" && \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ldd /opt/amazon/ofi-nccl/lib/libnccl-net.so | grep -q "libfabric.so.1 => /opt/amazon/efa/lib/"
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV PATH="/opt/amazon/efa/bin:$PATH"
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV

ARG TORCH_VERSION="2.13.0"
ARG TORCHVISION_VERSION="0.28.0"
ARG TORCHAUDIO_VERSION="2.11.0"
ARG TORCHCODEC_VERSION="0.16.0"
ARG TORCH_CUDA_INDEX=cu129
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv && \
    uv pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        torchcodec==${TORCHCODEC_VERSION} \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX} && \
    uv pip install ninja "protobuf>=6.30.0" && \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.__version__.endswith('+${TORCH_CUDA_INDEX}'), torch.__version__"

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib/python3.12/site-packages/torch/lib

ARG OPENCV_PYTHON_VERSION=5.0.0.93
RUN mkdir -p /opt/opencv-wheels && \
    ENABLE_HEADLESS=1 CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    pip wheel --no-deps --no-binary opencv-python-headless \
        opencv-python-headless==${OPENCV_PYTHON_VERSION} -w /opt/opencv-wheels && \
    ls -la /opt/opencv-wheels

RUN printf '%s\n' \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_INDEX}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_INDEX}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_INDEX}" \
        "torchcodec==${TORCHCODEC_VERSION}+${TORCH_CUDA_INDEX}" \
        "triton==$(python -c 'import importlib.metadata as m; print(m.version("triton"))')" \
        > /app/torch-constraints.txt && \
    if [ "${TORCH_CUDA_INDEX#cu12}" != "${TORCH_CUDA_INDEX}" ]; then \
        printf '%s\n' "cuda-bindings<13" "cuda-python<13" >> /app/torch-constraints.txt; \
    fi && \
    cat /app/torch-constraints.txt
ENV PIP_CONSTRAINT=/app/torch-constraints.txt
ENV UV_CONSTRAINT=/app/torch-constraints.txt
ENV TORCH_CUDA_INDEX=${TORCH_CUDA_INDEX}
ENV EXPECT_TORCH_VERSION=${TORCH_VERSION}+${TORCH_CUDA_INDEX}
COPY scripts/check-torch-stack.sh /app/check-torch-stack.sh
RUN chmod +x /app/check-torch-stack.sh && /app/check-torch-stack.sh base

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

ENV TORCHINDUCTOR_MAX_AUTOTUNE=0
ENV TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
ENV TORCH_COMPILE_DISABLE=1

FROM nvidia AS spark

ENV BASE_NAME=spark

FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv \
    openmpi-bin libopenmpi-dev libpmix-dev slurm-wlm libslurm-dev \
    cmake

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate

ARG TORCH_VERSION="2.13.0"
ARG TORCHVISION_VERSION="0.28.0"
ARG TORCHAUDIO_VERSION="2.11.0"
ARG TORCHCODEC_VERSION="0.14.0"

RUN pip install uv && \
    uv pip install \
        torch==${TORCH_VERSION}+cpu \
        torchvision==${TORCHVISION_VERSION}+cpu \
        torchaudio==${TORCHAUDIO_VERSION}+cpu \
        torchcodec==${TORCHCODEC_VERSION}+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install ninja

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib64/python3.12/site-packages/torch/lib

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

FROM rocm/primus:v26.5 AS amd

ENV BASE_NAME=amd

ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV PATH="/opt/venv/bin:/opt/openmpi/bin:${PATH}"
RUN ln -sfn /opt/openmpi /opt/ompi-rocm && \
    ln -sfn /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel /opt/rocm && \
    pip install uv && uv pip install ninja "setuptools>=77"

RUN ln -sfn /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi/amdsmi \
        /opt/venv/lib/python3.12/site-packages/amdsmi
RUN apt-get update -y && apt-get install -y \
        slurm-wlm libslurm-dev libpmix-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:${LD_LIBRARY_PATH:-}:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/openmpi/lib

FROM public.ecr.aws/docker/library/node:24.2.0 AS scalarlm_ui_builder

WORKDIR /build

COPY ui/package.json ui/package-lock.json* ./

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY ui/ .
RUN npm run build

FROM ${BASE_NAME} AS vllm

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl git ccache vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && if [ "$BASE_NAME" = "nvidia" ] || [ "$BASE_NAME" = "spark" ]; then \
        apt-get install -y gcc-13 g++-13 \
        && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 10 --slave /usr/bin/g++ g++ /usr/bin/g++-13; \
    else \
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12; \
    fi \
    && gcc --version | head -1

ARG INSTALL_ROOT=/app/cray

WORKDIR ${INSTALL_ROOT}

RUN pip install setuptools-scm

ARG VLLM_SOURCE=remote
ARG VLLM_BRANCH=v0.32.0
ARG VLLM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_FORK_PATCH=vllm-v0.32.0-scalarlm.patch

COPY scripts/build-copy-vllm.sh scripts/vllm_patches/apply_patches.py \
     scripts/vllm_patches/vllm-v0.32.0-scalarlm.patch ${INSTALL_ROOT}/

RUN --mount=type=bind,source=./vllm,target=/workspace/vllm,rw \
    VLLM_FORK_PATCH="${VLLM_FORK_PATCH}" \
    bash ${INSTALL_ROOT}/build-copy-vllm.sh ${VLLM_SOURCE} ${INSTALL_ROOT}/vllm \
    /workspace/vllm ${VLLM_REPO} ${VLLM_BRANCH} ${VLLM_COMMIT}

WORKDIR ${INSTALL_ROOT}/vllm

ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG VLLM_TARGET_DEVICE=cpu

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}
ENV CMAKE_BUILD_TYPE=Release

COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/requirements-vllm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm.txt && \
    if [ -f ${INSTALL_ROOT}/vllm/tools/use_existing_torch.py ]; then \
        UET=${INSTALL_ROOT}/vllm/tools/use_existing_torch.py; \
    else \
        UET=${INSTALL_ROOT}/vllm/use_existing_torch.py; \
    fi && \
    python "$UET" --prefix && \
    if [ "$VLLM_TARGET_DEVICE" = "cuda" ] && [ -f requirements/build/cuda.txt ]; then \
        uv pip install --no-compile --no-cache-dir -r requirements/build/cuda.txt; \
    fi && \
    if [ "$VLLM_TARGET_DEVICE" = "cuda" ] && python -c "import sys, torch; sys.exit(0 if torch.version.cuda.startswith('12.') else 1)"; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/; s/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt && \
        echo "CUDA 12 build: swapped [cu13] extras in requirements/cuda.txt"; \
    fi

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/app/cray/vllm/.deps \
    NPROC=$(nproc) && \
    MEM_BASED=$(free -g | awk '/^Mem:/ {print int($2/4)}') && \
    COMPUTED=$(( NPROC < MEM_BASED ? NPROC : MEM_BASED )) && \
    export MAX_JOBS=$(( COMPUTED < 16 ? COMPUTED : 16 )) && \
    echo "MAX_JOBS=${MAX_JOBS} (nproc=${NPROC}, mem-based=${MEM_BASED}, capped at 16)" && \
    pip install --no-build-isolation -e . --verbose \
        ${TORCH_CUDA_INDEX:+--extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}} && \
    uv pip install "safetensors>=0.8.0" && \
    if ls /opt/opencv-wheels/opencv_python_headless-*.whl >/dev/null 2>&1; then \
        uv pip uninstall opencv-python-headless opencv-python && \
        uv pip install --no-deps /opt/opencv-wheels/opencv_python_headless-*.whl && \
        if find "$(python -c 'import site; print(site.getsitepackages()[0])')" \
                -name 'libcrypto-*.so*' -o -name 'libssl-*.so*' | grep -q opencv; then \
            echo "ERROR: opencv still bundles its own OpenSSL" >&2; exit 1; \
        fi && \
        echo "opencv-python-headless replaced with the source build (system OpenSSL)"; \
    fi

RUN if [ -x /app/check-torch-stack.sh ]; then /app/check-torch-stack.sh after-vllm; fi

WORKDIR ${INSTALL_ROOT}

FROM vllm AS infra

RUN apt-get update -y  \
    && apt-get install -y build-essential \
    less curl wget net-tools vim iputils-ping strace gdb python3-dbg python3-dev \
    dmidecode \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/vllm"

COPY ./infra/requirements-megatron.txt ${INSTALL_ROOT}/requirements-megatron.txt
COPY ./infra/requirements-megatron-cpu.txt ${INSTALL_ROOT}/requirements-megatron-cpu.txt
COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt

RUN \
    if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        uv pip install --no-deps --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron.txt; \
    fi && \
    if [ "$VLLM_TARGET_DEVICE" = "cpu" ]; then \
        uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron-cpu.txt; \
    fi && \
    uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt && \
    uv pip install --no-compile --no-cache-dir scikit-learn && \
    python -c "import transformers, tokenizers; print(f'transformers {transformers.__version__} + tokenizers {tokenizers.__version__} import OK')" && \
    if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        python -c "import sklearn; print('sklearn import OK', sklearn.__version__)"; \
    fi && \
    if [ "$VLLM_TARGET_DEVICE" = "cpu" ]; then \
        python -c "import peft, sentence_transformers, torch, torchaudio, torchao, torchcodec, torchvision, vllm._C; assert torch.version.cuda is None; assert '+cpu' in torch.__version__, torch.__version__; assert '+cpu' in torchvision.__version__, torchvision.__version__; assert '+cpu' in torchaudio.__version__, torchaudio.__version__; assert '+cpu' in torchcodec.__version__, torchcodec.__version__"; \
    fi

RUN if [ -x /app/check-torch-stack.sh ]; then /app/check-torch-stack.sh after-infra; fi

RUN mkdir -p ${INSTALL_ROOT}/jobs ${INSTALL_ROOT}/nfs

COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

COPY --from=scalarlm_ui_builder /build/dist /app/ui-bundle

WORKDIR ${INSTALL_ROOT}

RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:/usr/local/lib/slurm
ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf
ENV VLLM_CPU_MOE_PREPACK=0

RUN if [ "$VLLM_TARGET_DEVICE" = "cuda" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && [ ! -x "$VIRTUAL_ENV/bin/torchrun" ]; then \
        printf '#!%s/bin/python\nimport sys\nfrom torch.distributed.run import main\nif __name__=="__main__":\n    sys.exit(main())\n' "$VIRTUAL_ENV" > "$VIRTUAL_ENV/bin/torchrun" && \
        chmod +x "$VIRTUAL_ENV/bin/torchrun" && \
        "$VIRTUAL_ENV/bin/python" -c "import torch, cryptography; from torch.distributed.run import main" && \
        echo "venv torchrun installed at $VIRTUAL_ENV (resolves system torch + venv deps)"; \
    fi
