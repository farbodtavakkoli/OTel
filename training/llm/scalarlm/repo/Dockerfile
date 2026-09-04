ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:26.07-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages && \
    . $VIRTUAL_ENV/bin/activate

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ARG LD_LIBRARY_PATH=""
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/hpcx/ompi/lib:/usr/local/lib/python3.12/dist-packages/torch/lib

ARG TORCH_VERSION="2.13.0"
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv && \
    uv pip install ninja && \
    pip install --upgrade "protobuf>=6.30.0"

ENV PIP_CONSTRAINT=""

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

ENV TORCHINDUCTOR_MAX_AUTOTUNE=0
ENV TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
ENV TORCH_COMPILE_DISABLE=1

###############################################################################
# NVIDIA DGX SPARK BASE IMAGE
# aarch64 Grace CPU + Blackwell GPU (SM 12.0). The NGC PyTorch image is
# multi-arch, so pulling this tag on linux/arm64 yields the aarch64 variant.
FROM nvidia AS spark

ENV BASE_NAME=spark

###############################################################################
# CPU BASE IMAGE
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

# Put torch on the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib64/python3.12/site-packages/torch/lib

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
# MI355X (gfx950) port: rocm-base-mi300:v0.9926 ships torch 2.9, but the
# vllm-fork pin (v0.27.1 era, #223) needs torch >= 2.12 stable-ABI headers
# (torch/csrc/stable/accelerator.h). rocm/primus provides torch 2.12+rocm7.15
# built for gfx942;gfx950. Same base swap the tensorwavecloud AMD PR (#5) made.
FROM rocm/primus:v26.5 AS amd

ENV BASE_NAME=amd

#RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

# primus keeps the venv at /opt/venv, OpenMPI at /opt/openmpi, and ROCm as a
# pip SDK (_rocm_sdk_devel) with no /opt/rocm. Alias both to the conventional
# paths: plenty of ROCm tooling probes /opt/rocm, and /opt/ompi-rocm is the
# layout the ROCm OpenMPI builds expect. (These also served the gpu_aware_mpi
# extension's hardcoded paths before PR #5 removed it.)
# Slurm + PMIx came from the old base; install them here like the cpu stage.
ENV PATH="/opt/venv/bin:/opt/openmpi/bin:${PATH}"
RUN ln -sfn /opt/openmpi /opt/ompi-rocm && \
    ln -sfn /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel /opt/rocm && \
    pip install uv && uv pip install ninja "setuptools>=77"

# vLLM detects ROCm by importing `amdsmi`. The PyPI amdsmi (7.0.x) ABI-mismatches
# primus's libamd_smi.so.26: it sees 8 GPUs alone but 0 once torch has loaded the
# SDK's library first. Use the SDK's own matching python bindings instead.
RUN ln -sfn /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi/amdsmi \
        /opt/venv/lib/python3.12/site-packages/amdsmi
RUN apt-get update -y && apt-get install -y \
        slurm-wlm libslurm-dev libpmix-dev \
    && rm -rf /var/lib/apt/lists/*

# _rocm_sdk_core/lib MUST come first. primus ships libamd_comgr.so.3 and
# librocprofiler-sdk.so.1 in BOTH _rocm_sdk_core and _rocm_sdk_devel, and the base
# image's LD_LIBRARY_PATH points at devel. torch loads core's copies, so a training
# process ends up with two copies of each -> rocprofiler aborts ("REGISTER_LIBRARY
# already set ... not overriding") or LLVM aborts ("Option 'spirv-expand-step'
# registered more than once"). Prepending core makes every soname lookup resolve to
# ONE copy; devel stays on the path because it supplies librocm-openblas and the
# rest of the rocm_sysdeps shared objects.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:${LD_LIBRARY_PATH:-}:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/openmpi/lib

###############################################################################
# ScalarLM React UI build stage
#
# Builds the first-party React SPA from ./ui into /build/dist and hands it off
# to the final stage as /app/ui-bundle. No Node.js runtime is carried forward;
# the final image only contains the static bundle.
###############################################################################
FROM node:24.2.0 AS scalarlm_ui_builder

WORKDIR /build

# Copy package metadata first so Docker can cache `npm ci` across source edits.
COPY ui/package.json ui/package-lock.json* ./

# Lockfile may not exist on a fresh checkout; fall back to npm install.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY ui/ .
RUN npm run build

###############################################################################
# VLLM BUILD STAGE
FROM ${BASE_NAME} AS vllm

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl git ccache vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

WORKDIR ${INSTALL_ROOT}

# Install build dependencies FIRST
RUN pip install setuptools-scm

# Configure vLLM source - can use either local directory or remote repo.
#
# Pin the exact vllm-fork v0.27.1 revision validated with this PyTorch
# generation. VLLM_BRANCH still selects the clone source, while an explicit
# empty VLLM_COMMIT opts back into tracking that branch tip for development.
ARG VLLM_SOURCE=remote
ARG VLLM_BRANCH=main
ARG VLLM_COMMIT=281955fc63c823e8a439db82bf02ad9fffb199e2
ARG VLLM_REPO=https://github.com/supermassive-intelligence/vllm-fork.git

# Handle vLLM source - support both local and remote modes.
# build-copy-vllm.sh and apply_patches.py are copied in a single COPY
# step so the image stays under Docker's 127-layer overlay-fs stacking
# cap. The script sources apply_patches.py from its own SCRIPT_DIR.
COPY scripts/build-copy-vllm.sh scripts/vllm_patches/apply_patches.py ${INSTALL_ROOT}/

# Handle vLLM source - single RUN command with conditional mount
# For remote: clone from repository
# For local: mount and copy from ./vllm directory
RUN --mount=type=bind,source=./vllm,target=/workspace/vllm,rw \
    bash ${INSTALL_ROOT}/build-copy-vllm.sh ${VLLM_SOURCE} ${INSTALL_ROOT}/vllm \
    /workspace/vllm ${VLLM_REPO} ${VLLM_BRANCH} ${VLLM_COMMIT}

WORKDIR ${INSTALL_ROOT}/vllm

# Set build environment variables for CPU compilation
ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG VLLM_TARGET_DEVICE=cpu

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}
ENV CMAKE_BUILD_TYPE=Release

# vLLM dependencies
COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/requirements-vllm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm.txt && \
    python ${INSTALL_ROOT}/vllm/use_existing_torch.py --prefix

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/app/cray/vllm/.deps \
    NPROC=$(nproc) && \
    MEM_BASED=$(free -g | awk '/^Mem:/ {print int($2/4)}') && \
    COMPUTED=$(( NPROC < MEM_BASED ? NPROC : MEM_BASED )) && \
    export MAX_JOBS=$(( COMPUTED < 16 ? COMPUTED : 16 )) && \
    echo "MAX_JOBS=${MAX_JOBS} (nproc=${NPROC}, mem-based=${MEM_BASED}, capped at 16)" && \
    pip install --no-build-isolation -e . --verbose && \
    uv pip install "safetensors>=0.8.0" 

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

# The gpu_aware_mpi C++ extension is gone (PR #5): collectives now run through
# torch.distributed + NCCL/RCCL (infra/cray_infra/training/distributed.py), so
# there is no per-ROCm-release extension to compile here. OpenMPI itself stays in
# the image — mpirun is still the multi-node bootstrap for torchrun.

RUN apt-get update -y  \
    && apt-get install -y build-essential \
    less curl wget net-tools vim iputils-ping strace gdb python3-dbg python3-dev \
    dmidecode \
    && rm -rf /var/lib/apt/lists/*

# Setup python path
ENV PYTHONPATH="${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/vllm"

# Megatron dependencies (GPU only)
# note this has to happen after vllm because it overrides some packages installed by vllm
COPY ./infra/requirements-megatron.txt ${INSTALL_ROOT}/requirements-megatron.txt
COPY ./infra/requirements-megatron-cpu.txt ${INSTALL_ROOT}/requirements-megatron-cpu.txt
COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt

RUN \
    if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        # --no-deps so these wheels don't drag in transitive deps that
        # clobber the versions vllm installed above (this layer runs after
        # vllm on purpose). All megatron deps are pure-Python wheels — no
        # source build, so no MAX_JOBS cap or --no-build-isolation needed
        # (those were only for flash-attn, which is no longer installed).
        uv pip install --no-deps --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron.txt; \
    fi && \
    # CPU-only dependencies must not replace the ROCm base's binary packages.
    if [ "$VLLM_TARGET_DEVICE" = "cpu" ]; then \
        uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron-cpu.txt; \
    fi && \
    uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt && \
    # sentence_transformers (embedding training mode) imports scikit-learn eagerly
    # (sentence_transformers.util.similarity -> sklearn.metrics). The megatron
    # requirements install --no-deps, so sklearn is never pulled in and the
    # embedding path dies at import on the GPU image. Install it explicitly.
    uv pip install --no-compile --no-cache-dir scikit-learn && \
    # Fail the BUILD if transformers cannot be imported.
    #
    # transformers validates its pins (tokenizers, safetensors) at import time,
    # and everything above installs either with --no-deps or in an order that
    # can leave a transitive package on an incompatible version. When that
    # happens the image still builds, still pushes, and still passes any gate
    # that only exercises the client SDK — it just cannot start an api server,
    # because vllm imports transformers on the way up. mi355-v1.1 and v1.2 both
    # shipped that way (tokenizers 0.23.1 vs the required <=0.23.0).
    #
    # One line here turns a silent broken publish into a failed build.
    python -c "import transformers, tokenizers; print(f'transformers {transformers.__version__} + tokenizers {tokenizers.__version__} import OK')" && \
    # The megatron deps install --no-deps, so sentence_transformers' transitive
    # scikit-learn is otherwise missing and the embedding path only fails when a
    # job runs. Assert sklearn imports here. (peft / sentence_transformers pull in
    # transformer_engine, which dlopens libcuda.so.1 — absent at build time — so
    # they cannot be import-checked in the builder; sklearn is CUDA-free.)
    if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        python -c "import sklearn; print('sklearn import OK', sklearn.__version__)"; \
    fi && \
    if [ "$VLLM_TARGET_DEVICE" = "cpu" ]; then \
        python -c "import peft, sentence_transformers, torch, torchaudio, torchao, torchcodec, torchvision, vllm._C; assert torch.version.cuda is None; assert '+cpu' in torch.__version__, torch.__version__; assert '+cpu' in torchvision.__version__, torchvision.__version__; assert '+cpu' in torchaudio.__version__, torchaudio.__version__; assert '+cpu' in torchcodec.__version__, torchcodec.__version__"; \
    fi

RUN mkdir -p ${INSTALL_ROOT}/jobs ${INSTALL_ROOT}/nfs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

# ScalarLM React UI static bundle — served by FastAPI at /app/*.
# Path must match SCALARLM_UI_BUNDLE_DIR in infra/cray_infra/api/fastapi/setup_ui.py.
COPY --from=scalarlm_ui_builder /build/dist /app/ui-bundle

WORKDIR ${INSTALL_ROOT}

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:/usr/local/lib/slurm
ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf
ENV VLLM_CPU_MOE_PREPACK=0

# train_job_entrypoint.sh launches workers with a bare `torchrun`. On the NVIDIA
# base, torch lives in the system dist-packages so `torchrun` resolves to
# /usr/local/bin/torchrun with a `#!/usr/bin/python3` shebang — the SYSTEM
# interpreter, which cannot see venv-only deps (cryptography, peft, …) and every
# worker dies at `import cryptography` (via cray_infra.huggingface.get_hf_token).
# The NVIDIA venv is built --system-site-packages, so its interpreter sees BOTH
# system torch and the venv deps. Install a venv torchrun that runs under it so
# the unmodified entrypoint resolves torchrun (venv PATH is first) correctly.
#
# NVIDIA-ONLY (VLLM_TARGET_DEVICE=cuda). The AMD image installs torch INTO its
# venv (/opt/venv) so torchrun is already venv-bound there — and its venv is NOT
# at $VIRTUAL_ENV=/app/.venv, so this must never touch the AMD build. Guard on
# cuda specifically and on the venv python actually existing, and only add a
# torchrun if one isn't already present, so this is a no-op everywhere but NVIDIA.
RUN if [ "$VLLM_TARGET_DEVICE" = "cuda" ] && [ -x "$VIRTUAL_ENV/bin/python" ] && [ ! -x "$VIRTUAL_ENV/bin/torchrun" ]; then \
        printf '#!%s/bin/python\nimport sys\nfrom torch.distributed.run import main\nif __name__=="__main__":\n    sys.exit(main())\n' "$VIRTUAL_ENV" > "$VIRTUAL_ENV/bin/torchrun" && \
        chmod +x "$VIRTUAL_ENV/bin/torchrun" && \
        "$VIRTUAL_ENV/bin/python" -c "import torch, cryptography; from torch.distributed.run import main" && \
        echo "venv torchrun installed at $VIRTUAL_ENV (resolves system torch + venv deps)"; \
    fi
