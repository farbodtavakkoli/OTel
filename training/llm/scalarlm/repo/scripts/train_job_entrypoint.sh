#!/bin/bash
set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset PYTORCH_NVML_BASED_CUDA_CHECK

LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${LOCAL_DIRECTORY}/ml:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"
NODEFILE="$(mktemp)"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${NODEFILE}"
NUM_NODES="$(wc -l < "${NODEFILE}")"
MASTER_ADDR="$(head -n 1 "${NODEFILE}")"

if [ "${NUM_NODES}" -gt 1 ]; then
    UNRESOLVABLE=""
    while read -r node_name; do
        [ -z "${node_name}" ] && continue
        if ! getent hosts "${node_name}" >/dev/null 2>&1; then
            UNRESOLVABLE="${UNRESOLVABLE} ${node_name}"
        fi
    done < "${NODEFILE}"

    if [ -n "${UNRESOLVABLE}" ]; then
        echo "[launcher] FATAL: slurm node name(s) do not resolve:${UNRESOLVABLE}" >&2
        echo "[launcher]        SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
        echo "[launcher]        'scontrol show hostnames' must yield names this container can" >&2
        echo "[launcher]        connect to. Fix the NodeName entries in slurm.conf, add the" >&2
        echo "[launcher]        names to DNS/\$(hostname) resolution, or set NodeAddr so the" >&2
        echo "[launcher]        names map to reachable addresses." >&2
        rm -f "${NODEFILE}"
        exit 1
    fi
    echo "[launcher] all ${NUM_NODES} slurm hostnames resolve" >&2
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[launcher] nodes=${NUM_NODES} gpus_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}" >&2

ulimit -l unlimited 2>/dev/null || echo "[launcher] note: cannot raise locked-memory limit (not permitted); continuing" >&2

if [ "${NUM_NODES}" -le 1 ]; then

    rm -f "${NODEFILE}"
    exec torchrun \
        --nnodes=1 \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${LOCAL_DIRECTORY}/ml/cray_megatron/main.py" "$@"
fi

if [ "${NUM_NODES}" -gt 1 ] && [ -z "${NCCL_IB_HCA:-}" ] && command -v ibv_devinfo >/dev/null 2>&1; then
    _hca_common="$(
        mpirun --allow-run-as-root --hostfile "${NODEFILE}" \
               -np "${NUM_NODES}" -npernode 1 --bind-to none \
            bash -c 'ibv_devinfo 2>/dev/null | awk "/hca_id/{h=\$2} /state:.*PORT_ACTIVE/{print h}" | sort -u' \
            2>/dev/null \
        | sort | uniq -c \
        | awk -v n="${NUM_NODES}" '$1 == n {printf "%s%s", (c++ ? "," : ""), $2}'
    )" || _hca_common=""

    if [ -n "${_hca_common}" ]; then
        export NCCL_IB_HCA="${_hca_common}"
        echo "[launcher] NCCL_IB_HCA=${NCCL_IB_HCA} (intersection across ${NUM_NODES} nodes)" >&2
    else
        echo "[launcher] warning: could not derive a common RDMA device set; leaving NCCL_IB_HCA unset." >&2
        echo "[launcher]          if the first collective hangs, pin NCCL_IB_HCA to devices ACTIVE on every node." >&2
    fi
fi

MPIRUN_FORWARD=(-x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH
                -x CRAY_TRAINING_JOB_CONFIG_PATH -x PYTHONUNBUFFERED
                -x PYTORCH_CUDA_ALLOC_CONF)
for _var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA NCCL_DEBUG \
            CRAY_TRAIN_DEBUG CRAY_FAULT_HANDLER; do
    if [ -n "${!_var:-}" ]; then
        MPIRUN_FORWARD+=(-x "${_var}")
    fi
done

trap 'rm -f "${NODEFILE}"' EXIT

mpirun --allow-run-as-root \
    --hostfile "${NODEFILE}" \
    -np "${NUM_NODES}" \
    -npernode 1 \
    --bind-to none \
    "${MPIRUN_FORWARD[@]}" \
    bash -c '
        set -Eeuo pipefail
        unset PYTORCH_NVML_BASED_CUDA_CHECK
        exec torchrun \
            --nnodes="'"${NUM_NODES}"'" \
            --nproc-per-node="'"${GPUS_PER_NODE}"'" \
            --node_rank="${OMPI_COMM_WORLD_RANK}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            "'"${LOCAL_DIRECTORY}"'/ml/cray_megatron/main.py" "$@"
    ' _ "$@"
