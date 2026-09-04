#!/bin/bash

# Thin shell wrapper: set env vars then `exec` mpirun, so slurm's
# `--signal=B:TERM@N` lands on mpirun and is forwarded to each rank, whose
# handler in main.py sets the stop_flag.

set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# expandable_segments uses growable virtual address ranges so freed
# blocks of one size can satisfy a later allocation of a different
# size — without it, gradient checkpointing's recompute pattern
# fragments the caching allocator and reserved memory grows step
# over step (especially on Gemma-4-class models with alternating
# sliding/full-attention activation shapes). PyTorch 2.1+.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# `import vllm` sets this process-wide and sbatch inherits it. Left set, ROCm reports
# 0 devices inside the trainer and the whole run silently falls back to CPU while
# still reporting COMPLETED.
unset PYTORCH_NVML_BASED_CUDA_CHECK

LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${LOCAL_DIRECTORY}/ml:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- NCCL/RCCL launcher (tensorwavecloud/ScalarLM PR #5) ---------------------
# ONE torchrun per node, each spawning one worker per GPU. nproc-per-node comes
# from Slurm's ntasks-per-node, NOT PR #5's OMPI_COMM_WORLD_LOCAL_SIZE, which is
# 1 on a single node and would quietly train an 8-GPU job on one GPU.
GPUS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE%%(*}"   # "8(x2)" on heterogeneous allocations
NODEFILE="$(mktemp)"
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${NODEFILE}"
NUM_NODES="$(wc -l < "${NODEFILE}")"
MASTER_ADDR="$(head -n 1 "${NODEFILE}")"

# Slurm node names are arbitrary labels and may not resolve, but mpirun has to
# CONNECT to them. Unchecked, the failure looks identical to a collectives
# problem. Check up front and name the host that failed.
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

# Best-effort: raising locked-memory helps RDMA/NCCL pinning on multi-node, but
# an unprivileged container cannot change it. Must not be fatal under `set -e` —
# it exits the batch script before torchrun ever starts, and slurm then relaunches
# the job in a loop that never makes progress.
ulimit -l unlimited 2>/dev/null || echo "[launcher] note: cannot raise locked-memory limit (not permitted); continuing" >&2

if [ "${NUM_NODES}" -le 1 ]; then
    # Single node: torchrun IS the launcher. exec keeps slurm's --signal=B:TERM@N
    # landing directly on it, and torchrun forwards SIGTERM to every worker, so
    # the stop_flag/checkpoint path behaves as it did under mpirun.
    rm -f "${NODEFILE}"
    exec torchrun \
        --nnodes=1 \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${LOCAL_DIRECTORY}/ml/cray_megatron/main.py" "$@"
fi

# Multi-node: mpirun bootstraps one torchrun per node; every collective still
# goes through NCCL. This script runs on node 0 ONLY — remote ranks re-inherit
# their own node's environment, so anything set above must be forwarded
# explicitly or it applies to node 0 alone.

# Pin NCCL_IB_HCA to the RDMA devices ACTIVE on EVERY allocated node. Nodes with
# differing device lists make RCCL hang in the first collective with no error.
# Operator-set value always wins; best-effort, failure leaves it unset.
# See "Multi-node" in readme_scalarlm.md.
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

# Forward the socket-interface hints only when the operator set them; the right
# interface is cluster-specific, so there is no safe default to invent here. If
# cross-node rendezvous hangs or picks a wrong NIC, set NCCL_SOCKET_IFNAME (and
# GLOO_SOCKET_IFNAME) in the server environment and they will propagate.
MPIRUN_FORWARD=(-x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH
                -x CRAY_TRAINING_JOB_CONFIG_PATH -x PYTHONUNBUFFERED
                -x PYTORCH_CUDA_ALLOC_CONF)
for _var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA NCCL_DEBUG \
            CRAY_TRAIN_DEBUG CRAY_FAULT_HANDLER; do
    if [ -n "${!_var:-}" ]; then
        MPIRUN_FORWARD+=(-x "${_var}")
    fi
done

# Clean up the nodefile even when mpirun fails: `set -e` would otherwise abort
# the script at the failing mpirun and leak the temp file (the old
# `MPIRUN_EXIT=$?` line after it was unreachable on failure for the same reason).
trap 'rm -f "${NODEFILE}"' EXIT

mpirun --allow-run-as-root \
    --hostfile "${NODEFILE}" \
    -np "${NUM_NODES}" \
    -npernode 1 \
    --bind-to none \
    "${MPIRUN_FORWARD[@]}" \
    bash -c '
        set -Eeuo pipefail
        # Must be repeated here: see the note above. On nodes 2+ this is the only
        # place it runs, and leaving it set makes torch.cuda.is_available() false
        # inside the trainer -> silent CPU training that still reports COMPLETED.
        unset PYTORCH_NVML_BASED_CUDA_CHECK
        exec torchrun \
            --nnodes="'"${NUM_NODES}"'" \
            --nproc-per-node="'"${GPUS_PER_NODE}"'" \
            --node_rank="${OMPI_COMM_WORLD_RANK}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            "'"${LOCAL_DIRECTORY}"'/ml/cray_megatron/main.py" "$@"
    ' _ "$@"
