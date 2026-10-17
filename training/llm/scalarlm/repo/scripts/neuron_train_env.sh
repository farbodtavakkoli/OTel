unset NKI_COMPILER_FRONTEND NEURON_CC_FLAGS PJRT_DEVICE NEURON_VISIBLE_DEVICES
export PATH="/opt/venv/bin:${PATH}"
export PYTHONPATH="/app/cray/infra:/app/cray/sdk"
export LD_LIBRARY_PATH="/opt/aws/neuron/lib"
NEURON_CORES=""
for DEVICE in $(ls /dev | grep -E '^neuron[0-9]+$' | sed 's/^neuron//' | sort -n); do
    for CORE in 0 1 2 3; do
        NEURON_CORES="${NEURON_CORES} $((DEVICE * 4 + CORE))"
    done
done
NEURON_TASKS="${SLURM_NTASKS_PER_NODE:-1}"
export NEURON_RT_VISIBLE_CORES="$(echo ${NEURON_CORES} | tr ' ' '\n' | tail -n "${NEURON_TASKS%%(*}" | paste -sd, -)"
