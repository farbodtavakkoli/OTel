#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run the slurm discovery service
python $LOCAL_DIRECTORY/../infra/cray_infra/slurm/discovery/discover_clusters.py

# PR #5: only the controller node runs slurmctld. Every node starting its own
# controller is harmless on a single-node deployment but wrong as soon as there
# is more than one node — the extra controllers fight over the same state
# directory. slurmd still runs everywhere (it is the per-node worker daemon).
SLURM_CONF_PATH="${SLURM_CONF:-/app/cray/nfs/slurm.conf}"
CONTROLLER=""
if [ -f "$SLURM_CONF_PATH" ]; then
    CONTROLLER=$(grep '^SlurmctldHost=' "$SLURM_CONF_PATH" | cut -d= -f2- | tr -d '[:space:]' || true)
fi

THIS_HOST=$(hostname)
if [ -z "$CONTROLLER" ] || [ "$THIS_HOST" = "$CONTROLLER" ] || [ "${THIS_HOST%%.*}" = "$CONTROLLER" ]; then
    echo "Starting slurmctld on ${THIS_HOST} (controller=${CONTROLLER:-unset})"
    slurmctld
else
    echo "Skipping slurmctld on ${THIS_HOST} (controller is ${CONTROLLER})"
fi

slurmd
