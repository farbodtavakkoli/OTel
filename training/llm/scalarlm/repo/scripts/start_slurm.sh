#!/bin/bash
set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python $LOCAL_DIRECTORY/../infra/cray_infra/slurm/discovery/discover_clusters.py

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
