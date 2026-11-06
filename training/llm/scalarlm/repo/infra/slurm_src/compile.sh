#!/bin/bash
set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SOURCE_FILE=$LOCAL_DIRECTORY/cgroup_docker.c
INCLUDE_PATH=/usr/include

gcc -I$INCLUDE_PATH -Wall -fPIC -shared -o $LOCAL_DIRECTORY/cgroup_docker.so $SOURCE_FILE

if [ "$(uname -m)" == "x86_64" ]; then
    TARGET="x86_64-linux-gnu"
elif [ "$(uname -m)" == "aarch64" ]; then
    TARGET="aarch64-linux-gnu"
else
    echo "Unsupported architecture"
    exit 1
fi

if [ "$BASE_NAME" == "amd" ] || [ "$BASE_NAME" == "nvidia" ]; then
    sed -i -e 's/CgroupPlugin=cgroup\/docker/CgroupPlugin=cgroup\/v1/g' /app/cray/infra/slurm_configs/cgroup.conf
else
    cp /app/cray/infra/slurm_src/cgroup_docker.so /usr/lib/$TARGET/slurm-wlm/cgroup_docker.so
fi

mkdir -p /var/log/slurm
mkdir -p /var/spool/slurmd
chown slurm:slurm /var/spool/slurmd 2>/dev/null || true

