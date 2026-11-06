#!/bin/bash
set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$LOCAL_DIRECTORY/start_slurm.sh

python -m cray_infra.one_server.main

