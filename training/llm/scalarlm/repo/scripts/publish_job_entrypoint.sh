#!/bin/bash
set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

export PYTHONPATH="$LOCAL_DIRECTORY/ml:$LOCAL_DIRECTORY/infra:${PYTHONPATH:-}"

python -m adapters.merge_lora_and_push "$@"
