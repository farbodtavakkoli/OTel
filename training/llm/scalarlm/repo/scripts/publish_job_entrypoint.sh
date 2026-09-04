#!/bin/bash
# Entrypoint for the publish-to-HF SLURM job.
#
# All arguments are forwarded verbatim to `adapters.merge_lora_and_push`.
# The HuggingFace token is expected to be in $HF_TOKEN, set via
# `sbatch --export=...,HF_TOKEN=<value>` by launch_publish_job.py.
# The token never lands on this script's argv or in any file we write.

set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

# Make `adapters` and `cray_infra` importable.
export PYTHONPATH="$LOCAL_DIRECTORY/ml:$LOCAL_DIRECTORY/infra:${PYTHONPATH:-}"

python -m adapters.merge_lora_and_push "$@"
