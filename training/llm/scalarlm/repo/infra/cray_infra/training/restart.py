"""
Manual restart of a non-running training job.

Companion to the auto-relaunch reconciler in restart_megatron_jobs.py
(which re-queues TRAINING/QUEUED jobs that fell out of squeue on a
SLURM slice timeout — see docs/training-lifecycle.md §5.4). This module
covers the operator-initiated case from the UI: a FAILED or CANCELLED
job whose user wants to resume from the last checkpoint.

Resume itself is automatic — TrainingLoop.resume_from_checkpoint
(ml/cray_megatron/megatron/training_loop.py) reads the latest
checkpoint and replays optimizer/scheduler/RNG/data-cursor state on
every slice start. All we have to do here is clear the prior error
fields from status.json and call start_slurm_job again.
"""

import json
import logging
import os

import yaml
from fastapi import HTTPException, status

from cray_infra.api.fastapi.routers.request_types.train_request import TrainResponse
from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.training.get_training_job_info import get_training_job_status
from cray_infra.training.launch_training_job import start_slurm_job

logger = logging.getLogger(__name__)


# Statuses we'll accept restart from. FAILED is the headline case
# (sbatch failure, OOM, crash). CANCELLED is included so a user who
# cancelled and now wants to pick up where they left off can use the
# same button — same disk state, same restart path. QUEUED/TRAINING
# would double-launch a live job; COMPLETED has nothing to resume.
_RESTARTABLE_STATUSES = frozenset({"FAILED", "CANCELLED"})


async def restart(job_hash: str):
    logger.info(f"Restart request received for job hash: {job_hash}")

    if job_hash == "latest":
        job_hash = get_latest_model()

    job_status, job_directory_path = get_training_job_status(job_hash)

    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job was not found at {job_directory_path}",
        )

    current = job_status.get("status")
    if current not in _RESTARTABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot restart job in state {current!r}; restart is only "
                f"allowed from {sorted(_RESTARTABLE_STATUSES)}. Cancel the "
                "job first if it is currently running."
            ),
        )

    config_filepath = os.path.join(job_directory_path, "config.yaml")
    try:
        with open(config_filepath, "r") as file:
            job_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job config was not found at {job_directory_path}",
        )

    reset_status_for_restart(job_directory_path)

    # start_slurm_job rewrites status/job_id/start_time on top of our
    # cleared row via run_sbatch's own write_job_status; the error/
    # output fields we just dropped stay dropped because that writer
    # merges new fields into existing rather than truncating.
    start_slurm_job(job_config)

    refreshed, _ = get_training_job_status(job_hash)
    return TrainResponse(
        job_status=refreshed,
        job_config=job_config,
        deployed=False,
    )


def reset_status_for_restart(job_directory_path: str) -> None:
    """
    Drop the FAILED/CANCELLED-era error fields and set status to QUEUED.

    The TrainDetail UI surfaces `job_status.error` directly; leaving it
    in place after a successful restart would show a stale red error
    chip next to a QUEUED badge. `output` is the captured sbatch
    stderr from the previous failure (launch_training_job.py:320) —
    same reason to drop.
    """
    status_filepath = os.path.join(job_directory_path, "status.json")
    try:
        with open(status_filepath, "r") as file:
            job_status = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        job_status = {}

    job_status.pop("error", None)
    job_status.pop("output", None)
    job_status["status"] = "QUEUED"

    with open(status_filepath, "w") as file:
        json.dump(job_status, file)
