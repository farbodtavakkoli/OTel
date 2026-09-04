from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.launch_training_job import start_slurm_job

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.util.get_config import get_config


import traceback
import os
import json
import time
import yaml
import subprocess

import logging

logger = logging.getLogger(__name__)


async def restart_megatron_jobs():
    logger.info("Restarting Megatron jobs")

    # Get all the jobs that are running
    all_jobs = get_running_jobs()

    # Get slurm jobs that are running
    slurm_job_names = await get_slurm_jobs()

    logger.info(f"Slurm jobs running: {slurm_job_names}")

    # Filter out the jobs that are already running
    jobs = filter_running_jobs(all_jobs, slurm_job_names)

    # Restart all the jobs
    async for job in jobs:
        await restart_job(job)

    # Get slurm jobs that are running again
    slurm_job_names = await get_slurm_jobs()

    # If any are still running, keep the server alive
    if slurm_job_names:
        logger.info("Jobs are still running, keeping the server alive")
        await keep_alive()


async def get_running_jobs():
    config = get_config()

    # training jobs are in the training job directory
    # they are subdirectories which should have a file called status.json

    # get all the directories in the training job directory
    # that have a status.json file
    if not os.path.exists(config["training_job_directory"]):
        return

    for path in os.listdir(config["training_job_directory"]):
        root = os.path.join(config["training_job_directory"], path)
        if os.path.exists(os.path.join(root, "status.json")):
            try:
                with open(os.path.join(root, "status.json")) as f:
                    status = json.load(f)
            except Exception as e:
                logger.error(f"Error reading status.json for job {root}: {e}")
                # print exception info
                traceback.print_exc()
                continue

            if not isinstance(status, dict):
                logger.error(
                    "status.json for job %s is not a JSON object (got %s)",
                    root,
                    type(status).__name__,
                )
                continue

            if "status" not in status:
                logger.warning(
                    f"status.json for job {root} has no 'status' key, skipping"
                )
                continue

            job_status = status["status"]
            if (
                job_status == TrainingJobStatus.TRAINING
                or job_status == TrainingJobStatus.QUEUED
            ):
                yield root


async def get_slurm_jobs():
    squeue_output = subprocess.check_output(
        ["squeue", '--format="%.18i %.9P %.128j %.8u %.8T %.10M %.9l %.6D %R"']
    )

    # Here is an example of the output of squeue
    # JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    # 1234      gpu  jobname  username  R       0:10      1 node1
    # 5678      gpu  jobname  username  R       0:10      1 node2
    # 91011      gpu  jobname  username  R       0:10      1 node3
    # 121314      gpu  jobname  username  R       0:10      1 node4

    # We want to get the job name from the third column
    # We also want to remove the header
    jobs = squeue_output.decode().split("\n")[1:]

    job_names = []

    for job in jobs:
        if job:
            job_fields = job.strip().split()
            if len(job_fields) > 3:
                job_names.append(job_fields[3])

    return job_names


async def filter_running_jobs(all_jobs, slurm_job_names):
    async for job in all_jobs:
        if os.path.basename(job) not in slurm_job_names:
            yield job


def _read_status(job):
    status_path = os.path.join(job, "status.json")
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path) as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_status(job, status):
    status_path = os.path.join(job, "status.json")
    try:
        with open(status_path, "w") as f:
            json.dump(status, f)
    except OSError as e:
        logger.error(f"Could not persist restart bookkeeping for {job}: {e}")


def should_restart_job(job, now=None):
    """Bounded restart policy (PR #5).

    Relaunching a job whose slurm allocation vanished is right for preemption,
    but when a job dies during startup for a deterministic reason (a bad
    entrypoint, an unavailable device) the same job is resubmitted forever —
    observed here as a job reaching attempt 24 in a few minutes, each attempt
    failing identically. Three guards:

      * a recent heartbeat means the job IS alive; slurm just has not registered
        it yet (or squeue was queried mid-transition) — do not double-submit;
      * a cooldown spaces out attempts so a failing job cannot spin;
      * a hard cap gives up and marks the job FAILED instead of looping.
    """
    cfg = get_config()
    now = now if now is not None else time.time()
    status = _read_status(job)

    last_updated = status.get("last_updated")
    heartbeat = cfg.get("training_heartbeat_seconds", 600)
    if isinstance(last_updated, (int, float)) and (now - last_updated) < heartbeat:
        logger.info(
            f"Not restarting {job}: heartbeat {now - last_updated:.0f}s ago "
            f"(< {heartbeat}s) — job is alive"
        )
        return False, status

    last_restart = status.get("last_restart_time")
    cooldown = cfg.get("job_restart_cooldown_seconds", 300)
    if isinstance(last_restart, (int, float)) and (now - last_restart) < cooldown:
        logger.info(
            f"Not restarting {job}: last restart {now - last_restart:.0f}s ago "
            f"(< {cooldown}s cooldown)"
        )
        return False, status

    restart_count = int(status.get("restart_count", 0) or 0)
    max_restarts = cfg.get("max_job_restart_count", 3)
    if restart_count >= max_restarts:
        logger.error(
            f"Not restarting {job}: exhausted {restart_count}/{max_restarts} restarts"
        )
        if status.get("status") != "FAILED":
            status["status"] = "FAILED"
            status["error"] = (
                f"Exceeded max_job_restart_count ({max_restarts}); "
                "the job failed to start repeatedly."
            )
            _write_status(job, status)
        return False, status

    return True, status


async def restart_job(job):
    allowed, status = should_restart_job(job)
    if not allowed:
        return

    logger.info(f"Restarting job: {job}")

    # Get the job config
    with open(os.path.join(job, "config.yaml")) as f:
        config = yaml.safe_load(f)

    status["restart_count"] = int(status.get("restart_count", 0) or 0) + 1
    status["last_restart_time"] = time.time()
    _write_status(job, status)

    start_slurm_job(config)


async def keep_alive():
    config = get_config()
    session = get_global_session()
    try:
        async with session.get(config["api_url"] + "/v1/health/keepalive") as resp:
            assert resp.status == 200
    except Exception as e:
        logger.error(f"Error keeping the server alive: {e}")
