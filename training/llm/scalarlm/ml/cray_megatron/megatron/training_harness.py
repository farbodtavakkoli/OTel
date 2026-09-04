from cray_infra.util.get_config import get_config
from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.main_rank_only import main_rank_only

import torch

import os
import json

import logging

logger = logging.getLogger(__name__)

from cray_infra.training.train_debug import is_train_debug_enabled

import sys
import time


def _trace_harness(msg: str) -> None:
    """CRAY_TRAIN_DEBUG tracing for status writes."""
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[rank={rank}] harness [{time.monotonic():.3f}]: {msg}\n")
    sys.stderr.flush()


class TrainingHarness:
    def update_status(self, status, metadata={}):
        _trace_harness(f"update_status enter status={status}")

        current_status = get_status()

        current_status["status"] = status
        # Heartbeat: a job untouched within training_heartbeat_seconds is treated as dead.
        current_status["last_updated"] = time.time()
        for key, value in metadata.items():
            current_status[key] = value

        save_status(current_status)

    def checkpoint(self, checkpoint_state, checkpoint_name):
        job_config = get_job_config()

        checkpoint_path = os.path.join(job_config["job_directory"], checkpoint_name)

        torch.save(checkpoint_state, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def get_status(self):
        return get_status()


def get_status():
    try:
        with open(os.path.join(get_training_job_directory(), "status.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading job status: {e}")
        return {"status": "unknown"}


def get_training_job_directory():
    job_config = get_job_config()

    return job_config["job_directory"]

@main_rank_only
def save_status(job_status):
    try:
        contents = json.dumps(job_status)
    except Exception as e:
        logger.error(f"Error serializing job status: {e}")
        return

    with open(os.path.join(get_training_job_directory(), "status.json"), "w") as f:
        f.write(contents)
