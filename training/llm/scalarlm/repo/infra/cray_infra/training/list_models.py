from cray_infra.api.fastapi.routers.request_types.list_models_response import (
    ListModelsResponse,
)

from cray_infra.training.get_training_job_info import get_training_job_status

from cray_infra.training.vllm_model_manager import get_vllm_model_manager
from cray_infra.training.get_latest_model import get_start_time

from cray_infra.util.get_config import get_config

import os

import logging

logger = logging.getLogger(__name__)


async def list_models():
    logger.info("Listing models")

    config = get_config()

    models = []

    # Synology FIRST so it wins on hash collisions in the dedup below
    # (`if name not in job_dirs`). Synology is the canonical post-COMPLETED
    # archive written by the post-completion copy; PVC entries are kept
    # for in-progress runs that haven't been archived yet, so PVC-only
    # hashes still surface, but a hash present in both shows the archived
    # copy.
    bases = []
    synology_dir = "/mnt/synology/jobs"
    if os.path.exists(synology_dir):
        bases.append(synology_dir)

    if os.path.exists(config["training_job_directory"]):
        bases.append(config["training_job_directory"])

    if not bases:
        return ListModelsResponse(models=models)

    registered_models = set(get_vllm_model_manager().get_registered_models())

    # Filter to entries that look like training job directories. The jobs
    # folder is often a mount point whose filesystem injects artifacts
    # (lost+found on ext*, .Trash-* on some NAS setups); `launch_training_job`
    # writes config.yaml at creation time, so use that as the ground truth.
    job_dirs = {}
    for base in bases:
        if not os.path.isdir(base):
            continue
        try:
            for name in os.listdir(base):
                job_path = os.path.join(base, name)
                if os.path.isdir(job_path) and os.path.isfile(os.path.join(job_path, "config.yaml")):
                    if name not in job_dirs:
                        job_dirs[name] = job_path
        except Exception as e:
            logger.error(f"Error scanning directory {base}: {e}")

    model_names = list(job_dirs.keys())
    model_names.sort(
        key=lambda x: get_start_time(job_dirs[x]),
        reverse=True,
    )

    for model_name in model_names:
        job_status, job_directory = get_training_job_status(model_name)

        status = {}

        if job_status is not None:
            status = job_status

        models.append(
            {
                "name": model_name,
                "deployed": model_name in registered_models,
                "status": status.get("status", "UNKNOWN"),
                "start_time": status.get("start_time", 0),
                "step": get_current_step(status),
                "max_steps": status.get("max_steps", 0),
                "train_time": get_train_time(status),
                "loss": get_loss(status),
            }
        )

    return ListModelsResponse(models=models)

def get_current_step(status):
    history = status.get("history", [])

    if len(history) == 0:
        return 0

    last_step = history[-1].get("step", 0)

    return last_step + 1

def get_train_time(status):
    history = status.get("history", [])

    if len(history) == 0:
        return 0

    last_time = history[-1].get("time", 0)

    return last_time

def get_loss(status):

    # Compute expontial moving average of loss with alpha=0.9
    history = status.get("history", [])

    if len(history) == 0:
        return 0.0

    alpha = 0.9

    ema_loss = history[0].get("loss", 0.0)
    for entry in history[1:]:
        loss = entry.get("loss", 0.0)
        ema_loss = alpha * ema_loss + (1 - alpha) * loss

    return ema_loss
