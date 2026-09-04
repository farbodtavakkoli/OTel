from cray_infra.util.get_config import get_config

import os
import json
import logging

logger = logging.getLogger(__name__)


def get_latest_model():
    config = get_config()
    bases = []
    if os.path.exists(config["training_job_directory"]):
        bases.append(config["training_job_directory"])
    
    synology_dir = "/mnt/synology/jobs"
    if os.path.exists(synology_dir):
        bases.append(synology_dir)

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

    if not job_dirs:
        raise FileNotFoundError("No training jobs found")

    models = list(job_dirs.keys())
    models.sort(
        key=lambda x: get_start_time(job_dirs[x]),
        reverse=True,
    )

    model_name = models[0]

    return model_name


def get_start_time(path):
    try:
        with open(os.path.join(path, "status.json")) as f:
            status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"Could not read status.json in {path}")
        return 0

    if "history" not in status:
        logger.warning(f"No history found in status.json in {path}")
        return 0

    return status.get("start_time", 0)
