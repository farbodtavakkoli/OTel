from cray_infra.util.default_job_config import JobConfig

import yaml
import os
import logging

logger = logging.getLogger(__name__)

_EXPECTED_UNDECLARED_KEYS = {
    "job_directory",
    "training_data_path",
    "dataset_hash",
}

_warned_dropped_keys = False


def get_job_config():
    job_config_path = get_job_config_path()

    with open(job_config_path, "r") as stream:
        raw_job_config = yaml.safe_load(stream) or {}

    job_config = JobConfig(**raw_job_config).dict()

    _warn_about_dropped_keys(raw_job_config, job_config)

    return job_config


def _warn_about_dropped_keys(raw_job_config, job_config):
    global _warned_dropped_keys
    if _warned_dropped_keys:
        return

    dropped = set(raw_job_config) - set(job_config) - _EXPECTED_UNDECLARED_KEYS
    if dropped:
        logger.warning(
            "job config: %d key(s) in config.yaml are NOT declared on JobConfig "
            "and were DROPPED — they will have no effect: %s. Declare them in "
            "cray_infra/util/default_job_config.py to make them take effect.",
            len(dropped),
            ", ".join(sorted(dropped)),
        )
    _warned_dropped_keys = True


def get_job_config_path():
    assert (
        "CRAY_TRAINING_JOB_CONFIG_PATH" in os.environ
    ), "CRAY_TRAINING_JOB_CONFIG_PATH not set"
    return os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"]
