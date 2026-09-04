from cray_infra.util.default_job_config import JobConfig

import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Keys the server adds to config.yaml for its own bookkeeping that the training
# code is not expected to consume. Listing them keeps the warning below signal.
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

    # fill in missing values with defaults
    job_config = JobConfig(**raw_job_config).dict()

    _warn_about_dropped_keys(raw_job_config, job_config)

    return job_config


def _warn_about_dropped_keys(raw_job_config, job_config):
    """Log any train_args key that this schema silently discarded.

    JobConfig runs with pydantic's default extra="ignore", so a key that is not
    declared as a field vanishes between config.yaml and the training code with
    no error. The setting simply has no effect, which is indistinguishable from
    a broken feature — and it has bitten this codebase repeatedly:

      * hsdp_shard_size / reshard_after_forward — the whole HSDP path was
        unreachable; build_hsdp_mesh() could never leave its mesh=None branch.
      * upload_to_hf / hf_repo_id / hf_upload_token — the HF upload feature
        returned on its first line, so the flag was a silent no-op.
      * lora_config.use_rslora — training used alpha/r while the merge step,
        which reads the raw YAML, used alpha/sqrt(r): a mis-scaled adapter.

    Each cost real debugging time and none of them raised. One warning at load
    turns this entire bug class into a log line. Fires once per process.
    """
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
