"""Push a finished checkpoint to the HuggingFace Hub."""

import logging
import os

logger = logging.getLogger(__name__)

# Offline switches a deployment may set so boot cannot stall on tokenizer downloads.
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def upload_to_hf_if_enabled():
    """Push the final checkpoint to the Hub when the job asked for it."""
    from cray_infra.training.distributed import get_rank
    from cray_infra.util.get_job_config import get_job_config
    from cray_megatron.models.get_latest_checkpoint_path import (
        get_latest_checkpoint_path,
    )

    if get_rank() != 0:
        return

    try:
        job_config = get_job_config()
    except Exception as e:
        logger.warning("Could not load config for HF upload: %s", e)
        return

    if not job_config.get("upload_to_hf", False):
        return

    hf_repo_id = job_config.get("hf_repo_id", "")
    hf_token = job_config.get("hf_upload_token", "")
    if not hf_repo_id:
        logger.warning("upload_to_hf enabled but hf_repo_id is empty; skipping upload.")
        return
    if not hf_token:
        logger.warning(
            "upload_to_hf enabled but hf_upload_token is empty; skipping upload."
        )
        return

    checkpoint_path = get_latest_checkpoint_path()
    if not checkpoint_path:
        logger.warning("No checkpoint found to upload to HuggingFace.")
        record_hf_upload_result(error="no checkpoint found to upload")
        return

    do_upload(hf_repo_id, hf_token, checkpoint_path)


def do_upload(hf_repo_id, hf_token, checkpoint_path):
    """Perform the upload with offline mode suspended for its duration."""
    # Clear the offline flags for the upload only, then restore them.
    saved_offline = {k: os.environ.get(k) for k in OFFLINE_VARS}
    overridden = [k for k, v in saved_offline.items() if v not in (None, "0")]
    for key in OFFLINE_VARS:
        os.environ.pop(key, None)
    if overridden:
        logger.info(
            "upload_to_hf: temporarily clearing %s so the upload can reach the Hub",
            ", ".join(overridden),
        )

    # huggingface_hub caches HF_HUB_OFFLINE at import time, so clearing the env is not enough.
    saved_constant = None
    try:
        from huggingface_hub import constants as _hf_constants

        saved_constant = _hf_constants.HF_HUB_OFFLINE
        if saved_constant:
            _hf_constants.HF_HUB_OFFLINE = False
            logger.info(
                "upload_to_hf: suspending huggingface_hub offline mode "
                "(constants.HF_HUB_OFFLINE True -> False) for the upload"
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not suspend huggingface_hub offline mode: %s", e)

    try:
        import huggingface_hub

        logger.info("Uploading checkpoint to HuggingFace repo %s", hf_repo_id)
        api = huggingface_hub.HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=os.path.basename(checkpoint_path),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint {os.path.basename(checkpoint_path)}",
        )
        logger.info("Uploaded checkpoint to %s", hf_repo_id)
        record_hf_upload_result(repo_id=hf_repo_id)
    except Exception as e:
        # Non-fatal: the job stays COMPLETED and the failure is surfaced through status.json.
        logger.error("Failed to upload to HuggingFace: %s", e)
        record_hf_upload_result(repo_id=hf_repo_id, error=str(e))
    finally:
        for key, value in saved_offline.items():
            if value is not None:
                os.environ[key] = value
        if saved_constant:
            try:
                from huggingface_hub import constants as _hf_constants

                _hf_constants.HF_HUB_OFFLINE = saved_constant
            except Exception:  # pragma: no cover - defensive
                pass


def record_hf_upload_result(repo_id=None, error=None):
    """Surface the upload outcome in status.json without changing job status."""
    try:
        from cray_megatron.megatron.training_harness import get_status, save_status

        status = get_status()
        status["hf_upload"] = "failed" if error else "ok"
        if repo_id:
            status["hf_upload_repo"] = repo_id
        if error:
            status["hf_upload_error"] = error
        save_status(status)
    except Exception as e:
        logger.warning("Could not record HF upload result in status: %s", e)
