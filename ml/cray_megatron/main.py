from cray_infra.training.training_harness import TrainingHarness
from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.huggingface.get_hf_token import get_hf_token

from ml.get_local_job_config import load_local_training_config
from cray_megatron.models.get_latest_checkpoint_path import get_latest_checkpoint_path
from gpu_aware_mpi import get_rank

import traceback
import sys
import os
import warnings
from gpu_aware_mpi import finalize_mpi

# Perhaps only print on rank 1
def print_exception():
    if get_rank() != 0:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)


try:
    from cray_megatron.megatron.megatron_trainer import MegatronTrainer
except Exception as e:
    print_exception()

import signal
import logging


def main():

    harness = TrainingHarness()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    try:
        setup_logging()
        setup_signal_handler(harness)

        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
        
        upload_to_hf_if_enabled()
    except Exception as e:
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        #raise e

    finalize_mpi()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("cray_infra.training.distribution_strategy.fsdp.fsdp").setLevel(
        logging.INFO
    )
    # Suppress noisy urllib3 HTTP request logs (HEAD/GET requests to HuggingFace)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Dedupe warnings - only show warnings from rank 0 to avoid duplicates across processes
    if get_rank() != 0:
        warnings.filterwarnings("ignore", message="Full backward hook")
        # Suppress transformers warnings on non-rank-0 (use_cache warning uses logger.warning_once)
        logging.getLogger("transformers").setLevel(logging.ERROR)

def setup_signal_handler(harness):
    def signal_handler(sig, frame):
        logger.warn("Received signal: ", sig)
        harness.update_status(status=TrainingJobStatus.QUEUED)

        sys.exit(0)

    signal.signal(signal.SIGCONT, signal_handler)


def upload_to_hf_if_enabled():
    if get_rank() != 0:
        return
    
    logger = logging.getLogger(__name__)
    
    try:
        job_config = load_local_training_config()
    except Exception as e:
        logger.warning(f"Could not load local training config for HF upload: {e}")
        return
    
    if not job_config.get("upload_to_hf", False):
        return
    
    hf_repo_id = job_config.get("hf_repo_id", "")
    hf_token = job_config.get("hf_upload_token", "")
    
    if not hf_repo_id:
        logger.warning("upload_to_hf is enabled but hf_repo_id is not set. Skipping upload.")
        return
    
    if not hf_token:
        logger.warning("upload_to_hf is enabled but hf_upload_token is not set. Skipping upload.")
        return
    
    checkpoint_path = get_latest_checkpoint_path()
    if not checkpoint_path:
        logger.warning("No checkpoint found to upload.")
        return
    
    try:
        from huggingface_hub import HfApi
        
        logger.info(f"Uploading checkpoint to HuggingFace: {hf_repo_id}")
        
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
        
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=os.path.basename(checkpoint_path),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint {os.path.basename(checkpoint_path)}",
        )
        
        logger.info(f"Successfully uploaded checkpoint to {hf_repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")


main()
