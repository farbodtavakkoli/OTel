from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.huggingface.get_hf_token import get_hf_token

from cray_megatron.megatron.training_harness import TrainingHarness
from cray_megatron.megatron import stop_flag
from cray_megatron.models.get_latest_checkpoint_path import get_latest_checkpoint_path
from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.main_rank_only import is_main_rank
# Upload lives in its own module: importing cray_megatron.main initialises torch.distributed.
from cray_megatron.hf_upload import upload_to_hf_if_enabled

import traceback
import sys
import os
from cray_infra.training.distributed import finalize_mpi, get_rank, init as init_distributed
from cray_infra.training.train_debug import (
    is_fault_handler_enabled,
    is_train_debug_enabled,
)

import faulthandler
import time

# A rank stuck in a collective raises nothing; faulthandler dumps every thread stack on a timer.
if is_fault_handler_enabled():
    faulthandler.enable()
    faulthandler.dump_traceback_later(timeout=600, repeat=True)


def _boot(msg: str) -> None:
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[rank={rank}] boot pid={os.getpid()} t={time.monotonic():.3f}: {msg}\n")
    sys.stderr.flush()

_boot("main.py entered")


def print_exception():
    print(f"Rank {get_rank()} hit exception")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)


try:
    from cray_megatron.megatron.megatron_trainer import MegatronTrainer
except Exception as e:
    print_exception()

import signal
import logging

logger = logging.getLogger(__name__)

def main():

    # Bring up the process group first: init() sets the device so model and communicator match.
    init_distributed()

    harness = TrainingHarness()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    try:
        setup_logging()
        setup_signal_handler(harness)

        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()

        # After train(), so the post-loop checkpoint is already on disk. Best-effort, rank 0 only.
        upload_to_hf_if_enabled()
    except Exception as e:
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        raise e

    finalize_mpi()


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    # Central noisy-library list lives in infra/cray_infra/util/quiet_loggers.py.
    try:
        from cray_infra.util.quiet_loggers import quiet_noisy_loggers
        quiet_noisy_loggers()
    except Exception:
        logging.getLogger("filelock").setLevel(logging.WARNING)

    logging.getLogger("cray_megatron.megatron.distribution.fsdp").setLevel(
        logging.INFO
    )

def setup_signal_handler(harness):
    def signal_handler(sig, frame):
        # Do not sys.exit here: set the flag and let TrainingLoop unwind through its checkpoint.
        logger.warning("Received signal %s — requesting graceful stop", sig)
        stop_flag.request_stop(signal_number=sig)

    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


main()
