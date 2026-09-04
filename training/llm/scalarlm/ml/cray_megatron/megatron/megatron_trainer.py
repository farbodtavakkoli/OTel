import os
from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.print_logo import print_logo

from cray_megatron.megatron.training_loop import TrainingLoop, get_max_steps
from cray_megatron.megatron.training_harness import TrainingHarness
from cray_megatron.megatron import stop_flag

import sys

import logging

logger = logging.getLogger(__name__)

from cray_infra.training.train_debug import is_train_debug_enabled

import sys
import time


def _trace_trainer(msg: str) -> None:
    """CRAY_TRAIN_DEBUG tracing for the outer trainer steps (PR #5)."""
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[rank={rank}] trainer [{time.monotonic():.3f}]: {msg}\n")
    sys.stderr.flush()


class MegatronTrainer:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

    def train(self):
        self.train_loop()

    def train_loop(self):
        self.training_harness.update_status(
            status=TrainingJobStatus.TRAINING, metadata={"max_steps": get_max_steps()}
        )

        print_logo()

        TrainingLoop(self.training_harness).train()

        # TrainingLoop._finalize_slice already wrote the right status; COMPLETED would clobber it.
        if not stop_flag.was_stop_requested():
            self.training_harness.update_status(status=TrainingJobStatus.COMPLETED)
