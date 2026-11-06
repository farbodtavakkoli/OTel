from cray_infra.training.distributed import finalize_mpi

from cray_infra.training.training_job_status import TrainingJobStatus

import contextlib

@contextlib.contextmanager
def training_job_context():
    from cray_megatron.megatron.training_harness import TrainingHarness

    harness = TrainingHarness()

    harness.update_status(TrainingJobStatus.TRAINING)

    try:
        yield harness
        harness.update_status(TrainingJobStatus.COMPLETED)
    except Exception as e:
        harness.update_status(TrainingJobStatus.FAILED, metadata={"error": str(e)})
    finally:
        finalize_mpi()
