from cray_infra.training.distributed import finalize_mpi

from cray_infra.training.training_job_status import TrainingJobStatus

import contextlib

@contextlib.contextmanager
def training_job_context():
    # TrainingHarness lives in the ml tree (cray_megatron.megatron.training_harness),
    # not in cray_infra. This module previously imported it from
    # `cray_infra.training.training_harness`, which has never existed -- so
    # `training_job_context()` raised ModuleNotFoundError on import and every
    # custom Slurm job that used it died before running a line of user code.
    # (Pre-existing upstream bug, present at baseline and on main; it went
    # unnoticed because the tests that use this context manager only printed
    # their results and were never executed.)
    #
    # Imported lazily and inside the function: cray_infra is the server-side
    # layer and must not take a module-level dependency on the ml tree, which is
    # only guaranteed to be importable inside a training job (the launcher puts
    # the job's own ml/ first on PYTHONPATH).
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


