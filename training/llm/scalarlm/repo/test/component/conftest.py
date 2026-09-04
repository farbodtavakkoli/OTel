"""
Component-layer fixtures.

Component tests run against real, locally-spawned services (slurmctld
inside the CPU container, in-process FastAPI via start_cray_server, etc.)
but stub anything that would require GPUs, large model downloads, or
multi-node orchestration. The runner (cmd/test_command.sh) starts
slurmctld before invoking pytest at this level, so these tests can
shell out to `sbatch` / `scontrol` / `squeue` directly.
"""

import os
import shutil
import subprocess

import pytest


def _slurm_available() -> bool:
    """True iff `squeue` is on PATH and responds without erroring. Used
    by the slurm_running fixture to skip cleanly on hosts without slurm
    (a contributor running pytest on their laptop), while the runner
    inside the CPU image always satisfies it."""
    if shutil.which("squeue") is None:
        return False
    try:
        return (
            subprocess.run(
                ["squeue"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0
        )
    except OSError:
        return False


@pytest.fixture
def slurm_running():
    """Skip the test when slurm isn't reachable. The CPU image's start
    script ensures slurmctld/slurmd are up before pytest fires; outside
    that context the test simply skips rather than failing noisily."""
    if not _slurm_available():
        pytest.skip("slurm not available — run inside the CPU container via "
                    "./scalarlm test --level component")
