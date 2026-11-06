import os
import shutil
import subprocess

import pytest


def _slurm_available() -> bool:
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
    if not _slurm_available():
        pytest.skip("slurm not available — run inside the CPU container via "
                    "./scalarlm test --level component")
