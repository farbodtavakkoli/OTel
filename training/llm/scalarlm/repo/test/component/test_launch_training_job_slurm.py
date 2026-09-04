"""
Component-level test for launch_training_job against a real slurmctld.

The unit tests in test/unit/test_launch_training_job_relaunch.py stub
out subprocess and config; this one runs the real flow: an actual
`sbatch` call against the slurmctld that the test runner starts, with
a tiny shell-only entrypoint so we don't pull in torch/mpirun. The
point is to catch breakage in the boundary the unit tests can't see —
that the constructed sbatch argv parses and slurm accepts the
--signal=B:TERM@N flag (the long-jobs grace window, training-lifecycle.md
§5.4).
"""

import asyncio
import json
import os
import subprocess
import textwrap
import time

import pytest


def _scancel(job_id: str) -> None:
    if not job_id:
        return
    subprocess.run(
        ["scancel", job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _job_in_squeue(job_id: str) -> bool:
    out = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and out.stdout.strip() == job_id


def _write_noop_entrypoint(path: str) -> None:
    """Tiny stand-in for scripts/train_job_entrypoint.sh. Just exits 0
    after a short sleep — enough for the test to observe the job in
    squeue before it completes, without needing python/torch/mpirun.
    launch_training_job copies this verbatim into the job dir and
    string-substitutes REPLACE_CONFIG_PATH (which we ignore here)."""
    with open(path, "w") as f:
        f.write(textwrap.dedent("""\
            #!/bin/bash
            # CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH
            echo "[noop-entrypoint] started"
            sleep 2
            echo "[noop-entrypoint] done"
        """))
    os.chmod(path, 0o755)


@pytest.fixture
def job_workspace(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    entrypoint = tmp_path / "noop_entrypoint.sh"
    _write_noop_entrypoint(str(entrypoint))

    monkeypatch.setenv("SCALARLM_TRAINING_JOB_DIRECTORY", str(jobs_dir))
    monkeypatch.setenv("SCALARLM_TRAIN_JOB_ENTRYPOINT", str(entrypoint))
    monkeypatch.setenv("SCALARLM_MAX_TRAIN_TIME", "60")
    monkeypatch.setenv("SCALARLM_EXTRA_TRAINING_SECONDS", "0")
    monkeypatch.setenv("SCALARLM_SIGNAL_GRACE_SECONDS", "10")

    submitted_jobs = []
    yield jobs_dir, submitted_jobs
    for jid in submitted_jobs:
        _scancel(jid)


def test_sbatch_accepts_signal_flag_end_to_end(slurm_running, job_workspace):
    """End-to-end: launch_training_job builds an sbatch argv with the
    --signal=B:TERM@N flag and a real slurmctld accepts it. status.json
    records the job_id; the id is reachable via squeue/sacct. A unit-
    level stub can't catch a sbatch-template regression where slurm
    rejects --signal=B:TERM@N — only a real sbatch round-trip can."""
    from cray_infra.training.launch_training_job import launch_training_job

    jobs_dir, submitted = job_workspace
    job_hash = "test-component-launch"
    train_args = {
        "job_directory": str(jobs_dir / job_hash),
        "training_data_path": "ignored-by-noop-entrypoint",
        "dataset_hash": "deadbeef",
        "timeout": 30,
        "gpus": 0,
        "nodes": 1,
    }

    result = asyncio.run(launch_training_job(train_args))
    job_id = str(result["job_id"])
    submitted.append(job_id)

    status_path = os.path.join(train_args["job_directory"], "status.json")
    with open(status_path) as f:
        status = json.load(f)
    assert status["status"] == "QUEUED"
    assert status["job_id"] == job_id
    assert "start_time" in status

    # If sbatch had rejected the --signal flag with EINVAL, we wouldn't
    # have gotten a job_id back at all — so any non-empty id implicitly
    # proves the flag was accepted. We additionally confirm the job is
    # visible to slurm via squeue or (for very-short-lived jobs) sacct.
    deadline = time.time() + 5
    seen_in_squeue = False
    while time.time() < deadline:
        if _job_in_squeue(job_id):
            seen_in_squeue = True
            break
        time.sleep(0.2)
    if not seen_in_squeue:
        sacct = subprocess.run(
            ["sacct", "-n", "-j", job_id, "-o", "JobID"],
            capture_output=True,
            text=True,
        )
        assert job_id in sacct.stdout, (
            f"job_id {job_id} not in squeue nor sacct — sbatch likely "
            f"didn't actually queue. sacct out: {sacct.stdout!r}"
        )
