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
