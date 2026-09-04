"""
Unit tests for the manual restart endpoint backing
infra/cray_infra/training/restart.py.

Contract:
- Only FAILED or CANCELLED jobs may be restarted (everything else
  returns 409).
- Restart clears the stale `error`/`output` fields from status.json
  so the UI's error chip disappears the moment the badge flips to
  QUEUED. Other fields (history, job_directory, …) are preserved.
- Restart re-launches via start_slurm_job — does NOT go through
  launch_training_job (which would short-circuit on the existing
  status.json).
- 404 when the job isn't on disk; 404 when its config.yaml is missing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException

from cray_infra.training import restart as restart_mod


def _write_job(
    tmp_path,
    *,
    hash_id="abc123",
    status="FAILED",
    config=None,
    extra_status_fields=None,
):
    """Lay out a minimal training-job directory like the server expects."""
    job_dir = tmp_path / hash_id
    job_dir.mkdir()
    status_doc = {
        "status": status,
        "job_id": "42",
        "start_time": 1234567890,
        "error": "previous-run-error",
        "output": "previous-run-stderr",
    }
    if extra_status_fields:
        status_doc.update(extra_status_fields)
    (job_dir / "status.json").write_text(json.dumps(status_doc))
    cfg = config or {"job_directory": str(job_dir), "llm_name": "m", "gpus": 0}
    (job_dir / "config.yaml").write_text(yaml.dump(cfg))
    return job_dir, status_doc


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    """Point training_job_directory at tmp_path for every config lookup."""

    def _get_config():
        return {"training_job_directory": str(tmp_path)}

    monkeypatch.setattr(
        "cray_infra.training.get_training_job_info.get_config", _get_config
    )
    return _get_config


@pytest.mark.asyncio
async def test_restart_failed_job_clears_error_and_calls_start_slurm_job(
    tmp_path, fake_config
):
    job_dir, _ = _write_job(tmp_path, status="FAILED")

    with patch.object(restart_mod, "start_slurm_job") as start:
        response = await restart_mod.restart("abc123")

    # 1. sbatch was invoked with the job's persisted config.
    assert start.call_count == 1
    submitted_config = start.call_args.args[0]
    assert submitted_config["job_directory"] == str(job_dir)

    # 2. status.json now reads QUEUED and the previous-run error/output
    #    are gone so the UI's red error chip clears alongside the badge.
    on_disk = json.loads((job_dir / "status.json").read_text())
    assert on_disk["status"] == "QUEUED"
    assert "error" not in on_disk
    assert "output" not in on_disk

    # 3. The response carries the refreshed status row (start_slurm_job
    #    is mocked, so it's exactly what we just wrote).
    assert response.job_status["status"] == "QUEUED"
    assert response.deployed is False


@pytest.mark.asyncio
async def test_restart_cancelled_job_is_allowed(tmp_path, fake_config):
    """CANCELLED is the second restart-from state (user paused, now
    resuming) — confirm the same path works."""
    job_dir, _ = _write_job(tmp_path, status="CANCELLED")

    with patch.object(restart_mod, "start_slurm_job") as start:
        await restart_mod.restart("abc123")

    assert start.call_count == 1
    assert json.loads((job_dir / "status.json").read_text())["status"] == "QUEUED"


@pytest.mark.parametrize("running_status", ["QUEUED", "TRAINING", "COMPLETED"])
@pytest.mark.asyncio
async def test_restart_rejects_non_restartable_status(
    tmp_path, fake_config, running_status
):
    """
    Restarting a job that's still running would double-launch; restarting
    a completed job has nothing to resume. Both should 409 without
    touching status.json or invoking sbatch.
    """
    job_dir, original = _write_job(tmp_path, status=running_status)

    with patch.object(restart_mod, "start_slurm_job") as start:
        with pytest.raises(HTTPException) as exc:
            await restart_mod.restart("abc123")

    assert exc.value.status_code == 409
    assert running_status in exc.value.detail or repr(running_status) in exc.value.detail
    assert start.call_count == 0
    # status.json untouched.
    assert json.loads((job_dir / "status.json").read_text()) == original


@pytest.mark.asyncio
async def test_restart_missing_job_returns_404(tmp_path, fake_config):
    with patch.object(restart_mod, "start_slurm_job") as start:
        with pytest.raises(HTTPException) as exc:
            await restart_mod.restart("does-not-exist")
    assert exc.value.status_code == 404
    assert start.call_count == 0


@pytest.mark.asyncio
async def test_restart_preserves_unrelated_status_fields(tmp_path, fake_config):
    """history, job_directory, and any other fields the trainer wrote
    must survive — only the failure-era ones get dropped."""
    job_dir, _ = _write_job(
        tmp_path,
        status="FAILED",
        extra_status_fields={
            "history": [{"step": 10, "loss": 1.2}],
            "max_steps": 100,
        },
    )

    with patch.object(restart_mod, "start_slurm_job"):
        await restart_mod.restart("abc123")

    on_disk = json.loads((job_dir / "status.json").read_text())
    assert on_disk["history"] == [{"step": 10, "loss": 1.2}]
    assert on_disk["max_steps"] == 100
    assert on_disk["status"] == "QUEUED"
