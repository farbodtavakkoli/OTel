"""Regression tests for scanning persisted training-job status files."""

import asyncio
import json
import logging
from unittest.mock import Mock

import pytest

import cray_infra.training.restart_megatron_jobs as subject


def _write_status(tmp_path, job_name, contents, *, raw=False):
    job_dir = tmp_path / job_name
    job_dir.mkdir()
    status_path = job_dir / "status.json"
    if raw:
        status_path.write_text(contents)
    else:
        status_path.write_text(json.dumps(contents))
    return job_dir


def _collect_running_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subject,
        "get_config",
        lambda: {"training_job_directory": str(tmp_path)},
    )

    async def collect():
        return [job async for job in subject.get_running_jobs()]

    return asyncio.run(collect())


def test_get_running_jobs_yields_only_training_and_queued(monkeypatch, tmp_path):
    training = _write_status(tmp_path, "training", {"status": "TRAINING"})
    queued = _write_status(tmp_path, "queued", {"status": "QUEUED"})
    _write_status(tmp_path, "completed", {"status": "COMPLETED"})

    assert set(_collect_running_jobs(monkeypatch, tmp_path)) == {
        str(training),
        str(queued),
    }


def test_get_running_jobs_warns_and_skips_missing_status(monkeypatch, tmp_path, caplog):
    _write_status(tmp_path, "missing-status", {"message": "still starting"})
    print_exc = Mock()
    monkeypatch.setattr(subject.traceback, "print_exc", print_exc)
    caplog.set_level(logging.WARNING, logger=subject.__name__)

    assert _collect_running_jobs(monkeypatch, tmp_path) == []
    assert "has no 'status' key, skipping" in caplog.text
    print_exc.assert_not_called()


@pytest.mark.parametrize("payload", [None, [], "TRAINING"])
def test_get_running_jobs_logs_and_skips_non_object_json(
    monkeypatch, tmp_path, caplog, payload
):
    _write_status(tmp_path, "non-object", payload)
    print_exc = Mock()
    monkeypatch.setattr(subject.traceback, "print_exc", print_exc)
    caplog.set_level(logging.ERROR, logger=subject.__name__)

    assert _collect_running_jobs(monkeypatch, tmp_path) == []
    assert "is not a JSON object" in caplog.text
    print_exc.assert_not_called()


def test_get_running_jobs_does_not_log_large_non_object_payload(
    monkeypatch, tmp_path, caplog
):
    marker = "private-payload-marker"
    _write_status(tmp_path, "large-non-object", marker + ("x" * 100_000))
    caplog.set_level(logging.ERROR, logger=subject.__name__)

    assert _collect_running_jobs(monkeypatch, tmp_path) == []
    assert "is not a JSON object (got str)" in caplog.text
    assert marker not in caplog.text
    assert len(caplog.text) < 1_000


def test_get_running_jobs_keeps_traceback_for_malformed_json(
    monkeypatch, tmp_path, caplog
):
    _write_status(tmp_path, "malformed", "{", raw=True)
    print_exc = Mock()
    monkeypatch.setattr(subject.traceback, "print_exc", print_exc)
    caplog.set_level(logging.ERROR, logger=subject.__name__)

    assert _collect_running_jobs(monkeypatch, tmp_path) == []
    assert "Error reading status.json" in caplog.text
    print_exc.assert_called_once_with()


def test_get_running_jobs_does_not_call_null_status_missing(
    monkeypatch, tmp_path, caplog
):
    _write_status(tmp_path, "null-status", {"status": None})
    caplog.set_level(logging.WARNING, logger=subject.__name__)

    assert _collect_running_jobs(monkeypatch, tmp_path) == []
    assert "has no 'status' key" not in caplog.text
