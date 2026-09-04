"""
Unit tests for cray_infra.training.list_checkpoints.

Powers the checkpoint dropdown on the Publish-to-HF modal — every
`checkpoint_<step>.pt` in the job directory, newest first.
"""

import os
from unittest.mock import patch

import pytest

from cray_infra.training import list_checkpoints as mod


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "deadbeef"
    d.mkdir()
    return d


@pytest.fixture
def resolve(job_dir):
    with patch(
        "cray_infra.training.list_checkpoints.get_job_directory_for_hash",
        return_value=str(job_dir),
    ):
        yield


def _touch(path, mtime: float):
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))


def test_returns_checkpoints_newest_first(job_dir, resolve):
    _touch(job_dir / "checkpoint_5.pt", 1_000_000.0)
    _touch(job_dir / "checkpoint_100.pt", 1_000_500.0)
    _touch(job_dir / "checkpoint_27.pt", 1_000_300.0)
    out = mod.list_checkpoints("deadbeef")
    steps = [c["step"] for c in out["checkpoints"]]
    assert steps == [100, 27, 5]


def test_includes_mtime(job_dir, resolve):
    _touch(job_dir / "checkpoint_1.pt", 1_234_567.5)
    out = mod.list_checkpoints("deadbeef")
    assert out["checkpoints"][0]["mtime"] == pytest.approx(1_234_567.5)


def test_skips_non_checkpoint_files(job_dir, resolve):
    _touch(job_dir / "checkpoint_3.pt", 1_000_000.0)
    _touch(job_dir / "config.yaml", 1_000_000.0)
    _touch(job_dir / "checkpoint_3.txt", 1_000_000.0)
    _touch(job_dir / "checkpoint_x.pt", 1_000_000.0)  # bad step
    out = mod.list_checkpoints("deadbeef")
    assert [c["name"] for c in out["checkpoints"]] == ["checkpoint_3.pt"]


def test_returns_empty_list_when_no_checkpoints(job_dir, resolve):
    _touch(job_dir / "config.yaml", 1_000_000.0)
    out = mod.list_checkpoints("deadbeef")
    assert out == {"checkpoints": []}
