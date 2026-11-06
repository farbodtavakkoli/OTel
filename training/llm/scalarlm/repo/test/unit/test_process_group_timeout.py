from datetime import timedelta
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from cray_infra.training import distributed
from cray_infra.util.default_job_config import JobConfig


REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "timeout-test",
}


def test_process_group_timeout_defaults_to_one_hour():
    config = JobConfig(**REQUIRED_FIELDS)
    assert config.process_group_timeout_seconds == 3600
    assert config.model_dump()["process_group_timeout_seconds"] == 3600


@pytest.mark.parametrize("seconds", [1, 7200])
def test_process_group_timeout_accepts_positive_override(seconds):
    config = JobConfig(**REQUIRED_FIELDS, process_group_timeout_seconds=seconds)
    assert config.model_dump()["process_group_timeout_seconds"] == seconds


@pytest.mark.parametrize("seconds", [0, -1, None, "invalid", 1.5])
def test_process_group_timeout_rejects_invalid_values(seconds):
    with pytest.raises(ValidationError) as error:
        JobConfig(**REQUIRED_FIELDS, process_group_timeout_seconds=seconds)
    assert error.value.errors()[0]["loc"] == ("process_group_timeout_seconds",)


@pytest.mark.parametrize("seconds", [None, 1, 7200])
def test_nccl_init_uses_timeout_from_job_config(tmp_path, monkeypatch, seconds):
    config = dict(REQUIRED_FIELDS)
    if seconds is not None:
        config["process_group_timeout_seconds"] = seconds
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CRAY_TRAINING_JOB_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(distributed, "_trace", Mock())
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    calls = Mock()
    monkeypatch.setattr(distributed.torch.cuda, "set_device", calls.set_device)
    monkeypatch.setattr(distributed.dist, "init_process_group", calls.init_process_group)

    distributed.init()

    calls.set_device.assert_called_once_with(2)
    calls.init_process_group.assert_called_once_with(
        backend="nccl", timeout=timedelta(seconds=3600 if seconds is None else seconds)
    )
    assert [call[0] for call in calls.mock_calls] == ["set_device", "init_process_group"]


def test_nccl_init_already_initialized_does_not_load_config(monkeypatch):
    monkeypatch.setattr(distributed, "_trace", Mock())
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    load_config = Mock()
    init_process_group = Mock()
    monkeypatch.setattr(distributed, "get_job_config", load_config)
    monkeypatch.setattr(distributed.dist, "init_process_group", init_process_group)

    distributed.init()

    load_config.assert_not_called()
    init_process_group.assert_not_called()
