import builtins
import json
import logging
from types import SimpleNamespace

import pytest
import torch

import cray_megatron.megatron.training_harness as harness
from cray_megatron.megatron.training_harness import TrainingHarness, get_status
from cray_megatron.megatron.training_loop import TrainingLoop


@pytest.fixture
def job_dir(tmp_path, monkeypatch):
    directory = tmp_path / "job"
    directory.mkdir()
    data_path = tmp_path / "dataset.jsonl"
    data_path.write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"job_directory: {directory}",
                f"training_data_path: {data_path}",
                "dataset_hash: testhash",
            ]
        )
    )
    monkeypatch.setenv("CRAY_TRAINING_JOB_CONFIG_PATH", str(config_path))
    return directory


@pytest.fixture(autouse=True)
def rank_zero(monkeypatch):
    import cray_megatron.collectives.main_rank_only as m

    m._in_main_rank_only = False
    monkeypatch.setattr(m, "get_rank", lambda: 0)
    monkeypatch.setattr(m, "barrier", lambda: None)
    yield
    m._in_main_rank_only = False


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_missing_status_returns_unknown_without_error(job_dir, caplog):
    caplog.set_level(logging.DEBUG)
    assert get_status() == {"status": "unknown"}
    assert _errors(caplog) == []


def test_existing_status_is_returned_unchanged(job_dir, caplog):
    contents = {"status": "TRAINING", "history": [{"step": 1, "loss": 0.5}], "max_steps": 5}
    (job_dir / "status.json").write_text(json.dumps(contents))
    caplog.set_level(logging.DEBUG)
    assert get_status() == contents
    assert _errors(caplog) == []


def test_malformed_status_logs_error(job_dir, caplog):
    (job_dir / "status.json").write_text("{not json")
    caplog.set_level(logging.DEBUG)
    assert get_status() == {"status": "unknown"}
    assert any("Error loading job status" in r.getMessage() for r in _errors(caplog))


def test_permission_denied_logs_error(job_dir, caplog, monkeypatch):
    (job_dir / "status.json").write_text(json.dumps({"status": "TRAINING"}))
    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if str(path).endswith("status.json"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny)
    caplog.set_level(logging.DEBUG)
    assert get_status() == {"status": "unknown"}
    assert any("Error loading job status" in r.getMessage() for r in _errors(caplog))


def test_update_status_on_new_job_writes_status_without_error(job_dir, caplog):
    caplog.set_level(logging.DEBUG)
    TrainingHarness().update_status("TRAINING", metadata={"max_steps": 5})
    written = json.loads((job_dir / "status.json").read_text())
    assert written["status"] == "TRAINING"
    assert written["max_steps"] == 5
    assert "last_updated" in written
    assert _errors(caplog) == []
    assert harness.get_status()["status"] == "TRAINING"


def _device_line(device, caplog):
    stub = SimpleNamespace(
        training_state=SimpleNamespace(model_info={"distribution_strategy": {"device": device}})
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cray_megatron.megatron.training_loop"):
        TrainingLoop.print_device_info(stub)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Using device")]
    assert len(lines) == 1
    return lines[0]


def test_device_log_full_cuda_device(caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _device_line(torch.device("cuda:3"), caplog) == "Using device cuda:3"


def test_device_log_single_gpu_integer(caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _device_line(0, caplog) == "Using device cuda:0"


def test_device_log_cpu(caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _device_line(torch.device("cpu"), caplog) == "Using device cpu"


class _NeuronDevice:
    type = "neuron"
    index = None

    def __str__(self):
        return "neuron"


def test_device_log_neuron(caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _device_line(_NeuronDevice(), caplog) == "Using device neuron"


@pytest.mark.parametrize("device", [torch.device("cuda:3"), 0, torch.device("cpu")])
def test_device_log_never_doubles_cuda(device, caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device != torch.device("cpu"))
    assert "cuda:cuda" not in _device_line(device, caplog)
