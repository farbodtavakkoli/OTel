import os

import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FSDP2 requires a GPU"
)


@pytest.fixture
def job_config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"

    def write(adapter_type):
        path.write_text(
            "job_directory: /tmp\n"
            "training_data_path: /tmp/dataset.jsonlines\n"
            "dataset_hash: smoke\n"
            "hsdp_shard_size: 1\n"
            "reshard_after_forward: false\n"
            f"adapter_type: {adapter_type}\n"
        )

    monkeypatch.setenv("CRAY_TRAINING_JOB_CONFIG_PATH", str(path))
    return write


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4)
        self.register_buffer("scale", torch.full((4,), 2.0))
        for param in self.base.parameters():
            param.requires_grad_(False)


def _wrapped(seed):
    from cray_infra.training.distributed import cuda_device
    from cray_megatron.megatron.distribution.pytorch_fsdp import PyTorchFSDP

    torch.manual_seed(seed)
    return PyTorchFSDP(_Tiny().to(cuda_device()))


def _full(wrapped, name):
    return wrapped.unwrap_model()[name].float()


def test_adapter_run_saves_only_trainable_params_and_buffers(job_config):
    job_config("lora")
    saved = _wrapped(0).unwrap_model()
    assert set(saved) == {"adapter.weight", "adapter.bias", "scale"}


def test_full_parameter_run_saves_every_param(job_config):
    job_config("none")
    saved = _wrapped(0).unwrap_model()
    assert {"base.weight", "base.bias", "adapter.weight", "adapter.bias"} <= set(saved)


def test_adapter_checkpoint_restores_trainable_and_keeps_frozen(job_config):
    job_config("lora")
    saved = _wrapped(0).unwrap_model()

    fresh = _wrapped(1)
    job_config("none")
    frozen_before = {k: _full(fresh, k).clone() for k in ("base.weight", "base.bias")}
    job_config("lora")
    assert not torch.allclose(_full(fresh, "adapter.weight"), saved["adapter.weight"].float())

    expected = {k: v.clone().float() for k, v in saved.items()}
    fresh.load_unwrapped_model(saved)

    for name in ("adapter.weight", "adapter.bias"):
        assert torch.allclose(_full(fresh, name), expected[name]), name
    job_config("none")
    for name, before in frozen_before.items():
        assert torch.equal(_full(fresh, name), before), name


def test_adapter_load_raises_on_unrelated_state_dict(job_config):
    job_config("lora")
    fresh = _wrapped(0)
    with pytest.raises(RuntimeError, match="matched no trainable parameter"):
        fresh.load_unwrapped_model({"totally.unrelated": torch.zeros(2)})


def test_adapter_load_raises_on_extra_keys(job_config):
    job_config("lora")
    saved = _wrapped(0).unwrap_model()
    saved["adapter.extra_lora_A.weight"] = torch.zeros(2, 4)
    with pytest.raises(RuntimeError):
        _wrapped(1).load_unwrapped_model(saved)


def test_full_parameter_load_stays_strict(job_config):
    job_config("none")
    saved = _wrapped(0).unwrap_model()
    del saved["base.weight"]
    with pytest.raises(RuntimeError):
        _wrapped(1).load_unwrapped_model(saved)
