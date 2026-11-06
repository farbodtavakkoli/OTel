import json
import os

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from cray_megatron.megatron.distribution.no_distribution import NoDistribution
from cray_megatron.megatron.training_loop import (
    TrainingLoop,
    _load_trained_parameters,
)
from cray_megatron.megatron.training_harness import TrainingHarness


@pytest.fixture
def job_config_env(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    data_path = tmp_path / "dataset.jsonl"
    data_path.write_text("")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"job_directory: {job_dir}",
                f"training_data_path: {data_path}",
                "dataset_hash: testhash",
                "max_steps: 10",
                "steps_per_checkpoint: 5",
                "cuda_memory_log_interval: 0",
                "timeout: 3600",
                "adapter_type: tokenformer",
            ]
        )
    )
    monkeypatch.setenv("CRAY_TRAINING_JOB_CONFIG_PATH", str(config_path))
    return job_dir


@pytest.fixture(autouse=True)
def rank_zero(monkeypatch):
    import cray_megatron.collectives.main_rank_only as m

    m._in_main_rank_only = False
    monkeypatch.setattr(m, "get_rank", lambda: 0)
    monkeypatch.setattr(m, "barrier", lambda: None)
    yield
    m._in_main_rank_only = False


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.base_proj = nn.Linear(hidden, hidden)
        self.adapter = nn.Parameter(torch.zeros(hidden, hidden))


class _Decoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden)
        self.layers = nn.ModuleList([_Block(hidden)])


class _CausalLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.model = _Decoder(hidden)
        self.lm_head = nn.Linear(hidden, hidden)


def _set_trainable(causal_lm):
    for p in causal_lm.parameters():
        p.requires_grad = False
    trainable = set()
    for name, p in causal_lm.named_parameters():
        if name.endswith(".adapter") or name.startswith("lm_head"):
            p.requires_grad = True
            trainable.add(name)
    return trainable


def test_resolver_loads_into_depth_one_wrapper():
    hidden = 4
    trained = _CausalLM(hidden)
    _set_trainable(trained)
    with torch.no_grad():
        trained.lm_head.weight.fill_(3.0)
        trained.model.layers[0].adapter.fill_(7.0)

    from cray_megatron.megatron.training_loop import filter_checkpoint

    state_dict = filter_checkpoint(trained, trained.state_dict())
    assert {"lm_head.weight", "model.layers.0.adapter"} <= set(state_dict)
    assert all(not k.startswith("model.model") for k in state_dict)

    fresh = NoDistribution(_CausalLM(hidden))

    target = _load_trained_parameters(fresh, state_dict)

    assert target is fresh.model
    assert torch.all(fresh.model.lm_head.weight == 3.0)
    assert torch.all(fresh.model.model.layers[0].adapter == 7.0)


def test_resolver_loads_into_depth_two_wrapper():
    hidden = 4

    class _Peft(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

    trained_inner = _CausalLM(hidden)
    _set_trainable(trained_inner)
    with torch.no_grad():
        trained_inner.model.layers[0].adapter.fill_(5.0)

    from cray_megatron.megatron.training_loop import filter_checkpoint

    state_dict = filter_checkpoint(trained_inner, trained_inner.state_dict())

    fresh = NoDistribution(_Peft(_CausalLM(hidden)))

    target = _load_trained_parameters(fresh, state_dict)

    assert target is fresh.model.model
    assert torch.all(fresh.model.model.model.layers[0].adapter == 5.0)


def test_resolver_raises_when_nothing_aligns():
    fresh = NoDistribution(_CausalLM(4))
    bogus = {"totally.unrelated.key": torch.zeros(2)}

    with pytest.raises(RuntimeError, match="could not align"):
        _load_trained_parameters(fresh, bogus)


def test_resolver_raises_on_partial_namespace_mismatch():
    fresh = NoDistribution(_CausalLM(4))
    sd = {"lm_head.weight": torch.zeros(4, 4), "ghost.adapter": torch.zeros(4, 4)}

    with pytest.raises(RuntimeError, match="unexpected keys"):
        _load_trained_parameters(fresh, sd)


def _make_loop(job_dir, wrapper):
    harness = TrainingHarness()
    loop = TrainingLoop(training_harness=harness)
    loop.training_state.model_info = {
        "model": wrapper,
        "distribution_strategy": {"device": "cpu"},
        "model_config": None,
    }
    params = [p for p in wrapper.parameters() if p.requires_grad]
    loop.training_state.optimizer = AdamW(params, lr=1e-3)
    loop.training_state.scheduler = torch.optim.lr_scheduler.LinearLR(
        loop.training_state.optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=10,
    )
    return loop


def test_full_checkpoint_resume_restores_trained_weights(job_config_env):
    job_dir = job_config_env
    hidden = 4

    trained = _CausalLM(hidden)
    _set_trainable(trained)
    with torch.no_grad():
        trained.lm_head.weight.fill_(2.5)
        trained.model.layers[0].adapter.fill_(-1.25)
    wrapper = NoDistribution(trained)

    loop = _make_loop(job_dir, wrapper)
    loop.training_state.current_step = 5
    loop.training_state.epoch = 1
    loop.training_state.data_cursor = 12
    loop.training_state.nan_steps = 0

    (job_dir / "status.json").write_text(
        json.dumps({"status": "TRAINING", "history": [{"step": 5, "loss": 0.1}]})
    )

    loop.checkpoint()

    checkpoints = [f for f in os.listdir(job_dir) if f.startswith("checkpoint_")]
    assert checkpoints == ["checkpoint_5.pt"]

    fresh = _CausalLM(hidden)
    _set_trainable(fresh)
    with torch.no_grad():
        fresh.lm_head.weight.zero_()
        fresh.model.layers[0].adapter.zero_()
    fresh_wrapper = NoDistribution(fresh)

    loop.training_state.model_info["model"] = fresh_wrapper
    loop.training_state.optimizer = AdamW(
        [p for p in fresh_wrapper.parameters() if p.requires_grad], lr=1e-3
    )
    loop.training_state.scheduler = torch.optim.lr_scheduler.LinearLR(
        loop.training_state.optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=10,
    )

    loop.resume_from_checkpoint()

    assert torch.allclose(
        fresh.lm_head.weight, torch.full((hidden, hidden), 2.5)
    ), "lm_head weights were not restored on resume"
    assert torch.allclose(
        fresh.model.layers[0].adapter, torch.full((hidden, hidden), -1.25)
    ), "adapter weights were not restored on resume"

    assert loop.training_state.current_step == 6
    assert loop.training_state.epoch == 1
    assert loop.training_state.data_cursor == 12
