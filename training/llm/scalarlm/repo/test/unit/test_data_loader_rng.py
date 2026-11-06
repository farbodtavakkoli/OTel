import importlib

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    module = importlib.import_module("cray_megatron.megatron.dataset.data_loader")
    monkeypatch.setattr(module, "load_dataset", lambda model, tokenizer, epoch: [torch.tensor(epoch * 10 + i) for i in range(3)])
    return module


@pytest.mark.parametrize("config,expected_seed", [
    ({"batch_size": 1, "seed": 17, "sampling_seed": 91}, 17),
    ({"batch_size": 1, "seed": 0, "sampling_seed": 91}, 0),
    ({"batch_size": 1, "seed": None, "sampling_seed": 91}, 91),
    ({"batch_size": 1}, 42),
])
def test_loader_seed_uses_training_seed_then_sampling_seed(module, monkeypatch, config, expected_seed):
    monkeypatch.setattr(module, "get_job_config", lambda: config)
    loader = module.DataLoader(None, None, starting_epoch=5)
    assert loader.loader.generator.initial_seed() == expected_seed + 5


def test_iterator_creation_resume_skip_and_epoch_rollover_preserve_cpu_rng(module, monkeypatch):
    monkeypatch.setattr(module, "get_job_config", lambda: {"batch_size": 1, "seed": 42})
    state = torch.get_rng_state().clone()
    loader = module.DataLoader(None, None)
    iterator = iter(loader)
    assert [int(next(iterator)) for _ in range(4)] == [0, 1, 2, 10]
    assert loader.epoch == 1
    assert loader.loader.generator.initial_seed() == 43
    resumed = module.DataLoader(None, None, starting_epoch=1)
    resume_iterator = iter(resumed)
    assert int(next(resume_iterator)) == 10
    assert int(next(resume_iterator)) == int(next(iterator)) == 11
    assert torch.equal(state, torch.get_rng_state())


def test_same_epoch_reconstruction_reproduces_iterator_seed_without_resetting_epoch(module, monkeypatch):
    monkeypatch.setattr(module, "get_job_config", lambda: {"batch_size": 1, "seed": 42})
    initial = iter(module.DataLoader(None, None))
    seed0 = initial.iterator._base_seed
    for _ in range(4):
        next(initial)
    seed1 = initial.iterator._base_seed
    resumed = iter(module.DataLoader(None, None, starting_epoch=1))
    assert seed1 == resumed.iterator._base_seed
    assert seed0 != seed1
    assert resumed.epoch == 1
