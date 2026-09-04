"""
Regression tests for checkpoint resume restoring trained weights.

Bug under test (the "loss jumps on checkpoint boundaries" report): the save
path serializes a filtered, trainable-only state_dict relative to an *inner*
module (``filter_checkpoint(model.model, ...)`` for the no-distribution / DDP
path), but ``resume_from_checkpoint`` loaded it back into the *outer*
distribution wrapper with ``strict=False``. The extra ``model.`` prefix on
the wrapper's own keys meant nothing matched, ``strict=False`` swallowed the
mismatch, and every trained weight was silently dropped on resume — so the
model reverted toward its base weights at each restart while the optimizer /
scheduler / RNG restored correctly, producing a loss spike that re-descended.

These tests pin two things:
  1. ``_load_trained_parameters`` resolves the right descendant module across
     wrapper depths (tokenformer = depth 1, LoRA = depth 2) and refuses to
     silently no-op when nothing aligns.
  2. A full ``TrainingLoop.checkpoint()`` -> ``resume_from_checkpoint()``
     round-trip on the NoDistribution path actually restores the trained
     trainable parameters into a freshly-constructed (base-weight) model.
"""

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


# ---------------------------------------------------------------------------
# Fixtures: a job config on disk + a rank-0 stub for @main_rank_only.
# ---------------------------------------------------------------------------


@pytest.fixture
def job_config_env(tmp_path, monkeypatch):
    """Write a minimal JobConfig YAML and point the loader at it.

    Only job_directory / training_data_path / dataset_hash are required by the
    pydantic model; everything else takes its default. steps_per_checkpoint
    and timeout exist so the callbacks construct cleanly.
    """
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
    """Run @main_rank_only bodies inline (rank 0, no real MPI barriers)."""
    import cray_megatron.collectives.main_rank_only as m

    m._in_main_rank_only = False
    monkeypatch.setattr(m, "get_rank", lambda: 0)
    monkeypatch.setattr(m, "barrier", lambda: None)
    yield
    m._in_main_rank_only = False


# ---------------------------------------------------------------------------
# Model doubles mirroring the real wrapper nesting.
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # Stands in for a tokenformer-wrapped mlp: a trainable adapter param
        # alongside a frozen base projection.
        self.base_proj = nn.Linear(hidden, hidden)
        self.adapter = nn.Parameter(torch.zeros(hidden, hidden))


class _Decoder(nn.Module):
    """Mirrors GemmaModel: `.embed_tokens` + `.layers`."""

    def __init__(self, hidden):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, hidden)
        self.layers = nn.ModuleList([_Block(hidden)])


class _CausalLM(nn.Module):
    """Mirrors GemmaForCausalLM: `.model` (decoder) + `.lm_head`."""

    def __init__(self, hidden):
        super().__init__()
        self.model = _Decoder(hidden)
        self.lm_head = nn.Linear(hidden, hidden)


def _set_trainable(causal_lm):
    """Freeze everything, then unfreeze the adapter + lm_head — the same shape
    of 'small trainable subset of a mostly-frozen model' the adapter paths
    produce. Returns the set of trainable parameter names."""
    for p in causal_lm.parameters():
        p.requires_grad = False
    trainable = set()
    for name, p in causal_lm.named_parameters():
        if name.endswith(".adapter") or name.startswith("lm_head"):
            p.requires_grad = True
            trainable.add(name)
    return trainable


# ---------------------------------------------------------------------------
# _load_trained_parameters — unit-level namespace resolution.
# ---------------------------------------------------------------------------


def test_resolver_loads_into_depth_one_wrapper():
    """NoDistribution(CausalLM): keys are relative to wrapper.model."""
    hidden = 4
    trained = _CausalLM(hidden)
    _set_trainable(trained)
    # Give the trainable params recognizable non-zero values.
    with torch.no_grad():
        trained.lm_head.weight.fill_(3.0)
        trained.model.layers[0].adapter.fill_(7.0)

    # Save exactly like TrainingLoop.checkpoint() does for this path.
    from cray_megatron.megatron.training_loop import filter_checkpoint

    state_dict = filter_checkpoint(trained, trained.state_dict())
    # Saved keys are relative to the inner CausalLM (no NoDistribution prefix).
    assert {"lm_head.weight", "model.layers.0.adapter"} <= set(state_dict)
    assert all(not k.startswith("model.model") for k in state_dict)

    fresh = NoDistribution(_CausalLM(hidden))

    target = _load_trained_parameters(fresh, state_dict)

    # Resolver picked the inner CausalLM, not the NoDistribution wrapper.
    assert target is fresh.model
    assert torch.all(fresh.model.lm_head.weight == 3.0)
    assert torch.all(fresh.model.model.layers[0].adapter == 7.0)


def test_resolver_loads_into_depth_two_wrapper():
    """LoRA-style nesting: keys relative to wrapper.model.model."""
    hidden = 4

    class _Peft(nn.Module):  # PeftModel double: `.model` is the CausalLM
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
    """A checkpoint whose keys match no module must fail loudly, not no-op."""
    fresh = NoDistribution(_CausalLM(4))
    bogus = {"totally.unrelated.key": torch.zeros(2)}

    with pytest.raises(RuntimeError, match="could not align"):
        _load_trained_parameters(fresh, bogus)


def test_resolver_raises_on_partial_namespace_mismatch():
    """If the best target still leaves trained keys unplaced, refuse to
    continue rather than load a partial set of weights."""
    fresh = NoDistribution(_CausalLM(4))
    sd = {"lm_head.weight": torch.zeros(4, 4), "ghost.adapter": torch.zeros(4, 4)}

    with pytest.raises(RuntimeError, match="unexpected keys"):
        _load_trained_parameters(fresh, sd)


# ---------------------------------------------------------------------------
# Full round-trip through the real TrainingLoop save + resume methods.
# ---------------------------------------------------------------------------


def _make_loop(job_dir, wrapper):
    """Build a TrainingLoop wired to `wrapper`, with a real optimizer +
    scheduler over the trainable params (so optimizer/scheduler state has
    something to restore too)."""
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
    """End-to-end: train -> checkpoint -> fresh model -> resume must bring the
    trained weights back. This is the direct regression for the reported
    checkpoint-boundary loss jump."""
    job_dir = job_config_env
    hidden = 4

    # --- "trained" model: trainable params bumped to known values ---
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

    # status.json must exist with a history list for resume to read it back.
    (job_dir / "status.json").write_text(
        json.dumps({"status": "TRAINING", "history": [{"step": 5, "loss": 0.1}]})
    )

    loop.checkpoint()

    # The checkpoint actually hit disk.
    checkpoints = [f for f in os.listdir(job_dir) if f.startswith("checkpoint_")]
    assert checkpoints == ["checkpoint_5.pt"]

    # --- fresh model with base (zero-ish, distinct) weights ---
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

    # Weights restored to the trained values (the bug left these at 0).
    assert torch.allclose(
        fresh.lm_head.weight, torch.full((hidden, hidden), 2.5)
    ), "lm_head weights were not restored on resume"
    assert torch.allclose(
        fresh.model.layers[0].adapter, torch.full((hidden, hidden), -1.25)
    ), "adapter weights were not restored on resume"

    # And the bookkeeping resumed at the next step.
    assert loop.training_state.current_step == 6
    assert loop.training_state.epoch == 1
    assert loop.training_state.data_cursor == 12
