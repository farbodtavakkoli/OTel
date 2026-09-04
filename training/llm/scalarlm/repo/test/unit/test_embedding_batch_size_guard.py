"""Embedding training must refuse batch_size < 2 instead of silently no-op'ing.

CoSENT is a pairwise RANKING loss: it compares pairs against each other within a
batch. With a batch of one there is nothing to rank, so the loss is exactly 0.0
at every step, no gradient flows, and the job still reports COMPLETED having
learned nothing.

Measured before the guard:
    batch_size 1 -> COMPLETED, 30 steps, loss 0.0 at every step
    batch_size 8 -> COMPLETED, loss 4.02, 10.66, 5.70, 0.45, ... 0.00097

`batch_size` defaults to 1 in the CLI, so the DEFAULT embedding job was the
broken one. A silent false success is worse than a hard failure.
"""

import pytest

from cray_megatron.megatron.dataset import load_embedding_dataset as led


def _patch_config(monkeypatch, **overrides):
    config = {"batch_size": 1, "max_token_block_size": 128}
    config.update(overrides)
    monkeypatch.setattr(led, "get_job_config", lambda: config)
    return config


def test_batch_size_one_is_rejected(monkeypatch):
    _patch_config(monkeypatch, batch_size=1)

    with pytest.raises(ValueError) as excinfo:
        led._reject_degenerate_batch_size()

    message = str(excinfo.value)
    assert "batch_size >= 2" in message
    assert "pairwise" in message.lower()


def test_batch_size_zero_is_rejected(monkeypatch):
    _patch_config(monkeypatch, batch_size=0)

    with pytest.raises(ValueError):
        led._reject_degenerate_batch_size()


@pytest.mark.parametrize("batch_size", [2, 8, 32])
def test_workable_batch_sizes_are_accepted(monkeypatch, batch_size):
    _patch_config(monkeypatch, batch_size=batch_size)

    # Must not raise.
    led._reject_degenerate_batch_size()


def test_guard_runs_before_any_data_is_touched(monkeypatch):
    """The guard must fire at dataset-build time, not after N steps of zero loss."""
    _patch_config(monkeypatch, batch_size=1)

    def _explode(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("dataset generator ran despite a degenerate batch_size")

    monkeypatch.setattr(led, "make_dataset_generator", _explode)

    with pytest.raises(ValueError):
        led.load_embedding_dataset(model=None, tokenizer=None, epoch=0)
