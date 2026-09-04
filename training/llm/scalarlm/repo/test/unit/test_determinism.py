"""
Unit tests for cray_megatron.megatron.determinism.apply_seed — the training-RNG
seed that makes LoRA init + canvas corruption + SC mask reproducible, so a
DiffusionGemma config can be seed-swept for a clean-basin draw instead of relying
on luck. Torch-only; no training harness.
"""

import torch

from cray_megatron.megatron.determinism import apply_seed
from cray_megatron.megatron.diffusion_corruption import corrupt_canvas


def test_apply_seed_makes_global_rng_corruption_reproducible():
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()

    # Two runs that each seed then corrupt via the global RNG (generator=None, as
    # the training loop does) must be byte-identical.
    apply_seed(123)
    a = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.5)
    apply_seed(123)
    b = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.5)
    assert torch.equal(a, b)


def test_different_seeds_diverge():
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()
    apply_seed(1)
    a = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.5)
    apply_seed(2)
    b = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.5)
    assert not torch.equal(a, b)


def test_apply_seed_none_is_noop():
    # None returns False and must not disturb the RNG stream: a draw taken after
    # apply_seed(None) equals the draw taken without calling it, given the same
    # prior seed.
    assert apply_seed(None) is False

    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    assert apply_seed(None) is False
    got = torch.rand(4)
    assert torch.equal(expected, got)


def test_apply_seed_returns_true_when_applied():
    assert apply_seed(42) is True
