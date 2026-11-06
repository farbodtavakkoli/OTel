import torch

from cray_megatron.megatron.determinism import apply_seed
from cray_megatron.megatron.diffusion_corruption import corrupt_canvas


def test_apply_seed_makes_global_rng_corruption_reproducible():
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()

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
    assert apply_seed(None) is False

    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    assert apply_seed(None) is False
    got = torch.rand(4)
    assert torch.equal(expected, got)


def test_apply_seed_returns_true_when_applied():
    assert apply_seed(42) is True
