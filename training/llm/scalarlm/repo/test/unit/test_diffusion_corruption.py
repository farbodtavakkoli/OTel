"""
Unit tests for cray_megatron.megatron.diffusion_corruption.corrupt_canvas.

DiffusionGemma trains by corrupting a clean canvas with uniform-vocabulary noise
(sample t ~ U(eps, 1) per example, replace each supervised position with a random
vocab token w.p. t — no [MASK] token) and learning to denoise it. These tests pin
the invariants: replacements stay in-vocabulary, padding is never corrupted, the
schedule is reproducible under a seeded generator, and high t corrupts heavily.
Torch-only — no training harness or GPU.
"""

import torch

from cray_megatron.megatron.diffusion_corruption import corrupt_canvas


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_corruption_stays_in_vocab():
    canvas = torch.tensor([[5, 6, 7, 0, 0]])
    labels = torch.tensor([[5, 6, 7, -100, -100]])
    vocab = 50
    out = corrupt_canvas(canvas, labels, vocab, eps=0.5, generator=_gen(0))
    assert out.shape == canvas.shape
    assert out.dtype == canvas.dtype
    assert int(out.min()) >= 0
    assert int(out.max()) < vocab


def test_padding_never_corrupted():
    # Even at maximal corruption (t ~= 1), padding positions (labels == -100) are
    # supervised=False and must retain the clean pad token.
    canvas = torch.tensor([[5, 6, 7, 0, 0]])
    labels = torch.tensor([[5, 6, 7, -100, -100]])
    out = corrupt_canvas(canvas, labels, vocab_size=50, eps=1.0 - 1e-9, generator=_gen(1))
    assert out[0, 3].item() == 0
    assert out[0, 4].item() == 0


def test_corruption_reproducible_with_seed():
    canvas = torch.tensor([[5, 6, 7, 8, 9]])
    labels = torch.tensor([[5, 6, 7, 8, 9]])
    a = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42))
    b = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42))
    assert torch.equal(a, b)


def test_high_t_corrupts_most_supervised_positions():
    # t forced near 1 => nearly every supervised position is replaced. With a large
    # vocab, collisions (random == clean) are negligible, so most positions differ.
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()
    out = corrupt_canvas(canvas, labels, vocab_size=1000, eps=1.0 - 1e-6, generator=_gen(7))
    changed = int((out != canvas).sum())
    assert changed > 10


def test_protect_prefix_keeps_anchor_clean_at_max_t():
    # Tier-2 anchor: position 0 is supervised (label != -100) but must stay clean
    # every step. Even at t ~= 1, protect_prefix=1 keeps it equal to the input while
    # the rest of the supervised canvas is heavily corrupted.
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()  # fully supervised, no padding
    out = corrupt_canvas(
        canvas, labels, vocab_size=1000, eps=1.0 - 1e-6, generator=_gen(7), protect_prefix=1
    )
    assert out[0, 0].item() == canvas[0, 0].item()  # anchor untouched
    assert int((out[:, 1:] != canvas[:, 1:]).sum()) > 10  # rest still corrupted


def test_protect_prefix_zero_matches_default():
    canvas = torch.tensor([[5, 6, 7, 8, 9]])
    labels = canvas.clone()
    a = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42))
    b = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42), protect_prefix=0)
    assert torch.equal(a, b)


def test_low_t_corrupts_few_positions():
    # t drawn from [eps, 1) with a tiny eps and a seed that yields a small t => few
    # positions flip. This is a distributional check, not an exact count, so allow
    # a generous bound: at eps=0.001 the expected corruption fraction is low.
    canvas = torch.arange(2, 102).reshape(1, 100)
    labels = canvas.clone()
    out = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.001, generator=_gen(3))
    changed = int((out != canvas).sum())
    # Some corruption is possible but it must not be near-total at this eps/seed.
    assert changed < 90
