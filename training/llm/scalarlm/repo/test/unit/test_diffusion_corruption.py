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
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()
    out = corrupt_canvas(canvas, labels, vocab_size=1000, eps=1.0 - 1e-6, generator=_gen(7))
    changed = int((out != canvas).sum())
    assert changed > 10


def test_protect_prefix_keeps_anchor_clean_at_max_t():
    canvas = torch.arange(2, 22).reshape(1, 20)
    labels = canvas.clone()
    out = corrupt_canvas(
        canvas, labels, vocab_size=1000, eps=1.0 - 1e-6, generator=_gen(7), protect_prefix=1
    )
    assert out[0, 0].item() == canvas[0, 0].item()
    assert int((out[:, 1:] != canvas[:, 1:]).sum()) > 10


def test_protect_prefix_zero_matches_default():
    canvas = torch.tensor([[5, 6, 7, 8, 9]])
    labels = canvas.clone()
    a = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42))
    b = corrupt_canvas(canvas, labels, 50, 0.3, generator=_gen(42), protect_prefix=0)
    assert torch.equal(a, b)


def test_low_t_corrupts_few_positions():
    canvas = torch.arange(2, 102).reshape(1, 100)
    labels = canvas.clone()
    out = corrupt_canvas(canvas, labels, vocab_size=1000, eps=0.001, generator=_gen(3))
    changed = int((out != canvas).sum())
    assert changed < 90
