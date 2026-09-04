"""Runnable demonstration of the NaRA prototype against our diffusion corruption.

Run:
    PYTHONPATH=ml uv run --with pytest --with torch python -m pytest \
        test/unit/test_nara_prototype.py -v

These tests assert the three properties that make NaRA "correct" as a drop-in for our
DiffusionGemma decoder-only LoRA, using the real ``corrupt_canvas`` as the noise source:

1. At init, NaRA == plain LoRA (the zero-init mapper => Ceff == I residual).
2. C(lambda) actually depends on the noise level once the mapper is non-trivial.
3. The two-stage schedule freezes/unfreezes the mapper as intended, and a few
   optimizer steps on a toy canvas-denoising loss reduce the loss (grad flows).
"""

import torch
import torch.nn as nn

from cray_megatron.megatron.diffusion_corruption import corrupt_canvas
from adapters.nara_prototype import (
    NaRAConfig,
    NaRAContext,
    NaRALinear,
    inject_nara,
)


def _toy_canvas(batch=4, length=16, vocab=64, seed=0):
    """A clean canvas with a few padded (label == -100) tail positions per row."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, length), generator=g)
    labels = ids.clone()
    labels[:, -3:] = -100  # simulate padding
    ids[:, -3:] = 0
    return ids, labels


def test_corrupt_canvas_returns_noise_level():
    """The hook NaRA needs: corrupt_canvas can now hand back the per-example lambda."""
    ids, labels = _toy_canvas()
    g = torch.Generator().manual_seed(1)
    out = corrupt_canvas(ids, labels, vocab_size=64, eps=0.001, generator=g,
                         return_noise_level=True)
    assert isinstance(out, tuple) and len(out) == 2
    decoder_input_ids, t = out
    assert decoder_input_ids.shape == ids.shape
    assert t.shape == (ids.shape[0], 1)
    assert torch.all(t >= 0.001) and torch.all(t < 1.0)  # t ~ U(eps, 1)

    # Backward compatible: default still returns just the tensor.
    g2 = torch.Generator().manual_seed(1)
    only = corrupt_canvas(ids, labels, vocab_size=64, eps=0.001, generator=g2)
    assert torch.is_tensor(only)


def test_nara_equals_plain_lora_at_init():
    """Zero-init mapper => Ceff == I => delta is exactly the plain-LoRA delta."""
    torch.manual_seed(0)
    cfg = NaRAConfig(r=8, lora_alpha=16, lora_dropout=0.0)
    base = nn.Linear(32, 48, bias=False)
    ctx = NaRAContext(cfg)
    layer = NaRALinear(base, ctx, cfg)
    # Give B a nonzero value so the adapter is actually active (init B is zero).
    with torch.no_grad():
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))

    x = torch.randn(4, 10, 32)
    t = torch.rand(4, 1) * 0.999 + 0.001
    ctx.set_noise_level(t)  # mapper is zero-init => C == 0 => Ceff == I

    got = layer(x)
    # Reference plain-LoRA delta: scaling * (x @ A^T) @ B^T
    ref = base(x) + cfg.lora_alpha / cfg.r * (x @ layer.lora_A.T) @ layer.lora_B.T
    assert torch.allclose(got, ref, atol=1e-5), "NaRA must reduce to plain LoRA at init"


def test_ceff_depends_on_noise_level():
    """Once the mapper's last layer is non-zero, Ceff varies with lambda."""
    torch.manual_seed(0)
    cfg = NaRAConfig(r=8)
    ctx = NaRAContext(cfg)
    # Perturb the zero-init last layer so C(lambda) != 0.
    last = [m for m in ctx.mapper.net if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn_like(last.weight) * 0.5)

    ctx.set_noise_level(torch.tensor([[0.1]]))
    ceff_low = ctx.ceff.clone()
    ctx.set_noise_level(torch.tensor([[0.9]]))
    ceff_high = ctx.ceff.clone()

    assert not torch.allclose(ceff_low, ceff_high), "Ceff should move with the noise level"
    # Per-example batching: a (B,1) lambda yields a (B, r, r) stack.
    ctx.set_noise_level(torch.rand(5, 1))
    assert ctx.ceff.shape == (5, cfg.r, cfg.r)


def test_two_stage_grad_flow():
    """Stage 1 freezes the mapper (A/B only); stage 2 trains it too."""
    cfg = NaRAConfig(r=8)
    ctx = NaRAContext(cfg)
    ctx.set_training_stage(1)
    assert all(not p.requires_grad for p in ctx.mapper.parameters())
    ctx.set_training_stage(2)
    assert all(p.requires_grad for p in ctx.mapper.parameters())


def test_toy_denoising_step_reduces_loss():
    """End-to-end: corrupt a canvas, condition NaRA on its lambda, train a toy denoiser,
    and confirm the loss drops. Exercises the exact call sequence prod would use."""
    torch.manual_seed(0)
    vocab, dim, length = 64, 32, 16
    ids, labels = _toy_canvas(batch=8, length=length, vocab=vocab)

    # Toy "denoiser": embed -> frozen linear tower (NaRA-adapted) -> vocab head.
    class ToyDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.proj = nn.Linear(dim, dim)   # will be NaRA-adapted
            self.head = nn.Linear(dim, vocab)

        def forward(self, tokens):
            return self.head(torch.tanh(self.proj(self.emb(tokens))))

    model = ToyDenoiser()
    ctx = inject_nara(model, target_modules=["proj"], config=NaRAConfig(r=8))
    assert isinstance(model.proj, NaRALinear)

    # Freeze everything except the adapter + mapper (LoRA-style), like create_lora_model.
    trainable = []
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    trainable += [p for p in model.parameters() if p.requires_grad]
    trainable += list(ctx.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-2)

    def step():
        g = torch.Generator().manual_seed(123)
        decoder_input_ids, t = corrupt_canvas(
            ids, labels, vocab_size=vocab, eps=0.001, generator=g, return_noise_level=True
        )
        ctx.set_noise_level(t)                 # <-- the NaRA hook
        logits = model(decoder_input_ids)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab).float(), labels.reshape(-1), ignore_index=-100
        )

    first = step().item()
    for _ in range(50):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    last = step().item()

    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    test_corrupt_canvas_returns_noise_level()
    test_nara_equals_plain_lora_at_init()
    test_ceff_depends_on_noise_level()
    test_two_stage_grad_flow()
    test_toy_denoising_step_reduces_loss()
    print("all NaRA prototype checks passed")
