"""Wiring test: config flag -> create_lora_model NaRA branch -> checkpoint keys ->
the training-step noise-level hook. Exercises the real integration seams on a tiny
model (the actual DiffusionGemma-26B needs a GPU). See ADR 0012.

Run:
    PYTHONPATH=ml:infra uv run --with pytest --with torch --with pydantic \
        python -m pytest test/unit/test_nara_wiring.py -v
"""

import torch
import torch.nn as nn
import pytest

from adapters.nara_prototype import NaRALinear, NaRAContext, find_nara_context


class TinyModel(nn.Module):
    """Two 'decoder' linears with the leaf names resolve_target_modules would emit."""

    def __init__(self, dim=16, vocab=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        h = self.o_proj(torch.tanh(self.q_proj(h)))
        return self.lm_head(h)


class _Backbone(nn.Module):
    """Inner backbone (the module ``unwrap_model`` checkpoints via ``self.model``)."""

    def __init__(self, dim=16, vocab=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        return self.o_proj(torch.tanh(self.q_proj(h)))


class WrappedModel(nn.Module):
    """Reproduces the HF ``…ForXXX`` outer/inner split that hid the checkpoint bug:
    the targeted linears live in ``self.model`` (the backbone), and ``unwrap_model``
    saves ``self.model.state_dict()`` — so a context registered on the OUTER wrapper
    is a sibling of the saved sub-tree and gets dropped. Mirrors DiffusionGemma."""

    def __init__(self, dim=16, vocab=32):
        super().__init__()
        self.model = _Backbone(dim, vocab)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def _patch_job_config(monkeypatch, nara_block):
    """Point create_lora_model.get_job_config at a fixed config dict."""
    import adapters.create_lora_model as clm

    cfg = {
        "lora_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                        "target_modules": ["q_proj", "o_proj"]},
        "diffusion": {"nara": nara_block},
    }
    monkeypatch.setattr(clm, "get_job_config", lambda: cfg)
    return cfg


def test_flag_off_uses_plain_lora(monkeypatch):
    """nara.enabled=False (or absent) must NOT take the NaRA path — falls through to
    PEFT LoRA. We only assert the branch is skipped (no NaRALinear injected)."""
    from adapters.create_lora_model import _nara_config

    assert _nara_config({"diffusion": {"nara": {"enabled": False}}}) is None
    assert _nara_config({"diffusion": {}}) is None
    assert _nara_config({}) is None
    assert _nara_config({"diffusion": {"nara": {"enabled": True}}}) == {"enabled": True}


def test_nara_branch_injects_and_marks_trainable(monkeypatch):
    from adapters.create_lora_model import create_lora_model

    _patch_job_config(monkeypatch, {"enabled": True, "c_scale": 0.1})
    model = create_lora_model(TinyModel(), device="cpu", train_lm_head=False)

    # Both targeted linears became NaRALinear; lm_head did not.
    assert isinstance(model.q_proj, NaRALinear)
    assert isinstance(model.o_proj, NaRALinear)
    assert not isinstance(model.lm_head, NaRALinear)

    # One shared context is reachable and registered exactly once.
    ctx = find_nara_context(model)
    assert isinstance(ctx, NaRAContext)
    n_ctx = sum(1 for m in model.modules() if isinstance(m, NaRAContext))
    assert n_ctx == 1, "context must be registered once, not per-layer"

    # Base weights frozen; adapter + mapper trainable.
    assert not model.q_proj.base.weight.requires_grad
    assert model.q_proj.lora_A.requires_grad and model.q_proj.lora_B.requires_grad
    assert any(p.requires_grad for p in ctx.mapper.parameters())


def test_checkpoint_keys_include_mapper_no_duplicates(monkeypatch):
    """filter_checkpoint saves requires_grad params. The NaRA branch must contribute
    lora_A/lora_B AND the shared mapper — the mapper exactly once (weakref, not a
    per-layer submodule)."""
    from adapters.create_lora_model import create_lora_model, filter_checkpoint

    _patch_job_config(monkeypatch, {"enabled": True})
    model = create_lora_model(TinyModel(), device="cpu", train_lm_head=False)

    saved = filter_checkpoint(model, model.state_dict())
    lora_keys = [k for k in saved if "lora_" in k]
    mapper_keys = [k for k in saved if "nara_context.mapper" in k]

    assert any("q_proj.lora_A" in k for k in lora_keys)
    assert any("o_proj.lora_B" in k for k in lora_keys)
    assert mapper_keys, "shared hypernetwork weights must be checkpointed"
    # Mapper registered once => no 'q_proj.context...'/'o_proj.context...' duplicates.
    assert not [k for k in saved if ".context." in k]


def test_checkpoint_survives_outer_wrapper(monkeypatch):
    """Regression: with a HF-style outer/inner wrapper, the REAL save path
    (model.unwrap_model() -> filter_checkpoint(self.model, ...)) must still carry the
    shared mapper. Before the fix the context sat on the outer wrapper, outside the
    checkpointed backbone, and the mapper was silently dropped (plain-LoRA .pt)."""
    from adapters.create_lora_model import create_lora_model

    _patch_job_config(monkeypatch, {"enabled": True})
    model = create_lora_model(WrappedModel(), device="cpu", train_lm_head=False)

    saved = model.unwrap_model()  # the exact path the training loop checkpoints with
    mapper_keys = [k for k in saved if "nara_context.mapper" in k]
    lora_keys = [k for k in saved if "lora_" in k]

    assert mapper_keys, f"mapper dropped from checkpoint; got keys={sorted(saved)[:8]}"
    assert any("q_proj.lora_A" in k for k in lora_keys)
    assert any("o_proj.lora_B" in k for k in lora_keys)
    # Exactly one shared mapper, no per-layer duplication.
    assert not [k for k in saved if ".context." in k]
    # Keys are backbone-relative (no leaked outer 'model.' prefix on the mapper).
    assert all(k.startswith("nara_context.mapper") for k in mapper_keys)


def test_training_step_hook_sets_noise_level(monkeypatch):
    """Simulate _diffusion_training_step_accumulate's hook: corrupt -> set_noise_level
    -> forward. Assert the adapter's cached Ceff reflects the corruption level and that
    a backward populates grads on lora + mapper."""
    from adapters.create_lora_model import create_lora_model
    from cray_megatron.megatron.diffusion_corruption import corrupt_canvas

    _patch_job_config(monkeypatch, {"enabled": True})
    model = create_lora_model(TinyModel(vocab=32), device="cpu", train_lm_head=True)
    ctx = find_nara_context(model)
    # Make the mapper non-trivial so Ceff actually depends on t.
    last = [m for m in ctx.mapper.net if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.weight.copy_(torch.randn_like(last.weight) * 0.3)

    ids = torch.randint(0, 32, (4, 12))
    labels = ids.clone()
    labels[:, -2:] = -100
    g = torch.Generator().manual_seed(7)
    decoder_input_ids, t = corrupt_canvas(ids, labels, vocab_size=32, eps=0.001,
                                          generator=g, return_noise_level=True)
    ctx.set_noise_level(t)                       # the hook under test
    assert ctx.ceff.shape == (4, 8, 8)           # per-example (B, r, r)

    logits = model(decoder_input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 32).float(), labels.reshape(-1), ignore_index=-100
    )
    loss.backward()
    assert model.q_proj.lora_A.grad is not None
    assert any(p.grad is not None for p in ctx.mapper.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
