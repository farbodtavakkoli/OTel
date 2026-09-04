"""
Unit tests for ml/tokenformer/tokenformer_surgeon.py.

Contract under test: docs/adapters.md §2 — TokenformerAdapter is zero at
init (tokenformer_p == 0), so wrapping the base model does not perturb
outputs; the surgeon wraps only `*.mlp` modules (and skips vision/audio
towers per the non-language-path guard in §2.3).
"""

import pytest
import torch
from torch import nn

from tokenformer.tokenformer_surgeon import (
    TokenformerAdapter,
    TokenformerSurgeon,
)


# Keep tokenformer_r / num_heads at their defaults (32, 4) by pointing
# SCALARLM_CONFIG_PATH at an empty YAML — the surgeon reads config inside
# TokenformerAdapter.__init__, so defaults must dominate.
@pytest.fixture
def _defaults_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(tmp_path / "none.yaml"))
    yield


class _ModelWithConfig(nn.Module):
    """Minimal nn.Module exposing `config.hidden_size` — the attribute the
    surgeon reads when choosing the adapter dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size_value = hidden_size

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.hidden_size = hidden_size


def _build_model_with_mlp(hidden_size: int) -> _ModelWithConfig:
    model = _ModelWithConfig(hidden_size)

    # Attach layers.{0,1} each with .mlp and .attention submodules.
    layers = nn.ModuleList()
    for _ in range(2):
        block = nn.Module()
        block.mlp = nn.Linear(hidden_size, hidden_size)
        block.attention = nn.Linear(hidden_size, hidden_size)
        layers.append(block)
    model.layers = layers
    model.lm_head = nn.Linear(hidden_size, hidden_size)
    return model


# ---- TokenformerAdapter (unit-level) --------------------------------------


def test_adapter_tokenformer_p_is_zero_at_init(_defaults_only):
    hidden_size = 16
    layer = nn.Linear(hidden_size, hidden_size)

    adapter = TokenformerAdapter(layer, hidden_size, device=torch.device("cpu"))

    assert torch.all(adapter.tokenformer_p == 0), (
        "tokenformer_p must initialize to zero so the adapter branch "
        "contributes nothing until training starts."
    )


def test_adapter_k_and_v_are_non_trivially_initialized(_defaults_only):
    hidden_size = 16
    layer = nn.Linear(hidden_size, hidden_size)

    adapter = TokenformerAdapter(layer, hidden_size, device=torch.device("cpu"))

    assert not torch.all(adapter.tokenformer_k == 0)
    assert not torch.all(adapter.tokenformer_v == 0)


def test_adapter_forward_equals_base_when_p_is_zero(_defaults_only):
    # When tokenformer_p is zero, the entire tokenformer branch is zero
    # regardless of k / v / query — so adapter(x) must equal layer(x).
    hidden_size = 16
    layer = nn.Linear(hidden_size, hidden_size)

    adapter = TokenformerAdapter(layer, hidden_size, device=torch.device("cpu"))

    x = torch.randn(4, hidden_size)

    with torch.no_grad():
        base = layer(x)
        wrapped = adapter(x)

    assert torch.allclose(wrapped, base, atol=1e-6)


def test_adapter_forward_diverges_from_base_once_p_is_nonzero(_defaults_only):
    hidden_size = 16
    layer = nn.Linear(hidden_size, hidden_size)

    adapter = TokenformerAdapter(layer, hidden_size, device=torch.device("cpu"))

    # Post-training simulation: bump tokenformer_p off zero.
    with torch.no_grad():
        adapter.tokenformer_p.add_(0.01)

    x = torch.randn(4, hidden_size)
    with torch.no_grad():
        base = layer(x)
        wrapped = adapter(x)

    assert not torch.allclose(wrapped, base, atol=1e-6)


def test_adapter_parameter_shapes_match_config(_defaults_only):
    hidden_size = 16
    num_heads = 4   # default tokenformer_num_heads
    r = 32          # default tokenformer_r

    adapter = TokenformerAdapter(
        nn.Linear(hidden_size, hidden_size),
        hidden_size,
        device=torch.device("cpu"),
    )

    assert tuple(adapter.tokenformer_k.shape) == (num_heads, hidden_size)
    assert tuple(adapter.tokenformer_v.shape) == (num_heads, hidden_size * r)
    assert tuple(adapter.tokenformer_p.shape) == (r, hidden_size)


# ---- TokenformerSurgeon._is_adapter_layer ---------------------------------


def test_surgeon_wraps_only_mlp_suffix(_defaults_only):
    model = _build_model_with_mlp(hidden_size=16)
    surgeon = TokenformerSurgeon(model, torch.device("cpu"))

    assert surgeon._is_adapter_layer("model.layers.0.mlp") is True
    assert surgeon._is_adapter_layer("model.layers.1.mlp") is True
    assert surgeon._is_adapter_layer("model.layers.0.attention") is False
    assert surgeon._is_adapter_layer("model.lm_head") is False


def test_surgeon_skips_non_language_towers(_defaults_only):
    model = _build_model_with_mlp(hidden_size=16)
    surgeon = TokenformerSurgeon(model, torch.device("cpu"))

    # Vision/audio tower MLPs have the right suffix but the wrong hidden_size
    # and aren't trained through the text loss — they must be excluded.
    assert (
        surgeon._is_adapter_layer("model.vision_tower.encoder.layers.0.mlp")
        is False
    )
    assert (
        surgeon._is_adapter_layer("model.audio_tower.layers.2.mlp") is False
    )
    assert (
        surgeon._is_adapter_layer("model.multi_modal_projector.mlp") is False
    )
    assert surgeon._is_adapter_layer("model.embed_vision.mlp") is False


def test_surgeon_mlp_is_substring_matched_on_last_component(_defaults_only):
    # `_is_adapter_layer` checks `"mlp" in parts[-1]`, so "mlp_proj" also
    # matches. Document the current behavior here — if someone tightens the
    # match to `parts[-1] == "mlp"` later, this test flips red.
    model = _build_model_with_mlp(hidden_size=16)
    surgeon = TokenformerSurgeon(model, torch.device("cpu"))

    assert surgeon._is_adapter_layer("model.layers.0.mlp_proj") is True


# ---- Surgery on a whole model --------------------------------------------


def test_surgeon_insert_adapter_modules_replaces_mlp_in_place(_defaults_only):
    hidden_size = 16
    model = _build_model_with_mlp(hidden_size)
    surgeon = TokenformerSurgeon(model, torch.device("cpu"))

    surgeon.insert_adapter_modules()

    # Every `layers[*].mlp` is now a TokenformerAdapter wrapping the original
    # nn.Linear.
    for idx, block in enumerate(model.layers):
        assert isinstance(block.mlp, TokenformerAdapter), (
            f"layers[{idx}].mlp was not wrapped"
        )
        assert isinstance(block.mlp.layer, nn.Linear)
        # Attention submodule is untouched.
        assert isinstance(block.attention, nn.Linear)
        assert not isinstance(block.attention, TokenformerAdapter)

    # lm_head is not wrapped.
    assert not isinstance(model.lm_head, TokenformerAdapter)


def test_surgery_preserves_outputs_at_init(_defaults_only):
    """End-to-end zero-perturbation check: the wrapped model must produce
    bitwise-close outputs to the unwrapped model, because tokenformer_p is
    zero at init."""
    hidden_size = 16
    model = _build_model_with_mlp(hidden_size)

    x = torch.randn(2, hidden_size)

    # Capture the original layers[0].mlp and its output for comparison.
    original_mlp = model.layers[0].mlp
    with torch.no_grad():
        original_out = original_mlp(x)

    TokenformerSurgeon(model, torch.device("cpu")).insert_adapter_modules()

    with torch.no_grad():
        wrapped_out = model.layers[0].mlp(x)

    assert torch.allclose(wrapped_out, original_out, atol=1e-6)
