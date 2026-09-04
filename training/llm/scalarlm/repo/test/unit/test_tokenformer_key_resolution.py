"""Unit tests for the Tokenformer adapter key resolver.

The resolver is injected into the vLLM fork by
`scripts/vllm_patches/apply_patches.py`. Its source lives in that script
as a string constant, so these tests exec it with a stub logger and
exercise the rules directly — no vLLM tree required.

The rule that matters most is #1 (exact match passes through untouched):
that is what keeps every already-trained checkpoint loading exactly as it
did before the patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import TOKENFORMER_KEY_RESOLVER_SRC  # noqa: E402


class _StubLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)


@pytest.fixture
def resolve():
    """The injected helper, with `logger` bound to a stub."""
    namespace = {"logger": _StubLogger()}
    exec(TOKENFORMER_KEY_RESOLVER_SRC, namespace)
    fn = namespace["_scalarlm_resolve_adapter_keys"]
    fn.stub_logger = namespace["logger"]
    return fn


class _Mapper:
    """Stand-in for vLLM's WeightsMapper with a prefix rule."""

    def __init__(self, prefixes):
        self._prefixes = prefixes

    def _map_name(self, key):
        for old, new in self._prefixes.items():
            if key.startswith(old):
                return new + key[len(old):]
        return key


class _Model:
    hf_to_vllm_mapper = None


def _model_with_mapper(mapper):
    class M:
        pass

    M.hf_to_vllm_mapper = mapper
    return M()


def test_exact_matches_pass_through_untouched(resolve):
    """Backward compatibility: a checkpoint whose keys already exist is
    neither renamed nor dropped."""
    tokenformers = {
        "model.layers.0.self_attn.q_proj.weight": "w0",
        "model.layers.0.mlp.tokenformer_p": "w1",
    }
    state_dict = dict.fromkeys(tokenformers, "live")

    out = resolve(_Model(), tokenformers, state_dict)

    assert out == tokenformers
    assert resolve.stub_logger.warnings == []


def test_mapper_rewrites_multimodal_prefix(resolve):
    """HF `model.language_model.*` resolves to vLLM `language_model.model.*`."""
    mapper = _Mapper({"model.language_model.": "language_model.model."})
    tokenformers = {"model.language_model.layers.0.mlp.tokenformer_p": "w"}
    state_dict = {"language_model.model.layers.0.mlp.tokenformer_p": "live"}

    out = resolve(_model_with_mapper(mapper), tokenformers, state_dict)

    assert out == {"language_model.model.layers.0.mlp.tokenformer_p": "w"}
    assert resolve.stub_logger.warnings == []


def test_mapper_declared_on_the_instance_is_used(resolve):
    """Several models (mimo_mtp, ernie_mtp, transformers backend) build
    hf_to_vllm_mapper in __init__ rather than as a class attribute. A
    class-only lookup would drop every key on those models."""
    model = _Model()
    model.hf_to_vllm_mapper = _Mapper({"model.": "language_model.model."})
    tokenformers = {"model.layers.0.mlp.tokenformer_p": "w"}
    state_dict = {"language_model.model.layers.0.mlp.tokenformer_p": "live"}

    out = resolve(model, tokenformers, state_dict)

    assert out == {"language_model.model.layers.0.mlp.tokenformer_p": "w"}


def test_unresolvable_keys_are_dropped_with_a_warning(resolve):
    """Vision-tower weights have no vLLM counterpart. Dropping them is what
    keeps EngineCore alive; the warning is what keeps it from being silent."""
    tokenformers = {
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": "w0",
        "model.layers.0.mlp.tokenformer_p": "w1",
    }
    state_dict = {"model.layers.0.mlp.tokenformer_p": "live"}

    out = resolve(_Model(), tokenformers, state_dict)

    assert out == {"model.layers.0.mlp.tokenformer_p": "w1"}
    assert "1 of 2" in resolve.stub_logger.warnings[0]


def test_mapper_result_still_absent_is_dropped(resolve):
    """A mapper rewrite that doesn't land on a real parameter is not
    trusted — the membership check is the authority, not the mapper."""
    mapper = _Mapper({"model.": "nowhere."})
    tokenformers = {"model.layers.0.self_attn.q_proj.weight": "w"}

    out = resolve(_model_with_mapper(mapper), tokenformers, {})

    assert out == {}
    assert "1 of 1" in resolve.stub_logger.warnings[0]


def test_broken_mapper_does_not_break_serving(resolve):
    """A mapper that raises falls back to the drop path rather than
    propagating out of adapter registration."""

    class Exploding:
        def _map_name(self, key):
            raise RuntimeError("mapper API drift")

    tokenformers = {"a": "w0", "b": "w1"}
    state_dict = {"b": "live"}

    out = resolve(_model_with_mapper(Exploding()), tokenformers, state_dict)

    assert out == {"b": "w1"}


def test_drop_warning_names_the_nearest_live_parameters(resolve):
    """The two naming bugs that each cost a 30-minute rebuild -- a LoRA
    wrapper nesting weights under `.base_layer.`, and vLLM fusing q/k/v
    into `qkv_proj` -- are both visible in this one line."""
    live = {
        "language_model.model.layers.0.self_attn.qkv_proj.base_layer.weight": "w",
        "language_model.model.layers.0.self_attn.o_proj.base_layer.weight": "w",
    }
    tokenformers = {"model.language_model.layers.0.self_attn.q_proj.weight": "w"}

    resolve(_Model(), tokenformers, live)

    diagnostic = " ".join(resolve.stub_logger.warnings)
    assert "qkv_proj.base_layer" in diagnostic


def test_nearest_keys_prefers_the_most_specific_match(resolve):
    """A name that does exist should report its own neighbours, not the
    whole layer."""
    live = {
        "language_model.model.layers.0.self_attn.o_proj.base_layer.weight": "w",
        "language_model.model.layers.0.mlp.tokenformer_p": "w",
    }
    tokenformers = {"model.language_model.layers.0.self_attn.o_proj.weight": "w"}

    resolve(_Model(), tokenformers, live)

    diagnostic = " ".join(resolve.stub_logger.warnings)
    assert "o_proj.base_layer" in diagnostic
    assert "tokenformer_p" not in diagnostic


def test_nearest_keys_handles_no_match_at_all(resolve):
    resolve(_Model(), {"totally.unrelated.key": "w"}, {"nothing": "w"})

    assert "(none)" in " ".join(resolve.stub_logger.warnings)


def test_empty_checkpoint_is_not_a_warning(resolve):
    out = resolve(_Model(), {}, {"x": "live"})

    assert out == {}
    assert resolve.stub_logger.warnings == []
