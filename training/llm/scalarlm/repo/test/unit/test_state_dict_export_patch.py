"""Unit tests for the ScalarLM state_dict export patch (gemma4 + llama).

The patch is injected into the vLLM fork by
`scripts/vllm_patches/apply_patches.py`; its source lives there as a
string constant, so these tests exec it against fakes and need no fork
checkout.

What it guards: Gemma4 alternates head_dim per layer (32 on some, 64 on
others), so the split sizes for `qkv_proj` differ per layer. The shared
`unpack_packed_modules_state_dict` helper derives a single split from the
model config, which raised

    split_with_sizes expects split_sizes to sum exactly to 1024,
    but got split_sizes=[256, 128, 128]

on a real tiny-random/gemma-4-dense run. The geometry below is taken from
that model.
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import STATE_DICT_EXPORT_SRC  # noqa: E402


class _Tensor:
    """Stand-in that records how it was sliced. Only shape/slicing matter."""

    def __init__(self, rows, tag):
        self.shape = (rows,)
        self.tag = tag

    def __getitem__(self, item):
        return ("slice", self.tag, item.start, item.stop)


def _layer(q_size, kv_size):
    return types.SimpleNamespace(
        self_attn=types.SimpleNamespace(q_size=q_size, kv_size=kv_size)
    )


# tiny-random/gemma-4-dense: 8 query heads, 4 kv heads, head_dim
# alternating 32 / 64 -> packed qkv rows alternating 512 / 1024.
LAYERS = [_layer(256, 128), _layer(512, 256), _layer(256, 128), _layer(512, 256)]

PREFIX = "language_model."


def _build(state_dict, layers, export_enabled=True):
    """exec the injected method inside a real class so super() resolves."""
    method = next(
        node
        for node in ast.walk(ast.parse(STATE_DICT_EXPORT_SRC.strip()))
        if isinstance(node, ast.FunctionDef) and node.name == "state_dict"
    )
    harness = (
        "class Base:\n"
        "    def state_dict(self, *, destination=None, prefix='', keep_vars=False):\n"
        "        return dict(SOURCE)\n"
        "\n"
        "class Model(Base):\n"
        "    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],\n"
        "                              'gate_up_proj': ['gate_proj', 'up_proj']}\n"
        "    config = None\n"
        "    def __init__(self, layers):\n"
        "        self.model = types.SimpleNamespace(layers=layers)\n"
        + textwrap.indent(ast.unparse(method), "    ")
    )
    captured = {}

    def fake_unpack(state_dict, *, prefix, packed_modules_mapping, config):
        captured["mapping"] = packed_modules_mapping
        return state_dict

    namespace = {
        "SOURCE": state_dict,
        "types": types,
        "scalarlm_state_dict_export_enabled": lambda: export_enabled,
        "unpack_packed_modules_state_dict": fake_unpack,
    }
    exec(harness, namespace)
    return namespace["Model"](layers), captured


@pytest.fixture
def packed_state_dict():
    return {
        f"{PREFIX}model.layers.{i}.self_attn.qkv_proj.weight": _Tensor(
            layer.self_attn.q_size + 2 * layer.self_attn.kv_size, i
        )
        for i, layer in enumerate(LAYERS)
    }


def test_per_layer_geometry_is_read_from_the_module(packed_state_dict):
    """Each layer splits by its own q_size/kv_size, not a shared config
    value -- layers 1 and 3 are twice the size of 0 and 2."""
    model, _ = _build(packed_state_dict, LAYERS)

    out = model.state_dict(None, PREFIX, False)

    for i, layer in enumerate(LAYERS):
        base = f"{PREFIX}model.layers.{i}.self_attn"
        q, kv = layer.self_attn.q_size, layer.self_attn.kv_size
        assert f"{base}.qkv_proj.weight" not in out
        assert out[f"{base}.q_proj.weight"] == ("slice", i, None, q)
        assert out[f"{base}.k_proj.weight"] == ("slice", i, q, q + kv)
        assert out[f"{base}.v_proj.weight"] == ("slice", i, q + kv, None)


def test_qkv_is_withheld_from_the_shared_helper(packed_state_dict):
    """Passing qkv_proj through would re-trigger the config-derived split
    that crashed; gate_up_proj must still be delegated."""
    model, captured = _build(packed_state_dict, LAYERS)

    model.state_dict(None, PREFIX, False)

    assert "qkv_proj" not in captured["mapping"]
    assert "gate_up_proj" in captured["mapping"]


def test_unrecognized_geometry_is_left_packed():
    """Losing a tensor for adapter matching is recoverable and logged;
    splitting it wrongly corrupts weights silently."""
    state_dict = {f"{PREFIX}model.layers.0.self_attn.qkv_proj.weight": _Tensor(999, 0)}
    model, _ = _build(state_dict, [_layer(256, 128)])

    out = model.state_dict(None, PREFIX, False)

    assert f"{PREFIX}model.layers.0.self_attn.qkv_proj.weight" in out
    assert not any("q_proj" in key for key in out)


def test_export_disabled_returns_state_dict_untouched(packed_state_dict):
    """Vanilla vLLM flows (sharded saves, dummy init) must be unaffected."""
    model, _ = _build(packed_state_dict, LAYERS, export_enabled=False)

    out = model.state_dict(None, PREFIX, False)

    assert out == packed_state_dict


def test_layers_without_attention_are_skipped(packed_state_dict):
    """PPMissingLayer placeholders have no self_attn."""
    layers = list(LAYERS)
    layers[1] = types.SimpleNamespace()
    model, _ = _build(packed_state_dict, layers)

    out = model.state_dict(None, PREFIX, False)

    assert f"{PREFIX}model.layers.1.self_attn.qkv_proj.weight" in out
    assert f"{PREFIX}model.layers.0.self_attn.q_proj.weight" in out


def test_non_qkv_keys_pass_through(packed_state_dict):
    packed_state_dict[f"{PREFIX}model.layers.0.self_attn.o_proj.weight"] = _Tensor(8, "o")
    model, _ = _build(packed_state_dict, LAYERS)

    out = model.state_dict(None, PREFIX, False)

    assert out[f"{PREFIX}model.layers.0.self_attn.o_proj.weight"].tag == "o"
