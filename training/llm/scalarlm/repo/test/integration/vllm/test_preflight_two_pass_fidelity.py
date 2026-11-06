#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEP_DIR = REPO_ROOT / "test" / "finetune_sweep"

_VLLM_DIR = str(REPO_ROOT / "vllm")
if _VLLM_DIR not in sys.path:
    sys.path.insert(0, _VLLM_DIR)

pytest.importorskip("torch")
pytest.importorskip("vllm.tokenformer.adapter_format")

FIDELITY_MODEL = "Qwen/Qwen2.5-0.5B"


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "preflight", _SWEEP_DIR / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_introspection(model_id: str) -> dict:
    pf = _load_preflight()
    ns: dict = {"__name__": "preflight_introspect_test"}
    exec(pf._INTROSPECT_SCRIPT, ns)
    return ns["run_introspection"](model_id)


def test_pass1_alone_overmaps_decoder_keys():
    from vllm.tokenformer.adapter_format import normalize_lora_key

    key = "model.layers.0.self_attn.q_proj.lora_A.default.weight"
    normalized = normalize_lora_key(key)
    assert normalized.startswith("language_model.model.layers."), (
        f"expected pass 1 to over-map the decoder key, got {normalized!r}")


def test_two_pass_normalization_overlaps_decoder_tree():
    result = _run_introspection(FIDELITY_MODEL)

    assert not result.get("error"), \
        f"meta-tree introspection failed: {result.get('error')}"
    assert result["predicted_ok"] is True, \
        f"two-pass overlap was zero — pass 2 regressed? result={result}"
    assert result["n_overlap"] > 0
    assert result["n_total"] > 0
