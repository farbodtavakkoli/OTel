#!/usr/bin/env python3
"""In-image two-pass fidelity test for the fine-tune sweep preflight.

Runs inside the cray image (needs torch + the vLLM fork); a no-op on the host.
For a decoder-only model whose vLLM tree is ``model.layers.*``
(``Qwen/Qwen2.5-0.5B``), the trainer's synthesized LoRA keys overlap the vLLM
module set ONLY AFTER BOTH passes of the fork's serve-time normalization:

  pass 1  ``normalize_lora_key``             -- statically OVER-maps
                                               ``model.layers.*`` to
                                               ``language_model.model.layers.*``
  pass 2  ``_renormalize_lora_sd_for_model`` -- detects the live ``model.layers.``
                                               prefix and corrects pass 1

A one-pass (or HF-meta) check would mispredict this model as a silent no-op and
wrongly skip it with ``PRECHECK_NO_OP``. This is the regression that justifies
running the preflight through the *real* ``_renormalize_lora_sd_for_model`` (see
docs/superpowers/specs/2026-06-11-finetune-sweep-dry-test-design.md and
ADR 0003's 2026-06-18 amendment).

The torch-free seams (key synthesis, overlap, JSON parsing, classify_result) are
covered on the host in test/unit/test_finetune_sweep_preflight.py; this file is
the in-image complement and is guarded with ``importorskip`` so it skips cleanly
where torch/vLLM are unavailable.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEP_DIR = REPO_ROOT / "test" / "finetune_sweep"

# Make the fork importable the same way the sibling in-image tests do; in the
# cray image the PYTHONPATH already includes it, so this is host-robustness only.
_VLLM_DIR = str(REPO_ROOT / "vllm")
if _VLLM_DIR not in sys.path:
    sys.path.insert(0, _VLLM_DIR)

# Skip the whole module off-image (no torch / no fork) rather than erroring.
pytest.importorskip("torch")
pytest.importorskip("vllm.tokenformer.adapter_format")

# A decoder-only model whose vLLM tree is `model.layers.*` — the exact case a
# one-pass check mispredicts. Already in the sweep manifest, so it is cached.
FIDELITY_MODEL = "Qwen/Qwen2.5-0.5B"


def _load_preflight():
    """Load the host-side preflight module (torch-free) to reach its REAL
    in-container script body, `_INTROSPECT_SCRIPT`."""
    spec = importlib.util.spec_from_file_location(
        "preflight", _SWEEP_DIR / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_introspection(model_id: str) -> dict:
    """Execute the preflight's ACTUAL in-container script body (not a copy) and
    return its result dict, so this test can never drift from what the preflight
    runs at sweep time. `__name__` is set to something other than `__main__` so
    the script's CLI block (which prints JSON) does not fire."""
    pf = _load_preflight()
    ns: dict = {"__name__": "preflight_introspect_test"}
    exec(pf._INTROSPECT_SCRIPT, ns)  # noqa: S102 -- running the real script body is the point
    return ns["run_introspection"](model_id)


def test_pass1_alone_overmaps_decoder_keys():
    """Pass 1 statically over-maps `model.layers.*` -> `language_model.model.
    layers.*`. Against Qwen2.5's real `model.layers.*` tree that lands nowhere —
    so a one-pass overlap check would predict a (false) no-op. This documents
    *why* pass 2 is required; the next test shows pass 2 fixing it."""
    from vllm.tokenformer.adapter_format import normalize_lora_key

    key = "model.layers.0.self_attn.q_proj.lora_A.default.weight"
    normalized = normalize_lora_key(key)
    assert normalized.startswith("language_model.model.layers."), (
        f"expected pass 1 to over-map the decoder key, got {normalized!r}")


def test_two_pass_normalization_overlaps_decoder_tree():
    """The full two-pass normalization (the real script body, against a meta
    device build of Qwen2.5's vLLM tree) lands the synthesized keys back on
    `model.layers.*` -> predicted_ok with non-zero overlap. A one-pass / HF-meta
    check would mispredict this model as a silent no-op."""
    result = _run_introspection(FIDELITY_MODEL)

    assert not result.get("error"), \
        f"meta-tree introspection failed: {result.get('error')}"
    assert result["predicted_ok"] is True, \
        f"two-pass overlap was zero — pass 2 regressed? result={result}"
    assert result["n_overlap"] > 0
    # Sanity: the synthesized target set is non-trivial (q/k/v/o + gate/up/down,
    # collapsed by q/k/v and gate/up fusion in the vLLM tree).
    assert result["n_total"] > 0
