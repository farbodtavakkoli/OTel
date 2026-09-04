"""
Unit tests for training_loop.build_adapter_metadata.

Contract under test: vllm/docs/training/adapter_format.md §Metadata
fields. A LoRA job must embed `lora_alpha` into the saved `.pt`, or the
inference-side loader falls back to `2 * rank` and silently mis-scales
the delta.
"""

from unittest.mock import patch

import pytest

# Module-level import pulls in the collectives layer, which is now
# cray_infra.training.distributed (torch.distributed) rather than the removed
# gpu_aware_mpi C++ extension. Importing it is side-effect free — the process
# group is only created on first use — so this is a plain import guard.
pytest.importorskip("cray_infra.training.distributed")

from cray_megatron.megatron.training_loop import build_adapter_metadata


def _with_job_config(cfg):
    return patch(
        "cray_megatron.megatron.training_loop.get_job_config",
        return_value=cfg,
    )


def test_lora_adapter_embeds_lora_alpha():
    cfg = {
        "adapter_type": "lora",
        "lora_config": {"r": 32, "lora_alpha": 16, "lora_dropout": 0.1},
    }
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {"lora_alpha": 16}


def test_lora_adapter_coerces_lora_alpha_to_int():
    # Pydantic round-trips config through dicts; make sure strings from
    # user-supplied YAML don't end up as strings in the saved metadata.
    cfg = {
        "adapter_type": "lora",
        "lora_config": {"r": 32, "lora_alpha": "16"},
    }
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta["lora_alpha"] == 16
    assert isinstance(meta["lora_alpha"], int)


def test_lora_adapter_passes_through_use_rslora():
    cfg = {
        "adapter_type": "lora",
        "lora_config": {
            "r": 32,
            "lora_alpha": 32,
            "use_rslora": True,
        },
    }
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {"lora_alpha": 32, "use_rslora": True}


def test_tokenformer_adapter_emits_empty_metadata():
    cfg = {
        "adapter_type": "tokenformer",
        "lora_config": {"r": 32, "lora_alpha": 32},
    }
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {}


def test_none_adapter_emits_empty_metadata():
    cfg = {"adapter_type": "none"}
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {}


def test_lora_with_missing_alpha_falls_back_to_empty():
    # Defensive: if the user omits lora_alpha we deliberately don't guess
    # — the loader's own default (2 * rank) then applies, which is at
    # least deterministic and documented.
    cfg = {
        "adapter_type": "lora",
        "lora_config": {"r": 32},
    }
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {}


def test_lora_with_no_lora_config_key_is_safe():
    cfg = {"adapter_type": "lora"}
    with _with_job_config(cfg):
        meta = build_adapter_metadata()
    assert meta == {}
