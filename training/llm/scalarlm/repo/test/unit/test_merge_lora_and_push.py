"""
Unit tests for the pure helpers in ml/adapters/merge_lora_and_push.py.

The CLI flow itself touches transformers + peft + huggingface_hub and
is exercised end-to-end by hand on a real job; here we cover the
checkpoint-discovery, key-classification, rank inference, and lora-
config-resolution layers so the wrong alpha or shape can't slip past.
"""

from pathlib import Path

import pytest
import torch

from adapters.merge_lora_and_push import (
    PEFT_OUTER_PREFIX,
    _prefix_for_peft_load,
    classify_state_dict,
    find_latest_checkpoint,
    infer_lora_rank,
    load_job_config,
    resolve_lora_config_args,
    strip_default_adapter_segment,
)


# ---- find_latest_checkpoint ---------------------------------------------


def test_find_latest_picks_highest_step(tmp_path):
    for step in (5, 100, 27):
        (tmp_path / f"checkpoint_{step}.pt").write_bytes(b"")
    assert find_latest_checkpoint(tmp_path).name == "checkpoint_100.pt"


def test_find_latest_ignores_non_matching_files(tmp_path):
    (tmp_path / "checkpoint_3.pt").write_bytes(b"")
    (tmp_path / "checkpoint_3.txt").write_bytes(b"")
    (tmp_path / "checkpoint_x.pt").write_bytes(b"")
    (tmp_path / "config.yaml").write_text("")
    assert find_latest_checkpoint(tmp_path).name == "checkpoint_3.pt"


def test_find_latest_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_checkpoint(tmp_path)


# ---- load_job_config ----------------------------------------------------


def test_load_job_config_reads_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "llm_name: foo/bar\nlora_config:\n  r: 8\n  lora_alpha: 16\n"
    )
    cfg = load_job_config(tmp_path)
    assert cfg["llm_name"] == "foo/bar"
    assert cfg["lora_config"] == {"r": 8, "lora_alpha": 16}


def test_load_job_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_job_config(tmp_path)


def test_load_job_config_empty_file_returns_dict(tmp_path):
    (tmp_path / "config.yaml").write_text("")
    assert load_job_config(tmp_path) == {}


# ---- classify_state_dict ------------------------------------------------


def test_classify_splits_three_buckets():
    sd = {
        # LoRA — `.lora_A.` segment
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight":
            torch.zeros(8, 16),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight":
            torch.zeros(16, 8),
        # Tokenformer — leaf match
        "model.layers.0.mlp.tokenformer_k": torch.zeros(4, 16),
        "model.layers.0.mlp.tokenformer_v": torch.zeros(4, 16),
        # Base override
        "lm_head.weight": torch.zeros(32_000, 16),
        "model.layers.0.input_layernorm.weight": torch.zeros(16),
    }
    out = classify_state_dict(sd)
    assert len(out["lora"]) == 2
    assert len(out["tokenformer"]) == 2
    assert len(out["base"]) == 2


def test_classify_lora_substring_does_not_steal_base_keys():
    # A param literally called `lora_config` shouldn't be classified as LoRA.
    sd = {"some.module.lora_config": torch.zeros(1)}
    out = classify_state_dict(sd)
    assert out["lora"] == {}
    assert "some.module.lora_config" in out["base"]


def test_classify_picks_up_lora_embedding_keys():
    sd = {
        "embed.lora_embedding_A": torch.zeros(8, 32_000),
        "embed.lora_embedding_B": torch.zeros(4096, 8),
    }
    out = classify_state_dict(sd)
    assert len(out["lora"]) == 2


# ---- infer_lora_rank ----------------------------------------------------


def test_infer_lora_rank_reads_lora_a_leading_dim():
    sd = {
        "x.lora_A.default.weight": torch.zeros(16, 4096),
        "x.lora_B.default.weight": torch.zeros(4096, 16),
    }
    assert infer_lora_rank(sd) == 16


def test_infer_lora_rank_raises_when_no_lora_a():
    with pytest.raises(ValueError):
        infer_lora_rank({"only_b.lora_B.default.weight": torch.zeros(4096, 8)})


# ---- resolve_lora_config_args -------------------------------------------


def _lora_keys(rank=8):
    return {
        "x.lora_A.default.weight": torch.zeros(rank, 4096),
        "x.lora_B.default.weight": torch.zeros(4096, rank),
    }


def test_resolve_uses_metadata_alpha_when_present():
    cfg = resolve_lora_config_args(
        job_config={"lora_config": {"r": 8, "lora_alpha": 4}},
        metadata={"lora_alpha": 16},
        lora_keys=_lora_keys(rank=8),
    )
    assert cfg["lora_alpha"] == 16
    assert cfg["_alpha_source"] == "metadata"


def test_resolve_falls_back_to_job_config_alpha_when_metadata_missing():
    cfg = resolve_lora_config_args(
        job_config={"lora_config": {"r": 8, "lora_alpha": 12}},
        metadata={},
        lora_keys=_lora_keys(rank=8),
    )
    assert cfg["lora_alpha"] == 12
    assert cfg["_alpha_source"] == "job_config"


def test_resolve_defaults_alpha_to_two_times_rank_when_unspecified():
    # Mirrors the vLLM-side adapter loader default.
    cfg = resolve_lora_config_args(
        job_config={},
        metadata={},
        lora_keys=_lora_keys(rank=32),
    )
    assert cfg["r"] == 32
    assert cfg["lora_alpha"] == 64
    assert cfg["_alpha_source"] == "default"


def test_resolve_cli_override_wins():
    cfg = resolve_lora_config_args(
        job_config={"lora_config": {"r": 8, "lora_alpha": 4}},
        metadata={"lora_alpha": 16},
        lora_keys=_lora_keys(rank=8),
        lora_alpha_override=99,
    )
    assert cfg["lora_alpha"] == 99
    assert cfg["_alpha_source"] == "cli"


def test_resolve_rank_comes_from_tensors_not_config():
    # Job config drifted (says r=8) but the actual saved tensors are r=16.
    # Trust the tensors.
    cfg = resolve_lora_config_args(
        job_config={"lora_config": {"r": 8, "lora_alpha": 16}},
        metadata={},
        lora_keys=_lora_keys(rank=16),
    )
    assert cfg["r"] == 16


def test_resolve_passes_use_rslora_through():
    cfg = resolve_lora_config_args(
        job_config={"lora_config": {"r": 8, "lora_alpha": 16, "use_rslora": True}},
        metadata={},
        lora_keys=_lora_keys(rank=8),
    )
    assert cfg["use_rslora"] is True


# ---- strip_default_adapter_segment --------------------------------------


def test_strip_drops_default_adapter_name():
    # Trainer's PEFT state_dict format → HF adapter-repo format.
    assert (
        strip_default_adapter_segment(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    )
    assert (
        strip_default_adapter_segment(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        )
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    )


def test_strip_handles_non_default_adapter_name():
    # PEFT supports multi-adapter setups where the segment isn't "default".
    assert (
        strip_default_adapter_segment(
            "base_model.model.x.lora_A.task_specific.weight"
        )
        == "base_model.model.x.lora_A.weight"
    )


def test_strip_handles_lora_embedding_keys():
    assert (
        strip_default_adapter_segment(
            "base_model.model.embed.lora_embedding_A.default.weight"
        )
        == "base_model.model.embed.lora_embedding_A.weight"
    )


def test_strip_is_idempotent_on_already_stripped_keys():
    # Already in HF shape (no adapter-name segment) — leave alone.
    assert (
        strip_default_adapter_segment(
            "base_model.model.x.lora_A.weight"
        )
        == "base_model.model.x.lora_A.weight"
    )


def test_strip_passes_through_non_lora_keys():
    assert (
        strip_default_adapter_segment("lm_head.weight")
        == "lm_head.weight"
    )
    assert (
        strip_default_adapter_segment(
            "model.layers.0.self_attn.q_proj.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )


# ---- _prefix_for_peft_load ----------------------------------------------
#
# Production bug this guards against: the trainer saves from inside
# the PEFT wrapper, so on-disk LoRA keys lack the `base_model.model.`
# outer prefix that PEFT's wrapped state_dict expects. Without this
# fix every key returned as "unexpected" from load_state_dict and the
# merge silently produced a no-op (uploading the unchanged base
# model). Tests pin the contract.


def test_prefix_added_to_unprefixed_keys():
    raw = {
        "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(8),
        "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.zeros(8),
    }
    prefixed = _prefix_for_peft_load(raw)
    expected = {
        f"{PEFT_OUTER_PREFIX}{k}": v for k, v in raw.items()
    }
    assert set(prefixed.keys()) == set(expected.keys())
    for k in expected:
        assert prefixed[k] is expected[k]


def test_prefix_preserves_keys_already_prefixed():
    """
    If someone re-runs the helper on already-prefixed input — or if a
    future trainer change starts saving with the prefix included —
    we must not double-prefix. Returns the same dict, untouched.
    """
    raw = {
        f"{PEFT_OUTER_PREFIX}model.x.lora_A.default.weight": torch.zeros(4),
    }
    out = _prefix_for_peft_load(raw)
    assert list(out.keys()) == list(raw.keys())


def test_prefix_handles_mixed_prefixed_and_unprefixed():
    raw = {
        "model.x.lora_A.default.weight": torch.zeros(4),
        f"{PEFT_OUTER_PREFIX}model.y.lora_A.default.weight": torch.zeros(4),
    }
    out = _prefix_for_peft_load(raw)
    assert f"{PEFT_OUTER_PREFIX}model.x.lora_A.default.weight" in out
    assert f"{PEFT_OUTER_PREFIX}model.y.lora_A.default.weight" in out
    # No double-prefixing.
    assert not any(
        k.startswith(f"{PEFT_OUTER_PREFIX}{PEFT_OUTER_PREFIX}") for k in out
    )


def test_prefix_empty_dict_passes_through():
    assert _prefix_for_peft_load({}) == {}


def test_prefix_preserves_tensor_identity():
    """The renamed dict must point at the same tensors — copying
    1202 LoRA tensors during merge would double peak memory."""
    t = torch.zeros(16)
    raw = {"model.foo.lora_A.default.weight": t}
    out = _prefix_for_peft_load(raw)
    assert next(iter(out.values())) is t
