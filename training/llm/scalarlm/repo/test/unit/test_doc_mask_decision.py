"""
Unit tests for cray_megatron.megatron.doc_mask.doc_mask_decision.

The trainer packs short documents into one block and builds a 4D
block-diagonal+causal attention mask so packed docs don't attend across each
other. Text decoders accept the 4D mask; some ...ForConditionalGeneration
wrappers (e.g. Gemma3ForConditionalGeneration) compute their loss internally and
index the logits by a *2D* attention_mask, so a 4D mask raises
`IndexError: too many indices for tensor of dimension 3` — the gemma-3-4b-it
TRAIN_FAILED. For those models we fall back to the 2D mask. These tests use
lightweight fakes (a dict batch + a config stub) so they don't import the heavy
training_loop module (gpu_aware_mpi / mpi).
"""

from types import SimpleNamespace

from cray_megatron.megatron.doc_mask import (
    BUILD,
    NONE,
    SKIP_MULTIMODAL,
    SKIP_SEQLEN,
    SKIP_SSM,
    doc_mask_decision,
    has_ssm_layers,
    is_multimodal,
)

CAP = 16384


def _text_config():
    return SimpleNamespace(model_type="llama")  # no vision_config attribute


def _multimodal_config():
    return SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8))


def _nemotron_h_config():
    # NemotronH: per-layer block-type list naming a mamba block + the
    # hybrid_override_pattern marker (verified against the real config.json).
    return SimpleNamespace(
        model_type="nemotron_h",
        layers_block_type=["mamba", "mamba", "attention", "mlp"],
        hybrid_override_pattern="M-M*-M-",
    )


def _falcon_h1_config():
    # FalconH1: Mamba-2 ‖ attention, no per-layer block-type list — recognized by
    # model_type alone.
    return SimpleNamespace(model_type="falcon_h1")


def test_packed_text_model_builds_4d_mask():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, 128, _text_config(), CAP) == BUILD


def test_multimodal_skips_4d_mask_even_when_packed_and_short():
    batch = {"document_ids": object()}
    # The whole point of the fix: a vision-config wrapper must NOT get the 4D mask.
    assert doc_mask_decision(batch, 128, _multimodal_config(), CAP) == SKIP_MULTIMODAL


def test_seqlen_over_cap_skips_for_text_model():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _text_config(), CAP) == SKIP_SEQLEN


def test_multimodal_skip_takes_precedence_over_seqlen():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _multimodal_config(), CAP) == SKIP_MULTIMODAL


def test_ssm_hybrid_skips_4d_mask_even_when_packed_and_short():
    # The Nemotron-H fix: a hybrid Mamba/SSM model must NOT get the 4D mask — its
    # kernel-less mixer path multiplies by a 2D mask and a 4D one broadcasts to 5D.
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, 128, _nemotron_h_config(), CAP) == SKIP_SSM
    # FalconH1 (no block-type list) is caught by model_type.
    assert doc_mask_decision(batch, 128, _falcon_h1_config(), CAP) == SKIP_SSM


def test_ssm_skip_takes_precedence_over_seqlen():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _nemotron_h_config(), CAP) == SKIP_SSM


def test_unpacked_ssm_batch_is_none():
    # No packing → no special handling needed, even for an SSM model.
    assert doc_mask_decision({}, 128, _nemotron_h_config(), CAP) == NONE


def test_has_ssm_layers_predicate():
    assert has_ssm_layers(_nemotron_h_config()) is True
    assert has_ssm_layers(_falcon_h1_config()) is True
    assert has_ssm_layers(_text_config()) is False
    assert has_ssm_layers(_multimodal_config()) is False
    assert has_ssm_layers(None) is False
    # A block-type list with no mamba block is not SSM.
    assert has_ssm_layers(
        SimpleNamespace(model_type="x", layers_block_type=["attention", "mlp"])
    ) is False


def test_unpacked_batch_is_none():
    assert doc_mask_decision({}, 128, _text_config(), CAP) == NONE
    # Even a multimodal model with no packing needs no special handling.
    assert doc_mask_decision({}, 128, _multimodal_config(), CAP) == NONE


def test_is_multimodal_predicate():
    assert is_multimodal(_multimodal_config()) is True
    assert is_multimodal(_text_config()) is False
    assert is_multimodal(None) is False
    # vision_config present but None must read as not-multimodal.
    assert is_multimodal(SimpleNamespace(vision_config=None)) is False
