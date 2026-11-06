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
    return SimpleNamespace(model_type="llama")


def _multimodal_config():
    return SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8))


def _nemotron_h_config():
    return SimpleNamespace(
        model_type="nemotron_h",
        layers_block_type=["mamba", "mamba", "attention", "mlp"],
        hybrid_override_pattern="M-M*-M-",
    )


def _falcon_h1_config():
    return SimpleNamespace(model_type="falcon_h1")


def test_packed_text_model_builds_4d_mask():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, 128, _text_config(), CAP) == BUILD


def test_multimodal_skips_4d_mask_even_when_packed_and_short():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, 128, _multimodal_config(), CAP) == SKIP_MULTIMODAL


def test_seqlen_over_cap_skips_for_text_model():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _text_config(), CAP) == SKIP_SEQLEN


def test_multimodal_skip_takes_precedence_over_seqlen():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _multimodal_config(), CAP) == SKIP_MULTIMODAL


def test_ssm_hybrid_skips_4d_mask_even_when_packed_and_short():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, 128, _nemotron_h_config(), CAP) == SKIP_SSM
    assert doc_mask_decision(batch, 128, _falcon_h1_config(), CAP) == SKIP_SSM


def test_ssm_skip_takes_precedence_over_seqlen():
    batch = {"document_ids": object()}
    assert doc_mask_decision(batch, CAP + 1, _nemotron_h_config(), CAP) == SKIP_SSM


def test_unpacked_ssm_batch_is_none():
    assert doc_mask_decision({}, 128, _nemotron_h_config(), CAP) == NONE


def test_has_ssm_layers_predicate():
    assert has_ssm_layers(_nemotron_h_config()) is True
    assert has_ssm_layers(_falcon_h1_config()) is True
    assert has_ssm_layers(_text_config()) is False
    assert has_ssm_layers(_multimodal_config()) is False
    assert has_ssm_layers(None) is False
    assert has_ssm_layers(
        SimpleNamespace(model_type="x", layers_block_type=["attention", "mlp"])
    ) is False


def test_unpacked_batch_is_none():
    assert doc_mask_decision({}, 128, _text_config(), CAP) == NONE
    assert doc_mask_decision({}, 128, _multimodal_config(), CAP) == NONE


def test_is_multimodal_predicate():
    assert is_multimodal(_multimodal_config()) is True
    assert is_multimodal(_text_config()) is False
    assert is_multimodal(None) is False
    assert is_multimodal(SimpleNamespace(vision_config=None)) is False
