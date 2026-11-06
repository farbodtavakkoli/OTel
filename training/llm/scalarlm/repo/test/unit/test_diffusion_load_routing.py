import pytest

from cray_megatron.megatron.doc_mask import is_diffusion, is_multimodal


def _diffusion_config():
    try:
        from transformers.models.diffusion_gemma import DiffusionGemmaConfig
    except Exception:
        pytest.skip("transformers build has no diffusion_gemma module")
    return DiffusionGemmaConfig()


def test_default_diffusion_config_is_diffusion():
    cfg = _diffusion_config()
    assert is_diffusion(cfg) is True


def test_default_diffusion_config_not_multimodal():
    cfg = _diffusion_config()
    assert is_multimodal(cfg) is False


def test_is_diffusion_false_for_plain_config():
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("gpt2")
    assert is_diffusion(cfg) is False


def test_is_diffusion_branch_ordered_before_multimodal():
    class FakeCfg:
        model_type = "diffusion_gemma"
        vision_config = object()

    cfg = FakeCfg()
    assert is_diffusion(cfg) is True
    assert is_multimodal(cfg) is True
    if is_diffusion(cfg):
        chosen = "diffusion"
    elif is_multimodal(cfg):
        chosen = "multimodal"
    else:
        chosen = "causal"
    assert chosen == "diffusion"
