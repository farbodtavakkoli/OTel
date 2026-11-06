BUILD = "build"
SKIP_MULTIMODAL = "skip_multimodal"
SKIP_SSM = "skip_ssm"
SKIP_SEQLEN = "skip_seqlen"
NONE = "none"

_SSM_MODEL_TYPES = {
    "nemotron_h",
    "falcon_h1",
    "granitemoehybrid",
    "bamba",
    "zamba2",
    "jamba",
}


def is_multimodal(model_config) -> bool:
    if model_config is None:
        return False
    return getattr(model_config, "vision_config", None) is not None


def is_diffusion(model_config) -> bool:
    if model_config is None:
        return False
    return getattr(model_config, "model_type", None) == "diffusion_gemma"


def has_ssm_layers(model_config) -> bool:
    if model_config is None:
        return False
    block_types = getattr(model_config, "layers_block_type", None)
    if block_types and any("mamba" in str(bt).lower() for bt in block_types):
        return True
    if getattr(model_config, "hybrid_override_pattern", None):
        return True
    model_type = getattr(model_config, "model_type", None) or ""
    return model_type in _SSM_MODEL_TYPES


def doc_mask_decision(batch, seq_len: int, model_config, max_4d_mask_seq_len: int) -> str:
    if "document_ids" not in batch:
        return NONE
    if is_multimodal(model_config):
        text_only_glimmer = getattr(model_config, "model_type", None) == "muse_glimmer" and not any(
            key in batch for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
        )
        if not text_only_glimmer:
            return SKIP_MULTIMODAL
    if has_ssm_layers(model_config):
        return SKIP_SSM
    if seq_len > max_4d_mask_seq_len:
        return SKIP_SEQLEN
    return BUILD
