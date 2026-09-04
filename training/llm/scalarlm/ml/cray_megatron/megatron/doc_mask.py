"""Decide how to handle packed-document attention masking per batch/model."""

# Decision outcomes for doc_mask_decision().
BUILD = "build"                    # construct the 4D block-diagonal+causal mask
SKIP_MULTIMODAL = "skip_multimodal"  # wrapper masks loss by a 2D mask; keep 2D
SKIP_SSM = "skip_ssm"              # hybrid Mamba/SSM mixer needs a 2D mask; keep 2D
SKIP_SEQLEN = "skip_seqlen"        # mask too large to materialize; keep 2D
NONE = "none"                      # batch isn't packed (no document_ids)

# Hybrid Mamba/SSM model_types that need a 2D mask.
_SSM_MODEL_TYPES = {
    "nemotron_h",
    "falcon_h1",
    "granitemoehybrid",
    "bamba",
    "zamba2",
    "jamba",
}


def is_multimodal(model_config) -> bool:
    """True for HF multimodal wrapper configs, which nest a ``vision_config``."""
    if model_config is None:
        return False
    return getattr(model_config, "vision_config", None) is not None


def is_diffusion(model_config) -> bool:
    """True for DiffusionGemma configs; must be checked BEFORE the multimodal fork."""
    if model_config is None:
        return False
    return getattr(model_config, "model_type", None) == "diffusion_gemma"



def has_ssm_layers(model_config) -> bool:
    """True for hybrid state-space model configs, which need a 2D padding mask."""
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
    """Return how to handle packed-document attention for this batch."""
    if "document_ids" not in batch:
        return NONE
    if is_multimodal(model_config):
        return SKIP_MULTIMODAL
    if has_ssm_layers(model_config):
        return SKIP_SSM
    if seq_len > max_4d_mask_seq_len:
        return SKIP_SEQLEN
    return BUILD
