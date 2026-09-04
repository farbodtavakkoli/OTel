"""Sampling params the served model will accept.

Kept free of heavy imports (torch, vllm) so it can be unit tested without
the full inference stack, same as vllm_cli_args.py.
"""

from __future__ import annotations


def sampling_params_for(app, request) -> dict:
    """Build the sampling kwargs for a generate request.

    Diffusion models (dLLMs) reject temperature, min_p, seed, min_tokens,
    logit_bias, bad_words and allowed_token_ids -- they denoise a canvas
    rather than sampling token by token, so those knobs have no meaning:

        ValueError: The temperature, min_p, seed, min_tokens, logit_bias,
        bad_words, and allowed_token_ids sampling parameters are not yet
        supported with diffusion models.

    temperature is the only one of those we ever set, so dropping it is
    enough. A model_config without `is_diffusion` (older vLLM) keeps the
    previous behavior rather than silently dropping temperature.
    """
    model_config = getattr(app.state.engine_client, "model_config", None)
    if getattr(model_config, "is_diffusion", False):
        return {}
    return {"temperature": request.get("temperature", 0.0)}
