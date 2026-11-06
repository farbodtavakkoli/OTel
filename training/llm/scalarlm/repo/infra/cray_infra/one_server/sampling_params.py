from __future__ import annotations


def sampling_params_for(app, request) -> dict:
    model_config = getattr(app.state.engine_client, "model_config", None)
    if getattr(model_config, "is_diffusion", False):
        return {}
    return {"temperature": request.get("temperature", 0.0)}
