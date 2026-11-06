from __future__ import annotations

import types

from cray_infra.one_server.sampling_params import sampling_params_for


def _app(is_diffusion):
    engine_client = types.SimpleNamespace(
        model_config=types.SimpleNamespace(is_diffusion=is_diffusion)
    )
    return types.SimpleNamespace(state=types.SimpleNamespace(engine_client=engine_client))


def test_autoregressive_model_gets_temperature():
    assert sampling_params_for(_app(False), {}) == {"temperature": 0.0}


def test_explicit_temperature_is_passed_through():
    params = sampling_params_for(_app(False), {"temperature": 0.7})

    assert params == {"temperature": 0.7}


def test_diffusion_model_gets_no_temperature():
    assert sampling_params_for(_app(True), {"temperature": 0.7}) == {}


def test_missing_model_config_falls_back_to_autoregressive():
    engine_client = types.SimpleNamespace(model_config=types.SimpleNamespace())
    app = types.SimpleNamespace(
        state=types.SimpleNamespace(engine_client=engine_client)
    )

    assert sampling_params_for(app, {}) == {"temperature": 0.0}


def test_absent_engine_client_attribute_does_not_raise():
    app = types.SimpleNamespace(
        state=types.SimpleNamespace(engine_client=types.SimpleNamespace())
    )

    assert sampling_params_for(app, {}) == {"temperature": 0.0}
