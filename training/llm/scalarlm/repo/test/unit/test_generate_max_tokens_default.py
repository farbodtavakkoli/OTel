"""
Pin the contract that /v1/generate plugs in a max_tokens default
when the SDK caller explicitly sends `max_tokens: null`.

Production motivation (same as chat completions, see #185): vLLM's
`request_output_to_completion_response` has a bare
`assert request.max_tokens is not None`
(vllm/entrypoints/openai/completion/serving.py:481). Passing None all
the way through generates the response and then crashes building it
with an empty `AssertionError`. The pydantic schema defaults the
field to 16 when omitted, but explicit `null` from JSON still arrives
here as None, so the handler has to defend.
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from cray_infra.api.fastapi.generate import generate as g
from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
)


def _capture_written_requests():
    """
    json.dump fires once with the requests list as the first arg.
    Capture it for inspection.
    """
    captured: list = []

    def fake_dump(obj, _fp):
        captured.append(obj)

    return captured, fake_dump


@pytest.fixture
def patched_generate(tmp_path, monkeypatch):
    """
    Patch the heavy bits of generate() so we can drive it without
    the real model manager, queue, or filesystem layout.
    """
    fake_config = {
        "model": "default-m",
        "default_max_output_tokens": 256,
        "upload_base_path": str(tmp_path),
    }
    fake_manager = MagicMock()
    fake_manager.find_model = lambda x: x  # accept any name

    with patch.object(g, "get_config", return_value=fake_config), \
         patch.object(g, "get_vllm_model_manager", return_value=fake_manager), \
         patch.object(g, "render_generate_entry", side_effect=lambda e, model: f"rendered:{e}"), \
         patch.object(g, "get_request_path", return_value=str(tmp_path / "req.json")), \
         patch.object(g, "push_into_queue", AsyncMock()), \
         patch.object(g, "poll_for_responses", AsyncMock(return_value=GenerateResponse(results=[]))):
        yield fake_config


def _build_request(**overrides) -> GenerateRequest:
    body = {"model": "default-m", "prompts": ["hi"]}
    body.update(overrides)
    return GenerateRequest(**body)


@pytest.mark.asyncio
async def test_explicit_null_max_tokens_gets_config_default(patched_generate):
    captured, fake_dump = _capture_written_requests()
    req = _build_request(max_tokens=None)

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert len(captured) == 1
    written = captured[0]
    assert written[0]["max_tokens"] == 256


@pytest.mark.asyncio
async def test_explicit_max_tokens_passes_through(patched_generate):
    captured, fake_dump = _capture_written_requests()
    req = _build_request(max_tokens=42)

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert captured[0][0]["max_tokens"] == 42


@pytest.mark.asyncio
async def test_omitted_field_uses_pydantic_default(patched_generate):
    """When the JSON body omits max_tokens entirely (pydantic's
    schema default of 16 fires), it passes through unchanged. The
    handler only intervenes for explicit None."""
    captured, fake_dump = _capture_written_requests()
    req = GenerateRequest(model="default-m", prompts=["hi"])  # no max_tokens

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert captured[0][0]["max_tokens"] == 16


@pytest.mark.asyncio
async def test_default_falls_back_to_128_when_config_missing(
    patched_generate,
):
    """If the operator removed default_max_output_tokens from cray-config,
    fall back to the in-code default of 128 — the queue must never
    pass None to vLLM, even on misconfigured pods."""
    patched_generate.pop("default_max_output_tokens")
    captured, fake_dump = _capture_written_requests()
    req = _build_request(max_tokens=None)

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert captured[0][0]["max_tokens"] == 128
