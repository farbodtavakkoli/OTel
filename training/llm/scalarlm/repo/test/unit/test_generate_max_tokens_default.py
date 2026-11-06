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
    captured: list = []

    def fake_dump(obj, _fp):
        captured.append(obj)

    return captured, fake_dump


@pytest.fixture
def patched_generate(tmp_path, monkeypatch):
    fake_config = {
        "model": "default-m",
        "default_max_output_tokens": 256,
        "upload_base_path": str(tmp_path),
    }
    fake_manager = MagicMock()
    fake_manager.find_model = lambda x: x

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
    captured, fake_dump = _capture_written_requests()
    req = GenerateRequest(model="default-m", prompts=["hi"])

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert captured[0][0]["max_tokens"] == 16


@pytest.mark.asyncio
async def test_default_falls_back_to_128_when_config_missing(
    patched_generate,
):
    patched_generate.pop("default_max_output_tokens")
    captured, fake_dump = _capture_written_requests()
    req = _build_request(max_tokens=None)

    with patch("builtins.open", mock_open()), \
         patch("json.dump", side_effect=fake_dump):
        await g.generate(req)

    assert captured[0][0]["max_tokens"] == 128
