from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cray_infra.api.fastapi.chat_completions import resolve_max_model_length as r


@pytest.fixture(autouse=True)
def _reset_cache():
    r.reset_cache_for_tests()
    yield
    r.reset_cache_for_tests()


def _fake_session(payload, status=200, raises=None):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=payload)

    @asynccontextmanager
    async def fake_get(url):
        if raises is not None:
            raise raises
        yield response

    session = MagicMock()
    session.get = fake_get
    return session


def _patches(session, *, vllm_url="http://vllm:8001", config_max=256):
    return [
        patch.object(r, "get_global_session", return_value=session),
        patch.object(
            r,
            "get_config",
            return_value={
                "vllm_api_url": vllm_url,
                "max_model_length": config_max,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_returns_vllm_reported_value():
    session = _fake_session(
        {"data": [{"id": "gemma-4-31B", "max_model_len": 65536}]}
    )
    with _patches(session)[0], _patches(session)[1]:
        out = await r.resolve_max_model_length("gemma-4-31B")
    assert out == 65536


@pytest.mark.asyncio
async def test_caches_after_first_lookup():
    session = _fake_session(
        {"data": [{"id": "m", "max_model_len": 4096}]}
    )
    session.get = MagicMock(side_effect=session.get)

    with _patches(session)[0], _patches(session)[1]:
        a = await r.resolve_max_model_length("m")
        b = await r.resolve_max_model_length("m")
        c = await r.resolve_max_model_length("m")
    assert a == b == c == 4096
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_falls_back_to_config_when_vllm_returns_non_200():
    session = _fake_session({}, status=503)
    with _patches(session, config_max=8192)[0], _patches(session, config_max=8192)[1]:
        out = await r.resolve_max_model_length("m")
    assert out == 8192


@pytest.mark.asyncio
async def test_falls_back_to_config_when_vllm_request_raises():
    session = _fake_session({}, raises=ConnectionError("vllm not ready"))
    with _patches(session, config_max=2048)[0], _patches(session, config_max=2048)[1]:
        out = await r.resolve_max_model_length("m")
    assert out == 2048


@pytest.mark.asyncio
async def test_falls_back_when_model_not_in_response():
    session = _fake_session(
        {"data": [{"id": "other-model", "max_model_len": 4096}]}
    )
    with _patches(session, config_max=128)[0], _patches(session, config_max=128)[1]:
        out = await r.resolve_max_model_length("our-model")
    assert out == 128


@pytest.mark.asyncio
async def test_falls_back_when_response_shape_is_unexpected():
    session = _fake_session({"unexpected": "shape"})
    with _patches(session, config_max=512)[0], _patches(session, config_max=512)[1]:
        out = await r.resolve_max_model_length("m")
    assert out == 512


@pytest.mark.asyncio
async def test_accepts_alternate_field_name_max_position_embeddings():
    session = _fake_session(
        {"data": [{"id": "m", "max_position_embeddings": 32768}]}
    )
    with _patches(session)[0], _patches(session)[1]:
        out = await r.resolve_max_model_length("m")
    assert out == 32768


@pytest.mark.asyncio
async def test_returns_zero_when_neither_vllm_nor_config_have_value():
    session = _fake_session({}, status=503)
    with _patches(session, config_max=0)[0], _patches(session, config_max=0)[1]:
        out = await r.resolve_max_model_length("m")
    assert out == 0


@pytest.mark.asyncio
async def test_per_model_cache_does_not_share_across_models():
    payloads = {
        "small": {"data": [{"id": "small", "max_model_len": 2048}]},
        "big": {"data": [{"id": "big", "max_model_len": 65536}]},
    }
    current = {"value": "small"}

    @asynccontextmanager
    async def fake_get(url):
        response = MagicMock()
        response.status = 200
        response.json = AsyncMock(return_value=payloads[current["value"]])
        yield response

    session = MagicMock()
    session.get = fake_get

    with _patches(session)[0], _patches(session)[1]:
        small = await r.resolve_max_model_length("small")
        current["value"] = "big"
        big = await r.resolve_max_model_length("big")

    assert small == 2048
    assert big == 65536


@pytest.mark.asyncio
async def test_lora_adapter_inherits_parent_max_model_len():
    session = _fake_session({
        "data": [
            {
                "id": "nvidia/Gemma-4-31B-IT-NVFP4",
                "parent": None,
                "max_model_len": 262144,
            },
            {
                "id": "adapter-hash",
                "parent": "nvidia/Gemma-4-31B-IT-NVFP4",
                "max_model_len": None,
            },
        ]
    })
    with _patches(session, config_max=256)[0], _patches(session, config_max=256)[1]:
        out = await r.resolve_max_model_length("adapter-hash")
    assert out == 262144


@pytest.mark.asyncio
async def test_lora_adapter_with_unknown_parent_falls_back_to_config():
    session = _fake_session({
        "data": [
            {
                "id": "adapter-hash",
                "parent": "missing-base",
                "max_model_len": None,
            },
        ]
    })
    with _patches(session, config_max=1024)[0], _patches(session, config_max=1024)[1]:
        out = await r.resolve_max_model_length("adapter-hash")
    assert out == 1024


@pytest.mark.asyncio
async def test_parent_chain_cycle_falls_back_to_config():
    session = _fake_session({
        "data": [
            {"id": "a", "parent": "b", "max_model_len": None},
            {"id": "b", "parent": "a", "max_model_len": None},
        ]
    })
    with _patches(session, config_max=512)[0], _patches(session, config_max=512)[1]:
        out = await r.resolve_max_model_length("a")
    assert out == 512


@pytest.mark.asyncio
async def test_adapter_with_own_max_model_len_does_not_walk_parent():
    session = _fake_session({
        "data": [
            {"id": "base", "parent": None, "max_model_len": 4096},
            {"id": "adapter", "parent": "base", "max_model_len": 8192},
        ]
    })
    with _patches(session)[0], _patches(session)[1]:
        out = await r.resolve_max_model_length("adapter")
    assert out == 8192


@pytest.mark.asyncio
async def test_concurrent_first_lookups_only_one_http_call():
    import asyncio

    call_count = {"n": 0}

    @asynccontextmanager
    async def fake_get(url):
        call_count["n"] += 1
        await asyncio.sleep(0)
        response = MagicMock()
        response.status = 200
        response.json = AsyncMock(
            return_value={"data": [{"id": "m", "max_model_len": 4096}]}
        )
        yield response

    session = MagicMock()
    session.get = fake_get

    with _patches(session)[0], _patches(session)[1]:
        results = await asyncio.gather(
            r.resolve_max_model_length("m"),
            r.resolve_max_model_length("m"),
            r.resolve_max_model_length("m"),
        )

    assert results == [4096, 4096, 4096]
    assert call_count["n"] == 1
