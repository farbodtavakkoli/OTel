"""Runtime chat tokenization must use vLLM's configured renderer safely."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest

from cray_infra.api.fastapi.chat_completions import (
    tokenize_chat_for_admission as tokenization,
)


def _chat_request(**overrides):
    request = {
        "model": "model-1",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "high",
        "chat_template_kwargs": {"reasoning_strength": "low"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    request.update(overrides)
    return request


def _session(*, status=200, body=None, text=""):
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=text)

    @asynccontextmanager
    async def post(url, json):
        response.request_url = url
        response.request_json = json
        yield response

    session = MagicMock()
    session.post = post
    return session, response


def test_payload_uses_supported_subset_and_never_forwards_template_override():
    payload = tokenization._build_tokenize_payload(
        model="resolved-model",
        chat_request=_chat_request(
            chat_template="{{ top-level untrusted }}",
            chat_template_kwargs={
                "reasoning_strength": "low",
                "reasoning_effort": "none",
                "chat_template": "{{ nested untrusted }}",
                "tokenize": True,
            },
        ),
    )

    assert payload["model"] == "resolved-model"
    assert payload["add_generation_prompt"] is True
    assert payload["continue_final_message"] is False
    assert payload["add_special_tokens"] is False
    assert payload["chat_template_kwargs"] == {
        "reasoning_strength": "low",
        "reasoning_effort": "high",
        "enable_thinking": True,
    }
    assert "chat_template" not in payload
    assert set(payload) == {
        "model",
        "messages",
        "tools",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
    }


def test_payload_revalidates_under_vllm_tokenize_protocol():
    """The exact v0.26 and v0.27 images both run this wire-contract test."""
    payload = tokenization._build_tokenize_payload(
        model="resolved-model",
        chat_request=_chat_request(),
    )

    parsed = TokenizeChatRequest.model_validate(payload)

    assert parsed.model == "resolved-model"
    assert parsed.add_generation_prompt is True
    assert parsed.continue_final_message is False
    assert parsed.add_special_tokens is False
    assert parsed.chat_template is None
    assert parsed.chat_template_kwargs == {
        "reasoning_strength": "low",
        "reasoning_effort": "high",
        "enable_thinking": True,
    }
    assert parsed.tools is not None
    assert "chat_template" not in payload
    assert set(payload) == {
        "model",
        "messages",
        "tools",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
    }


def test_tokenize_template_params_match_chat_completion_request():
    """Both exact images must render the same request-level template kwargs."""
    chat_wire = _chat_request(model="resolved-model")
    chat_request = ChatCompletionRequest.model_validate(chat_wire)
    tokenize_request = TokenizeChatRequest.model_validate(
        tokenization._build_tokenize_payload(
            model="resolved-model",
            chat_request=chat_wire,
        )
    )

    chat_params = chat_request.build_chat_params("server-template", "auto")
    tokenize_params = tokenize_request.build_chat_params("server-template", "auto")

    assert tokenize_params.chat_template == chat_params.chat_template
    assert tokenize_params.chat_template_kwargs == chat_params.chat_template_kwargs


@pytest.mark.asyncio
async def test_success_uses_vllm_tokenize_endpoint_and_runtime_values():
    session, response = _session(
        body={"count": 73, "max_model_len": 65536, "tokens": [1, 2, 3]}
    )
    with (
        patch.object(tokenization, "get_global_session", return_value=session),
        patch.object(
            tokenization,
            "get_config",
            return_value={"vllm_api_url": "http://vllm:8001"},
        ),
    ):
        result = await tokenization.tokenize_chat_for_admission(
            model="model-1",
            chat_request=_chat_request(),
        )

    assert result == tokenization.AdmissionTokenization(
        prompt_tokens=73,
        max_model_length=65536,
    )
    assert response.request_url == "http://vllm:8001/tokenize"
    assert response.request_json["model"] == "model-1"
    assert "chat_template" not in response.request_json


@pytest.mark.asyncio
async def test_deterministic_4xx_is_raised_with_vllm_message():
    session, _ = _session(
        status=400,
        body={"error": {"message": "chat template rejected messages"}},
    )
    with (
        patch.object(tokenization, "get_global_session", return_value=session),
        patch.object(
            tokenization,
            "get_config",
            return_value={"vllm_api_url": "http://vllm:8001"},
        ),
    ):
        with pytest.raises(tokenization.VLLMTokenizeRequestError) as exc_info:
            await tokenization.tokenize_chat_for_admission(
                model="model-1",
                chat_request=_chat_request(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "chat template rejected messages"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body",
    [
        (503, {"error": {"message": "starting"}}),
        (200, {"count": "wrong", "max_model_len": 65536}),
    ],
)
async def test_transient_or_malformed_response_fails_open(status, body):
    session, _ = _session(status=status, body=body)
    with (
        patch.object(tokenization, "get_global_session", return_value=session),
        patch.object(
            tokenization,
            "get_config",
            return_value={"vllm_api_url": "http://vllm:8001"},
        ),
    ):
        result = await tokenization.tokenize_chat_for_admission(
            model="model-1",
            chat_request=_chat_request(),
        )

    assert result is None


@pytest.mark.asyncio
async def test_network_error_fails_open():
    session = MagicMock()
    session.post.side_effect = OSError("connection refused")
    with (
        patch.object(tokenization, "get_global_session", return_value=session),
        patch.object(
            tokenization,
            "get_config",
            return_value={"vllm_api_url": "http://vllm:8001"},
        ),
    ):
        result = await tokenization.tokenize_chat_for_admission(
            model="model-1",
            chat_request=_chat_request(),
        )

    assert result is None


@pytest.mark.asyncio
async def test_slow_tokenize_response_times_out_and_fails_open():
    @asynccontextmanager
    async def slow_post(url, json):
        await asyncio.sleep(1)
        yield MagicMock()

    session = MagicMock()
    session.post = slow_post
    with (
        patch.object(tokenization, "get_global_session", return_value=session),
        patch.object(tokenization, "_TOKENIZE_TIMEOUT_SECONDS", 0.001),
        patch.object(
            tokenization,
            "get_config",
            return_value={"vllm_api_url": "http://vllm:8001"},
        ),
    ):
        result = await tokenization.tokenize_chat_for_admission(
            model="model-1",
            chat_request=_chat_request(),
        )

    assert result is None
