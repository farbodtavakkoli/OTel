"""Queue-backed chat requests must retain vLLM's structured semantics."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import Response
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_params_from_request,
)
from cray_infra.one_server import create_generate_worker as worker


@pytest.mark.asyncio
async def test_chat_worker_rehydrates_request_and_preserves_structure(monkeypatch):
    captured = {}

    async def fake_create_chat_completion(request, raw_request):
        captured["request"] = request
        captured["path"] = raw_request.scope["path"]
        return Response(
            content=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "I need a tool.",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Helsinki"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "total_tokens": 19,
                    },
                }
            ),
            media_type="application/json",
        )

    monkeypatch.setattr(worker, "create_chat_completion", fake_create_chat_completion)

    app = MagicMock()
    app.state.engine_client.check_health = AsyncMock()
    app.state.engine_client.model_config = MagicMock()
    queued = {
        "request_id": "req-1",
        "request_type": "generate",
        "prompt": "pre-rendered prompt used for accounting only",
        "chat_request": {
            "model": "muse",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_completion_tokens": 128,
            "include_reasoning": False,
            "reasoning_effort": "low",
            "thinking_token_budget": 96,
            "top_k": 64,
            "parallel_tool_calls": False,
            "chat_template_kwargs": {"reasoning_strength": "low"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        },
    }

    result = await worker.async_generate_task(queued, app)

    request = captured["request"]
    assert captured["path"] == "/v1/chat/completions"
    assert request.messages == [{"role": "user", "content": "weather?"}]
    assert request.tools[0].function.name == "get_weather"
    assert request.include_reasoning is False
    assert request.reasoning_effort == "low"
    assert request.thinking_token_budget == 96
    assert request.top_k == 64
    assert request.max_tokens is None
    assert request.max_completion_tokens == 128
    assert request.parallel_tool_calls is False
    assert request.chat_template_kwargs == {"reasoning_strength": "low"}
    assert result["response"] == ""
    assert result["reasoning"] == "I need a tool."
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["finish_reason"] == "tool_calls"
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 7
    assert result["token_count"] == 19
    app.state.engine_client.check_health.assert_awaited_once()


def test_json_schema_wire_alias_survives_queue_revalidation():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    request = ChatCompletionRequest(
        model="model-1",
        messages=[{"role": "user", "content": "return JSON"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": schema,
            },
        },
    )

    queued = _chat_params_from_request(request)
    assert queued["response_format"]["json_schema"]["schema"] == schema
    assert "json_schema" not in queued["response_format"]["json_schema"]

    rehydrated = ChatCompletionRequest(**queued)
    assert rehydrated.response_format.json_schema.json_schema == schema


def test_explicit_null_tool_choice_survives_filter_dump_and_revalidation():
    request = ChatCompletionRequest(
        model="model-1",
        messages=[{"role": "user", "content": "do not call a tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice=None,
    )

    queued = _chat_params_from_request(request)
    assert "tool_choice" in queued
    assert queued["tool_choice"] is None

    rehydrated = ChatCompletionRequest(**queued)
    assert rehydrated.tool_choice is None


def test_debug_log_sanitizer_recurses_lists_and_bounds_sensitive_data():
    original = {
        "requests": [
            {
                "prompt": "private prompt",
                "chat_request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "x" * 500},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64," + "A" * 5000
                                    },
                                },
                            ],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "d" * 500,
                            },
                        }
                    ]
                    * 25,
                },
            }
        ]
    }

    sanitized = worker.truncate_fields(original)

    assert original["requests"][0]["prompt"] == "private prompt"
    request = sanitized["requests"][0]
    assert request["prompt"] == "<redacted 14 chars>"
    assert request["chat_request"]["messages"][0]["content"].startswith("<redacted")
    tools = request["chat_request"]["tools"]
    assert len(tools) == worker._LOG_COLLECTION_LIMIT + 1
    assert tools[-1] == "<truncated 5 items>"
    assert tools[0]["function"]["description"].endswith("...")
    assert "data:image/png" not in repr(sanitized)


def test_debug_log_sanitizer_bounds_mapping_work_and_nesting():
    class BoundedItemsDict(dict):
        def items(self):
            for index, item in enumerate(super().items()):
                if index > worker._LOG_COLLECTION_LIMIT:
                    raise AssertionError("sanitizer consumed the complete mapping")
                yield item

    large = BoundedItemsDict((str(index), index) for index in range(100_000))
    nested = []
    cursor = nested
    for _ in range(worker._LOG_DEPTH_LIMIT + 5):
        child = []
        cursor.append(child)
        cursor = child

    sanitized = worker.truncate_fields({"large": large, "nested": nested})

    assert len(sanitized["large"]) == worker._LOG_COLLECTION_LIMIT + 1
    assert sanitized["large"]["<truncated>"] == "99980 keys"
    assert "<truncated nested value>" in repr(sanitized["nested"])


@pytest.mark.asyncio
async def test_process_requests_skips_sanitizer_when_debug_is_disabled(monkeypatch):
    monkeypatch.setattr(worker.logger, "isEnabledFor", lambda level: False)
    monkeypatch.setattr(
        worker,
        "truncate_fields",
        lambda value: pytest.fail("sanitizer should not run outside DEBUG logging"),
    )

    await worker.process_requests_task(MagicMock(), [])


@pytest.mark.asyncio
async def test_chat_worker_preserves_vllm_error_message(monkeypatch):
    async def fake_create_chat_completion(request, raw_request):
        return Response(
            content=json.dumps(
                {"error": {"message": "invalid tool choice", "type": "BadRequest"}}
            ),
            media_type="application/json",
            status_code=400,
        )

    monkeypatch.setattr(worker, "create_chat_completion", fake_create_chat_completion)

    app = MagicMock()
    app.state.engine_client.check_health = AsyncMock()
    queued = {
        "request_id": "req-error",
        "request_type": "chat_completions",
        "prompt": "accounting prompt",
        "chat_request": {
            "model": "model-1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": False,
        },
    }

    result = await worker.async_generate_task(queued, app)

    assert result == {
        "request_id": "req-error",
        "error": "invalid tool choice",
    }
    app.state.engine_client.check_health.assert_awaited_once()
