"""Unit tests for the pure-logic helpers behind the openai proxy router.

These tests exercise the private helper functions (_extract_token_count,
_filter_params, _ensure_usage_reported) to pin robust metrics extraction
and request sanitization. They live in `openai_v1_helpers` so the test
suite can run without standing up vllm / fastapi / aiohttp.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_params_from_request,
    _filter_chat_params,
    _extract_token_count,
    _filter_params,
    _ensure_usage_reported,
)

# ---- _extract_token_count --------------------------------------------------


def test_extract_token_count_from_json_body():
    payload = json.dumps({"usage": {"total_tokens": 42}}).encode("utf-8")
    assert _extract_token_count(payload) == 42


def test_extract_token_count_from_sse_stream():
    # Multi-event stream, usage in the last event
    sse_data = (
        'data: {"choices": [{"text": "hello"}]}\n\n'
        'data: {"choices": [], "usage": {"total_tokens": 10}}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")
    assert _extract_token_count(sse_data) == 10


def test_extract_token_count_handles_varying_whitespace_and_newlines():
    # OpenAI spec allows single \n or \r\n and varying space after data:
    sse_data = (
        'data:{"usage":{"total_tokens":1}}\n\n'
        'data:   {"usage":{"total_tokens":2}}\r\n\r\n'
        'data: {"usage": {"total_tokens": 3}}\n\n'
    ).encode("utf-8")
    assert _extract_token_count(sse_data) == 3


def test_extract_token_count_handles_multiline_data_events():
    # Per the SSE spec, a single event can have multiple `data:` lines;
    # their contents are concatenated with "\n" to form one decoded
    # value. In practice vLLM's OpenAI SSE emitter keeps each JSON on
    # one line, but we want the parser to handle spec-compliant input
    # correctly so a future emitter change doesn't silently break
    # token counting.
    sse_data = ('data: {"usage":\ndata: {"total_tokens": 7}}\n\n').encode("utf-8")
    assert _extract_token_count(sse_data) == 7


def test_extract_token_count_handles_partial_trailing_data():
    # If the buffer has extra garbage at the end but valid SSE events before it
    sse_data = (
        'data: {"usage": {"total_tokens": 5}}\n\ndata: [DONE]\n\nextra garbage'
    ).encode("utf-8")
    assert _extract_token_count(sse_data) == 5


def test_extract_token_count_handles_malformed_json_gracefully():
    assert _extract_token_count(b"{ invalid }") is None
    assert _extract_token_count(b"data: { malformed }\n\n") is None


def test_extract_token_count_returns_none_on_empty():
    assert _extract_token_count(b"") is None


def test_extract_token_count_handles_unicode_errors():
    # Payload with invalid utf-8 sequences
    payload = b'{"usage": {"total_tokens": 5}, "extra": "' + b"\xff" + b'"}'
    # Decoder should use "replace" and still find the usage field if possible
    assert _extract_token_count(payload) == 5


# ---- _filter_params --------------------------------------------------------


def test_filter_params_strips_unknown_keys():
    raw = {"model": "m1", "unknown": "u1", "temperature": 0.5}
    allowed = ("model", "temperature")
    filtered = _filter_params(raw, allowed)
    assert filtered == {"model": "m1", "temperature": 0.5}


def test_filter_params_strips_none_values():
    raw = {"model": "m1", "temperature": None}
    allowed = ("model", "temperature")
    filtered = _filter_params(raw, allowed)
    assert filtered == {"model": "m1"}


def test_filter_chat_params_preserves_vllm_controls_and_false():
    raw = {
        "model": "m1",
        "messages": [{"role": "user", "content": "hi"}],
        "include_reasoning": False,
        "reasoning_effort": "low",
        "thinking_token_budget": 512,
        "top_k": 64,
        "max_completion_tokens": 4096,
        "parallel_tool_calls": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "reasoning_strength": "low",
        },
    }

    assert _filter_chat_params(raw) == raw


def test_filter_chat_params_removes_unsupported_template_kwargs():
    filtered = _filter_chat_params(
        {
            "model": "m1",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "chat_template": "{{ untrusted }}",
                "tokenize": True,
            },
        }
    )

    assert filtered == {
        "model": "m1",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_chat_request_serializer_requests_aliases_and_preserves_explicit_null():
    request = MagicMock()
    request.model_fields_set = {"tool_choice"}
    request.tool_choice = None
    request.model_dump.return_value = {
        "model": "m1",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    }

    params = _chat_params_from_request(request)

    request.model_dump.assert_called_once_with(
        mode="json", exclude_none=True, by_alias=True
    )
    assert params["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "tool_choice" in params and params["tool_choice"] is None


# ---- _ensure_usage_reported ------------------------------------------------


def test_ensure_usage_reported_injects_field_on_stream():
    params = {"stream": True}
    _ensure_usage_reported(params)
    assert params["stream_options"] == {"include_usage": True}


def test_ensure_usage_reported_respects_existing_options():
    params = {"stream": True, "stream_options": {"existing": 1}}
    _ensure_usage_reported(params)
    assert params["stream_options"] == {"existing": 1, "include_usage": True}


def test_ensure_usage_reported_no_op_on_non_stream():
    params = {"stream": False}
    _ensure_usage_reported(params)
    assert "stream_options" not in params
