"""
Unit tests for build_chat_completion_response.

Contract: convert the worker's flat result dict into the OpenAI
ChatCompletion shape that AsyncOpenAI parses. The non-streaming chat
completions queue path was previously unparseable end-to-end because
the heartbeat streamer dumped the raw worker dict verbatim — the SDK
expects `id`/`object`/`choices`/`usage` at the top level.
"""

from cray_infra.api.fastapi.chat_completions.build_chat_completion_response import (
    CHAT_COMPLETION_OBJECT,
    build_chat_completion_response,
)


def test_happy_path_shape():
    result = {
        "request_id": "abc_0",
        "response": "hello world",
        "token_count": 42,
        "is_acked": True,
    }

    out = build_chat_completion_response(result=result, model="m-1")

    assert out["object"] == CHAT_COMPLETION_OBJECT
    assert out["id"] == "chatcmpl-abc_0"
    assert out["model"] == "m-1"
    assert isinstance(out["created"], int) and out["created"] > 0

    choice = out["choices"][0]
    assert choice["index"] == 0
    assert choice["message"] == {"role": "assistant", "content": "hello world"}
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"] is None

    assert out["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 42,
        "total_tokens": 42,
    }
    assert "error" not in out


def test_error_path_preserves_diagnostic_and_keeps_choices_parseable():
    """An error result must still produce a ChatCompletion-shaped body
    so the SDK doesn't fail to parse — but operators need the
    diagnostic too, so the worker error is exposed as `error`."""
    result = {"request_id": "abc_0", "error": "Invalid request type: foo"}

    out = build_chat_completion_response(result=result, model="m-1")

    # SDK must be able to parse this without exception.
    assert out["object"] == CHAT_COMPLETION_OBJECT
    assert out["choices"][0]["message"]["content"] == ""
    assert out["choices"][0]["finish_reason"] == "error"

    # Operator-facing error preserved.
    assert out["error"]["message"] == "Invalid request type: foo"
    assert out["error"]["type"] == "worker_error"


def test_missing_response_field_yields_empty_content_not_none():
    """If the worker resolved but didn't write a `response` field
    (rare race; could happen if finish_work received only
    token_count), we still emit valid JSON the SDK can parse."""
    result = {"request_id": "abc_0", "token_count": 5}

    out = build_chat_completion_response(result=result, model="m-1")

    assert out["choices"][0]["message"]["content"] == ""
    # Not an error — finish_reason stays "stop" because there was no
    # explicit error from the worker.
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 5


def test_preserves_structured_reasoning_from_vllm_chat_worker():
    result = {
        "request_id": "reasoning-1",
        "response": "OK",
        "reasoning": "I should answer briefly.",
        "finish_reason": "stop",
    }

    out = build_chat_completion_response(result=result, model="reasoning-model")

    message = out["choices"][0]["message"]
    assert message["content"] == "OK"
    assert message["reasoning"] == "I should answer briefly."
    assert out["choices"][0]["finish_reason"] == "stop"


def test_preserves_structured_tool_calls_from_vllm_chat_worker():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Helsinki"}',
            },
        }
    ]
    result = {
        "request_id": "tool-1",
        "response": "",
        "reasoning": "I need the weather tool.",
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls",
    }

    out = build_chat_completion_response(result=result, model="tool-model")

    choice = out["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning"] == "I need the weather tool."
    assert choice["message"]["tool_calls"] == tool_calls
    assert choice["finish_reason"] == "tool_calls"


def test_prompt_completion_split_preferred_over_token_count():
    """When the worker propagates the prompt/completion split (the
    normal post-fix path), surface it on `usage` and recompute
    `total_tokens` from the sum so the OpenAI shape is internally
    consistent."""
    result = {
        "request_id": "abc_0",
        "response": "hi",
        "prompt_tokens": 190000,
        "completion_tokens": 8,
        # token_count present but contradictory — split should win and
        # total should be recomputed from the sum, not echoed.
        "token_count": 0,
    }
    out = build_chat_completion_response(result=result, model="m")
    assert out["usage"] == {
        "prompt_tokens": 190000,
        "completion_tokens": 8,
        "total_tokens": 190008,
    }


def test_missing_token_count_zero_usage():
    result = {"request_id": "abc_0", "response": "hi"}
    out = build_chat_completion_response(result=result, model="m-1")
    assert out["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_id_falls_back_when_request_id_missing():
    """A result with no request_id (degenerate, but possible if some
    layer dropped it) still produces a `chatcmpl-…` id — the SDK
    rejects bodies without one."""
    out = build_chat_completion_response(result={"response": "hi"}, model="m")
    assert out["id"].startswith("chatcmpl-")
    assert len(out["id"]) > len("chatcmpl-")


def test_unwraps_group_level_shape():
    """
    Field-tested in the pod: _response.json-shaped dicts can arrive at
    the wrapper instead of the per-request shape, leaving the actual
    response buried under `results.<id>.response`. Symptom was empty
    `content` in the SDK reply.
    """
    group = {
        "results": {
            "abc_000000000": {
                "is_acked": True,
                "response": "[Tool: bash] {...}",
            }
        },
        "current_index": 1,
        "total_requests": 1,
        "work_queue_id": None,
    }

    out = build_chat_completion_response(result=group, model="m")

    assert out["choices"][0]["message"]["content"] == "[Tool: bash] {...}"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_unwrap_picks_first_entry_deterministically():
    """If multiple entries somehow appear (shouldn't today; defensive
    coverage), sort keys so two concurrent callers agree which entry
    is "first"."""
    group = {
        "results": {
            "abc_000000001": {"response": "B"},
            "abc_000000000": {"response": "A"},
        },
        "current_index": 2,
        "total_requests": 2,
    }
    out = build_chat_completion_response(result=group, model="m")
    assert out["choices"][0]["message"]["content"] == "A"


def test_unwrap_no_op_on_per_request_shape():
    """Per-request shape (the documented contract) must pass through
    unchanged."""
    out = build_chat_completion_response(
        result={"response": "direct", "is_acked": True}, model="m"
    )
    assert out["choices"][0]["message"]["content"] == "direct"
