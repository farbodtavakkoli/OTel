import time
from typing import Any

CHAT_COMPLETION_OBJECT = "chat.completion"


def build_chat_completion_response(
    *, result: dict[str, Any], model: str
) -> dict[str, Any]:
    result = _unwrap_group_dict(result)

    error = result.get("error")
    tool_calls = None if error else result.get("tool_calls")
    content: str | None = "" if error else (result.get("response") or "")
    reasoning = None if error else result.get("reasoning")

    if tool_calls and not content:
        content = None

    chat_id = _build_chat_id(result.get("request_id"))
    finish_reason = "stop" if not error else "error"
    prompt_tokens, completion_tokens, total_tokens = _resolve_usage(result)

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    response: dict[str, Any] = {
        "id": chat_id,
        "object": CHAT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.get("finish_reason") or finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    if error:
        response["error"] = {"message": error, "type": "worker_error"}

    return response


def _resolve_usage(result: dict[str, Any]) -> tuple[int, int, int]:
    prompt = result.get("prompt_tokens")
    completion = result.get("completion_tokens")
    if prompt is not None and completion is not None:
        p, c = int(prompt), int(completion)
        return p, c, p + c

    total = int(result.get("token_count") or 0)
    if prompt is not None:
        p = int(prompt)
        return p, max(0, total - p), total
    if completion is not None:
        c = int(completion)
        return max(0, total - c), c, total
    return 0, total, total


def _build_chat_id(request_id: Any) -> str:
    if isinstance(request_id, str) and request_id:
        return f"chatcmpl-{request_id}"
    return f"chatcmpl-{int(time.time() * 1000)}"


def _unwrap_group_dict(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result
    nested = result.get("results")
    if not isinstance(nested, dict) or not nested:
        return result
    first_key = next(iter(sorted(nested.keys())))
    candidate = nested[first_key]
    if not isinstance(candidate, dict):
        return result
    return candidate
