"""
Wrap the worker's flat result dict into the OpenAI ChatCompletion
shape the `AsyncOpenAI` SDK expects.

The worker (see `create_generate_worker.async_completion_task`)
returns:

    {
      "request_id": "<id>",
      "response": "<text>",            # present on success
      "error": "<message>",            # present on failure
      "token_count": <int>,            # total_tokens from vLLM
      "prompt_tokens": <int>,          # split from vLLM usage
      "completion_tokens": <int>,      # split from vLLM usage
      "is_acked": True,                # added by update_and_ack
      ...                              # original request fields
    }

We prefer the prompt/completion split when both are present and
recompute `total_tokens` from the sum so the OpenAI `usage` object is
internally consistent. When only `token_count` is available (older
worker, or a code path that didn't propagate the split), we fall back
to attributing it all to `completion_tokens` so `total_tokens` stays
accurate even if the split is approximate.
"""

import time
from typing import Any

CHAT_COMPLETION_OBJECT = "chat.completion"


def build_chat_completion_response(
    *, result: dict[str, Any], model: str
) -> dict[str, Any]:
    """
    `result` is the dict the handler's heartbeat streamer is about to
    JSON-dump — i.e. the worker output enriched by `update_and_ack`.
    `model` is captured from the original request so we don't have to
    rely on the worker echoing it back.

    The future resolution path *should* hand us the per-request dict
    (`{is_acked, response, ...}`), but the on-disk `_response.json`
    matches a group-level shape (`{results: {<id>: per_request_dict},
    current_index, total_requests, ...}`) and field-testing showed
    empty content unless we tolerate both. `_unwrap_group_dict`
    is a no-op on the per-request shape, so this defends without
    regressing the documented contract.
    """
    result = _unwrap_group_dict(result)

    error = result.get("error")
    tool_calls = None if error else result.get("tool_calls")
    content: str | None = "" if error else (result.get("response") or "")
    reasoning = None if error else result.get("reasoning")

    # OpenAI represents a tool-call-only assistant turn with null content.
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

    # OpenAI's error shape is a top-level `error` field; the SDK will
    # still parse the body if `choices` is present, so we preserve both
    # to give operators the diagnostic and the SDK a parseable result.
    if error:
        response["error"] = {"message": error, "type": "worker_error"}

    return response


def _resolve_usage(result: dict[str, Any]) -> tuple[int, int, int]:
    """Pull prompt/completion/total from the worker result.

    Prefer the split when both fields are present (recomputing total from
    the sum, since vLLM occasionally reports `total_tokens` slightly off
    from the sum on cache-hit paths). Fall back to `token_count` as the
    total when the split is missing — attribute it to completion so the
    `total_tokens` field stays meaningful even if the split is approximate.
    """
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
    """OpenAI ids are `chatcmpl-<token>`. Reuse the worker's request_id
    when it's a string so the inference browser can correlate; fall
    back to a timestamp-based id otherwise."""
    if isinstance(request_id, str) and request_id:
        return f"chatcmpl-{request_id}"
    return f"chatcmpl-{int(time.time() * 1000)}"


def _unwrap_group_dict(result: Any) -> dict[str, Any]:
    """
    If `result` is a group-level dict (has `results` mapping
    request_id → per_request_dict), pick the first per-request entry.

    The chat completions queue path packs requests one-per-row today
    (the coalescer batches at the file level, but each row that
    finishes resolves its own correlation_id), so picking the first
    entry is correct. If we ever get multi-prompt rows resolving a
    single cid, this needs to merge instead.

    Hands non-dicts straight back so the caller's `.get(...)` calls
    raise their natural AttributeError rather than silently returning
    empty content.
    """
    if not isinstance(result, dict):
        return result
    nested = result.get("results")
    if not isinstance(nested, dict) or not nested:
        return result
    # Mark the choice deterministic so two concurrent calls on the
    # same group don't disagree about which entry is "first".
    first_key = next(iter(sorted(nested.keys())))
    candidate = nested[first_key]
    if not isinstance(candidate, dict):
        return result
    return candidate
