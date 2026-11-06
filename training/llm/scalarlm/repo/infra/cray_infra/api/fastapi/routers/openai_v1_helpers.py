from __future__ import annotations

import json
from typing import Any, Optional

_USAGE_SCAN_TAIL_BYTES = 64 * 1024

_COMPLETION_ALLOWED_KEYS = (
    "model",
    "temperature",
    "prompt",
    "max_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "response_format",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "return_token_ids",
)

_CHAT_ALLOWED_KEYS = (
    "model",
    "temperature",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "response_format",
    "top_p",
    "stop",
    "seed",
    "include_reasoning",
    "reasoning_effort",
    "thinking_token_budget",
    "top_k",
    "parallel_tool_calls",
    "chat_template_kwargs",
    "presence_penalty",
    "frequency_penalty",
)

_ALLOWED_CHAT_TEMPLATE_KWARGS = frozenset({"enable_thinking", "reasoning_strength"})


def _filter_params(raw: dict, allowed: tuple) -> dict:
    return {k: v for k, v in raw.items() if v is not None and k in allowed}


def _sanitize_chat_template_kwargs(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: item for key, item in value.items() if key in _ALLOWED_CHAT_TEMPLATE_KWARGS
    }


def _chat_template_kwargs_for_render(
    value: object, reasoning_effort: object = None
) -> dict:
    kwargs = _sanitize_chat_template_kwargs(value)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs.setdefault("enable_thinking", reasoning_effort != "none")
    return kwargs


def _filter_chat_params(raw: dict) -> dict:
    params = {
        key: value
        for key, value in raw.items()
        if key in _CHAT_ALLOWED_KEYS and (value is not None or key == "tool_choice")
    }
    if "chat_template_kwargs" in params:
        kwargs = _sanitize_chat_template_kwargs(params["chat_template_kwargs"])
        if kwargs:
            params["chat_template_kwargs"] = kwargs
        else:
            params.pop("chat_template_kwargs")
    return params


def _chat_params_from_request(request: Any) -> dict:
    raw = request.model_dump(mode="json", exclude_none=True, by_alias=True)
    fields_set = getattr(request, "model_fields_set", set())
    if "tool_choice" in fields_set and getattr(request, "tool_choice", None) is None:
        raw["tool_choice"] = None
    return _filter_chat_params(raw)


def _ensure_usage_reported(params: dict) -> None:
    if not params.get("stream"):
        return
    opts = dict(params.get("stream_options") or {})
    opts.setdefault("include_usage", True)
    params["stream_options"] = opts


def _read_total_tokens(json_text: str) -> Optional[int]:
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    return int(total) if isinstance(total, (int, float)) else None


def _extract_token_count(payload: bytes) -> Optional[int]:
    if not payload:
        return None
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception:
        return None

    if "data:" in text:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        last: Optional[int] = None
        for event in normalized.split("\n\n"):
            data_lines = [
                line[5:].lstrip()
                for line in event.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            body = "\n".join(data_lines)
            if not body or body == "[DONE]":
                continue
            tokens = _read_total_tokens(body)
            if tokens is not None:
                last = tokens
        if last is not None:
            return last

    return _read_total_tokens(text)
