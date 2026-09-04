"""Pure-logic helpers for the OpenAI proxy router.

Kept free of vllm / fastapi / aiohttp imports so the logic stays
unit-testable without the full inference stack. ``openai_v1_router``
re-exports these names.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Tail-window for sniffing the upstream payload for `usage`. The terminal
# usage event in an OpenAI SSE stream is on the order of a few hundred
# bytes and always sits at the very end; 64 KB is more than enough
# headroom while bounding memory for very long completions.
_USAGE_SCAN_TAIL_BYTES = 64 * 1024

# Allowed keys on requests forwarded to vLLM. ``stream_options`` is
# included so callers can opt into usage reporting; we also force it on
# for streaming requests below so we can count tokens server-side.
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
    # vLLM chat extensions. Keep these on the shared allowlist so both
    # the direct streaming proxy and queue-backed non-streaming path honor
    # the same reasoning and sampling controls. In particular, filtering
    # must preserve the literal False used by include_reasoning.
    "include_reasoning",
    "reasoning_effort",
    "thinking_token_budget",
    "top_k",
    "parallel_tool_calls",
    "chat_template_kwargs",
    "presence_penalty",
    "frequency_penalty",
)

# Only template variables that ScalarLM has explicitly validated are accepted
# from callers. This is shared by the direct proxy, queue payload, and local
# accounting render so all three paths see the same request semantics.
_ALLOWED_CHAT_TEMPLATE_KWARGS = frozenset({"enable_thinking", "reasoning_strength"})


def _filter_params(raw: dict, allowed: tuple) -> dict:
    return {k: v for k, v in raw.items() if v is not None and k in allowed}


def _sanitize_chat_template_kwargs(value: object) -> dict:
    """Return only explicitly supported, JSON-safe template variables."""
    if not isinstance(value, dict):
        return {}
    return {
        key: item for key, item in value.items() if key in _ALLOWED_CHAT_TEMPLATE_KWARGS
    }


def _chat_template_kwargs_for_render(
    value: object, reasoning_effort: object = None
) -> dict:
    """Build the template kwargs vLLM derives from a chat request.

    vLLM 0.26 and 0.27 expose the validated effort to the template as
    ``reasoning_effort`` and also translate it into ``enable_thinking`` unless
    the caller supplied that safe template knob directly. Keep that rule in
    one dependency-light helper so the local rolling-upgrade fallback and
    vLLM's authoritative ``/tokenize`` request use identical request-level
    kwargs.
    """
    kwargs = _sanitize_chat_template_kwargs(value)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs.setdefault("enable_thinking", reasoning_effort != "none")
    return kwargs


def _filter_chat_params(raw: dict) -> dict:
    """Filter a chat request and constrain its nested template kwargs.

    ``tool_choice: null`` is presence-sensitive in vLLM: omitted means that
    the request validator may choose ``auto`` when tools are present, while an
    explicit null remains disabled. Preserve that one null-valued field while
    continuing to drop unrelated ``None`` defaults.
    """
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
    """Serialize a validated vLLM chat request for an HTTP/queue boundary.

    ``by_alias=True`` is required for nested OpenAI wire aliases such as the
    JSON-schema wrapper's ``schema`` key. Pydantic's ``exclude_none`` is useful
    for the many optional defaults, but it erases the semantic distinction for
    an explicitly null ``tool_choice``; restore that key only when the original
    model says the caller set it.
    """
    raw = request.model_dump(mode="json", exclude_none=True, by_alias=True)
    fields_set = getattr(request, "model_fields_set", set())
    if "tool_choice" in fields_set and getattr(request, "tool_choice", None) is None:
        raw["tool_choice"] = None
    return _filter_chat_params(raw)


def _ensure_usage_reported(params: dict) -> None:
    """For streaming requests, force vLLM to emit a final ``usage`` event
    so we can count tokens. OpenAI-compatible clients tolerate the extra
    field; the ScalarLM chat UI specifically reads it and surfaces
    tokens-per-message. Non-streaming responses always include usage in
    the final JSON body, so no opt-in is needed there.
    """
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
    """Best-effort scan for ``usage.total_tokens`` in either an SSE
    stream tail or a single JSON response body. Returns None if not
    found.
    """
    if not payload:
        return None
    try:
        text = payload.decode("utf-8", errors="replace")
    except Exception:
        return None

    # SSE path — look for the last event that contains a `usage` field.
    # Per the SSE spec:
    #   - events are separated by an empty line (\n\n, \r\n\r\n, or \r\r)
    #   - a single event can contain multiple `data:` lines whose values
    #     are concatenated with "\n" to form one decoded value
    # Normalize line endings to LF up-front so split("\n\n") handles all
    # separator forms; parse per-event (not per-line) so multi-line JSON
    # still resolves to a single object.
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

    # Non-streaming path — try parsing the whole tail as JSON.
    return _read_total_tokens(text)
