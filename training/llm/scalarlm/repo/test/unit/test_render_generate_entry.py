"""
Unit tests for render_generate_entry.

Each `prompts` entry in a /v1/generate request can be:
  - a bare string (legacy raw passthrough)
  - a dict with `prompt: str` (raw passthrough, equivalent form)
  - a dict with `messages: list[ChatMessage]` (rendered through chat template)

See docs/openai-chat-completions-queue.md §10. This is the single
function that owns the dispatch between those forms; both /v1/generate
and (potentially) other batch entrypoints route through it so the
rendering rules don't drift across paths.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.api.fastapi.chat_completions.render_generate_entry import (
    render_generate_entry,
)


def test_bare_string_entry_passes_through():
    assert render_generate_entry("hello", model="any") == "hello"


def test_dict_with_prompt_passes_through():
    assert render_generate_entry({"prompt": "hello"}, model="any") == "hello"


def test_dict_with_messages_renders_via_template():
    fake = "RENDERED"
    with patch(
        "cray_infra.api.fastapi.chat_completions.render_generate_entry.render_chat_template",
        return_value=fake,
    ) as renderer:
        out = render_generate_entry(
            {"messages": [{"role": "user", "content": "hi"}]},
            model="my-model",
        )
    assert out == fake
    renderer.assert_called_once_with(
        model="my-model",
        messages=[{"role": "user", "content": "hi"}],
        prompt=None,
    )


def test_dict_with_both_prompt_and_messages_raises_400():
    """Reuse the renderer's validation; surface as HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        render_generate_entry(
            {"prompt": "x", "messages": [{"role": "user", "content": "y"}]},
            model="any",
        )
    assert exc_info.value.status_code == 400


def test_dict_with_neither_prompt_nor_messages_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        render_generate_entry({}, model="any")
    assert exc_info.value.status_code == 400


def test_unsupported_type_raises_400():
    """Lists, ints, None — all rejected with 400 not 500."""
    for bogus in [[1, 2], 42, None]:
        with pytest.raises(HTTPException) as exc_info:
            render_generate_entry(bogus, model="any")
        assert exc_info.value.status_code == 400
