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
    for bogus in [[1, 2], 42, None]:
        with pytest.raises(HTTPException) as exc_info:
            render_generate_entry(bogus, model="any")
        assert exc_info.value.status_code == 400
