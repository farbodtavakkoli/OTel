from typing import Any

from fastapi import HTTPException

from cray_infra.api.fastapi.chat_completions.render_chat_template import (
    render_chat_template,
)


def render_generate_entry(entry: Any, *, model: str) -> str:
    if isinstance(entry, str):
        return entry

    if not isinstance(entry, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                "prompts entries must be either a string or a dict with "
                f"`prompt` or `messages`; got {type(entry).__name__}"
            ),
        )

    try:
        return render_chat_template(
            model=model,
            messages=entry.get("messages"),
            prompt=entry.get("prompt"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
