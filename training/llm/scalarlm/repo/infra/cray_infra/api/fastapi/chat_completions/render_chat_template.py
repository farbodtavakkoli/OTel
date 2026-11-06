import os
from typing import Any, Dict, List, Optional

from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_template_kwargs_for_render,
)

ChatMessage = Dict[str, Any]


_source_cache: Dict[str, str] = {}
_tokenizer_cache: Dict[str, Any] = {}


_TOKENIZER_MARKER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
)


def render_chat_template(
    *,
    model: str,
    messages: Optional[List[ChatMessage]],
    prompt: Optional[str],
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    has_messages = bool(messages)
    has_prompt = bool(prompt)

    if has_messages == has_prompt:
        raise ValueError(
            "render_chat_template requires exactly one of `messages` or "
            "`prompt` to be set; got "
            f"messages={'present' if has_messages else 'absent'}, "
            f"prompt={'present' if has_prompt else 'absent'}"
        )

    if has_prompt:
        return prompt

    tokenizer = _load_tokenizer(model)
    template_kwargs = _chat_template_kwargs_for_render(
        chat_template_kwargs, reasoning_effort
    )

    render_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **template_kwargs,
    }
    if tools is not None:
        render_kwargs["tools"] = tools

    return tokenizer.apply_chat_template(messages, **render_kwargs)


def _load_tokenizer(model: str) -> Any:
    source = _resolve_tokenizer_source(model)
    cached = _tokenizer_cache.get(source)
    if cached is not None:
        return cached
    tokenizer = _load_tokenizer_from_pretrained(source)
    _tokenizer_cache[source] = tokenizer
    return tokenizer


def count_prompt_tokens(prompt_text: str, *, model: str) -> int:
    return len(_load_tokenizer(model).encode(prompt_text))


def _resolve_tokenizer_source(model: str) -> str:
    cached = _source_cache.get(model)
    if cached is not None:
        return cached

    from cray_infra.util.get_config import get_config

    config = get_config()
    base_model = config.get("model", "")
    training_dir = config.get("training_job_directory", "/app/cray/jobs")

    if model and model != base_model:
        candidate_path = os.path.join(training_dir, model)
        if os.path.isdir(candidate_path):
            source = (
                candidate_path if _has_tokenizer_files(candidate_path) else base_model
            )
            _source_cache[model] = source
            return source

    _source_cache[model] = model
    return model


def _has_tokenizer_files(directory: str) -> bool:
    return any(
        os.path.isfile(os.path.join(directory, name))
        for name in _TOKENIZER_MARKER_FILES
    )


def _load_tokenizer_from_pretrained(source: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source)
