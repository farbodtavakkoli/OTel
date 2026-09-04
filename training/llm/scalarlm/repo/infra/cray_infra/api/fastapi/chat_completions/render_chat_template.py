"""
Shared chat template rendering for /v1/chat/completions and /v1/generate.

A request entry carries either `prompt: str` (raw passthrough) or
`messages: list[ChatMessage]` (conversation turns of one request).
This module is the single point that renders the latter into a model
input string via the model's tokenizer chat template, so both
endpoints produce byte-identical inputs to vLLM for equivalent
requests.

See docs/openai-chat-completions-queue.md §4.
"""

import os
from typing import Any, Dict, List, Optional

from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_template_kwargs_for_render,
)

ChatMessage = Dict[str, Any]


# Two-level cache. The request-level `model` name (which may be a
# training-job hash naming a LoRA adapter dir) maps to a *tokenizer
# source* — a path or HF repo id we hand to AutoTokenizer. Many adapter
# hashes resolve to the same base model tokenizer, so the instance
# cache is keyed by source rather than model name to avoid reloading
# the base tokenizer once per adapter.
_source_cache: Dict[str, str] = {}
_tokenizer_cache: Dict[str, Any] = {}


# Filenames that mark a directory as carrying its own tokenizer. A
# training-job dir that has none of these (the LoRA case — only `.pt`
# weights, see ml/cray_megatron/megatron/training_harness.py:32) gets
# resolved to the base model's tokenizer instead.
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
    """
    Render one request entry into a prompt string.

    Exactly one of `messages` or `prompt` must be a non-empty value.
    Empty list / empty string are rejected so silent
    misconfigurations on the caller's side surface as a clear
    ValueError rather than an opaque tokenizer or vLLM error
    downstream.

    `chat_template_kwargs` is restricted to ScalarLM's shared safe
    allowlist before it is passed to the tokenizer. vLLM derives
    `enable_thinking` from an explicit reasoning effort when the caller
    did not already set it; mirror that rule here so admission counts the
    prompt vLLM will actually serve. Tool definitions are likewise part
    of many model chat templates. Raw prompts bypass template rendering,
    so these values are ignored for that input shape.
    """
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
        return prompt  # type: ignore[return-value]

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
    """Return a cached tokenizer for `model`, loading on first use."""
    source = _resolve_tokenizer_source(model)
    cached = _tokenizer_cache.get(source)
    if cached is not None:
        return cached
    tokenizer = _load_tokenizer_from_pretrained(source)
    _tokenizer_cache[source] = tokenizer
    return tokenizer


def count_prompt_tokens(prompt_text: str, *, model: str) -> int:
    """
    Tokenize `prompt_text` with the same tokenizer used to render the
    chat template, returning the token count. Reuses
    `_tokenizer_cache` so the chat handler's pre-admission length
    check doesn't pay a second tokenizer load.
    """
    return len(_load_tokenizer(model).encode(prompt_text))


def _resolve_tokenizer_source(model: str) -> str:
    """
    Map a request-level model name to the source AutoTokenizer should
    load from.

    Trained models in ScalarLM are LoRA adapter directories at
    `{training_job_directory}/{hash}/` carrying only `.pt` weight files
    (see ml/cray_megatron/megatron/training_harness.py:32). They share
    the base model's tokenizer — passing the bare hash to
    `AutoTokenizer.from_pretrained` would make transformers treat it as
    an HF repo id and hit `huggingface.co/<hash>/...` → 401.

    Resolution order:
      1. cached → return cached source.
      2. local dir under training_job_directory with tokenizer files →
         use that path (forward-compat for full-checkpoint saves).
      3. local dir under training_job_directory without tokenizer files
         (the LoRA case) → fall back to base model from config.
      4. anything else → return as-is (lets AutoTokenizer treat it as
         an HF repo id or local path, the legacy path).
    """
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
    """
    Indirection point so tests can patch the network/disk-touching call
    without monkeypatching `transformers.AutoTokenizer` globally.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source)
