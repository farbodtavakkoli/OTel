"""
Pin the contract for `check_request_length`.

Production motivation: vLLM doesn't reject prompts that exceed the
per-request KV budget — the scheduler queues them and the request
stalls forever. The /inference browser shows them as `in_progress`
indefinitely; clients see whitespace heartbeats followed by a
client-side timeout. This pre-admission check fails fast with HTTP
400 instead.
"""

import pytest

from cray_infra.api.fastapi.chat_completions.check_request_length import (
    RequestTooLongError,
    check_request_length,
)


def test_passes_when_well_under_threshold():
    check_request_length(
        prompt_tokens=10, max_tokens=20, max_model_length=256
    )


def test_passes_at_exact_threshold():
    """prompt + max_tokens == max_model_length is fine; vLLM can fit it."""
    check_request_length(
        prompt_tokens=200, max_tokens=56, max_model_length=256
    )


def test_rejects_one_over_threshold():
    with pytest.raises(RequestTooLongError) as exc:
        check_request_length(
            prompt_tokens=200, max_tokens=57, max_model_length=256
        )
    err = exc.value
    assert err.prompt_tokens == 200
    assert err.max_tokens == 57
    assert err.total == 257
    assert err.max_model_length == 256


def test_rejects_when_prompt_alone_exceeds_threshold():
    """A prompt longer than the entire context window is unservable
    even with max_tokens=0."""
    with pytest.raises(RequestTooLongError):
        check_request_length(
            prompt_tokens=300, max_tokens=0, max_model_length=256
        )


def test_treats_max_tokens_none_as_zero():
    """The OpenAI SDK lets clients omit max_tokens; the prompt-side
    bound is still what matters for admission."""
    check_request_length(
        prompt_tokens=200, max_tokens=None, max_model_length=256
    )
    with pytest.raises(RequestTooLongError):
        check_request_length(
            prompt_tokens=300, max_tokens=None, max_model_length=256
        )


def test_short_circuits_when_max_model_length_is_zero():
    """Operator hasn't configured a cap → don't reject anything.
    Silent passthrough beats spurious 400s on misconfigured pods."""
    check_request_length(
        prompt_tokens=10_000, max_tokens=10_000, max_model_length=0
    )


def test_short_circuits_on_negative_max_model_length():
    """Defensive: negative cap doesn't make sense, but never reject."""
    check_request_length(
        prompt_tokens=999, max_tokens=999, max_model_length=-1
    )


def test_error_message_contains_the_numbers_operators_need():
    """The detail string ends up in the API response and the
    inference browser status row — operators read it directly."""
    try:
        check_request_length(
            prompt_tokens=512, max_tokens=128, max_model_length=256
        )
    except RequestTooLongError as exc:
        message = str(exc)
        assert "512" in message
        assert "128" in message
        assert "256" in message
        assert "640" in message  # the total
        assert "max_model_length" in message
    else:
        pytest.fail("expected RequestTooLongError")


def test_error_carries_structured_fields_for_inference_browser():
    """The browser pulls these out of the status row to render a
    human-friendly reason. They must not get lost in str()."""
    try:
        check_request_length(
            prompt_tokens=300, max_tokens=200, max_model_length=256
        )
    except RequestTooLongError as exc:
        assert exc.prompt_tokens == 300
        assert exc.max_tokens == 200
        assert exc.total == 500
        assert exc.max_model_length == 256
    else:
        pytest.fail("expected RequestTooLongError")
