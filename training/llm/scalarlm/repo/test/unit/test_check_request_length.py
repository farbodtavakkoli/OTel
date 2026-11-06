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
    with pytest.raises(RequestTooLongError):
        check_request_length(
            prompt_tokens=300, max_tokens=0, max_model_length=256
        )


def test_treats_max_tokens_none_as_zero():
    check_request_length(
        prompt_tokens=200, max_tokens=None, max_model_length=256
    )
    with pytest.raises(RequestTooLongError):
        check_request_length(
            prompt_tokens=300, max_tokens=None, max_model_length=256
        )


def test_short_circuits_when_max_model_length_is_zero():
    check_request_length(
        prompt_tokens=10_000, max_tokens=10_000, max_model_length=0
    )


def test_short_circuits_on_negative_max_model_length():
    check_request_length(
        prompt_tokens=999, max_tokens=999, max_model_length=-1
    )


def test_error_message_contains_the_numbers_operators_need():
    try:
        check_request_length(
            prompt_tokens=512, max_tokens=128, max_model_length=256
        )
    except RequestTooLongError as exc:
        message = str(exc)
        assert "512" in message
        assert "128" in message
        assert "256" in message
        assert "640" in message
        assert "max_model_length" in message
    else:
        pytest.fail("expected RequestTooLongError")


def test_error_carries_structured_fields_for_inference_browser():
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
