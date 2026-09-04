"""
Pin the contract for `format_exception`.

Production motivation: `str(AssertionError())` is `""`, and vLLM's
`request_output_to_completion_response` has a bare assert
(serving.py:481). The empty string then propagated through
finish_work and the chat completions wrapper as `error: ""`,
surfacing to the client as a 200 OK with empty `content` — silent
"success" for what was actually a hard failure.
"""

from cray_infra.one_server.format_exception import format_exception


def test_carries_type_when_str_is_empty():
    """Bare AssertionError — the production case that surfaced this."""
    out = format_exception(AssertionError())
    assert out == "AssertionError"
    assert out  # non-empty — the whole point


def test_carries_type_for_keyerror_with_empty_string_arg():
    out = format_exception(KeyError(""))
    assert "KeyError" in out


def test_includes_message_when_present():
    out = format_exception(RuntimeError("vllm crashed"))
    assert "RuntimeError" in out
    assert "vllm crashed" in out


def test_includes_message_for_assertion_with_text():
    out = format_exception(AssertionError("max_tokens is None"))
    assert "AssertionError" in out
    assert "max_tokens is None" in out


def test_handles_nested_exception_chain():
    """A wrapped exception carries the outer type/repr; the chain is
    preserved by exc_info logging elsewhere, not here."""
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        out = format_exception(outer)
    assert "RuntimeError" in out
    assert "outer" in out


def test_never_returns_empty_string():
    """The whole point: any exception, no matter how empty, gets a
    non-empty diagnostic. Otherwise the silent-success bug recurs."""
    for exc in (
        Exception(),
        AssertionError(),
        RuntimeError(),
        ValueError(),
        KeyError(""),
        TypeError(),
    ):
        out = format_exception(exc)
        assert out, f"format_exception returned empty for {type(exc).__name__}"


def test_handles_custom_exception_class():
    class CustomError(Exception):
        pass

    out = format_exception(CustomError())
    assert "CustomError" in out
