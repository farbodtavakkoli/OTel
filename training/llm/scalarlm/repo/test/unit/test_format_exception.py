from cray_infra.one_server.format_exception import format_exception


def test_carries_type_when_str_is_empty():
    out = format_exception(AssertionError())
    assert out == "AssertionError"
    assert out


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
