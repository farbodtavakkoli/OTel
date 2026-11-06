from types import SimpleNamespace

from cray_infra.one_server.decode_error_body import decode_error_body


def test_decodes_starlette_body_bytes():
    response = SimpleNamespace(body=b'{"detail":"adapter not found"}')
    assert decode_error_body(response) == '{"detail":"adapter not found"}'


def test_falls_back_to_content_attribute():
    response = SimpleNamespace(content=b"older shape")
    assert decode_error_body(response) == "older shape"


def test_prefers_body_over_content_when_both_present():
    response = SimpleNamespace(body=b"new", content=b"old")
    assert decode_error_body(response) == "new"


def test_handles_invalid_utf8_without_crashing():
    response = SimpleNamespace(body=b"\xff\xfe bad bytes")
    out = decode_error_body(response)
    assert "bad bytes" in out


def test_handles_neither_attribute():
    response = SimpleNamespace()
    assert decode_error_body(response) == ""


def test_handles_string_body():
    response = SimpleNamespace(body="already text")
    assert decode_error_body(response) == "already text"


def test_handles_bytearray_body():
    response = SimpleNamespace(body=bytearray(b"mutable"))
    assert decode_error_body(response) == "mutable"


def test_explicit_none_body_falls_through_to_content():
    response = SimpleNamespace(body=None, content=b"content fallback")
    assert decode_error_body(response) == "content fallback"
