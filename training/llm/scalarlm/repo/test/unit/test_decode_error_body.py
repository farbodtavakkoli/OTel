"""
Pin the contract for decode_error_body.

Production bug this guards against: the original adapter-load error
path read `JSONResponse.content`, but Starlette's JSONResponse
exposes rendered bytes via `.body` — `content` is only a constructor
parameter, no instance attribute. Every retry surfaced
`'JSONResponse' object has no attribute 'content'` instead of the
real upstream error, hiding why adapter loads were failing and making
the queue look like KV-cache exhaustion.
"""

from types import SimpleNamespace

from cray_infra.one_server.decode_error_body import decode_error_body


def test_decodes_starlette_body_bytes():
    """The real path: JSONResponse.body holds rendered JSON bytes."""
    response = SimpleNamespace(body=b'{"detail":"adapter not found"}')
    assert decode_error_body(response) == '{"detail":"adapter not found"}'


def test_falls_back_to_content_attribute():
    """
    Older aiohttp-shaped responses use `.content`. Don't lose them
    just because we now prefer `.body`.
    """
    response = SimpleNamespace(content=b"older shape")
    assert decode_error_body(response) == "older shape"


def test_prefers_body_over_content_when_both_present():
    """If both attributes exist, the modern Starlette `.body` wins."""
    response = SimpleNamespace(body=b"new", content=b"old")
    assert decode_error_body(response) == "new"


def test_handles_invalid_utf8_without_crashing():
    """Sanitised so a non-UTF8 payload never bubbles a UnicodeDecodeError
    out of the error path itself."""
    response = SimpleNamespace(body=b"\xff\xfe bad bytes")
    out = decode_error_body(response)
    assert "bad bytes" in out


def test_handles_neither_attribute():
    """Empty response object → empty string, not AttributeError."""
    response = SimpleNamespace()
    assert decode_error_body(response) == ""


def test_handles_string_body():
    """A pre-decoded string passes through via str()."""
    response = SimpleNamespace(body="already text")
    assert decode_error_body(response) == "already text"


def test_handles_bytearray_body():
    """Bytearray is a separate type from bytes; both should decode."""
    response = SimpleNamespace(body=bytearray(b"mutable"))
    assert decode_error_body(response) == "mutable"


def test_explicit_none_body_falls_through_to_content():
    """`getattr(..., 'body', None)` returns None if `.body` is set to
    None, not just absent. Make sure we still fall through."""
    response = SimpleNamespace(body=None, content=b"content fallback")
    assert decode_error_body(response) == "content fallback"
