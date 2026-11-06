from typing import Any


def decode_error_body(response: Any) -> str:
    body = getattr(response, "body", None)
    if body is None:
        body = getattr(response, "content", b"")
    if isinstance(body, (bytes, bytearray)):
        return body.decode("utf-8", errors="replace")
    return str(body)
