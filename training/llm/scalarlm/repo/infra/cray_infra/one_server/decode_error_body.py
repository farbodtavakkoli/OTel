"""
Pull a human-readable error string out of a FastAPI/Starlette
response (or aiohttp-shaped response). Lives in its own module so
unit tests can pin the body-vs-content contract without paying for
the create_generate_worker heavy-import chain (aiohttp, vllm,
transformers).

The original adapter-load error path read `response.content`, but
Starlette's JSONResponse exposes rendered bytes via `.body` —
`content` is constructor-only, no instance attribute. The
attribute-error blanked out every adapter-load failure log so we
never saw the real upstream cause; this helper preserves both
shapes so a future surface change can't silently regress us again.
"""

from typing import Any


def decode_error_body(response: Any) -> str:
    """
    Try `.body` (Starlette/FastAPI), fall back to `.content` (aiohttp /
    older shape), then to `str()`. Decodes bytes as UTF-8, replacing
    invalid sequences rather than raising — the error path itself must
    not throw.
    """
    body = getattr(response, "body", None)
    if body is None:
        body = getattr(response, "content", b"")
    if isinstance(body, (bytes, bytearray)):
        return body.decode("utf-8", errors="replace")
    return str(body)
