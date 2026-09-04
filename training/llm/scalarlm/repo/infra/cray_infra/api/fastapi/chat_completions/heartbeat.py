"""
Whitespace-heartbeat body generator for chunked-JSON responses.

This is the transport piece described in
`docs/openai-chat-completions-queue.md` §7. It lets a long-running
`/v1/chat/completions` handler keep its HTTP connection alive across
queue + processing time without requiring the OpenAI client to use
`stream=True`, raise `timeout`, or set `max_retries`.

Mechanism: while the work future hasn't resolved, emit one whitespace
byte every `heartbeat_interval_seconds`. Whitespace is RFC 7159 §2
valid JSON, so the OpenAI SDK's response parser (and any other JSON
parser) discards the leading bytes and only sees the final body. Each
byte resets httpx's `read` timeout and any middlebox idle timer in
the path.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Callable


HEARTBEAT_BYTE = b" "
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 4.0


async def stream_with_heartbeat(
    work_future: asyncio.Future,
    *,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    encode_body: Callable[[Any], bytes] | None = None,
) -> AsyncIterator[bytes]:
    """
    Yield whitespace bytes every `heartbeat_interval_seconds` until
    `work_future` resolves, then yield its result encoded as JSON.

    The future is `asyncio.shield`-wrapped so the heartbeat-tick
    timeout cancels only the waiter, never the underlying work.

    `encode_body` lets a caller transform the resolved value before it
    hits the wire — used by the chat completions handler to wrap the
    worker's flat result dict into the OpenAI ChatCompletion shape
    without coupling this transport to that vocabulary.
    """
    encoder = encode_body or _encode_json_body
    while True:
        try:
            result = await asyncio.wait_for(
                asyncio.shield(work_future),
                timeout=heartbeat_interval_seconds,
            )
            break
        except asyncio.TimeoutError:
            yield HEARTBEAT_BYTE

    yield encoder(result)


def _encode_json_body(value: Any) -> bytes:
    return json.dumps(value).encode("utf-8")
