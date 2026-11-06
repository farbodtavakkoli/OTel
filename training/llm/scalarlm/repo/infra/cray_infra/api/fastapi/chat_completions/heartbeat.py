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
