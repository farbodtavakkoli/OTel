"""
Per-prompt fan-out from the worker back to waiting handlers.

The coalescer assigns each chat request a correlation id (UUID) and
registers a Future under that id with the router; the FastAPI handler
awaits that Future. When the worker reports completion via
`update_and_ack` (`infra/cray_infra/api/fastapi/generate/update_and_ack.py`),
that path calls `router.resolve(cid, result)` and the handler's
`asyncio.wait_for` unblocks.

Designed to tolerate the post-disconnect race: a handler that closed
its connection has already called `unregister(cid)`, so a late
`resolve(cid, ...)` from the worker silently no-ops instead of raising.

Contract:
- register(cid)   → Future. Raises KeyError if cid already exists.
- resolve(cid, x) → set the Future and drop the mapping. Silent if cid
                    isn't registered (client disconnect raced the
                    worker).
- unregister(cid) → drop the mapping without resolving. For the
                    client-disconnect path; the future is left pending
                    and falls out of scope with the cancelled task.
- in_flight_count → current registration count. Used by the admission
                    controller for backpressure decisions and surfaced
                    as the `chat_in_flight` gauge (§13).

Single per-process singleton expected; no locking is needed because
all callers are FastAPI/asyncio coroutines on a single event loop.
"""

import asyncio
from typing import Any, Dict


class ResultRouter:
    def __init__(self) -> None:
        self._futures: Dict[str, asyncio.Future] = {}

    def register(self, correlation_id: str) -> asyncio.Future:
        if correlation_id in self._futures:
            raise KeyError(
                f"correlation_id {correlation_id!r} is already registered"
            )
        # Use the running loop. Production callers (FastAPI handlers,
        # update_and_ack) are always inside one; if a caller isn't,
        # surfacing the failure here is more useful than the silent
        # cross-loop bugs the deprecated `get_event_loop()` produced.
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._futures[correlation_id] = future
        return future

    def resolve(self, correlation_id: str, result: Any) -> None:
        future = self._futures.pop(correlation_id, None)
        if future is None:
            return
        if not future.done():
            future.set_result(result)

    def unregister(self, correlation_id: str) -> None:
        self._futures.pop(correlation_id, None)

    @property
    def in_flight_count(self) -> int:
        return len(self._futures)


_singleton: ResultRouter | None = None


def get_result_router() -> ResultRouter:
    global _singleton
    if _singleton is None:
        _singleton = ResultRouter()
    return _singleton
