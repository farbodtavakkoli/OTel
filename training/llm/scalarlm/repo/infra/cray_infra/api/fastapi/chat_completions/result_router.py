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
