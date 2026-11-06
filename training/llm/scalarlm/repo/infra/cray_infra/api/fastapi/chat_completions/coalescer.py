import asyncio
from typing import Any, Awaitable, Callable, List, Tuple


BatchEntry = Tuple[Any, str]
FlushCallback = Callable[[List[BatchEntry]], Awaitable[None]]
QueueDepthProvider = Callable[[], int]


class Coalescer:
    def __init__(
        self,
        *,
        packing_factor: int,
        window_seconds: float,
        bypass_threshold: int,
        flush_callback: FlushCallback,
        queue_depth_provider: QueueDepthProvider,
    ) -> None:
        if packing_factor < 1:
            raise ValueError("packing_factor must be >= 1")
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        if bypass_threshold < 0:
            raise ValueError("bypass_threshold must be >= 0")

        self.packing_factor = packing_factor
        self.window_seconds = window_seconds
        self.bypass_threshold = bypass_threshold
        self._flush_callback = flush_callback
        self._queue_depth_provider = queue_depth_provider

        self._lock = asyncio.Lock()
        self._batch: List[BatchEntry] = []
        self._flush_timer: asyncio.Task | None = None

    async def submit(self, request: Any, correlation_id: str) -> None:
        async with self._lock:
            self._batch.append((request, correlation_id))

            if self._queue_depth_provider() < self.bypass_threshold:
                await self._flush_locked()
            elif len(self._batch) >= self.packing_factor:
                await self._flush_locked()
            elif self._flush_timer is None:
                self._flush_timer = asyncio.create_task(
                    self._flush_after(self.window_seconds)
                )

    async def _flush_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._batch:
            self._cancel_flush_timer()
            return

        outgoing, self._batch = self._batch, []
        self._cancel_flush_timer()

        await self._flush_callback(outgoing)

    def _cancel_flush_timer(self) -> None:
        if self._flush_timer is None:
            return
        if not self._flush_timer.done():
            self._flush_timer.cancel()
        self._flush_timer = None
