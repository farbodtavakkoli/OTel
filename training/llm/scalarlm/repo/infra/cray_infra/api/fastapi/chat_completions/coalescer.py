"""
Per-process request coalescer for /v1/chat/completions.

Reduces SQLite write amplification by packing up to `packing_factor`
admitted requests into a single queue row. Under light load (queue
depth below `bypass_threshold`) every submit flushes immediately so
small bursts pay no batching tax. Under sustained load, batches form
naturally and one row holds N requests, giving the queue N× capacity.

See docs/openai-chat-completions-queue.md §6 for the full spec
including the sizing guidance for `packing_factor`.

Three flush triggers — only one fires for any given batch:

  1. SIZE   — `len(batch) >= packing_factor`
  2. TIME   — `window_seconds` elapsed since the *first* submit in the
              current batch. The timer is started on the transition
              empty → non-empty and is *not* reset by subsequent
              submits, so the first arrival's worst-case wait is
              bounded by `window_seconds`.
  3. BYPASS — queue depth was below `bypass_threshold` at submit time;
              flush immediately, no timer.

The flush callback is awaited inline. It is the callback's
responsibility to be fast (e.g. spawn a task for slow I/O and return
quickly) — submits during a flush queue up on the lock, so a slow
callback creates head-of-line blocking on the producer side.
"""

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
        """
        Drain the accumulator and run the callback. Must be called
        with `self._lock` held. Cancels any pending flush timer so a
        size-triggered or bypass-triggered flush doesn't leave a
        timer about to fire on an empty batch.
        """
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
