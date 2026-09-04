"""
Unit tests for Coalescer.

Contract (see docs/openai-chat-completions-queue.md §6):
- packing_factor: max requests per batch.
- window_seconds: max wait from the first request before flushing.
- bypass_threshold: when the upstream queue depth is below this,
  flush immediately on every submit (no batching tax under low load).

Three flush triggers, only one of which fires for any given batch:
  size           — accumulator hit packing_factor
  time           — window_seconds elapsed since first arrival
  bypass         — queue depth was below bypass_threshold at submit time

The callback receives the list of `(request, correlation_id)` tuples
that were in the batch, in submission order. The test ensures all
three triggers work in isolation and that they don't interfere when
they could fire in the same window.
"""

import asyncio
import time

import pytest

from cray_infra.api.fastapi.chat_completions.coalescer import Coalescer


def _make_coalescer(
    *,
    packing_factor: int = 10,
    window_seconds: float = 0.05,
    bypass_threshold: int = 10,
    queue_depth: int = 999,  # default high so bypass doesn't fire
) -> tuple[Coalescer, list[list]]:
    """Build a coalescer with a list-collecting callback. Returns (coalescer, flushes)."""
    flushes: list[list] = []

    async def collect(batch):
        flushes.append(list(batch))

    coalescer = Coalescer(
        packing_factor=packing_factor,
        window_seconds=window_seconds,
        bypass_threshold=bypass_threshold,
        flush_callback=collect,
        queue_depth_provider=lambda: queue_depth,
    )
    return coalescer, flushes


@pytest.mark.asyncio
async def test_bypass_under_threshold_flushes_immediately():
    """Low queue depth → submit returns after the flush has run."""
    coalescer, flushes = _make_coalescer(bypass_threshold=10, queue_depth=0)

    await coalescer.submit("req-1", "cid-1")

    assert flushes == [[("req-1", "cid-1")]]


@pytest.mark.asyncio
async def test_size_trigger_flushes_at_packing_factor():
    """Above the bypass threshold, packing_factor submits flush as one batch."""
    coalescer, flushes = _make_coalescer(packing_factor=3, queue_depth=999)

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")
    assert flushes == []  # not yet
    await coalescer.submit("c", "3")
    assert flushes == [[("a", "1"), ("b", "2"), ("c", "3")]]


@pytest.mark.asyncio
async def test_time_trigger_flushes_after_window():
    """Fewer-than-packing_factor submits flush when the window expires."""
    coalescer, flushes = _make_coalescer(
        packing_factor=10, window_seconds=0.05, queue_depth=999
    )

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")
    assert flushes == []

    # Wait past the window and then one event-loop tick for the timer task.
    await asyncio.sleep(0.08)

    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_size_trigger_cancels_pending_timer():
    """
    Batch fills before the timer fires → timer must cancel so it
    doesn't fire later and re-flush an empty (or wrong) batch.
    """
    coalescer, flushes = _make_coalescer(
        packing_factor=2, window_seconds=0.05, queue_depth=999
    )

    await coalescer.submit("a", "1")  # starts the timer
    await coalescer.submit("b", "2")  # size flush; cancels timer

    assert flushes == [[("a", "1"), ("b", "2")]]

    # Wait past the original window. No second flush should happen.
    await asyncio.sleep(0.1)

    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_subsequent_batch_after_flush_starts_new_timer():
    """Two batches in sequence; both flush correctly and independently."""
    coalescer, flushes = _make_coalescer(
        packing_factor=2, window_seconds=0.05, queue_depth=999
    )

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")

    await coalescer.submit("c", "3")
    await coalescer.submit("d", "4")

    assert flushes == [
        [("a", "1"), ("b", "2")],
        [("c", "3"), ("d", "4")],
    ]


@pytest.mark.asyncio
async def test_window_clock_starts_at_first_arrival_not_each_submit():
    """
    Three submits over 60ms with a 50ms window: the timer should fire
    based on the FIRST submit's clock, not be reset by later submits.
    """
    coalescer, flushes = _make_coalescer(
        packing_factor=10, window_seconds=0.05, queue_depth=999
    )

    t0 = time.monotonic()
    await coalescer.submit("a", "1")
    await asyncio.sleep(0.02)
    await coalescer.submit("b", "2")
    await asyncio.sleep(0.02)
    await coalescer.submit("c", "3")

    # Wait for the timer to fire (it was started at t0 + ~0.05).
    while not flushes and time.monotonic() - t0 < 0.5:
        await asyncio.sleep(0.005)

    assert flushes == [[("a", "1"), ("b", "2"), ("c", "3")]]
    elapsed = time.monotonic() - t0
    assert elapsed < 0.15, (
        f"flush fired at {elapsed:.3f}s — should have been near 0.05s "
        "from first submit, not reset by later submits"
    )


@pytest.mark.asyncio
async def test_flush_with_empty_batch_is_noop():
    """Edge case: timer fires after the batch was already drained."""
    coalescer, flushes = _make_coalescer(packing_factor=2, queue_depth=999)

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")  # size flush
    await asyncio.sleep(0.08)  # past the original window

    # No spurious second flush.
    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_concurrent_submits_serialize_through_lock():
    """
    Many concurrent submits land in batches whose contents and order
    are well-defined (no lost or duplicated entries).
    """
    coalescer, flushes = _make_coalescer(packing_factor=5, queue_depth=999)

    await asyncio.gather(*(coalescer.submit(f"req-{i}", f"cid-{i}") for i in range(15)))

    # 15 submits / packing_factor 5 = 3 full batches.
    assert sum(len(b) for b in flushes) == 15
    assert len(flushes) == 3
    flat = [entry for batch in flushes for entry in batch]
    assert {cid for _, cid in flat} == {f"cid-{i}" for i in range(15)}
