import asyncio
import time

import pytest

from cray_infra.api.fastapi.chat_completions.coalescer import Coalescer


def _make_coalescer(
    *,
    packing_factor: int = 10,
    window_seconds: float = 0.05,
    bypass_threshold: int = 10,
    queue_depth: int = 999,
) -> tuple[Coalescer, list[list]]:
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
    coalescer, flushes = _make_coalescer(bypass_threshold=10, queue_depth=0)

    await coalescer.submit("req-1", "cid-1")

    assert flushes == [[("req-1", "cid-1")]]


@pytest.mark.asyncio
async def test_size_trigger_flushes_at_packing_factor():
    coalescer, flushes = _make_coalescer(packing_factor=3, queue_depth=999)

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")
    assert flushes == []
    await coalescer.submit("c", "3")
    assert flushes == [[("a", "1"), ("b", "2"), ("c", "3")]]


@pytest.mark.asyncio
async def test_time_trigger_flushes_after_window():
    coalescer, flushes = _make_coalescer(
        packing_factor=10, window_seconds=0.05, queue_depth=999
    )

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")
    assert flushes == []

    await asyncio.sleep(0.08)

    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_size_trigger_cancels_pending_timer():
    coalescer, flushes = _make_coalescer(
        packing_factor=2, window_seconds=0.05, queue_depth=999
    )

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")

    assert flushes == [[("a", "1"), ("b", "2")]]

    await asyncio.sleep(0.1)

    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_subsequent_batch_after_flush_starts_new_timer():
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
    coalescer, flushes = _make_coalescer(
        packing_factor=10, window_seconds=0.05, queue_depth=999
    )

    t0 = time.monotonic()
    await coalescer.submit("a", "1")
    await asyncio.sleep(0.02)
    await coalescer.submit("b", "2")
    await asyncio.sleep(0.02)
    await coalescer.submit("c", "3")

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
    coalescer, flushes = _make_coalescer(packing_factor=2, queue_depth=999)

    await coalescer.submit("a", "1")
    await coalescer.submit("b", "2")
    await asyncio.sleep(0.08)

    assert flushes == [[("a", "1"), ("b", "2")]]


@pytest.mark.asyncio
async def test_concurrent_submits_serialize_through_lock():
    coalescer, flushes = _make_coalescer(packing_factor=5, queue_depth=999)

    await asyncio.gather(*(coalescer.submit(f"req-{i}", f"cid-{i}") for i in range(15)))

    assert sum(len(b) for b in flushes) == 15
    assert len(flushes) == 3
    flat = [entry for batch in flushes for entry in batch]
    assert {cid for _, cid in flat} == {f"cid-{i}" for i in range(15)}
