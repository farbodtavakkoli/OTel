"""
Unit tests for ResultRouter.

Contract (see docs/openai-chat-completions-queue.md §8):
- register(cid) creates a Future under that correlation id.
- resolve(cid, result) sets the Future's result and removes the entry.
- unregister(cid) removes the entry without resolving (for client disconnect).
- in_flight_count gauges currently-registered ids.
- resolve on an unknown cid is a silent no-op (post-disconnect race).
- register on a duplicate cid is a programmer error (raises).
"""

import asyncio

import pytest

from cray_infra.api.fastapi.chat_completions.result_router import ResultRouter


@pytest.mark.asyncio
async def test_register_returns_future_pending_until_resolve():
    router = ResultRouter()
    cid = "req-1"

    future = router.register(cid)
    assert not future.done()
    assert router.in_flight_count == 1

    router.resolve(cid, {"answer": 42})
    assert future.done()
    assert future.result() == {"answer": 42}
    assert router.in_flight_count == 0


@pytest.mark.asyncio
async def test_resolve_unknown_cid_is_silent_no_op():
    """
    Worker may produce a result for a cid whose handler already
    disconnected. That's normal; resolve must not raise.
    """
    router = ResultRouter()
    router.resolve("never-registered", {"answer": 0})
    assert router.in_flight_count == 0


@pytest.mark.asyncio
async def test_unregister_removes_without_resolving():
    """Client-disconnect path: drop the future, don't set a result."""
    router = ResultRouter()
    cid = "req-2"
    future = router.register(cid)

    router.unregister(cid)

    assert router.in_flight_count == 0
    assert not future.done()
    # A subsequent resolve under that cid must not raise either.
    router.resolve(cid, {"answer": "late"})
    assert not future.done()


@pytest.mark.asyncio
async def test_duplicate_register_raises():
    """A repeated cid is a bug in the caller; surface it loudly."""
    router = ResultRouter()
    router.register("dup")
    with pytest.raises(KeyError):
        router.register("dup")


@pytest.mark.asyncio
async def test_in_flight_count_tracks_concurrent_registrations():
    router = ResultRouter()
    futures = [router.register(f"req-{i}") for i in range(50)]
    assert router.in_flight_count == 50

    for i, fut in enumerate(futures[:20]):
        router.resolve(f"req-{i}", i)
    assert router.in_flight_count == 30

    for i in range(20, 35):
        router.unregister(f"req-{i}")
    assert router.in_flight_count == 15


@pytest.mark.asyncio
async def test_handler_workflow_resolve_unblocks_awaiter():
    """
    End-to-end shape of the production flow: handler awaits the future,
    a separate task simulates the worker calling resolve.
    """
    router = ResultRouter()
    cid = "req-await"
    future = router.register(cid)

    async def simulate_worker():
        await asyncio.sleep(0.02)
        router.resolve(cid, {"text": "hello"})

    asyncio.create_task(simulate_worker())
    result = await asyncio.wait_for(future, timeout=1.0)

    assert result == {"text": "hello"}
    assert router.in_flight_count == 0
