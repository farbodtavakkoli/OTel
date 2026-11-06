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
    router = ResultRouter()
    router.resolve("never-registered", {"answer": 0})
    assert router.in_flight_count == 0


@pytest.mark.asyncio
async def test_unregister_removes_without_resolving():
    router = ResultRouter()
    cid = "req-2"
    future = router.register(cid)

    router.unregister(cid)

    assert router.in_flight_count == 0
    assert not future.done()
    router.resolve(cid, {"answer": "late"})
    assert not future.done()


@pytest.mark.asyncio
async def test_duplicate_register_raises():
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
