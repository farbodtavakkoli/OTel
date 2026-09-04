"""
Integration of update_and_ack with the chat-completions ResultRouter.

When a per-prompt completion lands in update_and_ack and the
corresponding request_id has a stashed correlation_id (set by
fill_work_queue when the chat-completions JSON file was loaded), the
router's matching Future must be resolved. Requests without a cid
(the /v1/generate path) must not trigger a resolve call — the
behavior is endpoint-agnostic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cray_infra.api.fastapi.chat_completions.result_router import ResultRouter
from cray_infra.api.work_queue import correlation_id_map as cmap
from cray_infra.api.work_queue import update_and_ack as uaa_mod


@pytest.fixture(autouse=True)
def _reset_correlation_map():
    cmap._map.clear()
    yield
    cmap._map.clear()


@pytest.fixture
def fresh_router():
    return ResultRouter()


def _group_state(*, total: int = 1, work_queue_id: int = 42):
    return {
        "results": {},
        "current_index": 0,
        "total_requests": total,
        "work_queue_id": work_queue_id,
    }


@pytest.mark.asyncio
async def test_resolves_router_future_when_correlation_id_is_stashed(fresh_router):
    request_id = "abcd_000000000"
    correlation_id = "cid-xyz"

    future = fresh_router.register(correlation_id)
    await cmap.stash_correlation_id(request_id, correlation_id)

    queue = AsyncMock()
    group = _group_state(total=2)  # not finalized yet — keeps ack out of the test

    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ), patch(
        "cray_infra.api.work_queue.update_and_ack.get_result_router",
        return_value=fresh_router,
    ):
        await uaa_mod.update_and_ack(queue, request_id, {"response": "hi"})

    assert future.done()
    assert future.result()["response"] == "hi"


@pytest.mark.asyncio
async def test_no_resolve_when_no_correlation_id_stashed(fresh_router):
    """Generate-path requests don't carry a cid; resolve must not be called."""
    request_id = "wxyz_000000000"
    queue = AsyncMock()
    group = _group_state(total=2)

    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ), patch(
        "cray_infra.api.work_queue.update_and_ack.get_result_router",
        return_value=fresh_router,
    ):
        await uaa_mod.update_and_ack(queue, request_id, {"response": "x"})

    assert fresh_router.in_flight_count == 0


@pytest.mark.asyncio
async def test_correlation_id_is_consumed_on_first_resolve(fresh_router):
    """A second update_and_ack on the same id must not double-resolve."""
    request_id = "abcd_000000000"
    correlation_id = "cid-once"

    fresh_router.register(correlation_id)
    await cmap.stash_correlation_id(request_id, correlation_id)

    queue = AsyncMock()
    group = _group_state(total=3)

    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ), patch(
        "cray_infra.api.work_queue.update_and_ack.get_result_router",
        return_value=fresh_router,
    ):
        await uaa_mod.update_and_ack(queue, request_id, {"response": "first"})
        await uaa_mod.update_and_ack(queue, request_id, {"response": "second"})

    # Router cleared on first resolve; second call had nothing to resolve.
    assert fresh_router.in_flight_count == 0


@pytest.mark.asyncio
async def test_resolve_does_not_block_existing_disconnect_handling(fresh_router):
    """
    Client disconnected (cid unregistered) → resolve is silent no-op,
    update_and_ack still completes its existing work.
    """
    request_id = "abcd_000000000"
    correlation_id = "cid-gone"

    fresh_router.register(correlation_id)
    fresh_router.unregister(correlation_id)  # client disconnected
    await cmap.stash_correlation_id(request_id, correlation_id)

    queue = AsyncMock()
    group = _group_state(total=2)

    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ), patch(
        "cray_infra.api.work_queue.update_and_ack.get_result_router",
        return_value=fresh_router,
    ):
        # Must not raise, must update group state normally.
        await uaa_mod.update_and_ack(queue, request_id, {"response": "ignored"})

    assert group["current_index"] == 1
    assert group["results"][request_id]["is_acked"] is True
