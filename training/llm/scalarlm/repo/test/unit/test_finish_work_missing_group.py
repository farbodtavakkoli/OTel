"""
Unit tests for the finish_work path when the in-memory group state
has been wiped (uvicorn reload mid-batch, already-finalized group).

Before the fix, get_unfinished_result dereferenced a None group and
raised TypeError("'NoneType' object is not subscriptable"), bubbling
up as a 500 on /v1/generate/finish_work. The worker's retry-on-error
loop then hammered the API.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from cray_infra.api.work_queue.get_unfinished_result import get_unfinished_result
from cray_infra.api.work_queue import update_and_ack as uaa_mod


def _drain(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---- get_unfinished_result ----------------------------------------------


def test_returns_none_when_group_absent():
    # Mirrors the post-reload state: no group in memory for this hash.
    with patch(
        "cray_infra.api.work_queue.get_unfinished_result.get_in_memory_results",
        new=AsyncMock(return_value=None),
    ):
        result = _drain(get_unfinished_result("abcd_000000000"))
    assert result is None


def test_creates_placeholder_on_first_completion():
    group = {"results": {}, "current_index": 0, "total_requests": 3}
    with patch(
        "cray_infra.api.work_queue.get_unfinished_result.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ):
        result = _drain(get_unfinished_result("abcd_000000000"))
    assert result == {"is_acked": False}
    assert group["results"]["abcd_000000000"] is result


def test_returns_existing_entry_for_same_request_id():
    group = {
        "results": {"abcd_000000000": {"is_acked": False, "marker": "x"}},
        "current_index": 0,
        "total_requests": 1,
    }
    with patch(
        "cray_infra.api.work_queue.get_unfinished_result.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ):
        result = _drain(get_unfinished_result("abcd_000000000"))
    assert result["marker"] == "x"


# ---- update_and_ack: crash-free when group is gone -----------------------


def test_update_and_ack_drops_silently_when_group_absent():
    queue = AsyncMock()
    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=None),
    ):
        # Must not raise — worker needs a 200 so it doesn't retry.
        _drain(uaa_mod.update_and_ack(queue, "abcd_000000000", {"response": "hi"}))
    queue.ack.assert_not_called()


def test_update_and_ack_handles_first_completion_placeholder():
    # After get_unfinished_result created `{"is_acked": False}`, we
    # call update_and_ack with the real item. The guard must not
    # treat the placeholder as already-acked.
    group = {
        "results": {"abcd_000000000": {"is_acked": False}},
        "current_index": 0,
        "total_requests": 2,
        "work_queue_id": 42,
    }
    queue = AsyncMock()
    with patch(
        "cray_infra.api.work_queue.update_and_ack.get_in_memory_results",
        new=AsyncMock(return_value=group),
    ):
        _drain(
            uaa_mod.update_and_ack(
                queue, "abcd_000000000", {"response": "hello"}
            )
        )
    # current_index advanced because is_acked was False.
    assert group["current_index"] == 1
    assert group["results"]["abcd_000000000"]["response"] == "hello"
    assert group["results"]["abcd_000000000"]["is_acked"] is True
    # Group isn't finalized yet (1 < 2), so we shouldn't have acked.
    queue.ack.assert_not_called()
