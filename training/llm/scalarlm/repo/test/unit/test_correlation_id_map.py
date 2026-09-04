"""
Unit tests for correlation_id_map.

A tiny in-memory bridge between fill_work_queue (which reads the
request payload off disk and sees `correlation_id` per-request) and
update_and_ack (which only sees the per-request `request_id`). The
cid is stashed on load, popped on completion, and used to resolve
the result router future.
"""

import pytest

from cray_infra.api.work_queue import correlation_id_map as cmap
from cray_infra.api.work_queue.correlation_id_map import (
    pop_correlation_id,
    stash_correlation_id,
)


@pytest.fixture(autouse=True)
def _reset_map():
    """Each test starts with a fresh map; touching the private dict
    directly is the cleanest way and avoids depending on test order."""
    cmap._map.clear()
    yield
    cmap._map.clear()


@pytest.mark.asyncio
async def test_stash_then_pop_returns_value():
    await stash_correlation_id("req-1", "cid-1")
    assert await pop_correlation_id("req-1") == "cid-1"


@pytest.mark.asyncio
async def test_pop_unknown_returns_none():
    assert await pop_correlation_id("never-stashed") is None


@pytest.mark.asyncio
async def test_pop_consumes_entry():
    await stash_correlation_id("req-2", "cid-2")
    assert await pop_correlation_id("req-2") == "cid-2"
    assert await pop_correlation_id("req-2") is None


@pytest.mark.asyncio
async def test_independent_keys_dont_collide():
    await stash_correlation_id("a", "alpha")
    await stash_correlation_id("b", "beta")
    assert await pop_correlation_id("a") == "alpha"
    assert await pop_correlation_id("b") == "beta"
