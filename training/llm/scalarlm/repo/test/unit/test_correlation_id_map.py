import pytest

from cray_infra.api.work_queue import correlation_id_map as cmap
from cray_infra.api.work_queue.correlation_id_map import (
    pop_correlation_id,
    stash_correlation_id,
)


@pytest.fixture(autouse=True)
def _reset_map():
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
