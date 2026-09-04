"""
Verify that fill_work_queue stashes correlation_ids from a loaded
batch into the correlation_id_map so update_and_ack can later resolve
the matching ResultRouter futures.

Mixed batches (some entries have a cid, some don't) must not cross-
contaminate: only cids that were present in the JSON should be
stashed, by their per-entry sub-id.
"""

import hashlib
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from cray_infra.api.work_queue import correlation_id_map as cmap
from cray_infra.api.work_queue import get_work_item as gwi


@pytest.fixture(autouse=True)
def _reset():
    cmap._map.clear()
    gwi.in_memory_work_queue = []
    yield
    cmap._map.clear()
    gwi.in_memory_work_queue = []


def _write_batch(tmp_path, requests):
    """Mimic enqueue_coalesced_batch's filename convention."""
    contents = json.dumps(requests).encode("utf-8")
    contents_hash = hashlib.sha256(contents).hexdigest()
    path = os.path.join(tmp_path, f"{contents_hash}.json")
    with open(path, "wb") as fh:
        fh.write(contents)
    return path, contents_hash


def _make_queue(path):
    """A minimal mock of the work_queue interface used by fill_work_queue."""
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=({"path": path}, 7))
    queue.ack = AsyncMock()
    return queue


@pytest.mark.asyncio
async def test_stashes_cid_for_each_entry_with_correlation_id(tmp_path):
    requests = [
        {"prompt": "a", "correlation_id": "cid-a"},
        {"prompt": "b", "correlation_id": "cid-b"},
    ]
    path, group_id = _write_batch(tmp_path, requests)

    # Avoid the response-file-already-exists short-circuit.
    with patch.object(gwi, "group_request_id_to_response_path", return_value="/nonexistent/path"):
        await gwi.fill_work_queue(_make_queue(path))

    sub_id_0 = gwi.make_id(group_id, 0)
    sub_id_1 = gwi.make_id(group_id, 1)

    assert await cmap.pop_correlation_id(sub_id_0) == "cid-a"
    assert await cmap.pop_correlation_id(sub_id_1) == "cid-b"


@pytest.mark.asyncio
async def test_does_not_stash_when_correlation_id_absent(tmp_path):
    """Generate-path entries (no cid) must not produce stash entries."""
    requests = [{"prompt": "no-cid"}]
    path, group_id = _write_batch(tmp_path, requests)

    with patch.object(gwi, "group_request_id_to_response_path", return_value="/nonexistent/path"):
        await gwi.fill_work_queue(_make_queue(path))

    sub_id = gwi.make_id(group_id, 0)
    assert await cmap.pop_correlation_id(sub_id) is None


@pytest.mark.asyncio
async def test_mixed_batch_only_stashes_entries_with_cids(tmp_path):
    requests = [
        {"prompt": "with-cid", "correlation_id": "cid-yes"},
        {"prompt": "without-cid"},
        {"prompt": "with-cid-2", "correlation_id": "cid-also"},
    ]
    path, group_id = _write_batch(tmp_path, requests)

    with patch.object(gwi, "group_request_id_to_response_path", return_value="/nonexistent/path"):
        await gwi.fill_work_queue(_make_queue(path))

    assert await cmap.pop_correlation_id(gwi.make_id(group_id, 0)) == "cid-yes"
    assert await cmap.pop_correlation_id(gwi.make_id(group_id, 1)) is None
    assert await cmap.pop_correlation_id(gwi.make_id(group_id, 2)) == "cid-also"
