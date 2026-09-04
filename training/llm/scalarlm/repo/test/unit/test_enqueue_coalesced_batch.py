"""
Unit tests for enqueue_coalesced_batch.

Contract (see docs/openai-chat-completions-queue.md §6.3):
- A coalesced batch is written to disk as one JSON file under
  `upload_base_path`, named after the SHA-256 of its contents.
- A single SQLite row referencing that file is enqueued via
  `push_into_queue`.
- Identical batches produce the same path (dedup property carries
  over for free).
"""

import hashlib
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch import (
    enqueue_coalesced_batch,
)


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Patch the config-fetcher so the helper writes into a tmp dir."""
    target = tmp_path / "inference_requests"
    target.mkdir()

    fake_config = {"upload_base_path": str(target)}
    with patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.get_config",
        return_value=fake_config,
    ):
        yield target


@pytest.mark.asyncio
async def test_writes_request_file_and_calls_push(upload_dir):
    batch = [
        ({"prompt": "hi", "correlation_id": "c1"}, "c1"),
        ({"prompt": "hello", "correlation_id": "c2"}, "c2"),
    ]

    with patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.push_into_queue",
        new_callable=AsyncMock,
    ) as push:
        await enqueue_coalesced_batch(batch)

    assert push.await_count == 1
    request_count, path = push.await_args.args
    assert request_count == 2
    assert os.path.dirname(path) == str(upload_dir)
    assert os.path.exists(path)

    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk == [
        {"prompt": "hi", "correlation_id": "c1"},
        {"prompt": "hello", "correlation_id": "c2"},
    ]


@pytest.mark.asyncio
async def test_filename_is_sha256_of_contents(upload_dir):
    batch = [({"prompt": "x", "correlation_id": "c1"}, "c1")]

    with patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.push_into_queue",
        new_callable=AsyncMock,
    ) as push:
        await enqueue_coalesced_batch(batch)

    _, path = push.await_args.args
    expected_contents = json.dumps(
        [{"prompt": "x", "correlation_id": "c1"}]
    ).encode()
    expected_hash = hashlib.sha256(expected_contents).hexdigest()

    assert os.path.basename(path) == f"{expected_hash}.json"


@pytest.mark.asyncio
async def test_identical_batches_produce_same_path(upload_dir):
    """Dedup property: same content hashes to same path."""
    batch = [({"prompt": "same", "correlation_id": "c-a"}, "c-a")]

    with patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.push_into_queue",
        new_callable=AsyncMock,
    ) as push:
        await enqueue_coalesced_batch(batch)
        await enqueue_coalesced_batch(batch)

    paths = [call.args[1] for call in push.await_args_list]
    assert paths[0] == paths[1]


@pytest.mark.asyncio
async def test_empty_batch_raises(upload_dir):
    """An empty batch is a programmer error from the coalescer."""
    with pytest.raises(ValueError):
        await enqueue_coalesced_batch([])


@pytest.mark.asyncio
async def test_lazy_creates_upload_dir_when_missing(tmp_path):
    """
    Regression: on a pod that has never received a /v1/generate upload,
    upload_base_path doesn't exist yet (the /v1/generate path lazy-
    creates it via get_request_path). Chat completions would crash with
    FileNotFoundError on every request. Mirror the /v1/generate pattern
    and lazy-create here too.
    """
    target = tmp_path / "does-not-exist-yet"
    assert not target.exists()

    fake_config = {"upload_base_path": str(target)}
    with patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch.push_into_queue",
        new_callable=AsyncMock,
    ) as push:
        await enqueue_coalesced_batch([({"prompt": "x", "correlation_id": "c"}, "c")])

    assert target.is_dir()
    _, path = push.await_args.args
    assert os.path.dirname(path) == str(target)
    assert os.path.exists(path)
