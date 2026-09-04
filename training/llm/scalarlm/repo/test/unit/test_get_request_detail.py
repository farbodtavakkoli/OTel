"""
Unit tests for get_request_detail.

Contract (see docs/inference-request-browser.md §4.2):
- 64-char hex ids only — anything else is 400 before any FS call.
- Missing request file → 404.
- Missing optional files (response, status) → null in payload, no error.
- Files larger than the display cap → placeholder, not 500.
"""

import hashlib
import json
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.api.fastapi.generate.get_request_detail import (
    MAX_DISPLAY_BYTES,
    get_request_detail,
)


@pytest.fixture
def upload_dir(tmp_path):
    target = tmp_path / "inference_requests"
    target.mkdir()
    fake_config = {"upload_base_path": str(target)}
    with patch(
        "cray_infra.api.work_queue.group_request_id_to_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_status_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_response_path.get_config",
        return_value=fake_config,
    ):
        yield target


def _hex_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.mark.asyncio
async def test_rejects_non_hex_request_id(upload_dir):
    with pytest.raises(HTTPException) as exc:
        await get_request_detail("../../etc/passwd")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_rejects_short_id(upload_dir):
    with pytest.raises(HTTPException) as exc:
        await get_request_detail("abc")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_404_when_request_file_missing(upload_dir):
    rid = _hex_id("nope")
    with pytest.raises(HTTPException) as exc:
        await get_request_detail(rid)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_returns_request_only_when_response_and_status_absent(upload_dir):
    rid = _hex_id("ronly")
    payload = [{"prompt": "hi", "model": "m"}]
    with open(os.path.join(upload_dir, f"{rid}.json"), "w") as f:
        json.dump(payload, f)

    result = await get_request_detail(rid)
    assert result["request_id"] == rid
    assert result["request"] == payload
    assert result["response"] is None
    assert result["status"] is None


@pytest.mark.asyncio
async def test_returns_all_three_files(upload_dir):
    rid = _hex_id("full")
    request = [{"prompt": "hi", "model": "m"}]
    response = {"current_index": 1, "total_requests": 1, "results": {}}
    status = {"status": "completed", "current_index": 1, "total_requests": 1}
    with open(os.path.join(upload_dir, f"{rid}.json"), "w") as f:
        json.dump(request, f)
    with open(os.path.join(upload_dir, f"{rid}_response.json"), "w") as f:
        json.dump(response, f)
    with open(os.path.join(upload_dir, f"{rid}_status.json"), "w") as f:
        json.dump(status, f)

    result = await get_request_detail(rid)
    assert result["request"] == request
    assert result["response"] == response
    assert result["status"] == status
    assert result["request_mtime"] is not None
    assert result["response_mtime"] is not None


@pytest.mark.asyncio
async def test_oversize_request_returns_placeholder(upload_dir, monkeypatch):
    rid = _hex_id("huge")
    path = os.path.join(upload_dir, f"{rid}.json")

    # Write a small file, then lie about its size via a getsize patch
    # so we don't have to actually allocate 5+ MB on disk in CI.
    with open(path, "w") as f:
        json.dump([{"prompt": "tiny"}], f)

    real_getsize = os.path.getsize

    def fake_getsize(p):
        if p == path:
            return MAX_DISPLAY_BYTES + 1
        return real_getsize(p)

    monkeypatch.setattr(
        "cray_infra.api.fastapi.generate.get_request_detail.os.path.getsize",
        fake_getsize,
    )

    result = await get_request_detail(rid)
    assert isinstance(result["request"], dict)
    assert result["request"]["error"] == "too large to display"
    assert result["request"]["size_bytes"] == MAX_DISPLAY_BYTES + 1


@pytest.mark.asyncio
async def test_corrupt_response_returns_unreadable_placeholder(upload_dir):
    rid = _hex_id("corrupt-resp")
    with open(os.path.join(upload_dir, f"{rid}.json"), "w") as f:
        json.dump([{"prompt": "x"}], f)
    # Truncated response file (mid-write race in production).
    with open(os.path.join(upload_dir, f"{rid}_response.json"), "w") as f:
        f.write('{"current_index": 1, "tot')

    result = await get_request_detail(rid)
    assert result["request"] == [{"prompt": "x"}]
    assert isinstance(result["response"], dict)
    assert result["response"]["error"] == "unreadable"
