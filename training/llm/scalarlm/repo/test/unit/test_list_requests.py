"""
Unit tests for list_requests.

Contract (see docs/inference-request-browser.md §4.1):
- Returns request files only — `_response.json` and `_status.json`
  are never first-class rows.
- Sorted newest first; mtime cursor pagination is exclusive.
- Tolerates missing/corrupt files: surfaces the row with placeholders
  rather than 500ing.
"""

import hashlib
import json
import os
import time
from unittest.mock import patch

import pytest

from cray_infra.api.fastapi.generate.list_requests import list_requests


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    target = tmp_path / "inference_requests"
    target.mkdir()
    fake_config = {"upload_base_path": str(target)}
    with patch(
        "cray_infra.api.fastapi.generate.list_requests.get_config",
        return_value=fake_config,
    ):
        # The status/response path helpers also pull from get_config().
        with patch(
            "cray_infra.api.work_queue.group_request_id_to_status_path.get_config",
            return_value=fake_config,
        ), patch(
            "cray_infra.api.work_queue.group_request_id_to_response_path.get_config",
            return_value=fake_config,
        ):
            yield target


def _hex_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _write_request(upload_dir, seed: str, prompt: str = "hello world", *, model="m", request_count=1, mtime=None):
    rid = _hex_id(seed)
    payload = [{"prompt": prompt, "model": model, "request_type": "generate"}]
    payload.extend([{"prompt": "x", "model": model, "request_type": "generate"}] * (request_count - 1))
    path = os.path.join(upload_dir, f"{rid}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return rid, path


def _write_status(upload_dir, rid, status="completed", completed_at=None):
    path = os.path.join(upload_dir, f"{rid}_status.json")
    with open(path, "w") as f:
        json.dump(
            {
                "status": status,
                "current_index": 1,
                "total_requests": 1,
                "work_queue_id": 1,
                **({"completed_at": completed_at} if completed_at else {}),
            },
            f,
        )


def _write_response(upload_dir, rid):
    path = os.path.join(upload_dir, f"{rid}_response.json")
    with open(path, "w") as f:
        json.dump({"current_index": 1, "total_requests": 1, "results": {}}, f)


@pytest.mark.asyncio
async def test_returns_empty_when_dir_missing(tmp_path, monkeypatch):
    fake_config = {"upload_base_path": str(tmp_path / "does_not_exist")}
    with patch(
        "cray_infra.api.fastapi.generate.list_requests.get_config",
        return_value=fake_config,
    ):
        result = await list_requests(cursor=None, limit=50)
    assert result == {"rows": [], "next_cursor": None, "has_more": False}


@pytest.mark.asyncio
async def test_lists_request_files_newest_first(upload_dir):
    base = time.time()
    _write_request(upload_dir, "a", "first", mtime=base - 30)
    _write_request(upload_dir, "b", "second", mtime=base - 20)
    rid_c, _ = _write_request(upload_dir, "c", "third", mtime=base - 10)
    _write_status(upload_dir, rid_c, status="completed", completed_at=base - 9)
    _write_response(upload_dir, rid_c)

    result = await list_requests(cursor=None, limit=50)

    assert [r["prompt_preview"] for r in result["rows"]] == ["third", "second", "first"]
    assert result["rows"][0]["status"] == "completed"
    assert result["rows"][0]["has_response"] is True
    assert result["rows"][1]["status"] == "unknown"
    assert result["rows"][1]["has_response"] is False
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_pagination_cursor_excludes_previous_page(upload_dir):
    base = time.time()
    for i in range(5):
        _write_request(upload_dir, f"r{i}", f"prompt{i}", mtime=base - i)

    page1 = await list_requests(cursor=None, limit=2)
    assert len(page1["rows"]) == 2
    assert page1["has_more"] is True
    assert page1["next_cursor"] is not None

    page2 = await list_requests(cursor=page1["next_cursor"], limit=2)
    assert len(page2["rows"]) == 2

    page3 = await list_requests(cursor=page2["next_cursor"], limit=2)
    assert len(page3["rows"]) == 1
    assert page3["has_more"] is False
    assert page3["next_cursor"] is None

    # No id appears twice across the three pages.
    seen = [r["request_id"] for page in (page1, page2, page3) for r in page["rows"]]
    assert len(seen) == len(set(seen))


@pytest.mark.asyncio
async def test_skips_response_and_status_files(upload_dir):
    base = time.time()
    rid, _ = _write_request(upload_dir, "x", "p", mtime=base)
    _write_status(upload_dir, rid)
    _write_response(upload_dir, rid)
    # An orphan status without a request file must not show up.
    orphan = _hex_id("orphan")
    with open(os.path.join(upload_dir, f"{orphan}_status.json"), "w") as f:
        f.write("{}")

    result = await list_requests(cursor=None, limit=50)
    assert [r["request_id"] for r in result["rows"]] == [rid]


@pytest.mark.asyncio
async def test_corrupt_request_file_returns_placeholder(upload_dir):
    rid = _hex_id("bad")
    path = os.path.join(upload_dir, f"{rid}.json")
    with open(path, "w") as f:
        f.write("{not json")

    result = await list_requests(cursor=None, limit=50)
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["request_id"] == rid
    assert row["prompt_preview"] == "<unreadable>"
    assert row["request_count"] == 0


@pytest.mark.asyncio
async def test_long_prompts_are_truncated_with_ellipsis(upload_dir):
    long_prompt = "a" * 500
    _write_request(upload_dir, "long", long_prompt)
    result = await list_requests(cursor=None, limit=50)
    preview = result["rows"][0]["prompt_preview"]
    assert preview.endswith("…")
    assert len(preview) <= 121  # 120 chars + ellipsis


@pytest.mark.asyncio
async def test_limit_is_clamped(upload_dir):
    base = time.time()
    for i in range(3):
        _write_request(upload_dir, f"r{i}", f"p{i}", mtime=base - i)

    # limit=0 → clamped to 1
    one = await list_requests(cursor=None, limit=0)
    assert len(one["rows"]) == 1

    # limit=10000 → capped at MAX_LIMIT, but only 3 rows exist
    big = await list_requests(cursor=None, limit=10_000)
    assert len(big["rows"]) == 3


@pytest.mark.asyncio
async def test_non_hex_filenames_ignored(upload_dir):
    base = time.time()
    rid_good, _ = _write_request(upload_dir, "good", "p", mtime=base)
    # Extra files that match neither pattern should be ignored.
    with open(os.path.join(upload_dir, "not-a-hash.json"), "w") as f:
        f.write("[]")
    with open(os.path.join(upload_dir, "README.txt"), "w") as f:
        f.write("hello")

    result = await list_requests(cursor=None, limit=50)
    assert [r["request_id"] for r in result["rows"]] == [rid_good]
