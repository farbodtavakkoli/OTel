"""
Unit tests for infra/cray_infra/training/download_dataset.py.

Contract:
- No `limit` → stream the file verbatim, byte-for-byte.
- `limit=N` → stream the first N newline-terminated lines and stop.
- Content-Disposition carries an attachment filename keyed on the job
  hash prefix so multiple downloads in the same browser don't collide.
- 404 when dataset.jsonlines is missing; 400 when `limit < 1`.
"""

import json
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.training import download_dataset as dl_mod
from cray_infra.training.download_dataset import download_dataset


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "deadbeefcafe1234"
    d.mkdir()
    return d


def _write_dataset(job_dir, rows):
    path = os.path.join(job_dir, "dataset.jsonlines")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def resolve(job_dir):
    with patch(
        "cray_infra.training.download_dataset.get_job_directory_for_hash",
        return_value=str(job_dir),
    ):
        yield


async def _collect(response) -> bytes:
    """Iterate the StreamingResponse's body and concatenate.

    Starlette wraps a sync generator passed to StreamingResponse into
    an async iterator under the hood, so the test has to drive it from
    an async context.
    """
    parts = []
    async for chunk in response.body_iterator:
        parts.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(parts)


@pytest.mark.asyncio
async def test_full_download_streams_file_verbatim(job_dir, resolve):
    rows = [{"input": f"in-{i}", "output": f"out-{i}"} for i in range(5)]
    path = _write_dataset(job_dir, rows)
    expected = open(path, "rb").read()

    response = download_dataset("deadbeefcafe1234")

    assert response.media_type == "application/x-ndjson"
    assert await _collect(response) == expected


@pytest.mark.asyncio
async def test_limit_caps_to_first_n_lines(job_dir, resolve):
    rows = [{"input": f"in-{i}", "output": f"out-{i}"} for i in range(20)]
    _write_dataset(job_dir, rows)

    response = download_dataset("deadbeefcafe1234", limit=3)
    body = (await _collect(response)).decode()

    lines = body.splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["input"] == "in-0"
    assert json.loads(lines[2])["input"] == "in-2"


@pytest.mark.asyncio
async def test_limit_larger_than_file_emits_all_rows(job_dir, resolve):
    rows = [{"input": f"in-{i}"} for i in range(4)]
    _write_dataset(job_dir, rows)

    response = download_dataset("deadbeefcafe1234", limit=999)
    body = (await _collect(response)).decode()

    assert len(body.splitlines()) == 4


def test_content_disposition_uses_hash_prefix(job_dir, resolve):
    _write_dataset(job_dir, [{"x": 1}])

    response = download_dataset("deadbeefcafe1234")

    disp = response.headers["content-disposition"]
    # Job dir basename is the full hash; we trim to a 12-char prefix
    # so the filename stays readable but unambiguous in a downloads
    # folder.
    assert "deadbeefcafe" in disp
    assert disp.endswith('.jsonl"')


def test_content_disposition_marks_sampled_downloads(job_dir, resolve):
    _write_dataset(job_dir, [{"x": i} for i in range(10)])

    response = download_dataset("deadbeefcafe1234", limit=5)

    assert "first-5" in response.headers["content-disposition"]


def test_missing_dataset_returns_404(job_dir, resolve):
    # No dataset.jsonlines written.
    with pytest.raises(HTTPException) as exc:
        download_dataset("deadbeefcafe1234")
    assert exc.value.status_code == 404


def test_invalid_limit_returns_400(job_dir, resolve):
    _write_dataset(job_dir, [{"x": 1}])
    with pytest.raises(HTTPException) as exc:
        download_dataset("deadbeefcafe1234", limit=0)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_full_path_uses_chunked_reader_not_line_iteration(job_dir, resolve):
    """When limit is None, the streamer reads in 64 KiB byte chunks
    rather than iterating line-by-line — exercise with a row larger
    than _CHUNK_BYTES so a multi-chunk read happens."""
    rows = [{"x": "a" * 100_000}]  # > _CHUNK_BYTES → multi-chunk read
    path = _write_dataset(job_dir, rows)
    expected = open(path, "rb").read()

    response = download_dataset("deadbeefcafe1234")
    assert await _collect(response) == expected
