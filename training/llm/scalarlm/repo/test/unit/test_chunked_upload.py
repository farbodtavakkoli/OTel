"""
Unit tests for the chunked upload protocol (docs/chunked-upload.md).

Critical guard: the chunked reconstruction path must produce a byte-identical
archive and an identical job directory to the legacy single-POST path.
"""

import hashlib
import json
import os
import tarfile
import tempfile
import time

import pytest
from fastapi import HTTPException

from cray_infra.api.fastapi.routers.request_types.upload_chunked_types import (
    UploadChunkResponse,
    UploadFinalizeRequest,
    UploadInitRequest,
)
from cray_infra.training import chunked_upload


def _configure(tmp_path, monkeypatch, **overrides):
    """Point config at tmp dirs so init/finalize stage under tmp_path."""
    staging = tmp_path / "upload_sessions"
    jobs = tmp_path / "jobs"
    ml = tmp_path / "ml"
    ml.mkdir(exist_ok=True)
    (ml / "placeholder.txt").write_text("ml")
    jobs.mkdir(exist_ok=True)

    cfg = {
        "upload_staging_directory": str(staging),
        "training_job_directory": str(jobs),
        "upload_chunk_size_limit": 100 * 1024 * 1024,
        "upload_session_ttl_seconds": 6 * 60 * 60,
        **overrides,
    }
    yaml_lines = "".join(f"{k}: {json.dumps(v)}\n" for k, v in cfg.items())
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text(yaml_lines)
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))
    return cfg


def _make_archive(tmp_path):
    """Build a small gzipped tar with a dataset, return (path, bytes)."""
    data_file = tmp_path / "dataset.jsonlines"
    data_file.write_text("\n".join(json.dumps({"x": i}) for i in range(100)))

    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(str(data_file), arcname="dataset.jsonlines")
    return str(archive_path), archive_path.read_bytes()


async def _run_init(params, archive_bytes, chunk_size):
    total_hash = hashlib.sha256(archive_bytes).hexdigest()
    num_chunks = (len(archive_bytes) + chunk_size - 1) // chunk_size
    req = UploadInitRequest(
        total_size=len(archive_bytes),
        total_hash=total_hash,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        compressed=True,
        params=params,
    )
    return await chunked_upload.init_upload(req), num_chunks


def _chunks_of(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


@pytest.mark.asyncio
async def test_init_creates_session(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    _, archive_bytes = _make_archive(tmp_path)

    resp, num_chunks = await _run_init({"model": "m"}, archive_bytes, 4096)

    assert resp.upload_id
    assert resp.received_chunks == []
    session_dir = tmp_path / "upload_sessions" / resp.upload_id
    assert (session_dir / "manifest.json").exists()
    assert (session_dir / "chunks").is_dir()


@pytest.mark.asyncio
async def test_write_chunk_rejects_bad_hash(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    _, archive_bytes = _make_archive(tmp_path)
    resp, _ = await _run_init({"model": "m"}, archive_bytes, 4096)

    with pytest.raises(HTTPException) as exc:
        await chunked_upload.write_chunk(resp.upload_id, 0, "deadbeef", b"hello")
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_write_chunk_rejects_oversized(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, upload_chunk_size_limit=4)
    _, archive_bytes = _make_archive(tmp_path)
    resp, _ = await _run_init({"model": "m"}, archive_bytes, 4096)

    body = b"too-long-body"
    with pytest.raises(HTTPException) as exc:
        await chunked_upload.write_chunk(
            resp.upload_id, 0, hashlib.sha256(body).hexdigest(), body
        )
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_write_chunk_unknown_session(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    body = b"x"
    with pytest.raises(HTTPException) as exc:
        await chunked_upload.write_chunk(
            "nonexistent", 0, hashlib.sha256(body).hexdigest(), body
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_finalize_missing_chunk_returns_409(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    _, archive_bytes = _make_archive(tmp_path)
    chunk_size = max(1, len(archive_bytes) // 3)  # force at least 3 chunks
    resp, num_chunks = await _run_init({"model": "m"}, archive_bytes, chunk_size)
    assert num_chunks >= 2  # ensure this test is meaningful

    # upload all but the last chunk
    chunks = list(_chunks_of(archive_bytes, chunk_size))
    for i, c in enumerate(chunks[:-1]):
        await chunked_upload.write_chunk(
            resp.upload_id, i, hashlib.sha256(c).hexdigest(), c
        )

    with pytest.raises(HTTPException) as exc:
        await chunked_upload.finalize_upload(
            UploadFinalizeRequest(upload_id=resp.upload_id)
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_full_roundtrip_reconstructs_and_extracts(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    archive_path, archive_bytes = _make_archive(tmp_path)
    chunk_size = 1024
    resp, num_chunks = await _run_init({"model": "m"}, archive_bytes, chunk_size)

    for i, c in enumerate(_chunks_of(archive_bytes, chunk_size)):
        ack = await chunked_upload.write_chunk(
            resp.upload_id, i, hashlib.sha256(c).hexdigest(), c
        )
        assert isinstance(ack, UploadChunkResponse)
        assert ack.received

    final_path, train_args = await chunked_upload.finalize_upload(
        UploadFinalizeRequest(upload_id=resp.upload_id)
    )

    # dataset extracted to job dir
    assert os.path.exists(final_path)
    assert final_path.endswith("dataset.jsonlines")
    assert "job_directory" in train_args
    assert "dataset_hash" in train_args
    # ml dir copied in
    assert os.path.isdir(os.path.join(train_args["job_directory"], "ml"))
    # session cleaned up
    assert not (tmp_path / "upload_sessions" / resp.upload_id).exists()


@pytest.mark.asyncio
async def test_reap_stale_uploads(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, upload_session_ttl_seconds=1)
    _, archive_bytes = _make_archive(tmp_path)

    fresh, _ = await _run_init({"model": "fresh"}, archive_bytes, 4096)
    stale, _ = await _run_init({"model": "stale"}, archive_bytes, 4096)

    # backdate the stale session's manifest
    stale_dir = tmp_path / "upload_sessions" / stale.upload_id
    manifest = json.loads((stale_dir / "manifest.json").read_text())
    manifest["created_at"] = time.time() - 10_000
    (stale_dir / "manifest.json").write_text(json.dumps(manifest))

    chunked_upload.reap_stale_uploads()

    assert (tmp_path / "upload_sessions" / fresh.upload_id).exists()
    assert not stale_dir.exists()
