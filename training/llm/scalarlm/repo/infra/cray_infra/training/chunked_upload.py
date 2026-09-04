"""Session-based chunked upload for large training datasets.

Clients split a gzipped tar archive into chunks, each small enough to
pass through a proxy (e.g. Cloudflare's 100 MB body cap). The server
reassembles the archive here and hands it off to the standard training
pipeline once all chunks arrive.

See docs/chunked-upload.md for the full protocol specification.

State layout:
    /app/cray/upload_sessions/{upload_id}/
        manifest.json          — session metadata
        chunks/
            000000             — raw chunk bytes, written atomically
            000001
            ...
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid

from fastapi import HTTPException, status

from cray_infra.api.fastapi.routers.request_types.upload_chunked_types import (
    UploadChunkResponse,
    UploadFinalizeRequest,
    UploadInitRequest,
    UploadInitResponse,
)
from cray_infra.training.upload_training_data import extract_and_prepare_job
from cray_infra.util.get_config import get_config

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_upload(req: UploadInitRequest) -> UploadInitResponse:
    config = get_config()
    staging_root = config["upload_staging_directory"]

    upload_id = uuid.uuid4().hex
    session_dir = _session_dir(staging_root, upload_id)
    chunks_dir = os.path.join(session_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    manifest = {
        "upload_id": upload_id,
        "total_size": req.total_size,
        "total_hash": req.total_hash,
        "chunk_size": req.chunk_size,
        "num_chunks": req.num_chunks,
        "compressed": req.compressed,
        "params": req.params,
        "created_at": time.time(),
    }
    _write_manifest(session_dir, manifest)

    return UploadInitResponse(upload_id=upload_id, received_chunks=[])


# ---------------------------------------------------------------------------
# Write chunk
# ---------------------------------------------------------------------------

async def write_chunk(
    upload_id: str,
    chunk_index: int,
    chunk_hash: str,
    body: bytes,
) -> UploadChunkResponse:
    config = get_config()
    staging_root = config["upload_staging_directory"]
    chunk_size_limit = config["upload_chunk_size_limit"]

    session_dir = _session_dir(staging_root, upload_id)
    if not os.path.isdir(session_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session {upload_id!r} not found",
        )

    if len(body) > chunk_size_limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Chunk size {len(body)} exceeds server limit {chunk_size_limit}",
        )

    actual_hash = hashlib.sha256(body).hexdigest()
    if actual_hash != chunk_hash:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Chunk {chunk_index} hash mismatch: expected {chunk_hash}, got {actual_hash}",
        )

    chunk_filename = f"{chunk_index:06d}"
    chunks_dir = os.path.join(session_dir, "chunks")
    tmp_path = os.path.join(chunks_dir, chunk_filename + ".tmp")
    final_path = os.path.join(chunks_dir, chunk_filename)

    with open(tmp_path, "wb") as f:
        f.write(body)
    os.rename(tmp_path, final_path)

    return UploadChunkResponse(
        upload_id=upload_id,
        chunk_index=chunk_index,
        received=True,
        bytes_written=len(body),
    )


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

async def finalize_upload(req: UploadFinalizeRequest):
    """Reconstruct archive, verify integrity, extract, and return job args.

    Returns (final_dataset_filepath, train_args) — the same tuple that the
    legacy upload_training_data returns so the router can call
    launch_training_job identically.
    """
    config = get_config()
    staging_root = config["upload_staging_directory"]

    session_dir = _session_dir(staging_root, req.upload_id)
    if not os.path.isdir(session_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session {req.upload_id!r} not found",
        )

    manifest = _read_manifest(session_dir)
    num_chunks = manifest["num_chunks"]
    expected_total_size = manifest["total_size"]
    expected_hash = manifest["total_hash"]
    train_args = manifest["params"]

    chunks_dir = os.path.join(session_dir, "chunks")
    received = sorted(
        f for f in os.listdir(chunks_dir) if not f.endswith(".tmp")
    )

    if len(received) != num_chunks:
        received_indices = [int(f) for f in received]
        missing = sorted(set(range(num_chunks)) - set(received_indices))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Missing chunks: {missing}",
        )

    # Concatenate chunks into a temp archive
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_tar:
        tmp_tar_path = tmp_tar.name
        hasher = hashlib.sha256()
        total_written = 0
        for chunk_filename in received:
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            with open(chunk_path, "rb") as cf:
                data = cf.read()
                tmp_tar.write(data)
                hasher.update(data)
                total_written += len(data)

    try:
        if total_written != expected_total_size:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Reconstructed size {total_written} != expected {expected_total_size}",
            )

        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Archive hash mismatch: expected {expected_hash}, got {actual_hash}",
            )

        dataset_hash = _hash_file(tmp_tar_path)
        final_path, train_args = extract_and_prepare_job(
            tmp_tar_path, dataset_hash, train_args
        )
    finally:
        if os.path.exists(tmp_tar_path):
            os.remove(tmp_tar_path)
        shutil.rmtree(session_dir, ignore_errors=True)

    return final_path, train_args


# ---------------------------------------------------------------------------
# Stale session reaper
# ---------------------------------------------------------------------------

def reap_stale_uploads():
    config = get_config()
    staging_root = config["upload_staging_directory"]
    ttl = config["upload_session_ttl_seconds"]

    if not os.path.isdir(staging_root):
        return

    now = time.time()
    for entry in os.listdir(staging_root):
        session_dir = os.path.join(staging_root, entry)
        try:
            manifest = _read_manifest(session_dir)
            age = now - manifest.get("created_at", 0)
            if age > ttl:
                logger.info(f"Reaping stale upload session {entry} (age={age:.0f}s)")
                shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            # Corrupt or incomplete session dir — remove it
            logger.warning(f"Removing corrupt upload session dir: {session_dir}")
            shutil.rmtree(session_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_dir(staging_root: str, upload_id: str) -> str:
    return os.path.join(staging_root, upload_id)


def _write_manifest(session_dir: str, manifest: dict):
    with open(os.path.join(session_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def _read_manifest(session_dir: str) -> dict:
    with open(os.path.join(session_dir, "manifest.json")) as f:
        return json.load(f)


def _hash_file(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()
