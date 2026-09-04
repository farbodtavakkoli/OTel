from masint.util.make_api_url import make_api_url
from masint.util.get_session import get_session

import aiofiles
import aiohttp
import contextlib
import asyncio
import hashlib
import io
import json
import os
import tarfile
import tempfile
import jsonlines

import logging

logger = logging.getLogger(__name__)

# Datasets smaller than this threshold go through the legacy single-POST path.
# Default 90 MB — safely under the typical Cloudflare 100 MB cap.
_SINGLE_POST_THRESHOLD = int(os.environ.get("MASINT_SINGLE_POST_THRESHOLD", 90 * 1024 * 1024))

# Chunk size for the session-based upload. Must be <= server's upload_chunk_size_limit.
_UPLOAD_CHUNK_SIZE = int(os.environ.get("MASINT_UPLOAD_CHUNK_SIZE", 50 * 1024 * 1024))

_MAX_CHUNK_RETRIES = 5


async def submit_training_job(data, model_name, train_args, api_url):
    with make_training_archive(data) as archive_path:
        total_size = os.path.getsize(archive_path)

        if total_size <= _SINGLE_POST_THRESHOLD:
            logger.info(f"Archive {total_size} bytes — using single-POST upload")
            upload_url = make_api_url("v1/megatron/train", api_url=api_url)
            return await upload_async(archive_path, upload_url, train_args)

        logger.info(
            f"Archive {total_size} bytes exceeds threshold {_SINGLE_POST_THRESHOLD} — "
            "using chunked upload"
        )
        return await _upload_chunked(archive_path, total_size, train_args, api_url)


# ---------------------------------------------------------------------------
# Legacy single-POST path (small datasets)
# ---------------------------------------------------------------------------

async def upload_async(data_file_path, api_url, train_args):
    """Single-POST multipart upload. Kept public for submit_slurm_job."""
    async with get_session() as session:
        content_length = await _get_content_length(data_file_path, train_args)

        with _make_multipart_writer(data_file_path, train_args) as mp:
            headers = mp.headers
            headers["Content-Length"] = str(content_length)

            async with session.post(api_url, data=mp, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to upload data: {await resp.text()}")
                return await resp.json()


async def _get_content_length(data_file_path, train_args):
    with _make_multipart_writer(data_file_path, train_args) as mp:
        class Writer:
            def __init__(self):
                self.count = 0

            async def write(self, data):
                self.count += len(data)

        writer = Writer()
        await mp.write(writer)
        return writer.count


@contextlib.contextmanager
def _make_multipart_writer(data_file_path, train_args):
    with aiohttp.MultipartWriter("form-data") as mp:
        file_part = mp.append(_file_sender(data_file_path))
        file_part.set_content_disposition("form-data", name="file", filename="dataset")

        params_part = mp.append_json(train_args)
        params_part.set_content_disposition("form-data", name="params")

        yield mp


async def _file_sender(file_path):
    chunk_size = 64 * 1024
    async with aiofiles.open(file_path, "rb") as f:
        chunk = await f.read(chunk_size)
        while chunk:
            yield chunk
            chunk = await f.read(chunk_size)


# ---------------------------------------------------------------------------
# Chunked upload path (large datasets)
# ---------------------------------------------------------------------------

async def _upload_chunked(archive_path, total_size, train_args, api_url):
    total_hash = _hash_file(archive_path)
    num_chunks = (total_size + _UPLOAD_CHUNK_SIZE - 1) // _UPLOAD_CHUNK_SIZE

    init_url = make_api_url("v1/megatron/upload/init", api_url=api_url)
    chunk_url = make_api_url("v1/megatron/upload/chunk", api_url=api_url)
    finalize_url = make_api_url("v1/megatron/upload/finalize", api_url=api_url)

    # 1. Init
    init_payload = {
        "total_size": total_size,
        "total_hash": total_hash,
        "chunk_size": _UPLOAD_CHUNK_SIZE,
        "num_chunks": num_chunks,
        "compressed": True,
        "params": train_args,
    }
    async with get_session() as session:
        async with session.post(init_url, json=init_payload) as resp:
            if resp.status != 200:
                raise Exception(f"Upload init failed: {await resp.text()}")
            init_resp = await resp.json()

    upload_id = init_resp["upload_id"]
    already_received = set(init_resp.get("received_chunks", []))
    logger.info(f"Chunked upload session {upload_id}: {num_chunks} chunks, "
                f"{total_size} bytes, already received: {already_received}")

    # 2. Upload chunks
    async with aiofiles.open(archive_path, "rb") as f:
        for chunk_index in range(num_chunks):
            if chunk_index in already_received:
                logger.debug(f"Skipping already-received chunk {chunk_index}")
                await f.seek((chunk_index + 1) * _UPLOAD_CHUNK_SIZE)
                continue

            chunk_data = await f.read(_UPLOAD_CHUNK_SIZE)
            if not chunk_data:
                break

            await _upload_chunk_with_retry(
                chunk_url, upload_id, chunk_index, chunk_data
            )

    # 3. Finalize
    async with get_session() as session:
        async with session.post(finalize_url, json={"upload_id": upload_id}) as resp:
            if resp.status != 200:
                raise Exception(f"Upload finalize failed: {await resp.text()}")
            return await resp.json()


async def _upload_chunk_with_retry(chunk_url, upload_id, chunk_index, chunk_data):
    chunk_hash = hashlib.sha256(chunk_data).hexdigest()
    headers = {
        "X-Upload-Id": upload_id,
        "X-Chunk-Index": str(chunk_index),
        "X-Chunk-Hash": chunk_hash,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(chunk_data)),
    }

    delay = 1.0
    for attempt in range(_MAX_CHUNK_RETRIES):
        try:
            async with get_session() as session:
                async with session.post(
                    chunk_url, data=chunk_data, headers=headers
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"Chunk {chunk_index} uploaded successfully")
                        return
                    body = await resp.text()
                    if resp.status == 422:
                        # Hash mismatch — retry
                        logger.warning(
                            f"Chunk {chunk_index} hash mismatch (attempt {attempt + 1}): {body}"
                        )
                    elif resp.status == 413:
                        raise Exception(
                            f"Chunk {chunk_index} too large even for chunked path: {body}"
                        )
                    else:
                        logger.warning(
                            f"Chunk {chunk_index} got {resp.status} (attempt {attempt + 1}): {body}"
                        )
        except Exception as e:
            if attempt == _MAX_CHUNK_RETRIES - 1:
                raise
            logger.warning(
                f"Chunk {chunk_index} attempt {attempt + 1} failed: {e}; retrying in {delay:.1f}s"
            )

        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)

    raise Exception(f"Chunk {chunk_index} failed after {_MAX_CHUNK_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Archive building
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def make_training_archive(data):
    with make_data_file(data) as data_file_path:
        check_for_zero_length_file(data_file_path)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as archive_file:
            archive_path = archive_file.name

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(
                    data_file_path,
                    arcname="dataset.jsonlines",
                    filter=tar_info_strip_file_info,
                )

                ml_dir = find_ml_dir()
                if ml_dir is None:
                    logger.warning(
                        "ML directory not found. Skipping addition to "
                        "archive, using default ml directory from ScalarLM server."
                    )
                else:
                    tar.add(ml_dir, arcname="ml", filter=tar_info_strip_file_info)

            logger.debug(f"Archive created at {archive_path}")
            logger.debug(f"Archive size: {os.path.getsize(archive_path)} bytes")

            yield archive_path
        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)


def tar_info_strip_file_info(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    tarinfo.mtime = 0
    return tarinfo


def find_ml_dir():
    current_directory = os.path.join(os.getcwd(), "ml")
    if os.path.exists(current_directory):
        return current_directory

    peer_directory = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "ml"
    )
    if os.path.exists(peer_directory):
        return peer_directory


@contextlib.contextmanager
def make_data_file(data):
    if isinstance(data, str):
        with open(data, "rb") as f:
            yield f

    elif isinstance(data, io.BufferedIOBase):
        chunk_size = 64 * 1024
        with tempfile.NamedTemporaryFile() as f:
            for chunk in iter(lambda: data.read(chunk_size), b""):
                f.write(chunk)
            f.seek(0)
            f.flush()
            yield f.name

    elif isinstance(data, list):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name
        try:
            with jsonlines.open(tmp_path, mode="w") as writer:
                for item in data:
                    writer.write(item)
            yield tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def check_for_zero_length_file(file_path):
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"The file {file_path} is empty. Please provide valid data.")


def _hash_file(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()
