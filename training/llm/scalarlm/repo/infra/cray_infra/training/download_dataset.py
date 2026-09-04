"""
Stream a training job's dataset.jsonlines verbatim for download.

Distinct from get_dataset_slice (the in-browser viewer), which clips
every string field at 4 KiB and JSON-reparses every line — fine for
display but data-lossy and slow for download. The download path
serves the raw bytes so the file can round-trip back into a future
training submission unchanged.

Optional `limit` caps the number of lines for a "sample" download
without re-encoding (we stop reading early once the requested line
count is emitted).
"""

import logging
import os
from typing import Generator, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.training.get_training_job_info import get_job_directory_for_hash

logger = logging.getLogger(__name__)

# 64 KiB matches the upload-side hashing chunk in
# api/fastapi/generate/upload.py — same disk read pattern.
_CHUNK_BYTES = 64 * 1024


def download_dataset(
    job_hash: str, limit: Optional[int] = None
) -> StreamingResponse:
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    job_directory = get_job_directory_for_hash(job_hash)
    dataset_path = os.path.join(job_directory, "dataset.jsonlines")

    if not os.path.isfile(dataset_path):
        raise HTTPException(status_code=404, detail="dataset not found")

    job_basename = os.path.basename(job_directory)
    short = job_basename[:12]
    filename = (
        f"dataset-{short}-first-{limit}.jsonl"
        if limit
        else f"dataset-{short}.jsonl"
    )

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }

    return StreamingResponse(
        _stream_file(dataset_path, limit),
        media_type="application/x-ndjson",
        headers=headers,
    )


def _stream_file(
    path: str, limit: Optional[int]
) -> Generator[bytes, None, None]:
    """
    Yield bytes chunks of the dataset file. When `limit` is None we
    stream in 64 KiB blocks (fastest, no per-line work). When `limit`
    is set we iterate line-by-line and stop after `limit` newlines so
    we don't read the whole file just to throw most of it away.
    """
    if limit is None:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_BYTES)
                if not chunk:
                    return
                yield chunk
        return

    emitted = 0
    with open(path, "rb") as f:
        for raw in f:
            yield raw
            emitted += 1
            if emitted >= limit:
                return
