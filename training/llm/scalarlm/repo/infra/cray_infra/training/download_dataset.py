import logging
import os
from typing import Generator, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.training.get_training_job_info import get_job_directory_for_hash

logger = logging.getLogger(__name__)

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
