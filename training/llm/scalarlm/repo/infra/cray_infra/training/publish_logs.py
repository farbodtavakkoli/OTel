"""
NDJSON tail of the latest publish job's `publish.log`.

Same wire shape as `/v1/health/logs/{service}` so the client can
share its NDJSON stream consumer (`streamNdjsonLogOnce`). Resolves
the freshest `publish_*/publish.log` under the training-job
directory, then delegates to the shared `tail_log_file` helper.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from cray_infra.api.fastapi.health.log_file_tailer import tail_log_file
from cray_infra.training.get_training_job_info import get_job_directory_for_hash

logger = logging.getLogger(__name__)


def publish_logs_generator(
    job_hash: str,
    starting_line_number: int = 0,
    starting_byte_offset: int | None = None,
    tail: int | None = None,
    limit: int | None = None,
    before_byte_offset: int | None = None,
    before_count: int | None = None,
):
    log_path = _resolve_latest_publish_log(job_hash)
    return tail_log_file(
        str(log_path),
        starting_line_number=starting_line_number,
        starting_byte_offset=starting_byte_offset,
        tail=tail,
        limit=limit,
        before_byte_offset=before_byte_offset,
        before_count=before_count,
    )


def _resolve_latest_publish_log(job_hash: str) -> Path:
    job_directory = Path(get_job_directory_for_hash(job_hash))
    candidates = sorted(
        job_directory.glob("publish_*"), key=lambda p: p.name, reverse=True
    )
    for child in candidates:
        log_path = child / "publish.log"
        if log_path.is_file():
            return log_path
    raise HTTPException(
        status_code=404,
        detail="no publish.log found for this job",
    )
