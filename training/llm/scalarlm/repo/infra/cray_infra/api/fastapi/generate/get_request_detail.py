"""
Read-only detail endpoint for one inference request bundle. See
docs/inference-request-browser.md §4.2.

Returns the request batch JSON, the response file (or null), and the
status file (or null) for a given group_request_id. Files larger than
`MAX_DISPLAY_BYTES` are replaced with a placeholder rather than
serialized — operators can SSH for the truly enormous ones; the
browser is sized for the common case.
"""

import json
import logging
import os
import re
from typing import Any

from fastapi import HTTPException

from cray_infra.api.work_queue.group_request_id_to_path import (
    group_request_id_to_path,
)
from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)

logger = logging.getLogger(__name__)


REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_DISPLAY_BYTES = 5 * 1024 * 1024  # 5 MB


async def get_request_detail(request_id: str) -> dict[str, Any]:
    if not REQUEST_ID_PATTERN.match(request_id):
        # Validating up front blocks path-traversal attempts before any
        # filesystem call. The 64-char hex form is what the writers
        # produce, so no legitimate id is rejected here.
        raise HTTPException(
            status_code=400, detail="request_id must be a 64-char hex string"
        )

    request_path = group_request_id_to_path(request_id)
    if not os.path.exists(request_path):
        raise HTTPException(
            status_code=404, detail=f"request {request_id} not found"
        )

    response_path = group_request_id_to_response_path(request_id)
    status_path = group_request_id_to_status_path(request_id)

    return {
        "request_id": request_id,
        "request": _read_capped(request_path),
        "response": _read_optional(response_path),
        "status": _read_optional(status_path),
        "request_mtime": _safe_mtime(request_path),
        "response_mtime": _safe_mtime(response_path),
    }


def _read_capped(path: str) -> Any:
    """
    Read a file that we know exists. If it's larger than the display
    cap, return a placeholder. JSON errors get caught and turned into a
    placeholder too — a partially-written file (the response writer
    isn't always under our lock from the reader's perspective) should
    not crash the endpoint.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        logger.warning("getsize failed for %s: %s", path, exc)
        return {"error": "unreadable", "detail": str(exc)}

    if size > MAX_DISPLAY_BYTES:
        return {"error": "too large to display", "size_bytes": size}

    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("read failed for %s: %s", path, exc)
        return {"error": "unreadable", "detail": str(exc)}


def _read_optional(path: str) -> Any:
    """Same as `_read_capped` but returns None if the file is missing."""
    if not os.path.exists(path):
        return None
    return _read_capped(path)


def _safe_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None
