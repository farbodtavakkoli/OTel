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
MAX_DISPLAY_BYTES = 5 * 1024 * 1024


async def get_request_detail(request_id: str) -> dict[str, Any]:
    if not REQUEST_ID_PATTERN.match(request_id):
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
    if not os.path.exists(path):
        return None
    return _read_capped(path)


def _safe_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None
