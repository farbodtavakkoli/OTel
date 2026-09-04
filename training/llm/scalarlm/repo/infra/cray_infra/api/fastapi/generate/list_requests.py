"""
Read-only listing of inference request files persisted to
`upload_base_path`. See docs/inference-request-browser.md §4.1.

The shape on disk is three files per group_request_id (the SHA-256 of
the request batch contents):
  - `{hash}.json`        — the request batch (list of request dicts)
  - `{hash}_response.json` — the responses (or absent if still in flight)
  - `{hash}_status.json`   — the in_progress / completed marker

We treat `{hash}.json` as the canonical row. A status file with no
matching request file is dropped from the listing — that situation
shouldn't occur in normal operation, and surfacing it would be
misleading because there's nothing to display.
"""

import json
import logging
import os
import re
from typing import Any

from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


REQUEST_FILE_PATTERN = re.compile(r"^([0-9a-f]{64})\.json$")
PROMPT_PREVIEW_CHARS = 120
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


async def list_requests(cursor: float | None, limit: int) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_LIMIT))
    base_path = get_config()["upload_base_path"]

    if not os.path.isdir(base_path):
        return {"rows": [], "next_cursor": None, "has_more": False}

    entries = _scan_request_entries(base_path)
    # Newest first. Stable sort keeps deterministic order on mtime ties.
    entries.sort(key=lambda e: e[1], reverse=True)

    if cursor is not None:
        # Cursor is the mtime of the last row from the previous page.
        # Strict-less keeps us from double-listing it. Float equality
        # would only collide on exact-tie mtimes; we accept the tiny
        # risk of skipping a tied row over the bigger risk of dups.
        entries = [e for e in entries if e[1] < cursor]

    page = entries[:limit]
    rows = [_build_row(name, mtime, base_path) for name, mtime in page]

    has_more = len(entries) > limit
    next_cursor = page[-1][1] if (has_more and page) else None

    return {"rows": rows, "next_cursor": next_cursor, "has_more": has_more}


def _scan_request_entries(base_path: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    try:
        with os.scandir(base_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                m = REQUEST_FILE_PATTERN.match(entry.name)
                if not m:
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                out.append((entry.name, mtime))
    except OSError as exc:
        logger.warning("scandir of %s failed: %s", base_path, exc)
    return out


def _build_row(name: str, mtime: float, base_path: str) -> dict[str, Any]:
    request_id = name[: -len(".json")]
    request_path = os.path.join(base_path, name)

    try:
        size_bytes = os.path.getsize(request_path)
    except OSError:
        size_bytes = 0

    request_count, prompt_preview, model, request_type = _peek_request_file(request_path)
    status, completed_at = _peek_status_file(request_id)
    has_response = os.path.exists(group_request_id_to_response_path(request_id))

    return {
        "request_id": request_id,
        "mtime": mtime,
        "size_bytes": size_bytes,
        "request_count": request_count,
        "status": status,
        "completed_at": completed_at,
        "model": model,
        "request_type": request_type,
        "prompt_preview": prompt_preview,
        "has_response": has_response,
    }


def _peek_request_file(path: str) -> tuple[int, str, str, str]:
    """
    Read just enough of the request file to populate the listing row.
    Truncated/corrupt files return placeholders rather than 500ing —
    surfacing the row's existence is more useful than hiding it.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read request file %s: %s", path, exc)
        return 0, "<unreadable>", "unknown", "unknown"

    if not isinstance(data, list) or not data:
        return 0, "", "unknown", "unknown"

    first = data[0] if isinstance(data[0], dict) else {}
    prompt = first.get("prompt") or ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    preview = prompt[:PROMPT_PREVIEW_CHARS]
    if len(prompt) > PROMPT_PREVIEW_CHARS:
        preview += "…"

    model = first.get("model") or "unknown"
    request_type = first.get("request_type") or "unknown"
    return len(data), preview, model, request_type


def _peek_status_file(request_id: str) -> tuple[str, float | None]:
    path = group_request_id_to_status_path(request_id)
    if not os.path.exists(path):
        return "unknown", None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read status file %s: %s", path, exc)
        return "unknown", None
    return data.get("status") or "unknown", data.get("completed_at")
