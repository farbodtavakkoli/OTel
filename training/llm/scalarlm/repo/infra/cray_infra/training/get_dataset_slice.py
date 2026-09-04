"""
Serve a windowed view of a training job's dataset.jsonlines.

Called from GET /v1/megatron/train/{job_hash}/dataset. See
ui/docs/dataset-viewer.md for the full contract; this module implements
pagination + a case-insensitive substring filter with a bounded scan so
the endpoint can't be turned into an I/O hammer.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from cray_infra.training.get_training_job_info import get_job_directory_for_hash

logger = logging.getLogger(__name__)

MAX_LIMIT = 200
MAX_MATCH_SCAN_BYTES = 256 * 1024 * 1024  # 256 MiB — cap on `q`-scan work.
MAX_FIELD_BYTES = 4 * 1024  # 4 KiB per string field in the response payload.


def get_dataset_slice(
    job_hash: str,
    offset: int = 0,
    limit: int = 50,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if limit < 1 or limit > MAX_LIMIT:
        raise HTTPException(
            status_code=400, detail=f"limit must be in [1, {MAX_LIMIT}]"
        )

    job_directory = get_job_directory_for_hash(job_hash)
    dataset_path = os.path.join(job_directory, "dataset.jsonlines")

    if not os.path.isfile(dataset_path):
        raise HTTPException(status_code=404, detail="dataset not found")

    needle = q.lower() if q else None

    if needle is None:
        return _unfiltered_slice(dataset_path, offset, limit)
    return _filtered_slice(dataset_path, offset, limit, needle)


def _unfiltered_slice(path: str, offset: int, limit: int) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    total = 0
    end = offset + limit

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_index, raw_line in enumerate(f):
            total = line_index + 1
            if line_index < offset or line_index >= end:
                continue
            examples.append(_parse_example(line_index, raw_line))

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "matched": total,
        "truncated": False,
        "examples": examples,
    }


def _filtered_slice(
    path: str, offset: int, limit: int, needle: str
) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    total = 0
    matched = 0
    end = offset + limit
    scanned_bytes = 0
    truncated = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_index, raw_line in enumerate(f):
            total = line_index + 1
            scanned_bytes += len(raw_line)

            if needle in raw_line.lower():
                if offset <= matched < end:
                    examples.append(_parse_example(line_index, raw_line))
                matched += 1

            if scanned_bytes >= MAX_MATCH_SCAN_BYTES:
                truncated = True
                break

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "matched": matched,
        "truncated": truncated,
        "examples": examples,
    }


def _parse_example(index: int, raw_line: str) -> Dict[str, Any]:
    stripped = raw_line.rstrip("\n")
    try:
        parsed = json.loads(stripped) if stripped else {}
    except json.JSONDecodeError:
        return {
            "index": index,
            "raw": {"__parse_error__": stripped[:MAX_FIELD_BYTES]},
        }

    clipped, truncated_fields = _clip_strings(parsed)

    example: Dict[str, Any] = {"index": index, "raw": clipped}
    if isinstance(clipped, dict):
        if isinstance(clipped.get("input"), str):
            example["input"] = clipped["input"]
        if isinstance(clipped.get("output"), str):
            example["output"] = clipped["output"]
    if truncated_fields:
        example["truncated_fields"] = truncated_fields
    return example


def _clip_strings(value: Any, _path: str = "") -> tuple[Any, List[str]]:
    truncated: List[str] = []
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_FIELD_BYTES:
            return value.encode("utf-8")[:MAX_FIELD_BYTES].decode(
                "utf-8", errors="ignore"
            ), [_path or "."]
        return value, []
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            child, child_truncated = _clip_strings(v, f"{_path}.{k}" if _path else k)
            out[k] = child
            truncated.extend(child_truncated)
        return out, truncated
    if isinstance(value, list):
        out_list: List[Any] = []
        for i, v in enumerate(value):
            child, child_truncated = _clip_strings(v, f"{_path}[{i}]")
            out_list.append(child)
            truncated.extend(child_truncated)
        return out_list, truncated
    return value, []
