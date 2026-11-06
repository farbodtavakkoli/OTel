import hashlib
import json
import logging
import os
import time
from typing import Any

from cray_infra.api.work_queue.group_request_id_to_path import (
    group_request_id_to_path,
)
from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


def compute_request_hash(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_request_artifacts(
    *, request_hash: str, params: dict, endpoint_label: str
) -> None:
    try:
        base = get_config()["upload_base_path"]
        os.makedirs(base, exist_ok=True)

        entry = {
            "prompt": _derive_prompt_preview(params),
            "model": params.get("model", "unknown"),
            "request_type": _request_type_for(endpoint_label),
            "params": params,
        }
        with open(group_request_id_to_path(request_hash), "w") as f:
            json.dump([entry], f)

        status: dict[str, Any] = {
            "status": "in_progress",
            "current_index": 0,
            "total_requests": 1,
            "started_at": time.time(),
            "transport": "sse",
        }
        with open(group_request_id_to_status_path(request_hash), "w") as f:
            json.dump(status, f)
    except OSError as exc:
        logger.warning(
            "tee: failed to write request artifacts for %s: %s",
            request_hash,
            exc,
        )


def write_response_artifact(*, request_hash: str, sse_text: str) -> None:
    try:
        with open(group_request_id_to_response_path(request_hash), "w") as f:
            json.dump({"sse_response": sse_text}, f)

        status_path = group_request_id_to_status_path(request_hash)
        try:
            with open(status_path) as f:
                status = json.load(f)
        except (OSError, json.JSONDecodeError):
            status = {
                "current_index": 0,
                "total_requests": 1,
                "transport": "sse",
            }
        status["status"] = "completed"
        status["completed_at"] = time.time()
        status["current_index"] = 1
        with open(status_path, "w") as f:
            json.dump(status, f)
    except OSError as exc:
        logger.warning(
            "tee: failed to write response artifact for %s: %s",
            request_hash,
            exc,
        )


def _request_type_for(endpoint_label: str) -> str:
    if endpoint_label == "chat completions":
        return "chat_completions_streaming"
    return "completions_streaming"


def _derive_prompt_preview(params: dict) -> str:
    messages = params.get("messages")
    if isinstance(messages, list) and messages:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        return part["text"]
            break
    prompt = params.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return ""
