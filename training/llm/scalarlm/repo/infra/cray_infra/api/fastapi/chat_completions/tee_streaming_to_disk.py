"""
Side-channel that captures /v1/chat/completions and /v1/completions
streaming traffic to `upload_base_path`, mirroring the on-disk shape
the queue-backed paths produce. This lets the inference request
browser surface streaming requests too, without involving the queue.

The tee is best-effort: any disk error is logged and swallowed so
the user's stream is never blocked by an artifact write. Two
concurrent identical requests collide on the same content hash and
the second writer overwrites the first — same dedup property the
queue path has.
"""

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
    """
    Stable SHA-256 over the canonical JSON form of `params`. Matches
    the dedup convention of the queue path (`get_contents_hash` in
    `generate.py`); identical params hash to the same id, so the
    `_response.json` write is a free overwrite on the dedup case.
    """
    canonical = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_request_artifacts(
    *, request_hash: str, params: dict, endpoint_label: str
) -> None:
    """
    Called once before the upstream POST. Writes the request batch
    file (single-element list, matching what `enqueue_coalesced_batch`
    writes) and an in-progress status file. Errors are logged but
    never raised — the streaming response must not depend on disk.
    """
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
    """
    Called once after the upstream stream finishes (or errors). Writes
    the captured raw SSE bytes as the response file and flips status
    to `completed`. Failures swallowed for the same reason as
    `write_request_artifacts`.

    The response is dumped as `{"sse_response": "..."}` rather than
    the queue path's `{current_index, total_requests, results: {...}}`
    shape — SSE chat completions are deltas, not a single response
    dict, so forcing them into the queue shape would lose information
    operators want to see. The browser detail view JSON-pretty-prints
    whatever shape it gets.
    """
    try:
        with open(group_request_id_to_response_path(request_hash), "w") as f:
            json.dump({"sse_response": sse_text}, f)

        status_path = group_request_id_to_status_path(request_hash)
        try:
            with open(status_path) as f:
                status = json.load(f)
        except (OSError, json.JSONDecodeError):
            # The status file may have gone missing or never existed
            # if the request artifact write failed earlier. Synthesize
            # one so operators still see a coherent row.
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
    """
    What to put in the listing row's `prompt` field. Chat: the most
    recent user message — that's what an operator scanning for "did
    anyone ask about X" actually wants. Completions: the prompt
    string. Falls through to "" rather than synthesizing a
    placeholder; the browser handles empty previews.
    """
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
            # OpenAI's content-parts shape: [{"type":"text","text":"..."}].
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
