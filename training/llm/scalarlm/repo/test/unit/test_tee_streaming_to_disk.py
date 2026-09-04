"""
Unit tests for tee_streaming_to_disk.

Contract (see docs/inference-request-browser.md and
infra/cray_infra/api/fastapi/chat_completions/tee_streaming_to_disk.py):

- Identical params hash to identical request_ids (dedup property
  shared with the queue path).
- The request file is a single-element list whose first dict carries
  `prompt`, `model`, `request_type`, and the full `params` — exactly
  what list_requests reads to populate a row.
- The response artifact flips the status file to `completed` even
  if the original status file is missing or corrupt.
- All filesystem failures are swallowed and logged; nothing the tee
  does is allowed to bubble up and break the user's stream.
"""

import json
import os
from unittest.mock import patch

import pytest

from cray_infra.api.fastapi.chat_completions.tee_streaming_to_disk import (
    compute_request_hash,
    write_request_artifacts,
    write_response_artifact,
)


@pytest.fixture
def upload_dir(tmp_path):
    target = tmp_path / "inference_requests"
    target.mkdir()
    fake_config = {"upload_base_path": str(target)}
    with patch(
        "cray_infra.api.fastapi.chat_completions.tee_streaming_to_disk.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_status_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_response_path.get_config",
        return_value=fake_config,
    ):
        yield target


def test_hash_is_stable_across_key_order():
    a = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    b = {"stream": True, "messages": [{"role": "user", "content": "hi"}], "model": "m"}
    assert compute_request_hash(a) == compute_request_hash(b)


def test_hash_differs_for_different_params():
    a = {"model": "m", "prompt": "hi"}
    b = {"model": "m", "prompt": "hello"}
    assert compute_request_hash(a) != compute_request_hash(b)


def test_request_artifacts_write_chat_messages_preview(upload_dir):
    params = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "what is 3+3?"},
        ],
        "stream": True,
    }
    rid = compute_request_hash(params)
    write_request_artifacts(
        request_hash=rid, params=params, endpoint_label="chat completions",
    )

    request_file = upload_dir / f"{rid}.json"
    assert request_file.exists()
    with open(request_file) as f:
        data = json.load(f)
    assert isinstance(data, list)
    entry = data[0]
    # Most-recent user message wins — operators scanning rows want
    # what was *just* asked, not the system preamble.
    assert entry["prompt"] == "what is 3+3?"
    assert entry["model"] == "m"
    assert entry["request_type"] == "chat_completions_streaming"
    assert entry["params"] == params

    status_file = upload_dir / f"{rid}_status.json"
    assert status_file.exists()
    with open(status_file) as f:
        status = json.load(f)
    assert status["status"] == "in_progress"
    assert status["transport"] == "sse"


def test_request_artifacts_write_completions_prompt_preview(upload_dir):
    params = {"model": "m", "prompt": "complete this", "stream": True}
    rid = compute_request_hash(params)
    write_request_artifacts(
        request_hash=rid, params=params, endpoint_label="completions",
    )

    with open(upload_dir / f"{rid}.json") as f:
        entry = json.load(f)[0]
    assert entry["prompt"] == "complete this"
    assert entry["request_type"] == "completions_streaming"


def test_request_artifacts_handle_content_parts(upload_dir):
    """OpenAI multimodal content shape: [{type:'text', text:'...'}]."""
    params = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "describe this image"}],
            }
        ],
    }
    rid = compute_request_hash(params)
    write_request_artifacts(
        request_hash=rid, params=params, endpoint_label="chat completions",
    )
    with open(upload_dir / f"{rid}.json") as f:
        entry = json.load(f)[0]
    assert entry["prompt"] == "describe this image"


def test_request_artifacts_empty_preview_when_neither_field_present(upload_dir):
    params = {"model": "m"}
    rid = compute_request_hash(params)
    write_request_artifacts(
        request_hash=rid, params=params, endpoint_label="chat completions",
    )
    with open(upload_dir / f"{rid}.json") as f:
        entry = json.load(f)[0]
    assert entry["prompt"] == ""


def test_response_artifact_writes_response_and_flips_status(upload_dir):
    params = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    rid = compute_request_hash(params)
    write_request_artifacts(
        request_hash=rid, params=params, endpoint_label="chat completions",
    )

    sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
    write_response_artifact(request_hash=rid, sse_text=sse)

    with open(upload_dir / f"{rid}_response.json") as f:
        response = json.load(f)
    assert response["sse_response"] == sse

    with open(upload_dir / f"{rid}_status.json") as f:
        status = json.load(f)
    assert status["status"] == "completed"
    assert "completed_at" in status
    assert status["current_index"] == 1


def test_response_artifact_synthesises_status_when_request_artifact_missing(upload_dir):
    """
    The request-side write may have failed (full disk, races). The
    response-side write must still produce a row that list_requests
    can render — so it falls back to a synthesized status.
    """
    rid = compute_request_hash({"model": "m"})
    # No write_request_artifacts call: status file does not exist.
    write_response_artifact(request_hash=rid, sse_text="data: [DONE]\n\n")

    with open(upload_dir / f"{rid}_status.json") as f:
        status = json.load(f)
    assert status["status"] == "completed"
    assert status["current_index"] == 1


def test_disk_errors_are_swallowed(tmp_path, caplog):
    """A read-only base path must not crash the streaming response."""
    bad_path = tmp_path / "does-not-exist" / "and-cant-be-made"
    fake_config = {"upload_base_path": str(bad_path)}
    with patch(
        "cray_infra.api.fastapi.chat_completions.tee_streaming_to_disk.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_status_path.get_config",
        return_value=fake_config,
    ), patch(
        "cray_infra.api.work_queue.group_request_id_to_response_path.get_config",
        return_value=fake_config,
    ):
        # Force os.makedirs to fail so we hit the OSError branch.
        with patch(
            "cray_infra.api.fastapi.chat_completions.tee_streaming_to_disk.os.makedirs",
            side_effect=OSError("read-only fs"),
        ):
            # Must not raise.
            write_request_artifacts(
                request_hash="abcd",
                params={"model": "m"},
                endpoint_label="chat completions",
            )
        write_response_artifact(request_hash="abcd", sse_text="data: [DONE]\n\n")
