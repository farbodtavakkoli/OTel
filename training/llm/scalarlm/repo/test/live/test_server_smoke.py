"""Live inference smoke tests.

These tests intentionally assume the harness in ``run_live_server_tests.sh``
has provisioned a real ScalarLM server. They skip during ordinary pytest runs;
invoke them through ``./scalarlm test --level live`` so setup, readiness
checks, diagnostics, and teardown are all handled consistently.
"""

import json
import os

import pytest
import requests

from masint import SupermassiveIntelligence
from stream_assertions import assert_valid_completion_stream


BASE_URL = os.environ.get("SCALARLM_LIVE_URL", "").rstrip("/")
MODEL = os.environ.get("SCALARLM_LIVE_MODEL", "")
REQUEST_TIMEOUT = 120
pytestmark = pytest.mark.skipif(
    not BASE_URL or not MODEL,
    reason="live-server harness is not active",
)


def _assert_success(response: requests.Response) -> None:
    assert response.ok, (
        f"{response.request.method} {response.url} returned "
        f"{response.status_code}: {response.text[:2000]}"
    )


def test_health_reports_api_and_vllm_up():
    response = requests.get(f"{BASE_URL}/v1/health", timeout=REQUEST_TIMEOUT)
    _assert_success(response)

    health = response.json()
    assert health["api"] == "up"
    assert health["vllm"] == "up"


def test_models_lists_configured_model():
    response = requests.get(f"{BASE_URL}/v1/models", timeout=REQUEST_TIMEOUT)
    _assert_success(response)

    model_ids = {entry["id"] for entry in response.json()["data"]}
    assert MODEL in model_ids


def test_synchronous_scalarlm_completion_returns_openai_shape():
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "ScalarLM live test:",
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    _assert_success(response)

    result = response.json()
    assert isinstance(result["choices"][0]["text"], str)
    assert result["usage"]["total_tokens"] > 0


@pytest.mark.timeout(REQUEST_TIMEOUT)
def test_queued_sdk_generation_returns_all_results():
    client = SupermassiveIntelligence(api_url=BASE_URL)
    prompts = ["Count to two:", "Name one color:"]

    results = client.generate(prompts=prompts, model_name=MODEL, max_tokens=8)

    assert len(results) == len(prompts)
    assert all(isinstance(result, str) for result in results)


# ``requests`` resets its read timeout whenever a chunk arrives. The pytest
# timeout is the independent wall-clock deadline for a stream that never ends.
@pytest.mark.timeout(REQUEST_TIMEOUT)
def test_streaming_completion_emits_content_finish_reason_and_done():
    with requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Stream one short sentence:",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        _assert_success(response)
        events = []
        saw_done = False
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line.removeprefix("data: ")
            if payload == "[DONE]":
                saw_done = True
                break
            events.append(json.loads(payload))

    assert_valid_completion_stream(events)
    assert saw_done
