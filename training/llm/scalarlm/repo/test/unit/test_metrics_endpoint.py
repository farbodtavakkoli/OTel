"""
Tests for /v1/generate/metrics route handler. The mapping between
SQLiteAckQueue state → reported queue_depth is the part most likely
to silently regress (we shipped two bugs here already: len() didn't
include unacked items, then len()+unack_count didn't account for the
per-prompt fanout that push_into_queue records on each entry).
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from cray_infra.api.fastapi.generate.metrics import metrics
from cray_infra.generate import metrics as metrics_state


def _fake_queue(pending_prompts: int = 0):
    """Stand-in for InferenceWorkQueue exposing pending_request_count."""
    q = AsyncMock()
    q.pending_request_count = AsyncMock(return_value=pending_prompts)
    return q


def _drain(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _reset_metrics():
    metrics_state.generate_metrics = None


def setup_function(_fn):
    _reset_metrics()


# ---- queue depth surfaces per-prompt SDK work --------------------------


def test_queue_depth_uses_pending_request_count():
    # 7 prompts in flight → 7, regardless of how the worker batches.
    fake = _fake_queue(pending_prompts=7)
    with patch(
        "cray_infra.api.fastapi.generate.metrics.get_inference_work_queue",
        new=AsyncMock(return_value=fake),
    ):
        result = _drain(metrics())
    assert result["queue_depth"] == 7


def test_queue_depth_streaming_inflight_added_on_top():
    fake = _fake_queue(pending_prompts=4)
    metrics_state.get_metrics().record_streaming_start()
    metrics_state.get_metrics().record_streaming_start()
    with patch(
        "cray_infra.api.fastapi.generate.metrics.get_inference_work_queue",
        new=AsyncMock(return_value=fake),
    ):
        result = _drain(metrics())
    # 4 SDK prompts + 2 streaming = 6
    assert result["queue_depth"] == 6


# ---- error-path handling ----------------------------------------------


def test_queue_depth_clamps_negative_to_zero():
    fake = _fake_queue(pending_prompts=-3)
    with patch(
        "cray_infra.api.fastapi.generate.metrics.get_inference_work_queue",
        new=AsyncMock(return_value=fake),
    ):
        result = _drain(metrics())
    assert result["queue_depth"] == 0


def test_queue_depth_handles_value_error_from_persistqueue():
    fake = AsyncMock()
    fake.pending_request_count = AsyncMock(
        side_effect=ValueError("__len__() should return >= 0")
    )
    with patch(
        "cray_infra.api.fastapi.generate.metrics.get_inference_work_queue",
        new=AsyncMock(return_value=fake),
    ):
        result = _drain(metrics())
    assert result["queue_depth"] == 0


def test_queue_depth_handles_unreachable_queue():
    with patch(
        "cray_infra.api.fastapi.generate.metrics.get_inference_work_queue",
        new=AsyncMock(side_effect=RuntimeError("queue offline")),
    ):
        result = _drain(metrics())
    assert result["queue_depth"] == 0
