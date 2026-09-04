"""
Unit tests for cray_infra.generate.metrics.

Three pieces under test:

1. `sdk_queue_depth` (the SQLiteAckQueue ground truth) is summed with
   `streaming_inflight` to produce the surface queue_depth — neither
   path on its own is enough.
2. `streaming_inflight` is its own counter so the OpenAI-streaming
   path can be observed without touching the SDK queue, and is balanced
   per try/finally in `_wrap_with_metrics`.
3. token/s, request/s, and flop/s are *windowed* rates over the last
   RATE_WINDOW_SECONDS, not cumulative averages over process lifetime
   — otherwise the sparkline collapses into a flat line.
"""

import time
from unittest.mock import patch

from cray_infra.generate import metrics as metrics_mod
from cray_infra.generate.metrics import Metrics


def test_queue_depth_sums_sdk_and_streaming():
    m = Metrics()
    m.record_streaming_start()
    m.record_streaming_start()
    # SDK has 5 in SQLite, streaming has 2 inflight → 7 surface.
    result = m.get_all_metrics(sdk_queue_depth=5)
    assert result["queue_depth"] == 7


def test_queue_depth_zero_streaming_returns_sdk_only():
    m = Metrics()
    result = m.get_all_metrics(sdk_queue_depth=4)
    assert result["queue_depth"] == 4


def test_queue_depth_zero_sdk_returns_streaming_only():
    # The bug we hit: deployment was OpenAI-streaming-only, sdk queue
    # was always empty, but UI displayed 0 because we ignored streaming.
    m = Metrics()
    m.record_streaming_start()
    m.record_streaming_start()
    m.record_streaming_start()
    result = m.get_all_metrics(sdk_queue_depth=0)
    assert result["queue_depth"] == 3


def test_streaming_end_decrements_but_not_below_zero():
    m = Metrics()
    m.record_streaming_start()
    m.record_streaming_end()
    m.record_streaming_end()  # extra; should be a no-op
    assert m.streaming_inflight == 0


def test_no_sdk_param_falls_back_to_legacy_counter_plus_streaming():
    m = Metrics()
    m.record_new_request()
    m.record_new_request()
    m.record_streaming_start()
    # No sdk_queue_depth passed → use in-memory queue_depth (2) plus
    # streaming_inflight (1) = 3.
    assert m.get_all_metrics()["queue_depth"] == 3


# ---- windowed rates ------------------------------------------------------


def test_windowed_token_rate_with_three_recent_completions():
    m = Metrics()

    base = 1_000_000.0
    # Three completions evenly spaced over 30 seconds, 100 tokens each.
    for offset in (0.0, 15.0, 30.0):
        with patch("cray_infra.generate.metrics.time.time", return_value=base + offset):
            m.record_new_request()
            m.record_completed_request(token_count=100, flop_count=1_000)

    # Read at base + 30: window span is 30s, 300 tokens, 3 requests.
    with patch("cray_infra.generate.metrics.time.time", return_value=base + 30.0):
        result = m.get_all_metrics(sdk_queue_depth=0)
    assert result["token/s"] == 10.0
    assert result["request/s"] == 0.1
    assert result["flop/s"] == 100.0


def test_windowed_rate_drops_old_samples():
    m = Metrics()
    base = 1_000_000.0

    # An old completion just outside the window:
    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m.record_new_request()
        m.record_completed_request(token_count=999, flop_count=999)

    # Read RATE_WINDOW_SECONDS + 1 later — old sample must be pruned.
    with patch(
        "cray_infra.generate.metrics.time.time",
        return_value=base + metrics_mod.RATE_WINDOW_SECONDS + 1,
    ):
        result = m.get_all_metrics(sdk_queue_depth=0)
    assert result["token/s"] == 0
    assert result["request/s"] == 0
    assert result["flop/s"] == 0


def test_windowed_rate_returns_zero_when_no_completions():
    m = Metrics()
    result = m.get_all_metrics(sdk_queue_depth=0)
    assert result["token/s"] == 0
    assert result["request/s"] == 0
    assert result["flop/s"] == 0


def test_windowed_rate_is_not_cumulative():
    # Same total tokens completed, but spread across a fresh process
    # vs an old one: rate should be the same once they're inside the
    # window. Cumulative average would diverge.
    m_fresh = Metrics()
    m_old = Metrics()
    base = 1_000_000.0

    # Fresh process: 50 tokens completed at t=0.
    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m_fresh.record_new_request()
        m_fresh.record_completed_request(token_count=50, flop_count=0)
    # Old process: ran for 1 hour producing 50 tokens at the start,
    # then 50 more "now". Only the recent one is inside the window.
    with patch("cray_infra.generate.metrics.time.time", return_value=base - 3600):
        m_old.record_new_request()
        m_old.record_completed_request(token_count=50, flop_count=0)
    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m_old.record_new_request()
        m_old.record_completed_request(token_count=50, flop_count=0)

    # Read both at t=base + 0.001s (~ instantaneous after).
    with patch("cray_infra.generate.metrics.time.time", return_value=base + 0.001):
        fresh = m_fresh.get_all_metrics(sdk_queue_depth=0)
        old = m_old.get_all_metrics(sdk_queue_depth=0)

    # The fresh process and the old process see the same windowed
    # token rate even though their cumulative totals differ.
    assert fresh["token/s"] == old["token/s"]
