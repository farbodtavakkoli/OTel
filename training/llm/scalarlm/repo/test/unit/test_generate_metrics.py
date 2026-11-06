import time
from unittest.mock import patch

from cray_infra.generate import metrics as metrics_mod
from cray_infra.generate.metrics import Metrics


def test_queue_depth_sums_sdk_and_streaming():
    m = Metrics()
    m.record_streaming_start()
    m.record_streaming_start()
    result = m.get_all_metrics(sdk_queue_depth=5)
    assert result["queue_depth"] == 7


def test_queue_depth_zero_streaming_returns_sdk_only():
    m = Metrics()
    result = m.get_all_metrics(sdk_queue_depth=4)
    assert result["queue_depth"] == 4


def test_queue_depth_zero_sdk_returns_streaming_only():
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
    m.record_streaming_end()
    assert m.streaming_inflight == 0


def test_no_sdk_param_falls_back_to_legacy_counter_plus_streaming():
    m = Metrics()
    m.record_new_request()
    m.record_new_request()
    m.record_streaming_start()
    assert m.get_all_metrics()["queue_depth"] == 3


def test_windowed_token_rate_with_three_recent_completions():
    m = Metrics()

    base = 1_000_000.0
    for offset in (0.0, 15.0, 30.0):
        with patch("cray_infra.generate.metrics.time.time", return_value=base + offset):
            m.record_new_request()
            m.record_completed_request(token_count=100, flop_count=1_000)

    with patch("cray_infra.generate.metrics.time.time", return_value=base + 30.0):
        result = m.get_all_metrics(sdk_queue_depth=0)
    assert result["token/s"] == 10.0
    assert result["request/s"] == 0.1
    assert result["flop/s"] == 100.0


def test_windowed_rate_drops_old_samples():
    m = Metrics()
    base = 1_000_000.0

    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m.record_new_request()
        m.record_completed_request(token_count=999, flop_count=999)

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
    m_fresh = Metrics()
    m_old = Metrics()
    base = 1_000_000.0

    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m_fresh.record_new_request()
        m_fresh.record_completed_request(token_count=50, flop_count=0)
    with patch("cray_infra.generate.metrics.time.time", return_value=base - 3600):
        m_old.record_new_request()
        m_old.record_completed_request(token_count=50, flop_count=0)
    with patch("cray_infra.generate.metrics.time.time", return_value=base):
        m_old.record_new_request()
        m_old.record_completed_request(token_count=50, flop_count=0)

    with patch("cray_infra.generate.metrics.time.time", return_value=base + 0.001):
        fresh = m_fresh.get_all_metrics(sdk_queue_depth=0)
        old = m_old.get_all_metrics(sdk_queue_depth=0)

    assert fresh["token/s"] == old["token/s"]
