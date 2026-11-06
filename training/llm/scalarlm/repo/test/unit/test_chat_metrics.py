import time

import pytest

from cray_infra.generate.metrics import Metrics


def _exposed(metrics: Metrics) -> dict:
    return metrics.get_all_metrics()


def test_chat_admitted_increments_in_flight_and_total():
    m = Metrics()
    m.record_chat_admitted("cid-1")
    m.record_chat_admitted("cid-2")

    snapshot = _exposed(m)
    assert snapshot["chat_in_flight"] == 2
    assert snapshot["chat_total_count"] == 2
    assert snapshot["chat_admitted_429_count"] == 0


def test_chat_rejected_increments_429_and_total():
    m = Metrics()
    m.record_chat_rejected_429()
    m.record_chat_rejected_429()
    m.record_chat_rejected_429()

    snapshot = _exposed(m)
    assert snapshot["chat_admitted_429_count"] == 3
    assert snapshot["chat_total_count"] == 3
    assert snapshot["chat_in_flight"] == 0


def test_chat_resolved_decrements_in_flight_and_records_duration():
    m = Metrics()
    m.record_chat_admitted("cid-1")
    time.sleep(0.01)
    m.record_chat_resolved("cid-1")

    snapshot = _exposed(m)
    assert snapshot["chat_in_flight"] == 0
    assert snapshot["chat_request_duration_p50"] >= 0.005


def test_chat_unregistered_decrements_in_flight_without_duration():
    m = Metrics()
    m.record_chat_admitted("cid-1")
    m.record_chat_unregistered("cid-1")

    snapshot = _exposed(m)
    assert snapshot["chat_in_flight"] == 0
    assert snapshot["chat_request_duration_p50"] == 0


def test_chat_resolved_for_unknown_cid_is_noop():
    m = Metrics()
    m.record_chat_resolved("never-admitted")
    snapshot = _exposed(m)
    assert snapshot["chat_in_flight"] == 0


def test_chat_batch_size_percentiles():
    m = Metrics()
    for size in [1, 1, 1, 5, 5, 10, 10, 10, 10, 10]:
        m.record_chat_batch_size(size)

    snapshot = _exposed(m)
    assert 1 <= snapshot["chat_batch_size_p50"] <= 10
    assert snapshot["chat_batch_size_p99"] == 10


def test_chat_request_duration_percentiles_grow_with_samples():
    m = Metrics()

    for fake_duration in [0.05, 0.10, 0.50, 1.00, 5.00]:
        m.record_chat_admitted_with_clock("cid", start_time=0.0)
        m.record_chat_resolved_with_clock("cid", end_time=fake_duration)

    snapshot = _exposed(m)
    assert snapshot["chat_request_duration_p50"] > 0
    assert snapshot["chat_request_duration_p99"] >= snapshot["chat_request_duration_p50"]


def test_apparent_buffering_increments_within_tolerance():
    m = Metrics(
        buffering_check_proxy_timeout_seconds=60.0,
        buffering_match_threshold_seconds=0.5,
    )

    m.record_chat_admitted_with_clock("cid-a", start_time=0.0)
    m.record_chat_resolved_with_clock("cid-a", end_time=60.2)

    m.record_chat_admitted_with_clock("cid-b", start_time=0.0)
    m.record_chat_resolved_with_clock("cid-b", end_time=30.0)

    snapshot = _exposed(m)
    assert snapshot["chat_apparent_buffering_count"] == 1


def test_metrics_isolated_from_existing_counters():
    m = Metrics()
    m.record_chat_admitted("cid")
    m.record_chat_rejected_429()
    m.record_chat_resolved("cid")
    m.record_chat_batch_size(10)

    snapshot = _exposed(m)
    assert snapshot["queue_depth"] == 0
    assert snapshot["requests"] == 0
    assert snapshot["tokens"] == 0
