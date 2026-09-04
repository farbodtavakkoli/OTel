"""
Unit tests for admission control.

Two pieces:

  - is_over_high_water — pure function, checks (queue_depth +
    in_flight_count) against (admit_factor × max_num_seqs).
  - WaitEstimator — moving average of recent batch latencies, returned
    as the Retry-After hint when admission is denied.

See docs/openai-chat-completions-queue.md §5.
"""

import pytest

from cray_infra.api.fastapi.chat_completions.admission import (
    WaitEstimator,
    is_over_high_water,
)


# ---- is_over_high_water ----------------------------------------------------


@pytest.mark.parametrize(
    "queue_depth, in_flight, max_num_seqs, admit_factor, expected",
    [
        # Below threshold
        (0, 0, 256, 4, False),
        (100, 100, 256, 4, False),
        # Exactly at threshold (not "over")
        (1024, 0, 256, 4, False),
        (512, 512, 256, 4, False),
        # Just over threshold
        (1025, 0, 256, 4, True),
        (513, 512, 256, 4, True),
        # Way over
        (10000, 0, 256, 4, True),
        # admit_factor=1 (queue capped at max_num_seqs)
        (256, 0, 256, 1, False),
        (257, 0, 256, 1, True),
    ],
)
def test_is_over_high_water(
    queue_depth, in_flight, max_num_seqs, admit_factor, expected
):
    assert (
        is_over_high_water(
            queue_depth=queue_depth,
            in_flight_count=in_flight,
            max_num_seqs=max_num_seqs,
            admit_factor=admit_factor,
        )
        is expected
    )


# ---- WaitEstimator ---------------------------------------------------------


def test_no_samples_uses_default_latency():
    est = WaitEstimator(default_batch_latency_seconds=2.0, padding=1.5)
    # Overload ratio = (queue_depth - max_num_seqs) / max_num_seqs = 1.0
    # Estimate = 2.0 * 1.0 * 1.5 = 3.0
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(3.0)


def test_below_capacity_returns_zero_wait():
    """If queue depth is ≤ max_num_seqs, no overload — wait is zero."""
    est = WaitEstimator()
    assert est.estimate_wait_seconds(queue_depth=100, max_num_seqs=256) == 0
    assert est.estimate_wait_seconds(queue_depth=256, max_num_seqs=256) == 0


def test_recorded_samples_drive_moving_average():
    est = WaitEstimator(default_batch_latency_seconds=999.0, padding=1.0)
    for s in [1.0, 2.0, 3.0]:
        est.record_batch_latency_seconds(s)
    # Average = 2.0; overload = 1.0; padding = 1.0; expected = 2.0
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(2.0)


def test_padding_is_applied():
    est = WaitEstimator(default_batch_latency_seconds=4.0, padding=1.5)
    # 4.0 * 1.0 * 1.5 = 6.0
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(6.0)


def test_overload_ratio_scales_linearly():
    """A 4× overload yields a 4× larger wait than a 1× overload."""
    est = WaitEstimator(default_batch_latency_seconds=1.0, padding=1.0)

    one_x = est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256)
    four_x = est.estimate_wait_seconds(queue_depth=1280, max_num_seqs=256)

    assert four_x == pytest.approx(4 * one_x)


def test_sample_window_caps_history():
    """
    Old samples must roll off so the estimate tracks recent reality,
    not the entire run history.
    """
    est = WaitEstimator(
        default_batch_latency_seconds=999.0,
        padding=1.0,
        sample_size=4,
    )
    # Five samples; only the last four count.
    est.record_batch_latency_seconds(100.0)  # rolled off
    for s in [1.0, 2.0, 3.0, 4.0]:
        est.record_batch_latency_seconds(s)
    # Expected average = (1+2+3+4)/4 = 2.5
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(2.5)


def test_estimate_is_non_negative():
    """A misconfiguration shouldn't ever produce a negative Retry-After."""
    est = WaitEstimator(default_batch_latency_seconds=-5.0)
    # Even with a nonsense default, the result clamps to >= 0.
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) >= 0
