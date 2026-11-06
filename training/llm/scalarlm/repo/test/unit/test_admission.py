import pytest

from cray_infra.api.fastapi.chat_completions.admission import (
    WaitEstimator,
    is_over_high_water,
)


@pytest.mark.parametrize(
    "queue_depth, in_flight, max_num_seqs, admit_factor, expected",
    [
        (0, 0, 256, 4, False),
        (100, 100, 256, 4, False),
        (1024, 0, 256, 4, False),
        (512, 512, 256, 4, False),
        (1025, 0, 256, 4, True),
        (513, 512, 256, 4, True),
        (10000, 0, 256, 4, True),
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


def test_no_samples_uses_default_latency():
    est = WaitEstimator(default_batch_latency_seconds=2.0, padding=1.5)
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(3.0)


def test_below_capacity_returns_zero_wait():
    est = WaitEstimator()
    assert est.estimate_wait_seconds(queue_depth=100, max_num_seqs=256) == 0
    assert est.estimate_wait_seconds(queue_depth=256, max_num_seqs=256) == 0


def test_recorded_samples_drive_moving_average():
    est = WaitEstimator(default_batch_latency_seconds=999.0, padding=1.0)
    for s in [1.0, 2.0, 3.0]:
        est.record_batch_latency_seconds(s)
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(2.0)


def test_padding_is_applied():
    est = WaitEstimator(default_batch_latency_seconds=4.0, padding=1.5)
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(6.0)


def test_overload_ratio_scales_linearly():
    est = WaitEstimator(default_batch_latency_seconds=1.0, padding=1.0)

    one_x = est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256)
    four_x = est.estimate_wait_seconds(queue_depth=1280, max_num_seqs=256)

    assert four_x == pytest.approx(4 * one_x)


def test_sample_window_caps_history():
    est = WaitEstimator(
        default_batch_latency_seconds=999.0,
        padding=1.0,
        sample_size=4,
    )
    est.record_batch_latency_seconds(100.0)
    for s in [1.0, 2.0, 3.0, 4.0]:
        est.record_batch_latency_seconds(s)
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) == pytest.approx(2.5)


def test_estimate_is_non_negative():
    est = WaitEstimator(default_batch_latency_seconds=-5.0)
    assert est.estimate_wait_seconds(queue_depth=512, max_num_seqs=256) >= 0
