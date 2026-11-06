import datetime

import pytest

from cray_infra.training.launch_training_job import format_timedelta


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0-00:00:00"),
        (1, "0-00:00:01"),
        (59, "0-00:00:59"),
        (60, "0-00:01:00"),
        (3599, "0-00:59:59"),
        (3600, "0-01:00:00"),
        (3725, "0-01:02:05"),
        (86399, "0-23:59:59"),
        (86400, "1-00:00:00"),
        (90061, "1-01:01:01"),
        (2 * 86400 + 7200 + 30, "2-02:00:30"),
    ],
)
def test_format_timedelta_formats_dd_hh_mm_ss(seconds, expected):
    assert format_timedelta(datetime.timedelta(seconds=seconds)) == expected


def test_format_timedelta_pads_to_two_digits():
    out = format_timedelta(datetime.timedelta(seconds=9))
    assert out == "0-00:00:09"
