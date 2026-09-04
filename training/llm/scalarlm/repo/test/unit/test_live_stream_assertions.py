"""Regression tests for completion-stream false-positive prevention."""

import runpy
from pathlib import Path

import pytest

STREAM_ASSERTIONS = (
    Path(__file__).resolve().parents[1] / "live" / "stream_assertions.py"
)
ASSERT_VALID_STREAM = runpy.run_path(str(STREAM_ASSERTIONS))[
    "assert_valid_completion_stream"
]


def test_completion_stream_requires_text_and_terminal_finish_reason():
    ASSERT_VALID_STREAM(
        [
            {"choices": [{"text": "hello", "finish_reason": None}]},
            {"choices": [{"text": "", "finish_reason": "stop"}]},
        ]
    )


@pytest.mark.parametrize(
    "events",
    [
        [{"choices": [{}]}],
        [{"choices": [{"text": "", "finish_reason": "stop"}]}],
    ],
)
def test_completion_stream_rejects_missing_text(events):
    with pytest.raises(AssertionError):
        ASSERT_VALID_STREAM(events)


def test_completion_stream_rejects_missing_finish_reason():
    with pytest.raises(AssertionError):
        ASSERT_VALID_STREAM([{"choices": [{"text": "hello"}]}])
