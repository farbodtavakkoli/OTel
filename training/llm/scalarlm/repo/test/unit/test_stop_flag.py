import signal

import pytest

from cray_megatron.megatron import stop_flag


@pytest.fixture(autouse=True)
def reset_stop_flag():
    stop_flag.reset()
    yield
    stop_flag.reset()


def test_stop_flag_defaults_clear():
    assert stop_flag.was_stop_requested() is False
    assert stop_flag.last_signal() is None


def test_stop_flag_request_sets_latch_and_signal():
    stop_flag.request_stop(signal_number=signal.SIGTERM)
    assert stop_flag.was_stop_requested() is True
    assert stop_flag.last_signal() == signal.SIGTERM


def test_stop_flag_request_without_signal_keeps_prior_signal():
    stop_flag.request_stop(signal_number=signal.SIGTERM)
    stop_flag.request_stop()
    assert stop_flag.was_stop_requested() is True
    assert stop_flag.last_signal() == signal.SIGTERM


def test_stop_flag_distinguishes_sigterm_from_sigcont():
    stop_flag.request_stop(signal_number=signal.SIGCONT)
    assert stop_flag.last_signal() == signal.SIGCONT
    assert stop_flag.last_signal() != signal.SIGTERM


def test_stop_flag_reset_clears_both():
    stop_flag.request_stop(signal_number=signal.SIGTERM)
    stop_flag.reset()
    assert stop_flag.was_stop_requested() is False
    assert stop_flag.last_signal() is None
