"""
Unit tests for ml/cray_megatron/megatron/stop_flag.py.

Contract under test: docs/training-lifecycle.md §4.1, §5.4. Signal handlers
in main.py set the latch; TrainingLoop polls it; _finalize_slice uses
last_signal() to tell apart SIGTERM (slurm timeout → relaunch) from
SIGCONT (preempt → no relaunch). Tests must reset the module-level state
between cases since it is a global latch.
"""

import signal

import pytest

from cray_megatron.megatron import stop_flag


@pytest.fixture(autouse=True)
def reset_stop_flag():
    """Stop flag is module-level state. Wipe before and after every test
    so cases don't bleed into each other."""
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
    # _finalize_slice keys its relaunch decision on last_signal(). A
    # second handler invocation that omits the number must not clobber
    # the first one's signal back to None.
    stop_flag.request_stop(signal_number=signal.SIGTERM)
    stop_flag.request_stop()
    assert stop_flag.was_stop_requested() is True
    assert stop_flag.last_signal() == signal.SIGTERM


def test_stop_flag_distinguishes_sigterm_from_sigcont():
    # SIGTERM = slurm slice timeout → relaunch path.
    # SIGCONT = slurm preempt → no relaunch (slurm owns requeue).
    # _finalize_slice (training_loop.py) reads last_signal() to pick.
    stop_flag.request_stop(signal_number=signal.SIGCONT)
    assert stop_flag.last_signal() == signal.SIGCONT
    assert stop_flag.last_signal() != signal.SIGTERM


def test_stop_flag_reset_clears_both():
    stop_flag.request_stop(signal_number=signal.SIGTERM)
    stop_flag.reset()
    assert stop_flag.was_stop_requested() is False
    assert stop_flag.last_signal() is None
