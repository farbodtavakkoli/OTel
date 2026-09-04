"""Module-level signal latch for graceful training shutdown."""

_stop_requested = False
_last_signal = None


def request_stop(signal_number=None):
    global _stop_requested, _last_signal
    _stop_requested = True
    if signal_number is not None:
        _last_signal = signal_number


def was_stop_requested():
    return _stop_requested


def last_signal():
    return _last_signal


def reset():
    global _stop_requested, _last_signal
    _stop_requested = False
    _last_signal = None
