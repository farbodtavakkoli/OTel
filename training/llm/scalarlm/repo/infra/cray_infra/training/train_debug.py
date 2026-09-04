"""Debug switches for the training path (from tensorwavecloud/ScalarLM PR #5).

CRAY_TRAIN_DEBUG=1   -> per-rank stderr tracing of distributed init/collectives
CRAY_FAULT_HANDLER=1 -> faulthandler with a periodic traceback dump, which is how
                        you diagnose a rank wedged inside a collective (the whole
                        job hangs and no Python exception is ever raised).
"""

import os


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def is_train_debug_enabled() -> bool:
    return _env_flag("CRAY_TRAIN_DEBUG")


def is_fault_handler_enabled() -> bool:
    return _env_flag("CRAY_FAULT_HANDLER") or is_train_debug_enabled()
