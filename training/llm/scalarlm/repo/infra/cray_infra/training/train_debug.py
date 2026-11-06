import os


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def is_train_debug_enabled() -> bool:
    return _env_flag("CRAY_TRAIN_DEBUG")


def is_fault_handler_enabled() -> bool:
    return _env_flag("CRAY_FAULT_HANDLER") or is_train_debug_enabled()
