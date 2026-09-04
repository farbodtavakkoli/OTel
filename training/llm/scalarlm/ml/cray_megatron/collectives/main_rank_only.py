from functools import wraps
from cray_infra.training.distributed import get_rank, barrier
from cray_infra.training.train_debug import is_train_debug_enabled

import logging
import sys
import time

logger = logging.getLogger(__name__)


def _dist_ready() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_initialized()
    except Exception:
        return False


def _trace_main_rank_only(msg: str) -> None:
    """Emit a CRAY_TRAIN_DEBUG trace line identifying which barrier we are at."""
    if not is_train_debug_enabled():
        return
    if _dist_ready() and get_rank() != 0:
        return
    rank = get_rank() if _dist_ready() else "?"
    sys.stderr.write(f"[rank={rank}] main_rank_only [{time.monotonic():.3f}]: {msg}\n")
    sys.stderr.flush()

def is_main_rank():
    return get_rank() == 0


def log_if_main_rank(msg):
    """Log once for the whole job instead of once per rank."""
    if is_main_rank():
        logger.info(msg)


_in_main_rank_only = False

def main_rank_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _in_main_rank_only

        if _in_main_rank_only:
            return func(*args, **kwargs)

        _in_main_rank_only = True
        try:
            _trace_main_rank_only(f"{func.__name__}: pre-barrier-1")
            barrier()
            _trace_main_rank_only(
                f"{func.__name__}: post-barrier-1, is_main_rank={is_main_rank()}"
            )
            result = func(*args, **kwargs) if is_main_rank() else None
            _trace_main_rank_only(f"{func.__name__}: pre-barrier-2")
            barrier()
            _trace_main_rank_only(f"{func.__name__}: post-barrier-2")
            return result
        finally:
            _in_main_rank_only = False

    return wrapper
