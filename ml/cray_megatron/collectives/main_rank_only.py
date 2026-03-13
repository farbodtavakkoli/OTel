import logging

from gpu_aware_mpi import get_rank, barrier

logger = logging.getLogger(__name__)

def is_main_rank():
    return get_rank() == 0

def log_if_main_rank(msg):
    """Log info message only on main rank (rank 0)."""
    if get_rank() == 0:
        logger.info(msg)

def main_rank_only(func):
    def wrap_function(*args, **kwargs):
        result = None
        barrier()
        if is_main_rank():
            result = func(*args, **kwargs)
        barrier()
        return result

    return wrap_function
