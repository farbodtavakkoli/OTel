import logging
import os

from cray_infra.api.fastapi.health.log_file_tailer import (
    find_before_offset,
    find_tail_offset,
    tail_log_file,
)
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

__all__ = [
    "service_logs_generator",
    "get_service_log_file",
    "find_tail_offset",
    "find_before_offset",
]


def service_logs_generator(
    service_name: str,
    starting_line_number: int = 0,
    starting_byte_offset: int | None = None,
    tail: int | None = None,
    limit: int | None = None,
    before_byte_offset: int | None = None,
    before_count: int | None = None,
):
    return tail_log_file(
        get_service_log_file(service_name),
        starting_line_number=starting_line_number,
        starting_byte_offset=starting_byte_offset,
        tail=tail,
        limit=limit,
        before_byte_offset=before_byte_offset,
        before_count=before_count,
    )


def get_service_log_file(service_name: str) -> str:
    config = get_config()
    log_base_dir = config["log_directory"]

    service_log_file = os.path.join(log_base_dir, f"{service_name}.log")

    if not os.path.isfile(service_log_file):
        raise FileNotFoundError(f"Log file for service '{service_name}' not found.")

    return service_log_file
