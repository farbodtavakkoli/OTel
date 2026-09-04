"""
Stream a service log file as newline-delimited JSON.

Protocol (see health_router.py):

    GET /v1/health/logs/{service}
        ?tail=N                  # initial jump-to-end: seek to last N lines
        ?starting_byte_offset=B  # resume from a previous record's next_offset
        ?before_byte_offset=B
        &before_count=C          # scrollback: last C lines strictly before B
        ?starting_line_number=N  # (legacy) forward-skip; slow on large N
        ?limit=M                 # cap records in this response

Each yielded record is:

    {
      "line":         "<text>",
      "line_number":  <monotonic label — see below>,
      "byte_offset":  <start of this line in the file>,
      "next_offset":  <byte_offset + len(raw line including newline)>
    }

Positioning is by byte offset. `line_number` is a *label* that increments
across a tailing session but isn't guaranteed to match the file's global
line count — once we seek by byte we no longer know what global line
we landed on. The client drives its own counter.

Tailing logic lives in `log_file_tailer.tail_log_file`, shared with
the publish-job log endpoint.
"""

import logging
import os

from cray_infra.api.fastapi.health.log_file_tailer import (
    find_before_offset,
    find_tail_offset,
    tail_log_file,
)
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

# Re-exported so existing callers / tests importing from this module
# (e.g. `cray_infra.api.fastapi.health.service_logs_generator.find_tail_offset`)
# don't need to rewrite their imports.
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
