"""
Path-based NDJSON log tailer shared by every "tail this file" endpoint.

The protocol — query params + record shape — is documented in
`service_logs_generator.py`. That module wraps this helper with
service-name resolution; `publish_logs.py` does the same for publish
job logs. Anything new that wants the same byte-offset-resume
semantics should call `tail_log_file` directly with a resolved path.

The body of this file used to live inside `service_logs_generator.py`
verbatim — it moved here unchanged so both consumers stay byte-for-byte
identical in their tailing behaviour.
"""

import json
import logging
import os

import aiofiles

logger = logging.getLogger(__name__)

_NEWLINE = 0x0A  # ord(b"\n")
_BLOCK_SIZE = 64 * 1024


def tail_log_file(
    path: str,
    starting_line_number: int = 0,
    starting_byte_offset: int | None = None,
    tail: int | None = None,
    limit: int | None = None,
    before_byte_offset: int | None = None,
    before_count: int | None = None,
):
    """
    Async generator yielding NDJSON records from `path`. See the
    `service_logs_generator` docstring for the wire shape.
    """
    start_offset, end_offset = _resolve_range(
        path,
        starting_line_number=starting_line_number,
        starting_byte_offset=starting_byte_offset,
        tail=tail,
        before_byte_offset=before_byte_offset,
        before_count=before_count,
    )
    line_number_base = max(0, starting_line_number)

    logger.info(
        "Serving log %s from byte %d (end=%s, label_base=%d, limit=%s)",
        path,
        start_offset,
        end_offset,
        line_number_base,
        limit,
    )

    async def generate():
        yielded = 0
        line_number = line_number_base
        offset = start_offset
        async with aiofiles.open(path, mode="rb") as f:
            await f.seek(start_offset)
            while True:
                if limit is not None and yielded >= limit:
                    break
                if end_offset is not None and offset >= end_offset:
                    break
                raw = await f.readline()
                if not raw:
                    break
                next_offset = offset + len(raw)
                if end_offset is not None and next_offset > end_offset:
                    break
                line = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
                yield (
                    json.dumps(
                        {
                            "line": line,
                            "line_number": line_number,
                            "byte_offset": offset,
                            "next_offset": next_offset,
                        }
                    )
                    + "\n"
                )
                offset = next_offset
                line_number += 1
                yielded += 1

    return generate()


def _resolve_range(
    path: str,
    *,
    starting_line_number: int,
    starting_byte_offset: int | None,
    tail: int | None,
    before_byte_offset: int | None,
    before_count: int | None,
) -> tuple[int, int | None]:
    """Pick start_offset (and optional end_offset for before-window
    requests). Precedence: before-window > tail > byte_offset > line_number."""
    if (
        before_byte_offset is not None
        and before_count is not None
        and before_count > 0
    ):
        return find_before_offset(path, before_byte_offset, before_count), before_byte_offset

    if tail is not None and tail > 0:
        return find_tail_offset(path, tail), None

    if starting_byte_offset is not None and starting_byte_offset > 0:
        return starting_byte_offset, None

    if starting_line_number > 0:
        return _forward_skip_offset(path, starting_line_number), None

    return 0, None


def find_tail_offset(
    path: str, tail: int, block_size: int = _BLOCK_SIZE
) -> int:
    if tail <= 0:
        return 0
    end = os.path.getsize(path)
    if end == 0:
        return 0
    with open(path, "rb") as f:
        f.seek(end - 1)
        last = f.read(1)
    scan_end = end - 1 if last == b"\n" else end
    if scan_end <= 0:
        return 0
    return _find_offset_after_nth_newline_backward(path, scan_end, tail, block_size)


def find_before_offset(
    path: str, before: int, count: int, block_size: int = _BLOCK_SIZE
) -> int:
    if count <= 0 or before <= 0:
        return 0
    return _find_offset_after_nth_newline_backward(
        path, before, count + 1, block_size
    )


def _find_offset_after_nth_newline_backward(
    path: str, start_pos: int, k: int, block_size: int
) -> int:
    if k <= 0 or start_pos <= 0:
        return 0
    with open(path, "rb") as f:
        pos = start_pos
        remaining = k
        while pos > 0:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            for i in range(len(chunk) - 1, -1, -1):
                if chunk[i] == _NEWLINE:
                    remaining -= 1
                    if remaining == 0:
                        return pos + i + 1
    return 0


def _forward_skip_offset(path: str, starting_line_number: int) -> int:
    if starting_line_number <= 0:
        return 0
    with open(path, "rb", buffering=_BLOCK_SIZE) as f:
        seen = 0
        while seen < starting_line_number:
            line = f.readline()
            if not line:
                return f.tell()
            seen += 1
        return f.tell()
