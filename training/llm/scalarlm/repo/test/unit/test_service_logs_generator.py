"""
Unit tests for service_logs_generator.

The Metrics-page log panel needs three fast paths on top of this module:

- `tail=N` — initial jump-to-end without scanning the whole file.
- `starting_byte_offset=B` — resume a live tail after an EOF reconnect.
- `before_byte_offset=B & before_count=C` — scrollback: C lines strictly
  before B.

`line_number` is a monotonic label, not a global file index; the client
drives the counter. `byte_offset` + `next_offset` are the canonical
position tokens on the wire.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from cray_infra.api.fastapi.health import service_logs_generator as gen_mod


@pytest.fixture
def log_file(tmp_path):
    # 100 lines, each "line-<i>" plus trailing newline. `line-9` is 6 bytes;
    # `line-10` is 7 bytes, etc — intentionally heterogeneous so byte math
    # exercises variable-length records.
    p = tmp_path / "api.log"
    p.write_text("\n".join(f"line-{i}" for i in range(100)) + "\n")
    return p


@pytest.fixture
def resolve(log_file):
    with patch(
        "cray_infra.api.fastapi.health.service_logs_generator.get_service_log_file",
        return_value=str(log_file),
    ):
        yield


def _drain(agen) -> list[dict]:
    async def pump():
        out = []
        async for chunk in agen:
            out.append(json.loads(chunk))
        return out
    return asyncio.new_event_loop().run_until_complete(pump())


def _line_text_for(i: int) -> str:
    return f"line-{i}"


# ---- find_tail_offset ----------------------------------------------------


def test_find_tail_offset_zero(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert gen_mod.find_tail_offset(str(p), 5) == 0


def test_find_tail_offset_exact(log_file):
    # Last line starts at the byte after the 99th newline — i.e. the
    # (file_size - len("line-99\n")).
    full = log_file.read_text()
    expected = len(full) - len("line-99\n")
    assert gen_mod.find_tail_offset(str(log_file), 1) == expected


def test_find_tail_offset_larger_than_file_returns_zero(log_file):
    assert gen_mod.find_tail_offset(str(log_file), 10_000) == 0


def test_find_tail_offset_no_trailing_newline(tmp_path):
    p = tmp_path / "noeol.log"
    p.write_text("a\nb\nc")  # three lines, last missing \n
    # tail=2 → start of "b" = 2.
    assert gen_mod.find_tail_offset(str(p), 2) == 2
    # tail=1 → start of "c" = 4.
    assert gen_mod.find_tail_offset(str(p), 1) == 4


def test_find_tail_offset_crosses_block_boundary(tmp_path):
    # Force the backward scan to walk through multiple blocks by using
    # a tiny block_size.
    p = tmp_path / "many.log"
    p.write_text("\n".join(str(i) for i in range(200)) + "\n")
    off = gen_mod.find_tail_offset(str(p), 3, block_size=16)
    # Expected: start of "197\n".
    expected = p.read_text().index("197\n")
    assert off == expected


# ---- find_before_offset --------------------------------------------------


def test_find_before_offset_returns_c_lines_earlier(log_file):
    full = log_file.read_text()
    # before = start of "line-40\n"
    before = full.index("line-40\n")
    # Walking back 3 lines → start of "line-37".
    off = gen_mod.find_before_offset(str(log_file), before, 3)
    assert off == full.index("line-37\n")


def test_find_before_offset_clamps_to_zero(log_file):
    full = log_file.read_text()
    before = full.index("line-2\n")
    # Asking for 100 lines before "line-2" → starts at 0.
    assert gen_mod.find_before_offset(str(log_file), before, 100) == 0


# ---- generator: tail -----------------------------------------------------


def test_tail_yields_last_n_records(resolve):
    records = _drain(gen_mod.service_logs_generator("api", tail=5))
    assert [r["line"] for r in records] == [_line_text_for(i) for i in range(95, 100)]


def test_tail_records_carry_byte_and_next_offsets(resolve, log_file):
    records = _drain(gen_mod.service_logs_generator("api", tail=3))
    # next_offset of record N equals byte_offset of record N+1.
    for i in range(len(records) - 1):
        assert records[i]["next_offset"] == records[i + 1]["byte_offset"]
    # Final record's next_offset should equal filesize.
    assert records[-1]["next_offset"] == log_file.stat().st_size


def test_tail_labels_start_at_starting_line_number(resolve):
    records = _drain(
        gen_mod.service_logs_generator("api", tail=3, starting_line_number=42)
    )
    assert [r["line_number"] for r in records] == [42, 43, 44]


def test_tail_zero_falls_through_to_default(resolve):
    # tail=0 is treated as unset; starting_line_number=0 → stream from byte 0.
    records = _drain(gen_mod.service_logs_generator("api", tail=0))
    assert len(records) == 100
    assert records[0]["line"] == "line-0"


# ---- generator: starting_byte_offset -------------------------------------


def test_resume_by_byte_offset(resolve, log_file):
    full = log_file.read_text()
    offset = full.index("line-97\n")
    records = _drain(
        gen_mod.service_logs_generator("api", starting_byte_offset=offset)
    )
    assert [r["line"] for r in records] == ["line-97", "line-98", "line-99"]


def test_resume_by_byte_offset_with_label_base(resolve, log_file):
    full = log_file.read_text()
    offset = full.index("line-50\n")
    records = _drain(
        gen_mod.service_logs_generator(
            "api", starting_byte_offset=offset, starting_line_number=50
        )
    )
    assert records[0]["line_number"] == 50
    assert records[0]["byte_offset"] == offset


# ---- generator: before window (scrollback) -------------------------------


def test_before_window_returns_c_records_strictly_before_B(resolve, log_file):
    full = log_file.read_text()
    before = full.index("line-40\n")
    records = _drain(
        gen_mod.service_logs_generator(
            "api", before_byte_offset=before, before_count=3
        )
    )
    # We asked for 3 lines before line-40 → lines 37, 38, 39.
    assert [r["line"] for r in records] == ["line-37", "line-38", "line-39"]
    # The last record's next_offset must equal `before` exactly — the
    # window is strictly left-open-right-closed on boundaries.
    assert records[-1]["next_offset"] == before


def test_before_window_at_start_of_file(resolve, log_file):
    full = log_file.read_text()
    before = full.index("line-2\n")
    records = _drain(
        gen_mod.service_logs_generator(
            "api", before_byte_offset=before, before_count=100
        )
    )
    # Asking for more than the file has before line-2 → clamp to start:
    # lines 0, 1.
    assert [r["line"] for r in records] == ["line-0", "line-1"]


# ---- generator: limit ----------------------------------------------------


def test_limit_caps_records(resolve, log_file):
    full = log_file.read_text()
    offset = full.index("line-10\n")
    records = _drain(
        gen_mod.service_logs_generator(
            "api", starting_byte_offset=offset, limit=4
        )
    )
    assert [r["line"] for r in records] == [
        "line-10",
        "line-11",
        "line-12",
        "line-13",
    ]


# ---- legacy forward-skip -------------------------------------------------


def test_legacy_starting_line_number(resolve):
    # Preserved for backward compat / curl use. Walks from 0.
    records = _drain(
        gen_mod.service_logs_generator("api", starting_line_number=97)
    )
    assert [r["line"] for r in records] == ["line-97", "line-98", "line-99"]
    assert records[0]["line_number"] == 97


def test_zero_start_yields_full_file(resolve):
    records = _drain(gen_mod.service_logs_generator("api"))
    assert len(records) == 100
    assert records[0]["line"] == "line-0"
    assert records[-1]["line"] == "line-99"
