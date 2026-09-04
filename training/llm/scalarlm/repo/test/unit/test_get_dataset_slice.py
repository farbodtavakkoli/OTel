"""
Unit tests for infra/cray_infra/training/get_dataset_slice.py.

Contract under test: ui/docs/dataset-viewer.md §API. The endpoint is
thin — a pagination + substring-filter wrapper around dataset.jsonlines
— so these tests stand in for a component-layer test of the HTTP route.
"""

import json
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.training import get_dataset_slice as slice_mod
from cray_infra.training.get_dataset_slice import get_dataset_slice


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "deadbeef"
    d.mkdir()
    return d


def _write_dataset(job_dir, rows):
    path = os.path.join(job_dir, "dataset.jsonlines")
    with open(path, "w") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def resolve(job_dir):
    with patch(
        "cray_infra.training.get_dataset_slice.get_job_directory_for_hash",
        return_value=str(job_dir),
    ):
        yield


# ---- happy-path pagination -----------------------------------------------


def test_unfiltered_slice_returns_first_page(job_dir, resolve):
    _write_dataset(
        job_dir, [{"input": f"in-{i}", "output": f"out-{i}"} for i in range(10)]
    )

    result = get_dataset_slice("deadbeef", offset=0, limit=3)

    assert result["total"] == 10
    assert result["matched"] == 10
    assert result["offset"] == 0
    assert result["limit"] == 3
    assert result["truncated"] is False
    assert [e["index"] for e in result["examples"]] == [0, 1, 2]
    assert result["examples"][0]["input"] == "in-0"
    assert result["examples"][0]["output"] == "out-0"


def test_unfiltered_slice_respects_offset(job_dir, resolve):
    _write_dataset(job_dir, [{"input": f"i{i}", "output": f"o{i}"} for i in range(10)])

    result = get_dataset_slice("deadbeef", offset=7, limit=5)

    # Only 3 rows past offset=7 in a 10-row file.
    assert [e["index"] for e in result["examples"]] == [7, 8, 9]
    assert result["total"] == 10


def test_empty_file_returns_zero_total(job_dir, resolve):
    _write_dataset(job_dir, [])
    result = get_dataset_slice("deadbeef")
    assert result["total"] == 0
    assert result["examples"] == []


# ---- substring filter ----------------------------------------------------


def test_filter_matches_case_insensitive(job_dir, resolve):
    _write_dataset(
        job_dir,
        [
            {"input": "Hello world", "output": "hi"},
            {"input": "goodbye", "output": "bye"},
            {"input": "HELLO again", "output": "wave"},
        ],
    )

    result = get_dataset_slice("deadbeef", q="hello")

    assert result["total"] == 3
    assert result["matched"] == 2
    assert [e["index"] for e in result["examples"]] == [0, 2]


def test_filter_offset_applies_to_matches_not_lines(job_dir, resolve):
    rows = [{"input": "miss"}] * 20 + [{"input": "hit"}] * 5 + [{"input": "miss"}] * 5
    _write_dataset(job_dir, rows)

    # Skip the first 2 matches, take the next 10 (only 3 remain).
    result = get_dataset_slice("deadbeef", offset=2, limit=10, q="hit")

    assert result["matched"] == 5
    assert [e["index"] for e in result["examples"]] == [22, 23, 24]


def test_filter_with_no_matches_returns_empty(job_dir, resolve):
    _write_dataset(job_dir, [{"input": "a"}, {"input": "b"}])
    result = get_dataset_slice("deadbeef", q="zzz")
    assert result["matched"] == 0
    assert result["examples"] == []


# ---- robustness ----------------------------------------------------------


def test_malformed_line_is_counted_but_not_parsed(job_dir, resolve):
    path = os.path.join(job_dir, "dataset.jsonlines")
    with open(path, "w") as f:
        f.write(json.dumps({"input": "ok"}) + "\n")
        f.write("{not json at all\n")
        f.write(json.dumps({"input": "ok2"}) + "\n")

    result = get_dataset_slice("deadbeef")

    assert result["total"] == 3
    assert [e["index"] for e in result["examples"]] == [0, 1, 2]
    # The middle row surfaces the parse error rather than vanishing.
    assert "__parse_error__" in result["examples"][1]["raw"]


def test_long_field_is_clipped(job_dir, resolve, monkeypatch):
    # Shrink the clip so the test doesn't need to write megabytes.
    monkeypatch.setattr(slice_mod, "MAX_FIELD_BYTES", 16)

    huge = "x" * 1000
    _write_dataset(job_dir, [{"input": huge, "output": "short"}])

    result = get_dataset_slice("deadbeef")

    ex = result["examples"][0]
    assert len(ex["input"]) == 16
    assert ex["output"] == "short"
    assert ex["truncated_fields"] == ["input"]


def test_truncated_flag_set_when_scan_exceeds_budget(job_dir, resolve, monkeypatch):
    # Force the scan-budget tripwire to fire after a few rows.
    monkeypatch.setattr(slice_mod, "MAX_MATCH_SCAN_BYTES", 64)

    _write_dataset(
        job_dir,
        [{"input": "x" * 50, "output": "hit"} for _ in range(50)],
    )

    result = get_dataset_slice("deadbeef", q="hit")

    assert result["truncated"] is True
    assert result["total"] < 50


# ---- errors --------------------------------------------------------------


def test_missing_dataset_returns_404(job_dir, resolve):
    # Directory exists (resolve fixture), but no dataset.jsonlines inside.
    with pytest.raises(HTTPException) as exc:
        get_dataset_slice("deadbeef")
    assert exc.value.status_code == 404
    assert "dataset not found" in exc.value.detail


def test_negative_offset_rejected(job_dir, resolve):
    _write_dataset(job_dir, [{"input": "x"}])
    with pytest.raises(HTTPException) as exc:
        get_dataset_slice("deadbeef", offset=-1)
    assert exc.value.status_code == 400


def test_limit_above_cap_rejected(job_dir, resolve):
    _write_dataset(job_dir, [{"input": "x"}])
    with pytest.raises(HTTPException) as exc:
        get_dataset_slice("deadbeef", limit=10_000)
    assert exc.value.status_code == 400


def test_limit_zero_rejected(job_dir, resolve):
    _write_dataset(job_dir, [{"input": "x"}])
    with pytest.raises(HTTPException) as exc:
        get_dataset_slice("deadbeef", limit=0)
    assert exc.value.status_code == 400
