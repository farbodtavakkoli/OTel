"""
Unit tests for training_logs_generator.

Each training slice (resume) writes its own slurm-<job_id>.out; the generator
concatenates them into one continuously-numbered stream. Ordering must be by
*numeric* job id, not lexicographic — otherwise "slurm-1000.out" stitches
ahead of "slurm-999.out" (because '1' < '9'), putting a later slice before an
earlier one across any digit-width boundary. Because line numbers are assigned
by position in the concatenation, a wrong order also makes the UI's
line-number resume skip the misordered slice.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from cray_infra.training import training_logs_generator as gen_mod
from cray_infra.training.training_logs_generator import (
    _slurm_sort_key,
    training_logs_generator,
)


def _drain(agen) -> list[dict]:
    async def pump():
        out = []
        async for chunk in agen:
            out.append(json.loads(chunk))
        return out

    return asyncio.new_event_loop().run_until_complete(pump())


@pytest.fixture
def job_dir(tmp_path):
    # Two slices straddling the 999 -> 1000 digit-width boundary. Slice 999 is
    # the earlier job (smaller id); slice 1000 is the later resume.
    (tmp_path / "slurm-999.out").write_text("a0\na1\na2\n")
    (tmp_path / "slurm-1000.out").write_text("b0\nb1\n")
    return tmp_path


@pytest.fixture
def resolve(job_dir):
    with patch.object(
        gen_mod, "get_job_directory_for_hash", return_value=str(job_dir)
    ), patch.object(gen_mod, "get_config", return_value={}):
        yield job_dir


# ---- _slurm_sort_key ------------------------------------------------------


def test_sort_key_orders_numerically_across_digit_boundary():
    files = [
        "/jobs/x/slurm-1000.out",
        "/jobs/x/slurm-999.out",
        "/jobs/x/slurm-2.out",
    ]
    ordered = sorted(files, key=_slurm_sort_key)
    assert [p.rsplit("/", 1)[1] for p in ordered] == [
        "slurm-2.out",
        "slurm-999.out",
        "slurm-1000.out",
    ]


def test_sort_key_buckets_non_numeric_last():
    files = [
        "/jobs/x/slurm-1000.out",
        "/jobs/x/slurm-old.out",  # no numeric id at all
        "/jobs/x/slurm-5.out",
    ]
    ordered = sorted(files, key=_slurm_sort_key)
    assert [p.rsplit("/", 1)[1] for p in ordered] == [
        "slurm-5.out",
        "slurm-1000.out",
        "slurm-old.out",
    ]


# ---- end-to-end stitching order ------------------------------------------


def test_logs_stitched_in_chronological_job_id_order(resolve):
    rows = _drain(training_logs_generator("anyhash", starting_line_number=0))

    # Earlier slice (999) first, then later slice (1000); line numbers
    # continuous across the boundary.
    assert [r["line"] for r in rows] == ["a0", "a1", "a2", "b0", "b1"]
    assert [r["line_number"] for r in rows] == [0, 1, 2, 3, 4]


def test_starting_line_number_resumes_into_later_slice(resolve):
    # Resume cursor lands inside the second slice — it must still be reachable,
    # which it only is when the slices are in the right order.
    rows = _drain(training_logs_generator("anyhash", starting_line_number=3))

    assert [r["line"] for r in rows] == ["b0", "b1"]
    assert [r["line_number"] for r in rows] == [3, 4]


def test_missing_log_files_raises(tmp_path):
    with patch.object(
        gen_mod, "get_job_directory_for_hash", return_value=str(tmp_path)
    ), patch.object(gen_mod, "get_config", return_value={}):
        with pytest.raises(FileNotFoundError):
            # The generator does its directory scan eagerly (before returning
            # the async generator), so construction raises.
            training_logs_generator("anyhash", starting_line_number=0)
