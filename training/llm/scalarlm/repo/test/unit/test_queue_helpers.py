import asyncio
import os

import pytest

from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id
from cray_infra.api.work_queue.get_work_item import make_id, strip_request_id


def test_get_group_request_id_splits_on_first_underscore():
    assert get_group_request_id("abc_000000001") == "abc"


def test_get_group_request_id_no_underscore_returns_whole():
    assert get_group_request_id("abc") == "abc"


def test_get_group_request_id_multiple_underscores_takes_first_segment():
    assert get_group_request_id("sha256hex_extra_000000005") == "sha256hex"


def test_make_id_zero_pads_to_nine_digits():
    assert make_id("abc", 0) == "abc_000000000"
    assert make_id("abc", 5) == "abc_000000005"
    assert make_id("abc", 999999999) == "abc_999999999"


def test_make_id_handles_large_indices():
    out = make_id("abc", 1234567890)
    assert out.startswith("abc_")
    assert out.endswith("1234567890")


def test_strip_request_id_drops_json_suffix_and_dir():
    assert strip_request_id("/app/cray/inference_requests/abc123.json") == "abc123"


def test_strip_request_id_handles_bare_filename():
    assert strip_request_id("abc123.json") == "abc123"


@pytest.mark.asyncio
async def test_acquire_file_lock_creates_and_removes_sidecar(tmp_path):
    target = tmp_path / "data.json"
    target.write_text("{}")
    lock_path = tmp_path / "data.json.lock"

    async with acquire_file_lock(str(target), timeout=1, poll_interval=0.01):
        assert lock_path.exists()

    assert not lock_path.exists()


@pytest.mark.asyncio
async def test_acquire_file_lock_serializes_concurrent_holders(tmp_path):
    target = tmp_path / "data.json"
    target.write_text("{}")

    order = []

    async def hold(label: str, seconds: float):
        async with acquire_file_lock(str(target), timeout=2, poll_interval=0.01):
            order.append(f"{label}-in")
            await asyncio.sleep(seconds)
            order.append(f"{label}-out")

    await asyncio.gather(hold("a", 0.05), hold("b", 0.05))

    for label in ("a", "b"):
        in_idx = order.index(f"{label}-in")
        assert order[in_idx + 1] == f"{label}-out", order


@pytest.mark.asyncio
async def test_acquire_file_lock_times_out_when_held(tmp_path):
    target = tmp_path / "data.json"
    target.write_text("{}")
    (tmp_path / "data.json.lock").touch()

    with pytest.raises(TimeoutError):
        async with acquire_file_lock(
            str(target), timeout=0.1, poll_interval=0.01
        ):
            pass


@pytest.mark.asyncio
async def test_acquire_file_lock_releases_on_exception(tmp_path):
    target = tmp_path / "data.json"
    target.write_text("{}")
    lock_path = tmp_path / "data.json.lock"

    with pytest.raises(RuntimeError):
        async with acquire_file_lock(str(target), timeout=1, poll_interval=0.01):
            assert lock_path.exists()
            raise RuntimeError("boom")

    assert not lock_path.exists()
