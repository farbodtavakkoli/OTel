"""
Unit tests for the node-memory-reporting bits of discover_clusters.py:
``get_memory_mb`` and ``write_node_config``.
"""

from unittest.mock import patch

import pytest

from cray_infra.slurm.discovery import discover_clusters as dc


_missing = object()
_mib = 1024 * 1024


def _configure_memory_files(
    monkeypatch, tmp_path, meminfo, *, cgroup_v2=_missing, cgroup_v1=_missing
):
    paths = {
        "meminfo_path": tmp_path / "meminfo",
        "cgroup_v2_memory_limit_path": tmp_path / "memory.max",
        "cgroup_v1_memory_limit_path": tmp_path / "memory.limit_in_bytes",
    }
    paths["meminfo_path"].write_text(meminfo)
    if cgroup_v2 is not _missing:
        paths["cgroup_v2_memory_limit_path"].write_text(cgroup_v2)
    if cgroup_v1 is not _missing:
        paths["cgroup_v1_memory_limit_path"].write_text(cgroup_v1)
    for name, path in paths.items():
        monkeypatch.setattr(dc, name, str(path))
    return paths


def test_get_memory_mb_parses_mem_total_without_a_cgroup_controller(
    monkeypatch, tmp_path
):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal:       131072000 kB\nMemFree:        1000 kB\n",
    )
    assert dc.get_memory_mb() == 131072000 // 1024


def test_get_memory_mb_uses_smaller_cgroup_v2_limit(monkeypatch, tmp_path):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v2=str(384 * _mib),
    )
    assert dc.get_memory_mb() == 384


def test_get_memory_mb_uses_smaller_cgroup_v1_limit(monkeypatch, tmp_path):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v1=str(512 * _mib),
    )
    assert dc.get_memory_mb() == 512


def test_get_memory_mb_omits_capacity_when_v2_is_unlimited(monkeypatch, tmp_path):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v2="max\n",
    )
    assert dc.get_memory_mb() is None


def test_get_memory_mb_omits_capacity_for_v1_unlimited_sentinel(monkeypatch, tmp_path):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v1="9223372036854771712\n",
    )
    assert dc.get_memory_mb() is None


def test_get_memory_mb_never_exceeds_mem_total(monkeypatch, tmp_path):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 524288 kB\n",
        cgroup_v2=str(1024 * _mib),
    )
    assert dc.get_memory_mb() == 512


def test_get_memory_mb_returns_none_when_meminfo_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dc, "meminfo_path", str(tmp_path / "missing-meminfo"))
    assert dc.get_memory_mb() is None


def test_get_memory_mb_returns_none_on_permission_error():
    """A restricted container/sandbox can deny reading /proc/meminfo
    with PermissionError rather than FileNotFoundError -- both must be
    treated as "can't determine memory", not propagate as a crash."""
    with patch("builtins.open", side_effect=PermissionError):
        assert dc.get_memory_mb() is None


@pytest.mark.parametrize(
    "meminfo",
    [
        "MemTotal: not-a-number kB\n",
        "MemTotal: 0 kB\n",
        "MemTotal: -1024 kB\n",
        "MemTotal: 1024\n",
        "MemTotal: 1024 KB\n",
        "MemTotal: 1024 kB extra\n",
        "MemTotal: 9007199254740992 kB\n",
    ],
)
def test_get_memory_mb_rejects_malformed_or_out_of_range_memtotal(
    monkeypatch, tmp_path, meminfo
):
    _configure_memory_files(monkeypatch, tmp_path, meminfo)
    assert dc.get_memory_mb() is None


def test_get_memory_mb_rejects_positive_capacity_below_one_mib(monkeypatch, tmp_path):
    _configure_memory_files(monkeypatch, tmp_path, "MemTotal: 1023 kB\n")
    assert dc.get_memory_mb() is None


@pytest.mark.parametrize(
    "limit",
    ["", "not-a-number", "-1", "0", "max extra", "9223372036854775808"],
)
def test_get_memory_mb_rejects_invalid_cgroup_v2_limits(monkeypatch, tmp_path, limit):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v2=limit,
    )
    assert dc.get_memory_mb() is None


@pytest.mark.parametrize("limit", ["max", "-1", "0", "not-a-number"])
def test_get_memory_mb_rejects_invalid_cgroup_v1_limits(monkeypatch, tmp_path, limit):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v1=limit,
    )
    assert dc.get_memory_mb() is None


def test_get_memory_mb_does_not_fall_back_when_v2_limit_is_malformed(
    monkeypatch, tmp_path
):
    _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v2="malformed",
        cgroup_v1=str(256 * _mib),
    )
    assert dc.get_memory_mb() is None


def test_get_memory_mb_returns_none_when_cgroup_limit_is_unreadable(
    monkeypatch, tmp_path
):
    paths = _configure_memory_files(
        monkeypatch,
        tmp_path,
        "MemTotal: 1048576 kB\n",
        cgroup_v2=str(512 * _mib),
    )
    real_open = open

    def guarded_open(path, *args, **kwargs):
        if str(path) == str(paths["cgroup_v2_memory_limit_path"]):
            raise PermissionError("restricted cgroup")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=guarded_open):
        assert dc.get_memory_mb() is None


def _node(**overrides):
    node = {
        "hostname": "node1",
        "cpu_count": 64,
        "gpu_count": 0,
    }
    node.update(overrides)
    return node


def _fake_get_config(**overrides):
    config = {"max_gpus_per_node": 8}
    config.update(overrides)
    return config


def test_write_node_config_includes_real_memory_when_present():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=257723))
    assert "RealMemory=257723" in line
    assert line.startswith("NodeName=node1 CPUs=64 RealMemory=257723")


def test_write_node_config_omits_real_memory_when_missing():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=None))
    assert "RealMemory=" not in line


@pytest.mark.parametrize(
    "memory_mb",
    [0, -1, True, "257723", (1 << 63) // _mib],
)
def test_write_node_config_omits_invalid_real_memory(memory_mb):
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=memory_mb))
    assert "RealMemory=" not in line


def test_write_node_config_no_double_spaces_when_fields_missing():
    """Both RealMemory (no memory_mb) and Gres (no GPUs) can be absent
    at once; the rendered line must not have doubled-up whitespace from
    joining empty fields."""
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=None, gpu_count=0))
    assert "  " not in line
    assert line == "NodeName=node1 CPUs=64 State=UNKNOWN\n"


def test_write_node_config_includes_gres_when_gpus_present():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=257723, gpu_count=6))
    assert line == "NodeName=node1 CPUs=64 RealMemory=257723 Gres=gpu:6 State=UNKNOWN\n"
