"""
Unit tests for cray_infra.training.slurm_capacity.

These back the cluster-capacity numbers shown on the metrics page. The
earlier NFS-discovery implementation over-reported GPUs (raw host
device_count) and nodes (stale JSON files from rescheduled pods);
this layer switches to scontrol as the ground truth.
"""

import subprocess
from unittest.mock import patch

import pytest

from cray_infra.training import slurm_capacity


# Representative `scontrol show nodes` output. Two registered megatron
# pods, each capped at 1 GPU by max_gpus_per_node.
TWO_NODES_ONE_GPU_EACH = """\
NodeName=megatron-0 Arch=x86_64 CoresPerSocket=1
   CPUAlloc=0 CPUEfctv=64 CPUTot=64 CPULoad=0.12
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:1
   NodeAddr=megatron-0.svc State=IDLE

NodeName=megatron-1 Arch=x86_64
   Gres=gpu:1
   State=IDLE
"""

ONE_NODE_FOUR_GPU = """\
NodeName=trainer-0
   Gres=gpu:4
   State=IDLE
"""


def _run_result(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["scontrol", "show", "nodes"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---- get_slurm_nodes -----------------------------------------------------


def test_get_slurm_nodes_parses_multi_node_output():
    with patch("subprocess.run", return_value=_run_result(TWO_NODES_ONE_GPU_EACH)):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == [
        {"name": "megatron-0", "gpus": 1, "state": "IDLE"},
        {"name": "megatron-1", "gpus": 1, "state": "IDLE"},
    ]


def test_get_slurm_nodes_returns_gres_value():
    with patch("subprocess.run", return_value=_run_result(ONE_NODE_FOUR_GPU)):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == [{"name": "trainer-0", "gpus": 4, "state": "IDLE"}]


def test_get_slurm_nodes_handles_no_gres_line():
    # CPU-only node — scontrol omits Gres=gpu.
    with patch("subprocess.run", return_value=_run_result("NodeName=cpu-0\n   State=IDLE\n")):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == [{"name": "cpu-0", "gpus": 0, "state": "IDLE"}]


def test_get_slurm_nodes_returns_empty_when_scontrol_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == []


def test_get_slurm_nodes_returns_empty_when_scontrol_times_out():
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="scontrol", timeout=5),
    ):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == []


def test_get_slurm_nodes_returns_empty_on_nonzero_rc():
    with patch(
        "subprocess.run",
        return_value=_run_result(stdout="", returncode=1, stderr="Connection refused"),
    ):
        nodes = slurm_capacity.get_slurm_nodes()
    assert nodes == []


# ---- count_slurm_nodes ---------------------------------------------------


def test_count_slurm_nodes_matches_pod_count():
    with patch("subprocess.run", return_value=_run_result(TWO_NODES_ONE_GPU_EACH)):
        assert slurm_capacity.count_slurm_nodes() == 2


def test_count_slurm_nodes_falls_back_to_one_without_slurm():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert slurm_capacity.count_slurm_nodes() == 1


# ---- count_slurm_gpus ----------------------------------------------------


def test_count_slurm_gpus_sums_across_nodes():
    with patch("subprocess.run", return_value=_run_result(TWO_NODES_ONE_GPU_EACH)):
        assert slurm_capacity.count_slurm_gpus() == 2


def test_count_slurm_gpus_falls_back_to_cuda_device_count():
    # When slurmctld isn't up (local dev), fall back to torch.cuda.
    with patch("subprocess.run", side_effect=FileNotFoundError), \
        patch("torch.cuda.is_available", return_value=True), \
        patch("torch.cuda.device_count", return_value=2):
        assert slurm_capacity.count_slurm_gpus() == 2


def test_count_slurm_gpus_falls_back_to_zero_on_cpu_only_host():
    with patch("subprocess.run", side_effect=FileNotFoundError), \
        patch("torch.cuda.is_available", return_value=False):
        assert slurm_capacity.count_slurm_gpus() == 0


def test_count_slurm_gpus_reports_slurm_value_not_raw_host():
    # The core bug fix: scontrol caps a 4-GPU host to 1 via Gres=gpu:1
    # because max_gpus_per_node limited it. The count must reflect what
    # SLURM sees, not torch.cuda.device_count.
    with patch("subprocess.run", return_value=_run_result("NodeName=n0\n   Gres=gpu:1\n")), \
        patch("torch.cuda.device_count", return_value=4):
        assert slurm_capacity.count_slurm_gpus() == 1


# ---- is_healthy_state ----------------------------------------------------


@pytest.mark.parametrize(
    "state",
    ["IDLE", "MIXED", "ALLOCATED", "COMPLETING", "RESERVED", "RESUMING"],
)
def test_is_healthy_state_accepts_running_states(state):
    assert slurm_capacity.is_healthy_state(state) is True


@pytest.mark.parametrize(
    "state",
    ["DOWN", "FAIL", "UNKNOWN", "DRAIN", "FUTURE", ""],
)
def test_is_healthy_state_rejects_non_running_states(state):
    assert slurm_capacity.is_healthy_state(state) is False


def test_is_healthy_state_rejects_drain_flag_on_running_state():
    # `IDLE+DRAIN` — node is up but admin is draining it; shouldn't count
    # as healthy from a training perspective.
    assert slurm_capacity.is_healthy_state("IDLE+DRAIN") is False
    assert slurm_capacity.is_healthy_state("MIXED+DOWN") is False


def test_is_healthy_state_rejects_not_responding_suffix():
    # Trailing `*` means slurmctld hasn't heard from the slurmd in a while.
    assert slurm_capacity.is_healthy_state("IDLE*") is False
    assert slurm_capacity.is_healthy_state("MIXED*") is False
