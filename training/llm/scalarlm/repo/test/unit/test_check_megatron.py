"""
Unit tests for the megatron health check that appears at
GET /v1/health alongside api and vllm.

Contract:
  - All registered nodes healthy → "up"
  - Some unhealthy              → {"status": "down", "reason": "<H>/<T> ..."}
  - Zero registered nodes       → "up" (inference-only deployments)
  - scontrol unreachable        → "up" (same shape; we can't tell the
                                    difference from zero-configured,
                                    and we prefer quiet over false red)
"""

from unittest.mock import patch

from cray_infra.api.fastapi.health.check_megatron import get_megatron_health


def _with_nodes(nodes):
    return patch(
        "cray_infra.api.fastapi.health.check_megatron.get_slurm_nodes",
        return_value=nodes,
    )


def test_all_healthy_returns_up():
    with _with_nodes(
        [
            {"name": "m-0", "gpus": 1, "state": "IDLE"},
            {"name": "m-1", "gpus": 1, "state": "MIXED"},
        ]
    ):
        assert get_megatron_health() == "up"


def test_some_unhealthy_reports_ratio():
    with _with_nodes(
        [
            {"name": "m-0", "gpus": 1, "state": "IDLE"},
            {"name": "m-1", "gpus": 1, "state": "DOWN"},
            {"name": "m-2", "gpus": 1, "state": "IDLE*"},
        ]
    ):
        result = get_megatron_health()
    assert isinstance(result, dict)
    assert result["status"] == "down"
    assert "1/3" in result["reason"]
    # Both unhealthy nodes surface in the reason for operator debugging.
    assert "m-1=DOWN" in result["reason"]
    assert "m-2=IDLE*" in result["reason"]


def test_reason_truncates_many_unhealthy():
    nodes = [
        {"name": f"m-{i}", "gpus": 1, "state": "DOWN"} for i in range(10)
    ]
    with _with_nodes(nodes):
        result = get_megatron_health()
    # Reason shows 3 nodes + a "+N more" summary rather than listing all 10.
    reason = result["reason"]
    assert reason.count("=DOWN") == 3
    assert "+7 more" in reason


def test_zero_nodes_returns_up():
    with _with_nodes([]):
        assert get_megatron_health() == "up"


def test_drain_flag_counts_as_unhealthy():
    # IDLE+DRAIN is a common intentional state (admin draining a node)
    # that we report as unhealthy so the UI flags it.
    with _with_nodes([{"name": "m-0", "gpus": 1, "state": "IDLE+DRAIN"}]):
        result = get_megatron_health()
    assert isinstance(result, dict)
    assert "0/1" in result["reason"]


def test_unknown_state_counts_as_unhealthy():
    with _with_nodes([{"name": "m-0", "gpus": 1, "state": "UNKNOWN"}]):
        assert isinstance(get_megatron_health(), dict)
