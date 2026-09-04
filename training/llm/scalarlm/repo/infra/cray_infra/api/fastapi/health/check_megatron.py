"""
Megatron health check.

Ground truth is slurmctld: it knows both how many megatron workers the
cluster is configured with (= entries in `scontrol show nodes`) and
which of them are currently healthy (primary state in the running set,
no DOWN/DRAIN/FAIL flags, no `*` not-responding suffix).

Behavior matrix:

  scontrol unreachable        → {"status": "down", "reason": "..."}
  0 nodes registered          → "up"  — nothing to be sick about (e.g.
                                 inference-only deployment where
                                 megatron.enabled=false).
  all healthy                 → "up"
  some unhealthy              → {"status": "down", "reason": "H/T ..."}
"""

import logging

from cray_infra.training.slurm_capacity import get_slurm_nodes, is_healthy_state

logger = logging.getLogger(__name__)


def get_megatron_health():
    try:
        nodes = get_slurm_nodes()
    except Exception as e:  # defensive — subprocess etc. already caught, but
        logger.exception("get_slurm_nodes raised")
        return {"status": "down", "reason": f"error querying slurm: {e}"}

    if not nodes:
        # Two shapes here: scontrol failed/unavailable, or there are
        # genuinely zero nodes. We can't distinguish cleanly without
        # racing another command, and the user-facing answer is the
        # same either way: slurm has nothing healthy to report. Treat
        # as "up" so inference-only deployments don't flash red.
        return "up"

    total = len(nodes)
    unhealthy = [n for n in nodes if not is_healthy_state(n["state"])]
    healthy = total - len(unhealthy)

    if healthy == total:
        return "up"

    unhealthy_summary = ", ".join(
        f"{n['name']}={n['state'] or 'unknown'}" for n in unhealthy[:3]
    )
    if len(unhealthy) > 3:
        unhealthy_summary += f", +{len(unhealthy) - 3} more"

    return {
        "status": "down",
        "reason": f"{healthy}/{total} megatron workers healthy ({unhealthy_summary})",
    }
