import logging

from cray_infra.training.slurm_capacity import get_slurm_nodes, is_healthy_state

logger = logging.getLogger(__name__)


def get_megatron_health():
    try:
        nodes = get_slurm_nodes()
    except Exception as e:
        logger.exception("get_slurm_nodes raised")
        return {"status": "down", "reason": f"error querying slurm: {e}"}

    if not nodes:
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
