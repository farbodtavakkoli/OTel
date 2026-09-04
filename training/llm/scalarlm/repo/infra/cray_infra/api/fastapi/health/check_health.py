from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.api.fastapi.health.check_megatron import get_megatron_health
from cray_infra.util.get_config import get_config


async def check_health():
    vllm_health = await get_vllm_health()
    megatron_health = get_megatron_health()
    api_health = "up"
    all_health = get_all_health([api_health, vllm_health, megatron_health])
    return {
        "api": api_health,
        "vllm": vllm_health,
        "megatron": megatron_health,
        "all": all_health,
    }


def get_all_health(healths):
    """
    Aggregate sub-component statuses. A component's status is either the
    string "up" or a dict with `status: "down"`; anything else (or any
    "down") drags the overall health down.
    """
    def is_up(h) -> bool:
        if h == "up":
            return True
        if isinstance(h, dict) and h.get("status") == "up":
            return True
        return False

    ups = sum(1 for h in healths if is_up(h))
    if ups == len(healths):
        return "up"
    if ups == 0:
        return "down"
    return "mixed"


async def get_vllm_health():
    try:
        session = get_global_session()
        config = get_config()
        async with session.get(config["vllm_api_url"] + "/health") as resp:
            assert resp.status == 200
            return "up"
    except Exception as e:
        return {"status": "down", "reason": str(e)}
