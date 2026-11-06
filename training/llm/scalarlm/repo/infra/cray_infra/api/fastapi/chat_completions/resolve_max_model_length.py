import asyncio
import logging
from typing import Any

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


_cache: dict[str, int] = {}
_cache_lock = asyncio.Lock()


async def resolve_max_model_length(model: str) -> int:
    cached = _cache.get(model)
    if cached is not None:
        return cached

    async with _cache_lock:
        cached = _cache.get(model)
        if cached is not None:
            return cached

        resolved = await _query_vllm_for_max_model_len(model)
        if resolved is None:
            return _fallback_from_config()
        _cache[model] = resolved
        return resolved


def reset_cache_for_tests() -> None:
    _cache.clear()


async def _query_vllm_for_max_model_len(model: str) -> int | None:
    config = get_config()
    url = config["vllm_api_url"] + "/v1/models"
    try:
        session = get_global_session()
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning(
                    "resolve_max_model_length: vLLM /v1/models returned %s; "
                    "falling back to config",
                    resp.status,
                )
                return None
            payload = await resp.json()
    except Exception as exc:
        logger.warning(
            "resolve_max_model_length: HTTP error querying %s: %s; "
            "falling back to config",
            url,
            exc,
        )
        return None

    return _extract_max_model_len(payload, model)


def _extract_max_model_len(payload: Any, model: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None

    by_id: dict[str, dict] = {
        entry["id"]: entry
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }

    visited: set[str] = set()
    current_id: str | None = model
    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        entry = by_id.get(current_id)
        if entry is None:
            return None

        own_value = _coerce_max_len(entry)
        if own_value is not None:
            return own_value

        parent = entry.get("parent")
        current_id = parent if isinstance(parent, str) and parent else None

    return None


def _coerce_max_len(entry: dict) -> int | None:
    for field in ("max_model_len", "max_position_embeddings"):
        value = entry.get(field)
        if isinstance(value, int) and value > 0:
            return value
        try:
            coerced = int(value)
            if coerced > 0:
                return coerced
        except (TypeError, ValueError):
            continue
    return None


def _fallback_from_config() -> int:
    try:
        return int(get_config().get("max_model_length", 0))
    except (TypeError, ValueError):
        return 0
