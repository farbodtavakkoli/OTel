import asyncio
from typing import Dict, Optional

_lock = asyncio.Lock()
_map: Dict[str, str] = {}


async def stash_correlation_id(request_id: str, correlation_id: str) -> None:
    async with _lock:
        _map[request_id] = correlation_id


async def pop_correlation_id(request_id: str) -> Optional[str]:
    async with _lock:
        return _map.pop(request_id, None)
