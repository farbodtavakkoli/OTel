"""
Per-process map from per-prompt request_id to chat-completions
correlation_id.

The chat-completions handler attaches a `correlation_id` to each
queued request before serializing it to disk. When the worker pulls
the batch and begins dispatching prompts, `fill_work_queue` calls
`stash_correlation_id(request_id, cid)` for each one that carries a
cid. When `update_and_ack` records a result, it calls
`pop_correlation_id(request_id)` and, if non-None, resolves the
ResultRouter future for that cid.

The map is process-local and ephemeral; it is not durable across
restarts. That is fine — on restart the result router is also empty
(clients have all retried or disconnected), so the cid lookup
correctly returns None and no resolve happens. The eventual response
file on disk handles client reconnection via the existing
`/v1/generate/get_results` polling path.

`/v1/generate` requests don't carry a correlation_id; for those,
`pop_correlation_id` always returns None and `update_and_ack`'s hook
becomes a no-op. The two endpoint paths share update_and_ack without
behavioral interference.
"""

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
