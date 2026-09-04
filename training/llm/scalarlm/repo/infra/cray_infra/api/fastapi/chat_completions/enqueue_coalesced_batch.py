"""
Coalescer flush callback: turn one in-memory batch into one queue row.

The coalescer hands us a list of `(request_dict, correlation_id)`
tuples. We serialize the request dicts to a JSON file under
`upload_base_path`, name the file by the SHA-256 of its contents
(matching the existing `/v1/generate` upload path's dedup property),
and call `push_into_queue` to register a single SQLite row pointing
at that file.

The correlation id for each request is already inside the request
dict (the handler put it there before submitting to the coalescer)
so the worker can preserve it across the queue boundary; we don't
re-store it separately here.

See docs/openai-chat-completions-queue.md §6.3.
"""

import hashlib
import json
import os
from typing import Any, List, Tuple

from cray_infra.api.work_queue.push_into_queue import push_into_queue
from cray_infra.util.get_config import get_config


async def enqueue_coalesced_batch(batch: List[Tuple[Any, str]]) -> None:
    if not batch:
        raise ValueError("enqueue_coalesced_batch received an empty batch")

    requests = [request for request, _cid in batch]
    contents = json.dumps(requests).encode("utf-8")
    contents_hash = hashlib.sha256(contents).hexdigest()

    config = get_config()
    base_path = config["upload_base_path"]
    # Lazy-create the upload dir so a pod that's never received a
    # /v1/generate upload (which calls os.makedirs via get_request_path)
    # doesn't crash every chat-completions request with FileNotFoundError.
    # /v1/generate's upload.py and tee_streaming_to_disk.py use the same
    # pattern.
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, f"{contents_hash}.json")

    with open(path, "wb") as handle:
        handle.write(contents)

    await push_into_queue(len(requests), path)

    # Record the realized batch size for the chat_batch_size_p50/p99
    # histogram (docs §13). Imported lazily so this module remains
    # importable in unit tests that don't pull in metrics state.
    from cray_infra.generate.metrics import get_metrics

    get_metrics().record_chat_batch_size(len(requests))
