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
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, f"{contents_hash}.json")

    with open(path, "wb") as handle:
        handle.write(contents)

    await push_into_queue(len(requests), path)

    from cray_infra.generate.metrics import get_metrics

    get_metrics().record_chat_batch_size(len(requests))
