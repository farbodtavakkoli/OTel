from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.correlation_id_map import stash_correlation_id

from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)

from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)

import asyncio
import json
import os

import logging

logger = logging.getLogger(__name__)

lock = asyncio.Lock()
in_memory_work_queue = []


async def get_work_item(work_queue):
    global in_memory_work_queue
    global lock

    async with lock:
        if not in_memory_work_queue:
            await fill_work_queue(work_queue)

        if not in_memory_work_queue:
            return None, None

        item, id = in_memory_work_queue.pop(0)

    logger.info(f"Dispatching work item {id}")

    return item, id

async def get_work_item_no_wait(work_queue):
    global in_memory_work_queue
    global lock

    async with lock:
        if not in_memory_work_queue:
            return None, None

        item, id = in_memory_work_queue.pop(0)

    logger.info(f"Dispatching work item {id}")

    return item, id


async def fill_work_queue(work_queue):
    logger.debug("Filling work queue")

    while True:
        request, id = await work_queue.get()

        if request is None:
            logger.debug("Nothing in the work queue")
            return

        item_path = request["path"]

        group_request_id = strip_request_id(item_path)

        # Skip if already processed. Acking the duplicate is important:
        # `work_queue.get()` popped it in `unack` state, and without an
        # ack it stays checked out in SQLiteAckQueue forever. Duplicates
        # accumulate → the queue's internal counters drift → `len()`
        # eventually returns a negative number ("__len__() should return
        # >= 0") and the emit-metrics path starts failing.
        response_path = group_request_id_to_response_path(group_request_id)

        if os.path.exists(response_path):
            logger.debug(f"Skipping already processed request {group_request_id}")
            try:
                await work_queue.ack(id)
            except Exception as e:
                logger.debug(
                    f"Failed to ack duplicate request {group_request_id}: {e}"
                )
            continue

        async with acquire_file_lock(item_path):
            with open(item_path, "r") as f:
                requests = json.load(f)

            logger.debug(f"Loaded {len(requests)} requests from {item_path} to work queue")

            global in_memory_work_queue
            in_memory_work_queue = [
                (request, make_id(group_request_id, index))
                for index, request in enumerate(requests)
            ]

            # Stash chat-completions correlation_ids from this batch
            # so update_and_ack can later resolve the per-prompt
            # ResultRouter future. Generate-path entries (no cid) are
            # skipped — pop_correlation_id will return None for them
            # and the resolve hook becomes a no-op.
            for index, batch_request in enumerate(requests):
                if not isinstance(batch_request, dict):
                    continue
                cid = batch_request.get("correlation_id")
                if cid:
                    await stash_correlation_id(
                        make_id(group_request_id, index), cid
                    )

            break

def make_id(group_request_id, index):
    return f"{group_request_id}_{index:09d}"

def strip_request_id(item_path):
    base_name = os.path.basename(item_path)
    request_id, _ = os.path.splitext(base_name)
    return request_id
