from cray_infra.api.fastapi.chat_completions.result_router import (
    get_result_router,
)
from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.correlation_id_map import pop_correlation_id
from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id
from cray_infra.api.work_queue.get_in_memory_results import (
    get_in_memory_results,
    clear_in_memory_results,
)

import time
import json
import logging

logger = logging.getLogger(__name__)

async def update_and_ack(inference_work_queue, request_id, item):
    logger.info(f"Acknowledging request {request_id}")

    group_request_id = get_group_request_id(request_id)
    in_memory_results = await get_in_memory_results(group_request_id)

    # Defensive: the group may have been finalized+cleared between the
    # caller's get_unfinished_result and now. Also covers the case
    # where the caller decided None was fine and we got here anyway
    # (shouldn't happen today, but keeps the ack path crash-free).
    if in_memory_results is None:
        logger.warning(
            "update_and_ack: group state missing for %s; dropping.", request_id
        )
        return

    existing = in_memory_results["results"].get(request_id)
    if existing is not None and existing.get("is_acked"):
        # Was `{id}` (builtin), which logged the id() of the logger.warn
        # method itself — fixed while we're here.
        logger.warning(f"Request {request_id} is already acknowledged")
    else:
        in_memory_results["current_index"] += 1

    in_memory_results["results"][request_id] = item
    in_memory_results["results"][request_id]["is_acked"] = True

    # Resolve the chat-completions ResultRouter future if this
    # request_id was tagged with a correlation_id at fill_work_queue
    # time. /v1/generate requests carry no cid; pop returns None and
    # this becomes a no-op for them. A late completion whose handler
    # already disconnected also pops fine — router.resolve is silent
    # for unregistered cids.
    correlation_id = await pop_correlation_id(request_id)
    if correlation_id is not None:
        get_result_router().resolve(correlation_id, item)
        # Recording the duration here happens at the same instant the
        # router future resolves — close enough to "user-perceived
        # delivery" without instrumenting the chunked-JSON byte writer.
        from cray_infra.generate.metrics import get_metrics

        get_metrics().record_chat_resolved(correlation_id)

    if in_memory_results["current_index"] >= in_memory_results["total_requests"]:
        await finish_work_queue_item(request_id, inference_work_queue, in_memory_results)


async def finish_work_queue_item(request_id, inference_work_queue, in_memory_results):
    group_request_id = get_group_request_id(request_id)
    response_path = group_request_id_to_response_path(group_request_id)

    async with acquire_file_lock(response_path):

        # Save results to disk
        with open(response_path, "w") as response_file:
            json.dump(in_memory_results, response_file)

        # Update the status file
        status_path = group_request_id_to_status_path(group_request_id)

        # Load the existing status JSON – if it is missing or malformed we create a fresh one.
        try:
            with open(status_path, "r") as status_file:
                current_status = json.load(status_file)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(
                "Status file %s missing or malformed – creating fresh status.",
                status_path,
            )
            # Minimal status shape that matches what UI expects.
            current_status = {
                "status": "in_progress",
                "created_at": time.time(),
                "total_requests": in_memory_results.get("total_requests", 0),
                "current_index": 0,
            }

        # Mark the request as completed.
        current_status["status"] = "completed"
        current_status["completed_at"] = time.time()
        current_status["current_index"] = in_memory_results["current_index"]

        with open(status_path, "w") as status_file:
            json.dump(current_status, status_file)

    logger.info(f"Finished processing group request {group_request_id}")
    logger.info(f"Acknowledging work queue item {current_status['work_queue_id']}")

    await inference_work_queue.ack(current_status["work_queue_id"])

    await clear_in_memory_results(group_request_id)
