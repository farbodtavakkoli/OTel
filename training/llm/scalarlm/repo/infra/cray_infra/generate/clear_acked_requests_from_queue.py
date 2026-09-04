from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.inference_work_queue import (
    get_file_backed_inference_work_queue,
)
from cray_infra.util.get_config import get_config

import os
import time
import logging

logger = logging.getLogger(__name__)

ready_worker_idle_start_time = None


async def clear_acked_requests_from_queue():
    inference_work_queue = get_file_backed_inference_work_queue()

    starting_size = len(inference_work_queue)

    await inference_work_queue.clear_acked_data()

    ending_size = len(inference_work_queue)

    logger.info(f"Cleared {starting_size - ending_size} acked requests from the queue.")

    await restart_unacked_requests_from_queue(inference_work_queue)


async def restart_unacked_requests_from_queue(inference_work_queue):
    """
    Watchdog: requests that have been in unack state longer than the
    ack timeout get nack'd back into pending so a healthy worker can
    pick them up. Before we nack, check whether the response file is
    already on disk — that means the worker finished the job but
    hasn't (or couldn't) complete the finish_work round-trip yet.
    Nacking those would re-submit work that's already done and flood
    vLLM; ack them instead so they leave the queue cleanly.
    """
    global ready_worker_idle_start_time

    config = get_config()

    unacked_requests = await inference_work_queue.get_unacked_requests()

    resumed_count = 0
    acked_count = 0

    for request in unacked_requests:
        # If the worker already produced a response, ack this row instead
        # of restarting it. Prevents the watchdog from re-queueing work
        # that's finished but stuck between "result written" and
        # "finish_work returned 200."
        item_path = request["data"].get("path")
        if item_path:
            response_path = group_request_id_to_response_path(
                _strip_request_id(item_path)
            )
            if os.path.exists(response_path):
                try:
                    await inference_work_queue.ack(request["id"])
                    acked_count += request["data"].get("request_count", 1)
                except Exception as e:
                    logger.debug(
                        "Failed to ack already-responded request %s: %s",
                        request["id"],
                        e,
                    )
                continue

        time_since_submit = time.time() - request["data"]["timestamp"]
        ready_worker_idle_time = (
            0
            if ready_worker_idle_start_time is None
            else time.time() - ready_worker_idle_start_time
        )

        if (config["inference_work_queue_ack_timeout"] < time_since_submit) and (
            ready_worker_idle_time > config["inference_work_queue_idle_time"]
        ):
            await inference_work_queue.resume_unack_task(id=request["id"])
            resumed_count += request["data"].get("request_count", 1)

    if acked_count or resumed_count:
        logger.info(
            "Restart watchdog: acked %d already-responded, restarted %d stale requests.",
            acked_count,
            resumed_count,
        )


def _strip_request_id(item_path: str) -> str:
    # Mirrors get_work_item.strip_request_id; duplicated here to avoid
    # a circular import between the work queue and the generate package.
    base_name = os.path.basename(item_path)
    request_id, _ = os.path.splitext(base_name)
    return request_id


async def worker_ready():
    global ready_worker_idle_start_time
    ready_worker_idle_start_time = time.time()


async def worker_not_ready():
    global ready_worker_idle_start_time
    ready_worker_idle_start_time = None
