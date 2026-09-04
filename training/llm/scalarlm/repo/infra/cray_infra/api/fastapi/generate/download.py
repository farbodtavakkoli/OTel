from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.group_request_id_to_path import group_request_id_to_path
from cray_infra.api.work_queue.group_request_id_to_status_path import group_request_id_to_status_path
from cray_infra.api.work_queue.group_request_id_to_response_path import group_request_id_to_response_path

from cray_infra.util.get_config import get_config

from fastapi.responses import FileResponse

import json
import time
import asyncio
import logging

logger = logging.getLogger(__name__)


async def download(download_request):
    request_id = download_request.request_id

    config = get_config()

    timeout = config["response_timeout"]

    start_time = time.time()

    max_sleep_time = 2.0
    current_sleep_time = 0.1

    while time.time() - start_time < timeout:
        file_path = group_request_id_to_path(request_id)
        try:
            logger.debug(f"Checking status for request ID {request_id} at {file_path}")
            async with acquire_file_lock(file_path):
                status_path = group_request_id_to_status_path(request_id)
                with open(status_path, "r") as status_file:
                    status = json.load(status_file)
                    logger.debug(f"Status for request ID {request_id}: {status}")
                    if status["status"] == "completed":
                        response_path = group_request_id_to_response_path(request_id)
                        logger.debug(f"Serving response file at {response_path}")
                        return FileResponse(
                            response_path,
                            media_type="application/json",
                            filename=f"{request_id}_response.json",
                        )
                    else:
                        logger.debug(f"Request ID {request_id} not completed yet, retrying...")
                        await asyncio.sleep(current_sleep_time)
                        current_sleep_time = min(current_sleep_time * 2, max_sleep_time)
        except FileNotFoundError:
            logger.debug(f"File not found for request ID {request_id}, retrying...")
            await asyncio.sleep(current_sleep_time)
            current_sleep_time = min(current_sleep_time * 2, max_sleep_time)
