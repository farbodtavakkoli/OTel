from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def metrics():
    sdk_depth: int | None = None
    try:
        queue = await get_inference_work_queue()
        sdk_depth = await queue.pending_request_count()
    except ValueError as e:
        logger.debug("Queue len() drifted negative, reporting 0: %s", e)
        sdk_depth = 0
    except Exception as e:
        logger.debug("Falling back to in-memory queue_depth: %s", e)
    if sdk_depth is not None and sdk_depth < 0:
        sdk_depth = 0
    return get_metrics().get_all_metrics(sdk_queue_depth=sdk_depth)
