from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def metrics():
    """
    Get the current metrics.

    queue_depth is reported as
        SDK pending prompts + streaming_inflight

    Two paths feed it:

      - SDK path (`/v1/generate`): one queue entry per
        `llm.generate(...)` batch carrying `request_count` (number of
        prompts in that batch). `pending_request_count` sums
        `request_count` across every non-acked entry — matches the
        per-prompt fanout the user sees rather than the per-batch
        fanout that `__len__` and `unack_count` measure.
      - OpenAI-streaming path (`/v1/chat/completions`) proxies straight
        to vLLM and never touches SQLite; we count those via the
        in-memory `streaming_inflight` counter (try/finally-balanced
        so it doesn't drift).

    The in-memory `Metrics.queue_depth` running counter is still used
    internally to gate throughput math (epoch_time) and to feed
    instantaneous rates, but it's no longer the surface number — it
    drifts up on any error path that skips finish_work.
    """
    sdk_depth: int | None = None
    try:
        queue = await get_inference_work_queue()
        sdk_depth = await queue.pending_request_count()
    except ValueError as e:
        # persistqueue can raise ValueError when its internal counters
        # have drifted negative. Report 0 rather than the drifted
        # in-memory counter.
        logger.debug("Queue len() drifted negative, reporting 0: %s", e)
        sdk_depth = 0
    except Exception as e:
        logger.debug("Falling back to in-memory queue_depth: %s", e)
    if sdk_depth is not None and sdk_depth < 0:
        sdk_depth = 0
    return get_metrics().get_all_metrics(sdk_queue_depth=sdk_depth)
