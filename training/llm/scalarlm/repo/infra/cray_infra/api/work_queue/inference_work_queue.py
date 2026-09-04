
from cray_infra.util.get_config import get_config

import persistqueue

import asyncio
import time
import logging
import json

logger = logging.getLogger(__name__)

# Import observability metrics
try:
    from cray_infra.observability.prometheus_metrics import queue_depth, queue_wait_time_seconds
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Observability metrics not available for work queue")
    METRICS_AVAILABLE = False


class InferenceWorkQueue:
    def __init__(self, path, auto_resume=False):
        self.queue = persistqueue.SQLiteAckQueue(path, auto_resume=auto_resume)
        self.lock = asyncio.Lock()

    async def put(self, request):
        async with self.lock:
            # `timestamp` is read by clear_acked_requests_from_queue to decide
            # when to recycle a stuck unack'd row; kept for backwards
            # compatibility with that consumer. `enqueue_time` is the same
            # value under the new observability-friendly name used by
            # Prometheus queue_wait_time metrics below.
            now = time.time()
            request["timestamp"] = now
            request["enqueue_time"] = now
            request_id = self.queue.put(request)

            # Update queue depth metric
            if METRICS_AVAILABLE:
                try:
                    current_depth = len(self.queue)
                    model = request.get("model", "unknown")
                    queue_depth.labels(model=model).set(current_depth)

                    # Structured log
                    logger.info(json.dumps({
                        "event": "work_queued",
                        "request_id": request_id,
                        "model": model,
                        "queue_depth": current_depth,
                        "trace_id": request.get("trace_context", {}).get("traceparent", "unknown")
                    }))
                except Exception as e:
                    logger.debug(f"Failed to emit queue metrics: {e}")

            return request_id

    async def get(self):
        config = get_config()

        timeout = config["inference_work_queue_timeout"]

        start_time = time.time()

        item = None
        request_id = None

        while time.time() - start_time < timeout:
            try:
                async with self.lock:
                    raw_item = self.queue.get(block=False, raw=True)

                item = raw_item["data"]
                dequeue_time = time.time()
                # Both names set for the same reason as in put(): timestamp
                # is consumed by the ack-timeout recycle path, dequeue_time
                # by the metrics emitter below.
                item["timestamp"] = dequeue_time
                item["dequeue_time"] = dequeue_time
                request_id = raw_item["pqid"]

                # Emit queue wait time metric
                if METRICS_AVAILABLE and item:
                    try:
                        enqueue_time = item.get("enqueue_time", dequeue_time)
                        wait_time = dequeue_time - enqueue_time
                        model = item.get("model", "unknown")

                        queue_wait_time_seconds.labels(model=model).observe(wait_time)

                        # Update queue depth
                        current_depth = len(self.queue)
                        queue_depth.labels(model=model).set(current_depth)

                        # Structured log
                        logger.info(json.dumps({
                            "event": "work_dequeued",
                            "request_id": request_id,
                            "model": model,
                            "queue_wait_time_seconds": round(wait_time, 3),
                            "queue_depth": current_depth,
                            "trace_id": item.get("trace_context", {}).get("traceparent", "unknown")
                        }))
                    except Exception as e:
                        logger.debug(f"Failed to emit queue metrics: {e}")

                break

            except persistqueue.Empty:
                await asyncio.sleep(0.01)

        return item, request_id

    async def get_id(self, id):
        async with self.lock:
            return self.queue.get(block=False, id=id)

    async def get_nowait(self):
        try:
            async with self.lock:
                raw_item = self.queue.get(block=False, raw=True)

            item = raw_item["data"]
            item["timestamp"] = time.time()
            request_id = raw_item["pqid"]

        except persistqueue.Empty:
            item = None
            request_id = None

        return item, request_id

    async def get_if_finished(self, id):
        async with self.lock:
            results = self.queue.queue()

            for result in results:
                if result["id"] == id:
                    if int(result["status"]) == int(persistqueue.AckStatus.acked):
                        return result["data"]

        return None

    async def get_unacked_requests(self):
        async with self.lock:
            results = self.queue.queue()

            unacked_requests = []

            for result in results:
                if int(result["status"]) == int(persistqueue.AckStatus.unack):
                    unacked_requests.append(result)

            return unacked_requests

    async def update(self, id, item):
        async with self.lock:
            return self.queue.update(id=id, item=item)

    async def ack(self, id):
        async with self.lock:
            return self.queue.ack(id=id)

    async def update_and_ack(self, id, item):
        async with self.lock:
            result = self.queue.update(id=id, item=item)
            ack_result = self.queue.ack(id=id)

            return result, ack_result

    async def resume_unack_tasks(self):
        async with self.lock:
            self.queue.resume_unack_tasks()

    async def resume_unack_task(self, id):
        async with self.lock:
            self.queue.nack(id=id)

    async def clear_acked_data(self):
        async with self.lock:
            self.queue.clear_acked_data()

    async def unack_count(self):
        async with self.lock:
            return self.queue.unack_count()

    async def pending_request_count(self) -> int:
        """
        Total number of *prompts* in flight — sums `request_count`
        across every queue entry that's not yet acked.

        persistqueue's `__len__` only counts ready entries and
        `unack_count` only counts claimed-but-not-finished entries.
        Neither reflects the per-prompt fanout that push_into_queue
        records on each entry's payload, so a single
        `llm.generate(["a","b","c"])` looked like 1 / 0 to those
        counters but is 3 in-flight prompts to the user.
        """
        async with self.lock:
            results = self.queue.queue()
            total = 0
            for result in results:
                try:
                    status = int(result.get("status", 0))
                except (TypeError, ValueError):
                    continue
                # Skip terminal states (acked / ack_failed). Anything
                # else — ready, unack, inited — counts as in flight.
                if status in (
                    int(persistqueue.AckStatus.acked),
                    int(persistqueue.AckStatus.ack_failed),
                ):
                    continue
                data = result.get("data") or {}
                try:
                    count = int(data.get("request_count", 1))
                except (TypeError, ValueError):
                    count = 1
                if count < 1:
                    count = 1
                total += count
            return total

    async def clear_queue(self):
        async with self.lock:
            while not self.queue.empty():
                try:
                    raw_item = self.queue.get(block=False, raw=True)
                    request_id = raw_item["pqid"]
                    self.queue.ack(id=request_id)
                except persistqueue.Empty:
                    break

        await self.clear_acked_data()

    def __len__(self):
        return len(self.queue)

inference_work_queue = None
lock = asyncio.Lock()

async def get_inference_work_queue():
    global inference_work_queue
    global lock

    async with lock:
        if inference_work_queue is None:
            inference_work_queue = get_file_backed_inference_work_queue(auto_resume=True)

    return inference_work_queue


def get_file_backed_inference_work_queue(auto_resume=False):
    config = get_config()
    path = config["inference_work_queue_path"]

    return InferenceWorkQueue(path=path, auto_resume=auto_resume)

