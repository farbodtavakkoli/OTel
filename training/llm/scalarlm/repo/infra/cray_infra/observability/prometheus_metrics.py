"""Prometheus metrics for ScalarLM inference and training."""

import logging

logger = logging.getLogger(__name__)

# Try to import prometheus_client, but gracefully handle if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    PROMETHEUS_AVAILABLE = True

    # ==================== Inference Metrics ====================

    inference_requests_total = Counter(
        "scalarlm_inference_requests_total",
        "Total inference requests",
        ["model", "status"]
    )

    inference_duration_seconds = Histogram(
        "scalarlm_inference_duration_seconds",
        "Inference request duration in seconds",
        ["model"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120]
    )

    queue_depth = Gauge(
        "scalarlm_queue_depth",
        "Work queue depth by model",
        ["model"]
    )

    queue_wait_time_seconds = Histogram(
        "scalarlm_queue_wait_time_seconds",
        "Time request waits in queue before processing",
        ["model"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30]
    )

    adapter_load_time_seconds = Histogram(
        "scalarlm_adapter_load_time_seconds",
        "Time to load LoRA adapter",
        ["model"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
    )

    # ==================== Training Metrics ====================

    training_loss = Gauge(
        "scalarlm_training_loss",
        "Training loss per step",
        ["job_hash", "epoch"]
    )

    training_grad_norm = Gauge(
        "scalarlm_training_grad_norm",
        "Gradient norm per step",
        ["job_hash"]
    )

    training_mfu = Gauge(
        "scalarlm_training_mfu",
        "Model FLOP Utilization (actual FLOPS / theoretical peak)",
        ["job_hash"]
    )

    training_step_time_seconds = Histogram(
        "scalarlm_training_step_time_seconds",
        "Training step time in seconds",
        ["job_hash"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60]
    )

    def metrics_endpoint():
        """Return Prometheus metrics in text format."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

except ImportError as e:
    logger.warning(f"prometheus_client not available: {e}. Metrics will not be exported.")
    PROMETHEUS_AVAILABLE = False

    # Create no-op placeholders
    class NoOpMetric:
        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

    inference_requests_total = NoOpMetric()
    inference_duration_seconds = NoOpMetric()
    queue_depth = NoOpMetric()
    queue_wait_time_seconds = NoOpMetric()
    adapter_load_time_seconds = NoOpMetric()
    training_loss = NoOpMetric()
    training_grad_norm = NoOpMetric()
    training_mfu = NoOpMetric()
    training_step_time_seconds = NoOpMetric()

    def metrics_endpoint():
        """Return empty response when Prometheus not available."""
        from fastapi import Response
        return Response("Prometheus metrics not available", media_type="text/plain")
