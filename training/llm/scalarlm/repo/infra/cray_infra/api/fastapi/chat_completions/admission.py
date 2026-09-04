"""
Admission control for /v1/chat/completions.

Two responsibilities:

  - is_over_high_water(...) — pure threshold check used by the
    handler to decide whether to admit or reject with 429.
  - WaitEstimator             — moving average of recent batch
    completion times, exposed as a `Retry-After` hint when rejecting.

Both pieces are intentionally tiny and stateless-or-bounded-state so
the handler can call them on the hot path without contention.

See docs/openai-chat-completions-queue.md §5.
"""

from collections import deque
from typing import Deque


def is_over_high_water(
    *,
    queue_depth: int,
    in_flight_count: int,
    max_num_seqs: int,
    admit_factor: int,
) -> bool:
    """
    True if `queue_depth + in_flight_count` exceeds `admit_factor *
    max_num_seqs`. Equality is *not* over — the threshold is the last
    value we'll admit.
    """
    return (queue_depth + in_flight_count) > admit_factor * max_num_seqs


class WaitEstimator:
    """
    Bounded ring buffer of recent batch completion times. The
    `estimate_wait_seconds` method returns the `Retry-After` hint to
    send with a 429: the moving-average batch latency, scaled by the
    current overload ratio, padded by a constant factor to prevent
    retry storms when many clients receive identical hints and
    synchronize their next attempts.

    The estimate is non-negative and zero whenever queue depth is at
    or below `max_num_seqs` — callers shouldn't get a Retry-After
    when they're about to be admitted.
    """

    def __init__(
        self,
        *,
        default_batch_latency_seconds: float = 5.0,
        padding: float = 1.5,
        sample_size: int = 32,
    ) -> None:
        self._default = default_batch_latency_seconds
        self._padding = padding
        self._samples: Deque[float] = deque(maxlen=sample_size)

    def record_batch_latency_seconds(self, seconds: float) -> None:
        self._samples.append(seconds)

    def estimate_wait_seconds(
        self, *, queue_depth: int, max_num_seqs: int
    ) -> float:
        if max_num_seqs <= 0:
            return 0.0
        overload_ratio = (queue_depth - max_num_seqs) / max_num_seqs
        if overload_ratio <= 0:
            return 0.0

        avg_latency = (
            sum(self._samples) / len(self._samples)
            if self._samples
            else self._default
        )

        return max(0.0, avg_latency * overload_ratio * self._padding)
