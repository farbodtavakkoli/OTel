from collections import deque
from typing import Deque


def is_over_high_water(
    *,
    queue_depth: int,
    in_flight_count: int,
    max_num_seqs: int,
    admit_factor: int,
) -> bool:
    return (queue_depth + in_flight_count) > admit_factor * max_num_seqs


class WaitEstimator:
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
