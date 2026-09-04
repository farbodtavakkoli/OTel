
import time
from collections import deque
from typing import Dict

generate_metrics = None

# Rolling-window length for instantaneous rate metrics. token/s,
# request/s, and flop/s are reported as "completions in the last
# RATE_WINDOW_SECONDS / RATE_WINDOW_SECONDS" rather than a cumulative
# average over the lifetime of the process. The latter barely changed
# between polls and rendered a flat sparkline.
RATE_WINDOW_SECONDS = 60.0

# Bounded sample window for chat batch-size and request-duration
# histograms. Big enough for stable p50/p99 under realistic QPS, small
# enough that an O(n log n) sort at read time is trivial.
CHAT_HISTOGRAM_SAMPLE_SIZE = 1024


class Metrics:
    def __init__(
        self,
        *,
        buffering_check_proxy_timeout_seconds: float = 60.0,
        buffering_match_threshold_seconds: float = 0.5,
    ):
        self.queue_depth = 0
        # Concurrent OpenAI-streaming requests in flight. Driven by
        # _wrap_with_metrics in openai_v1_router.py; that path doesn't
        # touch the SQLiteAckQueue, so it has no other ground truth.
        # try/finally ensures every increment matches a decrement, so
        # this counter doesn't drift the way Metrics.queue_depth can.
        self.streaming_inflight = 0
        self.epoch_time = None

        self.total_completed_requests = 0
        self.total_completed_tokens = 0
        self.total_completed_flops = 0

        self.total_completed_response_time = 0.0

        # Per-completion window: (timestamp, tokens, flops). Pruned
        # to the last RATE_WINDOW_SECONDS in get_all_metrics.
        self._rate_window: deque = deque()

        # ----- chat-completions metrics (docs §13) -----
        # Six metrics surfaced through get_all_metrics. The chat path
        # is independent of the SDK path's queue_depth bookkeeping;
        # nothing here mutates the existing /v1/generate counters.
        self.chat_in_flight: int = 0
        self.chat_admitted_429_count: int = 0
        self.chat_total_count: int = 0
        self.chat_apparent_buffering_count: int = 0
        self._chat_batch_sizes: deque = deque(maxlen=CHAT_HISTOGRAM_SAMPLE_SIZE)
        self._chat_request_durations: deque = deque(maxlen=CHAT_HISTOGRAM_SAMPLE_SIZE)
        self._chat_start_times: Dict[str, float] = {}

        self._buffering_proxy_timeout = buffering_check_proxy_timeout_seconds
        self._buffering_match_threshold = buffering_match_threshold_seconds

    def record_completed_request(self, token_count: int, flop_count: int):
        """
        Record a completed request.
        """
        # Always feed the rate window — it doesn't depend on the
        # legacy queue_depth bookkeeping.
        now = time.time()
        self._rate_window.append(
            (now, int(token_count or 0), int(flop_count or 0))
        )

        if self.queue_depth <= 0:
            self.queue_depth = 0
            return

        self.total_completed_requests += 1

        if flop_count is not None:
            self.total_completed_flops += flop_count

        if token_count is not None:
            self.total_completed_tokens += token_count

        self.queue_depth -= 1

        time_since_epoch = now - self.epoch_time

        self.total_completed_response_time += time_since_epoch

        self.epoch_time = now

    def record_new_request(self):
        """
        Record a new request.
        """
        if self.queue_depth == 0:
            self.epoch_time = time.time()

        self.queue_depth += 1

    def record_streaming_start(self):
        """OpenAI-streaming path admission; increments streaming_inflight."""
        self.streaming_inflight += 1

    def record_streaming_end(self):
        """OpenAI-streaming path completion; decrements streaming_inflight."""
        if self.streaming_inflight > 0:
            self.streaming_inflight -= 1

    # ------------------------------------------------------------------
    # Chat-completions metrics (docs §13)
    # ------------------------------------------------------------------

    def record_chat_admitted(self, correlation_id: str) -> None:
        """Handler admitted a request: in_flight++, total++, log start time."""
        self.record_chat_admitted_with_clock(correlation_id, start_time=time.time())

    def record_chat_admitted_with_clock(
        self, correlation_id: str, *, start_time: float
    ) -> None:
        """Test seam — same as record_chat_admitted but takes the clock value."""
        self.chat_in_flight += 1
        self.chat_total_count += 1
        self._chat_start_times[correlation_id] = start_time

    def record_chat_rejected_429(self) -> None:
        """Admission denied a request with 429."""
        self.chat_admitted_429_count += 1
        self.chat_total_count += 1

    def record_chat_resolved(self, correlation_id: str) -> None:
        """Worker delivered a result for an admitted request."""
        self.record_chat_resolved_with_clock(correlation_id, end_time=time.time())

    def record_chat_resolved_with_clock(
        self, correlation_id: str, *, end_time: float
    ) -> None:
        start = self._chat_start_times.pop(correlation_id, None)
        if start is None:
            # Unknown cid — admitted before this Metrics instance
            # existed, or already cleaned up. Don't underflow in_flight.
            return

        if self.chat_in_flight > 0:
            self.chat_in_flight -= 1

        duration = max(0.0, end_time - start)
        self._chat_request_durations.append(duration)

        # Apparent-buffering heuristic: if the request landed within
        # `match_threshold` of a known proxy idle timeout, flag it.
        # See docs §13.2 — this is a signal, not a strict measurement.
        if abs(duration - self._buffering_proxy_timeout) <= self._buffering_match_threshold:
            self.chat_apparent_buffering_count += 1

    def record_chat_unregistered(self, correlation_id: str) -> None:
        """Client disconnected before resolution — drop the in_flight slot."""
        if self._chat_start_times.pop(correlation_id, None) is None:
            return
        if self.chat_in_flight > 0:
            self.chat_in_flight -= 1

    def record_chat_batch_size(self, size: int) -> None:
        """Coalescer flushed a batch of `size` requests as one queue row."""
        if size > 0:
            self._chat_batch_sizes.append(size)

    def get_all_metrics(self, sdk_queue_depth=None):
        """
        Get the current metrics.

        `sdk_queue_depth` is the live SQLiteAckQueue length passed in by
        the route handler — the ground truth for the SDK path. We add
        `streaming_inflight` to it so the reported queue_depth reflects
        BOTH submission paths (SDK via SQLite, OpenAI-streaming via the
        in-memory counter). When `sdk_queue_depth` is None we fall back
        to the legacy `queue_depth` counter, which is fine for callers
        that don't have access to the queue.
        """
        if sdk_queue_depth is not None:
            queue_depth = sdk_queue_depth + self.streaming_inflight
        else:
            queue_depth = self.queue_depth + self.streaming_inflight

        token_rate, request_rate, flop_rate = self._windowed_rates()

        return {
            "queue_depth": queue_depth,
            "requests": self.total_completed_requests,
            "tokens": self.total_completed_tokens,
            "total_time": self.total_completed_response_time,
            "token/s": token_rate,
            "request/s": request_rate,
            "flop/s": flop_rate,
            # Chat-completions metrics (docs §13).
            "chat_in_flight": self.chat_in_flight,
            "chat_admitted_429_count": self.chat_admitted_429_count,
            "chat_total_count": self.chat_total_count,
            "chat_admitted_429_rate": (
                self.chat_admitted_429_count / self.chat_total_count
                if self.chat_total_count
                else 0.0
            ),
            "chat_batch_size_p50": _percentile(self._chat_batch_sizes, 50),
            "chat_batch_size_p99": _percentile(self._chat_batch_sizes, 99),
            "chat_request_duration_p50": _percentile(self._chat_request_durations, 50),
            "chat_request_duration_p99": _percentile(self._chat_request_durations, 99),
            "chat_apparent_buffering_count": self.chat_apparent_buffering_count,
        }

    def _windowed_rates(self):
        """
        Compute (tokens, requests, flops) per second over the last
        RATE_WINDOW_SECONDS. Prunes the window in place. Returns zeros
        when the window is empty (no completions yet).
        """
        now = time.time()
        cutoff = now - RATE_WINDOW_SECONDS
        while self._rate_window and self._rate_window[0][0] < cutoff:
            self._rate_window.popleft()

        if not self._rate_window:
            return 0.0, 0.0, 0.0

        # Span is the elapsed seconds across the window. Use the full
        # configured window length once we have at least one full
        # window's worth of samples; otherwise use elapsed-since-first
        # so a fresh process doesn't divide by ~0.
        oldest_ts = self._rate_window[0][0]
        span = max(min(RATE_WINDOW_SECONDS, now - oldest_ts), 0.001)

        total_tokens = 0
        total_flops = 0
        for _, tokens, flops in self._rate_window:
            total_tokens += tokens
            total_flops += flops

        return (
            total_tokens / span,
            len(self._rate_window) / span,
            total_flops / span,
        )


def get_metrics() -> Metrics:
    """
    Get the metrics object.
    """
    global generate_metrics
    if generate_metrics is None:
        generate_metrics = Metrics()
    return generate_metrics


def _percentile(samples, percentile: int):
    """Simple percentile over a small bounded window. O(n log n) per
    read, called only from get_all_metrics, n <= CHAT_HISTOGRAM_SAMPLE_SIZE."""
    if not samples:
        return 0
    ordered = sorted(samples)
    k = min(int(len(ordered) * percentile / 100), len(ordered) - 1)
    return ordered[k]
