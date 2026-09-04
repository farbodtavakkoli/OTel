"""Live speed/ETA monitor for OSFT runs — polls training_metrics_0.jsonl; usage in readme_redhat.md."""

import json
import math
import os
import threading
from contextlib import contextmanager
from datetime import datetime

METRICS_FILENAME = "training_metrics_0.jsonl"


def count_data_samples(data_path: str) -> int:
    """Count the examples (lines) in a JSONL file by chunked newline counting."""
    count = 0
    last_byte = b"\n"
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte not in (b"\n", b""):
        count += 1
    return count


def compute_steps_per_epoch(data_path: str, effective_batch_size: int) -> int:
    """Return ceil(num_samples / effective_batch_size), at least 1."""
    num_samples = count_data_samples(data_path)
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be a positive integer")
    return max(1, math.ceil(num_samples / effective_batch_size))


def read_metrics(metrics_path: str) -> list[dict]:
    """Read all valid JSON lines from a live metrics file, skipping malformed ones."""
    metrics: list[dict] = []
    try:
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    # Partially-written trailing line; ignore.
                    continue
    except FileNotFoundError:
        return []
    return metrics


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def compute_speed_report(metrics: list[dict], steps_per_epoch: int, num_epochs: int) -> dict | None:
    """Build an ETA/throughput report from collected metrics, or None if no usable step yet."""
    if not metrics:
        return None

    last = metrics[-1]
    completed_steps = last.get("step")
    if completed_steps is None:
        return None

    first = metrics[0]
    total_steps = steps_per_epoch * num_epochs

    # Steady-state per-step time from timestamps between first and last logged step.
    elapsed = 0.0
    try:
        start_t = datetime.fromisoformat(first["timestamp"])
        end_t = datetime.fromisoformat(last["timestamp"])
        elapsed = (end_t - start_t).total_seconds()
    except (KeyError, ValueError, TypeError):
        elapsed = 0.0

    step_span = completed_steps - first.get("step", completed_steps)
    if step_span > 0 and elapsed > 0:
        per_step = elapsed / step_span
    else:
        # Only one step logged so far: fall back to its reported batch time.
        per_step = last.get("time_per_batch") or 0.0

    remaining_steps = max(total_steps - completed_steps, 0)
    total_remaining = remaining_steps * per_step

    # Steps completed / remaining within the current epoch (self-consistent with the
    # computed steps_per_epoch).
    steps_into_epoch = completed_steps % steps_per_epoch if steps_per_epoch else 0
    if steps_into_epoch == 0 and completed_steps > 0:
        steps_left_in_epoch = 0 # exactly on an epoch boundary
    else:
        steps_left_in_epoch = steps_per_epoch - steps_into_epoch
    epoch_remaining = steps_left_in_epoch * per_step

    peak_mem = max(
        (m["peak_memory_usage_GB"] for m in metrics
        if m.get("peak_memory_usage_GB") is not None),
        default=None,
    )
    peak_tps = max(
        (m["tokens_per_second"] for m in metrics
        if m.get("tokens_per_second") is not None),
        default=None,
    )
    val_losses = [m["val_loss"] for m in metrics if m.get("val_loss") is not None]
    last_val_loss = val_losses[-1] if val_losses else None

    return {
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "current_epoch": last.get("epoch"),
        "num_epochs": num_epochs,
        "elapsed_seconds": elapsed,
        "per_step_seconds": per_step,
        "total_remaining_seconds": total_remaining,
        "epoch_remaining_seconds": epoch_remaining,
        "peak_memory_usage_GB": peak_mem,
        "peak_tokens_per_second": peak_tps,
        "last_val_loss": last_val_loss,
    }


def format_report(report: dict) -> str:
    """Render a speed report dict (from compute_speed_report) as a printable block."""
    mem = report["peak_memory_usage_GB"]
    tps = report["peak_tokens_per_second"]
    vloss = report["last_val_loss"]
    mem_s = f"{mem:.2f} GB" if mem is not None else "N/A"
    tps_s = f"{tps:,.1f}" if tps is not None else "N/A"
    vloss_s = f"{vloss:.4f}" if vloss is not None else "N/A"
    return "\n".join([
    "",
    "⏱️ Speed / ETA report",
    "-" * 50,
    f" Progress: step {report['completed_steps']}/{report['total_steps']} "
    f"(epoch {report['current_epoch']}/{report['num_epochs']}, "
    f"{report['steps_per_epoch']} steps/epoch)",
    f" Elapsed / rate: {format_duration(report['elapsed_seconds'])} "
    f"(~{report['per_step_seconds']:.1f}s/step)",
    f" ETA current epoch: {format_duration(report['epoch_remaining_seconds'])}",
    f" ETA total remaining: {format_duration(report['total_remaining_seconds'])}",
    f" Max peak memory: {mem_s}",
    f" Max peak tokens/sec: {tps_s}",
    f" Last val loss: {vloss_s}",
    "-" * 50,
    ])


class SpeedMonitor(threading.Thread):
    """Background thread that polls the metrics file and prints ETA reports."""

    def __init__(self, ckpt_output_dir: str, steps_per_epoch: int, num_epochs: int,
    report_every_steps: int, poll_interval: float = 5.0):
        super().__init__(daemon=True)
        self.metrics_path = os.path.join(ckpt_output_dir, METRICS_FILENAME)
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.report_every_steps = max(1, report_every_steps)
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._last_reported_step = 0

    def stop(self):
        """Signal the monitor loop to exit."""
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self._maybe_report()
            self._stop_event.wait(self.poll_interval)
        # One last report so the final numbers are always shown.
        self._maybe_report(force=True)

    def _maybe_report(self, force: bool = False):
        report = compute_speed_report(
            read_metrics(self.metrics_path), self.steps_per_epoch, self.num_epochs
        )
        if report is None:
            return
        step = report["completed_steps"]
        if force or step - self._last_reported_step >= self.report_every_steps:
            print(format_report(report), flush=True)
            self._last_reported_step = step


@contextmanager
def speed_monitor(ckpt_output_dir: str, data_path: str, effective_batch_size: int,
num_epochs: int, report_every_steps: int,
poll_interval: float = 5.0):
    """Context manager running a SpeedMonitor for the with-block; always stopped/joined on exit."""
    steps_per_epoch = compute_steps_per_epoch(data_path, effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    print(
        f"📈 Speed monitor enabled: {steps_per_epoch:,} steps/epoch "
        f"({total_steps:,} total steps over {num_epochs} epochs); "
        f"reporting every {report_every_steps} step(s).",
        flush=True,
    )
    monitor = SpeedMonitor(
        ckpt_output_dir, steps_per_epoch, num_epochs, report_every_steps, poll_interval
    )
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()
        monitor.join(timeout=poll_interval + 1)