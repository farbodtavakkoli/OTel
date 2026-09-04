"""Summarize an OSFT run — peak memory, current step, duration, loss plot. Usage: python check_memory.py <ckpt_output_dir>."""
import os
import sys
import json
from datetime import datetime
from training_hub import plot_loss


def get_peak_mem(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = [json.loads(line) for line in f]

    peak_mems = [m.get("peak_memory_usage_GB", 0) for m in metrics]
    start_time = datetime.fromisoformat(metrics[0]['timestamp'])
    end_time = datetime.fromisoformat(metrics[-1]['timestamp'])
    return max(peak_mems), metrics[-1]['step'], end_time - start_time

if __name__ == "__main__":
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    plot_loss(ckpt_dir)
    peak_mem, current_step, training_duration = get_peak_mem(
        os.path.join(ckpt_dir, "training_metrics_0.jsonl")
    )
    print(f"Peak Memory Usage: {peak_mem} GB")
    print(f"Current step: {current_step}")
    print(f"Training duration: {training_duration}")