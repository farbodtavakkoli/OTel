import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from textwrap import wrap

import os
import platform
import subprocess
import re
import json
import torch
import math


def plot_roofline():
    machine_roofline = get_machine_roofline()

    plot_gemm_kernels_on_roofline(machine_roofline)
    plot_memcpy_on_roofline(machine_roofline)
    plot_forward_on_roofline(machine_roofline)
    plot_backward_on_roofline(machine_roofline)


def plot_gemm_kernels_on_roofline(machine_roofline):
    gemm_kernel_metrics = load_gemm_kernel_metrics()

    plot_kernels_on_roofline("GEMM", machine_roofline, gemm_kernel_metrics)


def load_gemm_kernel_metrics():
    metrics_path = "/app/cray/data/benchmark_gemm.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_memcpy_on_roofline(machine_roofline):
    memcpy_metrics = load_memcpy_metrics()

    plot_kernels_on_roofline("Memcpy", machine_roofline, memcpy_metrics)


def load_memcpy_metrics():
    metrics_path = "/app/cray/data/benchmark_memcpy.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    return benchmark_metrics


def plot_forward_on_roofline(machine_roofline):
    forward_metrics = load_forward_metrics()

    plot_kernels_on_roofline("Inference", machine_roofline, forward_metrics)


def load_forward_metrics():
    metrics_path = "/app/cray/data/benchmark_forward.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    benchmark_metrics = flatten_metrics(prefix="", metrics=benchmark_metrics)

    return benchmark_metrics


def plot_backward_on_roofline(machine_roofline):
    backward_metrics = load_backward_metrics()

    plot_kernels_on_roofline("Training", machine_roofline, backward_metrics)


def load_backward_metrics():
    metrics_path = "/app/cray/data/benchmark_backward.json"

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as metrics_file:
        benchmark_metrics = json.load(metrics_file)

    benchmark_metrics = flatten_metrics(prefix="", metrics=benchmark_metrics)

    return benchmark_metrics


def flatten_metrics(prefix, metrics):
    prefix = shorten_prefix(prefix)

    flat_metrics = {}
    is_leaf = False
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat_metrics.update(flatten_metrics(f"{prefix}{key}_", value))
        else:
            flat_metrics[f"{key}"] = value
            is_leaf = True

    if is_leaf:
        flat_metrics = {prefix: flat_metrics}

    return flat_metrics

def shorten_prefix(prefix):
    prefix = re.sub(r".*/", "", prefix)

    return prefix


def plot_kernels_on_roofline(kernel_type, machine_roofline, gemm_kernel_metrics):
    if gemm_kernel_metrics is None:
        return

    units = {
        "flop/s": "GFLOP/s",
        "memory_bandwidth": "GB/s",
        "factor": 1e9,
    }

    if machine_roofline["flops"] > 1e12:
        units["flop/s"] = "TFLOP/s"
        units["memory_bandwidth"] = "TB/s"
        units["factor"] = 1e12

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax.set_ylabel(f"Performance ({units['flop/s']})")
    ax.set_title(f"{kernel_type} Roofline Model for {get_processor_name()}")

    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]

    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "h"]

    max_intensity = (
        10 * machine_roofline["flops"] / machine_roofline["memory_bandwidth"]
    )
    min_flops = machine_roofline["memory_bandwidth"] / units["factor"]

    for i, (kernel_name, kernel_metrics) in enumerate(gemm_kernel_metrics.items()):
        intensity = kernel_metrics["operational_intensity"]
        flops = kernel_metrics["flop/s"] / units["factor"]

        color_idx = i % len(colors)
        marker_idx = i % len(markers)

        ax.plot(
            intensity,
            flops,
            marker=markers[marker_idx],
            markersize=8,
            markerfacecolor=colors[color_idx],
            markeredgecolor="black",
            markeredgewidth=1,
            linestyle="none",
            label="\n".join(wrap(kernel_name, 10)),
        )

        max_intensity = max(max_intensity, intensity)
        min_flops = min(min_flops, flops)

    memory_bandwidth_gflops = machine_roofline["memory_bandwidth"] / units["factor"]
    peak_flops_gflops = machine_roofline["flops"] / units["factor"]
    ridge_point_x = peak_flops_gflops / memory_bandwidth_gflops

    ax.plot(
        [0.01, ridge_point_x, max_intensity * 1.1],
        [
            0.01 * memory_bandwidth_gflops,
            peak_flops_gflops,
            peak_flops_gflops,
        ],
        linestyle="-",
        linewidth=2.5,
        color="#000000",
        label="Roofline",
    )

    ax.text(
        ridge_point_x * 0.01,
        peak_flops_gflops * .1,
        f"Memory Bound\n{memory_bandwidth_gflops:.1f} {units['memory_bandwidth']}",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        ha="center",
    )

    ax.text(
        ridge_point_x * 3,
        peak_flops_gflops * 1.5,
        f"Compute Bound\n{peak_flops_gflops:.1f} {units['flop/s']}",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        ha="center",
    )

    ax.plot(
        [ridge_point_x],
        [peak_flops_gflops],
        "kD",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.set_xlim(0.1, math.exp(math.log(max_intensity) * 1.1))
    ax.set_ylim(
        math.exp(math.log(min_flops) * 0.9),
        peak_flops_gflops * 3.0
    )

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
    )

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    save_to_file(fig, f"roofline_{kernel_type}.pdf")


def save_to_file(fig, filename):
    directory = "/app/cray/data"

    full_path = os.path.join(directory, filename)

    fig.savefig(full_path)


def get_machine_roofline():
    processor_name = get_processor_name()

    assert (
        processor_name in processor_table
    ), f"Processor {processor_name} not found in processor table"

    return processor_table[processor_name]


def get_processor_name():
    return "AMD Instinct MI300X"

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    assert platform.system() == "Linux"
    command = "lscpu"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    for line in all_info.split("\n"):
        if "Apple" in line:
            return "Apple M2"
    return "UNKNOWN"


processor_table = {
    "Apple M2": {
        "memory_bandwidth": 102.4e9,
        "peer_bandwidth": 102.4e9,
        "flops": 223.36e9,
    },
    "AMD Instinct MI300X": {
        "memory_bandwidth": 5.3e12,
        "peer_bandwidth": 50e9,
        "flops": 1.3e15,
    },
}
