from benchmark.pytorch.memcpy import benchmark_memcpy
from benchmark.pytorch.memcpy_peer import benchmark_memcpy_peer
from benchmark.pytorch.gemm import benchmark_gemm
from benchmark.pytorch.forward import benchmark_forward
from benchmark.pytorch.backward import benchmark_backward

from benchmark.roofline.plot_roofline import plot_roofline
from benchmark.roofline.plot_bandwidth_sweep import plot_bandwidth_sweep

import os

import logging

def main():
    setup_logging()

    if "HUGGING_FACE_HUB_TOKEN" not in os.environ and os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    benchmark_gemm()

    plot_roofline()
    plot_bandwidth_sweep()


def setup_logging():
    logging.basicConfig(level=logging.INFO)

main()
