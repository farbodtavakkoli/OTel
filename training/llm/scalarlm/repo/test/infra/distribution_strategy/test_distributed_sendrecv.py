"""Pytest coverage for cray_infra.training.distributed point-to-point ops.

Runs under torchrun automatically via the ``@pytest.mark.torchrun`` marker, or
directly, for example::

    torchrun --nnodes=1 --nproc-per-node=2 \\
        -m pytest test/infra/distribution_strategy/test_distributed_sendrecv.py -xvs
"""

import pytest
import torch

from cray_infra.training.distributed import get_rank

from distributed_benchmarks import run_sendrecv_benchmark


@pytest.mark.torchrun(nproc=2, arch="cpu")
def test_sendrecv_cpu(distributed_arch):
    bandwidth = run_sendrecv_benchmark(distributed_arch)
    if get_rank() == 0:
        assert bandwidth > 0


@pytest.mark.torchrun(nproc=2, arch="cuda")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sendrecv_cuda(distributed_arch):
    bandwidth = run_sendrecv_benchmark(distributed_arch)
    if get_rank() == 0:
        assert bandwidth > 0


@pytest.mark.torchrun(nproc=2, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_sendrecv_rocm(distributed_arch):
    bandwidth = run_sendrecv_benchmark(distributed_arch)
    if get_rank() == 0:
        assert bandwidth > 0
