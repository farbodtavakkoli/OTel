"""Pytest coverage for cray_infra.training.distributed collectives.

Runs under torchrun automatically via the ``@pytest.mark.torchrun`` marker, or
directly, for example::

    torchrun --nnodes=1 --nproc-per-node=2 \\
        -m pytest test/infra/distribution_strategy/test_distributed_collectives.py -xvs
"""

import pytest
import torch

from cray_infra.training.distributed import get_rank

from distributed_benchmarks import run_collectives_benchmark


@pytest.mark.torchrun(nproc=2, arch="cpu")
def test_collectives_cpu(distributed_arch):
    results = run_collectives_benchmark(distributed_arch)
    if get_rank() == 0:
        assert results, "rank 0 should collect benchmark results"
        for name, bandwidth in results.items():
            assert bandwidth > 0, f"{name} bandwidth should be positive"


@pytest.mark.torchrun(nproc=4, arch="cuda")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_collectives_cuda(distributed_arch):
    results = run_collectives_benchmark(distributed_arch)
    if get_rank() == 0:
        assert results
        for bandwidth in results.values():
            assert bandwidth > 0


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_collectives_rocm(distributed_arch):
    results = run_collectives_benchmark(distributed_arch)
    if get_rank() == 0:
        assert results
        for bandwidth in results.values():
            assert bandwidth > 0
