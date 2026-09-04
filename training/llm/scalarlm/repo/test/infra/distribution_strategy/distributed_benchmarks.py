"""Shared distributed collective benchmarks using cray_infra.training.distributed."""

import time

import torch

from cray_infra.training.distributed import (
    allgather,
    allreduce,
    barrier,
    finalize,
    get_rank,
    get_size,
    init,
    recv,
    reduce_scatter,
    send,
)

DATA_SIZE_COLLECTIVES = 4_194_304
DATA_SIZE_SENDRECV = 262_144 * 64
COLLECTIVE_WARMUP = 10
COLLECTIVE_ITERS = 100
SENDRECV_WARMUP = 10
SENDRECV_ITERS = 100


def create_buffer(arch: str, size: int, rank: int) -> torch.Tensor:
    if arch == "cuda":
        return torch.ones(size, dtype=torch.float32, device="cuda")
    if arch == "rocm":
        return torch.ones(size, dtype=torch.float32, device=f"cuda:{rank}")
    return torch.ones(size, dtype=torch.float32, device="cpu")


def benchmark_collective(
    collective_fn,
    arch: str,
    send_size: int,
    recv_size: int,
    expected_value: float,
    all_reduce_flag: bool = False,
    num_iters: int = COLLECTIVE_ITERS,
    warmup: int = COLLECTIVE_WARMUP,
) -> float:
    size = get_size()
    rank = get_rank()
    sendbuf = create_buffer(arch, send_size, rank).contiguous()
    recvbuf = torch.empty(recv_size, dtype=torch.float32, device=sendbuf.device).contiguous()

    for _ in range(warmup):
        collective_fn(sendbuf, recvbuf)

    barrier()
    t0 = time.time()
    for _ in range(num_iters):
        if all_reduce_flag:
            sendbuf = create_buffer(arch, send_size, rank).contiguous()
        collective_fn(sendbuf, recvbuf)
    barrier()
    dt = time.time() - t0

    if all_reduce_flag:
        assert torch.allclose(
            sendbuf, torch.full_like(sendbuf, expected_value), atol=1e-6
        ), "Verification failed"
    else:
        assert torch.allclose(
            recvbuf, torch.full_like(recvbuf, expected_value), atol=1e-6
        ), "Verification failed"

    datatype_bytes = 4
    total_data = send_size * datatype_bytes * 2 * size
    return (total_data / dt) / 1e9


def run_collectives_benchmark(arch: str) -> dict[str, float]:
    data_size = DATA_SIZE_COLLECTIVES
    size = get_size()

    collectives = {
        "AllGather": (
            lambda sbuf, rbuf: allgather(sbuf, rbuf),
            data_size,
            data_size * size,
            1.0,
            False,
        ),
        "ReduceScatter": (
            lambda sbuf, rbuf: reduce_scatter(sbuf, rbuf),
            data_size,
            data_size // size,
            size * 1.0,
            False,
        ),
        "AllReduce": (
            lambda sbuf, rbuf: allreduce(sbuf),
            data_size,
            1,
            size * 1.0,
            True,
        ),
    }

    results = {}
    for name, info in collectives.items():
        bw = benchmark_collective(info[0], arch, info[1], info[2], info[3], info[4])
        if get_rank() == 0:
            results[name] = bw
    return results


def _send_recv(sendbuf: torch.Tensor, iteration: int) -> None:
    sender = iteration % 2
    rank = get_rank()
    barrier()
    if rank == sender:
        send(sendbuf, (rank + 1) % 2)
    else:
        recv(sendbuf, (rank + 1) % 2)
    barrier()


def benchmark_send_recv(
    arch: str,
    data_size: int = DATA_SIZE_SENDRECV,
    num_iters: int = SENDRECV_ITERS,
    warmup: int = SENDRECV_WARMUP,
) -> float:
    rank = get_rank()
    sendbuf = create_buffer(arch, data_size, rank).contiguous()

    if rank == 1:
        torch.zero_(sendbuf)

    for i in range(0, warmup * 2, 2):
        _send_recv(sendbuf, i)

    barrier()
    t0 = time.time()
    for i in range(0, num_iters * 2, 2):
        _send_recv(sendbuf, i)
    barrier()
    dt = time.time() - t0

    assert torch.allclose(sendbuf, torch.full_like(sendbuf, 1), atol=1e-6), (
        "Verification failed"
    )

    total_data = data_size * 4 * num_iters
    return (total_data / dt) / 1e9


def run_sendrecv_benchmark(arch: str) -> float:
    assert get_size() == 2, "Send/recv benchmark requires exactly two ranks"
    return benchmark_send_recv(arch)


def setup_distributed() -> None:
    init()


def teardown_distributed() -> None:
    finalize()
