import pytest
import torch

from cray_infra.training.distributed import (
    allreduce,
    alltoall,
    get_rank,
    get_size,
)


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_alltoall_writes_back_to_cpu_buffer(distributed_arch):
    rank, world = get_rank(), get_size()
    chunk = 4

    send = torch.cat(
        [torch.full((chunk,), float(rank * world + j), dtype=torch.float32)
         for j in range(world)]
    )
    recv = torch.zeros(world * chunk, dtype=torch.float32)

    alltoall(send, recv)

    expected = torch.cat(
        [torch.full((chunk,), float(j * world + rank), dtype=torch.float32)
         for j in range(world)]
    )
    assert torch.equal(recv, expected), (
        f"rank {rank}: alltoall did not write back to the CPU buffer. "
        f"expected {expected.tolist()}, got {recv.tolist()}"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_allreduce_writes_back_to_cpu_buffer(distributed_arch):
    rank, world = get_rank(), get_size()

    tensor = torch.full((16,), float(rank + 1), dtype=torch.float32)
    allreduce(tensor)

    expected = float(world * (world + 1) // 2)
    assert torch.equal(tensor, torch.full((16,), expected, dtype=torch.float32)), (
        f"rank {rank}: expected all {expected}, got {tensor.tolist()[:8]}"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_fsdp_reduce_scatter_returns_mean_gradient(distributed_arch):
    from cray_megatron.megatron.distribution.fsdp import (
        collectives_reduce_scatter,
        cuda_device,
    )

    rank, world = get_rank(), get_size()

    numel = world * 8
    shard_size = numel // world
    metadata = {r: (numel, (numel,), shard_size, 0) for r in range(world)}

    grad = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=cuda_device())
    out = collectives_reduce_scatter(grad, metadata)

    mean_expected = (world + 1) / 2.0
    sum_expected = world * (world + 1) / 2.0
    got = out.float().mean().item()

    assert abs(got - mean_expected) < 1e-4, (
        f"rank {rank}: FSDP reduce_scatter returned {got}; expected the mean "
        f"{mean_expected}. ({sum_expected} would mean it is still summing, which "
        f"scales the effective learning rate by world_size.)"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_ddp_and_fsdp_agree_on_gradient_scaling(distributed_arch):
    from cray_megatron.megatron.distribution.fsdp import (
        collectives_reduce_scatter,
        cuda_device,
    )

    rank, world = get_rank(), get_size()

    ddp_grad = torch.full((16,), float(rank + 1), dtype=torch.float32)
    allreduce(ddp_grad)
    ddp_grad /= world

    numel = world * 8
    metadata = {r: (numel, (numel,), numel // world, 0) for r in range(world)}
    fsdp_grad = torch.full(
        (numel,), float(rank + 1), dtype=torch.float32, device=cuda_device()
    )
    fsdp_out = collectives_reduce_scatter(fsdp_grad, metadata)

    assert abs(ddp_grad[0].item() - fsdp_out.float().mean().item()) < 1e-4, (
        f"rank {rank}: DDP scaling {ddp_grad[0].item()} != FSDP scaling "
        f"{fsdp_out.float().mean().item()} -- the same learning_rate would mean "
        f"different things in the two strategies."
    )
