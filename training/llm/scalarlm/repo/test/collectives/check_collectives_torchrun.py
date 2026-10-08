"""Collective correctness at an arbitrary rank count, launched with torchrun (no Slurm, no server).

    torchrun --nnodes=1 --nproc-per-node=3 test/collectives/check_collectives_torchrun.py

Every rank contributes a value only it holds (rank + 1), so a wrong or missing reduction cannot
produce the expected result by accident. Checks allreduce (sum), allgather, reduce_scatter and
barrier through cray_infra.training.distributed (the NCCL/RCCL layer the trainer uses), on the
device that layer selected, for several buffer sizes and for fp32 + bf16 allreduce. Rank 0 prints
``RESULT: ALL COLLECTIVES CORRECT world=<n>`` on success; any failure raises on the failing rank.
"""
import torch

from cray_infra.training.distributed import (
    allgather,
    allreduce,
    barrier,
    finalize_mpi,
    get_rank,
    get_size,
    init,
    reduce_scatter,
)


def main():
    init()
    world, rank = get_size(), get_rank()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    checks = 0
    for n in (1, 7, 1024, 1 << 20):
        # allreduce SUM: expected world*(world+1)/2 in every slot
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.full((n,), float(rank + 1), dtype=dtype, device=device)
            allreduce(x)
            exp = float(world * (world + 1) // 2)
            assert torch.allclose(x.float(), torch.full((n,), exp, device=device)), f"allreduce {dtype} n={n} rank {rank}: {x[:4]}"
            checks += 1
        # allgather: slot k must hold k+1
        send = torch.full((n,), float(rank + 1), dtype=torch.float32, device=device)
        recv = torch.zeros((n * world,), dtype=torch.float32, device=device)
        allgather(send, recv)
        exp = torch.arange(1, world + 1, dtype=torch.float32, device=device).repeat_interleave(n)
        assert torch.equal(recv, exp), f"allgather n={n} rank {rank}"
        checks += 1
        # reduce_scatter: chunk r of the SUM -> every slot world*(world+1)/2 (inputs are constant)
        if n % world == 0 or n >= world:
            m = (n // world) * world
            send = torch.full((m,), float(rank + 1), dtype=torch.float32, device=device)
            recv = torch.zeros((m // world,), dtype=torch.float32, device=device)
            reduce_scatter(send, recv)
            exp = float(world * (world + 1) // 2)
            assert torch.allclose(recv, torch.full_like(recv, exp)), f"reduce_scatter n={m} rank {rank}"
            checks += 1
        barrier()
    barrier()
    if rank == 0:
        print(f"RESULT: ALL COLLECTIVES CORRECT world={world} checks_per_rank={checks} device={device} "
              f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else ''}", flush=True)
    finalize_mpi()


if __name__ == "__main__":
    main()
