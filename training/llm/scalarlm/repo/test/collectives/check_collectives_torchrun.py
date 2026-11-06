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
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.full((n,), float(rank + 1), dtype=dtype, device=device)
            allreduce(x)
            exp = float(world * (world + 1) // 2)
            assert torch.allclose(x.float(), torch.full((n,), exp, device=device)), f"allreduce {dtype} n={n} rank {rank}: {x[:4]}"
            checks += 1

        send = torch.full((n,), float(rank + 1), dtype=torch.float32, device=device)
        recv = torch.zeros((n * world,), dtype=torch.float32, device=device)
        allgather(send, recv)
        exp = torch.arange(1, world + 1, dtype=torch.float32, device=device).repeat_interleave(n)
        assert torch.equal(recv, exp), f"allgather n={n} rank {rank}"
        checks += 1

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
