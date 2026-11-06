import sys

import torch

from cray_infra.training.distributed import allreduce, finalize_mpi, get_rank, get_size

DTYPES = (
    ("float32", torch.float32),
    ("bfloat16", torch.bfloat16),
    ("float16", torch.float16),
)


def main():
    rank = get_rank()
    world_size = get_size()

    if torch.cuda.is_available():
        device = "cuda:%d" % (rank % torch.cuda.device_count())
    else:
        device = "cpu"

    failures = []

    for name, dtype in DTYPES:
        tensor = torch.ones(8, dtype=dtype, device=device)
        allreduce(tensor)

        got = float(tensor.float().mean().item())
        expected = float(world_size)

        ok = abs(got - expected) <= 0.01 * expected

        if rank == 0:
            print(
                f"{name:9s} expected {expected:.3f} got {got:.6g} "
                f"{'OK' if ok else 'CORRUPT'}"
            )

        if not ok:
            failures.append(f"{name}: expected {expected}, got {got}")

    if failures:
        if rank == 0:
            print("FAILED: " + "; ".join(failures))
        return 1

    if rank == 0:
        print(f"allreduce sums correctly for all dtypes across {world_size} ranks")
    return 0


if __name__ == "__main__":
    status = main()

    finalize_mpi()
    sys.exit(status)
