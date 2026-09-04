"""Regression test: allreduce must SUM low-precision floats, not their bit patterns.

MPI defines no bfloat16/float16 datatype. common.h's get_typesize() maps both to
MPI_SHORT, which is correct for the byte-moving collectives (send/recv/allgather)
that only need the right element width -- but mpi_allreduce is the one collective
that does arithmetic, and MPI_SUM over MPI_SHORT adds the raw IEEE bit patterns as
16-bit signed integers.

Before mpi_allreduce learned to promote to float32, this returned, for 1.0+1.0:
    bfloat16 -> inf        (0x3F80 + 0x3F80 = 0x7F00)
    float16  -> 32768.0    (0x3C00 + 0x3C00 = 0x7800)
which silently corrupted the classification loss sync (a 4-GPU run reported loss
0.0000 -> -5.2e35 while the identical 1-GPU run trained to 0.5015) and any bf16
gradient handed to DDP.backward_sync().

Run under mpirun, e.g.:
    mpirun --allow-run-as-root -n 2 python test/collectives/test-allreduce-dtypes.py
"""

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
        # Every rank contributes 1.0, so a correct MPI_SUM gives world_size.
        tensor = torch.ones(8, dtype=dtype, device=device)
        allreduce(tensor)

        got = float(tensor.float().mean().item())
        expected = float(world_size)

        # bf16 has ~3 decimal digits; a tolerance well inside its resolution
        # still separates a correct sum from bit-pattern garbage by many orders
        # of magnitude.
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
    # Every rank that touched MPI must finalize, or OpenMPI reports the run as an
    # abnormal termination and mpirun exits non-zero even though the test passed.
    finalize_mpi()
    sys.exit(status)
