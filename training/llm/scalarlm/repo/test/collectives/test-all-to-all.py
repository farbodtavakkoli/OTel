import scalarlm

import logging

scalarlm.api_url = "http://localhost:8000"


def main():
    llm = scalarlm.SupermassiveIntelligence()

    gpu_count = max(2, llm.get_gpu_count())

    status = llm.submit_slurm_job(
        code=get_code(), train_args={"gpus": gpu_count, "max_gpus": gpu_count}
    )

    print(status)


def get_code():
    return """
from cray_infra.training.distributed import get_size, get_rank, alltoall, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

# Per-rank chunk sizes. The total element count must be divisible by world size,
# which the previous `size // 4` byte-derived counts were not (size=4 gave 1
# element on 8 ranks -> ValueError), so this test could not have been passing.
chunk_sizes = [2 ** i for i in range(10)]

def alltoall_test(chunk):
    world_size = get_size()
    rank = get_rank()

    # Rank r sends value (r*world_size + j) to rank j, so after the exchange
    # rank r must hold (j*world_size + r) in chunk j. This validates ROUTING,
    # not just liveness: a transpose bug or a dropped write-back both fail.
    input_tensor = torch.cat(
        [torch.full((chunk,), float(rank * world_size + j), dtype=torch.float32)
         for j in range(world_size)]
    )
    output_tensor = torch.zeros(world_size * chunk, dtype=torch.float32)

    alltoall(input_tensor, output_tensor)

    expected = torch.cat(
        [torch.full((chunk,), float(j * world_size + rank), dtype=torch.float32)
         for j in range(world_size)]
    )
    assert torch.equal(output_tensor, expected), (
        f"alltoall chunk={chunk} rank={rank}: expected {expected.tolist()[:8]}..., "
        f"got {output_tensor.tolist()[:8]}..."
    )

    barrier()

    if rank == 0:
        print(f"[PASS] alltoall chunk={chunk} -> routing verified across {world_size} ranks")

with training_job_context():
    for chunk in chunk_sizes:
        alltoall_test(chunk)

    if get_rank() == 0:
        print("RESULT: all alltoall tests passed.")

"""


main()


