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
from cray_infra.training.distributed import get_size, get_rank, allreduce, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

block_sizes = [2 ** i for i in range(10)]

def allreduce_test(block):
    world_size = get_size()
    rank = get_rank()

    # Rank r contributes (r+1), so a correct SUM is world*(world+1)/2 -- a value
    # no rank holds locally. Contributing ones (as this test used to) cannot
    # distinguish a real reduction from no reduction at all.
    my_tensor = torch.full((block,), float(rank + 1), dtype=torch.float32)

    allreduce(my_tensor)

    expected = float(world_size * (world_size + 1) // 2)
    assert torch.equal(my_tensor, torch.full((block,), expected, dtype=torch.float32)), (
        f"allreduce block={block} rank={rank}: expected all {expected}, "
        f"got {my_tensor.tolist()[:8]}..."
    )

    barrier()

    if rank == 0:
        print(f"[PASS] allreduce block={block} -> {expected}")

with training_job_context():
    for block in block_sizes:
        allreduce_test(block)

    if get_rank() == 0:
        print("RESULT: all allreduce tests passed.")

"""


main()

