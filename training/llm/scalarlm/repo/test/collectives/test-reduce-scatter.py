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
from cray_infra.training.distributed import get_size, get_rank, reduce_scatter, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

block_sizes = [2 ** i for i in range(10)]

def reduce_scatter_test(block):
    world_size = get_size()
    rank = get_rank()

    # Rank r contributes (r+1) everywhere; each output shard is therefore the
    # SUM world*(world+1)/2. The old version summed ones into an output that was
    # pre-filled with ones and then printed the INPUT, so it could not fail.
    input_tensor = torch.full((world_size * block,), float(rank + 1), dtype=torch.float32)
    output_tensor = torch.zeros(block, dtype=torch.float32)

    reduce_scatter(input_tensor, output_tensor)

    expected = float(world_size * (world_size + 1) // 2)
    assert torch.equal(output_tensor, torch.full((block,), expected, dtype=torch.float32)), (
        f"reduce_scatter block={block} rank={rank}: expected all {expected}, "
        f"got {output_tensor.tolist()[:8]}..."
    )

    barrier()

    if rank == 0:
        print(f"[PASS] reduce_scatter block={block} -> {expected}")

with training_job_context():
    for block in block_sizes:
        reduce_scatter_test(block)

    if get_rank() == 0:
        print("RESULT: all reduce_scatter tests passed.")

"""


main()

