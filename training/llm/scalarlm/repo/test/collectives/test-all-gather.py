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
from cray_infra.training.distributed import get_size, get_rank, allgather, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

# Element counts, kept a multiple of world size so every rank contributes an
# equal block.
block_sizes = [2 ** i for i in range(10)]

def allgather_test(block):
    world_size = get_size()
    rank = get_rank()

    # Rank-distinguishable payload: rank r sends (r+1). A no-op collective, or
    # one that fills a GPU copy and leaves this CPU buffer untouched, cannot
    # produce the expected pattern. The previous version of this test filled the
    # output with ones and printed the INPUT, so it passed unconditionally.
    input_tensor = torch.full((block,), float(rank + 1), dtype=torch.float32)
    output_tensor = torch.zeros(world_size * block, dtype=torch.float32)

    allgather(input_tensor, output_tensor)

    expected = torch.cat(
        [torch.full((block,), float(r + 1), dtype=torch.float32) for r in range(world_size)]
    )
    assert torch.equal(output_tensor, expected), (
        f"allgather block={block} rank={rank}: expected {expected.tolist()[:8]}..., "
        f"got {output_tensor.tolist()[:8]}..."
    )

    barrier()

    if rank == 0:
        print(f"[PASS] allgather block={block} -> per-rank blocks 1..{world_size}")

with training_job_context():
    for block in block_sizes:
        allgather_test(block)

    if get_rank() == 0:
        print("RESULT: all allgather tests passed.")

"""


main()

