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
from cray_infra.training.distributed import get_size, get_rank, send, recv, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

block_sizes = [2 ** i for i in range(10)]

def send_recv_test(block):
    # cross the network bisection
    rank = get_rank()
    bisection_rank = get_size() // 2

    if bisection_rank == 0:
        print("Bisection rank is 0, skipping test to avoid deadlock.")
        return

    neighbor = (rank + bisection_rank) % get_size()

    if rank < bisection_rank:
        # Sender's payload identifies the sender.
        send(torch.full((block,), float(rank + 1), dtype=torch.float32), neighbor)
        print(f"[send] rank {rank} -> rank {neighbor} block={block}")
    else:
        # Receive into a ZEROED buffer and assert. The old test recv'd into a
        # buffer pre-filled with ones and only printed, so a recv() that filled a
        # GPU copy and never wrote back to this CPU tensor looked like success --
        # which is precisely the defect this stack had to fix.
        received_tensor = torch.zeros(block, dtype=torch.float32)
        recv(received_tensor, neighbor)

        expected = float(neighbor + 1)
        assert torch.equal(
            received_tensor, torch.full((block,), expected, dtype=torch.float32)
        ), (
            f"recv block={block} rank={rank} from {neighbor}: expected all "
            f"{expected}, got {received_tensor.tolist()[:8]}..."
        )
        print(f"[PASS] rank {rank} received {expected} from rank {neighbor} block={block}")

    barrier()

with training_job_context():
    for block in block_sizes:
        send_recv_test(block)

    if get_rank() == 0:
        print("RESULT: all send/recv tests passed.")


"""


main()
