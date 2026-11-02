"""Four-rank native FSDP2 optimizer-update and CPU-reference check."""

import json
import os
import torch
import torch.distributed as dist
import torch_neuronx
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group("neuron")
rank, world = dist.get_rank(), dist.get_world_size()
torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
try:
    torch.set_num_threads(2)
    torch.manual_seed(19)
    reference = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)
    )
    model.load_state_dict(reference.state_dict())
    x = torch.randn(world, 16)
    y = torch.arange(world) % 4
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    torch.nn.functional.cross_entropy(reference(x), y).backward()
    reference_optimizer.step()
    model.to("neuron")
    mesh = init_device_mesh("neuron", (world,))
    fully_shard(model[0], mesh=mesh, reshard_after_forward=True)
    fully_shard(model[2], mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(
        model(x[rank : rank + 1].to("neuron")), y[rank : rank + 1].to("neuron")
    )
    loss.backward()
    for p in model.parameters():
        if not (p.grad is not None):
            raise ValueError("Validation failed: p.grad is not None")
        if not (p.grad.to_local().device.type == "neuron"):
            raise ValueError(
                "Validation failed: p.grad.to_local().device.type == 'neuron'"
            )
    optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(
            actual.full_tensor().cpu(), expected, atol=2e-5, rtol=2e-4
        )
    print(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "status": "passed",
                "scope": "small_fsdp2_adamw_cpu_parity",
            }
        ),
        flush=True,
    )
finally:
    dist.destroy_process_group()
