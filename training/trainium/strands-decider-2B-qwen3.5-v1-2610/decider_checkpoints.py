"""Shared rank-checkpoint format, resume and full-export comparison."""

import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from decider_recipe_utils import require, write_json

COMPLETION_FILE = "complete.json"
DROPOUT_SEED_POLICY = "training-position-v1"


def local_tensor(value):
    """Return a local shard, or an already-local optimizer tensor."""
    return value.to_local() if isinstance(value, DTensor) else value


def rank_path(checkpoint, rank):
    return Path(checkpoint) / f"rank-{rank}.pt"


def read_completion(checkpoint):
    completion = json.loads((Path(checkpoint) / COMPLETION_FILE).read_text())
    require(isinstance(completion, dict), "Invalid checkpoint completion marker")
    for field in ("step", "world"):
        value = completion.get(field)
        require(type(value) is int and value > 0, "Invalid checkpoint " + field)
    require(
        all(
            rank_path(checkpoint, rank).is_file() for rank in range(completion["world"])
        ),
        "Checkpoint is missing one or more rank files",
    )
    return completion


def write_completion(checkpoint, step, world):
    require(
        all(rank_path(checkpoint, rank).is_file() for rank in range(world)),
        "Cannot complete a checkpoint with missing ranks",
    )
    write_json(Path(checkpoint) / COMPLETION_FILE, dict(step=step, world=world))


def load_rank_checkpoint(checkpoint, rank, completion=None):
    completion = read_completion(checkpoint) if completion is None else completion
    require(0 <= rank < completion["world"], "Invalid checkpoint rank")
    state = torch.load(
        rank_path(checkpoint, rank), map_location="cpu", weights_only=True
    )
    require(isinstance(state, dict), "Invalid checkpoint state")
    require(state.get("world") == completion["world"], "Checkpoint topology mismatch")
    require(state.get("step") == completion["step"], "Checkpoint step mismatch")
    require(isinstance(state.get("parameters"), dict), "Missing checkpoint parameters")
    return state


def capture_neuron_rng(collator):
    from torch_neuronx.python_ops.nki_kernels.rng_state import get_rng_state

    return dict(
        hardware_rng=get_rng_state().view(torch.int32).cpu(),
        cpu_rng=torch.get_rng_state(),
        neuron_rng=torch.neuron.get_rng_state(),
        collator_rng=collator.rng.getstate(),
    )


def restore_neuron_rng(state, collator):
    from torch_neuronx.python_ops.nki_kernels.rng_state import set_rng_state

    torch.neuron.set_rng_state(state["neuron_rng"])
    set_rng_state(state["hardware_rng"].to("neuron").unsqueeze(0))
    torch.set_rng_state(state["cpu_rng"])
    collator.rng.setstate(state["collator_rng"])


def save_checkpoint(
    checkpoint,
    model,
    optimizer,
    collator,
    *,
    step,
    rank,
    world,
    data_sha,
    signature,
    capture_rng=capture_neuron_rng,
):
    checkpoint = Path(checkpoint)
    if rank == 0:
        checkpoint.mkdir()
    dist.barrier()
    state = dict(
        parameters={
            name: local_tensor(parameter).detach().cpu()
            for name, parameter in model.named_parameters()
        },
        optimizer={
            name: {
                key: local_tensor(value).cpu()
                for key, value in optimizer.state[parameter].items()
            }
            for name, parameter in model.named_parameters()
        },
        step=step,
        world=world,
        data_sha=data_sha,
        signature=signature,
        dropout_seed_policy=DROPOUT_SEED_POLICY,
        **capture_rng(collator),
    )
    torch.save(state, rank_path(checkpoint, rank))
    dist.barrier()
    if rank == 0:
        write_completion(checkpoint, step, world)


def restore_checkpoint(
    checkpoint,
    model,
    optimizer,
    collator,
    *,
    rank,
    world,
    data_sha,
    signature,
    restore_rng=restore_neuron_rng,
):
    completion = read_completion(checkpoint)
    require(
        completion["world"] == world, "Incomplete or incompatible checkpoint topology"
    )
    state = load_rank_checkpoint(checkpoint, rank, completion)
    require(state.get("data_sha") == data_sha, "Resume dataset differs")
    require(
        state.get("signature") == signature,
        "Resume recipe, initialization or environment differs",
    )
    require(
        state.get("dropout_seed_policy") == DROPOUT_SEED_POLICY,
        "Resume seed policy differs",
    )
    require(
        all(
            key in state
            for key in ("hardware_rng", "cpu_rng", "neuron_rng", "collator_rng")
        ),
        "Checkpoint lacks RNG state; exact dropout resume is unavailable",
    )
    parameters = dict(model.named_parameters())
    require(
        set(state["parameters"]) == set(parameters), "Resume parameter names differ"
    )
    require(
        set(state.get("optimizer", {})) == set(parameters),
        "Resume optimizer names differ",
    )
    for name, parameter in parameters.items():
        require(
            state["parameters"][name].shape == local_tensor(parameter).shape,
            "Resume parameter shard shape differs: " + name,
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            local = local_tensor(parameter)
            local.copy_(state["parameters"][name].to(local.device))
    for name, parameter in parameters.items():
        device = local_tensor(parameter).device
        optimizer.state[parameter] = {
            key: (
                DTensor.from_local(
                    value.to(device),
                    parameter.device_mesh,
                    parameter.placements,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
                if isinstance(parameter, DTensor) and key != "step"
                else value if key == "step" else value.to(device)
            )
            for key, value in state["optimizer"][name].items()
        }
    restore_rng(state, collator)
    return state["step"]


def verify_checkpoint_export(checkpoint, model):
    """Compare full parameters to ordered rank shards using the shared reader."""
    completion = read_completion(checkpoint)
    pieces = {}
    identity = None
    for rank in range(completion["world"]):
        state = load_rank_checkpoint(checkpoint, rank, completion)
        current_identity = (
            state.get("signature"),
            state.get("data_sha"),
            state.get("dropout_seed_policy"),
        )
        if rank == 0:
            identity = current_identity
            pieces = {name: [] for name in state["parameters"]}
        require(current_identity == identity, "Checkpoint rank provenance differs")
        require(
            set(state["parameters"]) == set(pieces),
            "Checkpoint rank parameter names differ",
        )
        for name, value in state["parameters"].items():
            pieces[name].append(value)
        del state
        gc.collect()
    require(
        set(pieces) == dict(model.named_parameters()).keys(),
        "Export parameter names differ",
    )
    for name, parameter in model.named_parameters():
        require(
            torch.equal(parameter, torch.cat(pieces.pop(name), dim=0)),
            "Export differs from training shards: " + name,
        )
