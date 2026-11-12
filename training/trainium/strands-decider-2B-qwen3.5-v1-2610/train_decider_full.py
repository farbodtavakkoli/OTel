"""Architecture-preserving, four-rank full-weight Decider training orchestration."""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from decider_checkpoints import local_tensor, restore_checkpoint, save_checkpoint
from decider_recipe_utils import load_rows, require, StrictTokenizer, training_signature
from decider_training_step import DropoutSeed, StepConfig, train_step


def initialize_distributed(seed):
    import torch_neuronx  # Registers the native backend before group initialization.

    dist.init_process_group("neuron")
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch._dynamo.config.recompile_limit = 64
    return dist.get_rank(), dist.get_world_size()


def log_event(output, rank, event, **fields):
    record = dict(time=time.time(), rank=rank, event=event, **fields)
    with (output / f"rank-{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def load_training_data(path):
    from training.data.format import Example

    rows = [Example.from_dict(row) for row in load_rows(path)]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def configure_fsdp_model(model, world, dropout_seed):
    mesh = init_device_mesh("neuron", (world,))
    model.torso.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for layer in model.torso.layers:
        # Compile the layer body; FSDP hooks remain outside the compiled body.
        layer.forward = torch.compile(layer.forward, backend="neuron", dynamic=False)
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    fully_shard(model.torso.embed_tokens, mesh=mesh, reshard_after_forward=True)
    fully_shard(model.head, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    original_head = model.head.forward

    def head_fp32(*inputs):
        # This head is not activation-checkpointed. Seed by training position.
        if model.training:
            torch.neuron.manual_seed(dropout_seed.value)
        with torch.autocast("neuron", enabled=False):
            return original_head(*inputs)

    model.head.forward = head_fp32
    return model


def create_training_components(args, rank, world):
    from decider_model_io import load_full
    from training.data.collate import CollatorConfig, SystemOneCollator

    model = load_full(args.merged)
    model.config.temperature = 1.0
    model.config.temperature_by_kind = {}
    model.requires_grad_(True)
    model.to("neuron")
    collator = SystemOneCollator(
        StrictTokenizer(model.tokenizer, args.pad_to),
        CollatorConfig(max_length=args.pad_to, seed=args.seed + rank),
    )
    dropout_seed = DropoutSeed()
    configure_fsdp_model(model, world, dropout_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0,
        foreach=False,
    )
    return model, optimizer, collator, dropout_seed


def snapshot_parameters(model):
    return {
        name: local_tensor(parameter).detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def validate_parameter_changes(model, before, log, *, device="neuron"):
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], local_tensor(parameter).detach().cpu())
    ]
    # Vocabulary shards with no used tokens may remain unchanged locally.
    names = list(before)
    flags = torch.tensor([float(name in changed) for name in names], device=device)
    dist.all_reduce(flags)
    global_changed = [
        name for name, flag in zip(names, flags.cpu().tolist()) if flag > 0
    ]
    for prefix in ("head.", "torso.layers.", "torso.embed_tokens."):
        require(
            any(name.startswith(prefix) for name in global_changed),
            "No distributed parameter changed under " + prefix,
        )
    log("weights_changed", names=changed)
    log(
        "parameter_summary",
        tensors=len(before),
        changed_tensors=len(changed),
        global_changed_tensors=len(global_changed),
    )
    return global_changed


def gather_full_parameters(model, rank):
    # Every rank must take part in every gather, although only rank zero retains it.
    full_state = {}
    for name, parameter in model.named_parameters():
        value = parameter.full_tensor().detach().cpu()
        if rank == 0:
            full_state[name] = value
    return full_state


def export_final_model(
    model, args, *, rank, world, signature, total_steps, row_count, log
):
    from decider_model_io import load_full, save_full

    full_state = gather_full_parameters(model, rank)
    if rank == 0:
        exported = load_full(args.merged)
        exported.config.temperature = 1.0
        exported.config.temperature_by_kind = {}
        exported.load_state_dict(full_state, strict=True)
        save_full(exported, args.output / "final")
        log("full_export_saved", reload_pending=True)
        (args.output / "run.json").write_text(
            json.dumps(
                dict(
                    signature=signature,
                    steps=total_steps,
                    rows=row_count,
                    epochs=args.epochs,
                    world=world,
                    selection="last_step",
                    independent_quality_evaluation=False,
                ),
                indent=2,
            )
        )
    dist.barrier()


def run_training(args, rank, world, log):
    sys.path.insert(0, str(args.upstream))
    rows, data_sha = load_training_data(args.data)
    signature = training_signature(args, world)
    model, optimizer, collator, dropout_seed = create_training_components(
        args, rank, world
    )
    start_step = 0
    if args.resume:
        start_step = restore_checkpoint(
            args.resume,
            model,
            optimizer,
            collator,
            rank=rank,
            world=world,
            data_sha=data_sha,
            signature=signature,
        )
        log("resumed", step=start_step)
    before = snapshot_parameters(model)
    model.train()
    log(
        "training_start",
        parameters=sum(p.numel() for p in model.parameters()),
        world=world,
    )
    steps_per_epoch = (len(rows) + args.batch_size - 1) // args.batch_size
    total_steps = min(args.steps, args.epochs * steps_per_epoch)
    require(
        start_step < total_steps,
        "No remaining optimizer updates under the requested step limit",
    )
    config = StepConfig(rank=rank, world=world, pad_to=args.pad_to, seed=args.seed)
    for step in range(start_step, total_steps):
        offset = (step % steps_per_epoch) * args.batch_size
        group = rows[offset : offset + args.batch_size]
        train_step(
            model,
            optimizer,
            collator,
            group,
            step=step,
            config=config,
            dropout_seed=dropout_seed,
            log=log,
        )
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == total_steps:
            save_checkpoint(
                args.output / f"step-{step+1}",
                model,
                optimizer,
                collator,
                step=step + 1,
                rank=rank,
                world=world,
                data_sha=data_sha,
                signature=signature,
            )
    validate_parameter_changes(model, before, log)
    export_final_model(
        model,
        args,
        rank=rank,
        world=world,
        signature=signature,
        total_steps=total_steps,
        row_count=len(rows),
        log=log,
    )


def main(args):
    log = None
    try:
        rank, world = initialize_distributed(args.seed)
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        log = partial(log_event, args.output, rank)
        run_training(args, rank, world, log)
    except Exception:
        if log is not None:
            log("error", traceback=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "merged", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pad-to", type=int, default=768)
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Maximum total updates, including restored updates",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Global real examples per optimizer update",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args(argv)
    for name in ("pad_to", "steps", "epochs", "batch_size", "checkpoint_every"):
        require(getattr(args, name) > 0, name + " must be positive")
    require(0 < args.learning_rate < 1, "learning-rate must be between zero and one")
    require(
        0 <= args.seed < 2**31 and args.batch_size < 1_000_000,
        "seed/batch size exceeds supported range",
    )
    return args


if __name__ == "__main__":
    os.umask(0o077)
    main(parse_args())
