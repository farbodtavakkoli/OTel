"""One optimizer update: padded microbatches, loss scaling and gradient clipping."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from decider_checkpoints import local_tensor
from decider_recipe_utils import require


@dataclass(frozen=True)
class StepConfig:
    rank: int
    world: int
    pad_to: int
    seed: int
    # The CLI always uses Neuron. CPU permits isolated numerical unit tests.
    device: str = "neuron"


@dataclass
class DropoutSeed:
    value: int = 42


def prepare_batch(batch, *, pad_to, pad_token_id, device):
    length = batch["input_ids"].shape[1]
    require(
        length <= pad_to,
        f"Prompt length {length} exceeds padding {pad_to}; refusing truncation",
    )
    batch["input_ids"] = torch.nn.functional.pad(
        batch["input_ids"],
        (0, pad_to - length),
        value=pad_token_id,
    )
    batch["attention_mask"] = torch.nn.functional.pad(
        batch["attention_mask"],
        (0, pad_to - length),
    )
    return {key: value.to(device) for key, value in batch.items()}


def validate_and_clip_gradients(model, *, device):
    norm_sq = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        require(parameter.requires_grad and parameter.grad is not None, name)
        grad = local_tensor(parameter.grad)
        require(grad.device.type == device, name)
        require(torch.isfinite(grad.cpu()).all(), name)
        norm_sq += grad.float().square().sum()
    dist.all_reduce(norm_sq)
    scale = (1.0 / (norm_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        local_tensor(parameter.grad).mul_(scale)


def train_step(model, optimizer, collator, group, *, step, config, dropout_seed, log):
    require(bool(group), "Training batch is empty")
    optimizer.zero_grad(set_to_none=True)
    # Dummy slots contribute zero loss; world/group_size compensates FSDP's mean.
    for slot in range((len(group) + config.world - 1) // config.world):
        dropout_seed.value = (
            config.seed + step * 1_000_000 + slot * config.world + config.rank
        )
        index = config.rank + slot * config.world
        batch = prepare_batch(
            collator([group[index if index < len(group) else 0]]),
            pad_to=config.pad_to,
            pad_token_id=model.tokenizer.pad_token_id,
            device=config.device,
        )
        with torch.autocast(config.device, dtype=torch.bfloat16):
            loss = model(**batch)["loss"] * (
                config.world / len(group) if index < len(group) else 0.0
            )
        loss.backward()
        scalar = float(loss.detach().cpu())
        require(torch.isfinite(torch.tensor(scalar)), "Nonfinite microbatch loss")
        log(
            "microbatch", step=step + 1, slot=slot, loss=scalar, real=index < len(group)
        )
    validate_and_clip_gradients(model, device=config.device)
    optimizer.step()
    if config.device == "neuron":
        torch.neuron.synchronize()
        log(
            "optimizer_step",
            step=step + 1,
            peak_device_bytes=torch.neuron.max_memory_allocated(),
        )
    else:
        log("optimizer_step", step=step + 1)
