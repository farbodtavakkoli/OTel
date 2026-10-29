import logging

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from cray_infra.training.distributed import get_rank, get_size
from cray_infra.util.get_job_config import get_job_config

logger = logging.getLogger(__name__)

# Minimum children for a ModuleList to be treated as the transformer stack.
_MIN_TRANSFORMER_BLOCKS = 4


def _mesh_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "neuron"):
        return "neuron"
    return "cpu"


def _adapter_only_state_dict():
    return get_job_config().get("adapter_type", "none") not in (None, "", "none")


def _check_adapter_keys_match(model, checkpoint_keys):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    if not checkpoint_keys & trainable:
        raise RuntimeError(
            f"Checkpoint resume matched no trainable parameter: none of its "
            f"{len(checkpoint_keys)} keys, e.g. {sorted(checkpoint_keys)[:3]}, "
            f"names a requires_grad parameter of the live model."
        )
    missing = sorted(trainable - checkpoint_keys)
    if missing:
        raise RuntimeError(
            f"Checkpoint resume is missing {len(missing)} trainable parameter(s) "
            f"(missing keys), e.g. {missing[:3]}. Refusing to continue with "
            f"partially-restored weights."
        )
    unexpected = sorted(checkpoint_keys - set(model.state_dict().keys()))
    if unexpected:
        raise RuntimeError(
            f"Checkpoint resume left {len(unexpected)} trained "
            f"parameter(s) unloaded (unexpected keys), e.g. "
            f"{unexpected[:3]}. Refusing to continue with "
            f"partially-restored weights."
        )


def build_hsdp_mesh():
    job_config = get_job_config()
    shard_size = int(job_config.get("hsdp_shard_size", 1) or 1)
    world_size = get_size()

    if shard_size <= 1:
        return None, 1, world_size

    if world_size % shard_size != 0:
        raise ValueError(
            f"hsdp_shard_size={shard_size} does not divide world_size={world_size}"
        )

    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        _mesh_device_type(),
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    return mesh, replicate_size, shard_size


class PyTorchFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        rank = get_rank()
        mesh, replicate_size, shard_size = build_hsdp_mesh()
        job_config = get_job_config()

        # Trades peak memory against a re-all-gather in backward.
        self._fully_shard_kwargs = {
            "reshard_after_forward": bool(
                job_config.get("reshard_after_forward", False)
            )
        }
        if mesh is not None:
            self._fully_shard_kwargs["mesh"] = mesh

        if mesh is None:
            logger.info("Applying PyTorch FSDP2 (fully_shard) over %d ranks", get_size())
        else:
            logger.info(
                "Applying HSDP via FSDP2 - replicate=%d x shard=%d",
                replicate_size,
                shard_size,
            )

        # No activation checkpointing here: load_model.py owns gradient_checkpointing.

        # Bottom-up: shard the transformer blocks first, then the root module.
        self._apply_fsdp(model)
        fully_shard(model, **self._fully_shard_kwargs)

        self.model = model

    # Construction helpers

    def _transformer_stacks(self, model):
        stacks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > _MIN_TRANSFORMER_BLOCKS:
                if not any(name.startswith(prefix + ".") for prefix, _ in stacks):
                    stacks.append((name, module))
        return stacks

    def _apply_fsdp(self, model):
        stacks = self._transformer_stacks(model)
        if not stacks:
            logger.warning(
                "No ModuleList with more than %d children found; sharding only the "
                "root module. Check that this is the intended model architecture.",
                _MIN_TRANSFORMER_BLOCKS,
            )
            return

        for name, stack in stacks:
            for layer in stack:
                fully_shard(layer, **self._fully_shard_kwargs)
            logger.info("Applied fully_shard to %d transformer blocks in %s", len(stack), name)

    # Distribution strategy interface

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward_sync(self):
        return None

    def unwrap_model(self):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            self.model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=_adapter_only_state_dict(),
            ),
        )

    def unwrap_optimizer(self, optimizer):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
        )

        return get_optimizer_state_dict(
            self.model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def load_unwrapped_optimizer(self, optimizer, state_dict):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_optimizer_state_dict,
        )

        set_optimizer_state_dict(
            self.model,
            optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    def load_unwrapped_model(self, state_dict):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        adapter_only = _adapter_only_state_dict()
        if adapter_only:
            _check_adapter_keys_match(self.model, set(state_dict or {}))
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=not adapter_only,
            ),
        )

    def __getattr__(self, name):
        # Fall through to the wrapped model so callers can reach its attributes.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["model"], name)
