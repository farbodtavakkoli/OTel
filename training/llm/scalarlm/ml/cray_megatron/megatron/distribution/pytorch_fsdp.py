"""PyTorch FSDP2 (``fully_shard``) as an alternative distribution strategy."""

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
    """Device type string for init_device_mesh (ROCm GPUs report as "cuda")."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_hsdp_mesh():
    """Return (mesh, replicate_size, shard_size)."""
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
    """FSDP2 wrapper matching this tree's distribution-strategy interface."""

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

    # -- construction helpers ------------------------------------------------

    def _transformer_stack(self, model):
        """The first ModuleList large enough to be the transformer stack, else None."""
        for _, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > _MIN_TRANSFORMER_BLOCKS:
                return module
        return None

    def _apply_fsdp(self, model):
        """Shard each transformer block, not every module with parameters."""
        stack = self._transformer_stack(model)
        if stack is None:
            logger.warning(
                "No ModuleList with more than %d children found; sharding only the "
                "root module. Check that this is the intended model architecture.",
                _MIN_TRANSFORMER_BLOCKS,
            )
            return

        for layer in stack:
            fully_shard(layer, **self._fully_shard_kwargs)
        logger.info("Applied fully_shard to %d transformer blocks", len(stack))

    # -- distribution-strategy interface -------------------------------------

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward_sync(self):
        """No-op: FSDP2 reduce-scatters gradients during backward itself."""
        return None

    def unwrap_model(self):
        """Full, unsharded, CPU state dict -- what checkpoint() writes (collective)."""
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            self.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def unwrap_optimizer(self, optimizer):
        """Full, unsharded, CPU optimizer state - the counterpart to unwrap_model (collective)."""
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
        """Inverse of unwrap_optimizer: re-shard full optimizer state into this rank (collective)."""
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
        """Inverse of unwrap_model: re-shard a full state dict into this rank (collective)."""
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

    def __getattr__(self, name):
        # Fall through to the wrapped model so callers can reach its attributes.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["model"], name)
