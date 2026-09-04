from cray_infra.training.distributed import allreduce, get_size
from cray_infra.training.train_debug import is_train_debug_enabled

import torch
import torch.nn as nn


def _assert_uniform_participation(synced, world_size):
    """Fail loudly if ranks reduced different numbers of gradient tensors."""
    counts = torch.tensor([float(synced)], dtype=torch.float64)
    allreduce(counts)
    expected = float(synced) * world_size
    if counts.item() != expected:
        raise RuntimeError(
            "DDP gradient participation is not uniform across ranks: this rank "
            f"reduced {synced} tensors, cross-rank total is {counts.item():.0f} "
            f"(expected {expected:.0f} if every rank matched). Ranks are about to "
            "disagree on the collective sequence, which hangs the job. This "
            "wrapper supports dense models only -- data-dependent parameter "
            "participation (e.g. MoE expert routing) is not supported."
        )


class DDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def backward_sync(self):
        """All-reduce gradients across data-parallel ranks, then average."""
        world_size = get_size()

        if world_size == 1:
            return

        synced = 0
        for param_name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                allreduce(param.grad)
                param.grad /= world_size
                synced += 1

        if is_train_debug_enabled():
            _assert_uniform_participation(synced, world_size)
