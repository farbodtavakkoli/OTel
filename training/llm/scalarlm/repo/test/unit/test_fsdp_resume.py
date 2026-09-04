"""
Tests for FSDP checkpoint reshard-on-load.

unwrap_model() saves all-gathered FULL tensors, but on resume the live model
is sharded. load_unwrapped_model() must re-derive each rank's shard (the
inverse of shard_tensor) and copy it into the shard_<name> parameter, so a
resumed FSDP run restores its trained weights instead of silently reverting
to base (the checkpoint-boundary loss-jump bug, FSDP edition).

shard_full_tensor is exercised directly across world sizes (no MPI needed).
The full unwrap -> load round-trip runs through real SimpleFSDP at
world_size=1, where sharding is identity and the stubbed allgather is a copy.
"""

import torch
from torch import nn

from cray_megatron.megatron.distribution.fsdp import (
    SimpleFSDP,
    shard_full_tensor,
)


# ---- shard_full_tensor: inverse-of-shard, any world size ------------------


def _meta(original_numel, shape, shard_size, padding, world_size):
    return {r: (original_numel, tuple(shape), shard_size, padding) for r in range(world_size)}


def test_shard_full_tensor_no_padding():
    full = torch.arange(6, dtype=torch.float32)  # numel divisible by 2
    md = _meta(6, (6,), shard_size=3, padding=0, world_size=2)

    assert torch.equal(shard_full_tensor(full, md, 0, 2), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(shard_full_tensor(full, md, 1, 2), torch.tensor([3.0, 4.0, 5.0]))


def test_shard_full_tensor_with_padding():
    # numel=5 across 2 ranks -> shard_size=3, one zero of trailing padding.
    full = torch.arange(5, dtype=torch.float32)
    md = _meta(5, (5,), shard_size=3, padding=1, world_size=2)

    assert torch.equal(shard_full_tensor(full, md, 0, 2), torch.tensor([0.0, 1.0, 2.0]))
    # rank 1 gets [3, 4, <pad 0>]
    assert torch.equal(shard_full_tensor(full, md, 1, 2), torch.tensor([3.0, 4.0, 0.0]))


def test_shard_full_tensor_flattens_2d():
    full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    md = _meta(6, (2, 3), shard_size=2, padding=0, world_size=3)

    assert torch.equal(shard_full_tensor(full, md, 0, 3), torch.tensor([0.0, 1.0]))
    assert torch.equal(shard_full_tensor(full, md, 1, 3), torch.tensor([2.0, 3.0]))
    assert torch.equal(shard_full_tensor(full, md, 2, 3), torch.tensor([4.0, 5.0]))


def test_shard_full_tensor_reassembles_to_original():
    # Sharding all ranks then concatenating + trimming padding recovers the
    # original flat tensor — i.e. shard_full_tensor is a faithful split.
    full = torch.arange(7, dtype=torch.float32)
    world_size = 3
    shard_size = (7 + world_size - 1) // world_size  # 3
    padding = shard_size * world_size - 7  # 2
    md = _meta(7, (7,), shard_size, padding, world_size)

    shards = [shard_full_tensor(full, md, r, world_size) for r in range(world_size)]
    rebuilt = torch.cat(shards)[:7]
    assert torch.equal(rebuilt, full)


# ---- full unwrap -> load round-trip through real SimpleFSDP (world_size=1) -


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.out = nn.Linear(3, 2)


def test_fsdp_unwrap_then_load_restores_weights():
    torch.manual_seed(0)

    # "Trained" model with known (random, non-zero) weights.
    trained = SimpleFSDP(_Tiny())
    saved = trained.unwrap_model()
    assert {"lin.weight", "lin.bias", "out.weight", "out.bias"} <= set(saved)
    # Every saved tensor is non-trivial so a no-op load would be detectable.
    assert all(t.abs().sum() > 0 for t in saved.values())

    # Fresh model with zeroed weights (stands in for the base-init on resume).
    fresh_inner = _Tiny()
    with torch.no_grad():
        for p in fresh_inner.parameters():
            p.zero_()
    fresh = SimpleFSDP(fresh_inner)

    # Sanity: before load, fresh gathers to all-zeros.
    before = fresh.unwrap_model()
    assert all(t.abs().sum() == 0 for t in before.values())

    fresh.load_unwrapped_model(saved)

    # After load, gathering fresh reproduces the trained tensors.
    after = fresh.unwrap_model()
    for k, v in saved.items():
        assert torch.allclose(after[k], v), f"{k} not restored"


def test_fsdp_load_raises_on_unrelated_state_dict():
    fresh = SimpleFSDP(_Tiny())
    try:
        fresh.load_unwrapped_model({"totally.unrelated": torch.zeros(2)})
    except RuntimeError as e:
        assert "missing" in str(e).lower() or "matched no" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError on namespace mismatch")
