import torch
from torch import nn

from cray_megatron.megatron.distribution.fsdp import (
    SimpleFSDP,
    shard_full_tensor,
)


def _meta(original_numel, shape, shard_size, padding, world_size):
    return {r: (original_numel, tuple(shape), shard_size, padding) for r in range(world_size)}


def test_shard_full_tensor_no_padding():
    full = torch.arange(6, dtype=torch.float32)
    md = _meta(6, (6,), shard_size=3, padding=0, world_size=2)

    assert torch.equal(shard_full_tensor(full, md, 0, 2), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(shard_full_tensor(full, md, 1, 2), torch.tensor([3.0, 4.0, 5.0]))


def test_shard_full_tensor_with_padding():
    full = torch.arange(5, dtype=torch.float32)
    md = _meta(5, (5,), shard_size=3, padding=1, world_size=2)

    assert torch.equal(shard_full_tensor(full, md, 0, 2), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(shard_full_tensor(full, md, 1, 2), torch.tensor([3.0, 4.0, 0.0]))


def test_shard_full_tensor_flattens_2d():
    full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    md = _meta(6, (2, 3), shard_size=2, padding=0, world_size=3)

    assert torch.equal(shard_full_tensor(full, md, 0, 3), torch.tensor([0.0, 1.0]))
    assert torch.equal(shard_full_tensor(full, md, 1, 3), torch.tensor([2.0, 3.0]))
    assert torch.equal(shard_full_tensor(full, md, 2, 3), torch.tensor([4.0, 5.0]))


def test_shard_full_tensor_reassembles_to_original():
    full = torch.arange(7, dtype=torch.float32)
    world_size = 3
    shard_size = (7 + world_size - 1) // world_size
    padding = shard_size * world_size - 7
    md = _meta(7, (7,), shard_size, padding, world_size)

    shards = [shard_full_tensor(full, md, r, world_size) for r in range(world_size)]
    rebuilt = torch.cat(shards)[:7]
    assert torch.equal(rebuilt, full)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.out = nn.Linear(3, 2)


def test_fsdp_unwrap_then_load_restores_weights():
    torch.manual_seed(0)

    trained = SimpleFSDP(_Tiny())
    saved = trained.unwrap_model()
    assert {"lin.weight", "lin.bias", "out.weight", "out.bias"} <= set(saved)
    assert all(t.abs().sum() > 0 for t in saved.values())

    fresh_inner = _Tiny()
    with torch.no_grad():
        for p in fresh_inner.parameters():
            p.zero_()
    fresh = SimpleFSDP(fresh_inner)

    before = fresh.unwrap_model()
    assert all(t.abs().sum() == 0 for t in before.values())

    fresh.load_unwrapped_model(saved)

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
