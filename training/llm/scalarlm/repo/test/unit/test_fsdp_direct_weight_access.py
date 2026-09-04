"""SimpleFSDP must expose FULL parameters when a submodule's `.weight` is read
directly from a parent's forward.

SimpleFSDP shards each wrapped module's parameters and only all-gathers them
inside that module's own `forward` (FSDPLayer.forward_op). A model that reaches
into a child's `.weight` from the *parent's* forward therefore used to see the
sharded (and flattened) tensor and blow up:

    RuntimeError: size mismatch, got input (256), mat (256x262144), vec (524288)

That is not a diffusion quirk -- it is any weight-tying / soft-embedding /
custom-logit-head pattern. DiffusionGemma's self-conditioning path is the case
that surfaced it:

    soft_embeddings = torch.matmul(probs, self.embed_tokens.weight)

These tests run at world_size=1, where sharding is identity in VALUE but still
FLATTENS a 2-D parameter to 1-D -- so the shape regression reproduces without
MPI and without multiple GPUs.
"""

import torch
from torch import nn

from cray_megatron.megatron.distribution.fsdp import SimpleFSDP


class _ReadsChildWeightInParentForward(nn.Module):
    """Parent whose forward both CALLS the child and reads the child's .weight."""

    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, idx, probs):
        called = self.embed(idx)
        # The pattern that broke: direct parameter access from the parent.
        soft = torch.matmul(probs, self.embed.weight)
        return called, soft


def _model():
    """Build the model already on the accelerator.

    Real training materialises the model on the target device before the
    distribution strategy wraps it. Constructing on CPU here would leave the
    shard Parameters on CPU while SimpleFSDP's collectives run on the
    accelerator, and backward would then fail with a device mismatch that is an
    artefact of the harness rather than the behaviour under test.
    """
    torch.manual_seed(0)
    model = _ReadsChildWeightInParentForward()
    try:
        from cray_infra.training.distributed import cuda_device

        model = model.to(cuda_device())
    except Exception:
        pass
    return model


def _device_of(module):
    """Device SimpleFSDP will gather onto.

    The shard Parameters can still live on CPU while all_gather materialises the
    full tensor on the accelerator, so ask the same helper SimpleFSDP uses
    rather than reading a shard's device.
    """
    try:
        from cray_infra.training.distributed import cuda_device

        return cuda_device()
    except Exception:
        for p in module.parameters():
            return p.device
        return torch.device("cpu")


def _inputs(module, n=3):
    """Index / probability inputs on whatever device the (wrapped) model lives on."""
    device = _device_of(module)
    idx = torch.arange(n, device=device)
    probs = torch.zeros(n, 8, device=device)
    probs[:, 0] = 1.0
    return idx, probs


def test_direct_weight_access_keeps_full_shape():
    """`child.weight` read from the parent's forward must have the FULL 2-D shape."""
    inner = _model()
    expected_shape = tuple(inner.embed.weight.shape)

    wrapped = SimpleFSDP(inner)
    idx, probs = _inputs(wrapped)

    called, soft = wrapped(idx, probs)

    assert tuple(soft.shape) == (3, expected_shape[1])
    assert torch.isfinite(soft).all()


def test_direct_weight_access_matches_unwrapped_values():
    """The gathered weight must carry the same VALUES as the unsharded model."""
    wrapped = SimpleFSDP(_model())
    idx, probs = _inputs(wrapped)
    got_called, got_soft = wrapped(idx, probs)

    reference = _model().to(_device_of(wrapped))
    ref_called, ref_soft = reference(idx, probs)

    assert torch.allclose(got_called, ref_called, atol=1e-6)
    assert torch.allclose(got_soft, ref_soft, atol=1e-6)


def test_direct_weight_access_is_differentiable():
    """Gathering on attribute access must stay in the autograd graph."""
    wrapped = SimpleFSDP(_model())
    idx, probs = _inputs(wrapped)

    _, soft = wrapped(idx, probs)
    soft.sum().backward()

    grads = [
        p.grad for p in wrapped.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no parameter received a gradient through the .weight path"
    assert any(g.abs().sum() > 0 for g in grads)


def test_weight_is_resharded_after_forward():
    """The fix must not permanently materialise the full parameter."""
    inner = _model()
    full_numel = inner.embed.weight.numel()
    wrapped = SimpleFSDP(inner)
    idx, probs = _inputs(wrapped, n=1)
    wrapped(idx, probs)

    # After forward the shard parameter is what is registered/optimised.
    shard_params = [
        p for n, p in wrapped.named_parameters() if n.endswith("shard_weight")
    ]
    assert shard_params, "shard_weight parameter missing after forward"
    assert shard_params[0].numel() <= full_numel
