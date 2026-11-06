import torch
from torch import nn

from cray_megatron.megatron.distribution.fsdp import SimpleFSDP


class _ReadsChildWeightInParentForward(nn.Module):
    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)

    def forward(self, idx, probs):
        called = self.embed(idx)
        soft = torch.matmul(probs, self.embed.weight)
        return called, soft


def _model():
    torch.manual_seed(0)
    model = _ReadsChildWeightInParentForward()
    try:
        from cray_infra.training.distributed import cuda_device

        model = model.to(cuda_device())
    except Exception:
        pass
    return model


def _device_of(module):
    try:
        from cray_infra.training.distributed import cuda_device

        return cuda_device()
    except Exception:
        for p in module.parameters():
            return p.device
        return torch.device("cpu")


def _inputs(module, n=3):
    device = _device_of(module)
    idx = torch.arange(n, device=device)
    probs = torch.zeros(n, 8, device=device)
    probs[:, 0] = 1.0
    return idx, probs


def test_direct_weight_access_keeps_full_shape():
    inner = _model()
    expected_shape = tuple(inner.embed.weight.shape)

    wrapped = SimpleFSDP(inner)
    idx, probs = _inputs(wrapped)

    called, soft = wrapped(idx, probs)

    assert tuple(soft.shape) == (3, expected_shape[1])
    assert torch.isfinite(soft).all()


def test_direct_weight_access_matches_unwrapped_values():
    wrapped = SimpleFSDP(_model())
    idx, probs = _inputs(wrapped)
    got_called, got_soft = wrapped(idx, probs)

    reference = _model().to(_device_of(wrapped))
    ref_called, ref_soft = reference(idx, probs)

    assert torch.allclose(got_called, ref_called, atol=1e-6)
    assert torch.allclose(got_soft, ref_soft, atol=1e-6)


def test_direct_weight_access_is_differentiable():
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
    inner = _model()
    full_numel = inner.embed.weight.numel()
    wrapped = SimpleFSDP(inner)
    idx, probs = _inputs(wrapped, n=1)
    wrapped(idx, probs)

    shard_params = [
        p for n, p in wrapped.named_parameters() if n.endswith("shard_weight")
    ]
    assert shard_params, "shard_weight parameter missing after forward"
    assert shard_params[0].numel() <= full_numel
