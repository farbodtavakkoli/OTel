import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="distribution strategies require a GPU"
)


@pytest.fixture(autouse=True, scope="module")
def _job_config(tmp_path_factory):
    import os

    if os.environ.get("CRAY_TRAINING_JOB_CONFIG_PATH"):
        yield
        return

    path = tmp_path_factory.mktemp("jobcfg") / "config.yaml"
    path.write_text(
        "job_directory: /tmp\n"
        "training_data_path: /tmp/dataset.jsonlines\n"
        "dataset_hash: smoke\n"
        "hsdp_shard_size: 1\n"
        "reshard_after_forward: false\n"
    )
    os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"] = str(path)
    try:
        yield
    finally:
        os.environ.pop("CRAY_TRAINING_JOB_CONFIG_PATH", None)


def _device():
    from cray_infra.training.distributed import cuda_device

    return cuda_device()


class _TinyNet(nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, probs):
        hidden = self.proj(self.embed(idx))
        logits = self.head(hidden)
        soft = torch.matmul(probs, self.embed.weight)
        return logits, soft


def _build(strategy_name):
    from cray_megatron.megatron.distribution.ddp import DDP
    from cray_megatron.megatron.distribution.fsdp import SimpleFSDP

    torch.manual_seed(1234)
    model = _TinyNet().to(_device())

    if strategy_name == "ddp":
        return DDP(model)
    if strategy_name == "fsdp":
        return SimpleFSDP(model)
    if strategy_name == "pytorch_fsdp":
        from cray_megatron.megatron.distribution.pytorch_fsdp import PyTorchFSDP

        return PyTorchFSDP(model)
    raise AssertionError(f"unknown strategy {strategy_name}")


def _batch(n=8, vocab=16):
    device = _device()
    generator = torch.Generator(device="cpu").manual_seed(7)
    idx = torch.randint(0, vocab, (n,), generator=generator).to(device)
    probs = torch.zeros(n, vocab, device=device)
    probs[:, 0] = 1.0
    target = torch.randint(0, vocab, (n,), generator=generator).to(device)
    return idx, probs, target


def _train_a_few_steps(wrapped, steps=12):
    optimizer = torch.optim.AdamW(
        [p for p in wrapped.parameters() if p.requires_grad], lr=5e-2
    )
    idx, probs, target = _batch()
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, soft = wrapped(idx, probs)
        loss = loss_fn(logits, target) + soft.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().float().item()))
    return losses


ALL_STRATEGIES = ["ddp", "fsdp", "pytorch_fsdp"]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_wraps_and_runs_forward(strategy):
    wrapped = _build(strategy)
    idx, probs, _ = _batch()

    logits, soft = wrapped(idx, probs)

    assert torch.isfinite(logits).all(), f"{strategy}: non-finite logits"
    assert torch.isfinite(soft).all(), f"{strategy}: non-finite soft embeddings"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_direct_weight_access_has_full_shape(strategy):
    wrapped = _build(strategy)
    idx, probs, _ = _batch()

    _, soft = wrapped(idx, probs)

    assert soft.shape == (probs.shape[0], 8), (
        f"{strategy}: direct .weight access produced {tuple(soft.shape)}, "
        "expected (batch, dim) -- a sharded/flattened weight leaked out"
    )


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_loss_is_finite_every_step(strategy):
    losses = _train_a_few_steps(_build(strategy))

    assert all(
        torch.isfinite(torch.tensor(loss)) for loss in losses
    ), f"{strategy}: non-finite loss in {losses}"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_actually_learns(strategy):
    losses = _train_a_few_steps(_build(strategy))

    assert losses[-1] < losses[0], (
        f"{strategy}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}); "
        "this is the silent 'COMPLETED but learned nothing' failure mode"
    )


def _snapshot(wrapped):
    values = []
    for p in wrapped.parameters():
        if not p.requires_grad:
            continue
        tensor = p.detach()
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            tensor = to_local()
        values.append(tensor.clone().float().cpu())
    return values


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_strategy_updates_parameters(strategy):
    wrapped = _build(strategy)
    before = _snapshot(wrapped)

    _train_a_few_steps(wrapped)

    after = _snapshot(wrapped)
    assert before, f"{strategy}: no trainable parameters found"
    assert any(
        not torch.allclose(b, a) for b, a in zip(before, after)
    ), f"{strategy}: no parameter changed after training steps"
