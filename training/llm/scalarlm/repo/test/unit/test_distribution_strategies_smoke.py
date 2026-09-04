"""Single-node smoke test for all three distribution strategies.

This is the pytest form of the manual single-node smoke matrix (DDP, SimpleFSDP,
FSDP2) so it can be run on any box -- including H100 -- without standing up a
server, submitting a job and waiting for slurm.

For each strategy the test wraps a real model, runs real optimizer steps, and
asserts the things that actually distinguish "training" from "reporting
COMPLETED while learning nothing":

  * the loss is finite at every step (catches NaN/inf blowups),
  * the loss actually goes DOWN (catches the silent zero-gradient class of bug,
    e.g. embedding's CoSENT-at-batch-size-1, or gradients that never sync),
  * parameters actually CHANGE (catches an optimizer stepping on the wrong
    tensors -- the SimpleFSDP shard/full-tensor confusion),
  * a submodule parameter read directly from a parent's forward has the full
    logical shape (the `.weight` regression that broke DiffusionGemma).

Runs at world_size=1, which is what makes it a unit test. That still exercises
each strategy's wrap/forward/backward/step plumbing; what it does NOT cover is
cross-rank gradient agreement, which needs a real multi-rank run.

Requires: RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT set and one visible GPU.
Run it with, for example:

    docker run --rm --device=/dev/kfd --device=/dev/dri \\
      --security-opt seccomp=unconfined \\
      -e ROCR_VISIBLE_DEVICES=0 -e RANK=0 -e WORLD_SIZE=1 \\
      -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 -e LOCAL_RANK=0 \\
      <image> sh -c 'cd /app/cray && python -m pytest test/unit/test_distribution_strategies_smoke.py'

(On NVIDIA use `--gpus all` and CUDA_VISIBLE_DEVICES instead.)

IMPORTANT -- do NOT run these under `pytest --forked`. The repo's own
`cmd/test_command.sh` passes `--forked`, which runs each test in a forked
subprocess; a ROCm/CUDA context and an initialised torch.distributed do not
survive fork, so every GPU test here fails with an unrelated error. Run this
file (and test_fsdp_direct_weight_access.py / test_fsdp_resume.py) without
`--forked`.
"""

import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="distribution strategies require a GPU"
)


@pytest.fixture(autouse=True, scope="module")
def _job_config(tmp_path_factory):
    """FSDP2 (`pytorch_fsdp`) reads the job config for hsdp_shard_size /
    reshard_after_forward, so a config file must exist or it raises
    `CRAY_TRAINING_JOB_CONFIG_PATH not set`. DDP and SimpleFSDP do not need it.
    """
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
    """Small net whose forward ALSO reads a submodule's .weight directly.

    The direct read is deliberate: it is the DiffusionGemma self-conditioning
    pattern (`torch.matmul(probs, self.embed_tokens.weight)`) that SimpleFSDP
    used to break on, so every strategy is held to it.
    """

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, probs):
        hidden = self.proj(self.embed(idx))
        logits = self.head(hidden)
        # Direct submodule-parameter access from the PARENT forward.
        soft = torch.matmul(probs, self.embed.weight)
        return logits, soft


def _build(strategy_name):
    """Wrap _TinyNet in the requested strategy, exactly as load_model would."""
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
    """Real optimizer steps; returns the per-step losses."""
    optimizer = torch.optim.AdamW(
        [p for p in wrapped.parameters() if p.requires_grad], lr=5e-2
    )
    idx, probs, target = _batch()
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, soft = wrapped(idx, probs)
        # `soft` is included so the direct-.weight path is part of the graph.
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
    """`self.embed.weight` read from the parent forward must not be a shard."""
    wrapped = _build(strategy)
    idx, probs, _ = _batch()

    _, soft = wrapped(idx, probs)

    # probs is (n, vocab); embed.weight is (vocab, dim) -> soft must be (n, dim).
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
    """Loss must go DOWN -- the check that separates training from a no-op."""
    losses = _train_a_few_steps(_build(strategy))

    assert losses[-1] < losses[0], (
        f"{strategy}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}); "
        "this is the silent 'COMPLETED but learned nothing' failure mode"
    )


def _snapshot(wrapped):
    """Local values of every trainable parameter.

    FSDP2 parameters are DTensors; comparing those directly is not meaningful,
    so take the rank-local shard (`to_local`) when present.
    """
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
