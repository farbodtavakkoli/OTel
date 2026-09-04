"""FSDP gradient-mean correctness on the PADDED shard path, with real metadata.

`test_gradient_semantics.py` covers the gradient-mean fix only for a tensor whose
element count divides evenly across ranks, using a hand-built metadata dict. That
is the easy half. Real models are full of parameters whose numel is not a multiple
of world size, and for those `shard_tensor()` pads up to the next multiple and
records a non-zero `padding` in the metadata -- a different code path through
`collectives_reduce_scatter()`, including a trim on the last rank.

These tests drive that path with metadata produced by the real `shard_tensor()`,
so a divergence between the hand-built dict and the genuine one cannot hide a bug.

Run::

    torchrun --nnodes=1 --nproc-per-node=8 \\
        -m pytest test/infra/distribution_strategy/test_fsdp_padding_mean.py -xvs
"""

import pytest
import torch

from cray_infra.training.distributed import get_rank, get_size


def _reduce_scatter_with_real_metadata(numel):
    """Run the FSDP gradient reduce-scatter over a `numel`-element gradient.

    Metadata comes from the real `shard_tensor()`, so padding/shard_size are
    whatever production would compute. Rank r contributes (r+1) everywhere, so a
    correct MEAN is (world+1)/2 and a SUM would be world*(world+1)/2.
    """
    from cray_megatron.megatron.distribution.fsdp import (
        collectives_reduce_scatter,
        cuda_device,
        shard_tensor,
    )

    rank, world = get_rank(), get_size()
    device = cuda_device()

    # Real metadata for a parameter of this size.
    param = torch.zeros(numel, dtype=torch.float32, device=device)
    _, metadata = shard_tensor(param)

    original_numel, _, shard_size, padding = metadata[rank]

    grad = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    out = collectives_reduce_scatter(grad, metadata)

    return out, original_numel, shard_size, padding, world


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_padding_is_actually_exercised(distributed_arch):
    """Guard the guard: the chosen size must really produce padding > 0."""
    from cray_megatron.megatron.distribution.fsdp import cuda_device, shard_tensor

    world = get_size()
    numel = world * 3 + 1  # deliberately not divisible by world size
    param = torch.zeros(numel, dtype=torch.float32, device=cuda_device())
    _, metadata = shard_tensor(param)
    padding = metadata[get_rank()][3]

    assert padding > 0, (
        f"test is vacuous: numel={numel} on world={world} produced padding=0, "
        f"so the padded path is not being exercised"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_reduce_scatter_mean_with_padding(distributed_arch):
    """Every VALID element of the shard must carry the mean, not the sum.

    Padding elements are contributed as zeros by every rank, so they reduce to 0
    and stay 0 after the division; they are excluded from the check rather than
    asserted to be the mean.
    """
    rank, world = get_rank(), get_size()
    numel = world * 3 + 1

    out, original_numel, shard_size, padding, world = _reduce_scatter_with_real_metadata(numel)

    mean_expected = (world + 1) / 2.0
    sum_expected = world * (world + 1) / 2.0

    # Which elements of THIS rank's shard correspond to real parameter data?
    shard_start = rank * shard_size
    valid_in_shard = max(0, min(shard_size, original_numel - shard_start))
    valid_in_shard = min(valid_in_shard, out.numel())

    assert valid_in_shard > 0 or rank * shard_size >= original_numel, (
        f"rank {rank}: computed 0 valid elements for a shard that should hold data"
    )

    if valid_in_shard:
        got = out[:valid_in_shard]
        assert torch.allclose(
            got, torch.full_like(got, mean_expected), atol=1e-4
        ), (
            f"rank {rank} (numel={original_numel}, shard_size={shard_size}, "
            f"padding={padding}): expected valid elements to be the mean "
            f"{mean_expected}, got {got.tolist()}. "
            f"({sum_expected} would mean the world_size division is missing on "
            f"the padded path.)"
        )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_padded_and_unpadded_agree_on_scale(distributed_arch):
    """The padded path must not scale differently from the unpadded one.

    A world_size division applied in only one of the two branches would be an
    easy mistake to make and impossible to see in a loss curve.
    """
    world = get_size()

    padded_out, _, _, padding, _ = _reduce_scatter_with_real_metadata(world * 3 + 1)
    even_out, _, _, no_padding, _ = _reduce_scatter_with_real_metadata(world * 4)

    assert padding > 0 and no_padding == 0, (
        f"expected one padded and one unpadded case, got padding={padding} / {no_padding}"
    )

    # Compare the first element, which is real data on every rank in both cases.
    assert abs(padded_out[0].item() - even_out[0].item()) < 1e-4, (
        f"padded path scaled to {padded_out[0].item()} but unpadded path scaled "
        f"to {even_out[0].item()} -- the two branches disagree on gradient scale"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_shard_shape_matches_parameter_shard(distributed_arch):
    """The returned gradient must match the shape of the shard it updates.

    `shard_tensor()` gives every rank exactly `shard_size` elements (the last
    rank's shard includes the padding), so the gradient must be `shard_size` too
    or the optimizer step would fail on a shape mismatch.
    """
    from cray_megatron.megatron.distribution.fsdp import cuda_device, shard_tensor

    rank, world = get_rank(), get_size()
    numel = world * 3 + 1

    param = torch.zeros(numel, dtype=torch.float32, device=cuda_device())
    shard, metadata = shard_tensor(param)
    out, _, shard_size, _, _ = _reduce_scatter_with_real_metadata(numel)

    assert out.numel() == shard.numel(), (
        f"rank {rank}: gradient shard has {out.numel()} elements but the parameter "
        f"shard has {shard.numel()} (shard_size={shard_size}) -- the optimizer step "
        f"would fail on shape mismatch"
    )
