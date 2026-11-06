import pytest
import torch

from cray_infra.training.distributed import get_rank, get_size


def _reduce_scatter_with_real_metadata(numel):
    from cray_megatron.megatron.distribution.fsdp import (
        collectives_reduce_scatter,
        cuda_device,
        shard_tensor,
    )

    rank, world = get_rank(), get_size()
    device = cuda_device()

    param = torch.zeros(numel, dtype=torch.float32, device=device)
    _, metadata = shard_tensor(param)

    original_numel, _, shard_size, padding = metadata[rank]

    grad = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    out = collectives_reduce_scatter(grad, metadata)

    return out, original_numel, shard_size, padding, world


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_padding_is_actually_exercised(distributed_arch):
    from cray_megatron.megatron.distribution.fsdp import cuda_device, shard_tensor

    world = get_size()
    numel = world * 3 + 1
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
    rank, world = get_rank(), get_size()
    numel = world * 3 + 1

    out, original_numel, shard_size, padding, world = _reduce_scatter_with_real_metadata(numel)

    mean_expected = (world + 1) / 2.0
    sum_expected = world * (world + 1) / 2.0

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
    world = get_size()

    padded_out, _, _, padding, _ = _reduce_scatter_with_real_metadata(world * 3 + 1)
    even_out, _, _, no_padding, _ = _reduce_scatter_with_real_metadata(world * 4)

    assert padding > 0 and no_padding == 0, (
        f"expected one padded and one unpadded case, got padding={padding} / {no_padding}"
    )

    assert abs(padded_out[0].item() - even_out[0].item()) < 1e-4, (
        f"padded path scaled to {padded_out[0].item()} but unpadded path scaled "
        f"to {even_out[0].item()} -- the two branches disagree on gradient scale"
    )


@pytest.mark.torchrun(nproc=4, arch="rocm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA device is required")
def test_shard_shape_matches_parameter_shard(distributed_arch):
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
