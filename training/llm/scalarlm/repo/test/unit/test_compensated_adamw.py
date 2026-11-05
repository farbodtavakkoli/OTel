import io

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cray_megatron.megatron.compensated_adamw import CompensatedAdamW


def effective_parameter(param, optimizer):
    return param.detach().float() + optimizer.state[param]["residual"].float()


@pytest.mark.parametrize("steps", [5, 100])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_effective_parameter_tracks_fp32_adamw(steps, weight_decay):
    torch.manual_seed(41)
    initial = torch.linspace(-1, 1, 127).bfloat16()
    param = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.float())
    optimizer = CompensatedAdamW([param], lr=1e-5, weight_decay=weight_decay, chunk_size=17)
    reference_optimizer = torch.optim.AdamW([reference], lr=1e-5, weight_decay=weight_decay)
    quantized_reference = initial.float().clone()
    moment1 = torch.zeros_like(initial)
    moment2 = torch.zeros_like(initial)
    for index in range(steps):
        grad = (torch.sin(torch.arange(127).float() * 0.1 + index * 0.01) + 0.2).bfloat16()
        param.grad = grad.clone()
        reference.grad = grad.float()
        optimizer.step()
        reference_optimizer.step()
        next_moment1 = 0.9 * moment1.float() + 0.1 * grad.float()
        next_moment2 = 0.999 * moment2.float() + 0.001 * grad.float().square()
        update = (next_moment1 / (1 - 0.9 ** (index + 1))) / (
            (next_moment2 / (1 - 0.999 ** (index + 1))).sqrt() + 1e-8
        )
        quantized_reference = quantized_reference * (1 - 1e-5 * weight_decay) - 1e-5 * update
        rounded_param = quantized_reference.bfloat16()
        rounded_residual = (quantized_reference - rounded_param.float()).bfloat16()
        quantized_reference = rounded_param.float() + rounded_residual.float()
        moment1 = next_moment1.bfloat16()
        moment2 = next_moment2.bfloat16()
    torch.testing.assert_close(effective_parameter(param, optimizer), quantized_reference, rtol=0, atol=5e-6)
    error = (effective_parameter(param, optimizer) - reference.detach()).abs().max().item()
    print(f"steps={steps} decay={weight_decay} max_effective_error={error:.9g}")
    assert error < (1e-6 if steps == 5 else 1.2e-4)
    assert torch.isfinite(param).all()
    for name in ("exp_avg", "exp_avg_sq", "residual"):
        assert optimizer.state[param][name].dtype == torch.bfloat16
        assert torch.isfinite(optimizer.state[param][name]).all()
    assert optimizer.state[param]["step"].dtype == torch.float32
    assert set(optimizer.state[param]) == {"step", "exp_avg", "exp_avg_sq", "residual"}


def test_small_updates_eventually_change_bf16_weights():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = CompensatedAdamW([param], lr=1e-5, weight_decay=0, chunk_size=2)
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert torch.equal(param, torch.ones_like(param))
    assert torch.all(effective_parameter(param, optimizer) < 1)
    for _ in range(999):
        optimizer.step()
    assert torch.all(param < 1)


@pytest.mark.parametrize("maximize", [False, True])
def test_first_step_bias_correction_and_decay(maximize):
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16))
    grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    initial = param.detach().float().clone()
    optimizer = CompensatedAdamW(
        [param], lr=0.01, betas=(0.8, 0.95), eps=0.01, weight_decay=0.2, maximize=maximize
    )
    param.grad = grad
    optimizer.step()
    signed_grad = grad.float() * (-1 if maximize else 1)
    expected = initial * (1 - 0.01 * 0.2) - 0.01 * signed_grad / (grad.float().abs() + 0.01)
    torch.testing.assert_close(effective_parameter(param, optimizer), expected, rtol=0, atol=2e-6)


def test_checkpoint_resume_preserves_residual_and_is_exact():
    param = torch.nn.Parameter(torch.tensor([1.0, 0.5, -1.0], dtype=torch.bfloat16))
    optimizer = CompensatedAdamW([param], lr=1e-5, chunk_size=2)
    for index in range(13):
        param.grad = torch.tensor([0.1 + index * 0.01, -0.2, 0.3], dtype=torch.bfloat16)
        optimizer.step()
    buffer = io.BytesIO()
    torch.save({"param": param.detach(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    resumed = torch.nn.Parameter(checkpoint["param"].clone())
    resumed_optimizer = CompensatedAdamW([resumed], lr=0.1, chunk_size=99)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    assert resumed_optimizer.param_groups[0]["chunk_size"] == 2
    for key, value in optimizer.state[param].items():
        assert torch.equal(value, resumed_optimizer.state[resumed][key])
        assert value.dtype == resumed_optimizer.state[resumed][key].dtype
    for index in range(87):
        grad = torch.tensor([0.2, 0.01 + index * 0.001, -0.3], dtype=torch.bfloat16)
        param.grad = grad.clone()
        resumed.grad = grad.clone()
        optimizer.step()
        resumed_optimizer.step()
    assert torch.equal(param, resumed)
    for key, value in optimizer.state[param].items():
        assert torch.equal(value, resumed_optimizer.state[resumed][key])


class FloatAllocationRecorder(TorchDispatchMode):
    def __init__(self):
        self.sizes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        for output in outputs:
            if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
                self.sizes.append(output.numel())
        return result


def test_fp32_temporaries_are_bounded_by_chunk_size():
    param = torch.nn.Parameter(torch.ones(10007, dtype=torch.bfloat16))
    param.grad = torch.ones_like(param)
    optimizer = CompensatedAdamW([param], chunk_size=31)
    recorder = FloatAllocationRecorder()
    with recorder:
        optimizer.step()
        optimizer.step()
    assert recorder.sizes
    assert max(recorder.sizes) <= 31
    for value in optimizer.state[param].values():
        assert value.dtype == torch.bfloat16 or value.numel() == 1


def test_noncontiguous_parameters_and_gradients_match_contiguous():
    initial = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5).t()
    param = torch.nn.Parameter(initial.clone(memory_format=torch.preserve_format))
    contiguous = torch.nn.Parameter(initial.contiguous())
    assert not param.is_contiguous()
    optimizer = CompensatedAdamW([param], chunk_size=4)
    reference_optimizer = CompensatedAdamW([contiguous], chunk_size=4)
    for _ in range(5):
        grad = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5).t() + 1
        param.grad = grad
        contiguous.grad = grad.contiguous()
        optimizer.step()
        reference_optimizer.step()
    assert torch.equal(param, contiguous)
    assert torch.equal(optimizer.state[param]["residual"], reference_optimizer.state[contiguous]["residual"])


def test_missing_gradient_does_not_create_state_and_closure_runs_with_grad():
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = CompensatedAdamW([param])
    assert optimizer.step() is None
    assert not optimizer.state

    def closure():
        optimizer.zero_grad()
        loss = param.float().square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 2
    assert optimizer.state[param]["step"].item() == 1


def test_sparse_gradients_are_rejected():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    param.grad = torch.sparse_coo_tensor([[1]], [0.5], (4,), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="sparse gradients"):
        CompensatedAdamW([param]).step()


def test_non_bf16_parameters_are_rejected():
    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones_like(param)
    with pytest.raises(TypeError, match="requires BF16"):
        CompensatedAdamW([param]).step()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1},
        {"lr": float("nan")},
        {"eps": -1},
        {"weight_decay": -1},
        {"betas": (1, 0.9)},
        {"betas": (0.9, -0.1)},
        {"chunk_size": 0},
        {"chunk_size": 1.5},
        {"chunk_size": True},
        {"params": [{"params": [torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))], "chunk_size": 0}]},
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs):
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    params = kwargs.get("params", [param])
    options = {key: value for key, value in kwargs.items() if key != "params"}
    with pytest.raises(ValueError):
        CompensatedAdamW(params, **options)


def test_dtensor_state_full_checkpoint_round_trip(tmp_path):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = torch.nn.Module()
        model.register_parameter(
            "weight",
            torch.nn.Parameter(distribute_tensor(torch.ones(9, dtype=torch.bfloat16), mesh, [Shard(0)])),
        )
        optimizer = CompensatedAdamW(model.parameters(), lr=1e-5, chunk_size=4)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        for name in ("exp_avg", "exp_avg_sq", "residual"):
            value = optimizer.state[model.weight][name]
            assert isinstance(value, DTensor)
            assert value.dtype == torch.bfloat16
            assert value.placements == model.weight.placements
        options = StateDictOptions(full_state_dict=True)
        checkpoint = get_optimizer_state_dict(model, optimizer, options=options)
        assert checkpoint["state"]["weight"]["residual"].shape == (9,)
        assert not isinstance(checkpoint["state"]["weight"]["residual"], DTensor)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        resumed_model = torch.nn.Module()
        resumed_model.register_parameter("weight", torch.nn.Parameter(model.weight.detach().clone()))
        resumed_optimizer = CompensatedAdamW(resumed_model.parameters(), lr=1e-5, chunk_size=4)
        set_optimizer_state_dict(resumed_model, resumed_optimizer, checkpoint, options=options)
        resumed_model.weight.grad = torch.ones_like(resumed_model.weight)
        for _ in range(7):
            optimizer.step()
            resumed_optimizer.step()
        assert torch.equal(model.weight.to_local(), resumed_model.weight.to_local())
        for key, value in optimizer.state[model.weight].items():
            resumed_value = resumed_optimizer.state[resumed_model.weight][key]
            if isinstance(value, DTensor):
                assert isinstance(resumed_value, DTensor)
                assert torch.equal(value.to_local(), resumed_value.to_local())
            else:
                assert torch.equal(value, resumed_value)
    finally:
        torch.distributed.destroy_process_group()


def test_empty_shard_and_zero_gradient_are_supported():
    empty = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    empty.grad = torch.empty_like(empty)
    param.grad = torch.zeros_like(param)
    optimizer = CompensatedAdamW([empty, param], lr=1e-5, weight_decay=0.1)
    optimizer.step()
    assert optimizer.state[empty]["step"].item() == 1
    torch.testing.assert_close(effective_parameter(param, optimizer), torch.full((2,), 0.999999), rtol=0, atol=1e-7)
