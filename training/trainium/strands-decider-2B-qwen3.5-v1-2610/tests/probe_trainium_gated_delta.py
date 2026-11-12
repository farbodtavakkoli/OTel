"""Native GDN forward/backward diagnostic; not a full-model training test."""

import json
import traceback
import torch
import torch_neuronx
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

torch.manual_seed(7)
torch.set_num_threads(6)
shapes = [(1, 65, 2, 16)] * 3 + [(1, 65, 2)] * 2
inputs = [torch.randn(s) for s in shapes]
inputs[3] = -inputs[3].abs() * 0.1
inputs[4] = inputs[4].sigmoid()


def evaluate(fn, device):
    xs = [x.to(device).detach().requires_grad_() for x in inputs]
    out, _ = fn(*xs, use_qk_l2norm_in_kernel=True)
    out.square().mean().backward()
    if not (all((x.grad is not None and x.grad.device.type == device for x in xs))):
        raise ValueError(
            "Validation failed: all((x.grad is not None and x.grad.device.type == device for x in xs))"
        )
    return out.detach().cpu(), [x.grad.cpu() for x in xs]


try:
    reference, reference_grads = evaluate(torch_chunk_gated_delta_rule, "cpu")
    actual, grads = evaluate(torch_chunk_gated_delta_rule, "neuron")
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-3)
    for actual_grad, reference_grad in zip(grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-5, rtol=1e-3)
    print(
        json.dumps(
            {"status": "passed", "scope": "small_reference_gdn_eager_forward_backward"}
        )
    )
except Exception:
    print(json.dumps({"status": "failed", "traceback": traceback.format_exc()}))
    raise
