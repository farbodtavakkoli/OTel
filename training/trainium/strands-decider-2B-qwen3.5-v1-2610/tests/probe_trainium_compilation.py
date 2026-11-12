"""Synthetic native-Neuron compiled update; not a full Decider model test."""

import json
import importlib.metadata as metadata
import torch
import torch_neuronx  # noqa: F401

torch.manual_seed(42)
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)
).to("neuron")
x = torch.randn(2, 16).to("neuron")
y = torch.tensor([0, 2]).to("neuron")
before = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


def loss_fn(x, y):
    with torch.autocast("neuron", dtype=torch.bfloat16):
        logits = model(x)
    return torch.nn.functional.cross_entropy(logits.float(), y)


compiled = torch.compile(loss_fn, backend="neuron", dynamic=False)
loss = compiled(x, y)
loss.backward()
if not (torch.isfinite(loss.cpu()).all()):
    raise ValueError("Validation failed: torch.isfinite(loss.cpu()).all()")
for name, parameter in model.named_parameters():
    if not (parameter.device.type == "neuron"):
        raise ValueError(name)
    if not (parameter.grad is not None):
        raise ValueError(name)
    if not (parameter.grad.device.type == "neuron"):
        raise ValueError(name)
    if not (torch.isfinite(parameter.grad.cpu()).all()):
        raise ValueError(name)
optimizer.step()
changed = [
    name
    for name, parameter in model.named_parameters()
    if not torch.equal(before[name], parameter.detach().cpu())
]
if not (len(changed) == len(before)):
    raise ValueError(changed)
print(
    json.dumps(
        {
            "scope": "synthetic_compiled_optimizer_update",
            "status": "passed",
            "loss": float(loss.detach().cpu()),
            "device": str(next(model.parameters()).device),
            "changed_parameters": changed,
            "versions": {
                p: metadata.version(p)
                for p in ["torch", "torch-neuronx", "transformers", "peft"]
            },
            "full_decider_training_verified": False,
        },
        indent=2,
    )
)
