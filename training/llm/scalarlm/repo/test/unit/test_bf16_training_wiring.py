import ast
from pathlib import Path

import pytest
import torch

from cray_megatron.models.training_precision import prepare_training_precision

ROOT = Path(__file__).resolve().parents[2] / "ml/cray_megatron"


def test_bf16_storage_preserves_frozen_parameters():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.bias.requires_grad_(False)
    bias = model.bias.detach().clone()
    assert prepare_training_precision(model, "bfloat16") == torch.bfloat16
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.bias, bias)


def test_invalid_storage_rejected():
    with pytest.raises(ValueError, match="storage dtype"):
        prepare_training_precision(torch.nn.Linear(2, 2), "float16")


def test_chunked_grpo_logprobs_match_full_values_and_gradients():
    path = ROOT / "megatron/training_loop.py"
    tree = ast.parse(path.read_text())
    env = {"torch": torch}
    names = {"_row_logprobs", "_selected_logprobs"}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(path), "exec"), env)
    torch.manual_seed(42)
    logits = torch.randn(2, 259, 31, requires_grad=True)
    ref = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(31, (2, 259))
    actual = env["_selected_logprobs"](logits, targets)
    expected = ref.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, ref.grad)
    torch.testing.assert_close(env["_selected_logprobs"](logits.detach(), targets), expected.detach())


def test_grpo_and_sft_precision_are_wired():
    source = (ROOT / "models/load_model.py").read_text()
    assert 'in ("language_model", "grpo")' in source
    assert 'prepare_training_precision(model_info["model"], storage_dtype)' in source
    assert 'BF16 training storage requires compensated_adamw' in source
