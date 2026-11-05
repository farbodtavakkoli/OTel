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


@pytest.mark.parametrize("length", [127, 128, 129, 259])
def test_chunked_causal_loss_matches_masked_shifted_cross_entropy(length):
    from cray_megatron.megatron.chunked_loss import chunked_causal_loss
    torch.manual_seed(7)
    logits = torch.randn(2, length, 31, requires_grad=True)
    ref = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(31, (2, length))
    labels[:, ::3] = -100
    actual = chunked_causal_loss(logits, labels, 31)
    expected = torch.nn.functional.cross_entropy(ref[:, :-1].reshape(-1, 31), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, ref.grad)
    torch.testing.assert_close(chunked_causal_loss(logits.detach(), labels, 31), expected.detach())


def test_chunked_causal_loss_no_targets_is_finite_zero():
    from cray_megatron.megatron.chunked_loss import chunked_causal_loss
    logits = torch.randn(1, 129, 31, requires_grad=True)
    loss = chunked_causal_loss(logits, torch.full((1, 129), -100), 31)
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("mode", ["language_model", "grpo"])
def test_default_language_precision_is_compensated_bf16(mode):
    from cray_infra.util.default_job_config import JobConfig
    config = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode=mode)
    assert config.training_storage_dtype == "bfloat16"
    assert config.optimizer_type == "compensated_adamw"
    explicit = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode=mode, training_storage_dtype="float32")
    assert explicit.training_storage_dtype == "float32" and explicit.optimizer_type == "adamw"


def test_non_language_precision_defaults_remain_unchanged():
    from cray_infra.util.default_job_config import JobConfig
    config = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode="embedding")
    assert config.training_storage_dtype == "float32" and config.optimizer_type == "adamw"


def test_grpo_and_sft_precision_are_wired():
    source = (ROOT / "models/load_model.py").read_text()
    assert 'in ("language_model", "grpo")' in source
    assert 'prepare_training_precision(model_info["model"], storage_dtype)' in source
    assert 'BF16 training storage requires compensated_adamw' in source
