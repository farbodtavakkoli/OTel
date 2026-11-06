import ast
import logging
from pathlib import Path

import pytest
import torch

from cray_megatron.models.training_precision import prepare_training_precision

ROOT = Path(__file__).resolve().parents[2] / "ml/cray_megatron"
ROOT_CASES = [("repo", ROOT), ("mirror", ROOT.parents[2] / "ml/cray_megatron")]
ROOTS = [pytest.param(root, id=name) for name, root in ROOT_CASES if root.is_dir()]


def _get_optimizer(root, model, config):
    path = root / "megatron/training_loop.py"
    tree = ast.parse(path.read_text())
    env = {
        "get_job_config": lambda: config,
        "logger": logging.getLogger(__name__),
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_optimizer"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["get_optimizer"](model)


@pytest.mark.parametrize("root", ROOTS)
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("optimizer_type,expected", [
    (None, torch.optim.AdamW),
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
    ("rmsprop", torch.optim.RMSprop),
    ("unknown", torch.optim.AdamW),
])
def test_optimizer_dispatch_preserves_trainable_only(root, storage_dtype, optimizer_type, expected):
    model = torch.nn.Linear(2, 2, dtype=storage_dtype)
    model.bias.requires_grad_(False)
    config = {"learning_rate": 0.01}
    if optimizer_type is not None:
        config["optimizer_type"] = optimizer_type
    optimizer = _get_optimizer(root, model, config)
    assert type(optimizer) is expected
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert len(optimizer.param_groups[0]["params"]) == 1
    assert optimizer.param_groups[0]["params"][0] is model.weight


@pytest.mark.parametrize("optimizer_type", ["adamw", "sgd", "rmsprop", "unknown"])
def test_job_config_preserves_explicit_optimizer(optimizer_type):
    from cray_infra.util.default_job_config import JobConfig
    config = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", optimizer_type=optimizer_type)
    assert config.optimizer_type == optimizer_type
    assert config.training_storage_dtype == "bfloat16"


def test_bf16_storage_preserves_frozen_parameters():
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.bias.requires_grad_(False)
    bias = model.bias.detach().clone()
    assert prepare_training_precision(model, "bfloat16") == torch.bfloat16
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.bias, bias)


@pytest.mark.parametrize(
    "base_dtype,storage_dtype,expected_compute,expected_storage",
    [
        (torch.bfloat16, "bfloat16", torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, "float32", torch.bfloat16, torch.float32),
        (torch.float32, "bfloat16", None, torch.float32),
    ],
)
def test_peft_precision_with_promoted_adapters(
    base_dtype, storage_dtype, expected_compute, expected_storage
):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )).to(dtype=base_dtype)
    model = get_peft_model(base, LoraConfig(
        r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [(p, p.detach().clone()) for p in model.parameters() if not p.requires_grad]
    assert trainable and frozen
    assert {p.dtype for p in trainable} == {torch.float32}
    assert {p.dtype for p, _ in frozen} == {base_dtype}
    assert prepare_training_precision(model, storage_dtype) == expected_compute
    assert model._training_compute_dtype == expected_compute
    assert {p.dtype for p in trainable} == {expected_storage}
    optimizer = _get_optimizer(ROOT, model, {"learning_rate": 1e-2, "optimizer_type": "adamw"})
    assert type(optimizer) is torch.optim.AdamW
    before = [p.detach().clone() for p in trainable]
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=expected_compute is not None):
        loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and p.grad.dtype == expected_storage for p in trainable)
    optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(trainable, before))
    for param, old in frozen:
        assert param.dtype == base_dtype
        assert param.grad is None
        assert torch.equal(param, old)


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
def test_default_language_precision_is_adamw_bf16(mode):
    from cray_infra.util.default_job_config import JobConfig
    config = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode=mode)
    assert config.training_storage_dtype == "bfloat16"
    assert config.optimizer_type == "adamw"
    explicit = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode=mode, training_storage_dtype="float32")
    assert explicit.training_storage_dtype == "float32" and explicit.optimizer_type == "adamw"


def test_non_language_precision_defaults_remain_unchanged():
    from cray_infra.util.default_job_config import JobConfig
    config = JobConfig(job_directory="job", training_data_path="data", dataset_hash="hash", training_mode="embedding")
    assert config.training_storage_dtype == "float32" and config.optimizer_type == "adamw"


@pytest.mark.parametrize("root", ROOTS)
def test_grpo_and_sft_precision_are_wired(root):
    source = (root / "models/load_model.py").read_text()
    assert 'in ("language_model", "grpo")' in source
    assert 'prepare_training_precision(model_info["model"], storage_dtype)' in source
    assert 'if job_config.get("training_mode") == "language_model" and job_config.get("training_storage_dtype") == "bfloat16":' in source
    assert 'model_info["model"].loss_function = chunked_causal_loss' in source
