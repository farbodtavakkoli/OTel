import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from cray_infra.util.default_job_config import JobConfig
from cray_megatron.megatron.doc_mask import BUILD, SKIP_MULTIMODAL, doc_mask_decision
from cray_megatron.models.training_precision import prepare_training_precision

ROOT = Path(__file__).resolve().parents[2] / "ml/cray_megatron"


def functions(path, names, config):
    tree = ast.parse(path.read_text())
    env = {"torch": torch, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD,
           "RMSprop": torch.optim.RMSprop, "logger": logging.getLogger(__name__),
           "get_job_config": lambda: config}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(path), "exec"), env)
    return env


def dataset_functions(budget=4):
    return functions(ROOT / "megatron/dataset/load_language_model_dataset.py", {"get_pack_function", "get_max_position_embeddings", "add_eos_token"}, {"max_token_block_size": budget})


def fake_model(terminal=8):
    return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128, pad_token_id=0), generation_config=SimpleNamespace(eos_token_id=terminal))


@pytest.mark.parametrize("length", [0, 1, 3, 4])
def test_pack_preserves_short_and_exact_rows(length):
    data = {"input_ids": [list(range(length))], "labels": [list(range(length))], "attention_mask": [[1] * length]}
    result = dataset_functions()["get_pack_function"](fake_model())(data)
    if length == 0:
        assert result["input_ids"] == []
    else:
        assert result["input_ids"][0][:length] == list(range(length))
        assert result["labels"][0] == [-100] + list(range(1, length)) + [-100] * (4 - length)
        assert result["attention_mask"][0] == [1] * length + [0] * (4 - length)


def test_pack_document_boundaries_do_not_predict_next_document():
    data = {"input_ids": [[1, 2], [3, 4]], "labels": [[1, 2], [3, 4]], "attention_mask": [[1, 1], [1, 1]]}
    result = dataset_functions()["get_pack_function"](fake_model())(data)
    assert result["labels"] == [[-100, 2, -100, 4]]
    assert result["position_ids"] == [[0, 1, 0, 1]]
    assert result["document_ids"] == [[0, 0, 1, 1]]


@pytest.mark.parametrize("length", [5, 10])
def test_pack_overflow_fails_without_mutation(length):
    data = {"input_ids": [list(range(length))], "labels": [list(range(length))], "attention_mask": [[1] * length]}
    original = copy.deepcopy(data)
    with pytest.raises(ValueError, match=f"{length} tokens.*budget 4"):
        dataset_functions()["get_pack_function"](fake_model())(data)
    assert data == original


@pytest.mark.parametrize("key", ["labels", "attention_mask"])
def test_pack_rejects_misaligned_stream(key):
    data = {"input_ids": [[1, 2]], "labels": [[1, 2]], "attention_mask": [[1, 1]]}
    data[key][0].pop()
    with pytest.raises(ValueError, match=f"misaligned {key}"):
        dataset_functions()["get_pack_function"](fake_model())(data)


def test_pack_rejects_zero_budget():
    with pytest.raises(ValueError, match="positive"):
        dataset_functions(0)["get_pack_function"](fake_model())


@pytest.mark.parametrize("ids,terminal,expected", [([1, 8], [7, 8], [1, 8]), ([1], [7, 8], [1, 8]), ([], 8, [8]), ([1, 9], None, [1, 9])])
def test_eos_is_idempotent_and_aligned(ids, terminal, expected):
    tokens = {"input_ids": [ids[:]], "labels": [ids[:]], "attention_mask": [[1] * len(ids)]}
    env = dataset_functions()
    for _ in range(2):
        env["add_eos_token"](tokens, fake_model(terminal), SimpleNamespace(eos_token_id=9))
    assert tokens == {"input_ids": [expected], "labels": [expected], "attention_mask": [[1] * len(expected)]}


@pytest.mark.parametrize("kind", ["linear", "cosine"])
@pytest.mark.parametrize("warmup", [0, 1, 2, 5])
def test_schedule_endpoints(kind, warmup):
    env = functions(ROOT / "megatron/training_loop.py", {"get_scheduler", "get_warmup_steps"}, {"warmup_steps": warmup, "scheduler_type": kind})
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.)
    scheduler = env["get_scheduler"](optimizer, 6)
    curve = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        curve.append(optimizer.param_groups[0]["lr"])
    assert curve[0] == pytest.approx(.001 if warmup else 1.)
    assert curve[warmup] == pytest.approx(1.)
    assert curve[-1] == pytest.approx(0., abs=1e-12)
    if kind == "linear" and not warmup:
        assert curve == pytest.approx([1, 5/6, 4/6, 3/6, 2/6, 1/6, 0])


@pytest.mark.parametrize("extra", [{"warmup_steps": -1}, {"warmup_steps": 6}, {"scheduler_type": "bad"}, {"max_steps": 0}])
def test_invalid_schedule_config_rejected(extra):
    config = {"job_directory": "job", "training_data_path": "data", "dataset_hash": "hash", "max_steps": 6}
    config.update(extra)
    with pytest.raises(ValidationError):
        JobConfig(**config)


def test_fp32_master_small_update_and_optimizer_resume():
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.)
    assert prepare_training_precision(model) == torch.bfloat16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    def step(m, opt):
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = m(torch.ones(1, 2))
            assert output.dtype == torch.bfloat16
            loss = output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    step(model, optimizer)
    assert model.weight.dtype == torch.float32
    assert torch.all(model.weight < 1.)
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.float32
    resumed = torch.nn.Linear(2, 2, bias=False)
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
    resumed_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(resumed, resumed_opt)
    assert torch.equal(model.weight, resumed.weight)
    for key in optimizer.state[model.weight]:
        assert torch.equal(optimizer.state[model.weight][key], resumed_opt.state[resumed.weight][key])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_explicit_non_bf16_dtype_preserved(dtype):
    model = torch.nn.Linear(2, 2, dtype=dtype)
    assert prepare_training_precision(model) is None
    assert model.weight.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 requires CUDA")
def test_fsdp2_bf16_compute_fp32_optimizer_checkpoint_resume(monkeypatch, tmp_path):
    from cray_megatron.megatron.distribution import pytorch_fsdp
    from cray_infra.training.distributed import cuda_device
    monkeypatch.setattr(pytorch_fsdp, "get_job_config", lambda: {"adapter_type": "none", "reshard_after_forward": True})
    def make():
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.fill_(1.)
        prepare_training_precision(model)
        wrapped = pytorch_fsdp.PyTorchFSDP(model)
        return wrapped, torch.optim.AdamW(wrapped.parameters(), lr=1e-5)
    def step(model, optimizer):
        optimizer.zero_grad()
        output = model(torch.ones(1, 2, device=cuda_device()))
        assert output.dtype == torch.bfloat16
        loss = output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model, optimizer = make()
    step(model, optimizer)
    saved = {"model": model.unwrap_model(), "optimizer": model.unwrap_optimizer(optimizer)}
    assert saved["model"]["weight"].dtype == torch.float32
    assert torch.all(saved["model"]["weight"] < 1.)
    for state in saved["optimizer"]["state"].values():
        assert state["exp_avg"].dtype == torch.float32
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    resumed, resumed_optimizer = make()
    restored = torch.load(path, weights_only=True)
    resumed.load_unwrapped_model(restored["model"])
    resumed.load_unwrapped_optimizer(resumed_optimizer, restored["optimizer"])
    step(model, optimizer)
    step(resumed, resumed_optimizer)
    assert torch.equal(model.unwrap_model()["weight"], resumed.unwrap_model()["weight"])
    expected = model.unwrap_optimizer(optimizer)["state"]
    actual = resumed.unwrap_optimizer(resumed_optimizer)["state"]
    for name, state in expected.items():
        for key, value in state.items():
            assert torch.equal(value, actual[name][key])


def test_glimmer_multimodal_fallback():
    config = SimpleNamespace(model_type="muse_glimmer", vision_config=object())
    assert doc_mask_decision({"document_ids": [0, 1]}, 2, config, 4) == BUILD
    assert doc_mask_decision({"document_ids": [0, 1], "pixel_values": object()}, 2, config, 4) == SKIP_MULTIMODAL


@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
def test_full_glimmer_document_isolation_finite_loss(layer_type):
    from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig, MuseGlimmerVisionConfig
    text = MuseGlimmerTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, layer_types=[layer_type], layer_rope_theta=[10000.], sliding_window=2)
    vision = MuseGlimmerVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
    config = MuseGlimmerConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), image_token_id=31, projector_hidden_size=16, out_hidden_size=16)
    config._attn_implementation = "sdpa"
    model = MuseGlimmerForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    changed = torch.tensor([[5, 6, 3, 4, 7]])
    docs = torch.tensor([[0, 0, 1, 1, 2]])
    padding = torch.tensor([[1, 1, 1, 1, 0]])
    mask = ((docs[:, :, None] == docs[:, None, :]) & torch.ones(5, 5, dtype=torch.bool).tril() & padding[:, None, :].bool())[:, None]
    labels = ids.clone()
    labels[:, [0, 2, 4]] = -100
    positions = torch.tensor([[0, 1, 0, 1, 0]])
    with torch.no_grad():
        a = model(ids, attention_mask=mask, position_ids=positions, labels=labels, use_cache=False)
        b = model(changed, attention_mask=mask, position_ids=positions, labels=labels, use_cache=False)
    assert torch.isfinite(a.loss)
    assert torch.equal(a.logits[:, 2:4], b.logits[:, 2:4])
