"""
Unit tests for render_chat_template.

Contract (see docs/openai-chat-completions-queue.md §4):
- Exactly one of `messages` or `prompt` must be set.
- `prompt` is a raw passthrough (returned unchanged).
- `messages` runs through the model's tokenizer chat template.
- Tokenizer instances are cached per-model — repeated calls on the
  same model don't reload.
"""

from unittest.mock import MagicMock, patch

import pytest

from cray_infra.api.fastapi.chat_completions import render_chat_template as rct


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache():
    """Each test gets a fresh cache so cross-test ordering doesn't matter."""
    rct._tokenizer_cache.clear()
    rct._source_cache.clear()
    yield
    rct._tokenizer_cache.clear()
    rct._source_cache.clear()


def _fake_tokenizer(rendered: str = "[fake-rendered]") -> MagicMock:
    tok = MagicMock()
    tok.apply_chat_template.return_value = rendered
    return tok


def test_prompt_is_passthrough_unchanged():
    out = rct.render_chat_template(model="any-model", messages=None, prompt="hello")
    assert out == "hello"


def test_messages_renders_via_tokenizer():
    fake = _fake_tokenizer("USER: hi\nASSISTANT: ")
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        out = rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
        )
    assert out == "USER: hi\nASSISTANT: "
    fake.apply_chat_template.assert_called_once()
    kwargs = fake.apply_chat_template.call_args.kwargs
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True


def test_safe_chat_template_kwargs_forwarded_to_tokenizer():
    fake = _fake_tokenizer()
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
            chat_template_kwargs={
                "enable_thinking": False,
                "reasoning_strength": "low",
            },
        )

    kwargs = fake.apply_chat_template.call_args.kwargs
    assert kwargs["enable_thinking"] is False
    assert kwargs["reasoning_strength"] == "low"


def test_tools_are_forwarded_for_admission_render():
    fake = _fake_tokenizer()
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
            tools=tools,
        )

    assert fake.apply_chat_template.call_args.kwargs["tools"] == tools


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [("none", False), ("low", True), ("high", True)],
)
def test_reasoning_effort_derives_enable_thinking(reasoning_effort, expected):
    fake = _fake_tokenizer()
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
            reasoning_effort=reasoning_effort,
        )

    kwargs = fake.apply_chat_template.call_args.kwargs
    assert kwargs["reasoning_effort"] == reasoning_effort
    assert kwargs["enable_thinking"] is expected


def test_explicit_enable_thinking_overrides_reasoning_effort():
    fake = _fake_tokenizer()
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
            chat_template_kwargs={"enable_thinking": False},
            reasoning_effort="high",
        )

    assert fake.apply_chat_template.call_args.kwargs["enable_thinking"] is False


def test_unrecognized_chat_template_kwargs_are_not_forwarded():
    fake = _fake_tokenizer()
    with patch.object(rct, "_load_tokenizer", return_value=fake):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
            chat_template_kwargs={
                "enable_thinking": False,
                "chat_template": "{{ untrusted }}",
                "tokenize": True,
            },
        )

    kwargs = fake.apply_chat_template.call_args.kwargs
    assert kwargs["enable_thinking"] is False
    assert "chat_template" not in kwargs
    assert kwargs["tokenize"] is False


def test_both_set_raises_value_error():
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(
            model="any-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt="also a prompt",
        )


def test_neither_set_raises_value_error():
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=None, prompt=None)


def test_empty_messages_list_treated_as_unset():
    """
    `messages=[]` is meaningless — no turns to render. Treat the same
    as unset so the caller learns about the bug from the ValueError
    rather than an opaque tokenizer failure.
    """
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=[], prompt=None)


def test_empty_prompt_treated_as_unset():
    """Same reasoning as the empty-messages case — empty string is suspicious."""
    with pytest.raises(ValueError, match="exactly one"):
        rct.render_chat_template(model="any-model", messages=None, prompt="")


def test_tokenizer_cached_per_model():
    """
    Repeated calls on the same model name must reuse the same tokenizer
    instance. Otherwise we'd reload the tokenizer (a multi-MB-disk-read,
    seconds-long operation) on every chat completion.
    """
    fake = _fake_tokenizer()
    with patch.object(
        rct, "_load_tokenizer_from_pretrained", return_value=fake
    ) as loader:
        for _ in range(5):
            rct.render_chat_template(
                model="repeat-model",
                messages=[{"role": "user", "content": "x"}],
                prompt=None,
            )
        assert loader.call_count == 1


def test_tokenizer_loaded_per_distinct_model():
    """Distinct model names get distinct tokenizers."""
    fake_a = _fake_tokenizer("A")
    fake_b = _fake_tokenizer("B")
    sequence = iter([fake_a, fake_b])
    with patch.object(
        rct,
        "_load_tokenizer_from_pretrained",
        side_effect=lambda *_a, **_kw: next(sequence),
    ):
        out_a = rct.render_chat_template(
            model="model-a",
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )
        out_b = rct.render_chat_template(
            model="model-b",
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )
    assert out_a == "A"
    assert out_b == "B"


def test_lora_adapter_dir_falls_back_to_base_tokenizer(tmp_path, monkeypatch):
    """
    A trained-model hash that names a LoRA adapter directory (only `.pt`
    files, no tokenizer config) must load the base model's tokenizer
    instead of being handed verbatim to AutoTokenizer — otherwise
    transformers treats the hash as an HF repo id and 401s against the
    Hub. Regression test for the production trace where
    `dcb1a490...c5b3bea24411c` resolved that way.
    """
    base_model = "base/model"
    adapter_hash = "dcb1a490f3a3a055469f7815cb59f5a4a8888adac6d2d320337c5b3bea24411c"
    adapter_dir = tmp_path / adapter_hash
    adapter_dir.mkdir()
    (adapter_dir / "checkpoint.pt").write_bytes(b"")

    def fake_get_config():
        return {"model": base_model, "training_job_directory": str(tmp_path)}

    from cray_infra.util import get_config as get_config_mod

    monkeypatch.setattr(get_config_mod, "get_config", fake_get_config)

    seen_sources: list[str] = []

    def fake_loader(source):
        seen_sources.append(source)
        return _fake_tokenizer()

    with patch.object(rct, "_load_tokenizer_from_pretrained", side_effect=fake_loader):
        rct.render_chat_template(
            model=adapter_hash,
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )

    assert seen_sources == [base_model]


def test_adapter_dir_with_tokenizer_files_uses_local_path(tmp_path, monkeypatch):
    """
    Forward-compat: if a future training job saves a full checkpoint
    (its own tokenizer.json next to the weights), prefer that local
    path over the base model — the local tokenizer is authoritative.
    """
    base_model = "base/model"
    adapter_hash = "full-checkpoint-hash"
    adapter_dir = tmp_path / adapter_hash
    adapter_dir.mkdir()
    (adapter_dir / "checkpoint.pt").write_bytes(b"")
    (adapter_dir / "tokenizer.json").write_text("{}")

    def fake_get_config():
        return {"model": base_model, "training_job_directory": str(tmp_path)}

    from cray_infra.util import get_config as get_config_mod

    monkeypatch.setattr(get_config_mod, "get_config", fake_get_config)

    seen_sources: list[str] = []

    def fake_loader(source):
        seen_sources.append(source)
        return _fake_tokenizer()

    with patch.object(rct, "_load_tokenizer_from_pretrained", side_effect=fake_loader):
        rct.render_chat_template(
            model=adapter_hash,
            messages=[{"role": "user", "content": "x"}],
            prompt=None,
        )

    assert seen_sources == [str(adapter_dir)]
