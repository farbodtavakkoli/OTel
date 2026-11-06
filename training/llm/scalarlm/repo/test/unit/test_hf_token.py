import pytest

from cray_infra.huggingface.get_hf_token import get_hf_token


def _empty_yaml_path(tmp_path, monkeypatch):
    path = tmp_path / "does-not-exist.yaml"
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(path))
    return path


def test_hf_token_env_wins_over_config_plaintext(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setenv("SCALARLM_HF_TOKEN", "from-config")

    assert get_hf_token() == "from-env"


def test_hf_token_env_wins_over_encrypted_fallback(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "from-env")

    assert get_hf_token() == "from-env"


def test_hf_token_config_plaintext_beats_encrypted(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_HF_TOKEN", "plaintext-token")

    assert get_hf_token() == "plaintext-token"


def test_hf_token_encrypted_fallback_decrypts_cleanly(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)

    token = get_hf_token()

    assert isinstance(token, str)
    assert len(token) > 0


def test_hf_token_empty_string_config_falls_through_to_encrypted(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.delenv("SCALARLM_HF_TOKEN", raising=False)

    token = get_hf_token()

    assert isinstance(token, str)
    assert len(token) > 0
