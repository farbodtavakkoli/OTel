"""
Unit tests for cray_infra.huggingface.get_hf_token.

Contract under test: docs/configuration.md §3 — four-way HF token
resolution (HF_TOKEN env → config.hf_token plaintext → Fernet-decrypted
fallback).
"""

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
    # No SCALARLM_HF_TOKEN → plaintext is empty → encrypted fallback would
    # normally kick in; HF_TOKEN short-circuits it.

    assert get_hf_token() == "from-env"


def test_hf_token_config_plaintext_beats_encrypted(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_HF_TOKEN", "plaintext-token")

    assert get_hf_token() == "plaintext-token"


def test_hf_token_encrypted_fallback_decrypts_cleanly(tmp_path, monkeypatch):
    # With no HF_TOKEN and no SCALARLM_HF_TOKEN, the loader must Fernet-
    # decrypt the baked-in `hf_encrypted_token` and return a non-empty string.
    # We don't assert on the token value itself — keeping secrets out of test
    # code — but the fact that the round-trip works is the contract.
    _empty_yaml_path(tmp_path, monkeypatch)

    token = get_hf_token()

    assert isinstance(token, str)
    assert len(token) > 0


def test_hf_token_empty_string_config_falls_through_to_encrypted(tmp_path, monkeypatch):
    # config["hf_token"] == "" triggers the encrypted-fallback branch.
    # The default in default_config.py is "" so we just need env clean.
    _empty_yaml_path(tmp_path, monkeypatch)
    # Explicitly assert no override:
    monkeypatch.delenv("SCALARLM_HF_TOKEN", raising=False)

    token = get_hf_token()

    assert isinstance(token, str)
    assert len(token) > 0
