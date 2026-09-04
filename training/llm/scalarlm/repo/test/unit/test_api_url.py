"""
Unit tests for SDK URL resolution (make_api_url, get_api_base).

Contract under test: docs/configuration.md §4 — six-level precedence order
for the API base URL.
"""

import masint
import scalarlm
from masint.util.get_api_base import get_api_base
from masint.util.make_api_url import make_api_url


def test_make_api_url_joins_with_single_slash():
    assert (
        make_api_url("v1/health", api_url="http://host:8000")
        == "http://host:8000/v1/health"
    )


def test_make_api_url_falls_back_to_get_api_base():
    # With a pristine env (autouse fixture clears everything), the fallback
    # is the localhost default.
    assert make_api_url("v1/health") == "http://localhost:8000/v1/health"


def test_make_api_url_explicit_beats_all_globals(monkeypatch):
    monkeypatch.setenv("SCALARLM_API_URL", "http://env")
    monkeypatch.setattr(scalarlm, "api_url", "http://attr", raising=False)

    assert (
        make_api_url("v1/health", api_url="http://explicit:9")
        == "http://explicit:9/v1/health"
    )


def test_get_api_base_defaults_to_localhost():
    assert get_api_base() == "http://localhost:8000"


def test_get_api_base_scalarlm_module_attr_is_highest_precedence(monkeypatch):
    monkeypatch.setenv("SCALARLM_API_URL", "http://env:1")
    monkeypatch.setenv("MASINT_API_URL", "http://env:2")
    monkeypatch.setattr(masint, "api_url", "http://masint-attr:3", raising=False)
    monkeypatch.setattr(scalarlm, "api_url", "http://scalarlm-attr:4", raising=False)

    assert get_api_base() == "http://scalarlm-attr:4"


def test_get_api_base_masint_attr_beats_env(monkeypatch):
    monkeypatch.setenv("SCALARLM_API_URL", "http://env")
    monkeypatch.setattr(masint, "api_url", "http://masint-attr", raising=False)
    # scalarlm.api_url stays None via clean_scalarlm_env

    assert get_api_base() == "http://masint-attr"


def test_get_api_base_scalarlm_env_beats_masint_env(monkeypatch):
    monkeypatch.setenv("SCALARLM_API_URL", "http://scalarlm-env")
    monkeypatch.setenv("MASINT_API_URL", "http://masint-env")

    assert get_api_base() == "http://scalarlm-env"


def test_get_api_base_masint_env_when_scalarlm_env_absent(monkeypatch):
    monkeypatch.setenv("MASINT_API_URL", "http://masint-env")

    assert get_api_base() == "http://masint-env"


def test_get_api_base_module_attr_of_none_is_ignored(monkeypatch):
    # Setting the attribute to None must fall through, not return "None".
    monkeypatch.setenv("SCALARLM_API_URL", "http://env")
    monkeypatch.setattr(scalarlm, "api_url", None, raising=False)
    monkeypatch.setattr(masint, "api_url", None, raising=False)

    assert get_api_base() == "http://env"
