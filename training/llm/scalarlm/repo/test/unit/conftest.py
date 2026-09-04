"""
Unit-layer fixtures.

The autouse `clean_scalarlm_env` fixture is the critical piece — every SDK
URL test and every config-override test depends on env state being pristine.
Without it, one test setting `SCALARLM_API_URL` would bleed into the next.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_scalarlm_env(monkeypatch):
    """
    Wipe every SCALARLM_*, MASINT_*, and HF token env var before each unit
    test. Reset `scalarlm.api_url` / `masint.api_url` module attributes too.

    Scope: unit tests only. Component and E2E tests rely on env set by the
    container / docker-compose, so this fixture deliberately does not live in
    the root conftest.
    """
    for var in list(os.environ):
        if var.startswith(("SCALARLM_", "MASINT_")):
            monkeypatch.delenv(var, raising=False)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    try:
        import scalarlm
        monkeypatch.setattr(scalarlm, "api_url", None, raising=False)
    except ImportError:
        pass

    try:
        import masint
        monkeypatch.setattr(masint, "api_url", None, raising=False)
    except ImportError:
        pass
