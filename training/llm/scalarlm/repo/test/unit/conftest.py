import os

import pytest


@pytest.fixture(autouse=True)
def clean_scalarlm_env(monkeypatch):
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
