"""`upload_to_hf` must not be silently defeated by offline mode.

Deployments commonly set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 so server boot
cannot stall on tokenizer downloads (DOCKER_IMAGE_MI355.md recommends exactly
that). Those same flags make huggingface_hub refuse the checkpoint upload:

    Cannot reach https://huggingface.co/api/repos/create:
    offline mode is enabled ... unset HF_HUB_OFFLINE

and because the upload is deliberately best-effort (a finished training run must
not be reported FAILED over a network hiccup) the exception was swallowed. The
user got a green COMPLETED job, no upload, and the only evidence buried in the
rank-0 log.

A job that explicitly asked to upload has explicitly asked for network, so the
upload clears the offline flags for its own duration, restores them afterwards,
and records the outcome in status.json.

Targets `do_upload` directly: `upload_to_hf_if_enabled` only adds rank-0 and
config gating on top, and reaching that would require torch.distributed.
"""

import os

import pytest

# cray_megatron.hf_upload is deliberately light -- importing cray_megatron.main
# would initialise torch.distributed and hang a unit test.
from cray_megatron import hf_upload


OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


class _RecordingApi:
    """Stand-in for HfApi that records the offline env seen mid-upload."""

    seen_env = {}

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, **kwargs):
        _RecordingApi.seen_env = {v: os.environ.get(v) for v in OFFLINE_VARS}

    def upload_file(self, **kwargs):
        return "https://huggingface.co/fake/commit"


@pytest.fixture
def upload_env(monkeypatch, tmp_path):
    """Stub HfApi and capture what gets recorded into status.json."""
    import huggingface_hub

    _RecordingApi.seen_env = {}
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingApi)

    recorded = {}
    monkeypatch.setattr(
        hf_upload,
        "record_hf_upload_result",
        lambda repo_id=None, error=None: recorded.update(
            {"repo_id": repo_id, "error": error}
        ),
    )

    checkpoint = tmp_path / "checkpoint_1.pt"
    checkpoint.write_text("weights")
    recorded["_checkpoint"] = str(checkpoint)
    return recorded


def _run(recorded):
    hf_upload.do_upload("someone/some-repo", "hf_faketoken", recorded["_checkpoint"])


def test_offline_flags_are_cleared_during_upload(monkeypatch, upload_env):
    for var in OFFLINE_VARS:
        monkeypatch.setenv(var, "1")

    _run(upload_env)

    # Inside the upload, offline mode must be gone.
    assert _RecordingApi.seen_env == {v: None for v in OFFLINE_VARS}


def test_offline_flags_are_restored_after_upload(monkeypatch, upload_env):
    for var in OFFLINE_VARS:
        monkeypatch.setenv(var, "1")

    _run(upload_env)

    for var in OFFLINE_VARS:
        assert os.environ.get(var) == "1", f"{var} was not restored"


def test_unset_offline_flags_are_not_invented(monkeypatch, upload_env):
    """Restoring must not create variables that were never set."""
    for var in OFFLINE_VARS:
        monkeypatch.delenv(var, raising=False)

    _run(upload_env)

    for var in OFFLINE_VARS:
        assert var not in os.environ


def test_successful_upload_is_recorded(monkeypatch, upload_env):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    _run(upload_env)

    assert upload_env["repo_id"] == "someone/some-repo"
    assert upload_env["error"] is None


def test_upload_failure_is_recorded_not_swallowed(monkeypatch, upload_env):
    """Training still succeeds, but the failure must stop being invisible."""

    class _FailingApi(_RecordingApi):
        def create_repo(self, **kwargs):
            raise RuntimeError("offline mode is enabled")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FailingApi)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    # Must not raise -- a finished run is not failed over an upload problem.
    _run(upload_env)

    assert upload_env["error"] is not None
    assert "offline" in upload_env["error"]
    # ... and the offline flag is still restored on the failure path.
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


# --- the constant, which is what actually blocked the upload -----------------
# Regression: clearing os.environ alone is not enough, because huggingface_hub
# caches HF_HUB_OFFLINE into constants at import time.


def test_offline_constant_is_suspended_during_upload(monkeypatch, upload_env):
    """constants.HF_HUB_OFFLINE must read False while the upload runs."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True, raising=False)

    seen = {}

    class _ConstantWatchingApi(_RecordingApi):
        def create_repo(self, **kwargs):
            seen["offline"] = constants.HF_HUB_OFFLINE
            seen["is_offline_mode"] = constants.is_offline_mode()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _ConstantWatchingApi)

    _run(upload_env)

    assert seen["offline"] is False
    assert seen["is_offline_mode"] is False


def test_offline_constant_is_restored_after_upload(monkeypatch, upload_env):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True, raising=False)

    _run(upload_env)

    assert constants.HF_HUB_OFFLINE is True


def test_offline_constant_restored_even_when_upload_fails(monkeypatch, upload_env):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True, raising=False)

    class _FailingApi(_RecordingApi):
        def create_repo(self, **kwargs):
            raise RuntimeError("boom")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FailingApi)

    _run(upload_env)

    assert constants.HF_HUB_OFFLINE is True
    assert upload_env["error"] is not None


def test_constant_untouched_when_not_offline(monkeypatch, upload_env):
    """No offline mode set -> the constant must be left exactly as it was."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False, raising=False)

    _run(upload_env)

    assert constants.HF_HUB_OFFLINE is False
