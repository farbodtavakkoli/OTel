import os

import pytest

from cray_megatron import hf_upload


OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


class _RecordingApi:
    seen_env = {}

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, **kwargs):
        _RecordingApi.seen_env = {v: os.environ.get(v) for v in OFFLINE_VARS}

    def upload_file(self, **kwargs):
        return "https://huggingface.co/fake/commit"


@pytest.fixture
def upload_env(monkeypatch, tmp_path):
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

    assert _RecordingApi.seen_env == {v: None for v in OFFLINE_VARS}


def test_offline_flags_are_restored_after_upload(monkeypatch, upload_env):
    for var in OFFLINE_VARS:
        monkeypatch.setenv(var, "1")

    _run(upload_env)

    for var in OFFLINE_VARS:
        assert os.environ.get(var) == "1", f"{var} was not restored"


def test_unset_offline_flags_are_not_invented(monkeypatch, upload_env):
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
    class _FailingApi(_RecordingApi):
        def create_repo(self, **kwargs):
            raise RuntimeError("offline mode is enabled")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FailingApi)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    _run(upload_env)

    assert upload_env["error"] is not None
    assert "offline" in upload_env["error"]
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_offline_constant_is_suspended_during_upload(monkeypatch, upload_env):
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
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False, raising=False)

    _run(upload_env)

    assert constants.HF_HUB_OFFLINE is False
