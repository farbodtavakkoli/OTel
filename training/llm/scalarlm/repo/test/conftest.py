"""
Root test configuration.

Responsibilities:
- Make repo-local packages importable regardless of the caller's CWD. Inside
  the CPU container the image's PYTHONPATH already includes these, but
  pinning them here keeps `pytest test/unit/...` working from any cwd.
- Expose a handful of shared fixtures (tiny datasets, stub VLLMModelManager,
  tmp workdir shaping) used across layers.

Fixtures that should apply only to a specific layer (e.g. env cleanup for the
unit suite) live in that layer's own conftest.py.
"""

import os
import sys
from typing import Dict, List, Optional

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ("infra", "sdk", "ml"):
    _path = os.path.join(REPO_ROOT, _subdir)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Tiny datasets — 16 rows each, the size budget from docs/test-plan.md §3.2.
# ---------------------------------------------------------------------------

TINY_LM = [
    {"input": f"q{i}", "output": f"a{i}"} for i in range(16)
]

TINY_EMBED = [
    {"query": f"q{i}", "positive": f"p{i}", "negative": f"n{i}"}
    for i in range(16)
]

TINY_CHAT = [
    {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }
    for i in range(16)
]


@pytest.fixture
def tiny_lm():
    return list(TINY_LM)


@pytest.fixture
def tiny_embed():
    return list(TINY_EMBED)


@pytest.fixture
def tiny_chat():
    return list(TINY_CHAT)


# ---------------------------------------------------------------------------
# Stub VLLMModelManager — replaces the production manager in unit/component
# tests that need resolver behavior without loading a real model.
# ---------------------------------------------------------------------------


class StubVLLMModelManager:
    """
    In-process stand-in for cray_infra.training.vllm_model_manager.VLLMModelManager.

    Contract mirrored from production:
    - find_model(name) returns the registered name, or the base model
      fallback, or None for the sentinel "__unknown__".
    - register_model(name) is idempotent; order of first insertion is
      preserved by get_registered_models().
    """

    DEFAULT = "tiny-random/gemma-4-dense"

    def __init__(
        self,
        default: str = DEFAULT,
        pre_registered: Optional[List[str]] = None,
    ):
        self._models: Dict[str, str] = {default: default}
        for name in pre_registered or []:
            self._models[name] = name

    def find_model(self, name: Optional[str]) -> Optional[str]:
        if name == "__unknown__":
            return None
        if name is None:
            return self._models.get(self.DEFAULT)
        return self._models.get(name, self._models.get(self.DEFAULT))

    def register_model(self, name: str, path: str = "") -> None:
        self._models.setdefault(name, name)

    def get_registered_models(self) -> List[str]:
        return list(self._models.values())


@pytest.fixture
def stub_vllm():
    """Fresh StubVLLMModelManager per test."""
    return StubVLLMModelManager()


@pytest.fixture
def make_stub_vllm():
    """Factory form when the test needs a non-default default or pre-registrations."""
    def _make(**kwargs) -> StubVLLMModelManager:
        return StubVLLMModelManager(**kwargs)
    return _make


# ---------------------------------------------------------------------------
# tmp_workdir — a scratch directory shaped like /app/cray/ so tests that touch
# training_job_directory, inference_work_queue_path, upload_base_path all
# land in the same tmp_path hierarchy.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """
    Create {tmp}/jobs/, {tmp}/inference_requests/, point SCALARLM_* env vars
    at them, and yield the root. The autouse env-cleanup fixture in
    test/unit/conftest.py ensures these don't leak across tests.
    """
    jobs = tmp_path / "jobs"
    reqs = tmp_path / "inference_requests"
    queue_db = tmp_path / "inference_work_queue.sqlite"
    logs = tmp_path / "logs"
    for d in (jobs, reqs, logs):
        d.mkdir()

    monkeypatch.setenv("SCALARLM_TRAINING_JOB_DIRECTORY", str(jobs))
    monkeypatch.setenv("SCALARLM_UPLOAD_BASE_PATH", str(reqs))
    monkeypatch.setenv("SCALARLM_INFERENCE_WORK_QUEUE_PATH", str(queue_db))
    monkeypatch.setenv("SCALARLM_LOG_DIRECTORY", str(logs))

    return tmp_path
