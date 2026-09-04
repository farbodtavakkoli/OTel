"""
Regression test: displayed train time must survive checkpoint resumes.

Bug under test: the history entry's ``time`` field was written as the
*current slice's* elapsed (``time.time() - start_time``), but ``start_time``
resets at every slice's ``on_train_begin``. Both the UI elapsed clock
(routes/train/TrainDetail.tsx) and the models-list ``train_time``
(training.list_models.get_train_time) read ``history[-1].time`` directly, so
they reset to ~0 on every resume — a 48h job showed ~13h.

``update_history`` must fold in ``accumulated_seconds_at_slice_start`` (the
wall time carried forward from prior slices, loaded by
``_load_accumulated_seconds`` and persisted by ``_finalize_slice``) so the
recorded time is cumulative across slices.
"""

import json

import pytest

from cray_megatron.megatron.training_loop import TrainingLoop
from cray_megatron.megatron.training_harness import TrainingHarness


@pytest.fixture
def job_config_env(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    data_path = tmp_path / "dataset.jsonl"
    data_path.write_text("")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"job_directory: {job_dir}",
                f"training_data_path: {data_path}",
                "dataset_hash: testhash",
                "steps_per_checkpoint: 5",
                "cuda_memory_log_interval: 0",
                "timeout: 3600",
                "training_history_length: 1024",
            ]
        )
    )
    monkeypatch.setenv("CRAY_TRAINING_JOB_CONFIG_PATH", str(config_path))
    return job_dir


@pytest.fixture(autouse=True)
def rank_zero(monkeypatch):
    import cray_megatron.collectives.main_rank_only as m

    m._in_main_rank_only = False
    monkeypatch.setattr(m, "get_rank", lambda: 0)
    monkeypatch.setattr(m, "barrier", lambda: None)
    yield
    m._in_main_rank_only = False


def _make_loop(job_dir):
    loop = TrainingLoop(training_harness=TrainingHarness())
    loop.training_state.history = []
    return loop


def test_history_time_includes_prior_slice_seconds(job_config_env, monkeypatch):
    import cray_megatron.megatron.training_loop as tl

    loop = _make_loop(job_config_env)

    # Slice started "now"; 10s of in-slice wall clock have elapsed; 48h of
    # prior slices were carried forward from status.json.
    prior_seconds = 48 * 60 * 60
    loop.training_state.start_time = 1_000.0
    loop.training_state.accumulated_seconds_at_slice_start = prior_seconds
    loop.training_state.current_step = 7
    loop.training_state.epoch = 2

    monkeypatch.setattr(tl.time, "time", lambda: 1_010.0)

    loop.update_history(loss=0.5)

    entry = loop.training_state.history[-1]
    # 48h + 10s, not just 10s.
    assert entry["time"] == pytest.approx(prior_seconds + 10.0)


def test_first_slice_time_is_just_in_slice_elapsed(job_config_env, monkeypatch):
    """With no prior slices (accumulated == 0) the behavior is unchanged:
    the recorded time is the current slice's elapsed."""
    import cray_megatron.megatron.training_loop as tl

    loop = _make_loop(job_config_env)
    loop.training_state.start_time = 500.0
    loop.training_state.accumulated_seconds_at_slice_start = 0.0
    loop.training_state.current_step = 1

    monkeypatch.setattr(tl.time, "time", lambda: 542.0)

    loop.update_history(loss=1.0)

    assert loop.training_state.history[-1]["time"] == pytest.approx(42.0)


def test_get_train_time_reads_accumulated_value(job_config_env, monkeypatch):
    """The backend list_models.get_train_time reads history[-1].time, so the
    fix at the source flows through to the models-list train_time too."""
    import cray_megatron.megatron.training_loop as tl
    from cray_infra.training.list_models import get_train_time

    loop = _make_loop(job_config_env)
    loop.training_state.start_time = 0.0
    loop.training_state.accumulated_seconds_at_slice_start = 100_000.0

    monkeypatch.setattr(tl.time, "time", lambda: 1.0)
    loop.update_history(loss=0.1)

    # Reconstruct the status.json shape get_train_time consumes.
    status = json.loads((job_config_env / "status.json").read_text())
    assert get_train_time(status) == pytest.approx(100_001.0)
