"""
Unit tests for the per-job `dtype` override.

Contract under test: train_args["dtype"] passed via the SDK round-trips
into JobConfig.dtype and wins over the global cray-config.yaml `dtype`
inside load_model. The FAQ documents this knob; before this fix the
field was missing from JobConfig (Pydantic silently dropped it) and
load_model read only the global config, so the train_args entry had
no effect.
"""

import pytest

from cray_infra.util.default_job_config import JobConfig


REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "abc",
}


def test_job_config_dtype_defaults_to_auto():
    cfg = JobConfig(**REQUIRED_FIELDS).dict()
    assert cfg["dtype"] == "auto"


def test_job_config_accepts_dtype_override():
    cfg = JobConfig(**REQUIRED_FIELDS, dtype="float32").dict()
    assert cfg["dtype"] == "float32"


@pytest.mark.parametrize("dtype", ["auto", "float32", "float16", "bfloat16"])
def test_job_config_accepts_documented_dtypes(dtype):
    cfg = JobConfig(**REQUIRED_FIELDS, dtype=dtype).dict()
    assert cfg["dtype"] == dtype


def _resolve(job_dtype, global_dtype):
    """Mirror load_model.py's dtype resolution so the precedence is testable
    without spinning up a 4-layer Gemma forward pass."""
    return job_dtype if job_dtype != "auto" else global_dtype


def test_resolution_prefers_job_dtype():
    assert _resolve(job_dtype="float32", global_dtype="bfloat16") == "float32"


def test_resolution_falls_back_to_global_when_job_is_auto():
    assert _resolve(job_dtype="auto", global_dtype="bfloat16") == "bfloat16"


def test_resolution_both_auto_stays_auto():
    assert _resolve(job_dtype="auto", global_dtype="auto") == "auto"
