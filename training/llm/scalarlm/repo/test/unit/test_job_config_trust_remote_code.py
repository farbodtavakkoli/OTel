"""
Unit tests for the per-job `trust_remote_code` override.

Contract under test: train_args["trust_remote_code"] passed via the SDK
round-trips into JobConfig.trust_remote_code and reaches load_model, so
models that ship custom modeling code (InternVL3, Molmo, GLM-4/ChatGLM,
...) can be TRAINED. Before this fix the field was missing from JobConfig
(Pydantic silently dropped it), so get_job_config() always returned the
False default and load_model called AutoConfig.from_pretrained WITHOUT
trust_remote_code — raising "contains custom code which must be executed
... pass trust_remote_code=True" and TRAIN_FAILED, even though config.yaml
carried `trust_remote_code: true`. Same failure mode as the historical
`dtype` drop (see test_job_config_dtype.py).
"""

from cray_infra.util.default_job_config import JobConfig


REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "abc",
}


def test_trust_remote_code_defaults_to_false():
    cfg = JobConfig(**REQUIRED_FIELDS).dict()
    assert cfg["trust_remote_code"] is False


def test_trust_remote_code_override_survives_roundtrip():
    # The regression: this key must NOT be dropped by the Pydantic model.
    cfg = JobConfig(**REQUIRED_FIELDS, trust_remote_code=True).dict()
    assert cfg["trust_remote_code"] is True


def test_trust_remote_code_present_in_default_dict():
    # load_model does job_config.get("trust_remote_code", False); the key
    # should exist explicitly (not rely on the .get default) so the field
    # is a first-class part of the job contract.
    assert "trust_remote_code" in JobConfig(**REQUIRED_FIELDS).dict()
