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
    cfg = JobConfig(**REQUIRED_FIELDS, trust_remote_code=True).dict()
    assert cfg["trust_remote_code"] is True


def test_trust_remote_code_present_in_default_dict():
    assert "trust_remote_code" in JobConfig(**REQUIRED_FIELDS).dict()
