from cray_infra.util.default_job_config import JobConfig


REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "abc",
}


def test_diffusion_defaults_to_none():
    cfg = JobConfig(**REQUIRED_FIELDS).dict()
    assert cfg["diffusion"] is None


def test_diffusion_block_survives_roundtrip():
    cfg = JobConfig(
        **REQUIRED_FIELDS,
        diffusion={"canvas_length": 128, "eps": 0.002},
    ).dict()
    assert cfg["diffusion"]["canvas_length"] == 128
    assert cfg["diffusion"]["eps"] == 0.002
    assert cfg["diffusion"]["self_conditioning_prob"] == 0.5
    assert cfg["diffusion"]["anchor_token"] is False
    assert cfg["diffusion"]["supervise_termination"] is False
    assert cfg["diffusion"]["pad_loss_weight"] == 1.0


def test_diffusion_supervise_termination_survives_roundtrip():
    cfg = JobConfig(
        **REQUIRED_FIELDS,
        diffusion={"supervise_termination": True, "pad_loss_weight": 0.3},
    ).dict()
    assert cfg["diffusion"]["supervise_termination"] is True
    assert cfg["diffusion"]["pad_loss_weight"] == 0.3


def test_diffusion_anchor_token_survives_roundtrip():
    cfg = JobConfig(**REQUIRED_FIELDS, diffusion={"anchor_token": True}).dict()
    assert cfg["diffusion"]["anchor_token"] is True


def test_diffusion_block_defaults_fill_missing_fields():
    cfg = JobConfig(**REQUIRED_FIELDS, diffusion={"canvas_length": 512}).dict()
    assert cfg["diffusion"]["canvas_length"] == 512
    assert cfg["diffusion"]["eps"] == 0.001
