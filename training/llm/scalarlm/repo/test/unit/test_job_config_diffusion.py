"""
Unit tests for the per-job `diffusion` block on JobConfig.

Contract under test: train_args["diffusion"] = {canvas_length, eps} passed via the
SDK round-trips into JobConfig.diffusion and reaches the DiffusionGemma dataset
loader / training step. Nested config blocks are exactly what the Pydantic model
silently drops when a field isn't declared (the same footgun as `trust_remote_code`
and `dtype`), so this pins that the block survives and defaults are sane.
"""

from cray_infra.util.default_job_config import JobConfig


REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "abc",
}


def test_diffusion_defaults_to_none():
    # Non-diffusion jobs carry no diffusion block; is_diffusion() gates whether it
    # is ever read, so None is the correct "not applicable" default.
    cfg = JobConfig(**REQUIRED_FIELDS).dict()
    assert cfg["diffusion"] is None


def test_diffusion_block_survives_roundtrip():
    cfg = JobConfig(
        **REQUIRED_FIELDS,
        diffusion={"canvas_length": 128, "eps": 0.002},
    ).dict()
    assert cfg["diffusion"]["canvas_length"] == 128
    assert cfg["diffusion"]["eps"] == 0.002
    # self_conditioning_prob is unspecified here -> its DiffusionConfig default.
    assert cfg["diffusion"]["self_conditioning_prob"] == 0.5
    # anchor_token defaults off so prior runs stay byte-identical.
    assert cfg["diffusion"]["anchor_token"] is False
    # Termination-supervision defaults off; pad weight uniform.
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
    # A partial block still parses; the unspecified field takes the DiffusionConfig
    # default (canvas_length 256, eps 0.001).
    cfg = JobConfig(**REQUIRED_FIELDS, diffusion={"canvas_length": 512}).dict()
    assert cfg["diffusion"]["canvas_length"] == 512
    assert cfg["diffusion"]["eps"] == 0.001
