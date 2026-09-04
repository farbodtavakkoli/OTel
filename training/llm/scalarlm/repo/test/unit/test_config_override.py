"""
Unit tests for cray_infra.util.get_config.

Contract under test: docs/configuration.md §1.1 — three-tier override
system (Pydantic defaults → YAML → SCALARLM_* env vars) with type-preserving
env coercion.
"""

import pytest

from cray_infra.util.get_config import get_config


def _empty_yaml_path(tmp_path, monkeypatch):
    """Point SCALARLM_CONFIG_PATH at a non-existent path so only defaults + env apply."""
    path = tmp_path / "does-not-exist.yaml"
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(path))
    return path


def test_config_defaults_when_no_yaml_and_no_env(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)

    cfg = get_config()

    assert cfg["model"] == "tiny-random/gemma-4-dense"
    assert cfg["api_url"] == "http://localhost:8000"
    assert cfg["max_model_length"] == 0
    assert cfg["gpu_memory_utilization"] == pytest.approx(0.40)
    assert cfg["tokenformer_r"] == 32
    assert cfg["tokenformer_num_heads"] == 4


def test_config_yaml_overrides_default(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text("model: other-model\nmax_model_length: 8192\n")
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))

    cfg = get_config()

    assert cfg["model"] == "other-model"
    assert cfg["max_model_length"] == 8192
    # Untouched fields still carry their default.
    assert cfg["api_url"] == "http://localhost:8000"


def test_config_env_overrides_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text("model: from-yaml\n")
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))
    monkeypatch.setenv("SCALARLM_MODEL", "from-env")

    cfg = get_config()

    assert cfg["model"] == "from-env"


def test_config_empty_yaml_is_safe(tmp_path, monkeypatch):
    # yaml.safe_load on an empty document returns None; the loader must not
    # blow up feeding None into Config(**...).
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text("")
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))

    cfg = get_config()

    assert cfg["model"] == "tiny-random/gemma-4-dense"


def test_config_env_casts_int(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_MAX_MODEL_LENGTH", "32768")

    cfg = get_config()

    assert cfg["max_model_length"] == 32768
    assert isinstance(cfg["max_model_length"], int)


def test_config_env_casts_float(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_GPU_MEMORY_UTILIZATION", "0.9")

    cfg = get_config()

    assert cfg["gpu_memory_utilization"] == pytest.approx(0.9)
    assert isinstance(cfg["gpu_memory_utilization"], float)


def test_config_env_preserves_string(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_DTYPE", "bfloat16")

    cfg = get_config()

    assert cfg["dtype"] == "bfloat16"


def test_config_env_rejects_invalid_int_cast(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_MAX_MODEL_LENGTH", "not-a-number")

    with pytest.raises(ValueError):
        get_config()


def test_config_env_rejects_invalid_float_cast(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_GPU_MEMORY_UTILIZATION", "0.9f")

    with pytest.raises(ValueError):
        get_config()


def test_config_env_non_field_is_ignored(tmp_path, monkeypatch):
    _empty_yaml_path(tmp_path, monkeypatch)
    monkeypatch.setenv("SCALARLM_NOT_A_REAL_FIELD", "ignored")

    cfg = get_config()

    # Field doesn't exist on the Pydantic model → never surfaces in the dict.
    assert "not_a_real_field" not in cfg


def test_config_yaml_extra_key_dropped_by_pydantic(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text("model: m1\nbogus_field: ignored\n")
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))

    cfg = get_config()

    assert cfg["model"] == "m1"
    assert "bogus_field" not in cfg


def test_config_every_get_call_rereads(tmp_path, monkeypatch):
    # documented in configuration.md §1.1: get_config has no cache.
    yaml_path = tmp_path / "cray-config.yaml"
    yaml_path.write_text("model: first\n")
    monkeypatch.setenv("SCALARLM_CONFIG_PATH", str(yaml_path))

    first = get_config()
    assert first["model"] == "first"

    yaml_path.write_text("model: second\n")
    second = get_config()
    assert second["model"] == "second"
