"""
Unit tests for the log-rotation wiring in
infra/cray_infra/one_server/main.py.

Without rotation, the API process writes
`/app/cray/nfs/logs/{vllm,megatron,api}.log` forever; on the gemma4
cluster (974 MB NFS PVC) this fills the volume in days, slurm.conf
writes start failing, and the pod becomes unusable.

The contract this module pins:
  - get_log_file_handlers attaches RotatingFileHandler instances, not
    plain FileHandlers.
  - maxBytes / backupCount are sourced from the cray config.
  - Rotation actually kicks in at the configured size (a real write
    test, because misconfigured rotation handlers silently degrade to
    "no rotation").
"""

import logging
import logging.handlers
import os
from unittest.mock import patch

import pytest

from cray_infra.one_server.log_handlers import get_log_file_handlers


@pytest.fixture
def fake_log_dir(tmp_path):
    target = tmp_path / "logs"
    fake_config = {
        "log_directory": str(target),
        "server_list": "api",
        "log_max_bytes": 1024,        # tiny on purpose so tests rotate
        "log_backup_count": 2,
    }
    with patch(
        "cray_infra.one_server.log_handlers.get_config",
        return_value=fake_config,
    ):
        yield target, fake_config


def test_attaches_rotating_handler(fake_log_dir):
    target, _ = fake_log_dir
    handlers = get_log_file_handlers()

    assert len(handlers) == 1
    handler = handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == 1024
    assert handler.backupCount == 2
    assert os.path.dirname(handler.baseFilename) == str(target)
    assert os.path.basename(handler.baseFilename) == "api.log"

    # Cleanup; the handler keeps an fd open.
    handler.close()


def test_creates_one_handler_per_server_in_list(tmp_path):
    fake_config = {
        "log_directory": str(tmp_path / "logs"),
        "server_list": "all",
        "log_max_bytes": 1024,
        "log_backup_count": 2,
    }
    with patch(
        "cray_infra.one_server.log_handlers.get_config",
        return_value=fake_config,
    ):
        handlers = get_log_file_handlers()

    try:
        names = sorted(os.path.basename(h.baseFilename) for h in handlers)
        assert names == ["api.log", "megatron.log", "vllm.log"]
        for h in handlers:
            assert isinstance(h, logging.handlers.RotatingFileHandler)
    finally:
        for h in handlers:
            h.close()


def test_rotation_actually_fires_at_max_bytes(fake_log_dir):
    """
    The most useful check: write enough bytes to cross maxBytes and
    verify that rotated `.1` / `.2` files appear and the live file
    stops growing past the cap. A misconfigured rotating handler
    (e.g. maxBytes=0) silently degrades to "never rotate"; this test
    fails loudly when that happens.
    """
    target, _ = fake_log_dir
    handlers = get_log_file_handlers()
    handler = handlers[0]

    test_logger = logging.getLogger("rotation-test")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)
    # Stop propagation so pytest's caplog doesn't intercept.
    test_logger.propagate = False

    try:
        # Each line is roughly 80 bytes; 200 of them blow well past
        # the 1024-byte cap and force multiple rotations.
        for i in range(200):
            test_logger.info("line %03d %s", i, "x" * 60)

        log_path = handler.baseFilename
        # Live file must be <= maxBytes after the final rotation. We
        # allow slop equal to one record width because rotation
        # decides post-write.
        assert os.path.getsize(log_path) <= 1024 + 200

        # backupCount=2 means we keep .1 and .2 rotated copies.
        assert os.path.exists(log_path + ".1"), "first rotation never fired"
        assert os.path.exists(log_path + ".2"), "second rotation never fired"
        assert not os.path.exists(log_path + ".3"), (
            "rotation kept too many backups — backupCount not honored"
        )
    finally:
        test_logger.removeHandler(handler)
        for h in handlers:
            h.close()


def test_handler_uses_config_values_not_hardcoded(tmp_path):
    """
    Edit safety: a future refactor that hardcodes a different default
    must not silently override the config values.
    """
    fake_config = {
        "log_directory": str(tmp_path / "logs"),
        "server_list": "api",
        "log_max_bytes": 7777,
        "log_backup_count": 9,
    }
    with patch(
        "cray_infra.one_server.log_handlers.get_config",
        return_value=fake_config,
    ):
        handlers = get_log_file_handlers()
    try:
        assert handlers[0].maxBytes == 7777
        assert handlers[0].backupCount == 9
    finally:
        for h in handlers:
            h.close()


def test_default_config_exposes_rotation_knobs():
    """Document the defaults so a careless edit to default_config.py
    forces a config-doc update."""
    from cray_infra.util.default_config import Config

    cfg = Config()
    assert cfg.log_max_bytes == 10 * 1024 * 1024
    assert cfg.log_backup_count == 5
