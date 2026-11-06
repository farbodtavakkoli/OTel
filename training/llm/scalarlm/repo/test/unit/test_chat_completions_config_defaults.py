from cray_infra.util.default_config import Config


def test_max_num_seqs_default():
    assert Config().max_num_seqs == 256


def test_chat_admit_factor_default():
    assert Config().chat_admit_factor == 4


def test_chat_coalescer_packing_factor_default():
    assert Config().chat_coalescer_packing_factor == 10


def test_chat_coalescer_window_ms_default():
    assert Config().chat_coalescer_window_ms == 50


def test_chat_coalescer_bypass_threshold_default():
    cfg = Config()
    assert cfg.chat_coalescer_bypass_threshold == 10
    assert cfg.chat_coalescer_bypass_threshold == cfg.chat_coalescer_packing_factor


def test_chat_heartbeat_interval_seconds_default():
    assert Config().chat_heartbeat_interval_seconds == 4


def test_chat_wait_estimator_defaults():
    cfg = Config()
    assert cfg.chat_wait_estimator_default_seconds == 5.0
    assert cfg.chat_wait_estimator_padding == 1.5
    assert cfg.chat_wait_estimator_sample_size == 32


def test_chat_buffering_detection_defaults():
    cfg = Config()
    assert cfg.chat_buffering_check_proxy_timeout_seconds == 60
    assert cfg.chat_buffering_match_threshold_seconds == 0.5
