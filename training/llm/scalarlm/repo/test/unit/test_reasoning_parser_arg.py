"""Focused tests for vLLM reasoning-parser startup configuration."""

from __future__ import annotations

import pytest

from cray_infra.one_server.vllm_cli_args import (
    build_vllm_cli_args,
    resolve_reasoning_parser,
)


@pytest.fixture(autouse=True)
def _clear_scalarlm_vllm_args_env(monkeypatch):
    monkeypatch.delenv("SCALARLM_VLLM_ARGS", raising=False)


def _base_config(**overrides):
    config = {
        "dtype": "auto",
        "gpu_memory_utilization": 0.85,
        "max_log_length": 100,
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": None,
        "enable_lora": False,
        "enable_tokenformer": True,
        "model": "",
        "reasoning_parser": None,
    }
    config.update(overrides)
    return config


def _reasoning_parser_args(config):
    return [
        arg
        for arg in build_vllm_cli_args(config)
        if arg.startswith("--reasoning-parser=")
    ]


def test_explicit_parser_is_forwarded():
    assert _reasoning_parser_args(_base_config(reasoning_parser="deepseek_r1")) == [
        "--reasoning-parser=deepseek_r1"
    ]


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-4B-Thinking-2507",
        "amd/Qwen3-30B-A3B-Thinking-2507-PTPC-FP8",
        "qwen/Qwen3.5-0.8B",
        "qwen/Qwen3.5-2B",
        "qwen/Qwen3.5-4B",
        "qwen/Qwen3.5-9B",
        "qwen/Qwen3.5-27B",
        "qwen/qWeN3.5-35B-A3B-FP8",
        "qwen/Qwen3.5-122B-A10B",
        "nvidia/Qwen3.5-122B-A10B-NVFP4",
        "nvidia/Qwen3.5-397B-A17B-NVFP4",
        "qwen/qWeN3.6-27B",
        "nvidia/Qwen3.6-35B-A3B-NVFP4-W4A4",
        "vendor/Qwen3.6-35B-A3B-GPTQ-W4A16-G32",
        "mgoin/Qwen3-0.6B-MXFP8",
        "/models/Qwen3-0.6B-Q4_K_M.gguf",
    ],
)
def test_supported_qwen3_checkpoint_is_detected_case_insensitively(model):
    assert _reasoning_parser_args(_base_config(model=model)) == [
        "--reasoning-parser=qwen3"
    ]


def test_third_party_name_containing_qwen3_does_not_trigger_detection():
    for model in ("org/My-Qwen3-Derivative", "org/prefixqwen3-32B"):
        assert not _reasoning_parser_args(_base_config(model=model))


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Reranker-8B",
        "Qwen/Qwen3-Text-Classifier-8B",
        "Qwen/Qwen3-Guard-Gen-8B",
        "Qwen/Qwen3Guard-Gen-8B",
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-ASR-1.7B",
        "Qwen/Qwen3.6-27B-Instruct",
        "vendor/Qwen3-999B-Thinking-2507",
        "vendor/Qwen3.5-999B",
        "vendor/Qwen3.5-0B-A0B",
        "vendor/Qwen3.5-35B-A17B",
        "vendor/Qwen3.6-999B",
        "vendor/Qwen3.6-0B-A0B",
        "vendor/Qwen3.6-35B-A17B",
        "vendor/Qwen3-0.6B-MXFP80",
        "vendor/Qwen3-0.6B-MXFP8-extra",
        "/models/Qwen3-0.6B-Q4_K_M.gguf.bak",
        "/models/Qwen3-0.6B-Q5_K_M.gguf",
    ],
)
def test_unsupported_qwen3_variant_or_packaging_is_excluded(model):
    assert not _reasoning_parser_args(_base_config(model=model))


def test_explicit_parser_can_override_a_non_reasoning_name_exclusion():
    assert _reasoning_parser_args(
        _base_config(
            model="Qwen/Qwen3-Next-80B-A3B-Thinking",
            reasoning_parser="deepseek_r1",
        )
    ) == ["--reasoning-parser=deepseek_r1"]


def test_qwen3_checkpoint_path_is_detected():
    for model in (
        "/models/Qwen3-32B",
        "/models/Qwen3-32B/",
        "/cache/Qwen3-4B-Thinking-2507///",
    ):
        assert _reasoning_parser_args(_base_config(model=model)) == [
            "--reasoning-parser=qwen3"
        ]


@pytest.mark.parametrize(
    "model",
    [
        (
            "/home/user/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-32B/snapshots/0123456789abcdef/"
        ),
        ("/models/hub/models--nvidia--Qwen3.5-397B-A17B-NVFP4/" "snapshots/deadbeef"),
        (r"C:\cache\models--Qwen--Qwen3-32B\snapshots\deadbeef"),
    ],
)
def test_hugging_face_snapshot_path_uses_encoded_repository_name(model):
    assert _reasoning_parser_args(_base_config(model=model)) == [
        "--reasoning-parser=qwen3"
    ]


def test_hugging_face_snapshot_revision_cannot_trigger_detection():
    model = (
        "/home/user/.cache/huggingface/hub/"
        "models--meta-llama--Llama-3.1-8B-Instruct/"
        "snapshots/Qwen3-32B"
    )
    assert not _reasoning_parser_args(_base_config(model=model))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "/mnt/models--Qwen--Qwen3-32B/snapshots/rev/Llama-3.1-8B",
            [],
        ),
        (
            "/mnt/models--meta-llama--Llama-3.1-8B/snapshots/rev/Qwen3-32B",
            ["--reasoning-parser=qwen3"],
        ),
        (
            "/mnt/models--meta-llama--Llama-3.1-8B/snapshots/rev/"
            "Qwen3-0.6B-Q4_K_M.gguf",
            ["--reasoning-parser=qwen3"],
        ),
    ],
)
def test_nested_snapshot_path_uses_actual_final_checkpoint(model, expected):
    assert _reasoning_parser_args(_base_config(model=model)) == expected


@pytest.mark.parametrize(
    "model",
    [
        "models--meta--/snapshots/Qwen3-32B",
        "models--Qwen-Qwen3-32B/snapshots/Qwen3-32B",
        "models--Qwen----Qwen3-32B/snapshots/Qwen3-32B",
        "models--Qwen--Qwen3-32B--extra/snapshots/Qwen3-32B",
        "models--Qwen--Qwen3-32B/not-snapshots/Qwen3-32B",
        "models--Qwen--Qwen3-32B/snapshots",
        ("models--Qwen--Qwen3-32B/snapshots/rev/" "models--meta--/snapshots/Qwen3-32B"),
    ],
)
def test_malformed_hugging_face_cache_path_does_not_trigger_detection(model):
    assert not _reasoning_parser_args(_base_config(model=model))


def test_unrecognized_checkpoint_does_not_guess_parser():
    assert not _reasoning_parser_args(
        _base_config(model="meta-llama/Llama-3.1-8B-Instruct")
    )


def test_explicit_parser_overrides_detection():
    assert _reasoning_parser_args(
        _base_config(model="Qwen/Qwen3-32B", reasoning_parser="qwen3_xml")
    ) == ["--reasoning-parser=qwen3_xml"]


def test_explicit_empty_string_disables_detection():
    assert not _reasoning_parser_args(
        _base_config(model="Qwen/Qwen3-32B", reasoning_parser="")
    )


def test_missing_config_key_still_allows_detection():
    config = _base_config(model="Qwen/Qwen3-32B")
    del config["reasoning_parser"]
    assert _reasoning_parser_args(config) == ["--reasoning-parser=qwen3"]


def test_none_model_is_safe():
    assert not _reasoning_parser_args(_base_config(model=None))


def test_exclusions_are_scoped_to_checkpoint_name():
    for model in ("coder-team/Qwen3-32B", "/models/next/Qwen3-32B"):
        assert _reasoning_parser_args(_base_config(model=model)) == [
            "--reasoning-parser=qwen3"
        ]


def test_positive_match_is_scoped_to_checkpoint_name():
    for model in ("qwen3-team/Llama-3.1-8B", "/models/qwen3/Llama-3.1-8B"):
        assert not _reasoning_parser_args(_base_config(model=model))


def test_env_override_wins_over_detection(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser=deepseek_r1")
    assert _reasoning_parser_args(_base_config(model="Qwen/Qwen3-32B")) == [
        "--reasoning-parser=deepseek_r1"
    ]


def test_env_override_wins_over_explicit_config(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser=deepseek_r1")
    assert _reasoning_parser_args(_base_config(reasoning_parser="qwen3_xml")) == [
        "--reasoning-parser=deepseek_r1"
    ]


def test_unrelated_env_override_does_not_change_parser(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--max-num-seqs=64")
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "qwen3_xml"
    )


def test_space_separated_env_override_is_supported(monkeypatch):
    monkeypatch.setenv(
        "SCALARLM_VLLM_ARGS", "--max-num-seqs=64 --reasoning-parser deepseek_r1"
    )
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "deepseek_r1"
    )


def test_last_valid_env_override_wins(monkeypatch):
    monkeypatch.setenv(
        "SCALARLM_VLLM_ARGS",
        "--reasoning-parser=first --reasoning-parser second",
    )
    assert resolve_reasoning_parser(_base_config()) == "second"


def test_malformed_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("SCALARLM_VLLM_ARGS", "--reasoning-parser --max-num-seqs=64")
    assert (
        resolve_reasoning_parser(_base_config(reasoning_parser="qwen3_xml"))
        == "qwen3_xml"
    )
