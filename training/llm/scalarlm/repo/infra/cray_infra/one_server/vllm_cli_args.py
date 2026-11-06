from __future__ import annotations

import os
import posixpath
import re


def build_vllm_cli_args(config: dict) -> list[str]:
    args = [
        f"--dtype={config['dtype']}",
        "--max-model-len=auto",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        f"--tensor-parallel-size={config['tensor_parallel_size']}",
        "--enable-auto-tool-choice",
        "--tool-call-parser=hermes",
        "--trust-remote-code",
    ]
    if config.get("enable_lora", False):
        args.append("--enable-lora")
    if config.get("enable_tokenformer", True):
        args.append("--enable-tokenformer")
    if config.get("limit_mm_per_prompt") is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")
    reasoning_parser = resolve_reasoning_parser(config)
    if reasoning_parser:
        args.append(f"--reasoning-parser={reasoning_parser}")
    return args


_REASONING_PARSER_FLAG_NAME = "--reasoning-parser"
_REASONING_PARSER_FLAG_PREFIX = _REASONING_PARSER_FLAG_NAME + "="


def _reasoning_parser_from_env_override() -> str | None:
    tokens = os.environ.get("SCALARLM_VLLM_ARGS", "").split()
    value = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith(_REASONING_PARSER_FLAG_PREFIX):
            value = token[len(_REASONING_PARSER_FLAG_PREFIX) :]
            i += 1
        elif (
            token == _REASONING_PARSER_FLAG_NAME
            and i + 1 < len(tokens)
            and not tokens[i + 1].startswith("-")
        ):
            value = tokens[i + 1]
            i += 2
        else:
            i += 1
    return value


def resolve_reasoning_parser(config: dict) -> str | None:
    env_override = _reasoning_parser_from_env_override()
    if env_override is not None:
        return env_override

    reasoning_parser = config.get("reasoning_parser")
    if reasoning_parser is None:
        reasoning_parser = _detect_reasoning_parser(config.get("model") or "")
    return reasoning_parser


_QWEN3_HYBRID_CHECKPOINTS = (
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
)
_QWEN3_THINKING_2507_CHECKPOINTS = (
    "qwen3-4b-thinking-2507",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-235b-a22b-thinking-2507",
)
_QWEN35_36_CHECKPOINTS = (
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b",
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
)

_PACKAGING_SUFFIX_PATTERN = (
    r"(?:"
    r"-(?:mxfp8|q4_k_m\.gguf)"
    r"|-(?:fp8|awq|gptq|gguf(?::[a-z0-9_]+)?|nvfp4|mxfp4|"
    r"w\d+a\d+|int\d+|quantized[._-]w\d+a\d+)"
    r"(?:-(?:dynamic|\d+bit|w\d+a\d+|g\d+|int\d+))*"
    r")?"
)


def _has_supported_packaging_suffix(checkpoint: str, base: str) -> bool:
    if checkpoint == base:
        return True
    if not checkpoint.startswith(base):
        return False
    suffix = checkpoint.removeprefix(base)
    return bool(re.fullmatch(_PACKAGING_SUFFIX_PATTERN, suffix))


def _is_supported_thinking_2507_checkpoint(checkpoint: str) -> bool:
    for base in _QWEN3_THINKING_2507_CHECKPOINTS:
        if checkpoint == base:
            return True
        if not checkpoint.startswith(base):
            continue
        suffix = checkpoint.removeprefix(base)
        if suffix.startswith("-ptpc"):
            suffix = suffix.removeprefix("-ptpc")
        if re.fullmatch(_PACKAGING_SUFFIX_PATTERN, suffix):
            return True
    return False


def _checkpoint_name(model: str) -> str:
    normalized = posixpath.normpath(model.strip().replace("\\", "/"))
    if normalized in ("", "."):
        return ""

    parts = tuple(part for part in normalized.split("/") if part)
    cache_indexes = tuple(
        index for index, part in enumerate(parts) if part.lower().startswith("models--")
    )
    if cache_indexes:
        if len(cache_indexes) != 1:
            return ""
        index = cache_indexes[0]
        encoded_repo = parts[index][len("models--") :]
        repo_parts = encoded_repo.split("--")
        if (
            len(repo_parts) != 2
            or not all(repo_parts)
            or index + 2 >= len(parts)
            or parts[index + 1].lower() != "snapshots"
        ):
            return ""
        if index + 2 == len(parts) - 1:
            return repo_parts[-1]
        return parts[-1]
    return parts[-1] if parts else ""


def _detect_reasoning_parser(model: str) -> str | None:
    checkpoint_lower = _checkpoint_name(model.lower())
    if any(
        _has_supported_packaging_suffix(checkpoint_lower, base)
        for base in _QWEN3_HYBRID_CHECKPOINTS
    ):
        return "qwen3"
    if _is_supported_thinking_2507_checkpoint(checkpoint_lower):
        return "qwen3"
    if any(
        _has_supported_packaging_suffix(checkpoint_lower, base)
        for base in _QWEN35_36_CHECKPOINTS
    ):
        return "qwen3"
    return None
