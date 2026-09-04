"""Unit tests for the conditional adapter args (--enable-lora,
--enable-tokenformer).

Pure-inference pods can opt out via SCALARLM_ENABLE_LORA=false, which
drops the --enable-lora vLLM CLI arg. This avoids wrapping every layer
in a LoRA shim (perf win, plus sidesteps the fused_moe wrapper bug on
some vllm-fork builds).

--enable-tokenformer is what lets the fork load the Tokenformer-keyed
.pt checkpoints the ScalarLM trainer emits; without it the fork's
LoRA-only manager rejects them and generation against a trained model
404s.
"""

from __future__ import annotations

from cray_infra.one_server.vllm_cli_args import build_vllm_cli_args


def _base_config(**overrides):
    cfg = {
        "dtype": "auto",
        "gpu_memory_utilization": 0.85,
        "max_log_length": 100,
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": None,
        "enable_lora": False,
        "enable_tokenformer": True,
    }
    cfg.update(overrides)
    return cfg


def test_enable_lora_true_includes_flag():
    args = build_vllm_cli_args(_base_config(enable_lora=True))
    assert "--enable-lora" in args


def test_enable_lora_false_omits_flag():
    args = build_vllm_cli_args(_base_config(enable_lora=False))
    assert "--enable-lora" not in args


def test_enable_lora_default_false():
    """Missing key defaults to False, matching default_config.Config -- a
    stale config yaml must not produce a different server."""
    cfg = _base_config()
    del cfg["enable_lora"]
    args = build_vllm_cli_args(cfg)
    assert "--enable-lora" not in args


def test_enable_tokenformer_true_includes_flag():
    args = build_vllm_cli_args(_base_config(enable_tokenformer=True))
    assert "--enable-tokenformer" in args


def test_enable_tokenformer_false_omits_flag():
    args = build_vllm_cli_args(_base_config(enable_tokenformer=False))
    assert "--enable-tokenformer" not in args


def test_enable_tokenformer_default_true():
    """Missing key defaults to True: the trainer only produces
    Tokenformer adapters, so a config yaml that predates this flag must
    still serve them."""
    cfg = _base_config()
    del cfg["enable_tokenformer"]
    args = build_vllm_cli_args(cfg)
    assert "--enable-tokenformer" in args


def test_default_selects_tokenformer_only_not_hybrid():
    """Tokenformer without LoRA is the shipped default. Both flags
    together select the HybridAdapterManager, whose LoRA sub-manager
    nests every targeted layer under `.base_layer.` -- which makes the
    attention weights in a ScalarLM checkpoint unmatchable, so they are
    silently dropped and the served model runs random attention."""
    args = build_vllm_cli_args(_base_config())
    assert "--enable-lora" not in args
    assert "--enable-tokenformer" in args


def test_hybrid_is_still_reachable_for_lora_adapters():
    """Servers dedicated to adapter_type="lora" jobs need the opposite
    setting; the knob must not become one-way."""
    args = build_vllm_cli_args(_base_config(enable_lora=True))
    assert "--enable-lora" in args
    assert "--enable-tokenformer" in args


def test_other_required_flags_always_present():
    """The LoRA toggle must not accidentally drop other required args."""
    args = build_vllm_cli_args(_base_config(enable_lora=False))
    assert "--trust-remote-code" in args
    assert "--enable-auto-tool-choice" in args
    assert "--tool-call-parser=hermes" in args
    assert any(a.startswith("--tensor-parallel-size=") for a in args)
    assert any(a.startswith("--gpu-memory-utilization=") for a in args)


def test_limit_mm_per_prompt_included_when_set():
    args = build_vllm_cli_args(_base_config(limit_mm_per_prompt='{"image":2}'))
    assert "--limit-mm-per-prompt={\"image\":2}" in args


def test_limit_mm_per_prompt_omitted_when_none():
    args = build_vllm_cli_args(_base_config(limit_mm_per_prompt=None))
    assert not any(a.startswith("--limit-mm-per-prompt=") for a in args)
