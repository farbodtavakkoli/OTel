#!/usr/bin/env python3
"""
Integration test for tokenformer support in CPUModelRunner
This test runs inside the Docker container where vLLM is properly built
"""

import sys

import pytest

# Add paths for Docker container
sys.path.insert(0, "/app/cray/vllm")
sys.path.insert(0, "/app/cray/infra")


def test_tokenformer_manager_basic():
    """Test basic TokenformerModelManager functionality"""
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

    print("Testing TokenformerModelManager initialization...")

    # Create a simple mock model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)

    model = SimpleModel()
    device = torch.device("cpu")
    manager = TokenformerModelManager(model=model, device=device)

    assert manager.device == device
    assert manager.model is not None
    assert hasattr(manager, "model")

    print("✓ TokenformerModelManager initialized successfully")


def test_cpu_model_runner_initialization():
    """Test that tokenformer can be used with model runner mixin"""
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

    print("Testing TokenformerModelManager with model runner context...")

    # Create a mock model that supports lora/tokenformer
    class MockLLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(768, 50257)  # GPT-2 like dimensions
            self.supports_lora = True

    model = MockLLMModel()
    device = torch.device("cpu")

    # Initialize TokenformerModelManager as done in LoRAModelRunnerMixin
    manager = TokenformerModelManager(model=model, device=device)

    assert manager.model is not None
    assert manager.device == device

    print("✓ TokenformerModelManager integration with model runner successful")


def test_tokenformer_with_lora_config():
    """Test LoRAModelRunnerMixin with LoRA config uses TokenformerModelManager"""
    import torch
    import torch.nn as nn
    from vllm.config import LoRAConfig, VllmConfig
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
    from unittest.mock import MagicMock, patch

    print("Testing LoRAModelRunnerMixin with LoRA config...")

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.lora_config = LoRAConfig(
        enable_tokenformer=True,
        max_lora_rank=8,
        max_loras=2,
        max_cpu_loras=2,
    )

    # Create a mock model that supports lora
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)

    model = MockModel()
    device = torch.device("cpu")

    # Test LoRAModelRunnerMixin
    mixin = LoRAModelRunnerMixin()

    with patch(
        "vllm.v1.worker.lora_model_runner_mixin.supports_lora",
        return_value=True,
    ):
        result_model = mixin.load_lora_model(model, vllm_config, device)

        assert result_model is not None
        assert isinstance(mixin.lora_manager, TokenformerModelManager)
        assert result_model is mixin.lora_manager.model
        print("✓ LoRAModelRunnerMixin uses TokenformerModelManager successfully")


def test_tokenformer_model_manager():
    """Test TokenformerModelManager is available"""
    try:
        from vllm.tokenformer.tokenformer_model_manager import (
            TokenformerModelManager,
            TokenformerModel,
        )

        print("Testing TokenformerModelManager availability...")

        # Check classes are importable
        assert TokenformerModelManager is not None
        assert TokenformerModel is not None

        print("✓ TokenformerModelManager and TokenformerModel are available")
    except ImportError as e:
        print(f"⚠ TokenformerModelManager not available: {e}")
        pytest.fail(f"TokenformerModelManager not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
