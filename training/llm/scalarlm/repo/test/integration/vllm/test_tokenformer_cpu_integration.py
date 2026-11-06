#!/usr/bin/env python3
import sys

import pytest

sys.path.insert(0, "/app/cray/vllm")
sys.path.insert(0, "/app/cray/infra")


def test_tokenformer_manager_basic():
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

    print("Testing TokenformerModelManager initialization...")

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
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

    print("Testing TokenformerModelManager with model runner context...")

    class MockLLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(768, 50257)
            self.supports_lora = True

    model = MockLLMModel()
    device = torch.device("cpu")

    manager = TokenformerModelManager(model=model, device=device)

    assert manager.model is not None
    assert manager.device == device

    print("✓ TokenformerModelManager integration with model runner successful")


def test_tokenformer_with_lora_config():
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

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)

    model = MockModel()
    device = torch.device("cpu")

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
    try:
        from vllm.tokenformer.tokenformer_model_manager import (
            TokenformerModelManager,
            TokenformerModel,
        )

        print("Testing TokenformerModelManager availability...")

        assert TokenformerModelManager is not None
        assert TokenformerModel is not None

        print("✓ TokenformerModelManager and TokenformerModel are available")
    except ImportError as e:
        print(f"⚠ TokenformerModelManager not available: {e}")
        pytest.fail(f"TokenformerModelManager not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
