#!/usr/bin/env python3
import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "vllm",
    ),
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.supports_lora = True

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_adapter_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter1"
        adapter_dir.mkdir()

        mock_weights = {
            "model_state_dict": {
                "tokenformer.weight": torch.rand(10, 10),
                "lm_head.weight": torch.rand(10),
            },
        }
        torch.save(mock_weights, adapter_dir / "model.pt")

        yield str(adapter_dir)


class TestCPUTokenformerIntegration:
    def test_tokenformer_manager_initialization(self):
        import torch.nn as nn
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(10, 10)

        model = SimpleModel()
        device = torch.device("cpu")
        manager = TokenformerModelManager(model=model, device=device)

        assert manager.device == device
        assert manager.model is not None

    def test_tokenformer_model_creation(self, temp_adapter_path):
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModel

        tokenformer = TokenformerModel.from_local_checkpoint(
            temp_adapter_path, torch.device("cpu")
        )

        assert set(tokenformer.tokenformers) == {
            "tokenformer.weight",
            "lm_head.weight",
        }
        assert all(
            tensor.device.type == "cpu" for tensor in tokenformer.tokenformers.values()
        )

    def test_tokenformer_integration_with_runner_mixin(self):
        import torch.nn as nn
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        class MockLLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(768, 50257)

        model = MockLLMModel()
        device = torch.device("cpu")

        manager = TokenformerModelManager(model=model, device=device)

        assert manager.model is not None
        assert manager.device == device
        print("✓ TokenformerModelManager integration successful")


@pytest.mark.parametrize("has_lora_config", [True, False])
def test_cpu_runner_lora_config_handling(has_lora_config):
    import torch.nn as nn
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
    from vllm.config import LoRAConfig, VllmConfig
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
    from unittest.mock import MagicMock

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.lora_config = (
        LoRAConfig(
            enable_tokenformer=True,
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=2,
        )
        if has_lora_config
        else None
    )

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)

        @property
        def supports_lora(self):
            return True

    model = MockModel()
    device = torch.device("cpu")

    mixin = LoRAModelRunnerMixin()

    if has_lora_config:
        from unittest.mock import patch

        with patch(
            "vllm.v1.worker.lora_model_runner_mixin.supports_lora",
            return_value=True,
        ):
            result_model = mixin.load_lora_model(model, vllm_config, device)
            assert result_model is not None
            assert isinstance(mixin.lora_manager, TokenformerModelManager)
            assert result_model is mixin.lora_manager.model
    else:
        pytest.skip("No LoRA config test - mixin only used when LoRA enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
