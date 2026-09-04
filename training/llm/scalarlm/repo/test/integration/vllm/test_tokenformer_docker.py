#!/usr/bin/env python3
"""
Integration test for tokenformer in Docker container
This test should ONLY be run inside the Docker container where vLLM is properly built
"""

import sys

import pytest
import torch

# Add paths for Docker container
sys.path.insert(0, "/app/cray/vllm")
sys.path.insert(0, "/app/cray/infra")


@pytest.mark.asyncio
async def test_tokenformer_manager_in_docker():
    """Test TokenformerModelManager works in Docker environment"""
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

    # Create a simple mock model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)

    model = SimpleModel()
    device = torch.device("cpu")
    manager = TokenformerModelManager(model=model, device=device)

    assert manager is not None
    assert manager.device == device
    assert hasattr(manager, "_registered_adapters")
    assert manager.model is not None


@pytest.mark.asyncio
async def test_cpu_model_runner_with_tokenformer():
    """Test CPUModelRunner can use tokenformer in Docker"""
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.config import (
        VllmConfig,
        ModelConfig,
        CacheConfig,
        SchedulerConfig,
        ParallelConfig,
        LoRAConfig,
        DeviceConfig,
        LoadConfig,
    )

    # Create minimal configs for testing
    model_config = ModelConfig(
        model="microsoft/DialoGPT-small",
        tokenizer="microsoft/DialoGPT-small",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=256,
    )

    cache_config = CacheConfig(
        block_size=16, gpu_memory_utilization=0.3, cache_dtype="auto"
    )

    scheduler_config = SchedulerConfig.default_factory(
        max_num_batched_tokens=256,
        max_num_seqs=128,
        max_model_len=256,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    parallel_config = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1)

    device_config = DeviceConfig(device="cpu")
    load_config = LoadConfig()

    # Create VllmConfig
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=device_config,
        load_config=load_config,
        speculative_config=None,
        lora_config=None,
    )

    # Test that CPUModelRunner can be created successfully
    runner = CPUModelRunner(vllm_config, torch.device("cpu"))
    assert runner is not None
    assert runner.device == torch.device("cpu")

    # Test with LoRA config
    lora_config = LoRAConfig(
        enable_tokenformer=True, max_lora_rank=8, max_loras=2, max_cpu_loras=2
    )

    vllm_config_with_lora = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=device_config,
        load_config=load_config,
        speculative_config=None,
        lora_config=lora_config,
    )

    runner_with_lora = CPUModelRunner(vllm_config_with_lora, torch.device("cpu"))
    assert runner_with_lora is not None
    assert vllm_config_with_lora.lora_config is not None
    print("✓ Docker tokenformer integration test setup successful")


@pytest.mark.asyncio
async def test_tokenformer_imports():
    """Test all tokenformer-related imports work in Docker"""
    # These imports should all work
    from vllm.tokenformer.tokenformer_model_manager import (
        TokenformerModelManager,
        TokenformerModel,
    )
    from vllm.tokenformer.tokenformer_surgeon import (
        TokenformerAdapter,
        TokenformerSurgeon,
    )
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

    assert TokenformerModelManager is not None
    assert TokenformerModel is not None
    assert TokenformerSurgeon is not None
    assert TokenformerAdapter is not None
    assert LoRAModelRunnerMixin is not None


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("🐳 Testing Tokenformer in Docker Container")
        print("=" * 50)

        print("1️⃣ Testing tokenformer imports...")
        try:
            await test_tokenformer_imports()
            print("   ✅ Imports: PASSED")
        except Exception as e:
            print(f"   ❌ Imports: FAILED - {e}")
            return False

        print("\n2️⃣ Testing TokenformerModelManager...")
        try:
            await test_tokenformer_manager_in_docker()
            print("   ✅ TokenformerModelManager: PASSED")
        except Exception as e:
            print(f"   ❌ TokenformerModelManager: FAILED - {e}")
            return False

        print("\n3️⃣ Testing Docker integration...")
        try:
            await test_cpu_model_runner_with_tokenformer()
            print("   ✅ Docker integration: PASSED")
        except Exception as e:
            print(f"   ❌ Docker integration: FAILED - {e}")
            return False

        print("\n" + "=" * 50)
        print("🎉 ALL TOKENFORMER TESTS PASSED!")
        return True

    result = asyncio.run(run_tests())
    sys.exit(0 if result else 1)
