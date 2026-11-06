import torch
from torch import nn
from typing import Optional, Any, Dict, List, Type, ClassVar, Literal
import logging

from ..common.interfaces import SupportsTokenformer
from .tokenformer import TokenformerManager as TokenformerModelManager, create_tokenformer_manager

logger = logging.getLogger(__name__)


class ScalarLMModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scalarlm_initialized = False
        self._tokenformer_manager: Optional[TokenformerModelManager] = None

    def initialize_scalarlm_features(self, device: torch.device):
        if self._scalarlm_initialized:
            return

        logger.info("Initializing ScalarLM features")

        if hasattr(self, 'supports_tokenformer') and self.supports_tokenformer:
            self._tokenformer_manager = create_tokenformer_manager(self, device)
            logger.info("Tokenformer manager initialized")

        self._scalarlm_initialized = True

    def get_tokenformer_manager(self) -> Optional[TokenformerModelManager]:
        return self._tokenformer_manager


class ScalarLMTokenformerModel(ScalarLMModelMixin, SupportsTokenformer):
    supports_tokenformer: ClassVar[Literal[True]] = True

    def enable_tokenformer(self, config: dict) -> None:
        if not self._scalarlm_initialized:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda')
            self.initialize_scalarlm_features(device)

        if self._tokenformer_manager:
            logger.info("Enabling Tokenformer functionality")
        else:
            logger.warning("Tokenformer manager not available")

    def disable_tokenformer(self) -> None:
        if self._tokenformer_manager and self._tokenformer_manager.get_active_adapter():
            active_adapter = self._tokenformer_manager.get_active_adapter()
            self._tokenformer_manager.deactivate_adapter(active_adapter)
            logger.info("Tokenformer functionality disabled")


def create_scalarlm_model_adapter(base_model_class: Type) -> Type:
    class ScalarLMAdaptedModel(ScalarLMTokenformerModel, base_model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if hasattr(self, 'device'):
                device = self.device
            else:
                device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda')

            self.initialize_scalarlm_features(device)

    ScalarLMAdaptedModel.__name__ = f"ScalarLM{base_model_class.__name__}"
    ScalarLMAdaptedModel.__qualname__ = f"ScalarLM{base_model_class.__qualname__}"

    return ScalarLMAdaptedModel


def register_scalarlm_model_adapters():
    try:
        from vllm.model_executor.models.llama import LlamaForCausalLM
        from vllm.model_executor.models.gemma import GemmaForCausalLM
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

        from ..vllm.registry import get_registry

        registry = get_registry()

        models_to_adapt = [
            ("scalarlm_llama", LlamaForCausalLM),
            ("scalarlm_gemma", GemmaForCausalLM), 
            ("scalarlm_qwen2", Qwen2ForCausalLM),
        ]

        for model_name, base_class in models_to_adapt:
            try:
                adapted_class = create_scalarlm_model_adapter(base_class)
                registry.register_model(
                    model_name, 
                    adapted_class,
                    {"supports_tokenformer": True}
                )
                logger.info(f"Registered ScalarLM adapter: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to register {model_name}: {e}")

    except ImportError as e:
        logger.warning(f"Could not import vLLM models for adaptation: {e}")
    except Exception as e:
        logger.error(f"Error registering ScalarLM model adapters: {e}")


def create_scalarlm_llama_model(*args, **kwargs):
    from vllm.model_executor.models.llama import LlamaForCausalLM
    ScalarLMLlama = create_scalarlm_model_adapter(LlamaForCausalLM)
    return ScalarLMLlama(*args, **kwargs)


def create_scalarlm_gemma_model(*args, **kwargs):
    from vllm.model_executor.models.gemma import GemmaForCausalLM
    ScalarLMGemma = create_scalarlm_model_adapter(GemmaForCausalLM)
    return ScalarLMGemma(*args, **kwargs)


def create_scalarlm_qwen2_model(*args, **kwargs):
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    ScalarLMQwen2 = create_scalarlm_model_adapter(Qwen2ForCausalLM)
    return ScalarLMQwen2(*args, **kwargs)