from typing import Dict, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ScalarLMModelRegistry:
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._adapters: Dict[str, Any] = {}

    def register_model(self, model_name: str, model_class: Type, adapter_config: Optional[dict] = None):
        self._models[model_name] = model_class
        if adapter_config:
            self._adapters[model_name] = adapter_config
        logger.info(f"Registered ScalarLM model: {model_name}")

    def get_model_class(self, model_name: str) -> Optional[Type]:
        return self._models.get(model_name)

    def get_adapter_config(self, model_name: str) -> Optional[dict]:
        return self._adapters.get(model_name)

    def list_models(self) -> list:
        return list(self._models.keys())

    def is_registered(self, model_name: str) -> bool:
        return model_name in self._models


_registry = ScalarLMModelRegistry()


def register_scalarlm_models():
    logger.info("Registering ScalarLM models...")

    from ..model.models import register_scalarlm_model_adapters
    register_scalarlm_model_adapters()

    from ..model.models import (
        create_scalarlm_llama_model,
        create_scalarlm_gemma_model, 
        create_scalarlm_qwen2_model
    )

    _registry.register_model(
        "scalarlm_llama_factory", 
        create_scalarlm_llama_model,
        {"supports_tokenformer": True, "type": "factory"}
    )

    _registry.register_model(
        "scalarlm_gemma_factory",
        create_scalarlm_gemma_model, 
        {"supports_tokenformer": True, "type": "factory"}
    )

    _registry.register_model(
        "scalarlm_qwen2_factory",
        create_scalarlm_qwen2_model,
        {"supports_tokenformer": True, "type": "factory"}
    )

    logger.info(f"Registered {len(_registry.list_models())} ScalarLM models")


def get_scalarlm_model_class(model_name: str) -> Optional[Type]:
    return _registry.get_model_class(model_name)


def get_registry() -> ScalarLMModelRegistry:
    return _registry