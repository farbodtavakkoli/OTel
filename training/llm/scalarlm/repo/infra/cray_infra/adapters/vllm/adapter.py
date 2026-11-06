import logging
from typing import Any, Dict, Optional, Callable, Type, Protocol
import torch
from contextlib import contextmanager

from ..common.adapter_commons import set_config_provider, ConfigProvider
from ..model.tokenformer import create_tokenformer_manager, default_supports_tokenformer_check
from .attention_adapter import patch_vllm_attention_layer

logger = logging.getLogger(__name__)


class VLLMModelProtocol(Protocol):
    def named_modules(self): ...
    def named_parameters(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict: dict, strict: bool = True): ...
    def parameters(self): ...


class ScalarLMConfigProvider(ConfigProvider):
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
        self._scalarlm_config_loaded = False

    def get_config_value(self, key: str, default: Any = None) -> Any:
        if key in self._config:
            return self._config[key]

        if not self._scalarlm_config_loaded:
            self._load_scalarlm_config()

        return self._config.get(key, default)

    def _load_scalarlm_config(self):
        try:
            from cray_infra.util.get_config import get_config
            scalarlm_config = get_config()

            self._config.update({
                "tokenformer_cache_capacity": getattr(scalarlm_config, 'tokenformer_cache_capacity', 4),
                "enable_tokenformer": getattr(scalarlm_config, 'enable_tokenformer', True),
                "auto_patch_attention": getattr(scalarlm_config, 'auto_patch_attention', True),
            })

            logger.info("Loaded ScalarLM configuration")

        except ImportError:
            logger.info("ScalarLM config not available, using defaults")
            self._config.update({
                "tokenformer_cache_capacity": 4,
                "enable_tokenformer": True,
                "auto_patch_attention": True,
            })

        self._scalarlm_config_loaded = True


class ScalarLMAdapter:
    def __init__(
        self,
        config_provider: Optional[ConfigProvider] = None,
        supports_tokenformer_check: Optional[Callable] = None
    ):
        self.config_provider = config_provider or ScalarLMConfigProvider()
        self.supports_tokenformer_check = supports_tokenformer_check or default_supports_tokenformer_check
        self._enhanced_models: Dict[str, Any] = {}

        set_config_provider(self.config_provider)

    def enhance_vllm_model(
        self, 
        model: VLLMModelProtocol, 
        device: torch.device,
        model_id: Optional[str] = None
    ) -> "EnhancedModelWrapper":
        model_id = model_id or f"model_{id(model)}"

        if model_id in self._enhanced_models:
            return self._enhanced_models[model_id]

        logger.info(f"Enhancing vLLM model {model_id} with ScalarLM functionality")

        wrapper = EnhancedModelWrapper(
            model=model,
            device=device,
            config_provider=self.config_provider,
            supports_tokenformer_check=self.supports_tokenformer_check
        )

        self._enhanced_models[model_id] = wrapper
        return wrapper

    def create_tokenformer_manager(self, model: VLLMModelProtocol, device: torch.device):
        return create_tokenformer_manager(
            model=model,
            device=device,
            supports_tokenformer_check=self.supports_tokenformer_check
        )


class EnhancedModelWrapper:
    def __init__(
        self,
        model: VLLMModelProtocol,
        device: torch.device, 
        config_provider: ConfigProvider,
        supports_tokenformer_check: Callable
    ):
        self.vllm_model = model
        self.device = device
        self.config_provider = config_provider
        self._tokenformer_manager = None
        self._attention_patched = False

        self.supports_tokenformer = supports_tokenformer_check(model)

        if self.supports_tokenformer:
            self._initialize_tokenformer()

    def _initialize_tokenformer(self):
        logger.info("Initializing tokenformer for enhanced model")

        self._tokenformer_manager = create_tokenformer_manager(
            model=self.vllm_model,
            device=self.device,
            capacity=self.config_provider.get_config_value("tokenformer_cache_capacity", 4),
            supports_tokenformer_check=lambda m: True
        )

        if self.config_provider.get_config_value("auto_patch_attention", True):
            self._patch_attention_layers()

    def _patch_attention_layers(self):
        if self._attention_patched:
            return

        config = {"enable_tokenformer": True}

        for name, module in self.vllm_model.named_modules():
            if self._is_attention_layer(name):
                logger.debug(f"Patching attention layer: {name}")
                patch_vllm_attention_layer(module, config)

        self._attention_patched = True

    def _is_attention_layer(self, name: str) -> bool:
        attention_indicators = ['attn', 'attention', 'self_attention']
        return any(indicator in name.lower() for indicator in attention_indicators)

    @property
    def tokenformer_manager(self):
        return self._tokenformer_manager

    def __getattr__(self, name):
        return getattr(self.vllm_model, name)


class AdapterManager:
    def __init__(self):
        self._adapter: Optional[ScalarLMAdapter] = None
        self._initialized = False

    def initialize(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_provider: Optional[ConfigProvider] = None
    ):
        if self._initialized:
            logger.info("ScalarLM integration already initialized")
            return

        logger.info("Initializing ScalarLM integration with clean architecture")

        if config_provider is None:
            config_provider = ScalarLMConfigProvider(config)

        self._adapter = ScalarLMAdapter(config_provider=config_provider)
        self._initialized = True

        logger.info("ScalarLM integration initialized successfully")

    def enhance_model(self, model, device: torch.device, model_id: Optional[str] = None):
        if not self._initialized:
            raise RuntimeError("ScalarLM integration not initialized. Call initialize() first.")

        return self._adapter.enhance_vllm_model(model, device, model_id)

    def create_tokenformer_manager(self, model, device: torch.device):
        if not self._initialized:
            raise RuntimeError("ScalarLM integration not initialized. Call initialize() first.")

        return self._adapter.create_tokenformer_manager(model, device)

    def is_initialized(self) -> bool:
        return self._initialized


_manager = AdapterManager()


def init_adapters(
    config: Optional[Dict[str, Any]] = None,
    config_provider: Optional[ConfigProvider] = None
):
    _manager.initialize(config, config_provider)


def enhance_vllm_model(model, device: torch.device, model_id: Optional[str] = None):
    if not _manager.is_initialized():
        init_adapters()
    return _manager.enhance_model(model, device, model_id)


def create_tokenformer_manager(model, device: torch.device):
    if not _manager.is_initialized():
        init_adapters()
    return _manager.create_tokenformer_manager(model, device)


@contextmanager
def enhanced_vllm(model, device: torch.device, **config):
    init_adapters(config)

    enhanced_model = enhance_vllm_model(model, device)

    try:
        yield enhanced_model
    finally:
        pass

scalarlm_enhanced_vllm = enhanced_vllm


def _get_manager() -> AdapterManager:
    return _manager
