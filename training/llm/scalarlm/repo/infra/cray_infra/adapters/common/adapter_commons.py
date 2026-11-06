from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AdapterModel(ABC):
    def __init__(self, adapter_id: str):
        self.id = adapter_id


class AdapterModelManager(ABC):
    @abstractmethod
    def add_adapter(self, adapter: AdapterModel) -> bool:
        pass

    @abstractmethod
    def remove_adapter(self, adapter_id: str) -> bool:
        pass

    @abstractmethod
    def activate_adapter(self, adapter_id: str) -> bool:
        pass

    @abstractmethod
    def deactivate_adapter(self, adapter_id: str) -> bool:
        pass


def get_adapter(adapter_id: str, adapters: Dict[str, Any]) -> Optional[Any]:
    return adapters.get(adapter_id)


def list_adapters(adapters: Dict[str, Any]) -> Dict[str, Any]:
    return adapters.copy()


def remove_adapter(adapter_id: str, adapters: Dict[str, Any], remove_fn) -> bool:
    if adapter_id in adapters:
        remove_fn(adapter_id)
        return True
    return False


def deactivate_adapter(adapter_id: str, active_adapter_ref: list) -> bool:
    if active_adapter_ref and active_adapter_ref[0] == adapter_id:
        active_adapter_ref[0] = None
        return True
    return False


class ConfigProvider(Protocol):
    def get_config_value(self, key: str, default: Any = None) -> Any:
        ...


class DefaultConfigProvider:
    def __init__(self):
        self._config_cache = {}
        self._loaded = False

    def get_config_value(self, key: str, default: Any = None) -> Any:
        if not self._loaded:
            self._load_scalarlm_config()

        return self._config_cache.get(key, default)

    def _load_scalarlm_config(self):
        try:
            from cray_infra.util.get_config import get_config
            config = get_config()

            self._config_cache = {
                "tokenformer_cache_capacity": getattr(config, 'tokenformer_cache_capacity', 4),
                "enable_tokenformer": getattr(config, 'enable_tokenformer', True),
            }
            self._loaded = True

        except ImportError:
            logger.warning("ScalarLM config not available, using defaults")
            self._config_cache = {
                "tokenformer_cache_capacity": 4,
                "enable_tokenformer": True,
            }
            self._loaded = True


_config_provider: Optional[ConfigProvider] = None


def set_config_provider(provider: ConfigProvider):
    global _config_provider
    _config_provider = provider


def get_config_provider() -> ConfigProvider:
    global _config_provider
    if _config_provider is None:
        _config_provider = DefaultConfigProvider()
    return _config_provider