import torch
from torch import nn
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable
from collections import OrderedDict
import copy
import logging

from ..common.adapter_commons import AdapterModel, AdapterModelManager, get_config_provider
from ..vllm.attention_adapter import patch_vllm_attention_layer, AttentionType

logger = logging.getLogger(__name__)


class TokenformerModel(AdapterModel):
    def __init__(self, tokenformers: Dict[str, torch.Tensor], model_id: Optional[str] = None):
        super().__init__(model_id or str(hash(str(tokenformers.keys()))))
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(cls, model_dir: str, device: torch.device) -> "TokenformerModel":
        files = list(Path(model_dir).glob("*.pt"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

        checkpoint_file = files[0]
        tokenformers = {}

        state_dict = torch.load(checkpoint_file, map_location=device)
        module_state_dict = state_dict.get('model_state_dict', state_dict)

        for module, tensor in module_state_dict.items():
            if any(key in module for key in ("tokenformer", "lm_head")):
                logger.info(f"Loading {module} from {checkpoint_file}")
                tokenformers[module] = tensor.to(device)

        return cls(tokenformers)


class ModelStateManager:
    def __init__(self, model):
        self.model = model
        self.dtype = next(self.model.parameters()).dtype
        self.orig_lm_head = self._extract_lm_head_weights()

    def _extract_lm_head_weights(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.clone().to(self.dtype)
            for k, v in self.model.state_dict().items()
            if "lm_head" in k
        }

    def apply_tokenformer_weights(self, tokenformers: Dict[str, torch.Tensor]) -> bool:
        try:
            model_state_dict = self.model.state_dict()

            for key, value in self.orig_lm_head.items():
                model_state_dict[key] = value

            for key, value in tokenformers.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                else:
                    logger.warning(f"Tokenformer weight {key} not found in model")

            load_result = self.model.load_state_dict(model_state_dict, strict=False)

            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys: {load_result.unexpected_keys}")

            return True

        except Exception as e:
            logger.error(f"Failed to apply tokenformer weights: {e}")
            return False

    def remove_tokenformer_weights(self, tokenformer_keys: List[str]) -> bool:
        try:
            model_state_dict = self.model.state_dict()

            for key in tokenformer_keys:
                if "tokenformer_p" in key and key in model_state_dict:
                    nn.init.zeros_(model_state_dict[key])
                elif "lm_head" in key and key in self.orig_lm_head:
                    model_state_dict[key] = self.orig_lm_head[key]

            self.model.load_state_dict(model_state_dict, strict=False)
            return True

        except Exception as e:
            logger.error(f"Failed to remove tokenformer weights: {e}")
            return False


class TokenformerManager(AdapterModelManager):
    def __init__(
        self, 
        model,
        device: torch.device,
        capacity: Optional[int] = None,
        supports_tokenformer_check: Optional[Callable] = None
    ):
        self.base_model = model
        self.device = device
        self.state_manager = ModelStateManager(model)

        config_provider = get_config_provider()
        self.capacity = capacity or config_provider.get_config_value("tokenformer_cache_capacity", 4)

        self.tokenformer_supported = self._check_tokenformer_support(supports_tokenformer_check)

        if self.tokenformer_supported:
            self.model = self._patch_model_for_tokenformer()
        else:
            self.model = model

        self._registered_adapters: Dict[str, TokenformerModel] = {}
        self._active_adapter: Optional[str] = None
        self._lru_adapter_ids: List[str] = []

    def _check_tokenformer_support(self, check_function: Optional[Callable]) -> bool:
        if check_function:
            return check_function(self.base_model)

        return hasattr(self.base_model, 'supports_tokenformer') or \
               any('tokenformer' in str(param) for param in self.base_model.named_parameters())

    def _patch_model_for_tokenformer(self):
        model = self.base_model
        config = {"enable_tokenformer": True}

        for name, module in model.named_modules():
            if self._is_attention_layer(name, module):
                logger.debug(f"Patching attention layer: {name}")
                patch_vllm_attention_layer(module, config)

        return model

    def _is_attention_layer(self, name: str, module) -> bool:
        attention_indicators = ['attn', 'attention', 'self_attention']
        return any(indicator in name.lower() for indicator in attention_indicators)

    def add_adapter(self, adapter: TokenformerModel) -> bool:
        if len(self._registered_adapters) >= self.capacity:
            if self._lru_adapter_ids:
                lru_adapter_id = self._lru_adapter_ids.pop(0)
                self.remove_adapter(lru_adapter_id)

        self._registered_adapters[adapter.id] = adapter
        self._lru_adapter_ids.append(adapter.id)

        logger.info(f"Added Tokenformer adapter: {adapter.id}")
        return True

    def activate_adapter(self, adapter_id: str) -> bool:
        if adapter_id not in self._registered_adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return False

        if adapter_id == self._active_adapter:
            logger.info(f"Adapter {adapter_id} is already active")
            return False

        self._update_lru_position(adapter_id)

        logger.info(f"Activating Tokenformer adapter: {adapter_id}")

        adapter = self._registered_adapters[adapter_id]
        success = self.state_manager.apply_tokenformer_weights(adapter.tokenformers)

        if success:
            self._active_adapter = adapter_id

        return success

    def deactivate_adapter(self, adapter_id: str) -> bool:
        if adapter_id not in self._registered_adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return False

        logger.info(f"Deactivating Tokenformer adapter: {adapter_id}")

        adapter = self._registered_adapters[adapter_id]
        success = self.state_manager.remove_tokenformer_weights(list(adapter.tokenformers.keys()))

        if success and self._active_adapter == adapter_id:
            self._active_adapter = None

        return success

    def remove_adapter(self, adapter_id: str) -> bool:
        if adapter_id not in self._registered_adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return False

        if adapter_id == self._active_adapter:
            self.deactivate_adapter(adapter_id)

        del self._registered_adapters[adapter_id]
        if adapter_id in self._lru_adapter_ids:
            self._lru_adapter_ids.remove(adapter_id)

        logger.info(f"Removed Tokenformer adapter: {adapter_id}")
        return True

    def _update_lru_position(self, adapter_id: str):
        if adapter_id in self._lru_adapter_ids:
            self._lru_adapter_ids.remove(adapter_id)
        self._lru_adapter_ids.append(adapter_id)

    def list_adapters(self) -> Dict[str, TokenformerModel]:
        return self._registered_adapters.copy()

    def get_active_adapter(self) -> Optional[str]:
        return self._active_adapter


def create_tokenformer_manager(
    model, 
    device: torch.device, 
    capacity: Optional[int] = None,
    supports_tokenformer_check: Optional[Callable] = None
) -> TokenformerManager:
    return TokenformerManager(model, device, capacity, supports_tokenformer_check)


def default_supports_tokenformer_check(model) -> bool:
    if hasattr(model, 'supports_tokenformer'):
        return getattr(model, 'supports_tokenformer', False)

    for name, _ in model.named_parameters():
        if 'tokenformer' in name.lower():
            return True

    return False