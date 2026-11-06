from typing import Any, Optional, Protocol, TYPE_CHECKING
import torch
from enum import Enum
import logging

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    DECODER = "decoder"
    ENCODER = "encoder"
    ENCODER_DECODER = "encoder_decoder"


class AttentionMetadataAdapter:
    def __init__(self, vllm_attention_metadata: "AttentionMetadata"):
        self._vllm_metadata = vllm_attention_metadata

    def __getattr__(self, name):
        return getattr(self._vllm_metadata, name)

    def to_scalarlm_format(self) -> dict:
        return {
            "attention_type": AttentionType.DECODER,
            "vllm_metadata": self._vllm_metadata,
        }


class AttentionAdapterProtocol(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Any] = None,
        **kwargs
    ) -> torch.Tensor:
        ...


class VLLMAttentionAdapter:
    def __init__(self, vllm_attention_layer, scalarlm_config: Optional[dict] = None):
        self.vllm_layer = vllm_attention_layer
        self.config = scalarlm_config or {}
        self._tokenformer_enabled = self.config.get('enable_tokenformer', False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor, 
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Any] = None,
        attn_type: Optional[AttentionType] = None,
        **kwargs
    ) -> torch.Tensor:
        vllm_result = self.vllm_layer(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            **kwargs
        )

        if self._tokenformer_enabled:
            vllm_result = self._apply_tokenformer_processing(
                query, vllm_result, attn_type or AttentionType.DECODER
            )

        return vllm_result

    def _apply_tokenformer_processing(
        self, 
        query: torch.Tensor, 
        attention_output: torch.Tensor,
        attention_type: AttentionType
    ) -> torch.Tensor:
        if attention_type != AttentionType.DECODER:
            return attention_output

        seq_len = query.shape[0]

        if hasattr(self.vllm_layer, 'num_heads') and hasattr(self.vllm_layer, 'head_dim'):
            new_shape = [-1, self.vllm_layer.num_heads, seq_len, self.vllm_layer.head_dim]
            reshaped_query = torch.reshape(query, new_shape)
            reshaped_output = torch.reshape(attention_output, new_shape)

            processed_output = self._tokenformer_transform(reshaped_query, reshaped_output)

            return torch.reshape(processed_output, [-1, self.vllm_layer.num_heads * self.vllm_layer.head_dim])

        return attention_output

    def _tokenformer_transform(self, query: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output


def create_attention_adapter(vllm_attention_layer, config: Optional[dict] = None) -> VLLMAttentionAdapter:
    return VLLMAttentionAdapter(vllm_attention_layer, config)


def patch_vllm_attention_layer(layer, config: Optional[dict] = None):
    if hasattr(layer, '_scalarlm_adapted'):
        return layer

    original_forward = layer.forward
    adapter = create_attention_adapter(layer, config)

    def adapted_forward(*args, **kwargs):
        return adapter.forward(*args, **kwargs)

    layer.forward = adapted_forward
    layer._scalarlm_adapted = True
    layer._scalarlm_adapter = adapter

    return layer