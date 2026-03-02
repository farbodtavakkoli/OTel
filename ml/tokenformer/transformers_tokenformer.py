import torch
from torch import nn
from typing import Optional, Tuple
from tokenformer.tokenformer_surgeon import TokenformerAttentionAdapter, TokenformerSurgeon
from tokenformer.tokenformer_surgeon import TokenformerMLPAdapter, get_hidden_size

import logging

logger = logging.getLogger(__name__)

# Change made here - This file was modified to ensure it will work with TokenFormer for attention layers of Gemma3 architecture

# class TransformersTokenformerAttentionAdapter(TokenformerAttentionAdapter):
#     def __init__(self, layer, hidden_size, device: torch.device):
#         self.is_sliding = layer.is_sliding ## Change made here
#         super().__init__(layer, hidden_size, device)

#     def forward(self,
#         hidden_states: torch.Tensor, ## Change made here
#         position_embeddings: torch.Tensor = None, ## Change made here
#         attention_mask: Optional[torch.Tensor] = None, ## Change made here
#         past_key_values = None, ## Change made here
#         cache_position: Optional[torch.LongTensor] = None, ## Change made here
#         **kwargs, ## Change made here
#     ) -> torch.Tensor:
#         base_layer_results = self.layer(
#                                         hidden_states=hidden_states, ## Change made here
#                                         position_embeddings=position_embeddings, ## Change made here
#                                         attention_mask=attention_mask, ## Change made here
#                                         past_key_values=past_key_values, ## Change made here)
#                                         cache_position=cache_position, ## Change made here
#                                         **kwargs
#         )
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.layer.head_dim)
#         query = self.layer.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         return super().forward(query, base_layer_results)

class TransformersTokenformerSurgeon(TokenformerSurgeon):

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        super().__init__(model, device)


    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     position_embeddings: torch.Tensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],

    def update_attn(self, name, layer):
        """Try to wrap the layer with a TokenformerAttentionAdaptor."""
        if not self._is_attn_layer(name):
            return

        logger.info(f"Wrapping layer {name} with TokenformerMLPAdapter")

        adapter = TokenformerMLPAdapter(layer, get_hidden_size(self.model.config), device=self.device) # Change made here
        adapter.is_sliding = layer.is_sliding # Change made here
        # Wrap the layer with a TokenformerAttentionAdapter
        self._recursive_setattr(
            self.model,
            name,
            adapter # Change made here
            # TokenformerMLPAdapter(
            #     layer, get_hidden_size(self.model.config), device=self.device
            #     #layer, self.model.config.hidden_size, device=self.device 
            # ),
        )

        #self._recursive_setattr(self.model, name, TransformersTokenformerAttentionAdapter(layer, layer.head_dim, self.device))

