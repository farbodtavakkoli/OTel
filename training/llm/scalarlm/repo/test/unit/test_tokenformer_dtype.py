"""Regression tests for the tokenformer adapter dtype fix."""

import torch
from torch import nn

from tokenformer.tokenformer_surgeon import TokenformerAdapter


def test_resolve_dtype_bf16_layer():
    layer = nn.Linear(8, 8).to(torch.bfloat16)
    assert TokenformerAdapter._resolve_param_dtype(layer) == torch.bfloat16


def test_resolve_dtype_fp32_layer():
    layer = nn.Linear(8, 8)  # default float32
    assert TokenformerAdapter._resolve_param_dtype(layer) == torch.float32


def test_resolve_dtype_no_float_params_falls_back():
    # Integer-only module (stands in for packed/quantized weights): fall back
    # to the global default.
    class IntOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(
                torch.zeros(4, dtype=torch.int8), requires_grad=False
            )

    assert TokenformerAdapter._resolve_param_dtype(IntOnly()) == torch.float32


def test_resolve_dtype_none_layer_falls_back():
    assert TokenformerAdapter._resolve_param_dtype(None) == torch.float32


def test_resolve_dtype_first_float_param_wins_over_later_int():
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
            self.i = nn.Parameter(
                torch.zeros(2, dtype=torch.int64), requires_grad=False
            )

    assert TokenformerAdapter._resolve_param_dtype(Mixed()) == torch.bfloat16
