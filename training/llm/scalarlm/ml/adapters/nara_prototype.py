"""NaRA (Noise-aware LoRA) prototype — NOT wired into the production training loop."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NaRAConfig:
    r: int = 16                    # LoRA rank (also the size of the r x r core matrix)
    lora_alpha: int = 32           # scaling = lora_alpha / r, as in standard LoRA
    lora_dropout: float = 0.0
    fnn_hidden_1: int = 256        # shared hypernetwork hidden sizes
    fnn_hidden_2: int = 512
    noise_embed_dim: int = 128     # Gaussian-Fourier embedding width for lambda
    c_scale: float = 0.1           # the paper's eta: Ceff = c_scale * C(lambda) + I
    fourier_scale: float = 16.0


class GaussianFourierProjection(nn.Module):
    """Fixed (non-learnable) random Fourier features for a scalar noise level."""

    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)  # (B, 1)
        proj = 2.0 * math.pi * x * self.W.to(x.dtype)  # (B, embed_dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (B, embed_dim)


class NaRAMapper(nn.Module):
    """The single shared hypernetwork: noise embedding -> flattened (r*r) core matrix."""

    def __init__(self, r: int, in_dim: int, h1: int, h2: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, r * r),
        )
        self._reset()

    def _reset(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
        nn.init.zeros_(linears[-1].weight)  # zero_last: C == 0 at init
        nn.init.zeros_(linears[-1].bias)

    def forward(self, noise_emb: torch.Tensor) -> torch.Tensor:
        b = noise_emb.shape[0]
        return self.net(noise_emb).view(b, self.r, self.r)  # (B, r, r)


class NaRAContext(nn.Module):
    """Owns the shared mapper + noise embedding and the current core matrix ``Ceff``."""

    def __init__(self, config: NaRAConfig):
        super().__init__()
        self.config = config
        self.embed = GaussianFourierProjection(config.noise_embed_dim, config.fourier_scale)
        self.mapper = NaRAMapper(
            config.r, config.noise_embed_dim, config.fnn_hidden_1, config.fnn_hidden_2
        )
        self.register_buffer("_eye", torch.eye(config.r), persistent=False)
        self.ceff: Optional[torch.Tensor] = None  # (B, r, r) or (r, r); set per step
        self.training_stage: int = 2

    def set_training_stage(self, stage: int):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 (A/B only) or 2 (noise-aware)")
        self.training_stage = stage
        req = stage == 2
        for p in self.mapper.parameters():
            p.requires_grad_(req)

    def set_noise_level(self, noise_level: Optional[Union[float, torch.Tensor]]):
        """Compute and cache ``Ceff = c_scale * C(lambda) + I`` for this step."""
        if self.training_stage == 1 or noise_level is None:
            self.ceff = self._eye  # identity => behaves exactly like plain LoRA
            return
        if not torch.is_tensor(noise_level):
            noise_level = torch.tensor([noise_level], dtype=torch.float32)
        # Match the mapper's dtype: the model may have been cast to bf16/fp16 after injection.
        mapper_dtype = next(self.mapper.parameters()).dtype
        noise_level = noise_level.to(self._eye.device, mapper_dtype)
        emb = self.embed(noise_level)                 # (B, embed_dim)
        c = self.mapper(emb)                           # (B, r, r)
        self.ceff = self.config.c_scale * c + self._eye  # residual: I at init


class NaRALinear(nn.Module):
    """Wraps a frozen base ``nn.Linear`` and adds the noise-aware low-rank branch."""

    def __init__(self, base: nn.Linear, context: NaRAContext, config: NaRAConfig):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # Weakref, so the shared context is not duplicated as a submodule of every layer.
        self._ctx_ref = weakref.ref(context)
        self.scaling = config.lora_alpha / config.r
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

        in_f, out_f, r = base.in_features, base.out_features, config.r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays zero

    @property
    def context(self) -> "NaRAContext":
        ctx = self._ctx_ref()
        if ctx is None:
            raise RuntimeError("NaRAContext was garbage-collected; keep it registered "
                               "under the model (inject_nara does this).")
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        ceff = self.context.ceff
        if ceff is None:
            ceff = self.context._eye
        h = F.linear(self.dropout(x), self.lora_A)  # (..., r)
        if ceff.dim() == 3:
            # per-example (B, r, r): x is (B, S, r) -> einsum over the rank dim
            h = torch.einsum("bsr,brk->bsk", h, ceff.to(h.dtype))
        else:
            h = h @ ceff.to(h.dtype)                # shared (r, r)
        delta = F.linear(h, self.lora_B) * self.scaling
        return out + delta


#: Attribute/submodule name the shared context is registered under on the model.
NARA_CONTEXT_ATTR = "nara_context"


def inject_nara(model: nn.Module, target_modules, config: NaRAConfig) -> NaRAContext:
    """Replace every targeted ``nn.Linear`` with a ``NaRALinear`` sharing one ``NaRAContext``."""
    targets = set(target_modules)
    full_targets = {t for t in targets if "." in t}
    leaf_targets = {t for t in targets if "." not in t}
    context = NaRAContext(config)
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name in full_targets or name.split(".")[-1] in leaf_targets
        ):
            to_replace.append((name, module))
    if not to_replace:
        raise ValueError(f"inject_nara: no nn.Linear matched targets {target_modules}")
    for name, module in to_replace:
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, NaRALinear(module, context, config))
    # Register the context inside the checkpointed sub-tree, or the mapper never reaches disk.
    checkpoint_root = getattr(model, "model", model)
    if not isinstance(checkpoint_root, nn.Module):
        checkpoint_root = model
    checkpoint_root.add_module(NARA_CONTEXT_ATTR, context)
    return context


def find_nara_context(model: nn.Module) -> Optional[NaRAContext]:
    """Return the model's NaRAContext through any wrapping, or None if NaRA is not active."""
    for m in model.modules():
        if isinstance(m, NaRAContext):
            return m
    return None


def mark_nara_trainable(model: nn.Module) -> int:
    """Freeze everything but the NaRA branch and shared mapper; return the trainable count."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name or f"{NARA_CONTEXT_ATTR}." in name:
            p.requires_grad_(True)
            n += 1
    return n
