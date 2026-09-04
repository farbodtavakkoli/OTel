"""Resolve PEFT's "all-linear" target-modules shorthand ourselves."""

import re

import torch.nn as nn

# PEFT's sentinel for "adapt every linear layer except the output head".
ALL_LINEAR = "all-linear"

# A numbered per-expert projection (`...experts.0.w1`); grouped-expert MoEs have none.
_SEPARATE_EXPERT_RE = re.compile(r"(?:^|\.)experts\.\d+\.")


def _is_multimodal_model(model) -> bool:
    """True for HF multimodal wrappers, which nest a `vision_config` on their config."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "vision_config", None) is not None


def _language_decoder(model):
    """The text-decoder submodule to confine LoRA to, or None when no scoping is needed."""
    if not _is_multimodal_model(model):
        return None
    if not hasattr(model, "get_decoder"):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    return decoder


def _module_prefix(model, target) -> str | None:
    """The dotted name of `target` within `model` (by identity), or None if not found."""
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def _is_moe_model(model) -> bool:
    """True if the model has routed MoE expert submodules."""
    return any(
        ".experts" in name or ".block_sparse_moe" in name
        for name, _ in model.named_modules()
    )


def _ssm_mixer_prefixes(model) -> set[str]:
    """Dotted names of SSM mixer submodules not identifiable by path, found via `A_log`."""
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        if ".mamba" in name or ".linear_attn" in name:
            continue  # matched by name; no need for the signature fallback
        if any(pname == "A_log" for pname, _ in module.named_parameters(recurse=False)):
            prefixes.add(name)
    return prefixes


def _is_ssm_linear(name: str, ssm_prefixes: set[str]) -> bool:
    """True if the `nn.Linear` at `name` belongs to an SSM / linear-attention mixer."""
    if ".mamba" in name or ".linear_attn" in name:
        return True
    return any(name.startswith(prefix + ".") for prefix in ssm_prefixes)


def _has_ssm_layers(model) -> bool:
    """True if the model has SSM or linear-attention mixer submodules."""
    return any(
        ".mamba" in name or ".linear_attn" in name
        for name, _ in model.named_modules()
    ) or bool(_ssm_mixer_prefixes(model))


def _has_separate_experts(model) -> bool:
    """True if routed experts are per-expert `nn.Linear`s rather than grouped."""
    return any(
        isinstance(module, nn.Linear) and _SEPARATE_EXPERT_RE.search(name)
        for name, module in model.named_modules()
    )

def _moe_servable_linear_paths(model, output_embeddings, separate_experts=False) -> list[str]:
    """Full dotted paths of every servable `nn.Linear` in a MoE model."""
    ssm_prefixes = _ssm_mixer_prefixes(model)
    paths: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if module_name.endswith("lm_head"):  # head, when output_embeddings is None
            continue
        if _is_ssm_linear(module_name, ssm_prefixes):
            # SSM / linear-attention projections — not servable by the .pt LoRA path.
            continue
        # The MoE router / gate: adapting it would perturb expert selection.
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in ("gate", "router") or leaf.endswith("_gate"):
            continue
        # DeepSeek MLA latent projections: the serve-side kernel absorbs them.
        if leaf in ("kv_a_proj_with_mqa", "kv_b_proj"):
            continue
        # DiffusionGemma's router (`...router.proj`), which the leaf check above misses.
        if ".router." in module_name:
            continue
        # Per-expert projections are servable only in the separate-expert layout.
        if _SEPARATE_EXPERT_RE.search(module_name):
            if separate_experts:
                paths.append(module_name)
            continue
        if ".experts" in module_name:  # grouped/fused experts container — not .pt-serveable
            continue
        if ".block_sparse_moe" in module_name:  # GraniteMoe grouped experts + router.layer
            continue
        paths.append(module_name)
    return paths


def resolve_target_parameters(model) -> list[str]:
    """Leaf names of batched expert projections for `LoraConfig.target_parameters`."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "diffusion_gemma":
        return []
    leaves: set[str] = set()
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "experts":
            continue
        for pname, param in module.named_parameters(recurse=False):
            # A batched (num_experts, in, out) projection, not a router/norm scalar.
            if param.dim() >= 2:
                leaves.add(pname)
    return sorted(leaves)


def resolve_target_modules(model, target_modules):
    """Resolve the "all-linear" shorthand against the live `model`; pass anything else through."""
    if target_modules != ALL_LINEAR:
        return target_modules

    # Exclude the output projection by identity, so heads not named "lm_head" match.
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()

    decoder = _language_decoder(model)
    if decoder is not None:
        prefix = _module_prefix(model, decoder)
        if prefix is not None:
            def _under_decoder(name):
                return name == prefix or name.startswith(prefix + ".")
            if _is_moe_model(model):
                # Multimodal MoE: scope to the decoder AND apply the MoE-servable filter.
                separate = _has_separate_experts(model)
                paths = _moe_servable_linear_paths(
                    model, output_embeddings, separate_experts=separate)
                scoped = sorted(p for p in paths if _under_decoder(p))
                if scoped:
                    return scoped
                # Nothing matched under the decoder — fall through rather than adapt nothing.
            else:
                return sorted(
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, nn.Linear)
                    and module is not output_embeddings
                    and _under_decoder(name)
                )
        # get_decoder() returned a module we couldn't locate — fall through to the dense path.

    if _is_moe_model(model):
        # MoE: full paths for attention + dense MLP, plus separate-layout expert projections.
        separate = _has_separate_experts(model)
        paths = _moe_servable_linear_paths(model, output_embeddings, separate_experts=separate)
        if paths:
            return sorted(paths)
        # Nothing matched (unusual arch) — fall through to the dense path.

    if _has_ssm_layers(model):
        # Dense hybrid SSM: full paths for attention + MLP, excluding the SSM mixer subtree.
        ssm_prefixes = _ssm_mixer_prefixes(model)
        paths = sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and (output_embeddings is None or module is not output_embeddings)
            and not name.endswith("lm_head")
            and not _is_ssm_linear(name, ssm_prefixes)
        )
        if paths:
            return paths
        # Nothing outside the SSM mixer (pure Mamba) — fall through to the dense path.

    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        names.add(module_name.split(".")[-1])
    names.discard("lm_head")  # belt-and-suspenders when get_output_embeddings()==None

    return sorted(names)
