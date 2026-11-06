import re

import torch.nn as nn

ALL_LINEAR = "all-linear"

_SEPARATE_EXPERT_RE = re.compile(r"(?:^|\.)experts\.\d+\.")


def _is_multimodal_model(model) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "vision_config", None) is not None


def _language_decoder(model):
    if not _is_multimodal_model(model):
        return None
    if not hasattr(model, "get_decoder"):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    return decoder


def _module_prefix(model, target) -> str | None:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def _is_moe_model(model) -> bool:
    return any(
        ".experts" in name or ".block_sparse_moe" in name
        for name, _ in model.named_modules()
    )


def _ssm_mixer_prefixes(model) -> set[str]:
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        if ".mamba" in name or ".linear_attn" in name:
            continue
        if any(pname == "A_log" for pname, _ in module.named_parameters(recurse=False)):
            prefixes.add(name)
    return prefixes


def _is_ssm_linear(name: str, ssm_prefixes: set[str]) -> bool:
    if ".mamba" in name or ".linear_attn" in name:
        return True
    return any(name.startswith(prefix + ".") for prefix in ssm_prefixes)


def _has_ssm_layers(model) -> bool:
    return any(
        ".mamba" in name or ".linear_attn" in name
        for name, _ in model.named_modules()
    ) or bool(_ssm_mixer_prefixes(model))


def _has_separate_experts(model) -> bool:
    return any(
        isinstance(module, nn.Linear) and _SEPARATE_EXPERT_RE.search(name)
        for name, module in model.named_modules()
    )

def _moe_servable_linear_paths(model, output_embeddings, separate_experts=False) -> list[str]:
    ssm_prefixes = _ssm_mixer_prefixes(model)
    paths: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if module_name.endswith("lm_head"):
            continue
        if _is_ssm_linear(module_name, ssm_prefixes):
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in ("gate", "router") or leaf.endswith("_gate"):
            continue
        if leaf in ("kv_a_proj_with_mqa", "kv_b_proj"):
            continue
        if ".router." in module_name:
            continue
        if _SEPARATE_EXPERT_RE.search(module_name):
            if separate_experts:
                paths.append(module_name)
            continue
        if ".experts" in module_name:
            continue
        if ".block_sparse_moe" in module_name:
            continue
        paths.append(module_name)
    return paths


def resolve_target_parameters(model) -> list[str]:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "diffusion_gemma":
        return []
    leaves: set[str] = set()
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "experts":
            continue
        for pname, param in module.named_parameters(recurse=False):
            if param.dim() >= 2:
                leaves.add(pname)
    return sorted(leaves)


def resolve_target_modules(model, target_modules):
    if target_modules != ALL_LINEAR:
        return target_modules

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
                separate = _has_separate_experts(model)
                paths = _moe_servable_linear_paths(
                    model, output_embeddings, separate_experts=separate)
                scoped = sorted(p for p in paths if _under_decoder(p))
                if scoped:
                    return scoped
            else:
                return sorted(
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, nn.Linear)
                    and module is not output_embeddings
                    and _under_decoder(name)
                )

    if _is_moe_model(model):
        separate = _has_separate_experts(model)
        paths = _moe_servable_linear_paths(model, output_embeddings, separate_experts=separate)
        if paths:
            return sorted(paths)

    if _has_ssm_layers(model):
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

    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        names.add(module_name.split(".")[-1])
    names.discard("lm_head")

    return sorted(names)
