from __future__ import annotations

import glob
import logging
import os
import re
import time

import torch

from cray_infra.util.get_job_config import get_job_config

logger = logging.getLogger(__name__)

_CKPT = re.compile(r"checkpoint_(\d+)\.pt$")


def resolve_init_adapter_path(spec, job_directory):
    if spec.endswith(".pt") and os.path.isfile(spec):
        return spec
    candidates = [spec, os.path.join(os.path.dirname(job_directory.rstrip("/")), spec)]
    for d in candidates:
        if os.path.isdir(d):
            files = [p for p in glob.glob(os.path.join(d, "checkpoint_*.pt")) if _CKPT.search(p)]
            if not files:
                raise FileNotFoundError(f"init_adapter_from: no checkpoint_<step>.pt in {d}")
            return max(files, key=lambda p: int(_CKPT.search(p).group(1)))
    raise FileNotFoundError(
        f"init_adapter_from={spec!r}: not a checkpoint file, a directory, or a job hash under "
        f"{os.path.dirname(job_directory.rstrip('/'))}"
    )


def _candidates(model):
    out, module, seen = [], model, set()
    for _ in range(5):
        if module is None or id(module) in seen:
            break
        seen.add(id(module))
        out.append(module)
        module = getattr(module, "model", None)
    return out


def load_adapter_state_into(model, state_dict):
    keys = set(state_dict)
    best, best_hit = None, -1
    for cand in _candidates(model):
        hit = len(keys & set(cand.state_dict().keys()))
        if hit > best_hit:
            best, best_hit = cand, hit
    missing_in_model = sorted(keys - set(best.state_dict().keys())) if best is not None else sorted(keys)
    if best is None or missing_in_model:
        raise RuntimeError(
            f"init_adapter_from: {len(missing_in_model)} of {len(keys)} checkpoint tensors have no "
            f"parameter in the live model (adapter config differs?), e.g. {missing_in_model[:3]}"
        )
    res = best.load_state_dict(state_dict, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"init_adapter_from: unexpected keys {res.unexpected_keys[:3]}")
    return best_hit, type(best).__name__


def maybe_init_adapter_from(model):
    job_config = get_job_config()
    spec = job_config.get("init_adapter_from")
    if not spec:
        return None
    if job_config.get("adapter_type") == "none":
        raise ValueError("init_adapter_from requires an adapter (adapter_type lora/tokenformer)")

    t0 = time.time()
    path = resolve_init_adapter_path(spec, job_config["job_directory"])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    n_loaded, target = load_adapter_state_into(model, sd)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_nonzero_b = sum(1 for k, v in sd.items() if "lora_B" in k and bool(v.abs().sum() > 0))
    logger.info(
        "init_adapter_from: loaded %d/%d tensors from %s (step %s, metadata %s) into %s in %.1fs; "
        "live trainable tensors %d; lora_B non-zero in checkpoint %d",
        n_loaded, len(sd), path, ckpt.get("step"), ckpt.get("metadata"), target,
        time.time() - t0, n_trainable, n_nonzero_b,
    )
    if n_loaded != n_trainable:
        logger.warning(
            "init_adapter_from: checkpoint has %d tensors but the live adapter has %d trainable "
            "tensors — the adapter configs differ; continuing with the %d that matched",
            n_loaded, n_trainable, n_loaded,
        )
    return ckpt
