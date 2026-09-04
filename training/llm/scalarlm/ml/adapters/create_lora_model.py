from cray_infra.util.get_job_config import get_job_config

import time
import logging

from peft import get_peft_model, LoraConfig, TaskType

from adapters.resolve_target_modules import (
    resolve_target_modules,
    resolve_target_parameters,
)

logger = logging.getLogger(__name__)


def _create_nara_model(model, device, job_config, nara_cfg, train_lm_head):
    """NaRA prototype adapter creation — the PEFT path's contract with NaRALinear layers."""
    from adapters.nara_prototype import (
        NaRAConfig,
        inject_nara,
        mark_nara_trainable,
    )

    start = time.time()
    lora_config = dict(job_config["lora_config"])
    target_modules = resolve_target_modules(
        model, lora_config.get("target_modules", "all-linear")
    )
    if not isinstance(target_modules, list):
        raise ValueError(
            "NaRA prototype requires an explicit target-module list "
            f"(got {target_modules!r}); resolve_target_modules should have expanded it."
        )

    cfg = NaRAConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.0),
        c_scale=nara_cfg.get("c_scale", 0.1),
        fnn_hidden_1=nara_cfg.get("fnn_hidden_1", 256),
        fnn_hidden_2=nara_cfg.get("fnn_hidden_2", 512),
        noise_embed_dim=nara_cfg.get("noise_embed_dim", 128),
        fourier_scale=nara_cfg.get("fourier_scale", 16.0),
    )
    logger.info(f"NaRA config: {cfg}; {len(target_modules)} target modules")

    inject_nara(model, target_modules, cfg)
    n_trainable = mark_nara_trainable(model)

    if train_lm_head and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        model.lm_head.weight.requires_grad = True
        logger.info("lm_head weight set to trainable")

    add_methods(model)  # unwrap_model -> filter_checkpoint saves requires_grad params
    model = model.to(device)
    logger.info(
        f"create_nara_model completed in {time.time() - start:.2f}s, "
        f"{n_trainable} trainable NaRA params"
    )
    return model


def _nara_config(job_config):
    """Return the NaRA sub-config when NaRA is enabled for this diffusion job, else None."""
    diffusion = job_config.get("diffusion") or {}
    nara = diffusion.nara if hasattr(diffusion, "nara") else diffusion.get("nara")
    if nara is None:
        return None
    nara = nara if isinstance(nara, dict) else nara.model_dump()
    return nara if nara.get("enabled") else None


def create_lora_model(model, device, train_lm_head=False):
    overall_start = time.time()

    job_config = get_job_config()

    # NaRA (noise-aware LoRA) prototype path, gated by diffusion.nara.enabled.
    nara_cfg = _nara_config(job_config)
    if nara_cfg is not None:
        return _create_nara_model(model, device, job_config, nara_cfg, train_lm_head)

    # Step 1: Insert LoRA adapter modules
    logger.info("Starting LoRA adapter module insertion...")
    step2_start = time.time()
    lora_config = dict(job_config["lora_config"])

    # Resolve PEFT's "all-linear" shorthand ourselves: its expansion fails on some MoEs.
    lora_config["target_modules"] = resolve_target_modules(
        model, lora_config.get("target_modules", "all-linear")
    )

    # Grouped-expert MoEs get no LoRA unless their batched params are in `target_parameters`.
    expert_parameters = resolve_target_parameters(model)
    if expert_parameters:
        existing = lora_config.get("target_parameters") or []
        lora_config["target_parameters"] = sorted(set(existing) | set(expert_parameters))

    logger.info(f"LoRA config: {lora_config}")

    lora_model = get_peft_model(model, LoraConfig(**lora_config))
    add_methods(lora_model)
    lora_model = lora_model.to(device)
    step2_time = time.time() - step2_start
    logger.info(
        f"LoRA adapter module insertion completed: {step2_time:.2f}s ({step2_time/60:.1f} minutes)"
    )

    # Step 2: Count parameters for train_lm_head decision
    if train_lm_head is None:
        # Models over 100M params don't need lm_head training.
        logger.info("Counting parameters...")
        step3_start = time.time()
        param_count = count_parameters(lora_model)
        step3_time = time.time() - step3_start
        train_lm_head = param_count < 100_000_000
        logger.info(
            f"Parameter counting completed: {step3_time:.2f}s, count={param_count:,}, train_lm_head={train_lm_head}"
        )
    else:
        step3_time = 0

    # Step 3: Freeze all parameters (get_peft_model already froze the base model).
    logger.info("Freezing all parameters...")
    step4_start = time.time()
    frozen_count = 0
    for param in lora_model.parameters():
        param.requires_grad = False
        frozen_count += 1
    step4_time = time.time() - step4_start
    logger.info(
        f"Parameter freezing completed: {step4_time:.2f}s, frozen {frozen_count:,} parameters"
    )

    # Step 4: Unfreeze LoRA parameters
    logger.info("Unfreezing LoRA parameters...")
    step5_start = time.time()
    unfrozen_count = 0
    for name, param in lora_model.named_parameters():
        if any(
            module_name in name
            for module_name in [
                "lora_",  # LoRA A/B weight matrices
            ]
        ):
            param.requires_grad = True
            unfrozen_count += 1
    step5_time = time.time() - step5_start
    logger.info(
        f"LoRA parameter unfreezing completed: {step5_time:.2f}s, unfrozen {unfrozen_count:,} parameters"
    )

    # Step 5: Handle lm_head training. Accessed directly — it may be tied to the embeddings.
    step6_start = time.time()
    if train_lm_head:
        logger.info("Setting lm_head for training...")
        base = (
            lora_model.base_model if hasattr(lora_model, "base_model") else lora_model
        )
        if hasattr(base, "lm_head") and hasattr(base.lm_head, "weight"):
            base.lm_head.weight.requires_grad = True
            logger.info("lm_head weight set to trainable")
        else:
            logger.warning("lm_head or lm_head.weight not found")
    step6_time = time.time() - step6_start
    logger.info(f"lm_head handling completed: {step6_time:.2f}s")

    # Step 6: Log parameter gradients
    logger.info("Logging parameter gradients...")
    step7_start = time.time()
    log_param_gradients(lora_model, logger)
    step7_time = time.time() - step7_start
    logger.info(f"Parameter gradient logging completed: {step7_time:.2f}s")

    total_time = time.time() - overall_start
    logger.info(
        f"create_lora_model total time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    logger.info(
        f"Breakdown: adapter_insert={step2_time:.1f}s, param_count={step3_time:.1f}s, "
        f"freeze={step4_time:.1f}s, unfreeze={step5_time:.1f}s, "
        f"lm_head={step6_time:.1f}s, logging={step7_time:.1f}s"
    )

    return lora_model


def unwrap_model(self):
    model = self.model

    if hasattr(model, "unwrap_model"):
        return model.unwrap_model()
    else:
        return filter_checkpoint(model, model.state_dict())


def filter_checkpoint(model, state_dict):
    # Remove the layers without gradients
    saved_params = {}

    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            logger.info(f"Saving parameter {name}")
            saved_params[name] = state_dict[name]

    return saved_params


def add_methods(model):
    model.unwrap_model = unwrap_model.__get__(model)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def log_param_gradients(model, logger=logging.getLogger(__name__)):
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(
        f"Parameter summary: {trainable_count:,} trainable out of {total_count:,} total"
    )
