from tokenformer.tokenformer_surgeon import TokenformerSurgeon, is_non_language_path

import time
import logging

logger = logging.getLogger(__name__)

def create_tokenformer_model(model, device, train_lm_head=None):
    overall_start = time.time()

    # Step 1: Insert adapter modules
    logger.info("Starting adapter module insertion...")
    step2_start = time.time()
    tokenformer_model = TokenformerSurgeon(model, device).insert_adapter_modules()

    step2_time = time.time() - step2_start
    logger.info(
        f"Adapter module insertion completed: {step2_time:.2f}s ({step2_time/60:.1f} minutes)"
    )

    # Step 2: Count parameters for train_lm_head decision
    if train_lm_head is None:
        # Models over 100M params don't need lm_head training.
        logger.info("Counting parameters...")
        step3_start = time.time()
        param_count = count_parameters(tokenformer_model)
        step3_time = time.time() - step3_start
        train_lm_head = param_count < 100_000_000
        logger.info(
            f"Parameter counting completed: {step3_time:.2f}s, count={param_count:,}, train_lm_head={train_lm_head}"
        )

    # Step 3: Freeze all parameters
    logger.info("Freezing all parameters...")
    step4_start = time.time()
    frozen_count = 0
    for param in tokenformer_model.parameters():
        param.requires_grad = False
        frozen_count += 1
    step4_time = time.time() - step4_start
    logger.info(
        f"Parameter freezing completed: {step4_time:.2f}s, frozen {frozen_count:,} parameters"
    )

    # Step 4: Unfreeze tokenformer parameters
    logger.info("Unfreezing tokenformer parameters...")
    step5_start = time.time()
    unfrozen_count = 0
    skipped_non_language = 0
    for name, param in tokenformer_model.named_parameters():
        # Without this guard the substring matches below fire inside vision/audio towers.
        if is_non_language_path(name):
            skipped_non_language += 1
            continue
        if any(
            module_name in name
            for module_name in [
                "tokenformer",
                "q_proj",
                "k_proj",
                "v_proj",
                "norm",
                "rotary_emb",
                "embed_tokens",
                "input_layernorm",
                "post_attention_layernorm",
                "o_proj",
            ]
        ):
            param.requires_grad = True
            unfrozen_count += 1
    if skipped_non_language:
        logger.info(
            f"Kept {skipped_non_language:,} non-language (vision/audio tower) "
            f"parameters frozen"
        )
    step5_time = time.time() - step5_start
    logger.info(
        f"Tokenformer parameter unfreezing completed: {step5_time:.2f}s, unfrozen {unfrozen_count:,} parameters"
    )

    # Step 5: Handle lm_head training. Accessed directly — it may be tied to the embeddings.
    step6_start = time.time()
    if train_lm_head:
        logger.info("Setting lm_head for training...")
        if hasattr(tokenformer_model, "lm_head") and hasattr(
            tokenformer_model.lm_head, "weight"
        ):
            tokenformer_model.lm_head.weight.requires_grad = True
            logger.info("lm_head weight set to trainable")
        else:
            logger.warning("lm_head or lm_head.weight not found")
    step6_time = time.time() - step6_start
    logger.info(f"lm_head handling completed: {step6_time:.2f}s")

    # Step 6: Log parameter gradients
    logger.info("Logging parameter gradients...")
    step7_start = time.time()
    log_param_gradients(tokenformer_model, logger)
    step7_time = time.time() - step7_start
    logger.info(f"Parameter gradient logging completed: {step7_time:.2f}s")

    total_time = time.time() - overall_start
    logger.info(
        f"create_tokenformer_model total time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    logger.info(
        f"Breakdown: adapter_insert={step2_time:.1f}s, param_count={step3_time:.1f}s, freeze={step4_time:.1f}s, unfreeze={step5_time:.1f}s, lm_head={step6_time:.1f}s, logging={step7_time:.1f}s"
    )

    return tokenformer_model


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def log_param_gradients(model, logger=logging.getLogger(__name__)):
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(
        f"Parameter summary: {trainable_count:,} trainable out of {total_count:,} total"
    )
