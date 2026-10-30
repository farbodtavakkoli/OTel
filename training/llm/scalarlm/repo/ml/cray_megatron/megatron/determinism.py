import logging

import torch

logger = logging.getLogger(__name__)


def apply_seed(seed):
    if seed is None:
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(
        "Training RNG seeded with %d (deterministic LoRA init + corruption + SC mask).",
        seed,
    )
    return True
