import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(logits, targets):
    return F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="sum")


def chunked_causal_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, **kwargs):
    if ignore_index != -100:
        raise ValueError("Chunked causal loss supports ignore_index=-100")
    targets = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].reshape(-1)
    values = logits.reshape(-1, vocab_size)
    total = values.new_zeros((), dtype=torch.float32)
    for start in range(0, values.shape[0], 128):
        x, y = values[start:start + 128], targets[start:start + 128]
        if logits.requires_grad:
            total = total + checkpoint(_chunk_loss, x, y, use_reentrant=False)
        else:
            total = total + _chunk_loss(x, y)
    count = (targets != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
    count = torch.as_tensor(count, device=logits.device).clamp_min(1)
    return total / count
