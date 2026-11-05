import torch


def prepare_training_precision(model):
    compute_dtype = next(
        (p.dtype for p in model.parameters() if p.requires_grad and p.dtype == torch.bfloat16),
        None,
    )
    if compute_dtype is not None:
        for param in model.parameters():
            if param.requires_grad and param.is_floating_point():
                param.data = param.data.float()
    model._training_compute_dtype = compute_dtype
    return compute_dtype
