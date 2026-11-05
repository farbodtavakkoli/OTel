import torch


def prepare_training_precision(model, storage_dtype="float32"):
    compute_dtype = next(
        (p.dtype for p in model.parameters() if p.requires_grad and p.dtype == torch.bfloat16),
        None,
    )
    if storage_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"Unsupported training storage dtype: {storage_dtype}")
    if compute_dtype is not None:
        dtype = torch.float32 if storage_dtype == "float32" else torch.bfloat16
        for param in model.parameters():
            if param.requires_grad and param.is_floating_point():
                param.data = param.data.to(dtype)
    model._training_compute_dtype = compute_dtype
    return compute_dtype
