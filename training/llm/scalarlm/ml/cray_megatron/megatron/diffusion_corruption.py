"""Live canvas corruption for DiffusionGemma training."""

import torch


def corrupt_canvas(
    canvas_input_ids,
    canvas_labels,
    vocab_size,
    eps,
    generator=None,
    protect_prefix=0,
    return_noise_level=False,
):
    """Uniform-state discrete-diffusion corruption of a clean canvas."""
    device = canvas_input_ids.device
    batch_sz, canvas_len = canvas_input_ids.shape

    t = torch.rand(batch_sz, 1, device=device, generator=generator) * (1.0 - eps) + eps
    supervised = canvas_labels != -100
    if protect_prefix > 0:
        # Drop the prefix from the supervised (thus corruptible) set.
        supervised = supervised.clone()
        supervised[:, :protect_prefix] = False
    corrupt_mask = (
        torch.rand(batch_sz, canvas_len, device=device, generator=generator) < t
    ) & supervised
    random_tokens = torch.randint(
        0,
        vocab_size,
        (batch_sz, canvas_len),
        device=device,
        dtype=canvas_input_ids.dtype,
        generator=generator,
    )
    decoder_input_ids = torch.where(corrupt_mask, random_tokens, canvas_input_ids)
    if return_noise_level:
        return decoder_input_ids, t
    return decoder_input_ids
