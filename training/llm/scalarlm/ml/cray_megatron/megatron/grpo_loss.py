from __future__ import annotations

import torch


def grpo_token_loss(logp_new, logp_old, advantages, clip_eps=0.2):
    if logp_old is None:
        # On-policy single pass: r == 1 with d r / d logp_new == r, i.e. REINFORCE with A.
        logp_old = logp_new.detach()
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    adv = advantages.to(ratio.dtype).unsqueeze(1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)
    # A token is "clipped" when the clamp is what limits the objective: the ratio left the window
    # on the side where moving further would have increased r*A (TRL's low/high clip definition).
    is_clipped = ((ratio < 1.0 - clip_eps) & (adv < 0)) | ((ratio > 1.0 + clip_eps) & (adv > 0))
    return per_token, ratio, is_clipped


def grpo_loss(logp_new, logp_old, advantages, mask, clip_eps=0.2, kl_beta=0.0, logp_ref=None):
    mask = mask.to(logp_new.dtype)
    per_token, ratio, is_clipped = grpo_token_loss(logp_new, logp_old, advantages, clip_eps)

    if kl_beta > 0.0:
        if logp_ref is None:
            raise ValueError("kl_beta > 0 needs logp_ref (reference-policy log-probs)")
        d = logp_ref - logp_new
        per_token = per_token + kl_beta * (torch.exp(d) - d - 1.0)

    tok_per_seq = mask.sum(dim=1).clamp_min(1.0)
    per_seq = (per_token * mask).sum(dim=1) / tok_per_seq
    loss = per_seq.mean()

    n_tok = mask.sum()
    stats = {
        "loss": float(loss.detach()),
        "mean_ratio": float((ratio.detach() * mask).sum() / n_tok.clamp_min(1.0)),
        "clip_frac": float((is_clipped.to(mask.dtype) * mask).sum() / n_tok.clamp_min(1.0)),
        "tokens": int(n_tok),
        "sequences": int(mask.shape[0]),
    }
    return loss, stats
