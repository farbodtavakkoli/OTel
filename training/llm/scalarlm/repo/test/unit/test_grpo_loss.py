"""CPU tests for the GRPO clipped surrogate (ml/cray_megatron/megatron/grpo_loss.py).

What is pinned:
  * inside the clip window the loss is the plain ratio*A surrogate and its gradient flows;
  * outside the window, on the side where the clamp binds, the gradient w.r.t. logp_new is ZERO
    (that is the whole point of PPO clipping) and the token is reported in clip_frac;
  * on the side where the clamp does not bind (ratio far off but moving against A) the
    unclipped term is the min and the gradient is NOT zero (pessimistic bound);
  * on-policy (logp_old=None or logp_old == logp_new.detach()) the gradient equals REINFORCE's
    -A * d logp;
  * masked (prompt/padding) tokens contribute nothing; per-sequence mean then mean over rows.
"""
import math

import pytest
import torch

from cray_megatron.megatron.grpo_loss import grpo_loss, grpo_token_loss

EPS = 0.2


def _one(logp_new, logp_old, adv):
    lp_new = torch.tensor([[logp_new]], requires_grad=True)
    lp_old = None if logp_old is None else torch.tensor([[logp_old]])
    loss, stats = grpo_loss(lp_new, lp_old, torch.tensor([adv]), torch.ones(1, 1), clip_eps=EPS)
    loss.backward()
    return loss.item(), lp_new.grad.item(), stats


def test_inside_window_is_plain_surrogate_with_gradient():
    # ratio = exp(0.1) ~ 1.105, inside [0.8, 1.2]
    loss, grad, stats = _one(-1.0, -1.1, adv=2.0)
    ratio = math.exp(0.1)
    assert loss == pytest.approx(-ratio * 2.0)
    assert grad == pytest.approx(-ratio * 2.0)  # d(-r*A)/d logp_new = -r*A
    assert stats["clip_frac"] == 0.0
    assert stats["mean_ratio"] == pytest.approx(ratio)


def test_positive_advantage_ratio_above_window_is_clipped_zero_grad():
    # ratio = exp(0.5) ~ 1.65 > 1.2 with A > 0 -> clamp binds -> loss = -(1+eps)*A, grad 0
    loss, grad, stats = _one(-0.5, -1.0, adv=1.0)
    assert loss == pytest.approx(-(1 + EPS) * 1.0)
    assert grad == 0.0
    assert stats["clip_frac"] == 1.0


def test_negative_advantage_ratio_below_window_is_clipped_zero_grad():
    # ratio = exp(-0.5) ~ 0.61 < 0.8 with A < 0 -> clamp binds -> loss = -(1-eps)*A, grad 0
    loss, grad, stats = _one(-1.5, -1.0, adv=-1.0)
    assert loss == pytest.approx((1 - EPS) * 1.0)
    assert grad == 0.0
    assert stats["clip_frac"] == 1.0


def test_pessimistic_side_is_not_clipped():
    # ratio ~ 1.65 but A < 0: r*A < clip(r)*A, so the unclipped term is the min -> gradient flows
    loss, grad, stats = _one(-0.5, -1.0, adv=-1.0)
    ratio = math.exp(0.5)
    assert loss == pytest.approx(ratio)
    assert grad == pytest.approx(ratio)  # d(-r*A)/d logp = -r*A = +r
    assert stats["clip_frac"] == 0.0


def test_on_policy_equals_reinforce():
    # No old log-probs: ratio == 1, gradient == -A (REINFORCE with a baseline-normalised A)
    loss, grad, stats = _one(-2.0, None, adv=0.7)
    assert loss == pytest.approx(-0.7)
    assert grad == pytest.approx(-0.7)
    assert stats["mean_ratio"] == pytest.approx(1.0)
    assert stats["clip_frac"] == 0.0


def test_mask_and_reduction():
    # Row 0: 2 completion tokens, row 1: 1 completion token + 1 masked token.
    lp_new = torch.tensor([[-1.0, -2.0], [-0.5, -9.0]], requires_grad=True)
    lp_old = lp_new.detach().clone()
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([1.0, -3.0])
    loss, stats = grpo_loss(lp_new, lp_old, adv, mask)
    # ratio 1 everywhere: per-seq loss = -A; mean over rows = -(1 + -3)/2 = 1.0
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert lp_new.grad[1, 1].item() == 0.0  # masked token gets no gradient
    assert lp_new.grad[0, 0].item() == pytest.approx(-1.0 / 2 / 2)  # -A / tokens_in_seq / rows
    assert stats["tokens"] == 3 and stats["sequences"] == 2


def test_token_loss_shapes_and_clip_indicator():
    lp_new = torch.zeros(2, 3)
    lp_old = torch.tensor([[0.0, 0.5, -0.5], [0.0, 0.5, -0.5]])  # ratios 1, 0.61, 1.65
    adv = torch.tensor([1.0, -1.0])
    per_tok, ratio, clipped = grpo_token_loss(lp_new, lp_old, adv, clip_eps=EPS)
    assert per_tok.shape == ratio.shape == clipped.shape == (2, 3)
    # A>0: only ratio>1.2 counts as clipped; A<0: only ratio<0.8
    assert clipped.tolist() == [[False, False, True], [False, True, False]]


def test_kl_requires_reference():
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 1), None, torch.ones(1), torch.ones(1, 1), kl_beta=0.1)
