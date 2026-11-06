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
    loss, grad, stats = _one(-1.0, -1.1, adv=2.0)
    ratio = math.exp(0.1)
    assert loss == pytest.approx(-ratio * 2.0)
    assert grad == pytest.approx(-ratio * 2.0)
    assert stats["clip_frac"] == 0.0
    assert stats["mean_ratio"] == pytest.approx(ratio)


def test_positive_advantage_ratio_above_window_is_clipped_zero_grad():
    loss, grad, stats = _one(-0.5, -1.0, adv=1.0)
    assert loss == pytest.approx(-(1 + EPS) * 1.0)
    assert grad == 0.0
    assert stats["clip_frac"] == 1.0


def test_negative_advantage_ratio_below_window_is_clipped_zero_grad():
    loss, grad, stats = _one(-1.5, -1.0, adv=-1.0)
    assert loss == pytest.approx((1 - EPS) * 1.0)
    assert grad == 0.0
    assert stats["clip_frac"] == 1.0


def test_pessimistic_side_is_not_clipped():
    loss, grad, stats = _one(-0.5, -1.0, adv=-1.0)
    ratio = math.exp(0.5)
    assert loss == pytest.approx(ratio)
    assert grad == pytest.approx(ratio)
    assert stats["clip_frac"] == 0.0


def test_on_policy_equals_reinforce():
    loss, grad, stats = _one(-2.0, None, adv=0.7)
    assert loss == pytest.approx(-0.7)
    assert grad == pytest.approx(-0.7)
    assert stats["mean_ratio"] == pytest.approx(1.0)
    assert stats["clip_frac"] == 0.0


def test_mask_and_reduction():
    lp_new = torch.tensor([[-1.0, -2.0], [-0.5, -9.0]], requires_grad=True)
    lp_old = lp_new.detach().clone()
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([1.0, -3.0])
    loss, stats = grpo_loss(lp_new, lp_old, adv, mask)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert lp_new.grad[1, 1].item() == 0.0
    assert lp_new.grad[0, 0].item() == pytest.approx(-1.0 / 2 / 2)
    assert stats["tokens"] == 3 and stats["sequences"] == 2


def test_token_loss_shapes_and_clip_indicator():
    lp_new = torch.zeros(2, 3)
    lp_old = torch.tensor([[0.0, 0.5, -0.5], [0.0, 0.5, -0.5]])
    adv = torch.tensor([1.0, -1.0])
    per_tok, ratio, clipped = grpo_token_loss(lp_new, lp_old, adv, clip_eps=EPS)
    assert per_tok.shape == ratio.shape == clipped.shape == (2, 3)
    assert clipped.tolist() == [[False, False, True], [False, True, False]]


def test_kl_requires_reference():
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 1), None, torch.ones(1), torch.ones(1, 1), kl_beta=0.1)
