"""The factor-competition experiment (model.factor_weights: compete), outside the thesis.

Expected values are worked out by hand from the definition in model/attention_factors.py:
memberships a = softmax over FACTORS, w_F[k] = a[k] / sum_s a[k, s]. The sharp limit is the
one that gives the experiment its meaning: the factors partition the universe with equal
weights, and the residual of a name becomes its return minus its group's mean return.
"""

from __future__ import annotations

import math

import pytest
import torch

from afe.model.attention_factors import AttentionFactors, weight_stats

M, D = 6, 8


def model(K, weights="compete", sinkhorn_iters=0, lambda_ridge=1e-4):
    torch.manual_seed(0)
    return AttentionFactors(M, K, D, lambda_ridge, weights=weights, sinkhorn_iters=sinkhorn_iters).double()


def masked(scores, tradable):
    keep = tradable.unsqueeze(1)
    return scores.masked_fill(~keep, torch.finfo(scores.dtype).min), keep


def test_two_factors_three_names_by_hand():
    # name 0 prefers factor 1 three to one, name 1 factor 0 three to one, name 2 is indifferent
    l3 = math.log(3)
    scores = torch.tensor([[[0.0, l3, 0.0], [l3, 0.0, 0.0]]], dtype=torch.float64)   # (1, K=2, S=3)
    s, keep = masked(scores, torch.ones(1, 3, dtype=torch.bool))
    m = model(2)
    a = m.memberships(s, keep)[0]
    assert torch.allclose(a, torch.tensor([[1 / 4, 3 / 4, 1 / 2], [3 / 4, 1 / 4, 1 / 2]], dtype=torch.float64))
    # factor 0 holds 1/4 + 3/4 + 1/2 = 3/2 of membership, so its weights are a / 1.5
    w = a / a.sum(-1, keepdim=True)
    assert torch.allclose(w, torch.tensor([[1 / 6, 1 / 2, 1 / 3], [1 / 2, 1 / 6, 1 / 3]], dtype=torch.float64))


def test_one_factor_is_the_equal_weighted_universe():
    """With K = 1 every name has membership 1, so the factor is 1/n on each tradable name."""
    m = model(1)
    X = torch.randn(3, 10, M, dtype=torch.float64)
    tradable = torch.arange(10)[None, :] < torch.tensor([10, 7, 4])[:, None]
    w = m.factor_weights(X, tradable)
    for t, n in enumerate([10, 7, 4]):
        assert torch.allclose(w[t, 0, :n], torch.full((n,), 1 / n, dtype=torch.float64))
        assert (w[t, 0, n:] == 0).all()


def test_sharp_scores_partition_and_demean_within_groups():
    """Scores far apart: each name sits in its best factor only, with weight 1/n_k there, and
    the projection residual (ridge ~0) is the name's return minus its group's mean."""
    group = torch.tensor([0, 0, 1, 2, 1, 0, 2, 2, 2, 1])
    K, S = 3, len(group)
    scores = torch.nn.functional.one_hot(group, K).T.unsqueeze(0).double() * 200.0     # (1, K, S)
    s, keep = masked(scores, torch.ones(1, S, dtype=torch.bool))
    m = model(K, lambda_ridge=1e-12)
    a = m.memberships(s, keep)
    w = a / a.sum(-1, keepdim=True)
    n_k = torch.bincount(group, minlength=K).double()
    expected = torch.nn.functional.one_hot(group, K).T.double() / n_k[:, None]
    assert torch.allclose(w[0], expected, atol=1e-12)
    R = torch.randn(1, S, dtype=torch.float64)
    betaT = m.loadings(w)
    eps = R - torch.einsum("tsk,tk->ts", betaT, torch.einsum("tks,ts->tk", w, R))
    group_mean = torch.stack([R[0, group == k].mean() for k in range(K)])[group]
    assert torch.allclose(eps[0], R[0] - group_mean, atol=1e-8)


def test_memberships_sum_to_one_per_name_and_padding_is_empty():
    m = model(5)
    T, S = 4, 30
    X = torch.randn(T, S, M, dtype=torch.float64)
    tradable = torch.arange(S)[None, :] < torch.tensor([30, 20, 11, 1])[:, None]
    X = torch.where(tradable.unsqueeze(-1), X, torch.full_like(X, float("nan")))
    w = m.factor_weights(X, tradable)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(T, 5, dtype=torch.float64))
    assert (w[~tradable.unsqueeze(1).expand_as(w)] == 0).all()


@pytest.mark.parametrize("iters", [0, 3])
def test_dead_date_gives_zeros_and_a_finite_gradient(iters):
    m = model(4, sinkhorn_iters=iters)
    X = torch.randn(3, 12, M, dtype=torch.float64)
    tradable = torch.ones(3, 12, dtype=torch.bool)
    tradable[1] = False
    tradable[2, 7:] = False
    w = m.factor_weights(X, tradable)
    assert torch.isfinite(w).all() and (w[1] == 0).all()
    (w * torch.randn_like(w)).sum().backward()
    assert torch.isfinite(m.Q.grad).all() and torch.isfinite(m.W_K.grad).all()


def test_sinkhorn_balances_the_factors():
    """After enough iterations each factor holds n/K of membership and each name still 1."""
    K, S, n = 4, 40, 36
    scores = torch.randn(2, K, S, dtype=torch.float64) * 3
    tradable = torch.arange(S)[None, :] < n
    tradable = tradable.expand(2, S)
    s, keep = masked(scores, tradable)
    a = model(K, sinkhorn_iters=200).memberships(s, keep)
    assert torch.allclose(a.sum(dim=1)[tradable], torch.ones(2 * n, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(a.sum(dim=-1), torch.full((2, K), n / K, dtype=torch.float64), atol=1e-6)
    plain = model(K).memberships(s, keep).sum(dim=-1)
    assert (plain - n / K).abs().max() > 0.5      # the scores above do not balance on their own


def test_softmax_default_is_unchanged_and_state_dicts_match():
    torch.manual_seed(0)
    a = AttentionFactors(M, 5, D)
    torch.manual_seed(0)
    b = AttentionFactors(M, 5, D, weights="softmax")
    X = torch.randn(2, 9, M)
    tradable = torch.ones(2, 9, dtype=torch.bool)
    assert torch.equal(a.factor_weights(X, tradable), b.factor_weights(X, tradable))
    assert a.state_dict().keys() == model(5).state_dict().keys()   # no new parameters


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        AttentionFactors(M, 3, D, weights="sparsemax")
    with pytest.raises(ValueError):
        AttentionFactors(M, 3, D, sinkhorn_iters=5)


def test_weight_stats_by_hand():
    # two disjoint factors, two names each at 1/2: 2 effective names, no overlap, G = I/2
    w = torch.tensor([[[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]], dtype=torch.float64)
    tradable = torch.ones(1, 4, dtype=torch.bool)
    st = weight_stats(w, tradable, lambda_ridge=0.0, top=1)
    assert st["eff_names"] == pytest.approx(2.0) and st["top_share"] == pytest.approx(0.5)
    assert st["overlap"] == pytest.approx(0.0) and st["eff_factors"] == pytest.approx(2.0)
    # two copies of the same factor: overlap 1, the loading solve keeps one factor
    same = torch.tensor([[[0.25] * 4, [0.25] * 4]], dtype=torch.float64)
    st = weight_stats(same, tradable, lambda_ridge=1e-12)
    assert st["overlap"] == pytest.approx(1.0) and st["eff_factors"] == pytest.approx(1.0)
    assert st["names"] == 4
