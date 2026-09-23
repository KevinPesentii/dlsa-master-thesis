"""The cross-sectional softmax must see the universe of the date and nothing else.

The test that matters is the last one: running the masked model on a padded (T, S)
array has to give the same factor weights as running it on the (T, n_t) array that
holds only the names present. If that holds, the padding is invisible to the model.
"""

from __future__ import annotations

import math

import pytest
import torch

from afe.model.attention_factors import AttentionFactors

S, M, K, D, T = 500, 8, 6, 16, 12


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = AttentionFactors(n_features=M, n_factors=K, embedding_dim=D)
    X = torch.randn(T, S, M)
    R = torch.randn(T, S) * 0.02
    n = torch.randint(50, S, (T,))                       # a different universe each date
    tradable = torch.arange(S)[None, :] < n[:, None]
    X = torch.where(tradable.unsqueeze(-1), X, torch.full_like(X, float("nan")))
    R = torch.where(tradable, R, torch.full_like(R, float("nan")))
    return model, X, tradable, R


def test_padding_gets_no_weight(setup):
    model, X, tradable, R = setup
    w_F, betaT, eps = model(X, tradable, torch.nan_to_num(R))
    assert torch.isfinite(w_F).all(), "NaN in the padding leaked through the embedding"
    assert (w_F[~tradable.unsqueeze(1).expand_as(w_F)] == 0).all()
    assert (betaT[~tradable] == 0).all()
    assert (eps[~tradable] == 0).all()


def test_rows_sum_to_one_over_the_universe(setup):
    model, X, tradable, R = setup
    w_F = model.factor_weights(X, tradable)
    assert torch.allclose(w_F.sum(-1), torch.ones(T, K), atol=1e-5)


def test_dead_date_does_not_produce_nan(setup):
    """A date with no tradable name must give zeros and a finite gradient, not NaN."""
    model, X, tradable, R = setup
    dead = tradable.clone()
    dead[0] = False
    w_F = model.factor_weights(X, dead)
    assert torch.isfinite(w_F).all()
    assert (w_F[0] == 0).all()
    w_F.sum().backward()
    assert torch.isfinite(model.Q.grad).all()
    model.zero_grad()


def test_masking_equals_running_on_the_sub_universe(setup):
    """Padding must be invisible: same weights as a model that never saw the empty slots."""
    model, X, tradable, _ = setup
    padded = model.factor_weights(X, tradable)
    for t in range(T):
        n = int(tradable[t].sum())
        sub = X[t:t + 1, :n, :]
        alive = torch.ones(1, n, dtype=torch.bool)
        assert torch.allclose(model.factor_weights(sub, alive)[0], padded[t, :, :n], atol=1e-6)


def test_zeroing_after_the_softmax_is_wrong(setup):
    """The bug this module exists to prevent, quantified."""
    model, X, tradable, _ = setup
    m = tradable.unsqueeze(-1)
    Xz = torch.where(m, X, torch.zeros_like(X))
    scores = torch.einsum("kd,tsd->tks", model.Q, Xz @ model.W_K) / math.sqrt(model.d)
    naive = torch.softmax(scores, dim=-1) * tradable.unsqueeze(1)     # mask AFTER: wrong
    leaked = 1.0 - naive.sum(-1)
    assert leaked.mean() > 0.05, "the fixture is too easy to show the bug"
    assert not torch.allclose(naive.sum(-1), torch.ones(T, K), atol=1e-3)


def test_a_level_shared_by_all_names_does_not_move_the_weights_in_float32():
    """The cross-sectional medians are the same for every name on a date and are raw
    levels: med_Vol reaches 3.4e7 in 2008-2011. In exact arithmetic such a column cancels
    in the softmax over names; in float32 its offset in X @ W_K rounds the +-0.5 rank
    signal away unless X is centred per date. Weights sharpened x5, nearer a trained model."""
    torch.manual_seed(1)
    model = AttentionFactors(n_features=M, n_factors=K, embedding_dim=D)
    with torch.no_grad():
        model.W_K.mul_(5.0)
        model.Q.mul_(5.0)
    ranks = torch.rand(T, S, M - 1) - 0.5
    level = torch.linspace(1.0e7, 3.4e7, T).view(T, 1, 1).expand(T, S, 1)
    tradable = torch.arange(S)[None, :] < torch.randint(50, S, (T,))[:, None]
    with_level = model.factor_weights(torch.cat([ranks, level], -1), tradable)
    without = model.factor_weights(torch.cat([ranks, torch.zeros_like(level)], -1), tradable)
    assert (with_level - without).abs().sum(-1).max() < 1e-4          # L1 per factor and date
    exact = model.double().factor_weights(torch.cat([ranks, level], -1).double(), tradable)
    assert (with_level.double() - exact).abs().sum(-1).max() < 1e-4
