"""The one-step graph has to be causal, differentiable and consistent.

test_asset_space_return_equals_residual_space_return is the one that would catch a wrong
transpose in the composition: holding w on the assets must earn exactly what holding
w_port on the residual portfolios earns.
"""

from __future__ import annotations

import torch

from afe.model.attention_pipeline import AttentionArb, residual_windows, to_pool

T, P, S, M, L = 60, 40, 25, 5, 8


def panel(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, S, M, generator=g)
    R = torch.randn(T, S, generator=g) * 0.02
    idx = torch.stack([torch.randperm(P, generator=g)[:S] for _ in range(T)])
    univ = torch.ones(T, S, dtype=torch.bool)
    return X, R, univ, idx


def test_windows_are_strictly_causal():
    """Window of date t must hold eps of t-L..t-1 for that company, and nothing of t."""
    eps_pool = torch.arange(T * P, dtype=torch.float32).reshape(T, P)
    valid = torch.ones(T, P, dtype=torch.bool)
    idx = torch.stack([torch.arange(S) for _ in range(T)])
    w, tradable = residual_windows(eps_pool, valid, idx, L, cumulative=False)
    for i in (0, 5, T - L - 1):
        t = L + i
        for s in (0, S - 1):
            expected = eps_pool[t - L:t, idx[t, s]]
            assert torch.equal(w[i, s], expected)
    assert tradable.all()


def test_a_gap_in_membership_kills_the_window():
    eps_pool = torch.randn(T, P)
    valid = torch.ones(T, P, dtype=torch.bool)
    idx = torch.stack([torch.arange(S) for _ in range(T)])
    valid[L + 3, 7] = False                       # company 7 absent on one date
    _, tradable = residual_windows(eps_pool, valid, idx, L, cumulative=False)
    affected = [i for i in range(T - L) if not tradable[i, 7]]
    # date L+3 is inside the window of dates L+4..L+3+L, and is itself untradable
    assert affected == list(range(3, L + 4))


def test_gradient_reaches_the_factor_parameters():
    """If it does not, the model is two-step with extra steps."""
    X, R, univ, idx = panel()
    model = AttentionArb(n_features=M, n_factors=4, embedding_dim=8, lookback=L)
    out = model.forward_span(X, R, univ, idx, P)
    out["w"].abs().sum().backward()
    for name, p in [("W_K", model.factors.W_K), ("Q", model.factors.Q)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_asset_space_return_equals_residual_space_return():
    """w'R == w_port'eps, i.e. the composition and its transpose agree."""
    X, R, univ, idx = panel(1)
    model = AttentionArb(n_features=M, n_factors=4, embedding_dim=8, lookback=L)
    model.eval()
    with torch.no_grad():
        out = model.forward_span(X, R, univ, idx, P, cumulative=False, scale=1.0)
        w, w_port = out["w"], out["w_port"]
        _, _, eps = model.factors(X, univ, R)
        lhs = (w * R[L:]).sum(1)
        rhs = (w_port * eps[L:]).sum(1) / out["l1"].clamp(min=1e-12)
    assert torch.allclose(lhs, rhs, atol=1e-5), (lhs[:3], rhs[:3])


def test_padding_never_gets_weight():
    X, R, univ, idx = panel(2)
    univ[:, S - 5:] = False
    model = AttentionArb(n_features=M, n_factors=4, embedding_dim=8, lookback=L)
    with torch.no_grad():
        out = model.forward_span(X, R, univ, idx, P)
    assert (out["w"][:, S - 5:] == 0).all()
    assert torch.isfinite(out["w"]).all()
