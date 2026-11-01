"""policy.residual_window "fixed": the window of trade date t re-hedged with date t's model.

Definition (the spec, not the code): for the company i in slot s on t and tau = t-L..t-1,

    eps~_{i,tau} = R_{i,tau} - beta_{i,t}' (w_F,t R_tau)

with w_F,t, beta_t built from X row t (known at the close of t-1) and R_tau the returns on
tau of date t's members. Tradable: in the universe at t and a return on all L window days.
"""

from __future__ import annotations

import pytest
import torch

from afe.model.attention_pipeline import AttentionArb

T, P, S, M, L, K = 50, 30, 20, 4, 6, 3


def model():
    torch.manual_seed(0)
    m = AttentionArb(n_features=M, n_factors=K, embedding_dim=8, lookback=L)
    return m.eval()


def churning_panel(seed=0):
    """Slots reshuffled every day and some returns missing, as in the loader:
    in_universe = member and a return on the day, R 0 where there is none."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, S, M, generator=g)
    has_ret = torch.rand(T, P, generator=g) > 0.05
    R_pool = torch.randn(T, P, generator=g) * 0.02 * has_ret
    idx = torch.stack([torch.randperm(P, generator=g)[:S] for _ in range(T)])
    idx[:, -2:] = P                                    # two padding slots
    pad = torch.cat([has_ret, torch.zeros(T, 1, dtype=torch.bool)], 1)
    univ = pad.gather(1, idx)
    R = torch.cat([R_pool, torch.zeros(T, 1)], 1).gather(1, idx)
    return X, R, univ, idx, R_pool, has_ret


def run(m, X, R, univ, idx, R_pool, has_ret, window, **kw):
    with torch.no_grad():
        return m.forward_span(X, R, univ, idx, P, residual_window=window, R_pool=R_pool,
                              has_ret=has_ret, **kw)


def test_fixed_window_is_the_formula():
    """Every entry of the window by hand, one (t, s) at a time, with dense matrices."""
    m = model()
    X, R, univ, idx, R_pool, has_ret = churning_panel()
    out = run(m, X, R, univ, idx, R_pool, has_ret, "fixed", cumulative=False, scale=1.0)
    with torch.no_grad():
        w_F, betaT, _ = m.factors(X, univ, R)
    R_ext = torch.cat([R_pool, torch.zeros(T, 1)], 1)
    ok_ext = torch.cat([has_ret, torch.zeros(T, 1, dtype=torch.bool)], 1)
    n_rolling_blocked = 0
    for t in range(L, T):
        for s in range(S):
            c = int(idx[t, s])
            expect_ok = bool(univ[t, s]) and bool(ok_ext[t - L:t, c].all())
            assert bool(out["tradable"][t - L, s]) == expect_ok, (t, s)
            if not expect_ok:
                assert (out["windows"][t - L, s] == 0).all()
                continue
            member_all = all(bool(univ[tau][idx[tau] == c].any()) for tau in range(t - L, t))
            n_rolling_blocked += not member_all
            for j, tau in enumerate(range(t - L, t)):
                r_members = R_ext[tau, idx[t]]                       # date t's slots, on tau
                expect = R_ext[tau, c] - betaT[t, s] @ (w_F[t] @ r_members)
                assert torch.isclose(out["windows"][t - L, s, j], expect, atol=1e-6), (t, s, tau)
    # slots reshuffle daily, so most tradable names were not members on every window day:
    # under fixed they still trade (a return is enough), which is the case being tested
    assert n_rolling_blocked > 0


@pytest.mark.parametrize("window", ["rolling", "fixed"])
def test_window_and_weights_of_t_use_nothing_after_t_minus_1(window):
    """Prefix invariance: X rows after t (X row t holds the close of t-1) and returns from t
    on can be anything; the window and the weights of trade date t do not move."""
    m = model()
    X, R, univ, idx, R_pool, has_ret = churning_panel(1)
    base = run(m, X, R, univ, idx, R_pool, has_ret, window, normalise="window_vol", scale=2.0)
    g = torch.Generator().manual_seed(9)
    for t in (L, L + 7, T - 1):
        X2, R2, R_pool2, has2, univ2 = X.clone(), R.clone(), R_pool.clone(), has_ret.clone(), univ.clone()
        X2[t + 1:] = torch.randn(X2[t + 1:].shape, generator=g) * 50
        R2[t:] = torch.randn(R2[t:].shape, generator=g)
        R_pool2[t:] = torch.randn(R_pool2[t:].shape, generator=g)
        has2[t:] = torch.rand(has2[t:].shape, generator=g) > 0.5
        univ2[t + 1:] = torch.rand(univ2[t + 1:].shape, generator=g) > 0.5
        out = run(m, X2, R2, univ2, idx, R_pool2, has2, window, normalise="window_vol", scale=2.0)
        i = t - L
        assert torch.equal(out["tradable"][i], base["tradable"][i]), t
        assert torch.allclose(out["windows"][i], base["windows"][i], atol=1e-6), t
        assert torch.allclose(out["w"][i], base["w"][i], atol=1e-6), t


def test_constant_model_makes_fixed_equal_rolling():
    """X and membership the same on every date: each day's hedge is date t's hedge, so
    re-hedging the window with date t's model changes nothing, after the vol scaling,
    the cumulative sum and the input scale as well."""
    g = torch.Generator().manual_seed(2)
    X = torch.randn(1, S, M, generator=g).expand(T, S, M).clone()
    idx = torch.arange(S).expand(T, S).clone()
    idx[:, -3:] = P                                    # padding, the same every day
    univ = idx < P
    R_pool = torch.randn(T, P, generator=g) * 0.02
    has_ret = torch.ones(T, P, dtype=torch.bool)
    R = torch.cat([R_pool, torch.zeros(T, 1)], 1).gather(1, idx)
    m = model()
    kw = {"cumulative": True, "normalise": "window_vol", "scale": 3.0}
    roll = run(m, X, R, univ, idx, R_pool, has_ret, "rolling", **kw)
    fix = run(m, X, R, univ, idx, R_pool, has_ret, "fixed", **kw)
    assert torch.equal(roll["tradable"], fix["tradable"])
    assert roll["tradable"][:, :S - 3].all()
    assert torch.allclose(roll["windows"], fix["windows"], atol=1e-5)
    assert torch.allclose(roll["w"], fix["w"], atol=1e-6)
    assert torch.equal(roll["eps"], fix["eps"])        # the EV term sees the same residuals


def test_fixed_window_passes_gradient_to_the_factor_model():
    m = model().train()
    X, R, univ, idx, R_pool, has_ret = churning_panel(3)
    out = m.forward_span(X, R, univ, idx, P, residual_window="fixed", R_pool=R_pool, has_ret=has_ret)
    out["windows"].pow(2).sum().backward()             # through the window alone, not the composition
    for name, p in [("W_K", m.factors.W_K), ("Q", m.factors.Q)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
