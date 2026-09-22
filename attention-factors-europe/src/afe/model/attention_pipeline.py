"""One-step attention arbitrage: factors, residuals, policy and objective in one graph.

The two-step path fixes the residuals first (model/pca_factors.py) and trains the policy
on them. Here the factors are learned from the trading objective itself, which is the
paper's contribution (Section 3.3), so every residual has to stay attached to the graph:
the gradient of the net Sharpe reaches W_K and Q through eps.

That rules out trading.make_windows, which is NumPy and consumes a residual array that
already exists. Windows are built here with a gather, so they are differentiable.

Layout follows policy/trading.py. Two index spaces are in play and the distinction
matters: SLOT space (T, S) is who is in the universe on a date, and it is reshuffled
whenever membership changes; POOL space (T, n_pool) is a fixed column per company. A
window has to follow the company, not the slot, so residuals are scattered into pool
space before the window is cut.

A date is traded when the name is in the universe that day AND its residual is defined
on each of the L preceding days. The second condition is what make_windows enforced with
`isfinite`; with residuals built inside the graph the equivalent is membership on each
of those days.

Timing: X passed to forward_span must already be lagged, i.e. row t holds what was known
at the close of t-1. Nothing in this module shifts it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from afe.model.attention_factors import AttentionFactors, compose
from afe.policy.longconv import LongConvPolicy
from afe.policy import trading


def to_pool(x: torch.Tensor, idx: torch.Tensor, n_pool: int) -> torch.Tensor:
    """Slot space (T, S) -> pool space (T, n_pool). Padding lands in a discarded column."""
    out = torch.zeros(x.shape[0], n_pool + 1, dtype=x.dtype, device=x.device)
    return out.scatter_add(1, idx, x)[:, :-1]


def residual_windows(eps_pool: torch.Tensor, valid_pool: torch.Tensor, idx: torch.Tensor,
                     lookback: int, cumulative: bool = True, scale: float = 1.0):
    """Windows of past residuals per slot, differentiable in eps_pool.

    eps_pool: (T, P) residuals by company. valid_pool: (T, P) bool. idx: (T, S) company of
    each slot, P for padding. Returns windows (T-L, S, L) oldest first and tradable
    (T-L, S); row i corresponds to date L+i, and its window covers dates i..L+i-1, i.e.
    strictly before the date being traded.
    """
    T, P = eps_pool.shape
    S = idx.shape[1]
    L = lookback
    if T <= L:
        raise ValueError(f"span of {T} dates is too short for a lookback of {L}")

    cols = idx[L:]                                                    # (T-L, S)
    lags = torch.arange(L, device=eps_pool.device)
    rows = (torch.arange(T - L, device=eps_pool.device)[:, None] + lags[None, :])   # (T-L, L)

    flat = torch.cat([eps_pool, eps_pool.new_zeros(T, 1)], dim=1).reshape(-1)       # pad column
    gather_idx = rows[:, None, :] * (P + 1) + cols[:, :, None]        # (T-L, S, L)
    windows = flat[gather_idx.reshape(-1)].reshape(T - L, S, L)

    vflat = torch.cat([valid_pool, valid_pool.new_zeros(T, 1)], dim=1).reshape(-1)
    history_ok = vflat[gather_idx.reshape(-1)].reshape(T - L, S, L).all(dim=-1)
    own_row = torch.arange(L, T, device=eps_pool.device)[:, None] * (P + 1) + cols
    own_ok = vflat[own_row.reshape(-1)].reshape(T - L, S)
    tradable = history_ok & own_ok & (cols < P)

    windows = windows * tradable.unsqueeze(-1)
    if cumulative:
        windows = windows.cumsum(dim=-1)
    return windows * scale, tradable


class AttentionArb(nn.Module):
    """Attention factors + LongConv policy, trained end to end on the net Sharpe."""

    def __init__(self, n_features: int, n_factors: int = 30, embedding_dim: int = 32,
                 lambda_ridge: float = 1e-4, hidden: int = 32, lookback: int = 30,
                 dropout: float = 0.1, lambda_squash: float = 1e-3):
        super().__init__()
        self.factors = AttentionFactors(n_features, n_factors, embedding_dim, lambda_ridge)
        self.policy = LongConvPolicy(hidden, lookback, dropout, lambda_squash)
        self.lookback = lookback

    def forward_span(self, X, R_slot, in_universe, idx, n_pool, cumulative=True, scale=1.0):
        """X: (T,S,M) lagged characteristics. R_slot: (T,S). in_universe: (T,S) bool.
        idx: (T,S) int64 company column, n_pool for padding.

        Returns a dict for dates L..T-1: asset-space weights (L1-normalised), the slot
        tradability of those dates, and the residuals and returns of the whole span,
        which the explained-variance term needs.
        """
        L = self.lookback
        w_F, betaT, eps = self.factors(X, in_universe, R_slot)

        eps_pool = to_pool(eps, idx, n_pool)
        valid_pool = to_pool(in_universe.to(eps.dtype), idx, n_pool) > 0
        windows, tradable = residual_windows(eps_pool, valid_pool, idx, L, cumulative, scale)

        w_port = self.policy(windows) * tradable
        w = compose(w_port, w_F[L:], betaT[L:])
        l1 = w.abs().sum(dim=1, keepdim=True)
        w = w / torch.where(l1 > 0, l1, torch.ones_like(l1))
        return {"w": w, "tradable": tradable, "eps": eps, "R": R_slot,
                "in_universe": in_universe, "w_port": w_port, "l1": l1.squeeze(1),
                "idx": idx, "n_pool": n_pool}


def explained_variance(eps: torch.Tensor, R: torch.Tensor, mask: torch.Tensor,
                       idx: torch.Tensor | None = None, n_pool: int | None = None,
                       min_obs: int = 20) -> torch.Tensor:
    """The paper's second objective term, Section 3.3:

        (1/N) sum_i ( 1 - Var(e_i) / Var(R_i) )

    one explained-variance ratio per ASSET, over the dates of the block, then the plain
    average across assets. Every name counts equally, whatever its volatility.

    Asset means company, not slot: slots are reshuffled at every monthly rebalance, so the
    series are rebuilt in pool space through `idx` before the variances are taken. A name
    enters the average only if it is in the universe on at least `min_obs` dates of the
    block (the paper does not say how it treats partial histories; with 125-day blocks and
    a monthly universe this drops only names that enter or leave inside the block).
    Variances are demeaned and use 1/n, over the dates the name is present.

    Without idx the slots are taken to be assets, which is only right if membership does
    not change inside the span; kept for tests.

    Replaces the pooled 1 - SS(eps)/SS(R) of the first version, which weighted each name
    by its return variance, so the volatile names dominated the factors.
    """
    m = mask.to(eps.dtype)
    if idx is not None:
        e = to_pool(eps * m, idx, n_pool)
        r = to_pool(R * m, idx, n_pool)
        m = to_pool(m, idx, n_pool)
    else:
        e, r = eps * m, R * m
    n = m.sum(dim=0)
    keep = n >= min_obs
    n_safe = torch.where(keep, n, torch.ones_like(n))
    e_mean = e.sum(dim=0) / n_safe
    r_mean = r.sum(dim=0) / n_safe
    var_e = (((e - e_mean) * m) ** 2).sum(dim=0) / n_safe
    var_r = (((r - r_mean) * m) ** 2).sum(dim=0) / n_safe
    keep = keep & (var_r > 0)
    if not bool(keep.any()):
        return eps.new_zeros(())
    ratio = 1.0 - var_e[keep] / var_r[keep]
    return ratio.mean()


def explained_variance_pooled(eps: torch.Tensor, R: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """First version, pooled 1 - SS(eps)/SS(R). NOT the paper's formula; kept for comparison."""
    m = mask.to(eps.dtype)
    return 1.0 - ((eps * m) ** 2).sum() / (((R * m) ** 2).sum() + 1e-12)


def objective(out: dict, batch: trading.SlotBatch, tc: float, sc: float, lambda_var: float,
              subtract_rf: bool = False, w_prev_pool=None):
    """Loss and its parts. The Sharpe half reuses trading.net_returns, so turnover and
    short costs are measured in asset space exactly as in the two-step path."""
    gross, cost, net, turnover, short, wp = trading.net_returns(out["w"], batch, tc, sc,
                                                                w_prev_pool)
    valid = out["tradable"].any(dim=1)
    sharpe = trading.sharpe_loss(net, batch.rf, valid, subtract_rf)
    ev = explained_variance(out["eps"], out["R"], out["in_universe"], out["idx"], out["n_pool"])
    return sharpe - lambda_var * ev, {"sharpe_loss": sharpe, "ev": ev, "net": net,
                                      "turnover": turnover, "short": short, "w_pool": wp}


def slot_batch(R_slot, idx, rf, tradable, n_pool) -> trading.SlotBatch:
    """SlotBatch for the cost accounting. V, B and invvol belong to the PCA composition
    and are unused here, so they are empty."""
    z = R_slot.new_zeros(R_slot.shape + (0,))
    return trading.SlotBatch(windows=R_slot.new_zeros(R_slot.shape + (0,)), tradable=tradable,
                             V=z, B=z, invvol=R_slot.new_zeros(R_slot.shape), idx=idx,
                             R=R_slot, rf=rf, n_pool=n_pool)
