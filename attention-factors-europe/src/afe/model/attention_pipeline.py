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
of those days. That is policy.residual_window "rolling"; "fixed" re-hedges the whole
window with the trade date's factor model instead (fixed_residual_windows).

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


def _gather_windows(x_pool: torch.Tensor, idx: torch.Tensor, lookback: int) -> torch.Tensor:
    """(T, P) by company -> (T-L, S, L): row i, slot s holds x of dates i..L+i-1 for the
    company in slot s on date L+i, i.e. in the slot order of the date being traded. Padding
    (idx == P) reads a column of zeros (False for a bool x)."""
    T, P = x_pool.shape
    L = lookback
    if T <= L:
        raise ValueError(f"span of {T} dates is too short for a lookback of {L}")
    cols = idx[L:]                                                    # (T-L, S)
    lags = torch.arange(L, device=x_pool.device)
    rows = (torch.arange(T - L, device=x_pool.device)[:, None] + lags[None, :])     # (T-L, L)
    flat = torch.cat([x_pool, x_pool.new_zeros(T, 1)], dim=1).reshape(-1)           # pad column
    gather_idx = rows[:, None, :] * (P + 1) + cols[:, :, None]        # (T-L, S, L)
    return flat[gather_idx.reshape(-1)].reshape(T - L, idx.shape[1], L)


def residual_windows(eps_pool: torch.Tensor, valid_pool: torch.Tensor, idx: torch.Tensor,
                     lookback: int, cumulative: bool = True, scale: float = 1.0,
                     normalise: str = "none", vol_floor: float = 1e-4):
    """Windows of past residuals per slot, differentiable in eps_pool (policy.residual_window
    rolling: each day of the window keeps the hedge of its own date).

    eps_pool: (T, P) residuals by company. valid_pool: (T, P) bool. idx: (T, S) company of
    each slot, P for padding. Returns windows (T-L, S, L) oldest first and tradable
    (T-L, S); row i corresponds to date L+i, and its window covers dates i..L+i-1, i.e.
    strictly before the date being traded.
    """
    T, P = eps_pool.shape
    L = lookback
    windows = _gather_windows(eps_pool, idx, L)
    history_ok = _gather_windows(valid_pool, idx, L).all(dim=-1)
    cols = idx[L:]
    own_ok = torch.cat([valid_pool, valid_pool.new_zeros(T, 1)], dim=1)[L:].gather(1, cols)
    tradable = history_ok & own_ok & (cols < P)
    return _policy_input(windows, tradable, cumulative, scale, normalise, vol_floor)


def fixed_residual_windows(R_pool: torch.Tensor, has_ret: torch.Tensor, in_universe: torch.Tensor,
                           idx: torch.Tensor, w_F: torch.Tensor, betaT: torch.Tensor, lookback: int,
                           cumulative: bool = True, scale: float = 1.0, normalise: str = "none",
                           vol_floor: float = 1e-4):
    """Windows re-hedged with the factor model of the date being traded (policy.residual_window
    fixed). For trade date t and the company i in slot s on t, each day tau = t-L..t-1 is

        eps~_{i,tau} = R_{i,tau} - beta_{i,t}' (w_F,t R_tau)

    with w_F,t and beta_t of date t (built from X row t, known at the close of t-1), applied
    to the returns on tau of date t's members, whether or not they were members on tau. The
    window is then the past path of exactly the residual portfolio that is traded at t;
    `rolling` hedges each day with its own model, eps_tau = R_tau - beta_tau' (w_F,tau R_tau).

    Tradable: in the universe at t and a return on every one of the L window days. Unlike
    rolling, membership on those days is not needed. A constituent of w_F,t without a return
    on a window day counts as 0 in that day's factor return.

    R_pool, has_ret: (T, P) returns by company (0 where none) and where they exist.
    in_universe (T-L, S), w_F (T-L, K, S), betaT (T-L, S, K): those of the trade dates L..T-1.
    Differentiable in w_F and betaT; it forms K x L factor returns per date, no S x S matrix.
    """
    L = lookback
    R_win = _gather_windows(R_pool, idx, L)                           # (T-L, S, L)
    history_ok = _gather_windows(has_ret, idx, L).all(dim=-1)
    tradable = in_universe & history_ok & (idx[L:] < R_pool.shape[1])
    F = torch.einsum("tks,tsl->tkl", w_F, R_win)                     # date-t factors on tau
    windows = R_win - torch.einsum("tsk,tkl->tsl", betaT, F)
    return _policy_input(windows, tradable, cumulative, scale, normalise, vol_floor)


def _policy_input(windows, tradable, cumulative, scale, normalise, vol_floor):
    """Masking, normalisation, cumulative sum and scale, the same for both window kinds."""
    windows = windows * tradable.unsqueeze(-1)
    if normalise == "window_vol":
        # Each name's window divided by its own residual volatility over that window. Not in the
        # paper. Two reasons: a volatile name and a quiet one otherwise enter the policy with very
        # different amplitudes and the policy has to undo that itself; and the input scale stops
        # mattering, since the ratio is scale-free (policy.input_scale then only rescales a
        # standardised signal). Uses the L past residuals only, so it is as backward-looking as
        # the window itself. The floor keeps a near-zero-volatility window from exploding.
        sd = windows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(vol_floor)
        windows = windows / sd
    elif normalise != "none":
        raise ValueError(f"unknown policy.input_normalise: {normalise}")
    if cumulative:
        windows = windows.cumsum(dim=-1)
    return windows * scale, tradable


class AttentionArb(nn.Module):
    """Attention factors + LongConv policy, trained end to end on the net Sharpe."""

    def __init__(self, n_features: int, n_factors: int = 30, embedding_dim: int = 32,
                 lambda_ridge: float = 1e-4, hidden: int = 32, lookback: int = 30,
                 dropout: float = 0.1, lambda_squash: float = 1e-3, level_hidden: int = 0):
        super().__init__()
        self.factors = AttentionFactors(n_features, n_factors, embedding_dim, lambda_ridge,
                                        level_hidden)
        self.policy = LongConvPolicy(hidden, lookback, dropout, lambda_squash)
        self.lookback = lookback

    def forward_span(self, X, R_slot, in_universe, idx, n_pool, cumulative=True, scale=1.0,
                     normalise="none", residual_window="rolling", R_pool=None, has_ret=None):
        """X: (T,S,M) lagged characteristics. R_slot: (T,S). in_universe: (T,S) bool.
        idx: (T,S) int64 company column, n_pool for padding. residual_window "fixed" also
        needs R_pool / has_ret (T, n_pool), every company's returns (SlotPanel.R_pool).

        Returns a dict for dates L..T-1: asset-space weights (L1-normalised), the slot
        tradability of those dates, the policy input, and the residuals and returns of the
        whole span, which the explained-variance term needs (same-day residuals, whatever
        the window kind).
        """
        L = self.lookback
        w_F, betaT, eps = self.factors(X, in_universe, R_slot)

        if residual_window == "rolling":
            eps_pool = to_pool(eps, idx, n_pool)
            valid_pool = to_pool(in_universe.to(eps.dtype), idx, n_pool) > 0
            windows, tradable = residual_windows(eps_pool, valid_pool, idx, L, cumulative, scale,
                                                 normalise)
        elif residual_window == "fixed":
            if R_pool is None or has_ret is None:
                raise ValueError("residual_window fixed needs R_pool and has_ret")
            windows, tradable = fixed_residual_windows(R_pool, has_ret, in_universe[L:], idx,
                                                       w_F[L:], betaT[L:], L, cumulative, scale,
                                                       normalise)
        else:
            raise ValueError(f"unknown policy.residual_window: {residual_window}")

        w_port = self.policy(windows) * tradable
        w = compose(w_port, w_F[L:], betaT[L:])
        l1 = w.abs().sum(dim=1, keepdim=True)
        w = w / torch.where(l1 > 0, l1, torch.ones_like(l1))
        return {"w": w, "tradable": tradable, "windows": windows, "eps": eps, "R": R_slot,
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
              subtract_rf: bool = False, w_prev_pool=None, held=None, R_pool=None):
    """Loss and its parts. The Sharpe half reuses trading.net_returns, so turnover and
    short costs are measured in asset space exactly as in the two-step path. With `held`
    (pool-space positions from trading.execute) and the pool returns of the same days,
    the returns and costs are those of the executed book instead."""
    if held is None:
        gross, cost, net, turnover, short, wp = trading.net_returns(out["w"], batch, tc, sc,
                                                                    w_prev_pool)
    else:
        gross, cost, net, turnover, short, wp = trading.pool_net_returns(held, R_pool, tc, sc)
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
