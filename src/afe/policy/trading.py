"""From residual-space weights to asset-space weights, returns, costs and the objective.

Everything is in the "slot" layout of model/pca_factors.py: on each date the PCA set
occupies slots 0..n_t-1 of a fixed-width (T, 500) array, `idx` says which pool column a
slot holds, and padded slots have zero V, B, 1/vol and are never tradable.

Identities used (paper Section 3.2 and 3.3), with D = diag(vol):
    w_eps   = I - B V' D^-1                       residual composition, known at t-1
    w       = w_eps' w_port = w_port - D^-1 V (B' w_port)      asset-space weights
    R^port  = R_t' w                              == eps_t' w_port
    cost    = tc ||w_t - w_{t-1}||_1 + sc ||max(-w_t, 0)||_1     in ASSET space
The L1 normalisation ||w||_1 = 1 and the costs are applied in asset space, as in the
paper and in GPZ's train_test.py:100-116.  Turnover is measured on pool-aligned vectors,
so a name changing slot at a month end is not counted as a trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SlotBatch:
    """Inputs for a contiguous run of dates, already sliced to the block."""
    windows: torch.Tensor    # (T, S, L) residual windows, oldest first, 0 where undefined
    tradable: torch.Tensor   # (T, S) bool
    V: torch.Tensor          # (T, S, K)
    B: torch.Tensor          # (T, S, K)
    invvol: torch.Tensor     # (T, S)
    idx: torch.Tensor        # (T, S) int64, pool column, n_pool for padding
    R: torch.Tensor          # (T, S) realised return of the slot on the date, 0 if missing
    rf: torch.Tensor         # (T,)
    n_pool: int


def compose(w_port: torch.Tensor, b: SlotBatch) -> torch.Tensor:
    """Asset-space weights, L1-normalised per date. Dates with nothing tradable stay 0."""
    bw = torch.einsum("tsk,ts->tk", b.B, w_port)
    w = w_port - b.invvol * torch.einsum("tsk,tk->ts", b.V, bw)
    l1 = w.abs().sum(dim=1, keepdim=True)
    return w / torch.where(l1 > 0, l1, torch.ones_like(l1))


def to_pool(w: torch.Tensor, b: SlotBatch) -> torch.Tensor:
    """Scatter slot weights into (T, n_pool + 1); the last column collects the padding."""
    out = torch.zeros(w.shape[0], b.n_pool + 1, dtype=w.dtype, device=w.device)
    return out.scatter_add(1, b.idx, w)[:, :-1]


def net_returns(w: torch.Tensor, b: SlotBatch, tc: float, sc: float,
                w_prev_pool: torch.Tensor | None = None):
    """Gross return, cost and net return per date.  `w_prev_pool` is the pool-space
    position held before the first date of the block (zeros if None)."""
    gross = (w * b.R).sum(dim=1)
    wp = to_pool(w, b)
    prev = torch.zeros_like(wp[:1]) if w_prev_pool is None else w_prev_pool[None, :]
    turnover = (wp - torch.cat([prev, wp[:-1]])).abs().sum(dim=1)
    short = torch.clamp(-w, min=0).sum(dim=1)
    cost = tc * turnover + sc * short
    return gross, cost, gross - cost, turnover, short, wp


def sharpe_loss(net: torch.Tensor, rf: torch.Tensor, valid: torch.Tensor, subtract_rf: bool) -> torch.Tensor:
    """-(mean(net - rf) / sd(net)) over valid dates: the paper's objective without the
    explained-variance term, which is constant once the factors are fixed."""
    r = net[valid]
    excess = (r - rf[valid]) if subtract_rf else r
    return -(excess.mean() / (r.std() + 1e-12))


def make_windows(resid: np.ndarray, idx: np.ndarray, n: np.ndarray, t0: int, t1: int, L: int,
                 scale: float, cumulative: bool):
    """Residual windows for dates t0..t1-1 in slot layout: X[t, s] = resid[t-L : t, idx[t, s]].
    A slot is tradable when it is in the PCA set and all L residuals are finite."""
    T = t1 - t0
    S = idx.shape[1]
    X = np.zeros((T, S, L), dtype=np.float32)
    tradable = np.zeros((T, S), dtype=bool)
    pad = np.zeros((L, 1), dtype=np.float32)
    for i, t in enumerate(range(t0, t1)):
        m = int(n[t])
        if m == 0 or t < L:
            continue
        cols = idx[t, :m]
        win = resid[t - L:t][:, cols]                      # (L, m)
        ok = np.isfinite(win).all(axis=0)
        win = np.where(ok[None, :], win, pad)
        X[i, :m] = win.T
        tradable[i, :m] = ok
    if cumulative:
        X = np.cumsum(X, axis=-1)
    return X * scale, tradable
