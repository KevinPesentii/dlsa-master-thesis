"""The parametric benchmark of Table 2, "PCA + OU Thresh": an Ornstein-Uhlenbeck fit to each
residual's cumulative path, traded with the threshold rule of Avellaneda & Lee (2010) as
Guijarro-Ordonez, Pelger & Zanotti implement it (Section III.C and Appendix B).  Epstein et al.
use "the same implementation as in [19]".  GPZ's `preprocess_ou` (preprocess.py in the clone)
computes the same signal but feeds it to OU+FFN; their public code has no threshold rule.

For the cumulative residual path x_1..x_L of the last L residuals, the AR(1)
x_{l+1} = a + b x_l + e_l is fitted by OLS on the L-1 pairs.  The process is mean-reverting
only for 0 < b < 1, and then

    mu    = a / (1 - b)                   long-run mean
    sigma = sqrt(var(e) / (1 - b^2))      stationary sd, sigma_OU / sqrt(2 kappa)
    s     = (x_L - mu) / sigma            where today's level sits against the mean
    R2    = corr(x_l, x_{l+1})^2          fit of the AR(1)

The allocation has no state and nothing to train:

    w_eps = -1   if s >  c_thresh and R2 > c_crit      above the mean: short the residual
            +1   if s < -c_thresh and R2 > c_crit      below it: long
             0   otherwise, and whenever b is outside (0, 1)

GPZ's values are c_thresh = 1.25 and c_crit = 0.25.  c_crit is an R2 cutoff, not Avellaneda
& Lee's closing band, so a position is dropped as soon as |s| falls back below c_thresh.  The
weights then go through the same composition and L1 normalisation as the learned policies
(trading.compose).

GPZ's code adds 1e-6 to 1 - b and 1 - b^2.  It is kept, so the signal is theirs to float
precision; it only moves s on near-unit-root paths (b close to 1).
"""

from __future__ import annotations

import numpy as np

GPZ_EPS = 1e-6


def ou_signal(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s-score, R2 and the mean-reversion mask of cumulative paths x (..., L), oldest first.
    Population moments, as in GPZ's code.  s and R2 are 0 where the mask is False."""
    s, r2, _, ok = ou_fit(x)
    return s, r2, ok


def ou_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ou_signal plus the AR(1) slope b (1 where the mask is False), for the
    mean-reversion-speed filter kappa = -252 log b."""
    x = np.asarray(x, dtype=np.float64)
    X, Y = x[..., :-1], x[..., 1:]
    mX, mY = X.mean(axis=-1), Y.mean(axis=-1)
    dX, dY = X - mX[..., None], Y - mY[..., None]
    vX, vY, cov = (dX * dX).mean(axis=-1), (dY * dY).mean(axis=-1), (dX * dY).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = cov / vX
        a = mY - b * mX
        var_e = (Y - a[..., None] - b[..., None] * X).var(axis=-1)
        sigma = np.sqrt(var_e / (1 - b * b + GPZ_EPS))
        s = (x[..., -1] - a / (1 - b + GPZ_EPS)) / sigma
        r2 = cov * cov / (vX * vY)
    ok = (vX > 0) & (vY > 0) & (b > 0) & (b < 1) & (sigma > 0)
    return np.where(ok, s, 0.0), np.where(ok, r2, 0.0), np.where(ok, b, 1.0), ok


def threshold_weights(x: np.ndarray, tradable: np.ndarray, c_thresh: float = 1.25,
                      c_crit: float = 0.25) -> np.ndarray:
    """Residual-space weights in {-1, 0, 1}, float32, for paths x (..., L) and a mask (...)."""
    s, r2, ok = ou_signal(x)
    on = ok & np.asarray(tradable, dtype=bool) & (r2 > c_crit)
    w = np.zeros(s.shape, dtype=np.float32)
    w[on & (s > c_thresh)] = -1.0
    w[on & (s < -c_thresh)] = 1.0
    return w


def band_weights(s: np.ndarray, open_ok: np.ndarray, hold_ok: np.ndarray, sec: np.ndarray,
                 state: np.ndarray, c_open: float = 1.25, c_close_short: float = 0.75,
                 c_close_long: float = 0.50) -> np.ndarray:
    """Avellaneda & Lee's (2010, Section 5) bands, with memory: buy to open if s < -c_open,
    sell to open if s > c_open, close a short if s < c_close_short, close a long if
    s > -c_close_long (their 1.25, 1.25, 0.75, 0.50).  A diagnostic next to GPZ's
    memoryless rule (threshold_weights), not a thesis row.

    s, open_ok, hold_ok, sec: (T, S) by day and slot; sec is the slot's pool column, so a
    position follows the name across days.  open_ok gates new positions (signal defined and
    the filter passed); a held position is closed when hold_ok is False (signal undefined:
    b outside (0, 1), or the name left the tradable set).  A close and an opening on the
    same day are allowed (long to short in one step).  state: (P + 1,) position by pool
    column carried in, updated in place; names not in a day's slots are flat after it."""
    T = s.shape[0]
    w = np.zeros(s.shape, dtype=np.float32)
    for t in range(T):
        prev = state[sec[t]]
        cur = prev.copy()
        cur[(prev > 0) & ((s[t] > -c_close_long) | ~hold_ok[t])] = 0.0
        cur[(prev < 0) & ((s[t] < c_close_short) | ~hold_ok[t])] = 0.0
        flat = (cur == 0) & open_ok[t]
        cur[flat & (s[t] < -c_open)] = 1.0
        cur[flat & (s[t] > c_open)] = -1.0
        w[t] = cur
        state[:] = 0.0
        state[sec[t]] = cur
        state[-1] = 0.0                                   # padding column
    return w
