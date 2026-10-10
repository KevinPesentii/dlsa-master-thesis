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
    return np.where(ok, s, 0.0), np.where(ok, r2, 0.0), ok


def threshold_weights(x: np.ndarray, tradable: np.ndarray, c_thresh: float = 1.25,
                      c_crit: float = 0.25) -> np.ndarray:
    """Residual-space weights in {-1, 0, 1}, float32, for paths x (..., L) and a mask (...)."""
    s, r2, ok = ou_signal(x)
    on = ok & np.asarray(tradable, dtype=bool) & (r2 > c_crit)
    w = np.zeros(s.shape, dtype=np.float32)
    w[on & (s > c_thresh)] = -1.0
    w[on & (s < -c_thresh)] = 1.0
    return w
