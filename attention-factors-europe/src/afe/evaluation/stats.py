"""Inference for the thesis tables: Sharpe ratio standard errors, the paired Sharpe-difference
test, and alpha regressions, all with heteroskedasticity- and autocorrelation-consistent (HAC)
covariances of daily returns.

Sharpe ratios are annualised like evaluation.metrics (sqrt(252) * mean / sd, no risk-free
rate subtracted, Table 2's convention). The standard errors are the delta method on the
first two moments (Lo 2002, with a HAC long-run covariance instead of iid); the difference
test is the delta-method test of Ledoit and Wolf (2008, Section 3.1) with a Newey-West
(Bartlett) kernel and their automatic lag rule replaced by Newey and West's (1994)
floor(4 (T/100)^(2/9)). Their studentised bootstrap is not implemented.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

ANN = 252


def nw_lags(T: int) -> int:
    return int(math.floor(4 * (T / 100) ** (2 / 9)))


def hac(u: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West long-run covariance of the rows of u (T, k), demeaned inside."""
    u = np.asarray(u, dtype=float)
    u = u - u.mean(axis=0)
    T = len(u)
    L = nw_lags(T) if lags is None else lags
    S = u.T @ u / T
    for j in range(1, L + 1):
        G = u[j:].T @ u[:-j] / T
        S += (1 - j / (L + 1)) * (G + G.T)
    return S


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.sqrt(ANN) * r.mean() / r.std(ddof=1))


def sharpe_se(r: np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the annualised Sharpe ratio. With lags=0 and normal returns it is
    Lo's iid value sqrt((1 + SR_d^2 / 2) / T) * sqrt(252) up to the excess-kurtosis term."""
    r = np.asarray(r, dtype=float)
    m1, m2 = r.mean(), (r ** 2).mean()
    v = m2 - m1 ** 2
    g = np.array([m2 / v ** 1.5, -m1 / (2 * v ** 1.5)])
    S = hac(np.column_stack([r, r ** 2]), lags)
    return float(np.sqrt(ANN) * np.sqrt(g @ S @ g / len(r)))


def sharpe_diff(r1: np.ndarray, r2: np.ndarray, lags: int | None = None) -> dict:
    """Paired test of SR(r1) = SR(r2) on the same days (Ledoit-Wolf delta method, HAC)."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    a = np.array([r1.mean(), r2.mean(), (r1 ** 2).mean(), (r2 ** 2).mean()])
    v1, v2 = a[2] - a[0] ** 2, a[3] - a[1] ** 2
    grad = np.array([a[2] / v1 ** 1.5, -a[3] / v2 ** 1.5, -a[0] / (2 * v1 ** 1.5), a[1] / (2 * v2 ** 1.5)])
    S = hac(np.column_stack([r1, r2, r1 ** 2, r2 ** 2]), lags)
    se = float(np.sqrt(ANN) * np.sqrt(grad @ S @ grad / len(r1)))
    d = sharpe(r1) - sharpe(r2)
    # the moment form uses population sd; the reported difference uses ddof=1 as everywhere else
    z = d / se if se > 0 else float("nan")
    return {"diff": d, "se": se, "z": z, "p": float(2 * (1 - norm.cdf(abs(z)))), "T": len(r1),
            "mdd_5pct": 1.96 * se}


def ols_nw(y: np.ndarray, X: np.ndarray, names: list[str], lags: int | None = None) -> dict:
    """OLS of y on [1, X] with Newey-West standard errors. Returns coefficients, t-stats, R^2;
    alpha annualised (x 252)."""
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    ZZi = np.linalg.inv(Z.T @ Z / len(y))
    S = hac(Z * e[:, None], lags)
    V = ZZi @ S @ ZZi / len(y)
    t = b / np.sqrt(np.diag(V))
    r2 = 1 - e.var() / y.var()
    out = {"alpha_ann_pct": 100 * ANN * b[0], "t_alpha": t[0], "r2": r2, "T": len(y)}
    for n, bi, ti in zip(names, b[1:], t[1:]):
        out[f"b_{n}"], out[f"t_{n}"] = bi, ti
    return out
