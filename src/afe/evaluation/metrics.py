"""Out-of-sample performance in the units of Table 2 of Epstein et al. (2025).

SR, mu, sigma are annualised from daily returns: mu = 252 mean (in %), sigma = sqrt(252)
sd (in %), SR = mu / sigma.  Table 2's own rows satisfy SR = mu / sigma to the printed
precision (market: 8.61 / 20.37 = 0.42), so the reported SR does NOT subtract the
risk-free rate; only the training objective does.  Net numbers are after the paper's
5bp turnover and 1bp shorting cost.  Beta is the OLS slope on the daily equal-weighted
universe return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def annualised(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
    return {"SR": mu / sd if sd > 0 else np.nan, "mu_pct": 100 * mu, "sigma_pct": 100 * sd}


def beta(r: np.ndarray, mkt: np.ndarray) -> float:
    x = np.column_stack([np.ones(len(mkt)), mkt])
    return float(np.linalg.lstsq(x, r, rcond=None)[0][1])


def break_even_turnover_cost(gross: np.ndarray, turnover: np.ndarray, short: np.ndarray, sc: float) -> float:
    """Per-unit turnover cost at which the mean net return is zero, holding the shorting
    cost fixed.  In basis points."""
    t = turnover.mean()
    return float(1e4 * (gross.mean() - sc * short.mean()) / t) if t > 0 else np.nan


def performance(daily: pd.DataFrame, tc: float, sc: float, cost_grid_bps: list[float]) -> dict:
    """`daily` has columns date, gross, turnover, short, mkt_ew, rf (daily, decimal)."""
    d = daily.set_index("date")
    net = d["gross"] - tc * d["turnover"] - sc * d["short"]
    out = {"n_days": int(len(d)), "first": str(d.index[0].date()), "last": str(d.index[-1].date())}
    out["gross"] = annualised(d["gross"])
    out["net"] = annualised(net)
    out["beta_ew_market"] = beta(d["gross"].to_numpy(), d["mkt_ew"].to_numpy())
    out["turnover_daily"] = float(d["turnover"].mean())
    out["short_exposure"] = float(d["short"].mean())
    out["break_even_turnover_cost_bps"] = break_even_turnover_cost(
        d["gross"].to_numpy(), d["turnover"].to_numpy(), d["short"].to_numpy(), sc)
    out["net_SR_by_turnover_cost_bps"] = {
        str(c): annualised(d["gross"] - c * 1e-4 * d["turnover"] - sc * d["short"])["SR"] for c in cost_grid_bps}
    out["market_ew"] = annualised(d["mkt_ew"])
    by_year = d.groupby(d.index.year)
    out["by_year"] = {str(y): {"SR": annualised(g["gross"])["SR"],
                               "SR_net": annualised(g["gross"] - tc * g["turnover"] - sc * g["short"])["SR"]}
                      for y, g in by_year}
    return out


def table_row(m: dict) -> str:
    g, n = m["gross"], m["net"]
    return (f"{g['SR']:5.2f} {g['mu_pct']:6.2f} {g['sigma_pct']:5.2f}   "
            f"{n['SR']:5.2f} {n['mu_pct']:6.2f} {n['sigma_pct']:5.2f}   {m['beta_ew_market']:5.2f}")
