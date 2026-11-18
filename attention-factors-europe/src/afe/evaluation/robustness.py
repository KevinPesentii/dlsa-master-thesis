"""Saved portfolio weights re-scored under other implementation assumptions. Nothing is retrained.

A run's weights.parquet holds the TARGET weights: asset space, one row per (date, sec_id) of the slots
the policy traded, L1-normalised per date, before `trading.execute`. The runners book them as

    H = trading.execute(target, closed, lag)       position held over day t (pool space)
    gross_t = H_t' R_t,  turnover_t = ||H_t - H_{t-1}||_1 (H_0 = 0, refit seams included),
    short_t = ||max(-H_t, 0)||_1,  net_t = gross_t - tc turnover_t - sc short_t

where closed[k] says the name's market did not trade at the close before day k (Europe's holiday-filled
rows, returns.traded False): such a name keeps its position. A name that leaves the universe gets target 0
and is sold at the next open close; R is the numeraire return of returns.parquet, 0 where there is none.

`book` adds what the variants change: another lag, a per-name cost per unit traded instead of the flat tc
(half spread), a purchase tax rate * max(dH, 0) and another return matrix. Row t of dH is the trade done at
the close before day t, so a cost or tax that depends on the date uses that close (`trade_close`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from afe.data.return_gaps import check_mode, halted
from afe.evaluation import metrics
from afe.policy import trading


@dataclass
class Market:
    dates: pd.DatetimeIndex   # every close of the return panel, sorted
    cols: pd.Index            # sec_ids
    R: np.ndarray             # (T, P) numeraire return, 0 where none
    closed: np.ndarray | None  # (T, P) True where the name's market did not trade at that close
    rows: pd.DataFrame        # the returns.parquet rows of these names (date, sec_id, mktcap_lag, ...)


def load_market(data_dir: Path, sec_ids, start: str = "1997-01-01", return_gaps: str = "masked") -> Market:
    """The return panel of `data_dir` for `sec_ids` from `start` on (dates = every date of the file).
    `return_gaps` as the run's data.return_gaps: "closed" marks a halt inside a name's trading life
    closed (data/return_gaps.py), so a book re-scored here is executed as the run executed it."""
    check_mode(return_gaps)
    cols = pd.Index(sorted(set(sec_ids)), name="sec_id")
    ret = pd.read_parquet(data_dir / "returns.parquet", filters=[("date", ">=", pd.Timestamp(start))])
    ret["date"] = pd.to_datetime(ret["date"]).astype("datetime64[ns]")
    dates = pd.DatetimeIndex(np.sort(ret["date"].unique()))
    ret = ret[ret["sec_id"].isin(cols)]
    R = ret.pivot(index="date", columns="sec_id", values="ret").reindex(index=dates, columns=cols)
    R = R.to_numpy(dtype=np.float64)
    closed = None
    if "traded" in ret:
        nt = ret.loc[~ret["traded"].astype(bool), ["date", "sec_id"]].assign(one=True)
        closed = nt.pivot(index="date", columns="sec_id", values="one").reindex(index=dates, columns=cols).notna()
        closed = closed.to_numpy()
    if return_gaps == "closed":
        gaps = halted(data_dir, dates, cols)
        closed = gaps if closed is None else closed | gaps
    return Market(dates, cols, np.nan_to_num(R), closed, ret.reset_index(drop=True))


def wide(m: Market, long: pd.DataFrame, value: str) -> np.ndarray:
    """(T, P) matrix of `value` from a (date, sec_id, value) frame, NaN where missing."""
    w = long.pivot_table(index="date", columns="sec_id", values=value, aggfunc="sum")
    return w.reindex(index=m.dates, columns=m.cols).to_numpy(dtype=np.float64)


def targets(run_dir: Path, m: Market) -> np.ndarray:
    """Target weights of a run in pool space (a company in two slots on a date is summed), 0 elsewhere."""
    w = pd.read_parquet(run_dir / "weights.parquet")
    w["date"] = pd.to_datetime(w["date"]).astype("datetime64[ns]")
    return np.nan_to_num(wide(m, w, "w"))


def oos_index(m: Market, first: str, last: str) -> np.ndarray:
    t = np.flatnonzero((m.dates >= pd.Timestamp(first)) & (m.dates <= pd.Timestamp(last)))
    if not (np.diff(t) == 1).all() or t[0] == 0:
        raise ValueError("out-of-sample days must be consecutive and preceded by a close")
    return t


def held(target: np.ndarray, m: Market, t_idx: np.ndarray, lag: int, stale: bool = True) -> np.ndarray:
    """Positions over the days t_idx, as the runners' daily_book: trading.execute on the closes before them."""
    closed = None
    if stale and m.closed is not None:
        closed = torch.from_numpy(m.closed[t_idx[0] - 1:t_idx[-1]])
    tgt = torch.from_numpy(target[t_idx])
    if not lag and closed is None:
        return tgt.numpy()
    return trading.execute(tgt, closed, lag).numpy()


def book(H: np.ndarray, R: np.ndarray, tc: float, sc: float, cost_per_unit: np.ndarray | None = None,
         tax_rate: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Daily gross, turnover, short and costs of the held book H on returns R (both (T, P), the same days).
    cost_per_unit: per-name cost of one unit traded (replaces tc); tax_rate: (buy, sell) per-name tax per
    unit bought (dH > 0: a new long or a short covered) and sold."""
    dH = np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1]), dtype=H.dtype))
    a = np.abs(dH)
    turnover = a.sum(axis=1)
    trade = tc * turnover if cost_per_unit is None else (cost_per_unit * a).sum(axis=1)
    tax = np.zeros(len(H))
    if tax_rate is not None:
        tax = (tax_rate[0] * np.clip(dH, 0, None) + tax_rate[1] * np.clip(-dH, 0, None)).sum(axis=1)
    short = np.clip(-H, 0, None).sum(axis=1)
    gross = (H * R).sum(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": turnover, "short": short, "trade_cost": trade,
                         "short_cost": sc * short, "tax": tax, "net": gross - trade - sc * short - tax})


def summarise(d: pd.DataFrame, sc: float) -> dict:
    """Table 2 units (metrics.annualised). break_even_bps: flat cost per unit of turnover at which the mean
    net return is zero, after the short fee and the taxes; eff_cost_bps: the trading cost actually charged
    per unit of turnover (5 under the flat cost)."""
    g, n = metrics.annualised(d["gross"]), metrics.annualised(d["net"])
    be = metrics.break_even_turnover_cost((d["gross"] - d["tax"]).to_numpy(), d["turnover"].to_numpy(),
                                          d["short"].to_numpy(), sc)
    return {"SR": g["SR"], "mu_pct": g["mu_pct"], "sigma_pct": g["sigma_pct"], "SR_net": n["SR"],
            "mu_net_pct": n["mu_pct"], "sigma_net_pct": n["sigma_pct"], "turnover": d["turnover"].mean(),
            "short": d["short"].mean(), "break_even_bps": be,
            "eff_cost_bps": 1e4 * d["trade_cost"].mean() / d["turnover"].mean(),
            "tax_pct": 100 * 252 * d["tax"].mean()}


# ------------------------------------------------------------------ spreads
MIN_DAYS = 10   # days per month an estimate needs, the spread_min_obs of the characteristic builds


def quoted_spread_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean of the closing quoted spread (ask - bid) / midpoint over the days with 0 < bid < ask
    (the rule of us_panel's Spread characteristic), >= MIN_DAYS such days. daily: sec_id, date, bid, ask."""
    ok = (daily["bid"] > 0) & (daily["ask"] > daily["bid"])
    d = daily[ok].assign(s=lambda x: (x["ask"] - x["bid"]) / ((x["ask"] + x["bid"]) / 2))
    g = d.groupby(["sec_id", d["date"].dt.to_period("M").rename("month")])["s"].agg(["mean", "count"])
    g = g[g["count"] >= MIN_DAYS]
    return g["mean"].rename("spread").reset_index().assign(month=lambda x: x["month"].dt.to_timestamp())


def abdi_ranaldo_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Abdi & Ranaldo (2017) close-high-low spread, monthly: S = sqrt(max(4 mean[(c_t - eta_t)(c_t - eta_t+1)], 0)),
    c = log close, eta = mid-range (log high + log low) / 2, over the pairs of consecutive traded days of the
    same line whose second day falls in the month, >= MIN_DAYS pairs. daily: sec_id, line, date, close, high,
    low (local prices of traded days)."""
    d = daily[(daily["low"] > 0) & (daily["high"] >= daily["low"]) & (daily["close"] > 0)]
    d = d.sort_values(["sec_id", "date"])
    c, eta = np.log(d["close"]), 0.5 * (np.log(d["high"]) + np.log(d["low"]))
    same = (d["sec_id"] == d["sec_id"].shift(-1)) & (d["line"] == d["line"].shift(-1))
    prod = ((c - eta) * (c - eta.shift(-1))).where(same)
    p = pd.DataFrame({"sec_id": d["sec_id"].shift(-1), "month": d["date"].shift(-1).dt.to_period("M"),
                      "x": prod}).dropna()
    g = p.groupby(["sec_id", "month"])["x"].agg(["mean", "count"])
    g = g[g["count"] >= MIN_DAYS]
    s = np.sqrt(np.clip(4 * g["mean"], 0, None))
    return s.rename("spread").reset_index().assign(month=lambda x: x["month"].dt.to_timestamp())


def daily_mean_monthly(daily: pd.DataFrame, col: str) -> pd.DataFrame:
    """Monthly mean of a daily estimate (e.g. the two-day Corwin-Schultz `cs` of europe_panel), >= MIN_DAYS."""
    g = daily.dropna(subset=[col]).groupby(["sec_id", daily["date"].dt.to_period("M").rename("month")])[col]
    g = g.agg(["mean", "count"])
    g = g[g["count"] >= MIN_DAYS]
    return g["mean"].rename("spread").reset_index().assign(month=lambda x: x["month"].dt.to_timestamp())


def half_spread(m: Market, monthly: pd.DataFrame, members: pd.DataFrame, t_idx: np.ndarray,
                winsor: float = 0.01) -> tuple[np.ndarray, dict]:
    """Half spread charged on row t of t_idx, i.e. on the trade at the close before day t: the full-spread
    estimate of the calendar month before that close's month (monthly: month, sec_id, spread), so it is
    known before the trade. Per month, winsorised at `winsor` / 1 - `winsor` over that month's universe
    members, and a missing value is the members' median of that month. Returns (T, P) and fill counts."""
    sp = monthly.dropna(subset=["spread"]).copy()
    sp["spread"] = sp["spread"].clip(lower=0.0)
    sp["use"] = sp["month"].dt.to_period("M") + 1          # estimate of month m is used in month m + 1
    mem = members.assign(use=members["month"].dt.to_period("M"))[["use", "sec_id"]]
    x = mem.merge(sp[["use", "sec_id", "spread"]], on=["use", "sec_id"], how="left")
    lo = x.groupby("use")["spread"].transform(lambda s: s.quantile(winsor))
    hi = x.groupby("use")["spread"].transform(lambda s: s.quantile(1 - winsor))
    x["spread"] = x["spread"].clip(lo, hi)
    med = x.groupby("use")["spread"].median()
    tab = x.pivot(index="use", columns="sec_id", values="spread")
    # non-members (a name held after leaving) keep their own estimate when there is one
    own = sp[~sp.set_index(["use", "sec_id"]).index.isin(x.set_index(["use", "sec_id"]).index)]
    if len(own):
        tab = tab.combine_first(own.pivot_table(index="use", columns="sec_id", values="spread"))
    trade_month = m.dates[t_idx - 1].to_period("M")
    S = tab.reindex(index=trade_month, columns=m.cols).to_numpy(dtype=np.float64)
    miss = np.isnan(S)
    S = np.where(miss, med.reindex(trade_month).to_numpy()[:, None], S)
    used = x[x["use"].isin(trade_month.unique())]
    info = {"member_months": int(len(used)), "member_months_filled": int(used["spread"].isna().sum())}
    return 0.5 * S, {**info, "_missing": miss}


# ------------------------------------------------------------------ transaction taxes
def load_tax_calendar(path: Path, include_optional: bool = False) -> list[dict]:
    rows = yaml.safe_load(Path(path).read_text())["taxes"]
    return [r for r in rows if include_optional or not r.get("optional", False)]


def tax_rates(m: Market, t_idx: np.ndarray, home: pd.Series, calendar: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(buy, sell): (T, P) tax per unit bought / sold on row t of t_idx (trade at the close before day t).
    home: sec_id -> ISO-2 country of incorporation. A calendar row applies from `start` (inclusive) to `end`
    (exclusive) to the companies of `country` whose prior-month cap (mktcap_lag at the trade close, EUR m,
    carried forward for a name held outside the universe) is at least `min_mktcap_eur_m`; rate in bp."""
    close = m.dates[t_idx - 1]
    cap = pd.DataFrame(wide(m, m.rows, "mktcap_lag"), index=m.dates, columns=m.cols).ffill().reindex(close)
    cap = cap.to_numpy()
    country = home.reindex(m.cols).to_numpy()
    buy, sell = np.zeros((len(t_idx), len(m.cols))), np.zeros((len(t_idx), len(m.cols)))
    for r in calendar:
        on = (close >= pd.Timestamp(r["start"])) & (close < pd.Timestamp(r.get("end") or "2100-01-01"))
        firms = country == r["country"]
        if not on.any() or not firms.any():
            continue
        big = cap >= float(r["min_mktcap_eur_m"]) if r.get("min_mktcap_eur_m") else np.ones(cap.shape, bool)
        rate = np.where(on[:, None] & firms[None, :] & big, float(r["rate_bp"]) * 1e-4, 0.0)
        buy += rate
        if r.get("side", "buy") == "both":
            sell += rate
    return buy, sell
