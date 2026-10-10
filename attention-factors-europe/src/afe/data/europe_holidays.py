"""Exchange holidays in the European returns table (the headline convention, decided 2026-10-10).

Before ~2010 Compustat Global has no row for a line on its own exchange's holidays, so a
national holiday leaves a member without a return on a pooled trading day, and the slot
model, which needs a return on every day of its 30-day lookback, drops the name for the
next 30 days. From ~2010 Compustat carries the price on those days instead (prcstd 5).

`fill_holiday_gaps` makes both periods the same:
- `traded` = the carrying line's close is a traded one (company_daily.traded, prcstd 10);
  False on Compustat's carried rows;
- a member's gap of at most `max_gap` pooled days with a return on both sides is filled
  with a zero LOCAL return: the numeraire return of a filled day is its currency's move
  against the numeraire since the previous pooled day, traded = False;
- the return of the day after the gap is divided by the filled days' FX moves, so the
  compounded return over the gap is unchanged.
With `execution.stale_when_closed` the policy keeps its position on a traded = False day
(src/afe/policy/trading.py), so the fill adds no trade at a price that did not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fill_holiday_gaps(ret: pd.DataFrame, universe: pd.DataFrame, fx: pd.DataFrame, traded: pd.DataFrame,
                      max_gap: int) -> tuple[pd.DataFrame, dict]:
    """`ret`: returns.parquet; `universe`: universe.parquet; `fx`: fx_to_numeraire_daily
    (currency, date, eur_per_unit); `traded`: (sec_id, date, traded). Returns the filled
    table (with a bool `traded` column) and counts for the build report."""
    r = ret.merge(traded[["sec_id", "date", "traded"]], on=["date", "sec_id"], how="left")
    r["traded"] = r["traded"].astype("boolean").fillna(True).astype(bool)
    cal = pd.DatetimeIndex(np.sort(r["date"].unique()))
    uni = universe.assign(ym=universe["month"].dt.to_period("M"))
    mem = pd.DataFrame({"date": cal, "ym": cal.to_period("M")}).merge(uni[["ym", "sec_id"]], on="ym")[["date", "sec_id"]]
    have = set(zip(r["sec_id"], cal.get_indexer(r["date"])))
    mem["ti"] = cal.get_indexer(mem["date"])
    miss = mem[[(s, t) not in have for s, t in zip(mem["sec_id"], mem["ti"])]].sort_values(["sec_id", "ti"], ignore_index=True)
    miss["run"] = ((miss["ti"].diff() != 1) | (miss["sec_id"] != miss["sec_id"].shift())).cumsum()
    runs = miss.groupby("run").agg(sec_id=("sec_id", "first"), t0=("ti", "min"), t1=("ti", "max"), n=("ti", "size"))
    bounded = [((s, a - 1) in have) and ((s, b + 1) in have) for s, a, b in zip(runs.sec_id, runs.t0, runs.t1)]
    runs = runs[(runs["n"] <= max_gap) & np.asarray(bounded, dtype=bool)]
    fill = miss[miss["run"].isin(runs.index)].copy()

    # currency, country and cap from the row before the gap
    prev = r.sort_values("date")[["date", "sec_id", "mktcap_lag", "country", "currency"]]
    fill = pd.merge_asof(fill.sort_values("date"), prev, on="date", by="sec_id", direction="backward")
    rate = fx.set_index(["currency", "date"])["eur_per_unit"]
    f_now = rate.reindex(pd.MultiIndex.from_arrays([fill["currency"], fill["date"]])).to_numpy()
    f_prev = rate.reindex(pd.MultiIndex.from_arrays([fill["currency"], cal[fill["ti"].to_numpy() - 1]])).to_numpy()
    fill["ret"] = (f_now / f_prev - 1).astype("float32")
    no_fx = ~np.isfinite(fill["ret"])
    fill.loc[no_fx, "ret"] = np.float32(0.0)
    fill["traded"] = False

    # the day after each gap: take the filled days' FX moves out of its gap-spanning return
    g = fill.groupby("run")["ret"].apply(lambda x: float(np.prod(1.0 + x.astype("float64"))))
    nxt = pd.DataFrame({"sec_id": runs.loc[g.index, "sec_id"].to_numpy(),
                        "date": cal[runs.loc[g.index, "t1"].to_numpy() + 1], "g": g.to_numpy()})
    r = r.merge(nxt, on=["sec_id", "date"], how="left")
    adj = r["g"].notna()
    r.loc[adj, "ret"] = ((1.0 + r.loc[adj, "ret"].astype("float64")) / r.loc[adj, "g"] - 1.0).astype("float32")
    r = r.drop(columns="g")

    out = pd.concat([r, fill[r.columns]], ignore_index=True).sort_values(["date", "sec_id"], ignore_index=True)
    for c in ("ret", "mktcap_lag"):
        out[c] = out[c].astype("float32")
    out["traded"] = out["traded"].astype(bool)
    stats = {"filled_rows": len(fill), "gaps": len(runs), "no_fx": int(no_fx.sum()),
             "next_day_adjusted": int(adj.sum()), "not_traded_rows": int((~out["traded"]).sum()),
             "filled_by_year": fill.groupby(fill["date"].dt.year).size().to_dict()}
    return out, stats
