"""Stage 2 of the US build: frozen raw tables -> the three tables of docs/schemas.md.

    returns.parquet    date, sec_id, ret, mktcap_lag          every day of every pool stock
    universe.parquet   month, sec_id, cap_rank                exactly N rows per month
    features.parquet   date, sec_id, char_*, med_*, rf        universe rows only

plus, for inspection, the un-normalised characteristics (characteristics_monthly.parquet,
characteristics_daily.parquet) and a build report.

Assembly rule, the one that matters for point-in-time correctness: a row (d, i) of
features.parquet carries the MONTHLY characteristics of stock i as of the end of the
month before d's month, and the DAILY characteristics as of d itself. Membership for
d's month was fixed at the same prior month end. Ranks and medians are cross-sectional
over that day's universe. `build(..., cutoff=t)` rebuilds from inputs truncated at t;
tests/test_no_lookahead.py-style prefix invariance means the rows dated t must not move.
Checked on the 2010 smoke build (cutoff 2010-09-15): features and universe identical,
returns identical on common rows. The returns table's ROW SET is not prefix-invariant by
design: the pool is "ever a member of the universe in the sample", so a full build also
carries the pre-entry history of stocks that join after t. Values never change.
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from afe import schemas
from afe.data import characteristics as ch
from afe.data import us_panel as up


def characteristic_names(cfg: dict) -> list[str]:
    return [n for theme in cfg["characteristics"].values() for n in theme]


# ------------------------------------------------------------------ characteristic tables


def monthly_characteristics(M: ch.MonthlyInputs, chars: list[ch.Characteristic]) -> pd.DataFrame:
    """Long table (month, permno) x monthly characteristics, un-normalised."""
    cols = {}
    for c in chars:
        w = c.fn(M).reindex(index=M.ret.index, columns=M.ret.columns)
        cols[c.name] = w.stack(future_stack=True)
    out = pd.DataFrame(cols)
    out.index.names = ["month", "permno"]
    return out


def daily_characteristics(D: ch.DailyInputs, chars: list[ch.Characteristic]) -> pd.DataFrame:
    cols = {c.name: c.fn(D).stack(future_stack=True) for c in chars}
    out = pd.DataFrame(cols)
    out.index.names = ["date", "permno"]
    return out


def annual_characteristics(raw: up.RawUS, co: pd.DataFrame, pool, months: pd.PeriodIndex,
                           chars: list[ch.Characteristic], cfg: dict) -> pd.DataFrame:
    """Long table (month, permno) x annual characteristics, each month carrying the
    fiscal year usable at that month end."""
    annual = up.annual_frame(raw.funda, raw.ccm, co, cfg)
    for c in chars:
        annual[c.name] = c.fn(annual)
    mapping = up.permno_gvkey(raw.ccm, pool, months)
    names = [c.name for c in chars]
    aligned = up.align_annual(annual, mapping, names, int(cfg["fundamentals"]["max_age_months"]))
    return aligned.set_index(["month", "permno"])[names]


# ------------------------------------------------------------------ assembly


def assemble_year(year: int, uni_members: dict, D: ch.DailyInputs, mc: pd.DataFrame, dc: pd.DataFrame,
                  names: list[str], rf: pd.Series, start: pd.Period, end: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    """features rows for one calendar year, plus the raw (pre-normalisation) coverage."""
    dates = D.ret.index[D.ret.index.year == year]
    dc = dc[dc.index.get_level_values("date").year == year]
    parts = []
    for M, dM in dates.groupby(dates.to_period("M")).items():
        if M < start or M > end or M not in uni_members:
            continue
        members = uni_members[M]
        idx = pd.MultiIndex.from_product([dM, members], names=["date", "permno"])
        rows = pd.DataFrame(index=idx)
        rows = rows.join(dc, how="left")                      # daily characteristics as of d
        prev = pd.MultiIndex.from_arrays([np.repeat(M - 1, len(rows)), rows.index.get_level_values("permno")],
                                         names=["month", "permno"])
        monthly_vals = mc.reindex(prev)                        # monthly/annual as of end of M-1
        monthly_vals.index = rows.index
        rows = rows.join(monthly_vals, how="left")
        parts.append(rows)
    # float64, not nullable Float64: pandas' masked-array rank ignores the NA mask and
    # ranks the garbage under it (diagnosed 2026-09-16 on the first build)
    raw = pd.concat(parts)[names].astype("float64")
    coverage = raw.notna().mean().rename(year)

    date_key = raw.index.get_level_values("date")
    g = raw.groupby(date_key)
    ranked = g.transform(ch.rank_quantile).astype("float64").fillna(0.0)
    with warnings.catch_warnings():  # an all-NaN day for one characteristic is a coverage fact, reported below
        warnings.simplefilter("ignore", RuntimeWarning)
        meds = g.transform("median").astype("float64").fillna(0.0)
    ranked.columns = [f"char_{c}" for c in names]
    meds.columns = [f"med_{c}" for c in names]
    feat = pd.concat([ranked, meds], axis=1).astype("float32")
    feat["rf"] = rf.reindex(date_key).to_numpy().astype("float32")
    feat = feat.reset_index()
    feat["sec_id"] = feat["permno"].astype(str)
    feat = feat.drop(columns="permno")
    feat["date"] = feat["date"].astype("datetime64[ns]")
    feat = feat[["date", "sec_id"] + schemas.feature_columns(feat)].sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(feat, schemas.FEATURES)
    return feat, coverage


def returns_table(daily: pd.DataFrame, co: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.DataFrame:
    """docs/schemas.md returns table: every day with a return for every pool stock,
    mktcap_lag = company cap (USD mn) at the end of the previous month."""
    d = daily[["date", "permno", "permco", "ret"]].copy()
    d["month_prev"] = d["date"].dt.to_period("M") - 1
    d = d[(d["month_prev"] + 1 >= start) & (d["month_prev"] + 1 <= end)]
    d = d.merge(co[["permco", "month", "cap_co"]].rename(columns={"month": "month_prev", "cap_co": "mktcap_lag"}),
                on=["permco", "month_prev"], how="left")
    d = d[d["ret"].notna() & d["mktcap_lag"].notna()]
    out = pd.DataFrame({"date": d["date"].astype("datetime64[ns]"), "sec_id": d["permno"].astype(str),
                        "ret": d["ret"].astype("float32"), "mktcap_lag": d["mktcap_lag"].astype("float32")})
    out = out.sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(out, schemas.RETURNS)
    return out


# ------------------------------------------------------------------ driver


def build(cfg: dict, raw: up.RawUS, out_dir: Path, cutoff: pd.Timestamp | None = None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = pd.Period(cfg["sample"]["start"], "M"), pd.Period(cfg["sample"]["end"], "M")
    if cutoff is not None:
        raw = raw.truncate(cutoff)
        end = min(end, pd.Period(cutoff, "M"))
    windows = cfg["windows"]
    names = characteristic_names(cfg)
    groups = ch.by_frequency(names)

    log("eligibility and companies")
    elig = up.eligible_monthly(raw.monthly, cfg)
    co = up.company_month(elig)
    calendar = raw.ff_daily.index
    uni, asof = up.build_universe(co, {**cfg, "sample": {**cfg["sample"], "end": str(end)}}, calendar)
    uni_members = {M: sorted(g["primary_permno"].astype(int).tolist()) for M, g in asof.groupby("month")}
    pool = sorted({p for ms in uni_members.values() for p in ms}) if cfg["pool"] == "universe" \
        else sorted(elig["permno"].unique().tolist())
    log(f"  universe {uni['month'].nunique()} months, pool {len(pool)} permnos")

    log("daily data")
    years = range(int(cfg["raw"]["start_year"]), end.year + 1)
    daily, stats = up.load_daily(raw.daily_dir, pool, years, raw.ff_daily, cutoff=cutoff, delisting=raw.delisting)
    D = up.daily_inputs(daily, raw.ff_daily, windows)
    M = up.monthly_inputs(raw.monthly, co, pool, stats, D, windows)
    n_del = int(raw.delisting["permno"].isin(pool).sum()) if len(raw.delisting) else 0
    log(f"  {len(daily):,} daily rows, {D.ret.shape[0]} days x {D.ret.shape[1]} stocks, "
        f"{n_del} delisting-day rows merged")

    log("characteristics")
    mc = monthly_characteristics(M, groups["monthly"])
    ac = annual_characteristics(raw, co, pool, M.ret.index, groups["annual"], cfg)
    mc = mc.join(ac, how="left")[[n for n in names if n in mc.columns or n in ac.columns]]
    dc = daily_characteristics(D, groups["daily"])
    mc.reset_index().to_parquet(out_dir / "characteristics_monthly.parquet", index=False)
    dc.reset_index().to_parquet(out_dir / "characteristics_daily.parquet", index=False)

    log("features")
    rf = raw.ff_daily["rf"]
    writer, coverage = None, []
    for year in range(start.year, end.year + 1):
        feat, cov = assemble_year(year, uni_members, D, mc, dc, names, rf, start, end)
        coverage.append(cov)
        table = pa.Table.from_pandas(feat, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "features.parquet", table.schema)
        writer.write_table(table)
        log(f"  {year}: {len(feat):,} rows")
    writer.close()

    log("returns and universe")
    ret = returns_table(daily, co, start, end)
    ret.to_parquet(out_dir / "returns.parquet", index=False)
    uni.to_parquet(out_dir / "universe.parquet", index=False)
    asof.to_parquet(out_dir / "universe_asof.parquet", index=False)

    report = build_report(cfg, uni, asof, ret, pd.DataFrame(coverage), mc, raw, out_dir)
    (out_dir / "build_report.txt").write_text(report)
    return {"universe": uni, "coverage": pd.DataFrame(coverage), "report": report}


def build_report(cfg, uni, asof, ret, coverage, mc, raw, out_dir: Path) -> str:
    L = [f"US dataset built {dt.datetime.now():%Y-%m-%d %H:%M}", f"config: {cfg}", "",
         f"universe: {uni['month'].nunique()} months x {cfg['sample']['universe_size']}, "
         f"{uni['sec_id'].nunique()} distinct permnos; returns.parquet {len(ret):,} rows, "
         f"{ret['sec_id'].nunique()} permnos", ""]
    if len(raw.delisting):
        dl = raw.delisting[raw.delisting["permno"].isin(ret["sec_id"].astype(int).unique())]
        key = dl.assign(sec_id=dl["permno"].astype(str))[["sec_id", "date"]]
        inret = ret.merge(key, on=["sec_id", "date"])
        um = uni.assign(month=uni["month"].dt.to_period("M"))[["month", "sec_id"]]
        held = inret.assign(month=inret["date"].dt.to_period("M")).merge(um, on=["month", "sec_id"])
        L += [f"Delisting-day rows (crsp_delisting.parquet, dlydelflg = Y) of pool permnos: {len(dl):,}; in "
              f"returns.parquet: {len(inret):,}; while a universe member that month: {len(held):,}; their returns: "
              f"mean {held['ret'].mean():+.4f}, median {held['ret'].median():+.4f}, min {held['ret'].min():+.3f}, "
              f"max {held['ret'].max():+.3f}", ""]
    else:
        L += ["Delisting-day rows: crsp_delisting.parquet NOT present; returns lack the delisting return (see wrds_us).", ""]
    yr = asof["month"].dt.year
    per_year = asof.groupby(yr).agg(cutoff_usd_bn=("cap_co", lambda s: s.min() / 1e3),
                                    largest_usd_bn=("cap_co", lambda s: s.max() / 1e3),
                                    multi_class=("n_classes", lambda s: (s > 1).mean()))
    L += ["Universe per year (cap in USD bn; multi_class = share of members with >1 class):",
          per_year.round(3).to_string(), ""]
    L += ["Coverage of each characteristic within the universe, share of rows non-missing before the "
          "median fill (rows = year):", coverage.round(3).to_string(), ""]

    comp = out_dir / "top500_monthly.parquet"
    if comp.exists():
        c = pd.read_parquet(comp)
        c["rank_month"] = c["datadate"].dt.to_period("M")
        months = pd.PeriodIndex(sorted(asof["month"].unique()), freq="M")
        mapping = up.permno_gvkey(raw.ccm, asof["primary_permno"].unique(), months - 1)
        x = asof.assign(rank_month=asof["month"] - 1).merge(
            mapping.rename(columns={"month": "rank_month", "permno": "primary_permno"}),
            on=["primary_permno", "rank_month"], how="left")
        rows = []
        for m, g in x.groupby("rank_month"):
            comp_set = set(c.loc[c["rank_month"] == m, "gvkey"])
            if comp_set:
                rows.append((m.year, len(set(g["gvkey"].dropna()) & comp_set) / len(g)))
        if rows:
            ov = pd.DataFrame(rows, columns=["year", "overlap"]).groupby("year")["overlap"].agg(["mean", "min"])
            L += ["Overlap with the Compustat-secm top-500 of step 1 (same ranking month, via CCM):",
                  ov.round(3).to_string(), ""]

    desc = mc.describe(percentiles=[0.05, 0.5, 0.95]).T[["count", "5%", "50%", "95%"]]
    L += ["Raw monthly characteristics over the pool (before ranking):", desc.round(4).to_string()]
    return "\n".join(L)
