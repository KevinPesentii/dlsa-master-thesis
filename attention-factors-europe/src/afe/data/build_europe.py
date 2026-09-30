"""Stage 2 of the European builds (europe17, europe12): frozen raw tables -> the tables of
docs/schemas.md in the ECU/EUR numeraire, plus the JKP table and the coverage tables the
availability report reads.

    shared/  returns.parquet, universe.parquet, features.parquet (schema tables),
             market_daily.parquet (top-N mkt_vw / mkt_ew / n, FF Europe mkt_ff, mkt = the
             market behind mktrf), factors_daily.parquet (mktrf, smb, hml, rf per config
             factors.source; smb_top_n / hml_top_n = the top N's own sort), rf_daily.parquet,
             fx_to_numeraire_daily.parquet, build_report.txt
    private/ returns_detail_daily.parquet, company_daily.parquet, universe_asof.parquet,
             characteristics_monthly.parquet, characteristics_daily.parquet,
             jkp_monthly.parquet, coverage_member_months.parquet, jkp_coverage.parquet

Assembly is build_us's: a features row (d, i) carries monthly and annual characteristics
of company i as of the end of the month before d's month and daily ones as of d; ranks and
medians are over that day's universe. `build(..., cutoff=t)` rebuilds from inputs
truncated at t (prefix invariance, as for the other builds).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from afe.data import build_us as bu
from afe.data import characteristics as ch
from afe.data import eu_panel as ep
from afe.data import europe_accounts as xa
from afe.data import europe_coverage as xc
from afe.data import europe_fx as efx
from afe.data import europe_panel as xp
from afe.data import us_panel as up


def build(cfg: dict, root: Path, out_dir: Path, inspect_dir: Path, cutoff: pd.Timestamp | None = None,
          log=print, jkp: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    inspect_dir.mkdir(parents=True, exist_ok=True)
    u, n = cfg["universe"], int(cfg["universe"]["size"])
    start, end = pd.Period(str(u["start"]), "M"), pd.Period(str(u["end"]), "M")
    raw = xp.load_raw(root, cfg)
    if cutoff is not None:
        raw = raw.truncate(cutoff)
        end = min(end, pd.Period(cutoff, "M"))
    names = bu.characteristic_names(cfg)
    groups = ch.by_frequency(names)
    windows = cfg["windows"]

    log("numeraire and daily data")
    fxtab = efx.conversion_table(raw.fx, int(cfg["filters"]["fx_max_stale_days"]))
    fxtab.to_parquet(out_dir / "fx_to_numeraire_daily.parquet", index=False)
    years = range(int(cfg["raw"]["daily_start_year"]), end.year + 1)
    daily = ep.load_daily(raw.daily_dir, raw.lines, years, cutoff)
    calendar = ep.trading_calendar(daily, int(cfg["returns"]["calendar_min_lines"]))
    dret = ep.daily_returns(daily, fxtab)
    del daily
    for c in ("gvkey", "iid", "curcdd"):
        dret[c] = dret[c].astype(str)
    dret["cs"] = xp.corwin_schultz(dret[["gvkey", "iid", "prchd", "prcld", "prccd", "prcstd"]])
    dret["high_eur"] = dret["prc_eur"] * dret["prchd"] / dret["prccd"]
    dret.to_parquet(inspect_dir / "returns_detail_daily.parquet", index=False)
    log(f"  {len(dret):,} listing-days, calendar {len(calendar):,} days {calendar.min().date()} .. {calendar.max().date()}")

    log("universe and returns")
    ranking = raw.ranking[raw.ranking["datadate"].dt.to_period("M") <= end - 1]
    uni, detail = ep.universe_lagged(ranking, calendar, str(start), str(end), n)
    uni.to_parquet(out_dir / "universe.parquet", index=False)
    detail.to_parquet(inspect_dir / "universe_asof.parquet", index=False)
    ret = ep.returns_table(dret, raw.mktcap, str(start), str(end), cfg["returns"]["delisting_return"], calendar)
    ret.to_parquet(out_dir / "returns.parquet", index=False)
    cd = xp.company_daily(dret, raw.mktcap, calendar)
    cd.to_parquet(inspect_dir / "company_daily.parquet", index=False)
    pool = sorted(raw.lines["gvkey"].unique())
    log(f"  universe {uni['month'].nunique()} months x {n}, returns {len(ret):,} rows, pool {len(pool):,} companies")

    log("market, rf, fundamentals, factors")
    members = xp.members_by_month(ranking, n)
    mkt = xp.value_weighted(cd, members).reindex(calendar)
    rf = xp.risk_free(root / cfg["risk_free"]["external_dir"], calendar, int(cfg["risk_free"]["day_count"]))
    rf_s = rf.set_index("date")["rf"]
    fc = cfg["factors"]
    if fc.get("source", "top_n") not in ("top_n", "ff_europe"):
        raise ValueError(f"factors.source: {fc['source']}")
    ffe = None
    if fc.get("source") == "ff_europe":
        tb = root / fc["tbill_monthly"] if fc.get("tbill_monthly") else None
        ffe = xp.ff_europe(root / fc["file"], fxtab, calendar, tb)
        mkt["mkt_ff"] = ffe["mkt"]
    f = xa.funda_numeraire(raw.funda, fxtab)
    annual = xa.annual_frame(f, raw.mktcap, cfg)
    for c in groups["annual"]:
        annual[c.name] = c.fn(annual)
    months = pd.period_range(calendar.min().to_period("M"), end, freq="M")
    mapping = pd.MultiIndex.from_product([pool, months], names=["permno", "month"]).to_frame(index=False)
    mapping["gvkey"] = mapping["permno"]
    a_names = [c.name for c in groups["annual"]]
    missing = cfg["normalisation"].get("missing", "median")
    aligned = up.align_annual(annual, mapping, a_names, int(cfg["fundamentals"]["max_age_months"]),
                              missing == "last_observed")
    beme = aligned[["month", "permno", "BEME"]].rename(columns={"permno": "gvkey"})
    fac = xp.smb_hml(cd, members, beme, int(fc["min_per_portfolio"])).reindex(calendar)
    own = pd.DataFrame({"mkt": mkt["mkt_vw"], "smb": fac["smb"], "hml": fac["hml"]}, index=calendar)
    used = own.copy()
    if ffe is not None and ffe["mkt"].first_valid_index() is not None:
        on = calendar >= ffe["mkt"].first_valid_index()                        # the top N's own before the file
        used.loc[on] = ffe.loc[on, ["mkt", "smb", "hml"]].to_numpy()
    mkt["mkt"] = used["mkt"]
    ff = pd.DataFrame({"mktrf": used["mkt"] - rf_s, "smb": used["smb"], "hml": used["hml"], "rf": rf_s}, index=calendar)
    ff.index.name = "date"
    mkt.rename_axis("date").reset_index().to_parquet(out_dir / "market_daily.parquet", index=False)
    ff.assign(smb_top_n=fac["smb"], hml_top_n=fac["hml"]).reset_index().to_parquet(
        out_dir / "factors_daily.parquet", index=False)
    rf.to_parquet(out_dir / "rf_daily.parquet", index=False)

    log("characteristics")
    carry = cfg["normalisation"]["carry_max"] if missing == "last_observed" else {"daily": 0, "monthly": 0}
    stats = xp.monthly_daily_stats(cd, ff)
    D = xp.daily_inputs(cd, ff, windows, calendar)
    mf = xp.monthly_frame(cd, raw.mktcap)
    M = xp.monthly_inputs(mf, pool, stats, D, windows)
    monthly = [resid_var_where_factors(c, ff, int(windows["variance_months"])) if c.name == "Resid_Var" else c
               for c in groups["monthly"]]
    mc = bu.monthly_characteristics(M, monthly, int(carry["monthly"]))
    ac = aligned.set_index(["month", "permno"])[a_names]
    mc = mc.join(ac, how="left")[[x for x in names if x in mc.columns or x in ac.columns]]
    dc = bu.daily_characteristics(D, groups["daily"], int(carry["daily"]))
    mc.reset_index().to_parquet(inspect_dir / "characteristics_monthly.parquet", index=False)
    dc.reset_index().to_parquet(inspect_dir / "characteristics_daily.parquet", index=False)

    log("features")
    uni_members = {m: sorted(g["gvkey"].astype(str).tolist()) for m, g in detail.groupby("month")}
    writer, coverage = None, []
    for year in range(start.year, end.year + 1):
        feat, cov = bu.assemble_year(year, uni_members, D, mc, dc, names, rf_s, start, end)
        coverage.append(cov)
        table = pa.Table.from_pandas(feat, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "features.parquet", table.schema)
        writer.write_table(table)
        log(f"  {year}: {len(feat):,} rows")
    writer.close()
    coverage = pd.DataFrame(coverage)

    log("coverage tables and JKP")
    cmm = xc.member_month_coverage(detail, mc, dc, names, calendar)
    cmm.to_parquet(inspect_dir / "coverage_member_months.parquet", index=False)
    inputs = xc.input_coverage(detail, cd, f, calendar)
    inputs.to_parquet(inspect_dir / "input_coverage_member_months.parquet", index=False)
    jkp_cov = build_jkp(cfg, root, detail, fxtab, rf, pool, end, inspect_dir, log) if jkp else None

    report = xc.build_report(cfg, uni, detail, ret, calendar, mkt, ff.assign(smb_top_n=fac["smb"], hml_top_n=fac["hml"]),
                             rf, coverage, cmm, fxtab, raw, f, jkp_cov)
    (out_dir / "build_report.txt").write_text(report, encoding="utf-8")
    return {"universe": uni, "coverage": coverage, "report": report}


def resid_var_where_factors(c: ch.Characteristic, ff: pd.DataFrame, k: int) -> ch.Characteristic:
    """Resid_Var only where every day of its k-month window has all three factors. The
    US factors never miss a day; the European SMB/HML do in a month whose 2x3 sort has an
    empty portfolio, and the per-month sums of characteristics.resid_var would then pool a
    month with factors and one without into one inconsistent regression."""
    have = ff[["mktrf", "smb", "hml"]].notna().all(axis=1)
    ok = have.groupby(have.index.to_period("M")).all()
    window_ok = ok.astype(float).rolling(k, min_periods=k).min() == 1.0

    def fn(M):
        v = c.fn(M)
        keep = window_ok.reindex(v.index).fillna(False).astype(bool)
        return v.mul(pd.Series(np.where(keep, 1.0, np.nan), index=v.index), axis=0)
    return ch.Characteristic(c.name, c.theme, c.frequency, c.reference, fn, c.doc)


def build_jkp(cfg: dict, root: Path, detail: pd.DataFrame, fxtab: pd.DataFrame, rf: pd.DataFrame, pool,
              end: pd.Period, inspect_dir: Path, log=print) -> pd.DataFrame | None:
    """The version's JKP table (jkp_monthly.parquet) and its coverage (jkp_coverage.parquet).
    Separate from the rest so it can run once the slow JKP pull is complete
    (scripts/build_europe_dataset.py --jkp-only reads the stage-2 outputs it needs)."""
    rf_s = rf.set_index("date")["rf"]
    rf_month = (1.0 + rf_s).groupby(rf_s.index.to_period("M")).prod() - 1.0
    raw = root / cfg["raw"]["dir"]
    jk = load_jkp(raw / "jkp", range(int(cfg["raw"]["jkp_start_year"]), end.year + 1), pool)
    if jk is None:
        log("  JKP: no files")
        return None
    mem = detail[["month", "gvkey", "iid", "country"]].copy()
    jt = xa.jkp_table(jk, mem, fxtab, rf_month)
    jt.to_parquet(inspect_dir / "jkp_monthly.parquet", index=False)
    cov = xc.jkp_coverage(jt, detail)
    cov.to_parquet(inspect_dir / "jkp_coverage.parquet", index=False)
    years = sorted(pd.to_datetime(jk["eom"]).dt.year.unique())
    log(f"  JKP: {len(jk):,} rows of {jk['gvkey'].nunique():,} companies, eom years {years[0]}-{years[-1]}; "
        f"table {len(jt):,} member-months")
    return cov


def build_jkp_only(cfg: dict, root: Path, log=print) -> pd.DataFrame | None:
    """build_jkp from the saved stage-2 outputs of this version."""
    out_dir, ins = root / cfg["output"]["dir"], root / cfg["output"]["inspect_dir"]
    detail = pd.read_parquet(ins / "universe_asof.parquet")
    detail["gvkey"], detail["iid"] = detail["gvkey"].astype(str), detail["iid"].astype(str)
    fxtab = pd.read_parquet(out_dir / "fx_to_numeraire_daily.parquet")
    rf = pd.read_parquet(out_dir / "rf_daily.parquet")
    pool = sorted(pd.read_csv(ins / "listings.csv", dtype=str)["gvkey"].unique())
    end = pd.Period(str(cfg["universe"]["end"]), "M")
    return build_jkp(cfg, root, detail, fxtab, rf, pool, end, ins, log)


def load_jkp(jkp_dir: Path, years, gvkeys) -> pd.DataFrame | None:
    """The union pull's JKP rows of this version's companies only (444 columns: memory)."""
    keep = sorted(set(gvkeys))
    parts = [pq.read_table(jkp_dir / f"{y}.parquet", filters=[("gvkey", "in", keep)]).to_pandas()
             for y in years if (jkp_dir / f"{y}.parquet").exists()]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out["gvkey"], out["iid"] = out["gvkey"].astype(str).astype(object), out["iid"].astype(str).astype(object)
    return out
