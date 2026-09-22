"""Stage 2a of the European build: raw tables -> EUR market data in docs/schemas.md form.

    python scripts/build_eu_returns.py [--config configs/eu_data.yaml] [--cutoff YYYY-MM-DD]

Writes to output.dir = data/eu/shared (gitignored):
  fx_to_eur_daily.parquet       (date, currency) -> eur_per_unit, dem_per_unit, sources
  returns.parquet               schema table: date, sec_id (gvkey), ret (EUR), mktcap_lag, country, currency
  universe.parquet              schema table: month (first trading day), sec_id, cap_rank
  market_daily.parquet/.csv     date, mkt_vw, mkt_ew, n: EUR return of the month's universe
  rf_daily.parquet              date, rf: daily simple euro money-market rate (if the ECB file exists)
  returns_build_report.txt      coverage, the numeraire seam, foreign-quoted lines, market stats
and to output.inspect_dir = data/eu/private (not shipped downstream):
  returns_detail_daily.parquet  (gvkey, iid, date): EUR price, local and EUR returns, volume

Conventions are in src/afe/data/eu_panel.py and configs/eu_data.yaml. --cutoff rebuilds
from inputs truncated at that date (prefix-invariance check, as for the US build).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import eu_panel as ep  # noqa: E402

pd.set_option("display.width", 200)


def load_rf(path: Path, day_count: int) -> pd.DataFrame | None:
    """ECB file: date, rate_pct_pa (annualised percent) -> daily simple rate, forward
    filled across non-quoted days."""
    if not path.exists():
        return None
    rf = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    rf["rf"] = pd.to_numeric(rf["rate_pct_pa"], errors="coerce") / 100.0 / day_count
    return rf[["date", "rf", "series"]] if "series" in rf else rf[["date", "rf"]]


def build_report(cfg, fxtab, dret, ret, uni, mkt, rf, calendar) -> str:
    L = [f"EU stage 2a build {dt.datetime.now():%Y-%m-%d %H:%M}; config: {cfg['currency']}, {cfg['returns']}, {cfg['market']}", ""]
    u_days = ret.merge(uni.assign(month=uni["month"].dt.to_period("M"))[["month", "sec_id"]],
                       left_on=[ret["date"].dt.to_period("M"), "sec_id"], right_on=["month", "sec_id"])
    L += [f"calendar: {len(calendar):,} pooled trading days {calendar.min().date()} .. {calendar.max().date()} "
          f"(>= {cfg['returns']['calendar_min_lines']} traded closes)",
          f"returns_detail: {len(dret):,} listing-days, {dret.groupby(['gvkey', 'iid'], observed=True).ngroups:,} listings; "
          f"EUR return present on {dret['ret'].notna().mean():.4f}, traded close on {dret['traded'].mean():.4f}",
          f"returns.parquet: {len(ret):,} rows, {ret['sec_id'].nunique():,} companies, {ret['date'].min().date()} .. {ret['date'].max().date()}",
          f"universe.parquet: {uni['month'].nunique()} months x {uni.groupby('month').size().iloc[0]}", ""]

    # coverage of the universe by returns, per year
    u = uni.assign(month=uni["month"].dt.to_period("M"))
    r = ret.assign(month=ret["date"].dt.to_period("M"))
    days_per_month = r.groupby("month")["date"].nunique()
    have = r.merge(u[["month", "sec_id"]], on=["month", "sec_id"]).groupby(["month", "sec_id"]).size().rename("days").reset_index()
    cov = u[["month", "sec_id"]].merge(have, on=["month", "sec_id"], how="left").fillna({"days": 0})
    cov["full"] = cov["days"] >= 0.8 * cov["month"].map(days_per_month)
    by_year = cov.groupby(cov["month"].dt.year).agg(members=("sec_id", "size"), any_return=("days", lambda s: (s > 0).mean()),
                                                    full_month=("full", "mean"))
    L += ["Universe members with a return in their month (share), by year:", by_year.round(4).to_string(), ""]
    missing = cov[cov["days"] == 0]
    L += [f"member-months with NO return at all: {len(missing):,} of {len(cov):,}", ""]

    # the numeraire seam at the euro changeover: EUR return through Compustat's synthetic
    # EUR of 1998-12-31 versus the return at the irrevocable conversion rates
    b = dret[dret["date"] == pd.Timestamp("1998-12-31")][["gvkey", "iid", "curcdd", "prccd", "ajexdi", "trfd"]]
    b = b.rename(columns={"curcdd": "ccy_before", "prccd": "p0", "ajexdi": "a0", "trfd": "t0"})
    j = dret[dret["date"] == pd.Timestamp("1999-01-04")].merge(b, on=["gvkey", "iid"])
    j["ccy_before"] = j["ccy_before"].astype(str)
    j = j[(j["curcdd"].astype(str) == "EUR") & j["ccy_before"].isin(ep.EURO_FIXED_RATES)]
    fixed = j["ccy_before"].map(ep.EURO_FIXED_RATES)
    j["ret_fixed"] = (j["prccd"] / j["ajexdi"] * j["trfd"]) / (j["p0"] / fixed / j["a0"] * j["t0"]) - 1
    j["artefact"] = j["ret"] - j["ret_fixed"]
    if len(j):
        L += ["1999-01-04, first EUR quotation day: EUR return (config numeraire: Compustat's synthetic EUR on 1998-12-31)",
              "minus the return at the irrevocable conversion rates, by the currency quoted on 1998-12-31 (one-day artefact):",
              j.groupby("ccy_before", observed=True)["artefact"].agg(["size", "mean", "min", "max"]).round(4).to_string(), ""]

    # foreign-quoted lines in the schema table
    fq = ret[ret["currency"] != "EUR"]
    L += [f"returns.parquet rows quoted in a non-EUR currency (FX move inside the EUR return): {len(fq):,} "
          f"({len(fq) / len(ret):.3%}); by currency: {fq['currency'].value_counts().to_dict()}",
          f"of which universe member-days: {len(fq.assign(month=fq['date'].dt.to_period('M')).merge(u, on=['month', 'sec_id'])):,}", ""]
    pre = ret[ret["date"] < "1999-01-01"]
    L += [f"rows before 1999 (legacy currencies via Compustat's synthetic EUR): {len(pre):,}; currencies "
          f"{pre['currency'].value_counts().head(8).to_dict()}", ""]

    # market
    m = mkt.set_index("date")
    yr = m.groupby(m.index.year).agg(mean_ann=("mkt_vw", lambda s: s.mean() * 252), vol_ann=("mkt_vw", lambda s: s.std() * np.sqrt(252)),
                                     ew_mean_ann=("mkt_ew", lambda s: s.mean() * 252), n_min=("n", "min"), n_med=("n", "median"), days=("n", "size"))
    L += ["Market (value-weighted EUR return of the universe) by year: annualised mean and vol, members with a return:",
          yr.round(3).to_string(), ""]
    L += [f"largest daily moves: {m['mkt_vw'].nsmallest(3).round(4).to_dict()} / {m['mkt_vw'].nlargest(3).round(4).to_dict()}", ""]
    if rf is not None:
        L += [f"rf_daily: {len(rf):,} days {rf['date'].min().date()} .. {rf['date'].max().date()}, "
              f"mean annualised {rf['rf'].mean() * 360:.4f}", ""]
    else:
        L += ["rf_daily: NOT built; euro money-market file missing (config risk_free.file).", ""]

    # delistings: companies whose returns end before the sample end
    last = ret.groupby("sec_id")["date"].max()
    ended = last[last < ret["date"].max() - pd.Timedelta(days=40)]
    L += [f"companies whose return series ends before the sample end (delisting, merger, or dormant line): {len(ended):,} "
          f"of {ret['sec_id'].nunique():,}; no delisting return applied (config returns.delisting_return = "
          f"{cfg['returns']['delisting_return']})", "gvkey is a zero-padded string: read CSVs with dtype={'sec_id': str}."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "eu_data.yaml")
    ap.add_argument("--cutoff", default=None, help="truncate every input at this date (prefix test)")
    ap.add_argument("--out", default=None, help="output directory (default output.dir, or output.inspect_dir/cutoff_<date>)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["raw"]["dir"]
    out = ROOT / cfg["output"]["dir"]
    inspect = ROOT / cfg["output"].get("inspect_dir", cfg["output"]["dir"])
    if args.out:
        out = inspect = Path(args.out)
    elif args.cutoff:
        out = inspect = inspect / f"cutoff_{args.cutoff}"
    out.mkdir(parents=True, exist_ok=True)
    inspect.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    cur, rc, u = cfg["currency"], cfg["returns"], cfg["universe"]
    n = int(u["size"])
    start, end = str(u["start"]), str(u["end"])
    if cutoff is not None:
        end = min(pd.Period(end, "M"), cutoff.to_period("M")).strftime("%Y-%m")

    raw = ep.load_raw(ROOT / cfg["universe"]["mktcap_table"], raw_dir)
    if cutoff is not None:
        raw = raw.truncate(cutoff)
    fxtab = ep.fx_conversion_table(raw.fx, switch=str(cur["numeraire_switch"]), dem_per_eur=float(cur["dem_per_eur"]))
    fxtab.to_parquet(out / "fx_to_eur_daily.parquet", index=False)
    print(f"fx table: {len(fxtab):,} rows, {fxtab['currency'].nunique()} currencies, "
          f"{fxtab['date'].min().date()} .. {fxtab['date'].max().date()}", flush=True)

    # pool: every home line of an ever-member (the lines stage 1 pulled)
    lines = pd.read_csv(raw_dir / "listings.csv", dtype=str)
    years = range(int(cfg["raw"]["daily_start_year"]), int(cfg["raw"]["daily_end_year"]) + 1)
    daily = ep.load_daily(raw.daily_dir, lines, years, cutoff)
    for c in ("gvkey", "iid", "curcdd"):
        daily[c] = daily[c].astype("category")
    print(f"daily rows: {len(daily):,}", flush=True)
    calendar = ep.trading_calendar(daily, int(rc["calendar_min_lines"]))
    dret = ep.daily_returns(daily, fxtab)
    del daily
    dret.to_parquet(inspect / "returns_detail_daily.parquet", index=False)
    print(f"returns detail: {len(dret):,} rows", flush=True)

    # universe with the one-month lag, then the schema returns table and the market
    ranking_end = (pd.Period(end, "M") - 1).strftime("%Y-%m")
    asof = ep.asof_ranking(raw.mktcap[raw.mktcap["datadate"] <= pd.Period(ranking_end, "M").end_time], n)
    first_universe = (pd.Period(start, "M") + 1).strftime("%Y-%m")
    uni, detail = ep.universe_lagged(asof, calendar, first_universe, end, n)
    uni.to_parquet(out / "universe.parquet", index=False)
    d2 = dret.copy()
    for c in ("gvkey", "iid", "curcdd"):
        d2[c] = d2[c].astype(str)
    ret = ep.returns_table(d2, raw.mktcap, first_universe, end, rc["delisting_return"], calendar)
    ret.to_parquet(out / "returns.parquet", index=False)
    mkt = ep.market_returns(ret, uni)
    mkt.to_parquet(out / "market_daily.parquet", index=False)
    mkt.round(8).to_csv(out / "market_daily.csv", index=False)
    print(f"universe: {uni['month'].nunique()} months; returns: {len(ret):,} rows; market: {len(mkt):,} days", flush=True)

    rf = load_rf(ROOT / cfg["risk_free"]["file"], int(cfg["risk_free"]["day_count"]))
    if rf is not None:
        rf = rf.set_index("date").reindex(calendar, method="ffill").rename_axis("date").reset_index()
        rf.to_parquet(out / "rf_daily.parquet", index=False)

    report = build_report(cfg, fxtab, dret, ret, uni, mkt, rf, calendar)
    (out / "returns_build_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
