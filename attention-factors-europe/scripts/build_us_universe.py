"""Build the monthly top-N US constituent list from Compustat Security Monthly.

    python scripts/build_us_universe.py [--config configs/universe_us_compustat.yaml] [--offline]

Writes to output.dir (gitignored):
  secm_extract.parquet          raw pull: common stock, USD, all countries
  fund_shares.parquet           fundq.cshoq / funda.csho at fiscal period ends
  company_month_mktcap.parquet  one row per (gvkey, month end), country-filtered
  top{N}_monthly.parquet/.csv   long: month end, cap_rank, gvkey, mktcap (USD mn), ...
  top{N}_gvkey_wide.csv         months x ranks of gvkey
  top{N}_mktcap_wide.csv        months x ranks of market cap
  top{N}_build_report.txt       coverage, seam audit, what the filters removed

This is the AS-OF table: month m holds the ranking by market cap at the END of m.
The universe for trading month m+1 is row m (the point-in-time lag in CLAUDE.md is
applied downstream, not here). --offline reuses the extracts on disk instead of WRDS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import compustat_us as cu  # noqa: E402


def uncovered_audit(secm, mcap, top, country_field, country, n) -> str:
    """Priced companies with no share count from any tier. Would any have made the
    top-N? Back-cast each from its FIRST cshom (future information, so this is a
    diagnostic of what the fallback misses, never an input to the ranking)."""
    s = secm[(secm[country_field] == country) & secm["datadate"].isin(top["datadate"].unique())]
    priced = s[["gvkey", "datadate"]].drop_duplicates()
    covered = mcap.loc[mcap["mktcap"].notna(), ["gvkey", "datadate"]]
    gap = priced.merge(covered, how="left", indicator=True).query("_merge == 'left_only'")
    gap = gap.drop(columns="_merge")

    first = (s[s["cshom"].notna()].sort_values("datadate")
             .drop_duplicates(["gvkey", "iid"])[["gvkey", "iid", "cshom", "ajexm"]]
             .rename(columns={"cshom": "cshom_0", "ajexm": "ajexm_0"}))
    rows = s.merge(gap, on=["gvkey", "datadate"]).merge(first, on=["gvkey", "iid"])
    rows["cap"] = rows["prccm"] * rows["cshom_0"] * rows["ajexm_0"] / rows["ajexm"] / 1e6
    back = rows.groupby(["gvkey", "datadate"], as_index=False).agg(cap=("cap", "sum"), conm=("conm", "first"))
    cutoff = top.groupby("datadate")["mktcap"].min().rename("cutoff")
    back = back.join(cutoff, on="datadate")
    # shells with absurd adjustment factors back-cast to trillions; keep names that were large at some point
    miss = back[(back["cap"] > back["cutoff"]) & back["gvkey"].isin(top["gvkey"])]
    yr = gap["datadate"].dt.year
    L = ["Priced company-months with NO share count (any tier), by year:",
         gap.groupby(yr).size().to_string(),
         f"of which back-castable from a later cshom: {len(back):,}; "
         f"back-cast cap above that month's top-{n} cutoff, among firms ever in the top-{n}: {len(miss):,} company-months"]
    if len(miss):
        L.append(miss.groupby(["gvkey", "conm"]).agg(months=("datadate", "size"), max_cap_bn=("cap", lambda x: x.max() / 1e3),
                                                     first=("datadate", "min"), last=("datadate", "max"))
                 .sort_values("months", ascending=False).head(20).to_string())
    return "\n".join(L)


def build_report(cfg, secm, mcap, top, top_all, n) -> str:
    field, country = cfg["filters"]["country_field"], cfg["filters"]["country"]
    L = [f"top-{n} US universe from comp.secm, built {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"config: {cfg}", "",
         f"extract rows {len(secm):,}  months {top['datadate'].nunique()}  "
         f"distinct gvkeys in top-{n} {top['gvkey'].nunique():,}", ""]

    year = top["datadate"].dt.year
    per_year = top.groupby(year).agg(
        cutoff_usd_bn=("mktcap", lambda s: s.min() / 1e3),
        largest_usd_bn=("mktcap", lambda s: s.max() / 1e3),
        share_fund=("shares_source", lambda s: (s == "fund").mean()),
        max_fund_age=("fund_age_months", "max"),
        reits=("sic", lambda s: (s == "6798").sum() / 12),
    )
    elig = mcap[mcap["datadate"].isin(top["datadate"].unique())]
    per_year["eligible_avg"] = elig.groupby(elig["datadate"].dt.year).size() / 12
    L += ["Per year (cutoff = smallest cap in the top-N, USD bn; reits = avg REIT count):",
          per_year.round(3).to_string(), ""]

    # Seam audit: where both measures exist, how similar are the top-N sets?
    both = mcap[mcap["mktcap_cshom"].notna() & mcap["mktcap_fund"].notna()]
    rows = []
    for d, g in both.groupby("datadate"):
        if len(g) < n:
            continue
        a = set(g.nlargest(n, "mktcap_cshom")["gvkey"])
        b = set(g.nlargest(n, "mktcap_fund")["gvkey"])
        rows.append((d.year, len(a & b) / n))
    if rows:
        seam = pd.DataFrame(rows, columns=["year", "overlap"]).groupby("year")["overlap"]
        L += [f"Seam audit, months where both share sources exist: share of the cshom top-{n} "
              f"also in the fundamentals top-{n}", seam.agg(["mean", "min"]).round(3).to_string(), ""]

    L += [uncovered_audit(secm, mcap, top, field, country, n), ""]

    # What the country filter removed, ranked by months it would have been in the top-N.
    foreign = top_all[top_all[field] != country]
    if len(foreign):
        removed = (foreign.groupby(["gvkey", "conm", field])
                   .agg(months=("datadate", "size"), max_cap_bn=("mktcap", lambda s: s.max() / 1e3),
                        first=("datadate", "min"), last=("datadate", "max"))
                   .sort_values("months", ascending=False))
        L += [f"Removed by {field} != {country}: {removed['months'].sum():,} company-months, "
              f"{len(removed)} companies. Top 25 by months:", removed.head(25).to_string(), ""]

    stale = top[top["fund_age_months"] > 3]
    L += [f"top-{n} rows built from a fundamentals share count older than 3 months: {len(stale):,}",
          "gvkey is a zero-padded string: read the CSVs with dtype={'gvkey': str}."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "universe_us_compustat.yaml")
    ap.add_argument("--offline", action="store_true", help="reuse the extracts on disk")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / cfg["output"]["dir"]
    out.mkdir(parents=True, exist_ok=True)

    start, end = str(cfg["sample"]["start"]), str(cfg["sample"]["end"])
    n = int(cfg["sample"]["universe_size"])
    stale = int(cfg["shares"]["fallback_max_stale_months"])
    y0, y1 = int(start[:4]) - 1, int(end[:4])  # one extra year so the fallback can carry in

    if args.offline:
        secm = pd.read_parquet(out / "secm_extract.parquet")
        fund = pd.read_parquet(out / "fund_shares.parquet")
        company = pd.read_parquet(out / "company_header.parquet")
    else:
        db = cu.connect(ROOT)
        secm = cu.fetch_secm(db, y0, y1, cfg["filters"]["currency"])
        fund = cu.fetch_fund_shares(db, y0, y1)
        company = cu.fetch_company(db)
        db.close()
        secm.to_parquet(out / "secm_extract.parquet", index=False)
        fund.to_parquet(out / "fund_shares.parquet", index=False)
        company.to_parquet(out / "company_header.parquet", index=False)
    print(f"extract: {len(secm):,} rows, {secm['gvkey'].nunique():,} gvkeys")

    mcap_all = cu.company_month_mktcap(secm, fund, max_stale_months=stale)
    mcap_all = mcap_all.merge(company[["gvkey", "sic", "loc", "dldte"]], on="gvkey", how="left")
    field, country = cfg["filters"]["country_field"], cfg["filters"]["country"]
    mcap = mcap_all[mcap_all[field] == country].reset_index(drop=True)
    mcap.to_parquet(out / "company_month_mktcap.parquet", index=False)
    print(f"company-months: {len(mcap):,} ({field} == {country})")

    top = cu.top_n_by_month(mcap, n, start, end)
    top_all = cu.top_n_by_month(mcap_all, n, start, end)

    tag = f"top{n}"
    top.to_parquet(out / f"{tag}_monthly.parquet", index=False)
    top.to_csv(out / f"{tag}_monthly.csv", index=False)
    cu.to_wide(top, "gvkey").to_csv(out / f"{tag}_gvkey_wide.csv")
    cu.to_wide(top, "mktcap").round(3).to_csv(out / f"{tag}_mktcap_wide.csv")
    report = build_report(cfg, secm, mcap, top, top_all, n)
    (out / f"{tag}_build_report.txt").write_text(report)
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
