"""Build the monthly company market cap table for the European leg from Compustat Global.

    python scripts/build_eu_mktcap.py [--config configs/universe_eu_compustat.yaml] [--offline]

Writes to output.dir (gitignored):
  secd_monthend_extract.parquet   raw pull: g_secd month-end rows, common + preferred, on
                                  exchanges in the configured countries
  security_header.parquet         g_security for those exchanges (excntry, isin, ...)
  company_header.parquet          g_company (prirow, fic, loc, sic, ...)
  fx_daily.parquet                g_exrt_dly GBP cross rates for the currencies seen
  company_month_mktcap.parquet    one row per (gvkey, month end), ALL companies with a cap,
                                  `eligible` flag; the .csv holds eligible rows only
  cutoffs_pooled_yearly.csv       cap at pooled ranks, median of the 12 month ends per year
  cutoffs_country_yearly.csv      same per country
  cutoffs_snapshots.csv           per-country cutoffs at the snapshot month ends
  composition_pooled.csv          country mix of the pooled top-N at the snapshot month ends
  mktcap_build_report.txt         coverage, cutoffs, what the filters removed

This is the AS-OF table: month m holds the cap at the END of m; the universe for
trading month m+1 is row m. No top-N is chosen here; that is what the report is for.
--offline reuses the extracts on disk instead of WRDS.
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

from afe.data import compustat_global as cg  # noqa: E402
from afe.data import compustat_us as cu  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def yearly_median_cutoffs(mcap: pd.DataFrame, ranks: list[int], by: list[str]) -> pd.DataFrame:
    """cutoff_table per month end, then the median over the month ends of each year."""
    t = cg.cutoff_table(mcap, ranks, by + ["datadate"]).reset_index()
    t["year"] = t["datadate"].dt.year
    return t.drop(columns="datadate").groupby(by + ["year"]).median()


def bn(df: pd.DataFrame) -> pd.DataFrame:
    """EUR mn -> EUR bn for the rank columns, counts untouched."""
    out = df.copy()
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out[rank_cols] = (out[rank_cols] / 1e3).round(2)
    return out


def build_report(cfg, secd, mcap_all, mcap, snaps, fx) -> str:
    ccy = cfg["filters"]["currency"]
    pooled_ranks, country_ranks = cfg["sizing"]["pooled_ranks"], cfg["sizing"]["country_ranks"]
    L = [f"European company market caps from comp.g_secd, built {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"config: {cfg}", "", f"All caps in {ccy} bn unless stated. Countries = exchange country of the primary issue.", ""]

    # --- extract coverage
    common = secd[secd["tpci"] == "0"]
    yr = common["datadate"].dt.year
    cov = common.groupby(yr).agg(rows=("gvkey", "size"), gvkeys=("gvkey", "nunique"),
                                 cshoc_share=("cshoc", lambda s: s.notna().mean()))
    cov["eligible_companies"] = mcap.groupby(mcap["datadate"].dt.year)["gvkey"].nunique()
    cov["avg_per_month"] = (mcap.groupby("datadate").size().groupby(lambda d: d.year).mean()).round(0)
    L += ["Extract, common shares (tpci = 0): rows and companies with month-end prices, share of",
          "rows with a share count (cshoc); eligible companies with a cap per year (distinct, avg per month):",
          cov.round(3).to_string(), ""]
    L += ["Quotation currency of common-share rows: " + str(common["curcdd"].value_counts().head(6).to_dict()),
          "Price status (prcstd) of common-share rows, with median volume and share of zero/NaN volume:",
          common.groupby("prcstd")["cshtrd"].agg(n="size", median_vol="median",
                                                 zero_share=lambda s: (s.fillna(0) == 0).mean()).round(3).to_string(), ""]

    # --- pooled cutoffs
    pooled = bn(yearly_median_cutoffs(mcap, pooled_ranks, []))
    L += [f"POOLED cutoffs: cap of the k-th largest company, median over the 12 month ends of each year ({ccy} bn):",
          pooled.to_string(), ""]

    # --- per-country counts and cutoffs at snapshots
    for d in snaps:
        m = mcap[mcap["datadate"] == d]
        if m.empty:
            L += [f"snapshot {d:%Y-%m}: no data", ""]
            continue
        tab = bn(cg.cutoff_table(m, country_ranks, ["country"]))
        tab.loc["POOLED"] = bn(cg.cutoff_table(m, country_ranks, [])).iloc[0]
        L += [f"Snapshot {d:%Y-%m-%d}: companies with a cap and cutoffs per country ({ccy} bn):",
              tab.to_string(), ""]
        r = cg.rank_within(m, [])
        comp = {n: r[r["cap_rank"] <= n]["country"].value_counts() for n in (200, 300, 500)}
        L += [f"Country composition of the pooled top-N at {d:%Y-%m-%d}:",
              pd.DataFrame(comp).fillna(0).astype(int).sort_values(500, ascending=False).T.to_string(), ""]

    # --- per-country yearly medians for the headline ranks (compact)
    cy = bn(yearly_median_cutoffs(mcap, country_ranks, ["country"]))
    for k in (20, 30, 40, 50):
        piv = cy[f"rank_{k}"].unstack("country")
        L += [f"Cap of the {k}-th largest company per country, yearly median ({ccy} bn):", piv.to_string(), ""]
    L += ["Companies with a cap per country, yearly median of the monthly count:",
          cy["n_companies"].unstack("country").round(0).astype("Int64").to_string(), ""]

    # --- audits
    cols = ["gvkey", "conm", "country", "fic", "prirow_country", "priusa", "datadate", "mktcap", "turnover"]
    r_all = cg.rank_within(mcap_all, ["datadate"])
    would = r_all[(r_all["cap_rank"] <= 500) & ~r_all["eligible"]]
    for label, sub in [("header link but NO active listing (dormant secondary line, or float too small to trade)",
                        would[would["header_link"]]),
                       ("no header link (foreign company on a local exchange)", would[~would["header_link"]])]:
        if len(sub):
            g = (sub.groupby(["gvkey", "conm", "country", "fic"])
                 .agg(months=("datadate", "size"), peak_cap=("mktcap", "max"), med_turnover=("turnover", "median"))
                 .sort_values("peak_cap", ascending=False).head(25))
            g["peak_cap"] = (g["peak_cap"] / 1e3).round(1)
            L += [f"Would rank in the pooled top-500 by cap but NOT eligible, {label}: {len(sub):,} company-months, "
                  f"{sub['gvkey'].nunique():,} companies. Largest 25 by peak cap ({ccy} bn):", g.round(7).to_string(), ""]
    indet = mcap[mcap["turnover"].isna()]
    L += [f"Eligible company-months kept with NO volume information in the window (screen indeterminate): "
          f"{len(indet):,} of {len(mcap):,}; by country: {indet['country'].value_counts().to_dict()}", ""]

    r = cg.rank_within(mcap, ["datadate"])
    top500 = r[r["cap_rank"] <= 500]
    mism = top500[top500["fic"] != top500["country"]]
    if len(mism):
        mm = (mism.groupby(["gvkey", "conm", "country", "fic"]).agg(months=("datadate", "size"), peak_cap=("mktcap", "max"))
              .sort_values("peak_cap", ascending=False).head(25))
        mm["peak_cap"] = (mm["peak_cap"] / 1e3).round(1)
        L += [f"Top-500 rows where incorporation (fic) differs from the home exchange country: "
              f"{len(mism):,} of {len(top500):,} company-months. Largest 25:", mm.to_string(), ""]

    multi = top500[top500["n_classes"] > 1]
    if len(multi):
        mc = multi.assign(second_share=1 - multi["mktcap_main_class"] / multi["mktcap"])
        mc = (mc.groupby(["gvkey", "conm", "country"]).agg(months=("datadate", "size"), peak_cap=("mktcap", "max"),
                                                              avg_other_class_share=("second_share", "mean"))
              .sort_values("peak_cap", ascending=False).head(20))
        mc["peak_cap"] = (mc["peak_cap"] / 1e3).round(1)
        L += [f"Top-500 company-months with more than one common share class after collapsing cross-listings: "
              f"{len(multi):,}. Largest 20:", mc.round(3).to_string(), ""]

    pref = secd[(secd["tpci"] == "1") & (secd["cshoc"] > 0) & (secd["prccd"] > 0)].copy()
    pref["datadate"] = pref["datadate"] + pd.offsets.MonthEnd(0)
    if len(pref):
        pref = pref.merge(mcap[["gvkey", "datadate", "mktcap", "country"]], on=["gvkey", "datadate"], how="left")
        pref = pref[pref["country"].isna() | (pref["country"] == pref["excntry"])].copy()
        fxp = cg.convert_to_currency(pref, fx, ccy)
        pref["pref_cap"] = pref["prccd"] / pref["qunit"].fillna(1) * fxp * pref["cshoc"] / 1e6
        pref = pref.sort_values("cshtrd", ascending=False).drop_duplicates(["gvkey", "datadate", "cshoc"])
        pp = pref.groupby(["gvkey", "datadate"]).agg(pref_cap=("pref_cap", "sum"), conm=("conm", "first"),
                                                     common_cap=("mktcap", "first")).reset_index()
        pp = pp.groupby(["gvkey", "conm"]).agg(months=("datadate", "size"), peak_pref_cap=("pref_cap", "max"),
                                               peak_common_cap=("common_cap", "max")).sort_values("peak_pref_cap", ascending=False)
        pp[["peak_pref_cap", "peak_common_cap"]] = (pp[["peak_pref_cap", "peak_common_cap"]] / 1e3).round(1)
        L += ["Preferred shares (tpci = 1) are NOT in the cap. Largest 20 by peak preferred cap, with the company's",
              f"common cap that month (NaN = no common listing, e.g. only prefs listed) ({ccy} bn):", pp.head(20).to_string(), ""]

    nocap = common[common["cshoc"].isna() | (common["cshoc"] <= 0)]
    late = mcap[(mcap["datadate"] - mcap["price_date"]).dt.days > 7]
    L += [f"Eligible company-months whose month-end row is more than 7 days before the calendar month end "
          f"(delisting month or stale line): {len(late):,} of {len(mcap):,}", ""]
    L += [f"Common-share rows with a price but no share count: {len(nocap):,} of {len(common):,}; "
          f"companies never getting a cap: {len(set(nocap['gvkey']) - set(mcap_all['gvkey'])):,}",
          "gvkey is a zero-padded string: read the CSVs with dtype={'gvkey': str}."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "universe_eu_compustat.yaml")
    ap.add_argument("--offline", action="store_true", help="reuse the extracts on disk")
    ap.add_argument("--refresh-headers", action="store_true",
                    help="with --offline: refetch g_security / g_company, keep the g_secd extract")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / cfg["output"]["dir"]
    out.mkdir(parents=True, exist_ok=True)

    start, end = str(cfg["sample"]["start"]), str(cfg["sample"]["end"])
    y0, y1 = int(start[:4]), int(end[:4])
    countries = list(cfg["countries"])
    ccy = cfg["filters"]["currency"]

    if args.offline:
        secd = pd.read_parquet(out / "secd_monthend_extract.parquet")
        fx = pd.read_parquet(out / "fx_daily.parquet")
        if args.refresh_headers:
            db = cu.connect(ROOT)
            security, company = cg.fetch_security_header(db, countries), cg.fetch_company_header(db)
            db.close()
            security.to_parquet(out / "security_header.parquet", index=False)
            company.to_parquet(out / "company_header.parquet", index=False)
        else:
            security = pd.read_parquet(out / "security_header.parquet")
            company = pd.read_parquet(out / "company_header.parquet")
    else:
        db = cu.connect(ROOT)
        security = cg.fetch_security_header(db, countries)
        company = cg.fetch_company_header(db)
        security.to_parquet(out / "security_header.parquet", index=False)
        company.to_parquet(out / "company_header.parquet", index=False)
        secd = cg.fetch_secd_monthend(db, y0, y1, countries)
        secd.to_parquet(out / "secd_monthend_extract.parquet", index=False)
        currencies = sorted(set(secd["curcdd"].dropna()) | {ccy})
        fx = cg.fetch_fx(db, currencies, y0, y1)
        fx.to_parquet(out / "fx_daily.parquet", index=False)
        db.close()
    print(f"extract: {len(secd):,} rows, {secd['gvkey'].nunique():,} gvkeys", flush=True)

    mcap_all = cg.company_month_mktcap(secd, company, security, fx, countries, ccy,
                                       tuple(cfg["filters"]["issue_types"]),
                                       min_turnover=float(cfg["filters"]["min_turnover"]),
                                       turnover_window=int(cfg["filters"]["turnover_window"]))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    mcap_all = mcap_all[(mcap_all["datadate"] >= lo) & (mcap_all["datadate"] <= hi)].reset_index(drop=True)
    mcap = mcap_all[mcap_all["eligible"]].reset_index(drop=True)
    mcap_all.to_parquet(out / "company_month_mktcap.parquet", index=False)
    mcap.to_csv(out / "company_month_mktcap.csv", index=False)
    print(f"company-months: {len(mcap_all):,} with a cap, {len(mcap):,} eligible, "
          f"{mcap['gvkey'].nunique():,} eligible companies", flush=True)

    ranks_p, ranks_c = cfg["sizing"]["pooled_ranks"], cfg["sizing"]["country_ranks"]
    bn(yearly_median_cutoffs(mcap, ranks_p, [])).to_csv(out / "cutoffs_pooled_yearly.csv")
    bn(yearly_median_cutoffs(mcap, ranks_c, ["country"])).to_csv(out / "cutoffs_country_yearly.csv")
    snaps = [pd.Timestamp(str(s)) + pd.offsets.MonthEnd(0) for s in cfg["sizing"]["snapshot_months"]]
    snaps = [mcap.loc[(mcap["datadate"].dt.year == s.year) & (mcap["datadate"].dt.month == s.month), "datadate"].max()
             for s in snaps]
    snaps = [s for s in snaps if pd.notna(s)]
    snap_tab = pd.concat({d: bn(cg.cutoff_table(mcap[mcap["datadate"] == d], ranks_c, ["country"])) for d in snaps},
                         names=["datadate"])
    snap_tab.to_csv(out / "cutoffs_snapshots.csv")
    comp_rows = []
    for d in snaps:
        r = cg.rank_within(mcap[mcap["datadate"] == d], [])
        for n in ranks_p:
            vc = r[r["cap_rank"] <= n]["country"].value_counts()
            comp_rows.append(pd.Series(vc, name=(d, n)))
    pd.DataFrame(comp_rows).fillna(0).astype(int).rename_axis(["datadate", "top_n"]).to_csv(out / "composition_pooled.csv")

    report = build_report(cfg, secd, mcap_all, mcap, snaps, fx)
    (out / "mktcap_build_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
