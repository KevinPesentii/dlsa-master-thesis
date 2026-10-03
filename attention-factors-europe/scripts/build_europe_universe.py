"""Step 1b of the European builds: cap table, as-of top-N ranking, and the lines to pull.

    python scripts/build_europe_universe.py --config configs/europe17_data.yaml

Reads the shared step-1 extract (extract.dir, from fetch_europe_extract.py) and writes to
output.inspect_dir (gitignored):
  company_month_mktcap.parquet  one row per (gvkey, month end): every company with a cap in
                                the version's exchange countries, in numeraire millions (ECU
                                to 1998-12, EUR from 1999-01), with the `eligible` flag of
                                compustat_global (header link AND turnover screen)
  ranking_asof.parquet          the top-N eligible companies at every month end from
                                universe.ranking_start; row m ranks on the cap at the END of
                                m, so it is the universe of month m+1 (the lag is applied in
                                stage 2, as for the US and euro-11 builds)
  listings.csv                  every home line (gvkey, iid) of a company ever in that
                                ranking, over all months of the cap table: the daily pull
  universe_build_report.txt     coverage, cutoffs, composition, what the screen could see
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
from afe.data import eu_panel as ep  # noqa: E402
from afe.data import europe_fx as efx  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def load_extract(ext: Path, countries: list[str]):
    secd = pd.concat([pd.read_parquet(p) for p in sorted((ext / "secd_monthend").glob("*.parquet"))], ignore_index=True)
    secd["datadate"] = pd.to_datetime(secd["datadate"]).astype("datetime64[ns]")
    secd = secd[secd["excntry"].isin(countries)].reset_index(drop=True)
    security = pd.read_parquet(ext / "security_header.parquet")
    company = pd.read_parquet(ext / "company_header.parquet")
    fx = pd.read_parquet(ext / "fx_daily.parquet")
    fx["datadate"] = pd.to_datetime(fx["datadate"]).astype("datetime64[ns]")
    return secd, security, company, fx


def official_ecu(cfg) -> pd.DataFrame | None:
    """Eurostat's official daily ECU rates (units per ECU), a check file named in the config."""
    p = cfg.get("currency", {}).get("official_ecu_check")
    if not p or not (ROOT / p).exists():
        return None
    return pd.read_csv(ROOT / p, parse_dates=["date"]).set_index("date")


def yearly_median_cutoffs(mcap: pd.DataFrame, ranks: list[int], by: list[str]) -> pd.DataFrame:
    t = cg.cutoff_table(mcap, ranks, by + ["datadate"]).reset_index()
    t["year"] = t["datadate"].dt.year
    return t.drop(columns="datadate").groupby(by + ["year"]).median()


def report(cfg, secd, mcap_all, asof, fx) -> str:
    n = int(cfg["universe"]["size"])
    mcap = mcap_all[mcap_all["eligible"]]
    L = [f"{cfg['market']}: cap table and as-of ranking, built {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"countries {cfg['countries']}; filters {cfg['filters']}; universe {cfg['universe']}", "",
         "Caps in numeraire bn: ECU to 1998-12-31, EUR from 1999-01-01 (src/afe/data/europe_fx.py). "
         "Country = exchange country of the company's most active listing.", ""]

    chk = efx.checks(fx, official_ecu(cfg))
    if "seam_1998" in chk:
        s = chk["seam_1998"]
        L += ["Numeraire check 1: ECU basket on 1998-12-31 from Compustat's cross rates vs the irrevocable euro "
              "conversion rates (which are that day's official ECU rates):", s.round(6).to_string(index=False), ""]
    L += ["Numeraire check 2: value of the old over the new basket on the revision day (1 = no jump):",
          chk["revisions"].round(6).to_string(index=False), ""]
    if "vs_official" in chk:
        L += ["Numeraire check 3: basket over the official daily ECU rate (Eurostat ert_bil_eur_d), minus 1: yearly "
              "mean and median absolute daily difference:", chk["vs_official"].round(5).to_string(), ""]
    for k, label in [("vs_compustat_ecu", "Compustat's own ECU quote (XEU, business days)"),
                     ("vs_compustat_eur", "Compustat's synthetic EUR (NOT used before 1999)")]:
        if k in chk:
            L += [f"Numeraire check: {label} per GBP over the basket ECU per GBP, minus 1, by year:",
                  chk[k].round(5).to_string(), ""]

    common = secd[secd["tpci"] == "0"].copy()
    common["year"] = common["datadate"].dt.year
    cov = common.groupby("year").agg(rows=("gvkey", "size"), gvkeys=("gvkey", "nunique"),
                                     with_shares=("cshoc", lambda s: (s.fillna(0) > 0).mean()),
                                     with_volume=("cshtrd", lambda s: s.notna().mean()))
    L += ["Extract, common shares: month-end rows, companies, share of rows with a share count and with volume:",
          cov.round(3).to_string(), ""]
    vol = common.groupby(["year", "excntry"])["cshtrd"].apply(lambda s: s.notna().mean()).unstack("excntry")
    L += ["Share of month-end rows with volume, by exchange country (the turnover screen needs it; blank "
          "before 1992 everywhere):", vol.round(2).to_string(), ""]

    yr = mcap["datadate"].dt.year
    per = mcap.groupby([yr, "country"]).size().unstack("country").div(mcap.groupby(yr)["datadate"].nunique(), axis=0)
    L += ["Eligible companies with a cap, average per month end, by home country:", per.round(0).astype("Int64").to_string(), ""]

    ranks = [50, 100, 150, 200, 250, 300, 500]
    pooled = yearly_median_cutoffs(mcap, ranks, [])
    rank_cols = [c for c in pooled.columns if c.startswith("rank_")]
    pooled[rank_cols] = pooled[rank_cols] / 1e3
    L += ["Pooled cap at rank k, median over the year's month ends (numeraire bn), eligible companies per month:",
          pooled.round(2).to_string(), ""]

    a = asof.assign(year=asof["datadate"].dt.year)
    months = a.groupby("year")["datadate"].nunique()
    comp = a.groupby(["year", "country"]).size().unstack("country").fillna(0).div(months, axis=0)
    L += [f"Composition of the as-of top-{n} (ranking month = year; its universe is the following month), "
          "average companies per month by home country:", comp.round(1).to_string(), ""]
    a["indeterminate"] = a["turnover"].isna()
    scr = a.groupby("year").agg(members=("gvkey", "size"), turnover_unknown=("indeterminate", "mean"),
                                fic_not_home=("fic", lambda s: 0.0), n_classes_gt1=("n_classes", lambda s: (s > 1).mean()))
    scr["fic_not_home"] = a.assign(x=a["fic"] != a["country"]).groupby("year")["x"].mean()
    L += [f"Top-{n} member-months per ranking year: share whose turnover screen had NO volume to look at "
          "(kept as active, the rule of compustat_global), share with incorporation != home country, share with "
          ">1 common class:", scr.round(3).to_string(), ""]

    r_all = cg.rank_within(mcap_all, ["datadate"])
    would = r_all[(r_all["cap_rank"] <= n) & ~r_all["eligible"]]
    for label, sub in [("header link but no active listing (dormant line or tiny float)", would[would["header_link"]]),
                       ("no header link (foreign company trading here)", would[~would["header_link"]])]:
        if len(sub):
            g = (sub.groupby(["gvkey", "conm", "country", "fic"])
                 .agg(first=("datadate", "min"), last=("datadate", "max"), months=("datadate", "size"),
                      peak_cap=("mktcap", "max"), med_turnover=("turnover", "median"))
                 .sort_values("peak_cap", ascending=False).head(25))
            g["peak_cap"] = (g["peak_cap"] / 1e3).round(1)
            g["first"], g["last"] = g["first"].dt.strftime("%Y-%m"), g["last"].dt.strftime("%Y-%m")
            L += [f"Would rank in the pooled top-{n} by cap but NOT eligible, {label}: {len(sub):,} company-months, "
                  f"{sub['gvkey'].nunique():,} companies. Largest 25:", g.to_string(), ""]

    mism = a[a["fic"] != a["country"]]
    if len(mism):
        mm = (mism.groupby(["gvkey", "conm", "country", "fic"]).agg(first=("datadate", "min"), last=("datadate", "max"),
              months=("datadate", "size"), peak_cap=("mktcap", "max")).sort_values("peak_cap", ascending=False).head(25))
        mm["peak_cap"] = (mm["peak_cap"] / 1e3).round(1)
        mm["first"], mm["last"] = mm["first"].dt.strftime("%Y-%m"), mm["last"].dt.strftime("%Y-%m")
        L += [f"Top-{n} companies incorporated outside their home exchange country, largest 25:", mm.to_string(), ""]

    ccy = a.groupby("year")["curcdd"].value_counts(normalize=True).unstack().fillna(0)
    L += [f"Quotation currency of the top-{n} lines (share of member-months):", ccy.round(3).to_string(), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    countries, f, u = list(cfg["countries"]), cfg["filters"], cfg["universe"]
    n = int(u["size"])
    out = ROOT / cfg["output"]["inspect_dir"]
    out.mkdir(parents=True, exist_ok=True)

    secd, security, company, fx = load_extract(ROOT / cfg["extract"]["dir"], countries)
    fxn = efx.with_numeraire(fx, int(f["fx_max_stale_days"]))
    print(f"{cfg['market']}: extract {len(secd):,} rows, {secd['gvkey'].nunique():,} gvkeys on {len(countries)} "
          f"exchange countries", flush=True)
    mcap_all = cg.company_month_mktcap(secd, company, security, fxn, countries, efx.NUMERAIRE,
                                       tuple(f["issue_types"]), min_turnover=float(f["min_turnover"]),
                                       turnover_window=int(f["turnover_window"]),
                                       min_class_share=float(f["min_class_share"]))
    mcap_all.to_parquet(out / "company_month_mktcap.parquet", index=False)
    print(f"cap table: {len(mcap_all):,} company-months, {int(mcap_all['eligible'].sum()):,} eligible", flush=True)

    lo = pd.Period(str(u["ranking_start"]), "M").to_timestamp(how="end").normalize()
    asof = ep.asof_ranking(mcap_all[mcap_all["datadate"] >= lo], n)
    short = asof.groupby("datadate").size()
    short = short[short < n]
    if len(short):
        print(f"WARNING: {len(short)} ranking months with fewer than {n} eligible companies: "
              f"{short.head(5).to_dict()}", flush=True)
    asof.to_parquet(out / "ranking_asof.parquet", index=False)
    last_rank = pd.Period(str(u["end"]), "M") - 1          # the ranking that sets the last universe month
    members = sorted(asof.loc[asof["datadate"].dt.to_period("M") <= last_rank, "gvkey"].unique())
    lines = (mcap_all[mcap_all["gvkey"].isin(members)][["gvkey", "iid"]].drop_duplicates()
             .sort_values(["gvkey", "iid"]).reset_index(drop=True))
    lines.to_csv(out / "listings.csv", index=False)
    print(f"ranking {asof['datadate'].min():%Y-%m} .. {asof['datadate'].max():%Y-%m}: {len(members):,} ever-members, "
          f"{len(lines):,} home lines", flush=True)

    rep = report(cfg, secd, mcap_all, asof, fx)
    (out / "universe_build_report.txt").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
