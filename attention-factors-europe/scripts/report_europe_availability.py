"""Availability report of the European builds up to a year (default 1997): the universe,
the market and accounting inputs, the 39 characteristics and JKP's, per year and country.

    python scripts/report_europe_availability.py --configs configs/europe17_data.yaml configs/europe12_data.yaml
                                                 [--last-year 1997] [--out data/europe/availability_report.md]

Every number is computed here from the build outputs (build_europe_universe.py,
build_europe_dataset.py); nothing is typed in. Writes the markdown report and, next to it,
the full tables as CSV (availability_tables/).
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

from afe.data import build_us as bu  # noqa: E402
from afe.data import characteristics as ch  # noqa: E402
from afe.data import compustat_global as cg  # noqa: E402
from afe.data import europe_fx as efx  # noqa: E402

# Nearest JKP column to each of the 39, and how close the definition is:
# same = same formula up to details, close = same idea, other window/scaling, proxy = related, none = no counterpart
JKP_MAP = {
    "r2_1": ("ret_1_0", "same"), "ST_Rev": ("ret_1_0", "same"), "r12_2": ("ret_12_1", "same"),
    "r12_7": ("ret_12_7", "same"), "r36_13": ("ret_36_12", "same"), "Ret_D1": (None, "none"),
    "Ret_W1": (None, "none"), "STD_W1": ("rvol_21d", "proxy"),
    "A2ME": ("at_me", "same"), "BEME": ("be_me", "same"), "C": ("cash_at", "same"), "CF": ("fcf_be", "close"),
    "CF2P": ("ocf_me", "close"), "Q": (None, "none"), "Lev": ("debt_bev", "proxy"), "E2P": ("ni_me", "close"),
    "Investment": ("at_gr1", "same"), "NOA": ("noa_at", "close"), "DPI2A": ("ppeinv_gr1a", "same"),
    "AT": ("assets", "same"), "LME": ("market_equity", "same"), "LTurnover": ("turnover_126d", "close"),
    "Rel2High": ("prc_highprc_252d", "same"), "Resid_Var": ("ivol_ff3_21d", "close"), "Spread": ("bidaskhl_21d", "same"),
    "SUV": (None, "none"), "Variance": ("rvol_21d", "close"), "Vol": ("dolvol_126d", "proxy"), "Beta": ("beta_252d", "close"),
    "PROF": ("gp_at", "proxy"), "CTO": ("at_turnover", "close"), "FC2Y": ("opex_at", "proxy"), "OP": ("ope_be", "same"),
    "PM": ("ebit_sale", "close"), "RNA": (None, "none"), "D2A": (None, "none"), "OA": ("oaccruals_at", "close"),
    "OL": ("opex_at", "close"), "PCM": ("gp_sale", "same"),
}


def md(df: pd.DataFrame, fmt: str = "{:.2f}", index: bool = True) -> str:
    """Markdown table; floats formatted with `fmt`, NaN as a dash."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join("-" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
             for r in d.itertuples(index=False)]
    return "\n".join(rows)


def load(cfg):
    ins = ROOT / cfg["output"]["inspect_dir"]
    out = {"mk": pd.read_parquet(ins / "company_month_mktcap.parquet"),
           "rank": pd.read_parquet(ins / "ranking_asof.parquet"),
           "cov": pd.read_parquet(ins / "coverage_member_months.parquet"),
           "inp": pd.read_parquet(ins / "input_coverage_member_months.parquet")}
    jc = ins / "jkp_coverage.parquet"
    out["jkp"] = pd.read_parquet(jc) if jc.exists() else None
    for k in ("mk", "rank"):
        out[k]["datadate"] = pd.to_datetime(out[k]["datadate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--last-year", type=int, default=1997)
    ap.add_argument("--out", default="data/europe/availability_report.md")
    args = ap.parse_args()
    cfgs = [yaml.safe_load(Path(p).read_text()) for p in args.configs]
    last = args.last_year
    out_path = ROOT / args.out
    tab_dir = out_path.parent / "availability_tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    data = {c["market"]: load(c) for c in cfgs}
    names = bu.characteristic_names(cfgs[0])
    theme = {n: ch.REGISTRY[n].theme for n in names}
    freq = {n: ch.REGISTRY[n].frequency for n in names}
    compare_years = [2000, 2005, 2010, 2015, 2020, 2025]

    L = [f"# European data to {last}: universe, inputs and characteristics", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by scripts/report_europe_availability.py from the build outputs of "
         + ", ".join(f"`{c['market']}`" for c in cfgs) + ". Numeraire: ECU to 1998-12-31, EUR from 1999-01-01. "
         "Universe month M = top 200 by cap at the end of M-1. \"Year\" of a universe table is the universe month's "
         "year; of a ranking table, the ranking month's.", ""]

    # ------------------------------------------------------------- universe
    L += ["## 1. Universe", ""]
    for c in cfgs:
        d = data[c["market"]]
        mk, rank = d["mk"], d["rank"]
        el = mk[mk["eligible"]]
        yr = el["datadate"].dt.year
        per = el.groupby([yr, "country"]).size().unstack("country").fillna(0).div(el.groupby(yr)["datadate"].nunique(), axis=0)
        per["total"] = per.sum(axis=1)
        per = per.loc[:last].round(0).astype(int)
        per.to_csv(tab_dir / f"{c['market']}_eligible_by_country.csv")
        cut = cg.cutoff_table(el, [50, 100, 150, 200, 300], ["datadate"]).reset_index()
        cut["year"] = cut["datadate"].dt.year
        cy = cut.drop(columns="datadate").groupby("year").median()
        cy[[x for x in cy.columns if x.startswith("rank_")]] /= 1e3
        cy.to_csv(tab_dir / f"{c['market']}_pooled_cutoffs.csv")
        r = rank.assign(year=rank["datadate"].dt.year)
        comp = r.groupby(["year", "country"]).size().unstack("country").fillna(0).div(r.groupby("year")["datadate"].nunique(), axis=0)
        comp.to_csv(tab_dir / f"{c['market']}_top200_composition.csv")
        blind = r.groupby("year").agg(turnover_unknown=("turnover", lambda s: s.isna().mean()),
                                      fic_not_home=("fic", lambda s: np.nan))
        blind["fic_not_home"] = r.assign(x=r["fic"] != r["country"]).groupby("year")["x"].mean()
        # the screen's blind years: members ranked while no volume existed, judged by their first
        # 12 month ends WITH a turnover measure (whenever that came)
        thr = float(c["filters"]["min_turnover"])
        early = r[r["turnover"].isna()]
        seen = mk[mk["gvkey"].isin(early["gvkey"].unique()) & mk["turnover"].notna()].sort_values("datadate")
        first = seen.groupby("gvkey").head(12).groupby("gvkey").agg(first_measured=("datadate", "min"),
                                                                    turnover_first_year=("turnover", "median"))
        e = early.groupby("gvkey").agg(conm=("conm", "first"), country=("country", "first"),
                                       blind_months=("datadate", "size"), peak_cap=("mktcap", "max")).join(first)
        e["verdict"] = np.where(e["turnover_first_year"].isna(), "never measured",
                                np.where(e["turnover_first_year"] >= thr, "passes", "would be screened"))
        e.to_csv(tab_dir / f"{c['market']}_blind_years_members.csv")
        verdict = e.groupby("verdict").agg(companies=("conm", "size"), member_months=("blind_months", "sum"))
        worst = e[e["verdict"] != "passes"].sort_values("blind_months", ascending=False).head(15).copy()
        worst["peak_cap"] = (worst["peak_cap"] / 1e3).round(1)
        worst["first_measured"] = pd.to_datetime(worst["first_measured"]).dt.strftime("%Y-%m")
        L += [f"### `{c['market']}`: {len(c['countries'])} exchange countries ({', '.join(c['countries'])})", "",
              f"Eligible companies with a cap, average per month end, by home country, ranking years to {last}:", "",
              md(per, index=True), "",
              "Pooled cap at rank k, median over the year's month ends (numeraire bn), and eligible companies per month:", "",
              md(cy.loc[[y for y in cy.index if y <= last or y in compare_years]], "{:.2f}"), "",
              f"Top-200 composition, average members per month by home country (ranking years to {last}, then "
              "selected later years):", "",
              md(comp.loc[[y for y in comp.index if y <= last or y in compare_years]], "{:.0f}"), "",
              "Share of top-200 member-months whose turnover screen had no volume to look at (kept as active), and "
              "share incorporated outside their home exchange country:", "",
              md(blind.loc[[y for y in blind.index if y <= last or y in compare_years]], "{:.2f}"), "",
              "Companies ranked in the top 200 while their turnover could not be measured, judged by the median "
              f"turnover of their first 12 month ends that have volume (threshold {thr:g}):", "",
              md(verdict), "", "Largest of those that fail or are never measured (by blind member-months):", "",
              md(worst[["conm", "country", "blind_months", "peak_cap", "first_measured", "turnover_first_year", "verdict"]],
                 "{:.2e}"), ""]

    # ------------------------------------------------------------- inputs
    L += ["## 2. Market and accounting inputs of the universe members", "",
          "Per universe member-month: share of the month's trading days with a return, a traded close, volume, "
          "a daily high and low, and a Corwin-Schultz spread estimate; share of member-months with a fiscal record "
          "usable at the end of M-1 (June rule, at most 30 months old) and the median age of that record in months.", ""]
    for c in cfgs:
        inp = data[c["market"]]["inp"].copy()
        inp["year"] = inp["month"].dt.year
        t = inp.groupby("year").agg(returns=("has_ret", "mean"), traded=("traded", "mean"), volume=("has_vol", "mean"),
                                    high_low=("has_hl", "mean"), spread_est=("has_cs", "mean"),
                                    fiscal_record=("has_funda", "mean"), record_age_m=("fund_age_m", "median"))
        t.to_csv(tab_dir / f"{c['market']}_inputs_by_year.csv")
        e = inp[inp["year"] <= last]
        bc = e.groupby("country").agg(member_months=("gvkey", "size"), volume=("has_vol", "mean"),
                                      high_low=("has_hl", "mean"), fiscal_record=("has_funda", "mean"))
        bc.to_csv(tab_dir / f"{c['market']}_inputs_by_country_to_{last}.csv")
        L += [f"### `{c['market']}`", "", md(t.loc[[y for y in t.index if y <= last or y in compare_years]]), "",
              f"By home country, universe months 1990-01 .. {last}-12:", "", md(bc), ""]

    # ------------------------------------------------------------- the 39
    L += ["## 3. The 39 characteristics (US definitions, docs/us_characteristics.md)", "",
          "Share of universe member-months with a value as the features table sees it: after the last-observed "
          "carry, before the cross-sectional median fill (annual and monthly ones at the end of M-1; daily ones as the "
          "share of the month's trading days). Spread is the Corwin-Schultz high-low estimator (Global has no quotes).", ""]
    for c in cfgs:
        cov = data[c["market"]]["cov"].copy()
        cov["year"] = cov["month"].dt.year
        by_year = cov.groupby("year")[names].mean().T
        by_year.insert(0, "freq", [freq[n] for n in names])
        by_year.insert(0, "theme", [theme[n] for n in names])
        by_year.to_csv(tab_dir / f"{c['market']}_characteristics_by_year.csv")
        cols = ["theme", "freq"] + [y for y in by_year.columns[2:] if y <= last] + \
               [y for y in by_year.columns[2:] if y in compare_years]
        e = cov[cov["year"] <= last]
        bc = e.groupby("country")[names].mean().T
        bc.to_csv(tab_dir / f"{c['market']}_characteristics_by_country_to_{last}.csv")
        L += [f"### `{c['market']}`: by universe year", "", md(by_year[cols].rename_axis("characteristic")), "",
              f"### `{c['market']}`: by home country, universe months to {last}", "", md(bc.rename_axis("characteristic")), ""]

    # ------------------------------------------------------------- JKP
    L += ["## 4. JKP Global Factor characteristics (`contrib.global_factor`)", "",
          "JKP rows of universe members as of the end of M-1 (the member's home line, else JKP's primary security). "
          "Coverage = share of member-months with a non-missing value, by the year of M-1 (JKP's `eom`).", ""]
    for c in cfgs:
        jk = data[c["market"]]["jkp"]
        if jk is None:
            L += [f"`{c['market']}`: no JKP table (pull not done).", ""]
            continue
        cols = [x for x in jk.columns if x not in ("year", "country", "n_member_months")]
        wsum = jk[cols].mul(jk["n_member_months"], axis=0).groupby(jk["year"]).sum()
        by_year = wsum.div(jk.groupby("year")["n_member_months"].sum(), axis=0)    # member-month weighted
        by_year.T.to_csv(tab_dir / f"{c['market']}_jkp_coverage_by_year.csv")
        chars = [x for x in cols if x != "has_row"]
        summ = pd.DataFrame({"has_row": by_year["has_row"],
                             "n_cols_ge_80pct": (by_year[chars] >= 0.8).sum(axis=1),
                             "n_cols_ge_50pct": (by_year[chars] >= 0.5).sum(axis=1),
                             "median_col_coverage": by_year[chars].median(axis=1)})
        summ.insert(0, "n_columns", len(chars))
        L += [f"### `{c['market']}`", "", md(summ.loc[[y for y in summ.index if y <= last or y in compare_years]]), ""]
        rows = []
        for n in names:
            col, q = JKP_MAP.get(n, (None, "none"))
            row = {"characteristic": n, "jkp_column": col or "-", "match": q}
            for y in [y for y in by_year.index if y <= last] + [y for y in compare_years if y in by_year.index]:
                row[str(y)] = float(by_year.loc[y, col]) if col in by_year.columns else np.nan
            rows.append(row)
        mt = pd.DataFrame(rows).set_index("characteristic")
        mt.to_csv(tab_dir / f"{c['market']}_jkp_counterparts.csv")
        L += [f"Nearest JKP column to each of the 39 and its coverage (`{c['market']}`):", "", md(mt), ""]
        low = by_year.loc[[y for y in by_year.index if y <= last], chars].mean().sort_values()
        L += [f"JKP columns with the lowest mean coverage over the years to {last} (20):", "",
              md(low.head(20).to_frame("coverage").rename_axis("jkp_column")), ""]

    # ------------------------------------------------------------- numeraire
    fx = pd.read_parquet(ROOT / cfgs[0]["extract"]["dir"] / "fx_daily.parquet")
    fx["datadate"] = pd.to_datetime(fx["datadate"])
    off = cfgs[0].get("currency", {}).get("official_ecu_check")
    official = pd.read_csv(ROOT / off, parse_dates=["date"]).set_index("date") if off and (ROOT / off).exists() else None
    chk = efx.checks(fx, official)
    L += ["## 5. Numeraire checks (ECU basket from Compustat's cross rates)", ""]
    if "vs_official" in chk:
        L += ["Basket over the official daily ECU rate of Eurostat (ert_bil_eur_d), minus 1: yearly mean and median "
              "absolute daily difference:", "", md(chk["vs_official"], "{:+.5f}"), ""]
    if "seam_1998" in chk:
        L += ["Basket on 1998-12-31 against the irrevocable conversion rates (units per ECU/EUR):", "",
              md(chk["seam_1998"].set_index("currency"), "{:.6f}"), ""]
    L += ["Old over new basket on the revision days:", "", md(chk["revisions"].set_index("revision_day"), "{:.6f}"), ""]
    for k, label in [("vs_compustat_ecu", "Compustat's own ECU quote over the basket, minus 1"),
                     ("vs_compustat_eur", "Compustat's synthetic EUR over the basket ECU, minus 1 (not used)")]:
        if k in chk:
            L += [f"{label}:", "", md(chk[k], "{:.5f}"), ""]
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out_path} and {tab_dir}")


if __name__ == "__main__":
    main()
