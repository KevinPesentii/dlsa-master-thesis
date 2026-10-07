"""Cap sanity checks to run after a build: does a company's cap use the same unit for its
price and its share count, and does it move with its price?

    python scripts/check_caps.py --config configs/europe17_data.yaml
    python scripts/check_caps.py --config configs/us_data.yaml

A cap is price x shares, so it cannot check itself. These checks compare it with numbers
that do not come from that product:

  dividends (Europe)  dividend yield of the price feed (g_secd `div` per traded unit over
                      the price) against the accounts' (funda dvc over our cap). A price
                      per depositary receipt of ten shares times a count of shares makes
                      the second yield ten times too small (Eurocommercial 1996-2005).
                      Scrip and special dividends add noise, so a firm-year is flagged
                      only when the yields differ by `--div-factor` in both windows (the
                      fiscal year and the year after) AND book-to-market against the
                      SIC2-year median of members points the same way.
  jumps (both)        month-end cap change against the pricing line's split-adjusted
                      price change: anything else is a change in the share count, so a
                      2x move of the cap at a flat price (a duplicate class, a unit
                      switch, a missing class) is flagged; and switches of the pricing
                      line that move the cap 2x.
  receipts (US)       a receipt's company cap against its issuer's home-market cap in
                      Compustat Global (us_panel.home_caps), after receipt_caps' guard.

Writes cap_checks.txt and one CSV per check to output.inspect_dir (Europe) or
data/us/private (US). Nothing is changed: a flag is a lead to look at, not a verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import us_panel as up  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 30)


def numeraire_factor(dates: pd.Series, ccy: pd.Series, fx: pd.DataFrame) -> np.ndarray:
    key = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy().astype("datetime64[ns]"),
                        "currency": ccy.fillna("NA").astype(str).to_numpy(), "_i": np.arange(len(dates))})
    m = pd.merge_asof(key.sort_values("date"), fx.sort_values("date"), on="date", by="currency",
                      tolerance=pd.Timedelta(days=10), direction="backward")
    return m.sort_values("_i")["eur_per_unit"].to_numpy()


def members(ins: Path, n: int) -> pd.DataFrame:
    r = pd.read_parquet(ins / "ranking_asof.parquet", columns=["gvkey", "datadate", "cap_rank"])
    return r[r["cap_rank"] <= n]


def europe_jumps(cfg: dict, ins: Path, mem: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mk = pd.read_parquet(ins / "company_month_mktcap.parquet",
                         columns=["gvkey", "datadate", "mktcap", "iid", "prccd", "fx", "n_classes", "country", "conm", "price_date"])
    ext = ROOT / cfg["extract"]["dir"] / "secd_monthend"
    adj = pd.concat([pd.read_parquet(p, columns=["gvkey", "iid", "datadate", "ajexdi"]) for p in sorted(ext.glob("*.parquet"))])
    adj = adj.rename(columns={"datadate": "price_date"}).drop_duplicates(["gvkey", "iid", "price_date"])
    mk = mk.merge(adj, on=["gvkey", "iid", "price_date"], how="left").merge(mem, on=["gvkey", "datadate"], how="left")
    mk["adj_price"] = mk["prccd"].astype(float) * mk["fx"].astype(float) / mk["ajexdi"].astype(float)
    mk = mk.sort_values(["gvkey", "datadate"])
    g = mk.groupby("gvkey")
    for c in ["datadate", "mktcap", "iid", "adj_price", "cap_rank", "n_classes", "country"]:
        mk["prev_" + c] = g[c].shift(1)
    mk = mk[(mk["prev_datadate"] == mk["datadate"] - pd.offsets.MonthEnd(1))
            & (mk["cap_rank"].notna() | mk["prev_cap_rank"].notna())].copy()
    mk["cap_change"] = np.log10(mk["mktcap"].astype(float) / mk["prev_mktcap"].astype(float))
    same = mk["iid"] == mk["prev_iid"]
    mk["unexplained"] = np.where(same, mk["cap_change"] - np.log10(mk["adj_price"] / mk["prev_adj_price"]), np.nan)
    cols = ["gvkey", "conm", "datadate", "prev_iid", "iid", "prev_mktcap", "mktcap", "prev_n_classes", "n_classes",
            "prev_country", "country", "prev_cap_rank", "cap_rank"]
    jumps = mk[same & (mk["unexplained"].abs() >= 0.3)][cols + ["unexplained"]]
    switches = mk[~same & (mk["cap_change"].abs() >= 0.3)][cols + ["cap_change"]]
    return jumps, switches


def europe_dividends(cfg: dict, ins: Path, mem: pd.DataFrame, factor: float) -> pd.DataFrame:
    raw = ROOT / cfg["raw"]["dir"]
    out_dir = ROOT / cfg["output"]["dir"]
    fx = pd.read_parquet(out_dir / "fx_to_numeraire_daily.parquet", columns=["date", "currency", "eur_per_unit"])
    fx["date"] = fx["date"].astype("datetime64[ns]")
    mk = pd.read_parquet(ins / "company_month_mktcap.parquet",
                         columns=["gvkey", "datadate", "mktcap", "iid", "prccd", "fx", "sic", "conm"])
    gv = set(mem["gvkey"])
    f = pd.read_parquet(raw / "g_funda.parquet", columns=["gvkey", "datadate", "curcd", "indfmt", "dvc", "dvt", "dv", "ceq", "seq"])
    f = f[f["gvkey"].isin(gv)].copy()
    f["datadate"] = pd.to_datetime(f["datadate"]).astype("datetime64[ns]")
    f = f.sort_values(["gvkey", "datadate", "indfmt"]).drop_duplicates(["gvkey", "datadate"])  # INDL before FS
    for c in ["dvc", "dvt", "dv", "ceq", "seq"]:
        f[c] = f[c].astype("float64")
    f["fx_acc"] = numeraire_factor(f["datadate"], f["curcd"], fx)
    f["month_end"] = f["datadate"] + pd.offsets.MonthEnd(0)                 # the cap table's key
    f = f.merge(mk.rename(columns={"datadate": "month_end"}), on=["gvkey", "month_end"], how="inner")
    f["price"] = f["prccd"].astype(float) * f["fx"].astype(float)
    f["div_acc"] = f["dvc"].fillna(f["dvt"]).fillna(f["dv"]) * f["fx_acc"]
    lines = f[["gvkey", "iid"]].drop_duplicates()
    d = pd.concat([pd.read_parquet(p, columns=["gvkey", "iid", "datadate", "div", "curcddv"]).merge(lines, on=["gvkey", "iid"])
                   for p in sorted((raw / "secd_daily").glob("*.parquet"))])
    d = d[d["div"].astype(float) > 0].copy()
    d["datadate"] = pd.to_datetime(d["datadate"]).astype("datetime64[ns]")
    d["div_num"] = d["div"].astype(float) * numeraire_factor(d["datadate"], d["curcddv"], fx)
    x = f[["gvkey", "iid", "datadate"]].reset_index().merge(d.rename(columns={"datadate": "exd"}), on=["gvkey", "iid"])
    lag = (x["exd"] - x["datadate"]).dt.days
    f["divps_fy"] = x[(lag > -366) & (lag <= 0)].groupby("index")["div_num"].sum()
    f["divps_next"] = x[(lag > 0) & (lag <= 366)].groupby("index")["div_num"].sum()
    f = f[f["div_acc"] > 0].copy()
    y_acc = f["div_acc"] / f["mktcap"].astype(float)
    f["ratio_fy"] = f["divps_fy"] / f["price"] / y_acc
    f["ratio_next"] = f["divps_next"] / f["price"] / y_acc
    be = f["ceq"].fillna(f["seq"]) * f["fx_acc"]
    f["lbm"] = np.log10(be.where(be > 0) / f["mktcap"].astype(float))
    f["sic2"], f["year"] = f["sic"].astype(str).str[:2], f["datadate"].dt.year
    f["lbm_dev"] = f["lbm"] - f.groupby(["sic2", "year"])["lbm"].transform("median")
    lf = np.log10(factor)
    lr = np.log10(f[["ratio_fy", "ratio_next"]].where(f[["ratio_fy", "ratio_next"]] > 0))
    both_high, both_low = (lr > lf).all(axis=1), (lr < -lf).all(axis=1)
    bm_agrees = (both_high & (f["lbm_dev"] < -0.5)) | (both_low & (f["lbm_dev"] > 0.5))
    ym = pd.MultiIndex.from_frame(mem.assign(year=mem["datadate"].dt.year)[["gvkey", "year"]].drop_duplicates())
    member_year = pd.MultiIndex.from_arrays([f["gvkey"], f["year"]]).isin(ym)
    flag = (both_high | both_low) & bm_agrees & member_year
    return f[flag][["gvkey", "conm", "datadate", "iid", "mktcap", "ratio_fy", "ratio_next", "lbm_dev"]]


def run_europe(cfg: dict, div_factor: float) -> None:
    ins = ROOT / cfg["output"]["inspect_dir"]
    mem = members(ins, int(cfg["universe"]["size"]))
    jumps, switches = europe_jumps(cfg, ins, mem)
    divs = europe_dividends(cfg, ins, mem, div_factor)
    for name, df in [("jumps", jumps), ("switches", switches), ("dividends", divs)]:
        df.to_csv(ins / f"cap_checks_{name}.csv", index=False)
    L = [f"{cfg['market']}: cap checks on the as-of top {cfg['universe']['size']} (scripts/check_caps.py)", "",
         f"1. Same pricing line, |cap change - price change| >= 2x: {len(jumps)} company-months, "
         f"{jumps['gvkey'].nunique()} companies (mergers, recapitalisations and spin-offs are real ones):",
         jumps.sort_values("datadate").to_string(index=False, max_rows=80), "",
         f"2. Pricing line switched and the cap moved 2x: {len(switches)}:",
         switches.sort_values("datadate").to_string(index=False, max_rows=80), "",
         f"3. Dividend yields of price feed and accounts {div_factor:g}x apart in both windows, book-to-market "
         f"agreeing: {len(divs)} member firm-years:", divs.to_string(index=False, max_rows=80)]
    (ins / "cap_checks.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def run_us(cfg: dict, tol: float) -> None:
    raw_dir, ins = ROOT / cfg["raw"]["dir"], ROOT / cfg["output"]["inspect_dir"]
    raw = up.load_raw(raw_dir)
    home = up.home_caps(raw.global_home)
    u = pd.read_parquet(ins / "universe_asof.parquet")
    u["cap_co"] = u["cap_co"].astype(float)
    u["key"] = u["rank_month"].dt.to_timestamp(how="end").dt.normalize()
    ccm = raw.ccm.copy()
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    lk = u[["primary_permno", "key"]].drop_duplicates().merge(ccm, left_on="primary_permno", right_on="permno")
    lk = lk[(lk["linkdt"] <= lk["key"]) & (lk["key"] <= lk["linkenddt"])]
    lk = lk.sort_values("linkprim").drop_duplicates(["primary_permno", "key"])[["primary_permno", "key", "gvkey"]]
    r = u[u["sharetype"] == "AD"].merge(lk, on=["primary_permno", "key"], how="left")
    r = r.merge(home.rename(columns={"month": "rank_month"}), on=["gvkey", "rank_month"], how="left")
    r["ratio"] = r["cap_co"] / (r["cap_home"] / 1000.0)           # cap_co in USD mn
    off = r[(r["ratio"] > tol) | (r["ratio"] < 1 / tol)]
    off = off[["primary_permno", "gvkey", "month", "cap_rank", "cap_source", "cap_co", "cap_home", "n_classes_home", "ratio"]]
    off.to_csv(ins / "cap_checks_receipts.csv", index=False)
    L = ["US: receipt company caps against their issuers' home-market caps (scripts/check_caps.py)", "",
         f"receipt member-months {len(r):,}, with a home cap {int(r['cap_home'].notna().sum()):,}; cap source "
         f"{r['cap_source'].value_counts().to_dict()}", "",
         f"Outside {1 / tol:.2f}..{tol:g}x the home cap after receipt_caps: {len(off)} member-months "
         f"(a receipt of one class of several, or a dual listing, sits below its company):",
         off.groupby("gvkey").agg(months=("month", "size"), first=("month", "min"), last=("month", "max"),
                                  ratio=("ratio", "median"), best_rank=("cap_rank", "min"),
                                  classes=("n_classes_home", "max"), source=("cap_source", "first"))
         .sort_values("months", ascending=False).to_string(max_rows=60)]
    (ins / "cap_checks.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--div-factor", type=float, default=4.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if "extract" in cfg:
        run_europe(cfg, args.div_factor)
    else:
        run_us(cfg, float(cfg.get("universe", {}).get("receipt_home_tol", 1.5)))


if __name__ == "__main__":
    main()
