"""Implementation robustness of the saved K = 30 books: every run's weights.parquet booked again under
other execution assumptions, each alone and then together. Nothing is retrained or tuned.

    python scripts/robustness_stored_weights.py --rows headline --out <dir> \
        --runs search_t4q=<runs dir> --runs search_t4q_eu=<runs dir> --runs pca_perk=<runs dir> ...

Variants (src/afe/evaluation/robustness.py):
  baseline        lag 0, 5bp x turnover + 1bp x short, numeraire returns: must reproduce metrics.json
  a_delay         weights traded one close later (trading.execute lag 1), costs on the delayed path
  b_spread        half spread x |dw| instead of 5bp: US closing quoted spread, Europe Corwin-Schultz
  b_spread_cs     Corwin-Schultz in both markets (US from scripts/fetch_us_highlow.py), for comparability
  b_spread_ar     Abdi-Ranaldo in both markets (cross-check)
  c_tax           Europe: + statutory purchase taxes (configs/tax_calendar.yaml); c_tax_ch adds Swiss
  d_local         Europe: local-currency returns instead of EUR
  e_all           a + b + c + d together (US: a + b)
Spread estimates: monthly, the estimate of the month before the trade's month, winsorised per month over
the universe, missing = that month's members' median. Writes runs.csv (spec x variant x seed),
summary.csv (mean, seed sd), table.tex (results_v2.tex), spread_levels.csv, fills.csv, non_eur_share.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data.europe_panel import corwin_schultz  # noqa: E402
from afe.evaluation import robustness as rb  # noqa: E402

ISO3 = {"AUT": "AT", "BEL": "BE", "BMU": "BM", "CHE": "CH", "DEU": "DE", "DNK": "DK", "ESP": "ES", "FIN": "FI",
        "FRA": "FR", "GBR": "GB", "GIB": "GI", "GRC": "GR", "IRL": "IE", "ITA": "IT", "JEY": "JE", "LUX": "LU",
        "NLD": "NL", "NOR": "NO", "PRT": "PT", "SWE": "SE", "USA": "US", "GGY": "GG", "IMN": "IM", "CYM": "KY"}
EURO_LEGACY = {"EUR", "DEM", "FRF", "ITL", "ESP", "NLG", "BEF", "ATS", "FIM", "PTE", "IEP", "LUF", "GRD"}
REPRO_TOL = 1e-4
PERIODS = [(1998, 2003), (2004, 2021), (1998, 2009), (2010, 2021)]   # mid-2003: CRSP NYSE closing quotes narrow 7x


def run_table(rows: list[dict], runs: dict[str, Path]) -> pd.DataFrame:
    out = []
    for r in rows:
        root = runs.get(r["matrix"], ROOT / "runs")
        t = pd.read_csv(root / "_matrix" / r["matrix"] / "summary.csv")
        t = t[(t["spec"] == r["spec"]) & (t["K"] == 30)]
        if t.empty:
            raise SystemExit(f"no K = 30 rows of {r['spec']} in {r['matrix']}")
        for x in t.itertuples():
            out.append({**r, "seed": int(x.seed), "run_dir": root / Path(x.run_dir.replace("\\", "/")).name})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ spread inputs
def us_spreads(mc: dict, pool: set[str]) -> dict[str, pd.DataFrame]:
    fr = []
    for d in mc["crsp_daily"]:
        for y in range(1997, 2022):
            x = pd.read_parquet(ROOT / d / f"{y}.parquet", columns=["permno", "date", "bid", "ask"])
            x["sec_id"] = x["permno"].astype(str)
            fr.append(x[x["sec_id"].isin(pool)])
    out = {"quoted": rb.quoted_spread_monthly(pd.concat(fr).drop_duplicates(["sec_id", "date"]))}
    hl = ROOT / mc["crsp_highlow"]
    if hl.exists():
        d = pd.read_parquet(hl)
        for c in ("high", "low", "close"):
            d[c] = d[c].astype("float64")             # nullable from WRDS; NA -> NaN
        d = d.assign(sec_id=d["permno"].astype(str))[lambda x: x["sec_id"].isin(pool)].sort_values(["sec_id", "date"])
        traded = d["close"] > 0                        # CRSP: negative price = bid/ask average, no trade
        cs = pd.DataFrame({"gvkey": d["sec_id"], "iid": "1", "prchd": d["high"], "prcld": d["low"],
                           "prccd": d["close"].abs(), "prcstd": np.where(traded, 10, 0)})
        d["cs"] = corwin_schultz(cs).to_numpy()
        out["cs"] = rb.daily_mean_monthly(d, "cs")
        out["ar"] = rb.abdi_ranaldo_monthly(d[traded].assign(line="1", close=d["close"].abs()))
    return out


def eu_daily(mc: dict) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / mc["company_daily"], columns=["gvkey", "iid", "date", "traded", "prchd", "prcld",
                                                             "prccd", "cs", "ret_local"])
    d = d.rename(columns={"gvkey": "sec_id", "iid": "line", "prccd": "close", "prchd": "high", "prcld": "low"})
    for c in ("close", "high", "low", "cs", "ret_local"):
        d[c] = d[c].astype("float64")
    d["traded"] = d["traded"].fillna(False).astype(bool)
    return d


def eu_spreads(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = d[d["traded"]]
    return {"cs": rb.daily_mean_monthly(t, "cs"), "ar": rb.abdi_ranaldo_monthly(t)}


def eu_local_returns(m: rb.Market, d: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Local-currency return matrix: company_daily ret_local where the row exists; a holiday-filled row
    without one has local return 0 (the fill's definition); any other missing row keeps the EUR return."""
    loc = rb.wide(m, d[d["sec_id"].isin(m.cols)], "ret_local")
    has = ~np.isnan(rb.wide(m, m.rows, "ret"))
    closed = m.closed if m.closed is not None else np.zeros_like(has)
    RL = np.where(np.isnan(loc), np.where(closed, 0.0, m.R), loc)
    RL = np.where(has, RL, 0.0)
    info = {"rows": int(has.sum()), "rows_without_local": int((has & np.isnan(loc)).sum())}
    return RL, info


# ------------------------------------------------------------------ main loop
def variants(market: str, primary: str, spreads: dict) -> dict[str, dict]:
    v = {"baseline": {}, "a_delay": {"lag": 1}, "b_spread": {"spread": primary}}
    for s in ("cs", "ar"):
        if s in spreads:
            v[f"b_spread_{s}"] = {"spread": s}
    if market == "eu":
        v.update({"c_tax": {"tax": "main"}, "c_tax_ch": {"tax": "with_ch"}, "d_local": {"local": True},
                  "e_all": {"lag": 1, "spread": primary, "tax": "main", "local": True}})
    else:
        v["e_all"] = {"lag": 1, "spread": primary}
    return v


def score_market(market: str, runs: pd.DataFrame, cfg: dict, out: Path) -> pd.DataFrame:
    mc = cfg["markets"][market]
    ids = set().union(*[set(pd.read_parquet(p / "weights.parquet", columns=["sec_id"])["sec_id"]) for p in runs.run_dir])
    m = rb.load_market(ROOT / mc["data_dir"], ids, return_gaps=mc.get("return_gaps", "masked"))  # as the runs
    members = pd.read_parquet(ROOT / mc["data_dir"] / "universe.parquet", columns=["month", "sec_id"])
    members["month"] = pd.to_datetime(members["month"])
    first = json.loads((runs.run_dir.iloc[0] / "metrics.json").read_text())
    t = rb.oos_index(m, first["first"], first["last"])
    print(f"{market}: {len(m.cols)} names, {len(t)} days", flush=True)

    daily = eu_daily(mc) if market == "eu" else None
    spreads = eu_spreads(daily) if market == "eu" else us_spreads(mc, set(m.cols))
    HS, fills = {}, []
    for name, monthly in spreads.items():
        HS[name], info = rb.half_spread(m, monthly, members, t, cfg["winsor"])
        fills.append({"market": market, "estimator": name, "member_months": info["member_months"],
                      "filled_share": info["member_months_filled"] / info["member_months"], "_miss": info["_missing"]})
        lv = members.assign(ym=members["month"].dt.to_period("M")).merge(
            monthly.assign(ym=monthly["month"].dt.to_period("M"))[["ym", "sec_id", "spread"]], on=["ym", "sec_id"], how="left")
        lv = lv[lv["month"].dt.year.between(1997, 2021)]
        lv = lv.groupby(lv["month"].dt.year)["spread"]
        lv.agg(["median", "mean", "count"]).assign(market=market, estimator=name).to_csv(
            out / "spread_levels.csv", mode="a", header=not (out / "spread_levels.csv").exists())
    taxes, R_local = {}, None
    if market == "eu":
        hdr = pd.read_parquet(ROOT / mc["company_header"], columns=["gvkey", "fic"]).drop_duplicates("gvkey")
        home = hdr.set_index("gvkey")["fic"].map(ISO3)
        cal = cfg["tax_calendar"]
        taxes = {"main": rb.tax_rates(m, t, home, rb.load_tax_calendar(ROOT / cal)),
                 "with_ch": rb.tax_rates(m, t, home, rb.load_tax_calendar(ROOT / cal, include_optional=True))}
        R_local, linfo = eu_local_returns(m, daily)
        print(f"  local returns: {linfo}", flush=True)
        non_eur_share(ROOT / mc["data_dir"], members, out)

    rows = []
    for r in runs.itertuples():
        st = json.loads((r.run_dir / "metrics.json").read_text())
        tgt = rb.targets(r.run_dir, m)
        H = {0: rb.held(tgt, m, t, 0), 1: rb.held(tgt, m, t, 1)}
        for vname, v in variants(market, mc["spread"], spreads).items():
            if v.get("spread") and v["spread"] not in HS:
                continue
            Hv = H[v.get("lag", 0)]
            R = R_local[t] if v.get("local") else m.R[t]
            d = rb.book(Hv, R, cfg["tc"], cfg["sc"], HS.get(v.get("spread")), taxes.get(v.get("tax")))
            s = rb.summarise(d, cfg["sc"])
            if vname == "baseline":
                check(s, st, r)
            yr = m.dates[t].year
            for lo, hi in PERIODS:
                sp = rb.summarise(d[(yr >= lo) & (yr <= hi)], cfg["sc"])
                s[f"SR_net_{lo}_{hi}"], s[f"eff_cost_bps_{lo}_{hi}"] = sp["SR_net"], sp["eff_cost_bps"]
            if v.get("spread"):
                dH = np.abs(np.diff(Hv, axis=0, prepend=0 * Hv[:1]))
                miss = next(f["_miss"] for f in fills if f["estimator"] == v["spread"])
                s["traded_filled_share"] = float((dH * miss).sum() / dH.sum())
            rows.append({"market": market, "model": r.model, "spec": r.spec, "seed": r.seed, "variant": vname,
                         "run": r.run_dir.name, **s})
        print(f"  {r.spec} s{r.seed}: done", flush=True)
    pd.DataFrame([{k: v for k, v in f.items() if k != "_miss"} for f in fills]).to_csv(
        out / "fills.csv", mode="a", header=not (out / "fills.csv").exists(), index=False)
    return pd.DataFrame(rows)


def check(s: dict, st: dict, r) -> None:
    """Step 0: the baseline booking must give the run's own metrics, or nothing else is reported."""
    ref = {"SR": st["gross"]["SR"], "SR_net": st["net"]["SR"], "turnover": st["turnover_daily"],
           "short": st["short_exposure"], "break_even_bps": st["break_even_turnover_cost_bps"]}
    bad = {k: (s[k], v) for k, v in ref.items() if abs(s[k] - v) > REPRO_TOL * max(1.0, abs(v))}
    if bad:
        raise SystemExit(f"{r.run_dir.name} ({r.spec} s{r.seed}) does not reproduce: {bad}")


def non_eur_share(data_dir: Path, members: pd.DataFrame, out: Path) -> None:
    """Share of universe member-days quoted outside EUR (any currency but EUR; outside the euro area's legacy
    currencies too), by year: how far the local-currency variant can move the book at most."""
    x = pd.read_parquet(data_dir / "returns.parquet", columns=["date", "sec_id", "currency"])
    x["date"] = pd.to_datetime(x["date"])
    x = x.assign(ym=x["date"].dt.to_period("M"))
    mem = members.assign(ym=members["month"].dt.to_period("M"))[["ym", "sec_id"]]
    x = x.merge(mem, on=["ym", "sec_id"])
    x = x[x["date"].dt.year.between(1998, 2021)]
    y = x.groupby(x["date"].dt.year).agg(non_eur=("currency", lambda c: (c != "EUR").mean()),
                                         non_euro_area=("currency", lambda c: (~c.isin(EURO_LEGACY)).mean()))
    y.to_csv(out / "non_eur_share.csv")


def retrained(cfg: dict, rows: str, runs: dict[str, Path]) -> pd.DataFrame:
    t = run_table(cfg["retrained_lag1"][rows], runs)
    out = []
    for r in t.itertuples():
        st = json.loads((r.run_dir / "metrics.json").read_text())
        out.append({"market": r.market, "model": r.model, "spec": r.spec, "seed": r.seed, "variant": "a_retrained_lag1",
                    "run": r.run_dir.name, "SR": st["gross"]["SR"], "mu_pct": st["gross"]["mu_pct"],
                    "sigma_pct": st["gross"]["sigma_pct"], "SR_net": st["net"]["SR"], "mu_net_pct": st["net"]["mu_pct"],
                    "sigma_net_pct": st["net"]["sigma_pct"], "turnover": st["turnover_daily"],
                    "short": st["short_exposure"], "break_even_bps": st["break_even_turnover_cost_bps"],
                    "eff_cost_bps": 1e4 * cfg["tc"], "tax_pct": 0.0})
    return pd.DataFrame(out)


TABLE_ROWS = [("baseline", "Baseline: 5bp per unit traded, 1bp short"),
              ("a_delay", "(a) Traded one close later"),
              ("a_retrained_lag1", "\\quad retrained with a one-day lag$^{\\dagger}$"),
              ("b_spread", "(b) Half spread (US quoted, Europe CS)"),
              ("b_spread_cs", "\\quad half spread, CS in both markets"),
              ("c_tax", "(c) + transaction taxes"),
              ("c_tax_ch", "\\quad incl.\\ Swiss turnover tax"),
              ("d_local", "(d) Local-currency returns"),
              ("e_all", "(e) All of (a)--(d)")]
COLS = [(mk, mo) for mk in ("us", "eu") for mo in ("attention", "pca_longconv", "pca_ou")]


NOTES = (r"January 1998 to December 2021, $K=30$, the weights saved by each run booked again; nothing is "
         r"retrained. Attention and PCA + LongConv: mean over five seeds, seed standard deviation of "
         r"$SR_{\mathrm{net}}$ in parentheses; OU: one run. Every variant keeps the 1bp short fee. (a) weights "
         r"traded one close later than in the baseline, costs on the delayed positions; $^{\dagger}$ not a "
         r"re-scoring: the model retrained with the one-day lag (three seeds, the $K=30$ values of its own search). "
         r"(b) half the bid-ask spread per unit traded instead of 5bp: monthly estimate of the month before the "
         r"trade, winsorised at 1/99\% over the universe, missing values set to that month's median; US: closing "
         r"quoted spread (CRSP), Europe: Corwin--Schultz high-low estimator. (c) statutory taxes on purchases by "
         r"country of incorporation (UK 50bp, Ireland 100bp, France 20/30bp from August 2012, Italy 12/10bp from "
         r"March 2013, Spain 20bp from January 2021; thresholds on prior-month market cap), on top of the "
         r"baseline cost; the Swiss row adds 7.5bp on each purchase and sale of Swiss companies. (d) local-currency "
         r"instead of euro returns. (e) (a)--(d) together; in the US (a) and (b). Break-even: cost per unit "
         r"traded at which the mean net return is zero, after the short fee and taxes. Trading cost charged: "
         r"mean spread cost per unit of turnover. Panel D: the same books from 2004, after the closing quotes of "
         r"NYSE stocks in CRSP narrow about sevenfold within mid-2003.")
EOL = " \\\\"   # LaTeX row end


def fmt(x: float, nd: int) -> str:
    v = f"{x:.{nd}f}"
    return v.replace("-", "$-$") if x < 0 else v


def make_table(summ: pd.DataFrame, out: Path) -> None:
    """Table in the style of results_v2.tex: panel A net SR (seed sd), panel B break-even cost (bp),
    panel C trading cost charged per unit traded (bp)."""
    s = summ.set_index(["variant", "market", "model"])

    def cell(v, mk, mo, col):
        if (v, mk, mo) not in s.index:
            return "n/a" if mk == "us" and v[0] in "cd" else "--"
        x = s.loc[(v, mk, mo)]
        if col == "SR_net":
            sd = x["SR_net_sd"]
            return fmt(x["SR_net"], 2) + (f" ({sd:.2f})" if np.isfinite(sd) else "")
        return fmt(x[col], 2 if col.startswith("SR") else 1)

    lines = [r"\begin{table}[htbp]", r"\caption{Implementation robustness of the $K=30$ models}\label{tab:implementation}",
             r"\footnotesize", r"\begin{adjustbox}{max width=\linewidth,center}", r"\begin{tabular}{lcccccc}",
             r"\toprule", r" & \multicolumn{3}{c}{US} & \multicolumn{3}{c}{Europe}" + EOL,
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r" & Attention & PCA+LongConv & PCA+OU & Attention & PCA+LongConv & PCA+OU" + EOL, r"\midrule"]
    for title, col in (("A. Net Sharpe ratio", "SR_net"), ("B. Break-even cost per unit traded (bp)", "break_even_bps"),
                       ("C. Trading cost charged per unit traded (bp)", "eff_cost_bps"),
                       ("D. Net Sharpe ratio, 2004--2021", "SR_net_2004_2021")):
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + "}}" + EOL)
        for v, label in TABLE_ROWS:
            if col == "eff_cost_bps" and not (v == "baseline" or v.startswith("b_") or v == "e_all"):
                continue
            if col == "SR_net_2004_2021" and v in ("a_retrained_lag1", "c_tax_ch"):
                continue
            lines.append(label + " & " + " & ".join(cell(v, mk, mo, col) for mk, mo in COLS) + EOL)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{adjustbox}", "", r"\notes{" + NOTES + "}", r"\end{table}"]
    (out / "table.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/robustness_k30.yaml")
    ap.add_argument("--rows", default="headline")
    ap.add_argument("--runs", action="append", default=[], help="<matrix>=<runs dir> (default: this repo's runs/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    runs = {k: Path(v) for k, v in (a.split("=", 1) for a in args.runs)}
    args.out.mkdir(parents=True, exist_ok=True)
    for f in ("spread_levels.csv", "fills.csv"):
        (args.out / f).unlink(missing_ok=True)
    table = run_table(cfg["rows"][args.rows], runs)
    res = pd.concat([score_market(mk, table[table.market == mk], cfg, args.out) for mk in ("us", "eu")]
                    + [retrained(cfg, args.rows, runs)], ignore_index=True)
    res.to_csv(args.out / "runs.csv", index=False, float_format="%.6g")
    num = [c for c in res.columns if res[c].dtype.kind == "f"]
    g = res.groupby(["market", "model", "variant"], sort=False)
    summ = g[num].mean().join(g["SR_net"].std(ddof=1).rename("SR_net_sd")).join(g.size().rename("n"))
    summ = summ.reset_index()
    summ.to_csv(args.out / "summary.csv", index=False, float_format="%.6g")
    make_table(summ, args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
