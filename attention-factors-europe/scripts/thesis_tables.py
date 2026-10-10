"""The thesis tables from the run matrix (configs/thesis_matrix.yaml) and the searches.

    python scripts/thesis_tables.py --out <dir> [--matrix runs/_matrix/thesis]
                                    [--main-search runs/<US headline search run>]

Writes <out>/tables/<name>.csv (full precision), .md and .tex (formatted, with caption and
notes), and <out>/tables/index.json (title, caption, notes, source of every table), which
scripts/thesis_handoff.py turns into the index. Every number comes from a run directory or a
data table of the build; the paper's numbers appear only in columns labelled "paper".
Statistics are per-seed then averaged (sd over seeds in parentheses); single-series results
(alphas) use the equal-weighted combination of a spec's seeds.
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
sys.path.insert(0, str(ROOT / "scripts"))

from afe.data.characteristics import REGISTRY  # noqa: E402
from afe.evaluation import metrics, stats  # noqa: E402
from afe.evaluation import thesis_data as td  # noqa: E402
from run_matrix import selected_values  # noqa: E402

MODELS = {"attention": "Attention factors", "pca_longconv": "PCA + LongConv", "pca_ou": "PCA + OU threshold"}
MARKETS = {"us": "US", "eu": "Europe"}
DATA = {"us": "data/us/shared", "eu": "data/europe17/holfill"}
THEMES = ["Past Returns", "Value", "Investment", "Profitability", "Intangibles", "Trading Frictions"]
GROUPS = {"past_returns": "Past returns", "investment": "Investment", "profitability": "Profitability",
          "intangibles": "Intangibles", "value": "Value", "trading_frictions": "Trading frictions"}
SUBPERIODS = [("1998-2009", "1998", "2009"), ("2010-2021", "2010", "2021")]
# statutory taxes on purchases (the floor of CLAUDE.md): country -> [(from, rate)]
TAXES = {"GB": [("1900-01-01", 0.005)], "FR": [("2012-08-01", 0.002), ("2017-01-01", 0.003)],
         "IT": [("2013-03-01", 0.0012), ("2014-01-01", 0.0010)]}
COUNTRY = {"GBR": "United Kingdom", "FRA": "France", "DEU": "Germany", "CHE": "Switzerland", "NLD": "Netherlands",
           "ITA": "Italy", "ESP": "Spain", "SWE": "Sweden"}   # universe_asof.country (ISO alpha-3)

INDEX: list[dict] = []


def spec(m: str, model: str) -> str:
    return f"{m}_{model}" if model != "attention" else f"{m}_attention"


# ------------------------------------------------------------------ writing


def fmt(x, d=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        return f"{x:,.{d}f}"
    return str(x)


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("–", "--").replace("−", "$-$"))


def write(out: Path, name: str, num: pd.DataFrame, disp: pd.DataFrame, title: str, caption: str,
          notes: list[str], source: str) -> None:
    """num: numbers (CSV); disp: the formatted table (strings), index = row labels."""
    d = out / "tables"
    d.mkdir(parents=True, exist_ok=True)
    num.to_csv(d / f"{name}.csv")
    cols = [str(c) for c in disp.columns]
    lab = disp.index.name or ""
    lines = [f"## {title}", "", caption, "", "| " + " | ".join([lab] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(disp.index, disp.values)]
    lines += [""] + [f"*{n}*" for n in notes]
    (d / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", rf"\caption{{{tex_escape(title)}}}",
           rf"\label{{tab:{name}}}", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
           " & ".join([tex_escape(lab)] + [tex_escape(c) for c in cols]) + r" \\", r"\midrule"]
    tex += [" & ".join([tex_escape(i)] + [tex_escape(v) for v in row]) + r" \\" for i, row in zip(disp.index, disp.values)]
    tex += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\parbox{\linewidth}{\scriptsize " +
            tex_escape(caption + " " + " ".join(notes)) + "}", r"\end{table}"]
    (d / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    INDEX.append({"name": name, "title": title, "caption": caption, "notes": notes, "source": source,
                  "files": [f"tables/{name}.{e}" for e in ("csv", "md", "tex")]})
    print(f"  {name}: {title}")


def per_seed_stats(M: td.Matrix, sp: str, K: int, seeds=None, period: tuple[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for seed, d in M.per_seed(sp, K, seeds).items():
        if period:
            d = d.loc[period[0]:period[1]]
        rows.append({"seed": seed, **td.perf(d)})
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return df.mean(numeric_only=True), df.std(numeric_only=True, ddof=1)


# ------------------------------------------------------------------ tables


def t01_sample(out: Path) -> None:
    years = [1998, 2002, 2006, 2010, 2014, 2018, 2021]
    num, disp = {}, {}
    comp = {}
    for m, mk in [("us", "us"), ("eu", "europe17")]:
        u = td.universe_by_month(mk)
        u["year"] = u["date"].dt.year
        jan = u[u["date"].dt.month == 1]
        g = jan.groupby("year")["cap_usd_bn"]
        stats_ = pd.DataFrame({"members": jan.groupby("year").size(), "min_cap": g.min(), "median_cap": g.median(),
                               "max_cap": g.max()}).loc[years]
        for c in stats_.columns:
            num[(MARKETS[m], c)] = stats_[c]
        disp[(MARKETS[m], "members")] = stats_["members"].map(fmt)
        disp[(MARKETS[m], "smallest ($bn)")] = stats_["min_cap"].map(lambda x: fmt(x, 1))
        disp[(MARKETS[m], "median ($bn)")] = stats_["median_cap"].map(lambda x: fmt(x, 1))
        disp[(MARKETS[m], "largest ($bn)")] = stats_["max_cap"].map(lambda x: fmt(x, 0))
        oos = u[(u["year"] >= 1998) & (u["year"] <= 2021)]
        comp[MARKETS[m]] = oos["sec_id"].nunique()
        if m == "eu":
            cc = jan[jan["year"].isin(years)].groupby(["year", "country"]).size().unstack(fill_value=0)
            top = list(COUNTRY)
            cc = cc.reindex(columns=top + [c for c in cc.columns if c not in top], fill_value=0)
            share = 100 * cc.div(cc.sum(axis=1), axis=0)
            share = pd.concat([share[top], share.drop(columns=top).sum(axis=1).rename("Other")], axis=1)
            share = share.rename(columns=COUNTRY)
    num_df = pd.DataFrame(num)
    disp_df = pd.DataFrame(disp)
    disp_df.columns = [f"{a} {b}" for a, b in disp_df.columns]
    disp_df.index.name = "January"
    write(out, "T01_sample", num_df, disp_df, "Table 1. The two universes",
          "Members in January and the market capitalisation that ranked them (previous month end), billions of USD: "
          "the 500 largest US listings (paper-style universe, CRSP CIZ) and the 200 largest European companies "
          "of europe17 (17 exchange countries, one line per company, Compustat Global; numeraire converted at "
          "the day's USD rate). US capitalisation is company level (all share classes).",
          [f"Distinct companies 1998-2021: US {comp['US']:,}, Europe {comp['Europe']:,}."],
          "data/us/private/universe_asof.parquet, data/europe17/private/universe_asof.parquet")
    share.index.name = "January"
    write(out, "T01b_europe_countries", share, share.map(lambda x: fmt(x, 1)),
          "Table 1b. Country composition of the European universe",
          "Share of the 200 members by country, percent, January.",
          ["Other: Belgium, Denmark, Finland, Norway, Ireland, Portugal, Austria, Luxembourg and Greece."],
          "data/europe17/private/universe_asof.parquet (country)")


def t02_characteristics(out: Path) -> None:
    lo, hi = pd.Period("1998-01", "M"), pd.Period("2021-12", "M")
    ec = pd.read_parquet(td.ROOT / "data/europe17/private/coverage_member_months.parquet")
    ec = ec[(ec["month"] >= lo) & (ec["month"] <= hi)]
    uni = pd.read_parquet(td.ROOT / "data/us/shared/universe.parquet")
    uni["ym"] = uni["month"].dt.to_period("M") - 1          # monthly characteristics as of the month before
    uni = uni[(uni["ym"] >= lo - 1) & (uni["ym"] <= hi - 1)]
    um = pd.read_parquet(td.ROOT / "data/us/private/characteristics_monthly.parquet")
    um["sec_id"] = um["permno"].astype(str)
    um["ym"] = pd.PeriodIndex(um["month"], freq="M")
    um = uni[["ym", "sec_id"]].merge(um, on=["ym", "sec_id"], how="left")
    rows = []
    for theme in THEMES:
        for name, c in sorted(((n, c) for n, c in REGISTRY.items() if c.theme == theme), key=lambda x: x[0].lower()):
            us_cov = um[name].notna().mean() if name in um.columns else np.nan
            eu_cov = ec[name].mean() if name in ec.columns else np.nan
            rows.append({"theme": theme, "characteristic": name, "frequency": c.frequency, "reference": c.reference,
                         "us_coverage": 100 * us_cov, "eu_coverage": 100 * eu_cov})
    t = pd.DataFrame(rows).set_index("characteristic")
    disp = t.assign(us_coverage=t["us_coverage"].map(lambda x: fmt(x, 0)),
                    eu_coverage=t["eu_coverage"].map(lambda x: fmt(x, 0)))
    disp.columns = ["Theme", "Frequency", "Source", "US coverage (%)", "Europe coverage (%)"]
    disp.index.name = "Characteristic"
    write(out, "T02_characteristics", t, disp, "Table 2. Firm characteristics by theme",
          "The 39 characteristics of Chen, Pelger and Zhu (2024) in their six themes, built identically for both "
          "markets and rank-normalised each day. Coverage: share of member-months 1998-2021 with the value "
          "observed before the fill (last observed value, then the cross-sectional median).",
          ["The model input adds the cross-sectional median of each characteristic and the risk-free rate: 79 "
           "features. US daily characteristics (Ret_D1, Ret_W1, STD_W1, Vol) are not in the monthly coverage file "
           "(–)."], "REGISTRY (src/afe/data/characteristics.py); coverage tables of both builds")


def search_dirs(matrix_dir: Path, main: Path) -> dict[str, Path]:
    return {"attention": main, **{k: matrix_dir / "searches" / k for k in ["us_raw", "us_lag1", "pca", "pca_lag1", "ou"]}}


def t03_hyperparameters(out: Path, matrix_dir: Path, main: Path) -> None:
    dirs = search_dirs(matrix_dir, main)
    labels = {"attention": "Attention (headline)", "us_raw": "Attention, raw input", "us_lag1": "Attention, lag 1",
              "pca": "PCA + LongConv", "pca_lag1": "PCA + LongConv, lag 1", "ou": "PCA + OU"}
    vals, info = {}, {}
    space = {}
    for k, d in dirs.items():
        man = json.loads((d / "manifest.json").read_text())["config"]
        w, v = selected_values(d)
        vals[labels[k]] = v
        S = man["search"]
        for key, spec_ in {**S["search"].get("space", {}), **{g: {"choice": x} for g, x in S["search"].get("grid", {}).items()}}.items():
            if "choice" in spec_:
                space.setdefault(key, "{" + ", ".join(str(c) for c in spec_["choice"]) + "}")
            else:
                r = f"log [{spec_['log'][0]:g}, {spec_['log'][1]:g}]"
                space.setdefault(key, r + (f", off w.p. {spec_['off']}" if spec_.get("off") else ""))
        mets = json.loads((d / "metrics.json").read_text())
        base = next((r for r in mets.get("finalists", []) if r["name"] == "base"), None)
        info[labels[k]] = {"candidate": w["name"], "val_net": w["val_net_SR"],
                           "base_val_net": base["val_net_SR"] if base else np.nan,
                           "points": mets.get("n_search_points"), "fixed": S.get("fixed") or {}}
    keys = ["objective.lambda_var", "policy.input_scale", "model.score_std_target", "model.lambda_ridge",
            "training.batch_days", "policy.input", "policy.input_normalise", "model.level_hidden",
            "execution.lag", "policy.c_thresh", "policy.c_crit"]
    t = pd.DataFrame({lab: {k: v.get(k, "") for k in keys} for lab, v in vals.items()})
    t.insert(0, "Search range", [space.get(k, "fixed" if k == "execution.lag" else "") for k in keys])
    t.insert(0, "Paper (Table 4)", [td.PAPER_T4.get(k, "–") if k in td.PAPER_T4 else "not stated" for k in keys])
    extra = pd.DataFrame({lab: {"selected candidate": i["candidate"], "validation net SR (winner)": fmt(i["val_net"]),
                                "validation net SR (base config)": fmt(i["base_val_net"]),
                                "candidates searched": i["points"]} for lab, i in info.items()})
    t = pd.concat([t, extra]).fillna("")
    t = t.map(lambda x: "off" if x is None else (f"{x:.3g}" if isinstance(x, float) else x))
    t.index.name = "Hyperparameter"
    write(out, "T03_hyperparameters", t, t, "Table 3. Hyperparameters and how they were selected",
          "Every reported return uses values selected on the first US training window: fitted on 1990-1995, "
          "scored by the net Sharpe ratio on 1996-1997, nothing later loaded (the paper's Appendix B protocol). "
          "Sobol points over the knobs the paper leaves open (128 for attention, 64 for PCA + LongConv, the 3 x 3 "
          "grid of Guijarro-Ordonez et al. for the OU thresholds); the best five and the base config rerun on "
          "seeds 0-2, the best mean wins. Europe uses the US values unchanged.",
          ["Fixed at the paper's Table 4 values: hidden 32, dropout 0.1, embedding 32, 30 epochs, one layer, "
           "learning rate 0.003, weight decay 0.05 on LongConv, squash 0.001, lookback 30. off = no score temperature."],
          "search runs (manifest.json, metrics.json): " + ", ".join(str(p.relative_to(td.ROOT)) for p in dirs.values()))


def table2(M: td.Matrix, m: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    num, disp = [], []
    for model in MODELS:
        sp = spec(m, model)
        for K in M.Ks(sp):
            ps = per_seed_stats(M, sp, K)
            mu, sd = mean_sd(ps)
            row = {"model": MODELS[model], "K": K, "seeds": len(ps), **mu.drop("seed").to_dict(),
                   "SR_sd": sd["SR"], "SR_net_sd": sd["SR_net"]}
            if m == "us":
                p = td.PAPER_T2[model].get(K)
                row.update({"paper_SR": p[0] if p else np.nan, "paper_SR_net": p[3] if p else np.nan})
            num.append(row)
    d0 = M.combination(spec(m, "attention"), 30)
    mk = metrics.annualised(d0["mkt_ew"])
    num.append({"model": "Market (equal-weighted universe)", "K": None, "seeds": None, "SR": mk["SR"], "mu": mk["mu_pct"],
                "sigma": mk["sigma_pct"], "SR_net": mk["SR"], "mu_net": mk["mu_pct"], "sigma_net": mk["sigma_pct"],
                "beta": 1.0, **({"paper_SR": td.PAPER_T2["market"][0], "paper_SR_net": td.PAPER_T2["market"][3]} if m == "us" else {})})
    t = pd.DataFrame(num)
    for _, r in t.iterrows():
        sdtxt = lambda v, s: fmt(v) + (f" ({fmt(s)})" if isinstance(s, float) and np.isfinite(s) else "")  # noqa: E731
        row = {"Model": r["model"], "K": fmt(int(r["K"])) if pd.notna(r["K"]) else "–",
               "SR": sdtxt(r["SR"], r.get("SR_sd")), "μ": fmt(r["mu"]), "σ": fmt(r["sigma"]),
               "SR net": sdtxt(r["SR_net"], r.get("SR_net_sd")), "μ net": fmt(r["mu_net"]), "β": fmt(r["beta"]),
               "Turnover": fmt(r.get("turnover"), 3), "Short": fmt(r.get("short"), 2),
               "Break-even (bp)": fmt(r.get("break_even_bps"), 1)}
        if m == "us":
            row.update({"Paper SR": fmt(r["paper_SR"]), "Paper SR net": fmt(r["paper_SR_net"])})
        disp.append(row)
    return t.set_index(["model", "K"]), pd.DataFrame(disp).set_index("Model")


def t04_t05(out: Path, M: td.Matrix) -> None:
    for m, name, num_ in [("us", "T04_us_table2", 4), ("eu", "T05_europe_table2", 5)]:
        t, disp = table2(M, m)
        cap = (f"Out-of-sample performance, January 1998 - December 2021, {MARKETS[m]}. SR, μ (%) and σ (%) are "
               "annualised from daily returns, gross and net of 5bp per unit turnover and 1bp per unit short "
               "position (the paper's costs); β on the equal-weighted universe; turnover = mean daily ||w_t - "
               "w_t-1||_1 with ||w||_1 = 1; short = mean ||max(-w, 0)||_1; break-even = the per-unit turnover cost "
               "at which the mean net return is zero (shorting cost kept). Mean over seeds 0-4 (sd in parentheses); "
               "PCA + OU is deterministic.")
        notes = ["Hyperparameters: Table 3 (validated on the first US window; Europe uses the US values)."]
        if m == "us":
            notes.append("Paper columns: Epstein, Wang, Choi and Pelger (2025), Table 2, for comparison only.")
        else:
            notes.append("Europe: europe17 top 200, ECU/EUR, holidays filled, positions held on non-traded closes; "
                         "training windows start in 1993 (8 years from 2001).")
        write(out, name, t, disp, f"Table {num_}. Out-of-sample performance, {MARKETS[m]}", cap, notes,
              "runs/_matrix/thesis (headline group): oos_daily.csv of every run")


def tax_cost(run: Path, countries: pd.Series) -> pd.Series:
    """Daily statutory tax on purchases (sum of positive weight changes by country x rate), in
    return units, from the run's target weights."""
    w = pd.read_parquet(run / "weights.parquet")
    w["date"] = pd.to_datetime(w["date"])
    W = w.pivot_table(index="date", columns="sec_id", values="w", aggfunc="sum", fill_value=0.0)
    buys = W.diff().clip(lower=0)
    buys.iloc[0] = W.iloc[0].clip(lower=0)
    cost = pd.Series(0.0, index=W.index)
    ctry = countries.reindex(W.columns)
    for c, sched in TAXES.items():
        cols = ctry.index[ctry == c]
        if len(cols) == 0:
            continue
        b = buys[cols].sum(axis=1)
        rate = pd.Series(0.0, index=W.index)
        for start, r in sched:
            rate[rate.index >= pd.Timestamp(start)] = r
        cost += b * rate
    return cost


def t06_costs(out: Path, M: td.Matrix) -> None:
    ret = pd.read_parquet(td.ROOT / DATA["eu"] / "returns.parquet", columns=["date", "sec_id", "country"])
    countries = ret.sort_values("date").drop_duplicates("sec_id", keep="last").set_index("sec_id")["country"]
    rows = []
    for m in MARKETS:
        for model in MODELS:
            sp = spec(m, model)
            per = []
            for r in M.runs(sp, 30).itertuples():
                d = td.daily(r.run)
                tc, sc = td._costs(r.run)
                row = {"gross_mu": 25200 * d["gross"].mean(), "turnover": d["turnover"].mean(),
                       "short": d["short"].mean(), "turnover_cost": 25200 * tc * d["turnover"].mean(),
                       "short_cost": 25200 * sc * d["short"].mean(), "net_mu": 25200 * d["net"].mean(),
                       "break_even_bps": metrics.break_even_turnover_cost(d["gross"].to_numpy(), d["turnover"].to_numpy(),
                                                                          d["short"].to_numpy(), sc)}
                for c in (10, 20):
                    row[f"SR_net_{c}bp"] = metrics.annualised(d["gross"] - c * 1e-4 * d["turnover"] - sc * d["short"])["SR"]
                if m == "eu":
                    tax = tax_cost(r.run, countries).reindex(d.index).fillna(0.0)
                    row["tax_cost"] = 25200 * tax.mean()
                    row["SR_net_tax"] = metrics.annualised(d["net"] - tax)["SR"]
                per.append(row)
            mu = pd.DataFrame(per).mean()
            rows.append({"market": MARKETS[m], "model": MODELS[model], **mu.to_dict()})
    t = pd.DataFrame(rows).set_index(["market", "model"])
    disp = pd.DataFrame({
        "Gross μ (%)": t["gross_mu"].map(fmt), "Turnover": t["turnover"].map(lambda x: fmt(x, 3)),
        "Short": t["short"].map(fmt), "Turnover cost (%)": t["turnover_cost"].map(fmt),
        "Short cost (%)": t["short_cost"].map(fmt), "Net μ (%)": t["net_mu"].map(fmt),
        "Break-even (bp)": t["break_even_bps"].map(lambda x: fmt(x, 1)),
        "SR net @10bp": t["SR_net_10bp"].map(fmt), "SR net @20bp": t["SR_net_20bp"].map(fmt),
        "Taxes (%)": t["tax_cost"].map(fmt) if "tax_cost" in t else "–",
        "SR net + taxes": t["SR_net_tax"].map(fmt) if "SR_net_tax" in t else "–"})
    disp.index = [f"{a}: {b}" for a, b in t.index]
    disp.index.name = "K = 30"
    write(out, "T06_cost_mechanism", t, disp, "Table 6. Where the gross return goes: turnover, shorting, taxes",
          "K = 30, 1998-2021, mean over seeds. Annual gross mean return, the two cost terms of the paper's model "
          "(5bp x turnover, 1bp x short position, in % a year), the net mean, the break-even turnover cost, and the "
          "net Sharpe ratio at 10bp and 20bp per unit turnover. Europe also: the statutory taxes on purchases "
          "(the floor of the cost argument) and the net Sharpe after them.",
          ["Taxes: UK stamp duty 50bp; French FTT 20bp from 2012-08-01, 30bp from 2017; Italian FTT 12bp from "
           "2013-03-01, 10bp from 2014; on the sum of positive changes of the target weights of names listed in "
           "those countries (other European transfer taxes, spreads, impact and borrow costs are not modelled). "
           "The taxes fall on purchases of the shares themselves: synthetic exposure (CFDs, total return "
           "swaps) is outside UK stamp duty and the French FTT, so this is the floor for a cash-equity "
           "implementation."],
          "runs/_matrix/thesis: oos_daily.csv, weights.parquet; data/europe17/holfill/returns.parquet (country)")


def t07_inference(out: Path, M: td.Matrix) -> None:
    rows = []
    for m in MARKETS:
        att = M.per_seed(spec(m, "attention"), 30)
        for model in ["pca_longconv", "pca_ou"]:
            other = M.per_seed(spec(m, model), 30)
            tests = []
            for s, a in att.items():
                b = other.get(s, next(iter(other.values())))        # OU: one deterministic run
                j = a[["net"]].join(b[["net"]], lsuffix="_a", rsuffix="_b", how="inner")
                tests.append(stats.sharpe_diff(j["net_a"].to_numpy(), j["net_b"].to_numpy()))
            T = pd.DataFrame(tests)
            rows.append({"comparison": f"{MARKETS[m]}: attention − {MODELS[model]}", "diff": T["diff"].mean(),
                         "se": T["se"].mean(), "mdd_5pct": T["mdd_5pct"].mean(), "p_max": T["p"].max(),
                         "n_sig": int((T["p"] < 0.05).sum()), "pairs": len(T), "days": int(T["T"].mean())})
    us, eu = M.per_seed("us_attention", 30), M.per_seed("eu_attention", 30)
    tests = []
    for s in us:
        j = us[s][["net"]].join(eu[s][["net"]], lsuffix="_a", rsuffix="_b", how="inner")
        tests.append(stats.sharpe_diff(j["net_a"].to_numpy(), j["net_b"].to_numpy()))
    T = pd.DataFrame(tests)
    rows.append({"comparison": "Attention: US − Europe (common days)", "diff": T["diff"].mean(), "se": T["se"].mean(),
                 "mdd_5pct": T["mdd_5pct"].mean(), "p_max": T["p"].max(), "n_sig": int((T["p"] < 0.05).sum()),
                 "pairs": len(T), "days": int(T["T"].mean())})
    single = []
    for m in MARKETS:
        for model in MODELS:
            per = [{"SR_net": stats.sharpe(d["net"]), "se": stats.sharpe_se(d["net"].to_numpy())}
                   for d in M.per_seed(spec(m, model), 30).values()]
            P = pd.DataFrame(per)
            single.append({"strategy": f"{MARKETS[m]}: {MODELS[model]}", "SR_net": P["SR_net"].mean(),
                           "se": P["se"].mean(), "min": P["SR_net"].min(), "max": P["SR_net"].max(), "seeds": len(P)})
    t = pd.DataFrame(rows).set_index("comparison")
    disp = pd.DataFrame({"Δ SR net": t["diff"].map(fmt), "HAC se": t["se"].map(fmt),
                         "Smallest resolvable Δ (5%)": t["mdd_5pct"].map(fmt), "Largest p": t["p_max"].map(lambda x: fmt(x, 3)),
                         "Pairs with p < 0.05": [f"{a} of {b}" for a, b in zip(t["n_sig"], t["pairs"])],
                         "Days": t["days"].map(fmt)}, index=t.index)
    write(out, "T07_inference", t, disp, "Table 7. Are the differences in net Sharpe ratios distinguishable from noise?",
          "Paired tests of equal net Sharpe ratios, K = 30, 1998-2021, each attention seed against the benchmark "
          "of the same seed (PCA + OU: its one run): the difference, its HAC standard error (Ledoit-Wolf delta "
          "method, Newey-West kernel), the smallest difference the sample resolves at 5% (1.96 se, Lo 2002), the "
          "largest p-value over the pairs and how many are below 0.05.",
          ["US vs Europe pairs the two markets' seed-s runs on the days both traded."],
          "runs/_matrix/thesis (headline): oos_daily.csv")
    s = pd.DataFrame(single).set_index("strategy")
    disp = pd.DataFrame({"SR net": s["SR_net"].map(fmt), "HAC se": s["se"].map(fmt), "Seed min": s["min"].map(fmt),
                         "Seed max": s["max"].map(fmt), "Seeds": s["seeds"].map(fmt)}, index=s.index)
    write(out, "T07b_sharpe_se", s, disp, "Table 7b. Net Sharpe ratios with standard errors (K = 30)",
          "Mean over seeds of the net Sharpe ratio and of its HAC standard error, and the range over seeds.",
          [], "runs/_matrix/thesis (headline): oos_daily.csv")


def t08_alphas(out: Path, M: td.Matrix) -> None:
    fac = {"us": td.us_factors().join(td.str_factor(td.ROOT / DATA["us"]), how="inner"),
           "eu": td.europe_factors().join(td.str_factor(td.ROOT / DATA["eu"]), how="inner")}
    sets = {"FF5 + Mom": ["mktrf", "smb", "hml", "rmw", "cma", "mom"], "FF5 + Mom + STR": ["mktrf", "smb", "hml", "rmw", "cma", "mom", "str"]}
    rows = []
    for m in MARKETS:
        for model in MODELS:
            d = M.combination(spec(m, model), 30)
            for label, cols in sets.items():
                j = d[["net"]].join(fac[m][cols], how="inner").dropna()
                r = stats.ols_nw(j["net"].to_numpy(), j[cols].to_numpy(), cols)
                rows.append({"strategy": f"{MARKETS[m]}: {MODELS[model]}", "factors": label, **r})
    t = pd.DataFrame(rows).set_index(["strategy", "factors"])
    names = ["mktrf", "smb", "hml", "rmw", "cma", "mom", "str"]
    disp = pd.DataFrame({"α (%/yr)": t["alpha_ann_pct"].map(fmt), "t(α)": t["t_alpha"].map(fmt),
                         **{n.upper() if n != "mktrf" else "MKT": t.get(f"b_{n}", pd.Series(np.nan, index=t.index)).map(fmt)
                            for n in names}, "R²": t["r2"].map(fmt)})
    disp.index = [f"{a} | {b}" for a, b in t.index]
    disp.index.name = "Strategy | factors"
    write(out, "T08_alphas", t, disp, "Table 8. Net returns on the factor models",
          "Daily net returns of the K = 30 strategies (equal-weighted combination of the seeds), 1998-2021, "
          "regressed on Fama-French five factors and momentum, with and without a short-term reversal factor; "
          "annualised alpha and its Newey-West t-statistic, loadings, R².",
          ["US: WRDS ff.fivefactors_daily (UMD). Europe: Ken French's Europe five factors and momentum (USD) in "
           "ECU/EUR, market in excess of the Bundesbank rate. STR: own daily factor per universe (bottom minus top "
           "30% of last month's return, equal-weighted), as Ken French has none for Europe. All t-statistics in "
           "the CSV."], "runs/_matrix/thesis; data/us/shared/raw/ff5_daily.parquet; Europe zips in "
                       "data/europe/private/raw/external; features.parquet (ST_Rev)")


def t09_subperiods(out: Path, M: td.Matrix) -> None:
    rows = []
    for m in MARKETS:
        for model in MODELS:
            sp = spec(m, model)
            row = {"strategy": f"{MARKETS[m]}: {MODELS[model]}"}
            for lab, a, b in SUBPERIODS:
                mu, _ = mean_sd(per_seed_stats(M, sp, 30, period=(a, b)))
                row[f"SR {lab}"], row[f"SR net {lab}"] = mu["SR"], mu["SR_net"]
            ext = f"{m}_attention_2022_2025"
            if model == "attention" and ext in set(M.summary["spec"]):
                mu, _ = mean_sd(per_seed_stats(M, ext, 30))
                row["SR 2022-2025"], row["SR net 2022-2025"] = mu["SR"], mu["SR_net"]
            rows.append(row)
    t = pd.DataFrame(rows).set_index("strategy")
    write(out, "T09_subperiods", t, t.map(fmt), "Table 9. Subperiods and the 2022-2025 holdout",
          "Gross and net Sharpe ratios by subperiod, K = 30, mean over seeds. 2022-2025: the attention model "
          "run past the paper's sample with the same hyperparameters and annual refits; no choice of the thesis "
          "used these years.", ["– : not run (the benchmarks stop in 2021)."],
          "runs/_matrix/thesis (headline, extension)")


def t10_ablation(out: Path, M: td.Matrix) -> None:
    rows = []
    for g in ["none", *GROUPS]:
        row = {"dropped": "none (baseline)" if g == "none" else GROUPS[g]}
        for m in MARKETS:
            sp = spec(m, "attention") if g == "none" else f"{m}_drop_{g}"
            ps = per_seed_stats(M, sp, 30, seeds=[0, 1, 2])
            mu, sd = mean_sd(ps)
            for k in ["SR", "mu", "sigma", "SR_net", "mu_net", "beta", "turnover"]:
                row[f"{MARKETS[m]} {k}"] = mu[k]
            row[f"{MARKETS[m]} SR_sd"] = sd["SR"]
        p = td.PAPER_T3[g]
        row["paper SR"], row["paper SR net"] = p[0], p[4]
        rows.append(row)
    t = pd.DataFrame(rows).set_index("dropped")
    disp = pd.DataFrame(index=t.index)
    for M_ in ["US", "Europe"]:
        disp[f"{M_} SR"] = [f"{fmt(a)} ({fmt(b)})" for a, b in zip(t[f"{M_} SR"], t[f"{M_} SR_sd"])]
        disp[f"{M_} SR net"] = t[f"{M_} SR_net"].map(fmt)
        disp[f"{M_} μ net"] = t[f"{M_} mu_net"].map(fmt)
        disp[f"{M_} β"] = t[f"{M_} beta"].map(fmt)
        disp[f"{M_} turnover"] = t[f"{M_} turnover"].map(lambda x: fmt(x, 3))
    disp["Paper SR"], disp["Paper SR net"] = t["paper SR"].map(fmt), t["paper SR net"].map(fmt)
    disp.index.name = "Dropped group"
    write(out, "T10_ablation", t, disp, "Table 10. Which characteristics drive the result",
          "Attention factors, K = 30, 1998-2021, each characteristic group removed from estimation and evaluation "
          "(its characteristics and their cross-sectional medians), same hyperparameters; mean over seeds 0-2 "
          "(sd of the gross SR in parentheses). The baseline is the headline model on the same seeds.",
          ["Paper columns: Epstein et al. (2025), Table 3 (US)."], "runs/_matrix/thesis (ablation, headline)")


def t11_robustness(out: Path, M: td.Matrix) -> None:
    rows_def = [("US", "Attention, headline", "us_attention"), ("US", "Attention, lag 1", "us_lag1"),
                ("US", "Attention, raw input", "us_raw_input"), ("US", "Attention, top 200", "us_top200"),
                ("US", "PCA + LongConv, headline", "us_pca_longconv"), ("US", "PCA + LongConv, lag 1", "us_pca_longconv_lag1"),
                ("Europe", "Attention, headline", "eu_attention"), ("Europe", "Attention, lag 1", "eu_lag1"),
                ("Europe", "Attention, raw input", "eu_raw_input"), ("Europe", "Attention, holidays unfilled", "eu_unfilled"),
                ("Europe", "Attention, europe12", "eu_europe12"),
                ("Europe", "PCA + LongConv, headline", "eu_pca_longconv"), ("Europe", "PCA + LongConv, lag 1", "eu_pca_longconv_lag1")]
    rows = []
    for mk, lab, sp in rows_def:
        ps = per_seed_stats(M, sp, 30, seeds=[0, 1, 2])
        mu, sd = mean_sd(ps)
        rows.append({"row": f"{mk}: {lab}", **mu.drop("seed").to_dict(), "SR_net_sd": sd["SR_net"]})
    t = pd.DataFrame(rows).set_index("row")
    base = {"US: Attention": t.loc["US: Attention, headline", "SR_net"], "Europe: Attention": t.loc["Europe: Attention, headline", "SR_net"],
            "US: PCA + LongConv": t.loc["US: PCA + LongConv, headline", "SR_net"],
            "Europe: PCA + LongConv": t.loc["Europe: PCA + LongConv, headline", "SR_net"]}
    t["delta_SR_net"] = [r - base[i.split(",")[0]] for i, r in zip(t.index, t["SR_net"])]
    disp = pd.DataFrame({"SR": t["SR"].map(fmt), "SR net": [f"{fmt(a)} ({fmt(b)})" for a, b in zip(t["SR_net"], t["SR_net_sd"])],
                         "Δ SR net": t["delta_SR_net"].map(fmt), "Turnover": t["turnover"].map(lambda x: fmt(x, 3)),
                         "Break-even (bp)": t["break_even_bps"].map(lambda x: fmt(x, 1)), "β": t["beta"].map(fmt)}, index=t.index)
    disp.index.name = "K = 30"
    write(out, "T11_robustness", t, disp, "Table 11. Robustness: one change at a time",
          "K = 30, 1998-2021, mean over seeds 0-2 (sd of the net SR in parentheses), Δ against the headline row "
          "of the same model and market on the same seeds. Lag 1: weights traded one close later. Raw input: the "
          "policy sees the 30 daily residuals (the paper's wording) instead of their cumulative path. Rows that "
          "change how the model trades (lag 1, raw input) use hyperparameters re-validated under that change; "
          "rows that change only the data keep the headline values (Table 3).",
          ["Top 200 (US): members up to cap rank 200 of the 500 build (ranks and medians of the 500 build). "
           "Holidays unfilled: Compustat Global as delivered (a holiday drops a country's names for 30 days before "
           "~2010). europe12: without Dublin, Lisbon, Vienna, Luxembourg and Athens."],
          "runs/_matrix/thesis (robustness, headline)")


def t12_years(out: Path, M: td.Matrix) -> None:
    cols = {}
    for m in MARKETS:
        for model in MODELS:
            ys = []
            for d in M.per_seed(spec(m, model), 30).values():
                ys.append(d.groupby(d.index.year)["net"].apply(lambda x: metrics.annualised(x)["SR"]))
            cols[f"{MARKETS[m]} {MODELS[model]}"] = pd.concat(ys, axis=1).mean(axis=1)
        ext = f"{m}_attention_2022_2025"
        if ext in set(M.summary["spec"]):
            ys = [d.groupby(d.index.year)["net"].apply(lambda x: metrics.annualised(x)["SR"])
                  for d in M.per_seed(ext, 30).values()]
            e = pd.concat(ys, axis=1).mean(axis=1)
            cols[f"{MARKETS[m]} {MODELS['attention']}"] = pd.concat([cols[f"{MARKETS[m]} {MODELS['attention']}"], e])
    t = pd.DataFrame(cols)
    t.index.name = "Year"
    write(out, "T12_net_sharpe_by_year", t, t.map(fmt), "Table A1. Net Sharpe ratio by year (K = 30)",
          "Annualised net Sharpe ratio within each calendar year, mean over seeds. 2022-2025: attention only "
          "(the holdout runs).", [], "runs/_matrix/thesis (headline, extension)")


def t13_searches(out: Path, matrix_dir: Path, main: Path) -> None:
    mets = json.loads((main / "metrics.json").read_text())
    rows = [{"name": r["name"], "val_net_SR": r["val_net_SR"], "se": r["se"], "gross_SR": r["gross_SR"],
             "turnover": r["turnover"], "values": json.dumps(r["overrides"])} for r in mets["finalists"]]
    t = pd.DataFrame(rows).set_index("name")
    disp = t.assign(val_net_SR=t["val_net_SR"].map(fmt), se=t["se"].map(fmt), gross_SR=t["gross_SR"].map(fmt),
                    turnover=t["turnover"].map(fmt))
    disp.columns = ["Validation net SR", "se", "Validation gross SR", "Turnover", "Values"]
    disp.index.name = "Finalist"
    write(out, "T13_search_finalists", t, disp, "Table A2. The attention search: finalists",
          f"US validation search ({mets.get('n_search_points')} Sobol points + base): the five best points and the "
          "base config on seeds 0-2, validation 1996-1997. The winner is the headline configuration.",
          ["sobol-009 is the configuration Kevin Pesenti selected earlier from the same space."],
          str(main.relative_to(td.ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "thesis")
    ap.add_argument("--matrix", type=Path, default=ROOT / "runs" / "_matrix" / "thesis")
    ap.add_argument("--main-search", type=Path, default=ROOT / "runs" / "20261010T110947Z_search_K30_val1996-1997")
    ap.add_argument("--only", nargs="*", help="table function names, e.g. t04_t05")
    args = ap.parse_args()
    M = td.Matrix(args.matrix, ROOT / "runs")
    jobs = {"t01_sample": lambda: t01_sample(args.out), "t02_characteristics": lambda: t02_characteristics(args.out),
            "t03_hyperparameters": lambda: t03_hyperparameters(args.out, args.matrix, args.main_search),
            "t04_t05": lambda: t04_t05(args.out, M), "t06_costs": lambda: t06_costs(args.out, M),
            "t07_inference": lambda: t07_inference(args.out, M), "t08_alphas": lambda: t08_alphas(args.out, M),
            "t09_subperiods": lambda: t09_subperiods(args.out, M), "t10_ablation": lambda: t10_ablation(args.out, M),
            "t11_robustness": lambda: t11_robustness(args.out, M), "t12_years": lambda: t12_years(args.out, M),
            "t13_searches": lambda: t13_searches(args.out, args.matrix, args.main_search)}
    for name, f in jobs.items():
        if not args.only or name in args.only:
            f()
    idx = args.out / "tables" / "index.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() and args.only else []
    keep = {e["name"]: e for e in old}
    keep.update({e["name"]: e for e in INDEX})
    idx.write_text(json.dumps(sorted(keep.values(), key=lambda e: e["name"]), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(INDEX)} tables -> {args.out / 'tables'}")


if __name__ == "__main__":
    main()
