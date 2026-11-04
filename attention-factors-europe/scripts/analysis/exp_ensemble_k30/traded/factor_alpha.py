"""Does the ensemble gain survive factor adjustment? 2026-11-04. Run from the repo root.

Every book of oos_daily_books.parquet regressed daily on three models: the equal-weighted universe return mkt_ew (the
runs' beta), FF5 + momentum, and both. Alpha, Newey-West t (5 lags), R2, and the information ratio alpha / residual vol,
gross and net. IR net is the net Sharpe of the book with its factor exposure hedged free of charge, so it flatters
every book a little; compare books, not levels. Net = gross - 5bp turnover - 1bp short.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from _common import ROOT, BASE, OUT, ANN, TC, SC

OUT.mkdir(exist_ok=True)
summ = pd.read_csv(BASE / "summary.csv")
daily = pd.read_csv(ROOT / "runs" / Path(summ.run_dir.iloc[0]).name / "oos_daily.csv", parse_dates=["date"]).set_index("date")
ff = pd.read_parquet(ROOT / "data/us/shared/raw/ff5_daily.parquet").set_index("date")
FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
F = ff[FF6].astype("float64").reindex(daily.index)   # WRDS decimals
assert F.notna().all().all(); daily = daily.astype({"mkt_ew": "float64", "rf": "float64"})
F["ew"] = daily["mkt_ew"] - daily["rf"]
MODELS = {"EW": ["ew"], "FF6": FF6, "FF6+EW": FF6 + ["ew"]}
d = pd.read_parquet(BASE / "oos_daily_books.parquet")
d["net"] = d.gross - TC * d.turnover - SC * d.short


def ols_nw(y, X, L=5):
    X = np.column_stack([np.ones(len(y)), X])
    XXi = np.linalg.inv(X.T @ X); b = XXi @ X.T @ y; e = y - X @ b
    u = X * e[:, None]; S = u.T @ u
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l]; S += (1 - l / (L + 1)) * (G + G.T)
    return b, np.sqrt(np.diag(XXi @ S @ XXi)), e


rows = []
for (w, kind, book), g in d.groupby(["w", "kind", "book"]):
    g = g.set_index("date").sort_index(); assert g.index.equals(F.index)
    base = {"w": w, "kind": kind, "book": book, "gross_SR": g.gross.mean() / g.gross.std(ddof=1) * np.sqrt(ANN),
            "net_SR": g.net.mean() / g.net.std(ddof=1) * np.sqrt(ANN), "sigma_pct": 100 * g.gross.std(ddof=1) * np.sqrt(ANN),
            "costs_pct": 100 * ANN * (g.gross - g.net).mean()}
    for name, cols in MODELS.items():
        X = F[cols].to_numpy()
        bg, sg, eg = ols_nw(g.gross.to_numpy(), X); bn, sn, en = ols_nw(g.net.to_numpy(), X)
        sd_e = eg.std(ddof=1) * np.sqrt(ANN)
        rows.append({**base, "model": name, "alpha_pct": 100 * ANN * bg[0], "t_alpha": bg[0] / sg[0], "alpha_net_pct": 100 * ANN * bn[0],
                     "t_alpha_net": bn[0] / sn[0], "r2": 1 - eg.var(ddof=1) / g.gross.var(ddof=1), "resid_vol_pct": 100 * sd_e,
                     "IR": ANN * bg[0] / sd_e, "IR_net": ANN * bn[0] / (en.std(ddof=1) * np.sqrt(ANN)),
                     "costs_over_resid": base["costs_pct"] / (100 * sd_e), **{f"b_{c}": v for c, v in zip(cols, bg[1:])}})
df = pd.DataFrame(rows); df.to_csv(OUT / "factor_alpha_books.csv", index=False)

groups = {"one sobol-036 run": lambda x: x.startswith("single:sobol-036:"), "one sobol-009 run": lambda x: x.startswith("single:sobol-009:"),
          "one run, any of 25": lambda x: x.startswith("single:"), "seeds: sobol-036 x 5": lambda x: x == "seeds:sobol-036",
          "configs: top 5": lambda x: re.fullmatch(r"configs:s\d", x) is not None, "all 25": lambda x: x == "all"}
out = []
for w in ("roll", "fixed"):
    for kind in ("avg_l1", "avg"):
        for mod in MODELS:
            for name, f in groups.items():
                k = "single" if name.startswith("one ") else kind
                s = df[(df.w == w) & (df.kind == k) & (df.model == mod) & df.book.map(f)]
                num = s.drop(columns=["w", "kind", "book", "model"]).mean(numeric_only=True)
                out.append({"w": w, "kind": kind, "model": mod, "group": name, "n": len(s), **num.to_dict()})
o = pd.DataFrame(out); o.to_csv(OUT / "factor_alpha_groups.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
keep = ["w", "model", "group", "n", "gross_SR", "net_SR", "sigma_pct", "alpha_pct", "t_alpha", "alpha_net_pct", "t_alpha_net", "r2",
        "resid_vol_pct", "IR", "IR_net", "costs_over_resid", "b_mktrf", "b_smb", "b_hml", "b_rmw", "b_cma", "b_umd", "b_ew"]
print(o[o.kind == "avg_l1"][keep].round(3).to_string(index=False))
