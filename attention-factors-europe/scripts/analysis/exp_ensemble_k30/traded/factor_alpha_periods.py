"""factor_alpha.py by sub-period: does the ensemble gain survive factor adjustment after 2010? 2026-11-04. Run from the repo root.

avg_l1 books of oos_daily_books.parquet, FF5 + momentum + equal-weighted market, betas estimated inside each period.
IR net = the net Sharpe of the book with its factor exposure hedged free of charge; EW only = the market alone.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from _common import ROOT, BASE, OUT, ANN, TC, SC

summ = pd.read_csv(BASE / "summary.csv")
daily = pd.read_csv(ROOT / "runs" / Path(summ.run_dir.iloc[0]).name / "oos_daily.csv", parse_dates=["date"]).set_index("date")
ff = pd.read_parquet(ROOT / "data/us/shared/raw/ff5_daily.parquet").set_index("date")
FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
F = ff[FF6].astype("float64").reindex(daily.index); F["ew"] = daily["mkt_ew"].astype("float64") - daily["rf"].astype("float64")
d = pd.read_parquet(BASE / "oos_daily_books.parquet"); d["net"] = d.gross - TC * d.turnover - SC * d.short
GROUPS = {"one sobol-036 run": lambda x: x.startswith("single:sobol-036:"), "one run, any of 25": lambda x: x.startswith("single:"),
          "seeds: sobol-036 x 5": lambda x: x == "seeds:sobol-036", "configs: top 5": lambda x: re.fullmatch(r"configs:s\d", x) is not None, "all 25": lambda x: x == "all"}
PERIODS = {"1998-2021": (None, None), "1998-2009": (None, "2009"), "2010-2021": ("2010", None)}


def fit(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    return ANN * b[0], e.std(ddof=1) * np.sqrt(ANN), b[1:]


rows = []
for (w, kind, book), g in d[d.kind.isin(["avg_l1", "single"])].groupby(["w", "kind", "book"]):
    g = g.set_index("date").sort_index()
    for per, (lo, hi) in PERIODS.items():
        x, f = g.loc[lo:hi], F.loc[lo:hi]
        ann = lambda s: ANN * s.mean()
        a6, s6, b6 = fit(x.net.to_numpy(), f[FF6 + ["ew"]].to_numpy()); a1, s1, b1 = fit(x.net.to_numpy(), f[["ew"]].to_numpy())
        rows.append({"w": w, "book": book, "period": per, "gross_pct": 100 * ann(x.gross), "net_pct": 100 * ann(x.net), "vol_pct": 100 * x.net.std(ddof=1) * np.sqrt(ANN),
                     "net_SR": ann(x.net) / (x.net.std(ddof=1) * np.sqrt(ANN)), "beta_ew": b1[0], "alpha_net_ew_pct": 100 * a1, "IR_net_ew": a1 / s1,
                     "alpha_net_all_pct": 100 * a6, "IR_net_all": a6 / s6, "resid_vol_pct": 100 * s6})
r = pd.DataFrame(rows); r.to_csv(OUT / "factor_alpha_periods_books.csv", index=False)
out = []
for w in ("roll", "fixed"):
    for per in PERIODS:
        for name, f_ in GROUPS.items():
            s = r[(r.w == w) & (r.period == per) & r.book.map(f_)]
            out.append({"w": w, "period": per, "group": name, "n": len(s), **s.drop(columns=["w", "book", "period"]).mean().to_dict()})
o = pd.DataFrame(out); o.to_csv(OUT / "factor_alpha_periods.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(o.round(3).to_string(index=False))
