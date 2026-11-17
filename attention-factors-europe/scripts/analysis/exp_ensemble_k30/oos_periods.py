"""Full-period and 5-year statistics of the K = 30 ensemble books (oos_daily_books.parquet, from oos_ensemble.py).

Books: one sobol-036 run (mean over its 5 seeds), seeds:sobol-036, configs (top 5 at one seed, mean over
configs:s0..4), the same with every candidate on a different seed (mean over configs:latin0..4), all 25, and one run
of any candidate (mean over the 25). A "mean over" row averages each statistic across its books, as the validation
rule scores configs. avg_l1 = gross exposure 1 every day (the decision book); avg = plain mean. Costs 5bp turnover,
1bp short. Periods: 1998-2002, 2003-07, 2008-12, 2013-17 and 2018-21 (4 years). Run from the repo root.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path.cwd() / "runs/_matrix/exp_ensemble_k30"
ANN, TC, SC = 252, 5e-4, 1e-4
PERIODS = {"1998-2021": ("1998", "2021"), "1998-2002": ("1998", "2002"), "2003-2007": ("2003", "2007"),
           "2008-2012": ("2008", "2012"), "2013-2017": ("2013", "2017"), "2018-2021": ("2018", "2021")}
GROUPS = {"one sobol-036 run": lambda b: b.startswith("single:sobol-036:"),
          "seeds: sobol-036 x 5": lambda b: b == "seeds:sobol-036",
          "configs: top 5, same seed": lambda b: re.fullmatch(r"configs:s\d", b),
          "configs: top 5, different seeds": lambda b: b.startswith("configs:latin"),
          "all 25": lambda b: b == "all",
          "one run, any candidate": lambda b: b.startswith("single:")}

def stats(d: pd.DataFrame) -> dict:
    g, net = d["gross"], d["gross"] - TC * d["turnover"] - SC * d["short"]
    sd = g.std(ddof=1) * np.sqrt(ANN)
    wealth = (1 + net).cumprod()
    return {"mu_pct": 100 * ANN * g.mean(), "sigma_pct": 100 * sd, "SR": ANN * g.mean() / sd,
            "costs_pct": 100 * ANN * (g - net).mean(), "mu_net_pct": 100 * ANN * net.mean(),
            "SR_net": net.mean() / net.std(ddof=1) * np.sqrt(ANN), "turnover": d["turnover"].mean(),
            "BE_bp": 1e4 * (g.mean() - SC * d["short"].mean()) / d["turnover"].mean(),
            "maxDD_net_pct": 100 * float((1 - wealth / wealth.cummax()).max())}

daily = pd.read_parquet(OUT / "oos_daily_books.parquet").set_index("date")
rows = []
for (w, kind, book), d in daily.groupby(["w", "kind", "book"]):
    for per, (a, b) in PERIODS.items():
        rows.append({"w": w, "kind": kind, "book": book, "period": per, "days": len(d.loc[a:b]), **stats(d.loc[a:b])})
books = pd.DataFrame(rows)
out = []
for kind in ("avg_l1", "avg"):
    for name, member in GROUPS.items():
        want = "single" if name.startswith("one ") else kind
        sel = books[(books.kind == want) & books.book.map(lambda x: bool(member(x)))]
        g = sel.groupby(["w", "period"]).mean(numeric_only=True).reset_index()
        g["n_books"] = sel.groupby(["w", "period"]).size().to_numpy()
        out.append(g.assign(kind=kind, group=name))
P = pd.concat(out)
P = P[["w", "kind", "group", "n_books", "period", "days", "mu_pct", "sigma_pct", "SR", "costs_pct", "mu_net_pct",
       "SR_net", "turnover", "BE_bp", "maxDD_net_pct"]]
P.to_csv(OUT / "oos_periods.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
order = {g: i for i, g in enumerate(GROUPS)}
for kind in ("avg_l1", "avg"):
    q = P[P.kind == kind].assign(o=lambda x: x.group.map(order))
    print(f"\n== {kind}: 1998-2021")
    print(q[q.period == "1998-2021"].sort_values(["w", "o"]).drop(columns=["o", "kind", "period"]).round(2).to_string(index=False))
    for col in ("mu_net_pct", "sigma_pct", "SR_net", "mu_pct", "SR", "costs_pct", "maxDD_net_pct"):
        print(f"\n-- {kind}: {col} by period")
        t = q[q.period != "1998-2021"].pivot_table(index=["w", "o", "group"], columns="period", values=col)
        print(t.droplevel("o").round(2).to_string())
