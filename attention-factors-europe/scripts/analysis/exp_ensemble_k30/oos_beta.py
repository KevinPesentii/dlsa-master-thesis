"""Market beta of the K = 30 ensemble books (oos_daily_books.parquet), 2026-11-02. Run from the repo root.

beta = OLS slope of the book's daily gross return on the equal-weighted market return mkt_ew, as
afe.evaluation.metrics.beta (the runs' beta_ew_market). mkt_ew is read from a run's oos_daily.csv (same
dates for every run). Checked first: the single runs' betas equal summary.csv. Also the net-return beta.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
import sys
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / "src"))
from afe.evaluation import metrics

OUT = ROOT / "runs/_matrix/exp_ensemble_k30"
TC, SC = 5e-4, 1e-4
summ = pd.read_csv(OUT / "summary.csv")
mkt = pd.read_csv(ROOT / "runs" / Path(summ.run_dir.iloc[0]).name / "oos_daily.csv", parse_dates=["date"]).set_index("date")["mkt_ew"]
d = pd.read_parquet(OUT / "oos_daily_books.parquet")
d["net"] = d.gross - TC * d.turnover - SC * d.short
rows = []
for (w, kind, book), g in d.groupby(["w", "kind", "book"]):
    g = g.set_index("date").sort_index()
    assert g.index.equals(mkt.index)
    rows.append({"w": w, "kind": kind, "book": book,
                 "beta": metrics.beta(g.gross.to_numpy(), mkt.to_numpy()),
                 "beta_net": metrics.beta(g.net.to_numpy(), mkt.to_numpy()),
                 "corr": np.corrcoef(g.gross, mkt)[0, 1]})
b = pd.DataFrame(rows)
b.to_csv(OUT / "oos_beta.csv", index=False)

# check: single runs against their summary.csv (new runs here; sobol-036 runs in exp_fixedwin_full)
full = pd.read_csv(ROOT / "runs/_matrix/exp_fixedwin_full/summary.csv")
ref = pd.concat([summ.assign(cand=summ.spec.str.extract(r"(sobol\d+)")[0]),
                 full[(full.K == 30)].assign(cand="sobol036")])
ref["w"] = np.where(ref.spec.str.contains("fixedwin"), "fixed", "roll")
s = b[b.kind == "single"].copy()
s["cand"] = s.book.str.extract(r"single:(sobol-\d+)")[0].str.replace("-", "")
s["seed"] = s.book.str.extract(r":s(\d)$")[0].astype(int)
m = s.merge(ref[["w", "cand", "seed", "beta"]], on=["w", "cand", "seed"], suffixes=("", "_summary"))
assert len(m) == 50, len(m)
print(f"single-run check: 50 runs, max |beta - summary.csv| = {np.abs(m.beta - m.beta_summary).max():.2e}")

groups = {"one sobol-036 run": lambda x: x.startswith("single:sobol-036:"), "seeds: sobol-036 x 5": lambda x: x == "seeds:sobol-036",
          "configs: top 5, same seed": lambda x: re.fullmatch(r"configs:s\d", x) is not None,
          "configs: top 5, different seeds": lambda x: x.startswith("configs:latin"), "all 25": lambda x: x == "all",
          "one run, any candidate": lambda x: x.startswith("single:")}
out = []
for w in ("roll", "fixed"):
    for kind in ("avg_l1", "avg"):
        for name, f in groups.items():
            k = "single" if name.startswith("one ") else kind
            sel = b[(b.w == w) & (b.kind == k) & b.book.map(f)]
            out.append({"w": w, "kind": kind, "group": name, "n": len(sel), "beta": sel.beta.mean(), "beta_min": sel.beta.min(),
                        "beta_max": sel.beta.max(), "beta_net": sel.beta_net.mean(), "corr": sel["corr"].mean()})
o = pd.DataFrame(out); o.to_csv(OUT / "oos_beta_groups.csv", index=False)
pd.set_option("display.width", 200)
print(o.round(3).to_string(index=False))
c = b[(b.kind == "single")].copy(); c["cand"] = c.book.str.extract(r"single:(sobol-\d+)")[0]
print(c.groupby(["w", "cand"]).beta.agg(["mean", "min", "max"]).round(3).to_string())
