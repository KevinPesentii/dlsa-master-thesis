"""The K = 30 books without depositary receipts, 2026-11-04. Run from the repo root.

The saved target weights with every receipt set to zero, booked as daily_book does (5bp on |dw|, 1bp on shorts). Nothing
is retrained: the models still saw receipts in training, used them in their factors and hedges, and the slots are not
refilled with the next largest names, so this measures what receipts contribute to the books that exist, not what a
receipt-free universe would earn. Variants, each book at exposure 1 before the cut:
  full               the book as it is; must equal oos_books.csv
  no_receipts        receipts removed, rescaled to exposure 1 every day
  no_receipts_cut    receipts removed, not rescaled, so exposure falls
  receipts_only      only the receipts, rescaled to exposure 1 every day
Per period: Sharpe, costs, turnover, break-even, EW beta, and hedged net = net Sharpe with FF5 + momentum + EW market
exposure hedged free of charge. Difference no_receipts - full: month-block bootstrap of return paths, training noise excluded.
"""
import argparse, sys
import numpy as np, pandas as pd
from _common import ROOT, OUT, ANN, TC, SC, SAMPLES, load, listing_types
sys.path.insert(0, str(ROOT / "scripts"))
from afe.evaluation import metrics
import ensemble_validation as ev

ap = argparse.ArgumentParser(); ap.add_argument("--w", default="roll"); ap.add_argument("--draws", type=int, default=1000); a = ap.parse_args()
D = load(a.w); W, R, dates, keys, ids, p, t_idx = D["W"], D["R"].astype(np.float64), D["dates"], D["keys"], D["ids"], D["p"], D["t_idx"]
T, n = R.shape
names, TYPE = listing_types(ids); keep = TYPE != 1
print("receipts in the pool:", int((~keep).sum()), "of", n, flush=True)
ff = pd.read_parquet(ROOT / "data/us/shared/raw/ff5_daily.parquet").set_index("date")
FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
F = ff[FF6].astype("float64").reindex(dates); assert F.notna().all().all()
ew = np.asarray(p.mkt_ew[t_idx], dtype=np.float64); F["ew"] = ew - p.rf.numpy()[t_idx].astype(np.float64)
masks = {}
for s, (lo, hi) in SAMPLES.items():
    m = np.ones(T, bool)
    if lo: m &= dates >= pd.Timestamp(f"{lo}-01-01")
    if hi: m &= dates <= pd.Timestamp(f"{hi}-12-31")
    masks[s] = m
idx = {k: i for i, k in enumerate(keys)}


def scale(w):
    l1 = np.abs(w).sum(1, keepdims=True)
    return np.divide(w, l1, out=np.zeros_like(w), where=l1 > 0)


def book(w):
    return {"gross": (w * R).sum(1), "turn": np.abs(np.diff(w, axis=0, prepend=np.zeros((1, n)))).sum(1), "short": np.clip(-w, 0, None).sum(1),
            "expo": np.abs(w).sum(1), "net_expo": w.sum(1)}


def sr(x): return x.mean() / x.std(ddof=1) * np.sqrt(ANN)


def stats(b, m):
    g, t_, sh = b["gross"][m], b["turn"][m], b["short"][m]; net = g - TC * t_ - SC * sh
    X = np.column_stack([np.ones(m.sum()), F.loc[m, FF6 + ["ew"]].to_numpy()]); c = np.linalg.lstsq(X, net, rcond=None)[0]; e = net - X @ c
    return {"gross_SR": sr(g), "net_SR": sr(net), "mu_pct": 100 * ANN * g.mean(), "mu_net_pct": 100 * ANN * net.mean(), "sigma_pct": 100 * g.std(ddof=1) * np.sqrt(ANN),
            "costs_pct": 100 * ANN * (g - net).mean(), "costs_over_vol": (g - net).mean() * ANN / (g.std(ddof=1) * np.sqrt(ANN)), "turnover": t_.mean(), "short": sh.mean(),
            "BE_bp": metrics.break_even_turnover_cost(g, t_, sh, SC), "beta_ew": np.cov(g, ew[m])[0, 1] / ew[m].var(ddof=1), "hedged_net": ANN * c[0] / (e.std(ddof=1) * np.sqrt(ANN)),
            "alpha_net_pct": 100 * ANN * c[0], "expo": b["expo"][m].mean(), "net_expo": b["net_expo"][m].mean()}


books = {f"single:{c}:s{s}": [(c, s)] for c, s in keys} | ev.ensembles(D["cands"], D["seeds"])
rows, kept = [], {}
for name, members in books.items():
    mean = sum(W[idx[k]].astype(np.float64) for k in members) / len(members)
    full = mean if len(members) == 1 else scale(mean)
    var = {"full": book(full), "no_receipts": book(scale(mean * keep)), "no_receipts_cut": book(full * keep), "receipts_only": book(scale(mean * ~keep))}
    for v, b in var.items():
        for s, m in masks.items():
            rows.append({"w": a.w, "book": name, "models": len(members), "variant": v, "period": s, **stats(b, m)})
    if name in ("all", "seeds:sobol-036"):
        kept[name] = {v: var[v]["gross"] - TC * var[v]["turn"] - SC * var[v]["short"] for v in ("full", "no_receipts")}
    print(name, f"{rows[-12]['net_SR']:.3f}" if len(rows) >= 12 else "", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / f"no_receipts_books_{a.w}.csv", index=False)

ref = pd.read_csv(ROOT / "runs/_matrix/exp_ensemble_k30/oos_books.csv"); ref = ref[ref.w == a.w]
ref = ref[((ref.kind == "single") | (ref.kind == "avg_l1"))].set_index("book")["net_SR"]
mine = df[(df.variant == "full") & (df.period == "1998-2021")].set_index("book")["net_SR"]
err = (mine - ref.reindex(mine.index)).abs(); assert err.notna().all() and err.max() < 1e-4, err.max()
print(f"full books = oos_books.csv on {len(err)} books, max |net SR difference| {err.max():.1e}", flush=True)

GROUPS = {"one sobol-036 run": lambda x: x.startswith("single:sobol-036:"), "mean of 25 runs": lambda x: x.startswith("single:"), "five seeds": lambda x: x == "seeds:sobol-036",
          "top 5 configs": lambda x: x.startswith("configs:s"), "all 25": lambda x: x == "all"}
out = []
for s in SAMPLES:
    for v in ("full", "no_receipts", "no_receipts_cut", "receipts_only"):
        for g, f in GROUPS.items():
            x = df[(df.period == s) & (df.variant == v) & df.book.map(f)]
            out.append({"w": a.w, "period": s, "variant": v, "group": g, "n": len(x), **x.drop(columns=["w", "book", "models", "variant", "period"]).mean().to_dict()})
o = pd.DataFrame(out); o.to_csv(OUT / f"no_receipts_groups_{a.w}.csv", index=False)

rng = np.random.default_rng(0); bs = []
for name, v in kept.items():
    for s, m in masks.items():
        x, y = v["full"][m], v["no_receipts"][m]; code = pd.factorize(dates[m].to_period("M"))[0]; blocks = [np.flatnonzero(code == c) for c in range(code.max() + 1)]
        diff = np.array([sr(y[i]) - sr(x[i]) for i in (np.concatenate([blocks[c] for c in rng.integers(0, len(blocks), len(blocks))]) for _ in range(a.draws))])
        bs.append({"w": a.w, "book": name, "period": s, "diff": sr(y) - sr(x), "se": diff.std(ddof=1), "p_no_receipts_better": (diff > 0).mean()})
bs = pd.DataFrame(bs); bs.to_csv(OUT / f"no_receipts_boot_{a.w}.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
print(o.round(3).to_string(index=False)); print(bs.round(3).to_string(index=False))
