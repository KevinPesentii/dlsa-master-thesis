"""Do the positions the models agree on earn more than the contested ones? 2026-11-04. Run from the repo root.

Every held stock-day goes into a bucket by how many models hold it on the same side as the average weight w_bar.
Linear, so the mean member's P&L in a bucket = w_bar's P&L there; what differs is exposure and trading. Per bucket:
share of member exposure / turnover / P&L, return, cost and net return per unit of member exposure (annualised,
5bp on |dw|, 1bp on shorts), and how much exposure survives netting (|w_bar| / mean |w_k|). Return also on stock
returns demeaned across held names and on PCA-5 residuals, so a net long bucket cannot earn the market. se = month-block
bootstrap of return paths. Also the same split by within-day tercile of position size (mean |w_k|).
"""
import argparse
import numpy as np, pandas as pd
from _common import OUT, ANN, TC, SC, SAMPLES, load, pca_resid

ap = argparse.ArgumentParser(); ap.add_argument("--w", default="roll"); a = ap.parse_args()
D = load(a.w); W, R, dates, keys = D["W"], D["R"], D["dates"], D["keys"]
T, n = R.shape
eps5 = pca_resid(5, dates, D["ids"])
print(f"exposure on names with no PCA-5 residual: {(np.abs(W) * np.isnan(eps5)).sum() / np.abs(W).sum():.4f}", flush=True)
OUT.mkdir(exist_ok=True)
GROUPS = {"all 25": (list(range(len(keys))), [15, 18, 20, 23, 25], ["<15", "15-17", "18-19", "20-22", "23-24", "25"]),
          "seeds sobol-036": ([i for i, k in enumerate(keys) if k[0] == "sobol-036"], [3, 4, 5], ["<3", "3", "4", "5"])}
NAMES = ["pnl", "pnl_dm", "pnl_eps5", "expo_mem", "expo_bar", "turn_mem", "turn_bar", "short_mem", "short_bar", "names"]
months = dates.to_period("M"); mcode = pd.factorize(months)[0]; rng = np.random.default_rng(0)


def tally(m_idx, edges):
    B = len(edges) + 2                               # last bucket = not held today: exits, trades only
    out = {k: np.zeros((T, B)) for k in NAMES}
    size = {k: np.zeros((T, 3 * B)) for k in ("pnl", "expo_mem", "turn_mem", "short_mem")}
    prev = np.zeros((len(m_idx), n)); prev_bar = np.zeros(n)
    for t in range(T):
        w = W[m_idx, t].astype(np.float64); bar = w.mean(0); m = np.abs(w).mean(0)
        act = m > 0
        same = (np.sign(w) == np.sign(bar)).sum(0)
        b = np.full(n, B - 1); b[act] = np.digitize(same[act], edges)
        r = R[t].astype(np.float64); rm = r[act].mean() if act.any() else 0.0
        e = np.where(np.isnan(eps5[t]), r - rm, eps5[t]).astype(np.float64)
        dw = np.abs(w - prev).mean(0); short = np.clip(-w, 0, None).mean(0)
        for k, v in (("pnl", bar * r), ("pnl_dm", bar * (r - rm)), ("pnl_eps5", bar * e), ("expo_mem", m), ("expo_bar", np.abs(bar)),
                     ("turn_mem", dw), ("turn_bar", np.abs(bar - prev_bar)), ("short_mem", short),
                     ("short_bar", np.clip(-bar, 0, None)), ("names", act * 1.0)):
            out[k][t] = np.bincount(b, weights=v, minlength=B)
        rank = np.zeros(n, dtype=int)
        if act.any():                                # size tercile within the day, among held names
            q = np.quantile(m[act], [1 / 3, 2 / 3]); rank[act] = np.digitize(m[act], q)
        b2 = b * 3 + rank
        for k, v in (("pnl", bar * r), ("expo_mem", m), ("turn_mem", dw), ("short_mem", short)):
            size[k][t] = np.bincount(b2, weights=v, minlength=3 * B)
        prev, prev_bar = w, bar
    return out, size


def summary(o, labels, group):
    rows = []
    for s, (lo, hi) in SAMPLES.items():
        mask = np.ones(T, bool)
        if lo: mask &= dates >= pd.Timestamp(f"{lo}-01-01")
        if hi: mask &= dates <= pd.Timestamp(f"{hi}-12-31")
        tot = {k: o[k][mask].sum() for k in NAMES}
        cost = TC * o["turn_mem"] + SC * o["short_mem"]
        M = pd.factorize(mcode[mask])[0]; nm = M.max() + 1
        by_m = lambda x: np.stack([np.bincount(M, weights=x[mask, j], minlength=nm) for j in range(x.shape[1])], 1)
        pm, em, cm = by_m(o["pnl"]), by_m(o["expo_mem"]), by_m(cost)
        draws = rng.integers(0, nm, size=(500, nm))
        rr = np.stack([pm[d].sum(0) / np.maximum(em[d].sum(0), 1e-12) for d in draws])
        nn = np.stack([(pm[d] - cm[d]).sum(0) / np.maximum(em[d].sum(0), 1e-12) for d in draws])
        for j, lab in enumerate(labels + ["not held"]):
            e, p_, c = o["expo_mem"][mask, j].sum(), o["pnl"][mask, j].sum(), cost[mask, j].sum()
            rows.append({"group": group, "sample": s, "bucket": lab, "names_day": o["names"][mask, j].mean(),
                         "share_expo": e / tot["expo_mem"], "share_turn": o["turn_mem"][mask, j].sum() / tot["turn_mem"],
                         "share_pnl": p_ / tot["pnl"], "share_cost": c / cost[mask].sum(),
                         "ret_per_expo_pct": 100 * ANN * p_ / e if e else np.nan, "ret_se": 100 * ANN * rr[:, j].std(ddof=1) if e else np.nan,
                         "cost_per_expo_pct": 100 * ANN * c / e if e else np.nan,
                         "net_per_expo_pct": 100 * ANN * (p_ - c) / e if e else np.nan, "net_se": 100 * ANN * nn[:, j].std(ddof=1) if e else np.nan,
                         "ret_dm_per_expo_pct": 100 * ANN * o["pnl_dm"][mask, j].sum() / e if e else np.nan,
                         "ret_eps5_per_expo_pct": 100 * ANN * o["pnl_eps5"][mask, j].sum() / e if e else np.nan,
                         "kept": o["expo_bar"][mask, j].sum() / e if e else np.nan,
                         "ret_per_bar_expo_pct": 100 * ANN * p_ / o["expo_bar"][mask, j].sum() if e else np.nan,
                         "share_expo_bar": o["expo_bar"][mask, j].sum() / tot["expo_bar"],
                         "share_turn_bar": o["turn_bar"][mask, j].sum() / tot["turn_bar"],
                         "turn_per_expo": o["turn_mem"][mask, j].sum() / e if e else np.nan,
                         "turn_bar_per_expo_bar": o["turn_bar"][mask, j].sum() / o["expo_bar"][mask, j].sum() if e else np.nan})
    return rows


rows, size_rows = [], []
for name, (idx, edges, labels) in GROUPS.items():
    o, sz = tally(idx, edges)
    rows += summary(o, labels, name)
    print(f"{name}: mean member exposure {o['expo_mem'].sum(1).mean():.4f}, turnover {o['turn_mem'].sum(1).mean():.4f}, "
          f"mu {100 * ANN * o['pnl'].sum(1).mean():.3f}%", flush=True)
    B = len(labels) + 1
    for j, lab in enumerate(labels):
        for q, ql in enumerate(("small", "mid", "large")):
            e, p_, c = sz["expo_mem"][:, j * 3 + q].sum(), sz["pnl"][:, j * 3 + q].sum(), (TC * sz["turn_mem"] + SC * sz["short_mem"])[:, j * 3 + q].sum()
            size_rows.append({"group": name, "bucket": lab, "size": ql, "share_expo": e / sz["expo_mem"].sum(),
                              "ret_per_expo_pct": 100 * ANN * p_ / e if e else np.nan, "cost_per_expo_pct": 100 * ANN * c / e if e else np.nan,
                              "net_per_expo_pct": 100 * ANN * (p_ - c) / e if e else np.nan})
df, sd = pd.DataFrame(rows), pd.DataFrame(size_rows)
df.to_csv(OUT / f"agreement_{a.w}.csv", index=False); sd.to_csv(OUT / f"agreement_size_{a.w}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.round(3).to_string(index=False))
print(sd.round(2).to_string(index=False))
