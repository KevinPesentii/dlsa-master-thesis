"""How fast does averaging saturate, and is the gain in seeds or in configs? 2026-11-04. Run from the repo root.

(1) Random subsets of the 25 models, N = 1..25, avg_l1 books (rescaled daily to exposure 1, booked as daily_book):
gross and net Sharpe, costs / vol. Against the equal-correlation formula SR(N) = SR_bar * sqrt(N / (1 + (N - 1) rho)).
(2) Grid: c configs x s seeds, every combination up to CAP random ones, the same seeds in every config.
(3) Eigenvalues of the 25 x 25 correlation of daily gross returns, effective number of models (sum l)^2 / sum l^2.
Subsets pick nothing: the mean over draws is what a size earns without choosing members.
"""
import argparse, itertools, time
import numpy as np, pandas as pd
from _common import BASE, OUT, ANN, TC, SC, load

ap = argparse.ArgumentParser(); ap.add_argument("--w", default="roll"); ap.add_argument("--draws", type=int, default=50); ap.add_argument("--cap", type=int, default=30)
a = ap.parse_args()
D = load(a.w); W, R, keys, cands, seeds = D["W"], D["R"], D["keys"], D["cands"], D["seeds"]
OUT.mkdir(exist_ok=True)
m, T, n = W.shape; R64 = R.astype(np.float64); rng = np.random.default_rng(0); t0 = time.time()


def book(idx):
    w = np.zeros((T, n))
    for i in idx: w += W[i]
    l1 = np.abs(w).sum(1, keepdims=True); w = np.divide(w, l1, out=np.zeros_like(w), where=l1 > 0)
    g = (w * R64).sum(1)
    turn = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, n)))).sum(1); short = np.clip(-w, 0, None).sum(1)
    net = g - TC * turn - SC * short
    sig, c = g.std(ddof=1) * np.sqrt(ANN), ANN * (g - net).mean()
    return {"gross_SR": g.mean() / g.std(ddof=1) * np.sqrt(ANN), "net_SR": net.mean() / net.std(ddof=1) * np.sqrt(ANN),
            "sigma_pct": 100 * sig, "costs_pct": 100 * c, "costs_over_vol": c / sig, "turnover": turn.mean(), "mu_pct": 100 * ANN * g.mean()}


check = book(range(m))
ref = pd.read_csv(BASE / "oos_books.csv"); ref = ref[(ref.w == a.w) & (ref.book == "all") & (ref.kind == "avg_l1")].iloc[0]
assert abs(check["net_SR"] - ref.net_SR) < 1e-4, (check["net_SR"], ref.net_SR)
print(f"all-25 book = oos_books.csv: net {check['net_SR']:.4f}, one book {time.time() - t0:.1f}s", flush=True)

series = np.stack([(W[i].astype(np.float64) * R64).sum(1) for i in range(m)])
C = np.corrcoef(series); up = C[np.triu_indices(m, 1)]
sr1 = np.mean(series.mean(1) / series.std(1, ddof=1)) * np.sqrt(ANN); rho = up.mean()
lam = np.linalg.eigvalsh(C)[::-1]
print(f"mean member gross SR {sr1:.3f}, mean pairwise corr {rho:.3f}, limit SR_bar / sqrt(rho) {sr1 / np.sqrt(rho):.2f}", flush=True)
print(f"eigenvalue shares {np.round(lam[:6] / m, 3).tolist()}, effective models {lam.sum() ** 2 / (lam ** 2).sum():.2f}", flush=True)
pd.DataFrame({"eig": lam, "share": lam / m}).to_csv(OUT / f"size_eigen_{a.w}.csv", index=False)

rows = []
for N in (1, 2, 3, 4, 5, 7, 10, 15, 20, 25):
    draws = [[i] for i in range(m)] if N == 1 else [list(range(m))] if N == 25 else [list(rng.choice(m, N, replace=False)) for _ in range(a.draws)]
    for idx in draws:
        rows.append({"N": N, "configs": len({keys[i][0] for i in idx}), **book(idx)})
    print(f"N={N} done, {time.time() - t0:.0f}s", flush=True)
r = pd.DataFrame(rows); r.to_csv(OUT / f"size_random_{a.w}.csv", index=False)
s = r.groupby("N").agg(draws=("net_SR", "size"), gross=("gross_SR", "mean"), net=("net_SR", "mean"), net_sd=("net_SR", "std"), net_min=("net_SR", "min"),
                       net_max=("net_SR", "max"), vol=("sigma_pct", "mean"), costs=("costs_pct", "mean"), cost_vol=("costs_over_vol", "mean"), turn=("turnover", "mean"))
s["gross_formula"] = [sr1 * np.sqrt(N / (1 + (N - 1) * rho)) for N in s.index]
s.to_csv(OUT / f"size_curve_{a.w}.csv")

g = []
ckey = {(c, sd): i for i, (c, sd) in enumerate(keys)}
for nc in range(1, len(cands) + 1):
    for ns in range(1, len(seeds) + 1):
        combos = list(itertools.product(itertools.combinations(cands, nc), itertools.combinations(seeds, ns)))
        if len(combos) > a.cap:
            combos = [combos[i] for i in rng.choice(len(combos), a.cap, replace=False)]
        for cs, ss in combos:
            g.append({"nc": nc, "ns": ns, "N": nc * ns, **book([ckey[(c, x)] for c in cs for x in ss])})
    print(f"grid configs={nc} done, {time.time() - t0:.0f}s", flush=True)
g = pd.DataFrame(g); g.to_csv(OUT / f"size_grid_{a.w}.csv", index=False)
pd.set_option("display.width", 200)
print(s.round(3).to_string())
for v in ("net_SR", "gross_SR", "costs_over_vol", "sigma_pct"):
    print(v); print(g.pivot_table(index="nc", columns="ns", values=v).round(2).to_string())
