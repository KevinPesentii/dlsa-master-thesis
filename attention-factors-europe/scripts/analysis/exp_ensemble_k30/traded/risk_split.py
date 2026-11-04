"""Is what the models disagree on leftover factor risk or stock picks? 2026-11-04. Run from the repo root.

Every book's daily return w'R splits exactly into w'(R - eps) (the part PCA-K explains) + w'eps (the residual), with
the PCA loadings of the 252 days before each date, the same benchmark for all models. Names without that history go
fully to the residual. Pieces: mean member, average w_bar, w_bar rescaled to exposure 1, disagreement w_k - w_bar; for
the five seeds of each candidate and for all 25. Also the mean pairwise correlation between models of each part.
"""
import argparse, itertools
import numpy as np, pandas as pd
from _common import OUT, ANN, SAMPLES, load, pca_resid

ap = argparse.ArgumentParser(); ap.add_argument("--w", default="roll"); a = ap.parse_args()
D = load(a.w); W, R, dates, keys = D["W"], D["R"], D["dates"], D["keys"]
OUT.mkdir(exist_ok=True)
m = len(keys); series = {}
for K in (1, 5, 30):
    eps = pca_resid(K, dates, D["ids"]); miss = np.isnan(eps)
    sysr = np.where(miss, 0.0, R - eps).astype(np.float64); idio = np.where(miss, R, eps).astype(np.float64)
    series[K] = (np.stack([(W[i] * sysr).sum(1) for i in range(m)]), np.stack([(W[i] * idio).sum(1) for i in range(m)]))
    print(f"PCA-{K}: exposure without a residual {np.mean([(np.abs(W[i]) * miss).sum() / np.abs(W[i]).sum() for i in range(m)]):.4f}", flush=True)
    del sysr, idio, eps


def describe(s, i, mask):
    s, i = s[mask], i[mask]; t = s + i
    vt, vs, vi = t.var(ddof=1), s.var(ddof=1), i.var(ddof=1)
    return {"sigma_pct": 100 * np.sqrt(ANN * vt), "sigma_sys_pct": 100 * np.sqrt(ANN * vs), "sigma_idio_pct": 100 * np.sqrt(ANN * vi),
            "share_sys": vs / vt, "share_idio": vi / vt, "share_cov": 2 * np.cov(s, i)[0, 1] / vt,
            "mu_pct": 100 * ANN * t.mean(), "mu_sys_pct": 100 * ANN * s.mean(), "mu_idio_pct": 100 * ANN * i.mean(),
            "SR_idio": i.mean() / i.std(ddof=1) * np.sqrt(ANN)}


groups = {"all 25": list(range(m))}
for c in D["cands"]:
    groups[f"seeds {c}"] = [i for i, k in enumerate(keys) if k[0] == c]
rows = []
for name, idx in groups.items():
    bar = sum(W[i].astype(np.float64) for i in idx) / len(idx)
    l1 = np.abs(bar).sum(1); scale = np.divide(1.0, l1, out=np.zeros_like(l1), where=l1 > 0)
    for K, (S, I) in series.items():
        sb, ib = S[idx].mean(0), I[idx].mean(0)
        for smp, (lo, hi) in SAMPLES.items():
            mask = np.ones(len(dates), bool)
            if lo: mask &= dates >= pd.Timestamp(f"{lo}-01-01")
            if hi: mask &= dates <= pd.Timestamp(f"{hi}-12-31")
            pieces = {"member (mean)": pd.DataFrame([describe(S[i], I[i], mask) for i in idx]).mean().to_dict(),
                      "average": describe(sb, ib, mask), "average rescaled": describe(sb * scale, ib * scale, mask),
                      "disagreement (mean)": pd.DataFrame([describe(S[i] - sb, I[i] - ib, mask) for i in idx]).mean().to_dict()}
            rows += [{"group": name, "K": K, "sample": smp, "piece": k, **v} for k, v in pieces.items()]
    print(f"{name} done", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / f"risk_split_{a.w}.csv", index=False)

corr = []
for K, (S, I) in series.items():
    for smp, (lo, hi) in SAMPLES.items():
        mask = np.ones(len(dates), bool)
        if lo: mask &= dates >= pd.Timestamp(f"{lo}-01-01")
        if hi: mask &= dates <= pd.Timestamp(f"{hi}-12-31")
        C = {"total": np.corrcoef((S + I)[:, mask]), "sys": np.corrcoef(S[:, mask]), "idio": np.corrcoef(I[:, mask])}
        for part, c in C.items():
            same = [c[i, j] for i, j in itertools.combinations(range(m), 2) if keys[i][0] == keys[j][0]]
            diff = [c[i, j] for i, j in itertools.combinations(range(m), 2) if keys[i][0] != keys[j][0]]
            s036 = [c[i, j] for i, j in itertools.combinations(range(m), 2) if keys[i][0] == keys[j][0] == "sobol-036"]
            corr.append({"K": K, "sample": smp, "part": part, "same_config": np.mean(same), "diff_config": np.mean(diff), "sobol036": np.mean(s036)})
cdf = pd.DataFrame(corr); cdf.to_csv(OUT / f"risk_split_corr_{a.w}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df[df["sample"] == "1998-2021"].round(3).to_string(index=False))
print(cdf.round(3).to_string(index=False))
