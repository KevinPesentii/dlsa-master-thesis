"""What averaging removes from the K = 30 books (rolling window), 2026-11-02. Run from the repo root.

For the five sobol-036 seeds and for all 25 models: each member's weights w_i split into the average w_bar
(what the members share) and w_i - w_bar (what averaging nets out). For each piece: gross exposure, gross
return and volatility (annualised, booked as daily_book does: w . R_pool, US has no closed mask, lag 0),
volatility per unit of exposure, and how concentrated it is (names held, effective number of positions
(sum|w|)^2 / sum w^2, share of exposure in the 20 largest positions). Also how much of a member's exposure
sits in names where it disagrees in sign with the average. A diagnostic, not a thesis row.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from afe.data import slots
import run_attention_us as rau

OUT, FULL = ROOT / "runs/_matrix/exp_ensemble_k30", ROOT / "runs/_matrix/exp_fixedwin_full"
ANN = 252
cfg = yaml.safe_load((ROOT / "configs/us_headline.yaml").read_text())
t0 = time.time()
p = slots.load_slots(ROOT / cfg["data"]["dir"], f"{rau.window_start(cfg, 1998):%Y-%m-%d}", "2021-12-31",
                     max_rank=cfg["sample"]["universe_size"])
t_idx = np.arange(p.dates.searchsorted(pd.Timestamp("1998-01-01")), len(p.dates))
oos_dates = p.dates[t_idx]
R = np.asarray(p.R_pool[t_idx], dtype=np.float64)
assert np.isfinite(R).all() and p.closed is None and cfg["execution"]["lag"] == 0
col = pd.Series(np.arange(p.n_pool), index=p.sec_ids[:p.n_pool])
print(f"panel {p.n_pool} names, {len(t_idx)} days, {time.time() - t0:.0f}s", flush=True)

def pool_weights(d: Path) -> np.ndarray:
    w = pd.read_parquet(d / "weights.parquet")
    r = oos_dates.get_indexer(w["date"]); c = col.reindex(w["sec_id"]).to_numpy()
    assert (r >= 0).all() and not np.isnan(c).any()
    wp = np.zeros((len(t_idx), p.n_pool), dtype=np.float32)
    np.add.at(wp, (r, c.astype(int)), w["w"].to_numpy())
    return wp

def describe(W: np.ndarray) -> dict:
    a = np.abs(W).astype(np.float64)
    l1 = a.sum(1); ret = (W * R).sum(1)
    on = l1 > 0
    top20 = -np.partition(-a, 20, axis=1)[:, :20].sum(1)
    sd = ret.std(ddof=1) * np.sqrt(ANN)
    return {"exposure": l1.mean(), "mu_pct": 100 * ANN * ret.mean(), "sigma_pct": 100 * sd,
            "sigma_per_exposure_pct": 100 * sd / l1.mean(), "mu_per_exposure_pct": 100 * ANN * ret.mean() / l1.mean(),
            "names": (a > 1e-9).sum(1).mean(), "eff_N": (l1[on] ** 2 / (a[on] ** 2).sum(1)).mean(),
            "top20_share": (top20[on] / l1[on]).mean()}

full, new = pd.read_csv(FULL / "summary.csv"), pd.read_csv(OUT / "summary.csv")
E = yaml.safe_load((ROOT / "configs/us_ensemble_val.yaml").read_text())
dirs = {}
for c in E["candidates"]:
    g = full[(full.spec == "us_attention") & (full.K == 30)] if c == "sobol-036" else new[new.spec == "us_k30_" + c.replace("-", "")]
    for _, r in g.iterrows():
        dirs[(c, int(r.seed))] = ROOT / "runs" / Path(r.run_dir).name
assert len(dirs) == 25
W = {k: pool_weights(d) for k, d in dirs.items()}
print(f"weights loaded, {time.time() - t0:.0f}s", flush=True)

rows = []
for group, members in (("seeds: sobol-036 x 5", [k for k in W if k[0] == "sobol-036"]), ("all 25", list(W))):
    bar = (sum(W[k].astype(np.float64) for k in members) / len(members)).astype(np.float32)
    mem = pd.DataFrame([describe(W[k]) for k in members]).mean()
    idio = pd.DataFrame([describe(W[k] - bar) for k in members]).mean()
    avg = pd.Series(describe(bar))
    l1 = np.abs(bar).sum(1, keepdims=True)
    scaled = pd.Series(describe(np.divide(bar, l1, out=np.zeros_like(bar), where=l1 > 0)))
    # sign disagreement: share of a member's exposure in names where it is opposite in sign to the average
    dis = np.mean([(np.abs(W[k]) * (np.sign(W[k]) * np.sign(bar) < 0)).sum() / np.abs(W[k]).sum() for k in members])
    for piece, s in (("member (mean)", mem), ("average w_bar", avg), ("average rescaled", scaled),
                     ("netted out w_i - w_bar (mean)", idio)):
        rows.append({"group": group, "piece": piece, **s.to_dict()})
    rows.append({"group": group, "piece": "member exposure opposite to w_bar (share)", "exposure": dis})
    print(f"{group} done, {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "netting.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
