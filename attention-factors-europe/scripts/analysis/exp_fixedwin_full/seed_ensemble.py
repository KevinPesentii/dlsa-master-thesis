"""Seed ensemble of runs/_matrix/exp_fixedwin_full: one book = the mean of the 5 seeds' target weights.

Re-books the saved weights (weights.parquet, slot weights by sec_id) with the runner's own daily_book, so
execution, stale closes, turnover and costs are exactly those of the single runs (checked first).
Two books per (window, K): `avg` = the plain mean (gross exposure below 1 where seeds disagree) and
`avg_l1` = the mean rescaled every day to ||w||_1 = 1, the single runs' exposure. No selection on
out-of-sample years: all five seeds, equal weight. A diagnostic, not a thesis row.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from afe.data import slots
from afe.evaluation import metrics
import run_attention_us as rau

MX = ROOT / "runs/_matrix/exp_fixedwin_full"
ANN, TC, SC = 252, 5e-4, 1e-4
cfg = yaml.safe_load((ROOT / "configs/us_headline.yaml").read_text())
assert cfg["execution"]["lag"] == 0
t0 = time.time()
p = slots.load_slots(ROOT / cfg["data"]["dir"], f"{rau.window_start(cfg, 1998):%Y-%m-%d}", "2021-12-31",
                     max_rank=cfg["sample"]["universe_size"])
t_idx = np.arange(p.dates.searchsorted(pd.Timestamp("1998-01-01")), len(p.dates))
oos_dates = p.dates[t_idx]
col = pd.Series(np.arange(p.n_pool), index=p.sec_ids[:p.n_pool])
assert col.index.is_unique
print(f"panel {p.n_pool} pool names, {len(t_idx)} oos days, closed mask {p.closed is not None}, {time.time() - t0:.0f}s", flush=True)

summ = pd.read_csv(MX / "summary.csv")
summ["dir"] = [ROOT / "runs" / Path(d).name for d in summ["run_dir"]]
W = {"us_attention": "roll", "us_attention_fixedwin": "fixed"}

def pool_weights(d: Path) -> np.ndarray:
    w = pd.read_parquet(d / "weights.parquet")
    r = oos_dates.get_indexer(w["date"]); c = col.reindex(w["sec_id"]).to_numpy()
    assert (r >= 0).all() and not np.isnan(c).any()
    wp = np.zeros((len(t_idx), p.n_pool), dtype=np.float32)
    np.add.at(wp, (r, c.astype(int)), w["w"].to_numpy())
    return wp

def sr(x): return x.mean() / x.std(ddof=1) * np.sqrt(ANN)
def summarise(daily: pd.DataFrame) -> dict:
    m = metrics.performance(daily, TC, SC, [])
    d = daily.set_index("date"); net = d["gross"] - TC * d["turnover"] - SC * d["short"]
    return {"SR": m["gross"]["SR"], "SR_net": m["net"]["SR"], "mu": m["gross"]["mu_pct"], "mu_net": m["net"]["mu_pct"],
            "sigma": m["gross"]["sigma_pct"], "turnover": m["turnover_daily"], "short": m["short_exposure"],
            "beta": m["beta_ew_market"], "BE": m["break_even_turnover_cost_bps"],
            "net_98_09": sr(net.loc[:"2009"]), "net_10_21": sr(net.loc["2010":])}

rows, books, checked = [], {}, 0
for (spec, K), g in summ.groupby(["spec", "K"]):
    w = W[spec]; acc = np.zeros((len(t_idx), p.n_pool)); l1 = []
    for _, r in g.sort_values("seed").iterrows():
        wp = pool_weights(r["dir"])
        if checked < 3 or K == 30:      # the re-booking must reproduce the run's own daily series
            ref = pd.read_csv(r["dir"] / "oos_daily.csv", parse_dates=["date"])
            mine = rau.daily_book(p, wp, t_idx, cfg)
            for c in ("gross", "turnover", "short"):
                err = np.abs(mine[c].to_numpy() - ref[c].to_numpy()).max()
                assert err < 1e-6, (r["dir"].name, c, err)
            checked += 1
        l1.append(np.abs(wp).sum(1).mean())
        acc += wp
    avg = acc / len(g)
    gl1 = np.abs(avg).sum(1)
    for kind, book in (("avg", avg), ("avg_l1", avg / np.where(gl1 > 0, gl1, 1)[:, None])):
        daily = rau.daily_book(p, book, t_idx, cfg)
        books[(w, K, kind)] = daily.set_index("date")
        rows.append({"w": w, "K": K, "book": kind, **summarise(daily), "gross_l1": float(np.abs(book).sum(1).mean()),
                     "single_l1": float(np.mean(l1))})
    print(f"{w:5s} K={K:3d} done ({len(g)} seeds), mean ||w_avg||_1 {gl1.mean():.3f}, {time.time() - t0:.0f}s", flush=True)
print(f"re-booking checked on {checked} runs: gross, turnover, short within 1e-6 of oos_daily.csv")

e = pd.DataFrame(rows)
single = pd.read_csv(MX / "compare_runs.csv").groupby(["w", "K"])[["SR", "SR_net", "turnover", "net_98_09", "net_10_21"]].mean()
e = e.join(single.add_prefix("single_"), on=["w", "K"])
e.to_csv(MX / "seed_ensemble.csv", index=False)
pd.set_option("display.width", 250)
cols = ["w", "K", "book", "gross_l1", "single_SR", "SR", "single_SR_net", "SR_net", "single_turnover", "turnover",
        "short", "BE", "single_net_98_09", "net_98_09", "single_net_10_21", "net_10_21"]
print(e.sort_values(["book", "w", "K"])[cols].round(3).to_string(index=False))

# fixed vs rolling, ensemble against ensemble: moving-block bootstrap over dates
n, B, reps = len(t_idx), 21, 2000
rng = np.random.default_rng(0)
starts = rng.integers(0, n - B, size=(reps, n // B + 1))
ix = (starts[:, :, None] + np.arange(B)).reshape(reps, -1)[:, :n]
def boot_sr(x): xb = x[ix]; return xb.mean(1) / xb.std(1, ddof=1) * np.sqrt(ANN)
out = []
for kind in ("avg_l1", "avg"):
    for K in sorted(e.K.unique()):
        f, r = books[("fixed", K, kind)], books[("roll", K, kind)]
        nf = f["gross"] - TC * f["turnover"] - SC * f["short"]; nr = r["gross"] - TC * r["turnover"] - SC * r["short"]
        b = boot_sr(nf.to_numpy()) - boot_sr(nr.to_numpy())
        out.append({"book": kind, "K": K, "d_net": sr(nf) - sr(nr), "boot_se": b.std(), "p_le0": float(np.mean(b <= 0)),
                    "corr_gross": np.corrcoef(f["gross"], r["gross"])[0, 1]})
o = pd.DataFrame(out); o.to_csv(MX / "seed_ensemble_fixed_vs_roll.csv", index=False)
print("\nensemble fixed - ensemble rolling, net SR (block bootstrap 21 days, 2000 reps)")
print(o.round(3).to_string(index=False))
