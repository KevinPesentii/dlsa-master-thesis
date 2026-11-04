"""Fixed vs rolling residual window, full sequential run: runs/_matrix/exp_fixedwin_full (5 seeds, K 1-100)."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ANN, TC, SC = 252, 5e-4, 1e-4
MX = Path("runs/_matrix/exp_fixedwin_full")
Ks, seeds = (1, 3, 5, 8, 15, 30, 100), range(5)
S = {"us_attention": "roll", "us_attention_fixedwin": "fixed"}

summ = pd.read_csv(MX / "summary.csv")
runs = {}
for _, r in summ.iterrows():
    d = Path("runs") / Path(r["run_dir"]).name
    m = json.loads((d / "metrics.json").read_text())
    daily = pd.read_csv(d / "oos_daily.csv", parse_dates=["date"]).set_index("date")
    daily["net"] = daily["gross"] - TC * daily["turnover"] - SC * daily["short"]
    runs[(S[r["spec"]], int(r["K"]), int(r["seed"]))] = (m, daily)
assert len(runs) == 70

def sr(x): return x.mean() / x.std(ddof=1) * np.sqrt(ANN)

# per run
rows = []
for (w, K, s), (m, d) in runs.items():
    assert m["residual_window"] == ("fixed" if w == "fixed" else "rolling")
    rows.append({"w": w, "K": K, "seed": s, "SR": sr(d["gross"]), "SR_net": sr(d["net"]),
                 "SR_net_metrics": m["net"]["SR"], "mu": m["gross"]["mu_pct"], "mu_net": m["net"]["mu_pct"],
                 "sigma": m["gross"]["sigma_pct"], "cost": m["gross"]["mu_pct"] - m["net"]["mu_pct"],
                 "turnover": m["turnover_daily"], "short": m["short_exposure"], "beta": m["beta_ew_market"],
                 "BE": m["break_even_turnover_cost_bps"], "n_traded": d["n_traded"].mean(),
                 "net_98_09": sr(d.loc[:"2009", "net"]), "net_10_21": sr(d.loc["2010":, "net"]),
                 "gross_98_09": sr(d.loc[:"2009", "gross"]), "gross_10_21": sr(d.loc["2010":, "gross"])})
t = pd.DataFrame(rows).sort_values(["K", "seed", "w"])
assert (t["SR_net"] - t["SR_net_metrics"]).abs().max() < 1e-6
t.drop(columns="SR_net_metrics").to_csv(MX / "compare_runs.csv", index=False)

# moving-block bootstrap over dates, the same blocks for every series
dates = runs[("roll", 30, 0)][1].index
n, B, reps = len(dates), 21, 2000
rng = np.random.default_rng(0)
starts = rng.integers(0, n - B, size=(reps, n // B + 1))
idx = (starts[:, :, None] + np.arange(B)).reshape(reps, -1)[:, :n]
def boot_sr(x):
    xb = x[idx]; return xb.mean(1) / xb.std(1, ddof=1) * np.sqrt(ANN)

cols = ["SR", "SR_net", "mu", "mu_net", "sigma", "cost", "turnover", "short", "beta", "BE", "n_traded",
        "net_98_09", "net_10_21", "gross_98_09", "gross_10_21"]
out = []
for K in Ks:
    a = t[(t.K == K) & (t.w == "roll")].set_index("seed")[cols]
    b = t[(t.K == K) & (t.w == "fixed")].set_index("seed")[cols]
    dd = b - a
    bn = np.mean([boot_sr(runs[("fixed", K, s)][1]["net"].to_numpy()) - boot_sr(runs[("roll", K, s)][1]["net"].to_numpy())
                  for s in seeds], axis=0)
    bg = np.mean([boot_sr(runs[("fixed", K, s)][1]["gross"].to_numpy()) - boot_sr(runs[("roll", K, s)][1]["gross"].to_numpy())
                  for s in seeds], axis=0)
    corr_fr = np.mean([np.corrcoef(runs[("fixed", K, s)][1]["gross"], runs[("roll", K, s)][1]["gross"])[0, 1] for s in seeds])
    corr_rr = np.mean([np.corrcoef(runs[("roll", K, s)][1]["gross"], runs[("roll", K, (s + 1) % 5)][1]["gross"])[0, 1] for s in seeds])
    yrs = []
    for s in seeds:
        f, r = runs[("fixed", K, s)][0]["by_year"], runs[("roll", K, s)][0]["by_year"]
        yrs.append(sum(f[y]["SR_net"] > r[y]["SR_net"] for y in f))
    row = {"K": K}
    for c in cols:
        row[f"roll_{c}"], row[f"fixed_{c}"], row[f"d_{c}"] = a[c].mean(), b[c].mean(), dd[c].mean()
    row.update({"roll_SR_net_sd": a["SR_net"].std(), "fixed_SR_net_sd": b["SR_net"].std(),
                "d_net_seed_sd": dd["SR_net"].std(), "d_net_seed_t": dd["SR_net"].mean() / (dd["SR_net"].std() / np.sqrt(5)),
                "d_net_seeds_pos": int((dd["SR_net"] > 0).sum()), "d_net_boot_se": bn.std(), "d_net_boot_p_le0": np.mean(bn <= 0),
                "d_gross_seed_sd": dd["SR"].std(), "d_gross_seeds_pos": int((dd["SR"] > 0).sum()), "d_gross_boot_se": bg.std(),
                "corr_fixed_roll": corr_fr, "corr_roll_roll": corr_rr, "years_fixed_wins_mean": np.mean(yrs)})
    out.append(row)
o = pd.DataFrame(out)
o.to_csv(MX / "compare_paired.csv", index=False)
pd.set_option("display.width", 250)
print(o[["K", "roll_SR", "fixed_SR", "d_SR", "d_gross_seeds_pos", "d_gross_boot_se",
         "roll_SR_net", "fixed_SR_net", "d_SR_net", "d_net_seed_sd", "d_net_seed_t", "d_net_seeds_pos", "d_net_boot_se", "d_net_boot_p_le0"]].round(3).to_string(index=False))
print()
print(o[["K", "roll_mu", "fixed_mu", "roll_mu_net", "fixed_mu_net", "roll_cost", "fixed_cost", "roll_sigma", "fixed_sigma",
         "roll_turnover", "fixed_turnover", "roll_short", "fixed_short", "roll_beta", "fixed_beta", "roll_BE", "fixed_BE",
         "roll_n_traded", "fixed_n_traded"]].round(2).to_string(index=False))
print()
print(o[["K", "roll_net_98_09", "fixed_net_98_09", "roll_net_10_21", "fixed_net_10_21", "roll_gross_98_09", "fixed_gross_98_09",
         "roll_gross_10_21", "fixed_gross_10_21", "years_fixed_wins_mean", "corr_fixed_roll", "corr_roll_roll"]].round(2).to_string(index=False))

# per-seed K = 30 and K = 100
print()
for K in (30, 100):
    a = t[(t.K == K) & (t.w == "roll")].set_index("seed"); b = t[(t.K == K) & (t.w == "fixed")].set_index("seed")
    print(f"K={K} per seed net roll", a["SR_net"].round(2).tolist(), "fixed", b["SR_net"].round(2).tolist(),
          "turnover roll", a["turnover"].round(3).tolist(), "fixed", b["turnover"].round(3).tolist())

# the baseline against the thesis row it repeats (same config, other machine)
th = pd.read_csv("runs/_matrix/thesis/summary.csv")
th = th[th.spec == "us_attention"].groupby("K")[["SR", "SR_net", "turnover"]].mean()
base = t[t.w == "roll"].groupby("K")[["SR", "SR_net", "turnover"]].mean()
print("\nbaseline (this machine) vs thesis us_attention row (2026-10-10):")
print(pd.concat({"thesis": th, "baseline": base}, axis=1).round(3).to_string())

# fixed by year, seed means, K = 30 and 100 (net)
yr = {}
for K in (30, 100):
    for w in ("roll", "fixed"):
        yr[(K, w)] = pd.DataFrame([runs[(w, K, s)][0]["by_year"] for s in seeds]).map(lambda v: v["SR_net"]).mean()
yt = pd.DataFrame(yr)
yt.to_csv(MX / "compare_by_year_net.csv")
print("\nnet SR by year, seed means:")
print(yt.round(2).T.to_string())
