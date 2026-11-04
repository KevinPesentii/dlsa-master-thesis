"""Pairwise daily-return correlations on the validation window (1996-1997), K = 30, rolling residual window.

- top 25: the 25 best points of the headline search at seed 0 (runs/_ensval/us_ensemble_val_top25, this laptop);
- top 5 x seeds 0-4 (runs/_ensval/us_ensemble_val, AWS): same config / different seed vs different config, the
  validation counterpart of seed_config_corr.py (out of sample);
- the same five at seed 0 on both machines: the float noise between them.
Every model's 1996-1997 weights re-booked with ensemble_validation's own booking (net SR checked against
trials.jsonl). A diagnostic, not a thesis row. Run from the repo root.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from afe.data import slots
from afe.evaluation import metrics
import run_attention_us as base
import search_validation as sv

OUT = ROOT / "runs/_matrix/exp_ensemble_k30"
TOP25, TOP5 = ROOT / "runs/_ensval/us_ensemble_val_top25", ROOT / "runs/_ensval/us_ensemble_val"
E = yaml.safe_load((ROOT / "configs/us_ensemble_val_top25.yaml").read_text())
cfg = sv.with_overrides(yaml.safe_load((ROOT / E["base_config"]).read_text()), E.get("fixed") or {})
first_oos, V = E["first_oos_year"], E["validation_years"]
ob = cfg["objective"]
p = slots.load_slots(ROOT / cfg["data"]["dir"], f"{base.window_start(cfg, first_oos):%Y-%m-%d}", f"{first_oos - 1}-12-31")
t_idx = np.arange(p.dates.searchsorted(pd.Timestamp(first_oos - V, 1, 1)), len(p.dates))


def booked(run_dir: Path) -> dict:
    """(name, seed) -> daily gross / net / turnover of every model in the run, checked against its trial."""
    trials = {(r["name"], r["seed"]): r for r in map(json.loads, (run_dir / "trials.jsonl").read_text().splitlines())}
    out = {}
    for (n, s), r in trials.items():
        d = base.daily_book(p, np.load(run_dir / "weights" / f"{n}_s{s}.npy").astype(np.float64), t_idx, cfg)
        net = d["gross"] - ob["turnover_cost"] * d["turnover"] - ob["short_cost"] * d["short"]
        assert abs(metrics.annualised(net.to_numpy(float))["SR"] - r["net_SR"]) < 1e-5, (run_dir.name, n, s)
        out[(n, s)] = pd.DataFrame({"gross": d["gross"].to_numpy(), "net": net.to_numpy(), "turnover": d["turnover"].to_numpy()})
    return out


def pairs(C: pd.DataFrame, keys: list, label) -> pd.DataFrame:
    return pd.DataFrame([{"a": a, "b": b, "group": label(a, b), "corr": C.loc[[a], [b]].to_numpy()[0, 0]}
                         for a, b in itertools.combinations(keys, 2)])


def show(df: pd.DataFrame, title: str):
    g = df.groupby("group")["corr"].agg(["mean", "median", "min", "max", "count"])
    print(f"\n{title}\n{g.round(3).to_string()}")


top25, top5 = booked(TOP25), booked(TOP5)
names = list(E["candidates"])
assert set(top25) == {(n, 0) for n in names}
sr = {n: metrics.annualised(top25[(n, 0)]["net"].to_numpy(float))["SR"] for n in names}
Cs, within = {}, {}                                  # kind -> 25 x 25 corr at seed 0; kind -> config -> mean over seed pairs

# 1. the 25 configurations at seed 0
for kind in ("gross", "net", "turnover"):
    X = pd.DataFrame({n: top25[(n, 0)][kind] for n in names})
    C = Cs[kind] = X.corr()
    top5_names = sorted({k[0] for k in top5})        # the five finalists of the search
    df = pairs(C, names, lambda a, b: "both top 5" if a in top5_names and b in top5_names else
               "one top 5" if a in top5_names or b in top5_names else "both ranks 6-25")
    show(df.assign(group="all 300 pairs"), f"[top 25, seed 0] {kind}: pairwise corr")
    if kind == "gross":
        show(df, "  by search rank")
        q = np.quantile(df["corr"], [0.05, 0.25, 0.5, 0.75, 0.95])
        print("  quantiles 5/25/50/75/95%:", np.round(q, 3))
        mean_other = ((C.sum() - 1) / (len(names) - 1)).sort_values()
        print("  each config's mean corr with the other 24 (seed-0 val net SR):")
        print("   " + "\n   ".join(f"{n}  {v:.3f}  ({sr[n]:.2f})" for n, v in mean_other.items()))
        top = df.sort_values("corr")
        print("  lowest 5 pairs:", ", ".join(f"{a}/{b} {c:.2f}" for a, b, c in top[["a", "b", "corr"]].head(5).to_numpy()))
        print("  highest 5 pairs:", ", ".join(f"{a}/{b} {c:.2f}" for a, b, c in top[["a", "b", "corr"]].tail(5).to_numpy()))
        # shared settings: do pairs that share a categorical knob move together more?
        o = {n: E["candidates"][n] for n in names}
        for knob in ("policy.input", "policy.input_normalise", "training.batch_days"):
            same = df.apply(lambda r: o[r.a][knob] == o[r.b][knob], axis=1)
            print(f"  {knob:24s} same {df['corr'][same].mean():.3f} (n={same.sum()})  different {df['corr'][~same].mean():.3f} (n={(~same).sum()})")
        tau = df.apply(lambda r: (o[r.a]["model.score_std_target"] is None) == (o[r.b]["model.score_std_target"] is None), axis=1)
        print(f"  {'temperature on/off':24s} same {df['corr'][tau].mean():.3f} (n={tau.sum()})  different {df['corr'][~tau].mean():.3f} (n={(~tau).sum()})")
        for knob in ("objective.lambda_var", "policy.input_scale", "model.lambda_ridge"):
            dist = df.apply(lambda r: abs(np.log10(o[r.a][knob]) - np.log10(o[r.b][knob])), axis=1)
            print(f"  {knob:24s} rank corr(|log10 gap|, pair corr) {pd.Series(dist).corr(df['corr'], method='spearman'):+.2f}")
        # the diversification an equal-weight average of the 25 would get, gross
        sd = X.std()
        rho_bar = (C.to_numpy()[np.triu_indices(25, 1)]).mean()
        print(f"  mean pairwise {rho_bar:.3f} -> sqrt(N / (1 + (N - 1) rho)) = {np.sqrt(25 / (1 + 24 * rho_bar)):.2f} for N = 25 equal-vol members"
              f" (member vol spread {sd.min() / sd.max():.2f} min/max)")

# 2. top 5 x seeds 0-4 on the validation window (AWS)
for kind in ("gross", "net", "turnover"):
    keys = sorted(top5)
    C = pd.DataFrame({k: top5[k][kind] for k in keys}).corr()
    df = pairs(C, keys, lambda a, b: "same config, diff seed" if a[0] == b[0] else
               "diff config, same seed" if a[1] == b[1] else "diff config, diff seed")
    show(df, f"[top 5 x seeds 0-4, AWS] {kind}")
    same = df[df.group == "same config, diff seed"]
    within[kind] = {c: float(same[same.a.map(lambda k: k[0]) == c]["corr"].mean()) for c in dict.fromkeys(k[0] for k in keys)}
    if kind == "gross":
        print("  within-config mean by candidate:", {c: round(v, 3) for c, v in within[kind].items()})

# 3. the same five, seed 0, laptop vs AWS
print("\n[laptop vs AWS, seed 0] gross corr, val net SR laptop / AWS")
for n in sorted({k[0] for k in top5}):
    a, b = top25[(n, 0)], top5[(n, 0)]
    print(f"  {n}  corr {np.corrcoef(a['gross'], b['gross'])[0, 1]:.4f}   net SR "
          f"{metrics.annualised(a['net'].to_numpy(float))['SR']:.2f} / {metrics.annualised(b['net'].to_numpy(float))['SR']:.2f}")

# 4. the matrix: off the diagonal the seed-0 runs of two configurations (laptop), on it the mean over the ten seed
# pairs of one configuration (seeds 0-4, AWS) where we have them (the top five), blank elsewhere; search rank order.
search = [json.loads(l) for l in (ROOT / "runs/20261010T110947Z_search_K30_val1996-1997/trials.jsonl").read_text().splitlines()]
rank = {r["name"]: r["net_SR"] for r in search if r["seed"] == 0}
order = sorted(names, key=lambda n: -rank[n])
for kind in ("gross", "net"):
    M = Cs[kind].loc[order, order].copy()
    for n in order:
        M.loc[n, n] = within[kind].get(n, np.nan)
    M.round(4).to_csv(OUT / f"val_top25_corr_matrix_{kind}.csv")
    if kind == "gross":
        short = [n.replace("sobol-", "") for n in order]
        print(f"\n[matrix, gross] diagonal = mean over seed pairs (top 5 only), off-diagonal = seed 0; search rank order")
        print(pd.DataFrame(M.to_numpy(), index=short, columns=short).round(2).to_string(na_rep="  -"))
d = (Cs["net"] - Cs["gross"]).abs().to_numpy().max()
print(f"\nmax |net - gross| corr over the 25 x 25 (seed 0): {d:.4f}")
