"""Pairwise daily-return correlations of the 50 OOS K=30 runs (5 candidates x seeds 0-4, rolling and fixed window):
same config / different seed vs different config / same seed vs different config / different seed.
Daily series from each run's oos_daily.csv (net = gross - 5bp turnover - 1bp short). A diagnostic, not a thesis row.
Run from the repo root."""
import itertools
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path.cwd()  # run from the repo root
TC, SC = 5e-4, 1e-4
OUT, FULL = ROOT / "runs/_matrix/exp_ensemble_k30", ROOT / "runs/_matrix/exp_fixedwin_full"
WINDOWS = {"roll": ("us_ensemble_val", "us_attention", "us_k30_"),
           "fixed": ("us_ensemble_val_fixedwin", "us_attention_fixedwin", "us_k30_fixedwin_")}
full, new = pd.read_csv(FULL / "summary.csv"), pd.read_csv(OUT / "summary.csv")
for w, (val_name, spec036, prefix) in WINDOWS.items():
    E = yaml.safe_load((ROOT / "configs" / f"{val_name}.yaml").read_text())
    cands, seeds = list(E["candidates"]), list(E["seeds"])
    series = {}
    for c in cands:
        g = full[(full.spec == spec036) & (full.K == 30)] if c == cands[0] else new[new.spec == prefix + c.replace("-", "")]
        assert sorted(g.seed) == seeds, (w, c)
        for _, r in g.iterrows():
            d = pd.read_csv(ROOT / "runs" / Path(r.run_dir).name / "oos_daily.csv", parse_dates=["date"]).set_index("date")
            series[(c, int(r.seed))] = d
    keys = list(series)
    for kind in ("gross", "net", "turnover"):
        if kind == "net":
            X = pd.DataFrame({k: v.gross - TC * v.turnover - SC * v["short"] for k, v in series.items()})
        else:
            X = pd.DataFrame({k: v[kind] for k, v in series.items()})
        C = X.corr()
        groups = {"same config, diff seed": [], "diff config, same seed": [], "diff config, diff seed": []}
        for a, b in itertools.combinations(keys, 2):
            g = ("same config, diff seed" if a[0] == b[0] else
                 "diff config, same seed" if a[1] == b[1] else "diff config, diff seed")
            groups[g].append(C.loc[[a], [b]].to_numpy()[0, 0])
        print(f"\n[{w}] {kind}: mean (min-max, n pairs)")
        for g, v in groups.items():
            print(f"  {g:24s} {np.mean(v):.3f} ({np.min(v):.3f}-{np.max(v):.3f}, n={len(v)})")
        if kind == "gross":
            print("  within-config mean by candidate:",
                  {c: float(round(np.mean([C.loc[[(c, s)], [(c, t)]].to_numpy()[0, 0] for s, t in itertools.combinations(seeds, 2)]), 3)) for c in cands})
            # config x config block means (diagonal = across seeds, off-diagonal = across configs)
            B = pd.DataFrame(index=cands, columns=cands, dtype=float)
            for c1, c2 in itertools.product(cands, cands):
                vals = [C.loc[[(c1, s)], [(c2, t)]].to_numpy()[0, 0] for s in seeds for t in seeds if (c1, s) != (c2, t)]
                B.loc[c1, c2] = np.mean(vals)
            print(B.round(3).to_string())
            # sub-periods
            for lo, hi in (("1998", "2009"), ("2010", "2021")):
                Cp = X.loc[lo:hi].corr()
                gp = {"same config": [], "diff config": []}
                for a, b in itertools.combinations(keys, 2):
                    gp["same config" if a[0] == b[0] else "diff config"].append(Cp.loc[[a], [b]].to_numpy()[0, 0])
                print(f"  {lo}-{hi}: " + ", ".join(f"{g} {np.mean(v):.3f}" for g, v in gp.items()))
