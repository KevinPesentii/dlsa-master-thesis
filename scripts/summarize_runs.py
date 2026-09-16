"""Collect runs/*_pca_longconv_K*/metrics.json into one table in the layout of Table 2.

    python scripts/summarize_runs.py [--glob "runs/*_pca_longconv_K*"] [--paper]

--paper prints the published "PCA Factors (Two-Step Approach)" rows underneath, marked
as such.  They are the authors' numbers, not ours; per CLAUDE.md they are for
comparison only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PAPER_PCA_TWO_STEP = {  # Epstein et al. (2025), Table 2, PCA Factors (Two-Step Approach)
    1: (2.26, 13.10, 5.79, 1.19, 6.98, 5.78, 0.10), 3: (2.76, 14.61, 5.30, 1.57, 8.29, 5.28, 0.07),
    5: (2.41, 14.10, 5.86, 1.30, 7.62, 5.84, 0.10), 8: (2.64, 14.88, 5.63, 1.50, 8.42, 5.61, 0.09),
    10: (2.66, 14.94, 5.61, 1.52, 8.48, 5.59, 0.09), 15: (2.56, 14.74, 5.75, 1.41, 8.08, 5.73, 0.09),
    30: (2.79, 15.15, 5.42, 1.57, 8.47, 5.40, 0.09), 100: (2.66, 14.36, 5.40, 1.44, 7.75, 5.38, 0.09),
}
COLS = ["K", "SR", "mu", "sigma", "SR_net", "mu_net", "sigma_net", "beta"]


def collect(pattern: str) -> pd.DataFrame:
    rows = []
    for d in sorted(ROOT.glob(pattern)):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        rows.append({
            "run": d.name, "K": m["K"], "seed": m["seed"],
            "loadings": cfg["factors"]["loading_window"] or "proj", "input": cfg["policy"]["input"],
            "SR": m["gross"]["SR"], "mu": m["gross"]["mu_pct"], "sigma": m["gross"]["sigma_pct"],
            "SR_net": m["net"]["SR"], "mu_net": m["net"]["mu_pct"], "sigma_net": m["net"]["sigma_pct"],
            "beta": m["beta_ew_market"], "turnover": m["turnover_daily"], "short": m["short_exposure"],
            "break_even_bp": m["break_even_turnover_cost_bps"], "days": m["n_days"],
            "first": m["first"], "last": m["last"],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*_pca_longconv_K*")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    df = collect(args.glob)
    if df.empty:
        print("no runs found")
        return
    pd.set_option("display.width", 200)
    df = df.sort_values(["loadings", "input", "seed", "K"])
    print(df.drop(columns=["run"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print("\nruns:")
    for r in df.itertuples():
        print(f"  K={r.K:<3} {r.run}")
    if args.paper:
        print("\nEpstein et al. (2025) Table 2, PCA Factors (Two-Step Approach) -- published, not ours:")
        print(pd.DataFrame([(k, *v) for k, v in PAPER_PCA_TWO_STEP.items()], columns=COLS)
              .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    main()
