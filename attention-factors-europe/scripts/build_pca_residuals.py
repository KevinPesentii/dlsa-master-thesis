"""Step one of the two-step benchmark: rolling PCA residuals and composition pieces.

    python scripts/build_pca_residuals.py [--config configs/us_pca_longconv.yaml]
                                          [--K 3 30] [--end 1992-12-31] [--out data/us/private/pca_test]
                                          [--loading-window 252]

Reads data/us/shared/returns.parquet + universe.parquet (+ raw daily files for the year before
the sample so that the first PCA window is complete), writes factors.out_dir/K<K>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.model import pca_factors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_pca_longconv.yaml")
    ap.add_argument("--K", type=int, nargs="*", help="override factors.n_factors")
    ap.add_argument("--end", help="override sample.end (YYYY-MM-DD), for quick checks")
    ap.add_argument("--out", help="override factors.out_dir")
    ap.add_argument("--loading-window", type=int, help="override factors.loading_window (0 = PCA projection)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    fc = cfg["factors"]
    Ks = args.K or fc["n_factors"]
    end = args.end or str(cfg["sample"]["end"])
    out = ROOT / (args.out or fc["out_dir"])
    lw = fc["loading_window"] if args.loading_window is None else args.loading_window

    t0 = time.time()
    panel = pca_factors.load_panel(ROOT / cfg["data"]["dir"], end, str(cfg["data"]["history_start"]),
                                   ROOT / cfg["data"]["raw_daily_dir"])
    print(f"panel: {panel.R.shape[0]} days x {panel.R.shape[1]} names, "
          f"{panel.dates[0]:%Y-%m-%d} to {panel.dates[-1]:%Y-%m-%d}, "
          f"members/day {panel.member.sum(1)[panel.member.any(1)].mean():.0f}  ({time.time() - t0:.0f}s)")
    meta = pca_factors.build(panel, Ks, fc["cov_window"], lw, out)
    print(meta)
    print(f"wrote {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
