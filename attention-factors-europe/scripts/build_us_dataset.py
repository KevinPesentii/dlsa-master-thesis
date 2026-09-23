"""Stage 2: build returns / universe / features for the US from the frozen raw tables.

    python scripts/build_us_dataset.py [--config configs/us_data.yaml] [--cutoff 2005-06-30]

Runs offline against raw.dir from stage 1. The schema tables go to output.dir
(data/us/shared), the inspection tables to output.inspect_dir (data/us/private).
--cutoff rebuilds from inputs truncated at a date (the prefix-invariance check of
CLAUDE.md) into <output.inspect_dir>/cutoff_<date>/ so it can be diffed against the full build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import build_us, us_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_data.yaml")
    ap.add_argument("--cutoff", help="truncate every input at this date (YYYY-MM-DD)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw = us_panel.load_raw(ROOT / cfg["raw"]["dir"])
    out = ROOT / cfg["output"]["dir"]
    inspect = ROOT / cfg["output"].get("inspect_dir", cfg["output"]["dir"])
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    if cutoff is not None:
        out = inspect = inspect / f"cutoff_{cutoff:%Y%m%d}"
    result = build_us.build(cfg, raw, out, cutoff=cutoff, inspect_dir=inspect)
    print(result["report"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
