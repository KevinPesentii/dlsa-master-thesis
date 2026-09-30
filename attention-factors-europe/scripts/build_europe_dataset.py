"""Stage 2 of the European builds: raw tables -> returns / universe / features in the
ECU/EUR numeraire, the JKP table and the coverage tables.

    python scripts/build_europe_dataset.py --config configs/europe17_data.yaml [--cutoff YYYY-MM-DD]

Inputs: the version's output.inspect_dir from build_europe_universe.py (cap table,
ranking, listings.csv) and the union raw pull of fetch_europe_raw.py. Outputs and
conventions: src/afe/data/build_europe.py. --cutoff rebuilds from inputs truncated at that
date into output.inspect_dir/cutoff_<date>/ (prefix-invariance check).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import build_europe  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--no-jkp", action="store_true", help="skip the JKP table (run it later with --jkp-only)")
    ap.add_argument("--jkp-only", action="store_true", help="only the JKP table, from the saved stage-2 outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out, ins = ROOT / cfg["output"]["dir"], ROOT / cfg["output"]["inspect_dir"]
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    if cutoff is not None:
        out = ins = ins / f"cutoff_{args.cutoff}"
    t0 = time.time()
    log = lambda *a, **k: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731
    if args.jkp_only:
        build_europe.build_jkp_only(cfg, ROOT, log)
        return
    res = build_europe.build(cfg, ROOT, out, ins, cutoff=cutoff, log=log, jkp=not args.no_jkp)
    print(res["report"])
    print(f"\nwrote {out} and {ins} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
