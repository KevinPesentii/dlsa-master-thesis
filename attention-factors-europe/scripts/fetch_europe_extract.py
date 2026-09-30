"""Step 1 of the European builds: month-end rows, headers and FX from WRDS, once.

    python scripts/fetch_europe_extract.py [--config configs/europe_extract.yaml] [--headers] [--fx]

Writes to output.dir (gitignored, licensed):
  secd_monthend/<year>.parquet   comp.g_secd rows with monthend = 1, common + preferred, on
                                 exchanges in `countries` (resumable: a year on disk is skipped)
  security_header.parquet        comp.g_security: every listing of companies with a line there
  company_header.parquet         comp.g_company
  fx_daily.parquet               comp.g_exrt_dly, units per GBP, the extract's quotation
                                 currencies plus fx.extra (ECU basket, XEU, EUR, USD)
The version cap tables (europe17, europe12) are cut from these by build_europe_universe.py.
--headers / --fx refetch only those tables.
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

from afe.data import compustat_global as cg  # noqa: E402
from afe.data import compustat_us as cu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "europe_extract.yaml")
    ap.add_argument("--headers", action="store_true", help="refetch the two header tables only")
    ap.add_argument("--fx", action="store_true", help="refetch the FX table only")
    ap.add_argument("--years", nargs=2, type=int, default=None,
                    help="month-end years to pull (default: the config's); with --reverse from the last one "
                         "down, so a second process can meet a first one in the middle. No headers, no FX.")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / cfg["output"]["dir"]
    (out / "secd_monthend").mkdir(parents=True, exist_ok=True)
    countries = list(cfg["countries"])
    y0, y1 = int(cfg["start_year"]), int(cfg["end_year"])
    only = args.headers or args.fx
    if args.years:
        years = range(args.years[0], args.years[1] + 1)
        db = cu.connect(ROOT)
        try:
            for y in (reversed(years) if args.reverse else years):
                path = out / "secd_monthend" / f"{y}.parquet"
                if not path.exists():
                    cg.fetch_secd_monthend(db, y, y, countries, tuple(cfg["issue_types"])).to_parquet(path, index=False)
                    print(f"  g_secd {y}: saved", flush=True)
        finally:
            db.close()
        return

    db = cu.connect(ROOT)
    try:
        if args.headers or not (out / "security_header.parquet").exists():
            cg.fetch_security_header(db, countries).to_parquet(out / "security_header.parquet", index=False)
            cg.fetch_company_header(db).to_parquet(out / "company_header.parquet", index=False)
            print("headers written", flush=True)
        if not only:
            for y in range(y0, y1 + 1):
                path = out / "secd_monthend" / f"{y}.parquet"
                if path.exists():
                    print(f"  g_secd {y}: exists, skipped", flush=True)
                    continue
                t0 = time.time()
                part = cg.fetch_secd_monthend(db, y, y, countries, tuple(cfg["issue_types"]))
                part.to_parquet(path, index=False)
                print(f"  g_secd {y}: saved in {time.time() - t0:.0f}s", flush=True)
        if args.fx or not only:
            ccy = set(cfg["fx"]["extra"])
            for p in sorted((out / "secd_monthend").glob("*.parquet")):
                ccy |= set(pd.read_parquet(p, columns=["curcdd"])["curcdd"].dropna().unique())
            fx = cg.fetch_fx(db, sorted(ccy), int(cfg["fx"]["start_year"]), y1 + 1)
            fx.to_parquet(out / "fx_daily.parquet", index=False)
            print(f"fx: {fx['tocurd'].nunique()} currencies, {fx['datadate'].min().date()} .. {fx['datadate'].max().date()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
