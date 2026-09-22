"""Stage 1: pull the raw US tables from WRDS once.

    python scripts/fetch_us_raw.py [--config configs/us_data.yaml] [--years 2025 2024 ...] [--delisting]

Resumable: yearly CRSP daily files that already exist are skipped, so a dropped
connection costs one year, not the whole pull. --years pulls only the listed daily years
(in that order, skipping existing files), so a second worker can run from the other end
of the range in parallel. --delisting pulls only crsp_delisting.parquet (seconds; added
2026-09-22 to an existing raw directory). Output goes to raw.dir from the config, with
a manifest.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import compustat_us, wrds_us  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_data.yaml")
    ap.add_argument("--years", type=int, nargs="+", help="pull only these CRSP daily years, in order")
    ap.add_argument("--delisting", action="store_true", help="pull only the delisting-day rows")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["raw"]["dir"]

    db = compustat_us.connect(ROOT)  # same .env / pgpass login as the universe build
    try:
        if args.delisting:
            df = wrds_us.fetch_crsp_delisting(db, cfg["raw"]["start_year"], cfg["raw"]["end_year"])
            df.to_parquet(raw_dir / "crsp_delisting.parquet", index=False)
            print(f"crsp_delisting: {len(df):,} rows -> {raw_dir / 'crsp_delisting.parquet'}", flush=True)
        elif args.years:
            raw_dir.joinpath("crsp_daily").mkdir(parents=True, exist_ok=True)
            for year in args.years:
                path = raw_dir / "crsp_daily" / f"{year}.parquet"
                if path.exists():
                    print(f"{year}: exists, skipped", flush=True)
                    continue
                df = wrds_us.fetch_crsp_daily_year(db, year)
                df.to_parquet(path, index=False)
                print(f"{year}: {len(df):,} rows -> {path}", flush=True)
        else:
            manifest = wrds_us.pull_all(db, raw_dir, cfg["raw"]["start_year"], cfg["raw"]["end_year"])
            print({k: v["rows"] for k, v in manifest["tables"].items()})
    finally:
        db.close()


if __name__ == "__main__":
    main()
