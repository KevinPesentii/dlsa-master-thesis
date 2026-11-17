"""Daily high, low and close of the US pool from CRSP CIZ, for the high-low spread cross-check of
scripts/robustness_stored_weights.py (the raw pull of fetch_us_raw.py has dlyhigh but not dlylow).

    python scripts/fetch_us_highlow.py --data-dir data/us/shared --out data/us/private/raw/crsp_highlow.parquet

Env py3.12-4338 (wrds package). Login as for every pull: WRDS_USERNAME from .env or the environment,
password from pgpass. Reads crsp.dsf_v2 only (never legacy crsp.dsf), every permno ever in the universe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data.compustat_us import connect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/us/shared")
    ap.add_argument("--out", type=Path, default=ROOT / "data/us/private/raw/crsp_highlow.parquet")
    ap.add_argument("--start", default="1997-01-01")
    ap.add_argument("--end", default="2021-12-31")
    args = ap.parse_args()
    permnos = sorted({int(s) for s in pd.read_parquet(args.data_dir / "universe.parquet", columns=["sec_id"])["sec_id"]})
    db = connect(ROOT)
    out = []
    for i in range(0, len(permnos), 500):
        chunk = ",".join(str(p) for p in permnos[i:i + 500])
        out.append(db.raw_sql(
            f"""select permno, dlycaldt as date, dlyhigh::float8 as high, dlylow::float8 as low,
                       dlyprc::float8 as close, dlybid::float8 as bid, dlyask::float8 as ask
                from crsp.dsf_v2
                where permno in ({chunk}) and dlycaldt between '{args.start}' and '{args.end}'""",
            date_cols=["date"]))
        print(f"  {min(i + 500, len(permnos))}/{len(permnos)} permnos, {sum(map(len, out)):,} rows", flush=True)
    db.close()
    df = pd.concat(out, ignore_index=True).sort_values(["permno", "date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
