"""Stage 2c: features_asof.parquet and returns_history.parquet of a finished build.

    python scripts/build_asof_tables.py --config configs/us_data.yaml
    python scripts/build_asof_tables.py --config configs/europe17_data.yaml

Reads output.dir (returns, universe, features), output.inspect_dir (characteristics_*,
and for Europe returns_detail_daily, universe_asof) and for the US raw.dir, and writes the two
tables into output.dir, and for Europe also output.holfill_dir: the features are the same
there, the history is taken against the filled returns. Nothing else moves. src/afe/data/asof.py
says why they exist, docs/schemas.md what is in them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import asof, build_us as bu  # noqa: E402


def read_ret(d: Path) -> pd.DataFrame:
    r = pd.read_parquet(d / "returns.parquet", columns=["date", "sec_id"])
    r["date"] = pd.to_datetime(r["date"]).astype("datetime64[ns]")
    r["sec_id"] = r["sec_id"].astype(object)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out, ins = ROOT / cfg["output"]["dir"], ROOT / cfg["output"]["inspect_dir"]
    names = bu.characteristic_names(cfg)

    print(f"{cfg['market']}: features_asof from {ins}", flush=True)
    dc = pd.read_parquet(ins / "characteristics_daily.parquet").set_index(["date", "permno"]).sort_index()
    mc = pd.read_parquet(ins / "characteristics_monthly.parquet").set_index(["month", "permno"]).sort_index()
    rows = pd.read_parquet(out / "features.parquet", columns=["date", "rf"]).drop_duplicates("date")
    rf = rows.assign(date=pd.to_datetime(rows["date"]).astype("datetime64[ns]")).set_index("date")["rf"].sort_index()
    asof.write_features(dc, mc, pd.read_parquet(out / "universe.parquet"), rf, names, out / "features_asof.parquet")
    del dc, mc

    us = "holfill_dir" not in cfg["output"]
    print("returns_history", flush=True)
    if us:
        raw = ROOT / cfg["raw"]["dir"]
        dirs = [raw / "crsp_daily", raw / "crsp_daily_extra"]
        h = asof.us_history(dirs, read_ret(out))
        h.to_parquet(out / "returns_history.parquet", index=False, compression="zstd")
        print(f"  {out}: {len(h):,} rows, {h['sec_id'].nunique():,} names", flush=True)
        return
    detail = pd.read_parquet(ins / "returns_detail_daily.parquet", columns=["gvkey", "iid", "date", "ret"])
    first = pd.read_parquet(ins / "universe_asof.parquet", columns=["month", "gvkey", "iid"])
    fill = ROOT / cfg["output"]["holfill_dir"]
    for d in (out, fill):
        h = asof.europe_history(detail, first, read_ret(d))
        h.to_parquet(d / "returns_history.parquet", index=False, compression="zstd")
        print(f"  {d}: {len(h):,} rows, {h['sec_id'].nunique():,} names", flush=True)
    shutil.copy2(out / "features_asof.parquet", fill / "features_asof.parquet")


if __name__ == "__main__":
    main()
