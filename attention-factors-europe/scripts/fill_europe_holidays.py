"""Stage 2b of the European builds: the holiday-filled schema tables the headline runs read.

    python scripts/fill_europe_holidays.py --config configs/europe17_data.yaml

Reads output.dir (returns, universe, features, fx_to_numeraire_daily) and
output.inspect_dir/company_daily.parquet (traded closes) of a finished stage 2, writes
output.holfill_dir: returns.parquet with gaps of at most returns.holiday_fill_max_gap pooled
days filled and a `traded` column (src/afe/data/europe_holidays.py), universe.parquet and
features.parquet copied unchanged. output.dir itself stays the unfilled build.
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

from afe import schemas  # noqa: E402
from afe.data.europe_holidays import fill_holiday_gaps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    src, ins = ROOT / cfg["output"]["dir"], ROOT / cfg["output"]["inspect_dir"]
    dst, max_gap = ROOT / cfg["output"]["holfill_dir"], int(cfg["returns"]["holiday_fill_max_gap"])
    dst.mkdir(parents=True, exist_ok=True)

    ret = pd.read_parquet(src / "returns.parquet")
    traded = pd.read_parquet(ins / "company_daily.parquet", columns=["gvkey", "date", "traded"])
    out, st = fill_holiday_gaps(ret, pd.read_parquet(src / "universe.parquet"),
                                pd.read_parquet(src / "fx_to_numeraire_daily.parquet"),
                                traded.rename(columns={"gvkey": "sec_id"}), max_gap)
    schemas.validate(out, schemas.RETURNS)
    out.to_parquet(dst / "returns.parquet", index=False)
    for f in ("universe.parquet", "features.parquet"):
        shutil.copy2(src / f, dst / f)

    report = (f"{cfg['market']} holiday fill from {cfg['output']['dir']}, max gap {max_gap} pooled days\n"
              f"filled member-days {st['filled_rows']:,} in {st['gaps']:,} gaps (no FX on {st['no_fx']}, left at 0)\n"
              f"returns rows {len(ret):,} -> {len(out):,}; traded False {st['not_traded_rows']:,} "
              f"({st['not_traded_rows'] / len(out):.2%}); next-day returns adjusted {st['next_day_adjusted']:,}\n"
              f"filled by year: {st['filled_by_year']}\n")
    (dst / "build_report.txt").write_text(report)
    print(report + f"wrote {dst}")


if __name__ == "__main__":
    main()
