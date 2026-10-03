"""Step 1c of the European builds: the raw WRDS tables for the union of the versions' lines.

    python scripts/fetch_europe_raw.py --configs configs/europe17_data.yaml configs/europe12_data.yaml
                                       [--tables daily funda fx jkp] [--add-lines]

Every config's output.inspect_dir must hold listings.csv (build_europe_universe.py). The
union of those lines and companies is pulled once into raw.dir of the first config
(the configs share it). Resumable: a year on disk is skipped. `--tables` restricts the run
so the daily file and JKP can be pulled by two processes at once (JKP scans are slow).
`--add-lines` (daily only): when a rebuilt universe prices a company by a line the pull
does not hold yet, fetch those lines for every year and append them to the year files;
listings.csv then lists everything pulled (old union plus the new lines).
  secd_daily/<year>.parquet   comp.g_secd, all fields of wrds_eu.SECD_DAILY_*, per listing-day
  g_funda.parquet             comp.g_funda INDL + FS, HIST_STD / I / C, wrds_eu.G_FUNDA_ITEMS
  jkp/<year>.parquet          contrib.global_factor, all 444 columns, every line of the companies
  fx_daily.parquet            comp.g_exrt_dly for every currency in secd_daily and g_funda
  listings.csv, members.csv   the union that was pulled
Nothing is converted or computed here (docs/eu_raw_data.md, "Nothing in raw/ is converted").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import compustat_global as cg  # noqa: E402
from afe.data import compustat_us as cu  # noqa: E402
from afe.data import wrds_eu  # noqa: E402


def union(cfgs) -> tuple[pd.DataFrame, list[str], list[str]]:
    lines = pd.concat([pd.read_csv(ROOT / c["output"]["inspect_dir"] / "listings.csv", dtype=str) for c in cfgs])
    lines = lines.drop_duplicates().sort_values(["gvkey", "iid"]).reset_index(drop=True)
    countries = sorted({x for c in cfgs for x in c["countries"]})
    return lines, sorted(lines["gvkey"].unique()), countries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tables", nargs="+", default=["funda", "daily", "jkp", "fx"],
                    choices=["funda", "daily", "jkp", "fx"])
    ap.add_argument("--add-lines", action="store_true",
                    help="daily only: append the union's lines that raw/listings.csv does not hold yet")
    args = ap.parse_args()
    cfgs = [yaml.safe_load(Path(p).read_text()) for p in args.configs]
    r = cfgs[0]["raw"]
    if any(c["raw"]["dir"] != r["dir"] for c in cfgs):
        raise ValueError("the configs must share raw.dir")
    raw = ROOT / r["dir"]
    for sub in ("secd_daily", "jkp"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    lines, gvkeys, countries = union(cfgs)
    if args.add_lines:
        if args.tables != ["daily"]:
            raise SystemExit("--add-lines appends daily rows only: pass --tables daily")
        held = pd.read_csv(raw / "listings.csv", dtype=str)
        new = lines.merge(held, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
        add_lines(raw, [tuple(p) for p in new.itertuples(index=False)], int(r["daily_start_year"]), int(r["daily_end_year"]))
        pd.concat([held, new]).sort_values(["gvkey", "iid"]).to_csv(raw / "listings.csv", index=False)
        return
    lines.to_csv(raw / "listings.csv", index=False)
    pd.Series(gvkeys, name="gvkey").to_csv(raw / "members.csv", index=False)
    pairs = list(lines.itertuples(index=False, name=None))
    y0, y1 = int(r["daily_start_year"]), int(r["daily_end_year"])
    print(f"union: {len(pairs):,} lines, {len(gvkeys):,} companies, countries {countries}; tables {args.tables}", flush=True)

    manifest_path = raw / f"manifest_{'_'.join(args.tables)}.json"
    manifest = {"pulled_at": dt.datetime.now().isoformat(timespec="seconds"), "n_listings": len(pairs),
                "n_companies": len(gvkeys), "countries": countries, "configs": [str(p) for p in args.configs], "tables": {}}

    def yearly(name, years, fetch):
        total = 0
        for y in years:
            path = raw / name / f"{y}.parquet"
            if path.exists():
                total += pq_rows(path)
                print(f"  {name} {y}: exists, skipped", flush=True)
                continue
            t0 = time.time()
            df = fetch(y)
            df.to_parquet(path, index=False)
            total += len(df)
            print(f"  {name} {y}: {len(df):>9,} rows in {time.time() - t0:.0f}s", flush=True)
        manifest["tables"][name] = {"rows": total, "file": f"{name}/<year>.parquet"}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    db = cu.connect(ROOT)
    try:
        if "funda" in args.tables:
            f = wrds_eu.fetch_g_funda(db, gvkeys, int(r["funda_start_year"]), y1)
            f.to_parquet(raw / "g_funda.parquet", index=False)
            manifest["tables"]["g_funda"] = {"rows": len(f)}
            print(f"  g_funda: {len(f):,} rows, {f['gvkey'].nunique():,} companies", flush=True)
        if "daily" in args.tables:
            yearly("secd_daily", range(y0, y1 + 1), lambda y: wrds_eu.fetch_secd_daily_year(db, y, pairs))
        if "jkp" in args.tables:
            # contrib.global_factor has no index a date filter can use: every query scans the
            # table (~4 min), so years are fetched in chunks and split into yearly files.
            years = [y for y in range(int(r["jkp_start_year"]), y1 + 1) if not (raw / "jkp" / f"{y}.parquet").exists()]
            step = int(r.get("jkp_chunk_years", 7))
            for i in range(0, len(years), step):
                chunk = years[i:i + step]
                t0 = time.time()
                df = fetch_jkp_years(db, chunk[0], chunk[-1], gvkeys, countries)
                for y in chunk:
                    df[df["eom"].dt.year == y].to_parquet(raw / "jkp" / f"{y}.parquet", index=False)
                print(f"  jkp {chunk[0]}-{chunk[-1]}: {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)
            manifest["tables"]["jkp"] = {"rows": sum(pq_rows(p) for p in (raw / "jkp").glob("*.parquet")),
                                         "file": "jkp/<year>.parquet"}
        if "fx" in args.tables:
            ccy = {"EUR", "USD", "GBP"}
            for p in (raw / "secd_daily").glob("*.parquet"):
                ccy |= set(pd.read_parquet(p, columns=["curcdd"])["curcdd"].dropna().unique())
            if (raw / "g_funda.parquet").exists():
                ccy |= set(pd.read_parquet(raw / "g_funda.parquet", columns=["curcd"])["curcd"].dropna().unique())
            fx = cg.fetch_fx(db, sorted(ccy), int(r.get("fx_start_year", y0)), y1 + 1)
            fx.to_parquet(raw / "fx_daily.parquet", index=False)
            manifest["tables"]["fx_daily"] = {"rows": len(fx), "currencies": sorted(ccy)}
    finally:
        db.close()
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print({k: v["rows"] for k, v in manifest["tables"].items()})


def add_lines(raw: Path, pairs: list[tuple[str, str]], y0: int, y1: int) -> None:
    """Append every year of comp.g_secd for `pairs` to raw/secd_daily/<year>.parquet."""
    print(f"adding {len(pairs)} lines to the daily pull: {pairs}", flush=True)
    if not pairs:
        return
    db = cu.connect(ROOT)
    try:
        for y in range(y0, y1 + 1):
            path = raw / "secd_daily" / f"{y}.parquet"
            add = wrds_eu.fetch_secd_daily_year(db, y, pairs)
            df = add
            if path.exists():
                df = pd.concat([pd.read_parquet(path), add], ignore_index=True).drop_duplicates(
                    ["gvkey", "iid", "datadate"], keep="first")
            df.to_parquet(path, index=False)
            print(f"  secd_daily {y}: {len(add):,} rows fetched, {len(df):,} in the file", flush=True)
    finally:
        db.close()


def pq_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def fetch_jkp_years(db, y0: int, y1: int, gvkeys, countries) -> pd.DataFrame:
    """wrds_eu.fetch_jkp_year over several years in one query (same columns, same float32 narrowing)."""
    sql = f"""
        select * from contrib.global_factor
        where excntry in ({", ".join(f"'{c}'" for c in countries)})
          and gvkey in ({", ".join(f"'{g}'" for g in gvkeys)})
          and eom between '{y0}-01-01' and '{y1}-12-31'
    """
    df = db.raw_sql(sql, date_cols=["eom", "date"])
    keep64 = ("me", "me_company", "prc", "prc_local", "ret", "ret_local", "ret_exc", "fx")
    for c in df.columns:
        if str(df[c].dtype) == "float64" and c not in keep64:
            df[c] = df[c].astype("float32")
    return df


if __name__ == "__main__":
    main()
