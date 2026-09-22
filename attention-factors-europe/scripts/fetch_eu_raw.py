"""Stage 1 of the European build: the top-N universe and the raw WRDS tables behind it.

    python scripts/fetch_eu_raw.py [--config configs/eu_data.yaml] [--universe-only]

Step 1 (offline, from the cap table of build_eu_mktcap.py):
  data/eu/private/raw/universe_top500_monthly.parquet/.csv   AS-OF: row m = the N largest eligible
                                                      companies by cap at the END of m, with
                                                      the listing (iid) that priced the cap
  data/eu/private/raw/universe_top500_gvkey_wide.csv          months x ranks of gvkey
  data/eu/private/raw/listings.csv                            the (gvkey, iid) lines pulled daily
Step 2 (WRDS, resumable per year): secd_daily/<year>.parquet, g_funda.parquet,
  jkp/<year>.parquet, fx_daily.parquet, contrib_factors_daily.parquet, ff_daily.parquet,
  ff_monthly.parquet, manifest.json. See src/afe/data/wrds_eu.py and docs/eu_raw_data.md.

The point-in-time lag (universe for trading month m+1 = row m) is applied downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe.data import compustat_global as cg  # noqa: E402
from afe.data import compustat_us as cu  # noqa: E402
from afe.data import wrds_eu, wrds_us  # noqa: E402


def build_universe(cfg) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Top-N AS-OF table, the listings to pull, and the ever-member gvkeys."""
    u = cfg["universe"]
    mcap = pd.read_parquet(ROOT / u["mktcap_table"])
    elig = mcap[mcap["eligible"]].reset_index(drop=True)
    top = cu.top_n_by_month(elig, int(u["size"]), str(u["start"]), str(u["end"]))
    cols = ["datadate", "cap_rank", "gvkey", "mktcap", "country", "iid", "exchg", "isin", "conm",
            "n_classes", "n_listings", "turnover", "curcdd", "prccd", "cshoc", "fic", "sic", "price_date"]
    top = top[cols].reset_index(drop=True)
    members = sorted(top["gvkey"].unique())
    # every home line of an ever-member, over ALL months of the cap table
    lines = (mcap[mcap["gvkey"].isin(members)][["gvkey", "iid"]].drop_duplicates()
             .sort_values(["gvkey", "iid"]).reset_index(drop=True))
    return top, lines, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "eu_data.yaml")
    ap.add_argument("--universe-only", action="store_true", help="write the universe files, no WRDS")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["raw"]["dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg["universe"]["size"])

    top, lines, members = build_universe(cfg)
    tag = f"top{n}"
    top.to_parquet(raw_dir / f"universe_{tag}_monthly.parquet", index=False)
    top.to_csv(raw_dir / f"universe_{tag}_monthly.csv", index=False)
    cu.to_wide(top, "gvkey").to_csv(raw_dir / f"universe_{tag}_gvkey_wide.csv")
    lines.to_csv(raw_dir / "listings.csv", index=False)
    print(f"universe: {top['datadate'].nunique()} months x {n}, {len(members):,} ever-members, "
          f"{len(lines):,} listings to pull; countries {sorted(top['country'].unique())}", flush=True)
    if args.universe_only:
        return

    countries = sorted(pd.read_parquet(ROOT / cfg["universe"]["mktcap_table"], columns=["country"])["country"].unique())
    r = cfg["raw"]
    db = cu.connect(ROOT)
    try:
        manifest = wrds_eu.pull_all(
            db, raw_dir, pairs=list(lines.itertuples(index=False, name=None)), gvkeys=members,
            countries=list(countries), daily_start=int(r["daily_start_year"]), daily_end=int(r["daily_end_year"]),
            funda_start=int(r["funda_start_year"]), jkp_start=int(r["jkp_start_year"]),
            fetch_fx=cg.fetch_fx, fetch_ff=wrds_us.fetch_ff_factors,
            fx_start=int(r.get("fx_start_year", r["daily_start_year"])),
        )
    finally:
        db.close()
    print({k: v["rows"] for k, v in manifest["tables"].items()})


if __name__ == "__main__":
    main()
