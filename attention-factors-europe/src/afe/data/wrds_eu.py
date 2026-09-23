"""Stage 1 of the European build: pull the raw tables from WRDS once, store as parquet.

Same split as `wrds_us`: nothing here computes a characteristic. The universe (which
companies, which listing) comes from the step-1 cap table built by
`compustat_global` / `scripts/build_eu_mktcap.py`; this module pulls the daily market
data, annual fundamentals, FX and the JKP characteristics for those companies only,
so the transfer is ~1/10 of the full country files.

Tables (names as on WRDS PostgreSQL):
  comp.g_secd            Compustat Global security daily: close, high, low, open, split
                         factor (ajexdi), total-return factor (trfd), shares, volume,
                         dividends, quotation currency. Total return in local currency is
                         (prccd/ajexdi*trfd)_t / (prccd/ajexdi*trfd)_{t-1} - 1, the JKP
                         definition. No delisting return exists in Global.
  comp.g_funda           annual fundamentals, datafmt HIST_STD / popsrc I / consol C,
                         BOTH industrial (INDL) and financial-services (FS) formats:
                         unlike North America, Global files banks and insurers under FS
                         only (Santander, ABN Amro, AIB, Generali), 346 of the 1,543
                         ever-members. `indfmt` is a column; prefer INDL where both exist
                         (23 companies). Reporting currency in `curcd`. Items mirror
                         `wrds_us.FUNDA_ITEMS`; NA-only items (pstkl, pstkrv, ni, gp, xad,
                         csho, ajex, prcc_f) are absent in Global and have substitutes
                         (nicon, revt-cogs, cshoi, ajexi, the daily close).
  comp.g_exrt_dly        daily cross rates, units of currency per GBP.
  contrib.global_factor  Jensen-Kelly-Pedersen characteristics, monthly, one row per
                         security (gvkey, iid); `ret`/`me` in USD, `ret_local`/`prc_local`
                         in quotation currency. Pulled in full width (444 columns) so the
                         US/EU characteristic intersection can be decided offline.
  contrib.factors_daily  country-level daily FF-style factors and rf (Germany, France,
                         Italy, Netherlands, Spain, UK, ... ) to 2018-02 only: a
                         cross-check, not the factor series of the build.
  ff.factors_daily       US factors and the US 1-month T-bill, the rf JKP use worldwide.
NUMERIC columns are cast to float8 in SQL (psycopg2 returns Decimal otherwise).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

# comp.g_secd columns for the daily file. `monthend` and `prcstd` (10 = traded close,
# 5 = carried price) are kept so downstream can filter stale days.
SECD_DAILY_NUMERIC = ["prccd", "prchd", "prcld", "prcod", "ajexdi", "trfd", "cshoc", "cshtrd",
                      "qunit", "div", "divd", "divsp", "cheqv", "prcstd", "exchg", "monthend"]
SECD_DAILY_TEXT = ["gvkey", "iid", "datadate", "curcdd", "curcddv"]

# comp.g_funda items: the US list where Global has the item, plus Global's substitutes
# and the cash-flow / payout items JKP need. Adding an item here makes it available.
G_FUNDA_ITEMS = [
    # balance sheet
    "at", "lt", "seq", "teq", "ceq", "pstk", "txditc", "txdb", "che", "ivao", "act", "lct",
    "dlc", "dltt", "mib", "txp", "wcap", "wcapch", "ppegt", "ppent", "invt", "rect", "ap",
    "intan", "gdwl",
    # income statement
    "sale", "revt", "cogs", "xsga", "xopr", "xrd", "xint", "oiadp", "oibdp", "ebit", "ebitda",
    "pi", "txt", "ib", "ibc", "nicon", "spi", "xido", "dp", "emp",
    # cash flow / payout / shares
    "capx", "oancf", "fincf", "ivncf", "aqc", "sstk", "prstkc", "dltis", "dltr", "dlcch",
    "chech", "dvc", "dvt", "dv", "cshoi", "cshpria", "ajexi",
]
G_FUNDA_FILTER = "indfmt in ('INDL', 'FS') and datafmt = 'HIST_STD' and popsrc = 'I' and consol = 'C'"

JKP_KEY_COLS = ["id", "gvkey", "iid", "excntry", "eom", "date", "curcd", "fx", "me", "me_company",
                "prc", "prc_local", "ret", "ret_local", "ret_exc", "primary_sec", "obs_main",
                "common", "exch_main", "comp_tpci", "comp_exchg", "source_crsp"]


def _pairs_sql(pairs: list[tuple[str, str]]) -> str:
    return ", ".join(f"('{g}', '{i}')" for g, i in pairs)


def _gvkeys_sql(gvkeys) -> str:
    return "(" + ", ".join(f"'{g}'" for g in gvkeys) + ")"


def fetch_secd_daily_year(db, year: int, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """One calendar year of comp.g_secd for the given (gvkey, iid) listings."""
    num = ", ".join(f"{c}::float8 as {c}" for c in SECD_DAILY_NUMERIC)
    sql = f"""
        select {", ".join(SECD_DAILY_TEXT)}, {num}
        from comp.g_secd
        where (gvkey, iid) in ({_pairs_sql(pairs)})
          and datadate between '{year}-01-01' and '{year}-12-31'
    """
    df = db.raw_sql(sql, date_cols=["datadate"])
    return _compact(df)


def fetch_g_funda(db, gvkeys, start_year: int, end_year: int) -> pd.DataFrame:
    items = ", ".join(f"{c}::float8 as {c}" for c in G_FUNDA_ITEMS)
    sql = f"""
        select gvkey, datadate, fyear, fyr, curcd, indfmt, {items}
        from comp.g_funda
        where {G_FUNDA_FILTER} and gvkey in {_gvkeys_sql(gvkeys)}
          and datadate between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(sql, date_cols=["datadate"])


def fetch_jkp_year(db, year: int, gvkeys, countries: list[str]) -> pd.DataFrame:
    """One year of contrib.global_factor, all columns, for the given companies. Float
    columns are narrowed to float32: 444 columns x 300k rows must stay small."""
    sql = f"""
        select * from contrib.global_factor
        where excntry in ({", ".join(f"'{c}'" for c in countries)})
          and gvkey in {_gvkeys_sql(gvkeys)}
          and eom between '{year}-01-01' and '{year}-12-31'
    """
    df = db.raw_sql(sql, date_cols=["eom", "date"])
    keep64 = ("me", "me_company", "prc", "prc_local", "ret", "ret_local", "ret_exc", "fx")
    for c in df.columns:
        if str(df[c].dtype) == "float64" and c not in keep64:
            df[c] = df[c].astype("float32")
    return df


def fetch_contrib_factors_daily(db) -> pd.DataFrame:
    return db.raw_sql("select * from contrib.factors_daily", date_cols=["date"])


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("prcstd", "exchg", "monthend"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int16")
        elif str(df[c].dtype) == "float64" and c not in ("cshoc",):
            df[c] = df[c].astype("float32")
    return df


def pull_all(db, raw_dir: Path, pairs: list[tuple[str, str]], gvkeys, countries: list[str],
             daily_start: int, daily_end: int, funda_start: int, jkp_start: int,
             fetch_fx, fetch_ff, fx_start: int | None = None, log=print) -> dict:
    """Run every fetch, write parquet, return the manifest (also written to raw_dir).
    Yearly files that exist are skipped, so a dropped connection costs one year."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "secd_daily").mkdir(exist_ok=True)
    (raw_dir / "jkp").mkdir(exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tables": {}}
    manifest.update({"pulled_at": dt.datetime.now().isoformat(timespec="seconds"),
                     "daily_start": daily_start, "daily_end": daily_end, "n_listings": len(pairs),
                     "n_companies": len(gvkeys), "countries": countries})

    def save(name, df, path):
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": int(len(df)), "file": str(path.relative_to(raw_dir))}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log(f"  {name}: {len(df):,} rows -> {path.name}", flush=True)

    def yearly(name, years, fetch):
        total = 0
        for year in years:
            path = raw_dir / name / f"{year}.parquet"
            if path.exists():
                total += pd.read_parquet(path, columns=["gvkey"]).shape[0]
                log(f"  {name} {year}: exists, skipped", flush=True)
                continue
            t0 = dt.datetime.now()
            df = fetch(year)
            df.to_parquet(path, index=False)
            total += len(df)
            log(f"  {name} {year}: {len(df):>9,} rows in {(dt.datetime.now() - t0).seconds}s", flush=True)
            manifest["tables"][name] = {"rows": total, "file": f"{name}/<year>.parquet"}
            manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest["tables"][name] = {"rows": total, "file": f"{name}/<year>.parquet"}

    if not (raw_dir / "g_funda.parquet").exists():
        save("g_funda", fetch_g_funda(db, gvkeys, funda_start, daily_end), raw_dir / "g_funda.parquet")
    if not (raw_dir / "contrib_factors_daily.parquet").exists():
        save("contrib_factors_daily", fetch_contrib_factors_daily(db), raw_dir / "contrib_factors_daily.parquet")
    if not (raw_dir / "ff_daily.parquet").exists():
        ffd, ffm = fetch_ff(db)
        save("ff_daily", ffd, raw_dir / "ff_daily.parquet")
        save("ff_monthly", ffm, raw_dir / "ff_monthly.parquet")

    yearly("secd_daily", range(daily_start, daily_end + 1), lambda y: fetch_secd_daily_year(db, y, pairs))
    yearly("jkp", range(jkp_start, daily_end + 1), lambda y: fetch_jkp_year(db, y, gvkeys, countries))

    # FX last: the currency list is the union of what the daily file and funda carry.
    ccy = set()
    for p in (raw_dir / "secd_daily").glob("*.parquet"):
        ccy |= set(pd.read_parquet(p, columns=["curcdd"])["curcdd"].dropna().unique())
    ccy |= set(pd.read_parquet(raw_dir / "g_funda.parquet", columns=["curcd"])["curcd"].dropna().unique())
    ccy |= {"EUR", "USD", "GBP"}
    save("fx_daily", fetch_fx(db, sorted(ccy), fx_start or daily_start, daily_end), raw_dir / "fx_daily.parquet")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
