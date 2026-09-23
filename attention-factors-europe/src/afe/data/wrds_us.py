"""Stage 1 of the US build: pull the raw tables from WRDS once and store them as parquet.

Nothing here computes a characteristic. The point of the split is that the expensive,
licence-bound part (an hour of transfer for ~50M daily rows) runs once, and the feature
logic in `characteristics.py` / `build_us.py` can be changed and re-run offline in
minutes against the same frozen inputs. A manifest records what was pulled and when.

Tables (all names as on WRDS PostgreSQL):
  crsp.dsf_v2              daily stock file, CIZ format. The delisting return is folded
                           into dlyret on the delisting day (dlydelflg = 'Y'), BUT on
                           that row CRSP blanks sharetype / securitytype / securitysubtype /
                           conditionaltype and sets tradingstatusflg to 'D', so the
                           CRSP_COMMON filter below drops it. Those rows are pulled
                           separately (crsp_delisting.parquet) and merged into the pool's
                           daily rows in stage 2. Found 2026-09-22: the first build had
                           no delisting returns at all (Lehman -60% on 2008-09-18).
  crsp.msf_v2              monthly stock file, CIZ format.
  crsp.stksecurityinfohist security header history (names, delisting codes).
  crsp.ccmxpf_lnkhist      CRSP-Compustat link (permno <-> gvkey with date ranges).
  comp.funda               Compustat annual fundamentals, the items the characteristics need.
  ff.factors_daily/monthly Fama-French three factors + momentum + risk-free rate.

The CRSP pulls keep every share class and issuer type that is a plain common share
(sharetype NS, securitytype EQTY, securitysubtype COM, regular-way, actively trading)
and store the remaining CIZ flags (usincflg, issuertype, primaryexch) as columns, so the
universe filter is a stage-2 decision that can be revisited without a new pull.
NUMERIC columns are cast to float8 in SQL: psycopg2 otherwise returns Decimal objects,
which is roughly 10x slower to transfer and convert.

Extra pull for a paper-style universe (`pull_extra`, added 2026-09-23). The paper's
Figures 3-4 name depositary receipts (Toyota, Mizuho, NatWest), MLP units (Energy
Transfer) and REIT trusts, none of which is a plain common share. Same CRSP tables with
the CRSP_EXTRA filter, into crsp_monthly_extra.parquet and crsp_daily_extra/<year>, plus:
  comp.secm                Compustat NA security monthly for the receipts' issues. CRSP's
                           cap of a receipt counts only receipts outstanding (Mizuho
                           2020-12: $0.2bn); secm has the underlying shares of the class
                           (cshom, from 1998-04) and the receipt ratio (adrrm, underlying
                           shares per receipt, monthly), so cshom * prccm / adrrm is the
                           class's cap in USD ($32bn).
  comp.funda currency      curcd (USD or CAD: North America reports Canadian filers in CAD,
                           receipt issuers in USD), native currency and receipt ratio for
                           every row of comp_funda.parquet, same key.
  comp.g_exrt_dly          daily GBP-based cross rates for the currencies above.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

# Plain common shares, trading regular-way and active. The remaining flags are kept as
# columns; see the module docstring.
CRSP_COMMON = ("sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM' "
               "and conditionaltype = 'RW' and tradingstatusflg = 'A'")

# The other share types an "all stocks" universe holds: depositary receipts (AD), units,
# i.e. MLP units and paired shares (UG), shares of beneficial interest, mostly REIT trusts
# (SB), certificates (CE). securitysubtype 'COM' keeps out the closed-end funds and ETFs
# that CRSP files under the same share types.
CRSP_EXTRA = ("sharetype in ('AD', 'UG', 'SB', 'CE') and securitytype = 'EQTY' "
              "and securitysubtype = 'COM' and conditionaltype = 'RW' and tradingstatusflg = 'A'")

# Compustat items behind the 39 characteristics (docs/us_characteristics.md), plus the
# identifiers needed to align them in time. Adding an item here is the only change
# needed to make it available to a new characteristic.
FUNDA_ITEMS = [
    "at", "lt", "seq", "ceq", "pstk", "pstkl", "pstkrv", "txditc", "txdb", "che", "ivao",
    "act", "lct", "dlc", "dltt", "mib", "txp", "wcap", "wcapch", "capx", "ni", "ib", "dp",
    "gp", "sale", "revt", "cogs", "xsga", "xrd", "xad", "xint", "oiadp", "ppegt", "invt",
    "csho", "ajex", "prcc_f",
]
FUNDA_FILTER = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"


def fetch_crsp_daily_year(db, year: int, where: str = CRSP_COMMON) -> pd.DataFrame:
    """One calendar year of crsp.dsf_v2 for plain common shares (or `where`)."""
    sql = f"""
        select permno, permco, dlycaldt as date,
               dlyret::float8 as ret, dlyprc::float8 as prc, dlyvol::float8 as vol,
               dlybid::float8 as bid, dlyask::float8 as ask, dlyhigh::float8 as high,
               dlycap::float8 as cap, shrout, dlycumfacpr::float8 as cumfacpr,
               usincflg, issuertype, primaryexch, siccd, sharetype
        from crsp.dsf_v2
        where {where} and dlycaldt between '{year}-01-01' and '{year}-12-31'
    """
    df = db.raw_sql(sql, date_cols=["date"])
    return _compact(df)


def fetch_crsp_delisting(db, start_year: int, end_year: int) -> pd.DataFrame:
    """Delisting-day rows of crsp.dsf_v2 (dlydelflg = 'Y'), every security, same
    columns as the daily file plus the flag and the previous trading date. ~24k rows
    over 1986-2025; stage 2 keeps the ones of pool permnos."""
    sql = f"""
        select permno, permco, dlycaldt as date,
               dlyret::float8 as ret, dlyprc::float8 as prc, dlyvol::float8 as vol,
               dlybid::float8 as bid, dlyask::float8 as ask, dlyhigh::float8 as high,
               dlycap::float8 as cap, shrout, dlycumfacpr::float8 as cumfacpr,
               usincflg, issuertype, primaryexch, siccd, dlydelflg, dlyprevdt
        from crsp.dsf_v2
        where dlydelflg = 'Y' and dlycaldt between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return _compact(db.raw_sql(sql, date_cols=["date", "dlyprevdt"]))


def fetch_crsp_monthly(db, start_year: int, end_year: int, where: str = CRSP_COMMON) -> pd.DataFrame:
    sql = f"""
        select permno, permco, mthcaldt as date,
               mthret::float8 as ret, mthretx::float8 as retx, mthprc::float8 as prc,
               mthvol::float8 as vol, mthcap::float8 as cap, shrout,
               mthcumfacpr::float8 as cumfacpr,
               usincflg, issuertype, primaryexch, siccd, sharetype
        from crsp.msf_v2
        where {where} and mthcaldt between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return _compact(db.raw_sql(sql, date_cols=["date"]))


def fetch_comp_secm(db, pairs: list[tuple[str, str]], start_year: int, end_year: int) -> pd.DataFrame:
    """comp.secm for the given (gvkey, iid) issues: underlying shares of the class (cshom),
    close (prccm, USD for a receipt), receipt ratio (adrrm), split factor, currency."""
    keys = ", ".join(f"('{g}', '{i}')" for g, i in pairs)
    sql = f"""
        select gvkey, iid, datadate, cshom::float8 as cshom, prccm::float8 as prccm,
               adrrm::float8 as adrrm, ajexm::float8 as ajexm, curcdm, tpci, exchg
        from comp.secm
        where (gvkey, iid) in ({keys}) and datadate between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(sql, date_cols=["datadate"])


def fetch_funda_currency(db, start_year: int, end_year: int) -> pd.DataFrame:
    """Reporting currency, native currency and receipt ratio of every comp.funda row that
    fetch_compustat_annual pulls (same filter and key), so comp_funda.parquet can be
    converted to USD without re-pulling it."""
    sql = f"""
        select gvkey, datadate, curcd, curncd, adrr::float8 as adrr
        from comp.funda
        where {FUNDA_FILTER} and datadate between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(sql, date_cols=["datadate"])


def fetch_security_info(db) -> pd.DataFrame:
    sql = """
        select permno, permco, secinfostartdt, secinfoenddt, securitynm, ticker, shareclass,
               primaryexch, usincflg, issuertype, securitytype, securitysubtype, sharetype,
               delactiontype, delstatustype, delreasontype, siccd
        from crsp.stksecurityinfohist
    """
    return db.raw_sql(sql, date_cols=["secinfostartdt", "secinfoenddt"])


def fetch_ccm_link(db) -> pd.DataFrame:
    """Company-level links only (LU, LC). P/C are the primary security of the company,
    J a secondary share class; all three map a permno to the company's gvkey."""
    sql = """
        select gvkey, linkprim, liid, linktype, lpermno::int as permno, lpermco::int as permco,
               linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU', 'LC') and linkprim in ('P', 'C', 'J') and lpermno is not null
    """
    return db.raw_sql(sql, date_cols=["linkdt", "linkenddt"])


def fetch_compustat_annual(db, start_year: int, end_year: int) -> pd.DataFrame:
    items = ", ".join(f"{c}::float8 as {c}" for c in FUNDA_ITEMS)
    sql = f"""
        select gvkey, datadate, fyear, fyr, {items}
        from comp.funda
        where {FUNDA_FILTER} and datadate between '{start_year}-01-01' and '{end_year}-12-31'
    """
    return db.raw_sql(sql, date_cols=["datadate"])


def fetch_ff_factors(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = db.raw_sql("select date, mktrf, smb, hml, rf, umd from ff.factors_daily", date_cols=["date"])
    monthly = db.raw_sql("select date, mktrf, smb, hml, rf, umd from ff.factors_monthly", date_cols=["date"])
    return daily, monthly


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow dtypes: 50M daily rows must fit on a 16GB laptop."""
    for c in df.columns:
        if c in ("permno", "permco", "siccd", "shrout"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
        elif str(df[c].dtype) == "float64":
            df[c] = df[c].astype("float32")
    return df


def pull_all(db, raw_dir: Path, start_year: int, end_year: int, log=print) -> dict:
    """Run every fetch, write parquet, return the manifest (also written to raw_dir)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "crsp_daily").mkdir(exist_ok=True)
    manifest = {"pulled_at": dt.datetime.now().isoformat(timespec="seconds"),
                "start_year": start_year, "end_year": end_year, "tables": {}}

    def save(name: str, df: pd.DataFrame, path: Path):
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": int(len(df)), "file": str(path.relative_to(raw_dir))}
        log(f"  {name}: {len(df):,} rows -> {path.name}")

    save("crsp_monthly", fetch_crsp_monthly(db, start_year, end_year), raw_dir / "crsp_monthly.parquet")
    save("crsp_secinfo", fetch_security_info(db), raw_dir / "crsp_secinfo.parquet")
    save("crsp_delisting", fetch_crsp_delisting(db, start_year, end_year), raw_dir / "crsp_delisting.parquet")
    save("ccm_link", fetch_ccm_link(db), raw_dir / "ccm_link.parquet")
    save("comp_funda", fetch_compustat_annual(db, start_year - 2, end_year), raw_dir / "comp_funda.parquet")
    ffd, ffm = fetch_ff_factors(db)
    save("ff_daily", ffd, raw_dir / "ff_daily.parquet")
    save("ff_monthly", ffm, raw_dir / "ff_monthly.parquet")

    total = 0
    for year in range(start_year, end_year + 1):
        path = raw_dir / "crsp_daily" / f"{year}.parquet"
        if path.exists():  # resumable: a killed pull picks up where it stopped
            total += pd.read_parquet(path, columns=["permno"]).shape[0]
            log(f"  crsp_daily {year}: exists, skipped")
            continue
        t0 = dt.datetime.now()
        df = fetch_crsp_daily_year(db, year)
        df.to_parquet(path, index=False)
        total += len(df)
        log(f"  crsp_daily {year}: {len(df):>9,} rows in {(dt.datetime.now() - t0).seconds}s")
    manifest["tables"]["crsp_daily"] = {"rows": total, "file": "crsp_daily/<year>.parquet"}

    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def pull_extra(db, raw_dir: Path, start_year: int, end_year: int, log=print) -> dict:
    """The tables a paper-style universe needs on top of pull_all (module docstring).
    Existing files are skipped, so a dropped connection costs one daily year. Needs
    ccm_link.parquet from pull_all: the receipts' Compustat issues come from the link."""
    from afe.data import compustat_global

    (raw_dir / "crsp_daily_extra").mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest_extra.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tables": {}}
    manifest.update({"pulled_at": dt.datetime.now().isoformat(timespec="seconds"), "crsp_filter": CRSP_EXTRA,
                     "start_year": start_year, "end_year": end_year})

    def save(name: str, fetch):
        path = raw_dir / f"{name}.parquet"
        if path.exists():
            log(f"  {name}: exists, skipped", flush=True)
            return pd.read_parquet(path)
        df = fetch()
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": int(len(df)), "file": path.name}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log(f"  {name}: {len(df):,} rows -> {path.name}", flush=True)
        return df

    monthly = save("crsp_monthly_extra", lambda: fetch_crsp_monthly(db, start_year, end_year, CRSP_EXTRA))
    total = 0
    for year in range(start_year, end_year + 1):
        path = raw_dir / "crsp_daily_extra" / f"{year}.parquet"
        if path.exists():
            total += pd.read_parquet(path, columns=["permno"]).shape[0]
            continue
        t0 = dt.datetime.now()
        df = fetch_crsp_daily_year(db, year, CRSP_EXTRA)
        df.to_parquet(path, index=False)
        total += len(df)
        log(f"  crsp_daily_extra {year}: {len(df):>8,} rows in {(dt.datetime.now() - t0).seconds}s", flush=True)
    manifest["tables"]["crsp_daily_extra"] = {"rows": total, "file": "crsp_daily_extra/<year>.parquet"}

    link = pd.read_parquet(raw_dir / "ccm_link.parquet")
    receipts = monthly.loc[monthly["sharetype"] == "AD", "permno"].astype("int64").unique()
    pairs = (link[link["permno"].astype("int64").isin(receipts) & link["liid"].notna()]
             [["gvkey", "liid"]].drop_duplicates())
    log(f"  receipts: {len(receipts):,} permnos, {len(pairs):,} Compustat issues", flush=True)
    secm = save("comp_secm_receipts",
                lambda: fetch_comp_secm(db, list(pairs.itertuples(index=False, name=None)), start_year, end_year))
    ccy = save("comp_funda_currency", lambda: fetch_funda_currency(db, start_year - 2, end_year))

    currencies = {"USD", "CAD"} | set(ccy["curcd"].dropna()) | set(secm["curcdm"].dropna())
    save("fx_daily", lambda: compustat_global.fetch_fx(db, sorted(currencies), start_year - 2, end_year))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
