"""Monthly top-N US universe by market cap from Compustat Security Monthly (comp.secm).

Market cap of a company at month end t is the sum over its common-stock issues
(tpci = '0', priced in USD) of prccm * cshom, the issue-level monthly share count. That
is the CRSP-permco-style definition: listed share classes are added up, unlisted ones
(e.g. Alphabet class B) are not counted.

cshom only exists from 1998-04 (so does cshoc in the daily file). Before that, and in
the rare later month where it is missing, the company-level share count from the
fundamentals files (fundq.cshoq, else funda.csho; millions) is carried forward from the
latest fiscal period end and multiplied by the price of the PRIMARY issue at t, with a
split adjustment from that same issue's ajexm. The company count is stated in units of
the primary issue at the fiscal date (Berkshire: A shares until 2010, B after), which is
why the price has to be the primary issue's and why the fallback is dropped when the
primary issue changed between the fiscal date and t. The copy of cshoq inside secm is
NOT used: it is attached to an arbitrary issue row (Berkshire's B row, priced 30x lower
than the A count it describes) and is missing for many firms before 1996.

`shares_source` says which tier was used. Both measures are kept for every month so the
seam can be audited on the overlap period. Assumption for the fallback: the fiscal-date
share count is known at month end. The count itself is observable in real time; the
reported figure appears with the 10-Q up to ~45 days later.

Header fields (exchg, fic, loc, conm, tic) carry a security's CURRENT value on every
historical row (primiss is historical). Filtering on exchg would drop Enron, Lehman,
WaMu and every other large firm that ended on the OTC market from the years it was
large, so there is no exchange filter: ranking on market cap already excludes small OTC
names. The country filter (fic by default) loses firms that redomiciled (Medtronic,
Eaton, ...) for their whole history; the build report lists what it removed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SECM_COLS = [
    "gvkey", "iid", "datadate", "tic", "conm", "cusip", "prccm", "cshom", "ajexm",
    "primiss", "tpci", "exchg", "fic", "curcdm",
]
NUMERIC = ["prccm", "cshom", "ajexm", "exchg"]
FUND_FILTER = "indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'"


# ----------------------------------------------------------------------------- WRDS


def read_env(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser for the gitignored .env file."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def connect(repo_root: Path):
    """WRDS connection. Username from .env or the environment; password from pgpass
    (or WRDS_PASSWORD in .env). Nothing is ever hard-coded here."""
    import wrds

    env = read_env(repo_root / ".env")
    kw = {}
    for key in ("WRDS_USERNAME", "WRDS_PASSWORD"):
        val = env.get(key) or os.environ.get(key)
        if val:
            kw[key.lower()] = val
    return wrds.Connection(**kw)


def fetch_secm(db, start_year: int, end_year: int, currency: str = "USD") -> pd.DataFrame:
    """All common-stock rows priced in `currency`, every country, one query per year.

    The country filter is applied later in pandas so the extract on disk can be
    re-filtered without going back to WRDS.
    """
    parts = []
    for year in range(start_year, end_year + 1):
        sql = f"""
            select {", ".join(SECM_COLS)}
            from comp.secm
            where tpci = '0' and curcdm = '{currency}' and prccm is not null
              and datadate between '{year}-01-01' and '{year}-12-31'
        """
        part = db.raw_sql(sql, date_cols=["datadate"])
        for c in NUMERIC:
            part[c] = pd.to_numeric(part[c], errors="coerce")
        print(f"  secm {year}: {len(part):>7,} rows", flush=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fetch_fund_shares(db, start_year: int, end_year: int) -> pd.DataFrame:
    """Company-level shares outstanding (millions) at fiscal period ends.

    fundq.cshoq first, funda.csho where the quarterly figure is missing. One row per
    (gvkey, datadate); datadate is always a month end in Compustat.
    """
    span = f"datadate between '{start_year}-01-01' and '{end_year}-12-31'"
    q = db.raw_sql(f"select gvkey, datadate, cshoq as shares_mn from comp.fundq "
                   f"where {FUND_FILTER} and cshoq is not null and {span}", date_cols=["datadate"])
    a = db.raw_sql(f"select gvkey, datadate, csho as shares_mn from comp.funda "
                   f"where {FUND_FILTER} and csho is not null and {span}", date_cols=["datadate"])
    q["fund_source"], a["fund_source"] = "fundq", "funda"
    fund = pd.concat([q, a], ignore_index=True)
    fund["shares_mn"] = pd.to_numeric(fund["shares_mn"], errors="coerce")
    fund = fund[fund["shares_mn"] > 0].sort_values(["gvkey", "datadate", "fund_source"],
                                                   ascending=[True, True, False])  # fundq first
    print(f"  fundamentals: {len(q):,} fundq + {len(a):,} funda rows", flush=True)
    return fund.drop_duplicates(["gvkey", "datadate"]).reset_index(drop=True)


def fetch_company(db) -> pd.DataFrame:
    """Header table: SIC (REIT flag), HQ country, delisting date/reason."""
    return db.raw_sql(
        "select gvkey, sic, loc, dldte, dlrsn from comp.company", date_cols=["dldte"]
    )


# ------------------------------------------------------------------- pure transforms


def company_month_mktcap(secm: pd.DataFrame, fund: pd.DataFrame,
                         max_stale_months: int = 12) -> pd.DataFrame:
    """One row per (gvkey, datadate) with market cap in USD millions.

    mktcap_cshom : sum over issues of prccm * cshom / 1e6
    mktcap_fund  : primary-issue prccm * fund shares * ajexm_at_fiscal_date / ajexm,
                   carried forward from the latest fiscal period end, at most
                   `max_stale_months` old, same primary issue at both dates
    mktcap       : mktcap_cshom, falling back to mktcap_fund
    """
    df = secm.sort_values(["gvkey", "iid", "datadate"], ignore_index=True)
    if df.duplicated(["gvkey", "iid", "datadate"]).any():
        raise ValueError("comp.secm extract has duplicate (gvkey, iid, datadate) rows")
    key = ["gvkey", "datadate"]
    df["cap_cshom"] = df["prccm"] * df["cshom"] / 1e6

    # One row per company-month: the primary issue, else the largest, else the lowest iid.
    prim = (
        df.assign(is_p=(df["primiss"] == "P"))
        .sort_values(key + ["is_p", "cap_cshom", "iid"], ascending=[True, True, False, False, True],
                     na_position="last")
        .drop_duplicates(key)
        .drop(columns="is_p")
        .reset_index(drop=True)
    )
    agg = df.groupby(key, sort=False).agg(
        mktcap_cshom=("cap_cshom", "sum"), n_issues_cshom=("cap_cshom", "count"),
        n_issues=("iid", "size"),
    ).reset_index()
    agg.loc[agg["n_issues_cshom"] == 0, "mktcap_cshom"] = np.nan
    out = prim.merge(agg, on=key, how="left").merge(fund, on=key, how="left")
    out = out.sort_values(key, ignore_index=True)

    # Carry the latest fiscal-date share count forward within the company.
    g = out.groupby("gvkey", sort=False)
    has_f = out["shares_mn"].notna()
    month_idx = out["datadate"].dt.year * 12 + out["datadate"].dt.month
    f_shares = g["shares_mn"].ffill()
    f_ajex = out["ajexm"].where(has_f).groupby(out["gvkey"]).ffill()
    f_iid = out["iid"].where(has_f).groupby(out["gvkey"]).ffill()
    f_age = month_idx - month_idx.where(has_f).groupby(out["gvkey"]).ffill()
    usable = f_age.le(max_stale_months) & (f_iid == out["iid"]) & out["ajexm"].gt(0)
    out["mktcap_fund"] = (out["prccm"] * f_shares * f_ajex / out["ajexm"]).where(usable)
    out["fund_age_months"] = f_age.where(usable)

    out["mktcap"] = out["mktcap_cshom"].fillna(out["mktcap_fund"])
    out["shares_source"] = np.select(
        [out["mktcap_cshom"].notna(), out["mktcap_fund"].notna()], ["cshom", "fund"], None
    )
    cols = key + ["mktcap", "shares_source", "mktcap_cshom", "mktcap_fund", "n_issues",
                  "fund_age_months", "iid", "tic", "conm", "cusip", "fic", "exchg", "prccm", "ajexm"]
    return out[cols]


def top_n_by_month(mcap: pd.DataFrame, n: int, start: str, end: str,
                   value: str = "mktcap") -> pd.DataFrame:
    """Rank companies by `value` within each month end and keep the top n.

    Raises if any month in [start, end] is missing or has fewer than n companies:
    a short month is a data problem, not something to paper over downstream.
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    m = mcap[(mcap["datadate"] >= lo) & (mcap["datadate"] <= hi) & mcap[value].notna()]
    m = m.sort_values(["datadate", value, "gvkey"], ascending=[True, False, True])
    m = m.assign(cap_rank=m.groupby("datadate").cumcount() + 1)
    top = m[m["cap_rank"] <= n].reset_index(drop=True)

    expected = pd.date_range(lo, hi, freq="ME")
    counts = top.groupby("datadate").size().reindex(expected, fill_value=0)
    short = counts[counts < n]
    if len(short):
        raise ValueError(f"{len(short)} month(s) have fewer than {n} companies:\n{short}")
    top["cap_rank"] = top["cap_rank"].astype("int16")
    return top


def to_wide(top: pd.DataFrame, value: str) -> pd.DataFrame:
    """months x ranks matrix of `value` (e.g. gvkey or mktcap)."""
    return top.pivot(index="datadate", columns="cap_rank", values=value)
