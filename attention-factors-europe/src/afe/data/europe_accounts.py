"""Accounting and JKP inputs of the European builds, in the numeraire (ECU to 1998-12-31,
EUR from 1999-01-01; europe_fx.py).

  fundamentals  g_funda amounts converted to the numeraire at the fiscal year end;
                ni = nicon, else ib + xido (nicon is 16-22% filled before 1999);
                mib, else mibt (mib is empty from ~2015); gp = revt - cogs; pstkrv /
                pstkl absent (book equity falls back to pstk), xad absent (zero, as in
                the US fill list); banks and insurers (indfmt FS) carry revt but no
                sale: sale = revt there, which is what Compustat North America's sale
                holds for banks. FS rows have no cogs, xsga, act, lct, che, capx at all.
                annual_frame then adds lags, book equity and December ME as
                us_panel.annual_frame does for the US; when a record is usable is the
                config's (fundamentals.availability).
  JKP           contrib.global_factor rows of the universe members as of the end of M-1,
                USD amounts converted at the month end, returns re-expressed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afe.data import characteristics as ch
from afe.data import eu_panel as ep
from afe.data import us_panel as up

# ------------------------------------------------------------------ fundamentals

AMOUNTS_EXCLUDED = ("cshoi", "cshpria", "ajexi", "emp")   # shares, factors, headcount: not currency


def funda_numeraire(funda: pd.DataFrame, fxtab: pd.DataFrame) -> pd.DataFrame:
    """g_funda in the numeraire at the fiscal year end, one row per (gvkey, datadate)
    (INDL preferred where both formats exist), with Global's item substitutes."""
    f = funda.copy()
    f["_fs"] = f["indfmt"] == "FS"
    f = f.sort_values(["gvkey", "datadate", "_fs"]).drop_duplicates(["gvkey", "datadate"], keep="first")
    amounts = [c for c in f.columns if c not in ("gvkey", "datadate", "fyear", "fyr", "curcd", "indfmt", "_fs")
               and c not in AMOUNTS_EXCLUDED]
    f[amounts] = f[amounts].astype("float64")
    f["fx_num"] = ep.eur_factor(f["datadate"], f["curcd"].fillna("NA"), fxtab)
    f[amounts] = f[amounts].mul(f["fx_num"], axis=0)
    # net income: nicon is 16-22% filled before 1999 (INDL); NI = IB + XIDO is Compustat's identity
    f["ni"] = f["nicon"].fillna(f["ib"] + f["xido"].fillna(0.0))
    # minority interest: mib is empty in Global from ~2015 (IFRS 10); the total noncontrolling interest replaces it
    if "mibt" in f:
        f["mib"] = f["mib"].fillna(f["mibt"])
    f["gp"] = f["revt"] - f["cogs"]
    f.loc[f["_fs"] & f["sale"].isna(), "sale"] = f.loc[f["_fs"] & f["sale"].isna(), "revt"]
    for c in ("pstkrv", "pstkl", "xad"):
        if c not in f:
            f[c] = np.nan
    return f


def annual_frame(f: pd.DataFrame, mktcap: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """us_panel.annual_frame for Global: lags by fyear, book equity, NOA, NWC, December
    company cap of the fiscal year's calendar year (numeraire mn, from the cap table),
    avail_month per config fundamentals."""
    fc = cfg["fundamentals"]
    f = f.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last").copy()
    for c in fc["fill_zero"]:
        f[c] = f[c].fillna(0.0)
    f["be"] = ch.book_equity(f)
    f["noa_level"] = ch.net_operating_assets(f)
    f["nwc"] = ch.noncash_working_capital(f)
    prev = f[["gvkey", "fyear"] + up.LAGGED].copy()
    prev["fyear"] = prev["fyear"] + 1
    prev = prev.drop_duplicates(["gvkey", "fyear"], keep="last")
    f = f.merge(prev.rename(columns={c: f"{c}_lag" for c in up.LAGGED}), on=["gvkey", "fyear"], how="left")
    f["dec_month"] = pd.PeriodIndex.from_fields(year=f["datadate"].dt.year, month=12, freq="M")
    cap = mktcap[["gvkey", "datadate", "mktcap"]].assign(dec_month=lambda t: t["datadate"].dt.to_period("M"))
    f = f.merge(cap[["gvkey", "dec_month", "mktcap"]].rename(columns={"mktcap": "me_dec"}),
                on=["gvkey", "dec_month"], how="left")
    if fc["availability"] == "june":
        f["avail_month"] = pd.PeriodIndex.from_fields(year=f["datadate"].dt.year + 1, month=6, freq="M")
    elif fc["availability"] == "lag_months":
        f["avail_month"] = (f["datadate"] + pd.DateOffset(months=int(fc["lag_months"]))).dt.to_period("M")
    else:
        raise ValueError(f"fundamentals.availability: {fc['availability']}")
    return f


# ------------------------------------------------------------------ JKP

# JKP columns in USD (amounts, prices) and in 1/USD (Amihud), converted to the numeraire at
# the month end; everything else is a ratio, a count, a share number or local currency.
JKP_USD = ["me", "me_company", "market_equity", "me_lag1", "prc", "prc_high", "prc_low", "dolvol",
           "dolvol_126d", "assets", "sales", "book_equity", "net_income", "enterprise_value", "intrinsic_value"]
JKP_PER_USD = ["ami_126d"]
JKP_DROP = ["ret_exc_lead1m"]   # next month's return: not information at the month end


def jkp_table(jkp: pd.DataFrame, members: pd.DataFrame, fxtab: pd.DataFrame, rf_month: pd.Series) -> pd.DataFrame:
    """JKP rows of the universe members: for universe month M the row dated at the end of
    M-1 (JKP is point in time as of `eom`), of the line that priced the company's cap then,
    else JKP's primary security of the company in the version's countries. USD columns
    converted with the numeraire rate at `eom`; `ret` (USD total return of month M-1)
    re-expressed in the numeraire; `ret_exc` = that minus the numeraire rf of the month.
    JKP's return-based characteristics stay as JKP computed them (from USD returns)."""
    j = jkp.copy()
    j["eom"] = pd.to_datetime(j["eom"]).astype("datetime64[ns]")
    j["month"] = j["eom"].dt.to_period("M") + 1
    mem = members[["month", "gvkey", "iid"]]
    exact = mem.merge(j, on=["month", "gvkey", "iid"], how="inner")
    rest = mem[~mem.set_index(["month", "gvkey"]).index.isin(exact.set_index(["month", "gvkey"]).index)]
    prim = j[j["primary_sec"] == 1].drop(columns="iid").drop_duplicates(["month", "gvkey"])
    fb = rest.merge(prim, on=["month", "gvkey"], how="inner").assign(jkp_line="primary_sec")
    out = pd.concat([exact.assign(jkp_line="home_line"), fb], ignore_index=True)
    x = ep.eur_factor(out["eom"], pd.Series("USD", index=out.index), fxtab)            # numeraire per USD at eom
    x_prev = ep.eur_factor(out["eom"] - pd.offsets.MonthEnd(1), pd.Series("USD", index=out.index), fxtab)
    for c in JKP_USD:
        if c in out:
            out[c] = out[c].astype("float64") * x
    for c in JKP_PER_USD:
        if c in out:
            out[c] = out[c].astype("float64") / x
    out["ret_usd"] = out["ret"]
    out["ret"] = (1.0 + out["ret"].astype("float64")) * (x / x_prev) - 1.0
    out["ret_exc"] = out["ret"] - out["eom"].dt.to_period("M").map(rf_month).astype("float64")
    return out.drop(columns=[c for c in JKP_DROP if c in out]).sort_values(["month", "gvkey"], ignore_index=True)
