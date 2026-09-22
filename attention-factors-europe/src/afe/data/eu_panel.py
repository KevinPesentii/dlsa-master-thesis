"""Stage 2a of the European build: frozen raw tables -> EUR market data in the schema.

Everything downstream sees EUR. The conventions, all of them config keys in
configs/eu_data.yaml:

  Numeraire.  EUR from `numeraire.switch` (1994-01-01) using Compustat's EUR series
              (synthetic, ECU-based, before 1999-01-01); before the switch the
              Deutsche Mark at the irrevocable 1.95583 DEM/EUR, so one continuous unit.
              DEM is carried as a second column: quoted to 2018-06 (Compustat stops
              there), EUR x 1.95583 afterwards. Any other pair is the ratio of two
              GBP-based quotes on the same day (`comp.g_exrt_dly`).
  Returns.    A listing's daily total return is computed on its EUR price:
              p_eur(t) = prccd/qunit x eur_per_unit(curcdd, t), then
              ret(t) = (p_eur/ajexdi*trfd)_t / (...)_{t-1} - 1. A GBP- or USD-quoted
              line therefore carries the currency move, and a line whose quotation
              switched DEM -> EUR on 1999-01-04 has no break. Local-currency returns
              are kept alongside. Returns are per (gvkey, iid); the company's return
              in month M is that of the line that priced its cap at the end of M-1,
              so a change of line never splices two price series.
  Universe.   Membership for month M = the top-N of the cap table at the end of M-1;
              `month` is the first trading day of M on the pooled calendar. sec_id is
              the gvkey (one per company; the listing is in the returns detail).
  Market.     Value-weighted EUR return of the month's universe, weights = cap at the
              end of M-1 (point in time); equal-weighted alongside. Built from the
              data, no external index.
  Delisting.  Compustat Global has no delisting return; the last row is the last
              trade. `delisting.return` in the config (default: none) can impute JKP's
              -30% on a security's final month; the report counts final months.

Reading order: load_raw -> fx_conversion_table -> daily_returns -> universe_lagged ->
returns_table -> market_returns. `truncate(cutoff)` on the raw bundle makes the
prefix-invariance test possible, as for the US.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from afe import schemas

DEM_PER_EUR = 1.95583
# irrevocable conversion rates, units of legacy currency per EUR (1999-01-01; GRD 2001-01-01)
EURO_FIXED_RATES = {"DEM": 1.95583, "FRF": 6.55957, "ITL": 1936.27, "NLG": 2.20371, "BEF": 40.3399,
                    "LUF": 40.3399, "ESP": 166.386, "ATS": 13.7603, "FIM": 5.94573, "IEP": 0.787564,
                    "PTE": 200.482, "GRD": 340.750, "EUR": 1.0}
ISO2 = {"AUT": "AT", "BEL": "BE", "DEU": "DE", "ESP": "ES", "FIN": "FI", "FRA": "FR", "IRL": "IE",
        "ITA": "IT", "LUX": "LU", "NLD": "NL", "PRT": "PT", "GRC": "GR", "GBR": "GB", "CHE": "CH",
        "SWE": "SE", "DNK": "DK", "NOR": "NO"}


@dataclass
class RawEU:
    mktcap: pd.DataFrame          # step-1 cap table, all companies, `eligible` flag
    fx: pd.DataFrame              # comp.g_exrt_dly rows: datadate, tocurd, exratd
    daily_dir: Path               # secd_daily/<year>.parquet

    def truncate(self, cutoff: pd.Timestamp) -> "RawEU":
        c = pd.Timestamp(cutoff)
        return RawEU(self.mktcap[self.mktcap["datadate"] <= c], self.fx[self.fx["datadate"] <= c], self.daily_dir)


def load_raw(data_dir: Path) -> RawEU:
    mk = pd.read_parquet(data_dir / "company_month_mktcap.parquet")
    fx = pd.read_parquet(data_dir / "raw" / "fx_daily.parquet")
    for df, col in [(mk, "datadate"), (fx, "datadate")]:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return RawEU(mk, fx, data_dir / "raw" / "secd_daily")


def load_daily(daily_dir: Path, pairs: pd.DataFrame | None, years, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """secd_daily rows for the given (gvkey, iid) pairs (all if None) and years."""
    parts = []
    for y in years:
        p = daily_dir / f"{y}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if pairs is not None:
            df = df.merge(pairs[["gvkey", "iid"]].drop_duplicates(), on=["gvkey", "iid"])
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    d["datadate"] = pd.to_datetime(d["datadate"]).astype("datetime64[ns]")
    if cutoff is not None:
        d = d[d["datadate"] <= pd.Timestamp(cutoff)]
    return d.sort_values(["gvkey", "iid", "datadate"], ignore_index=True)


# ------------------------------------------------------------------ currency


def fx_conversion_table(fx: pd.DataFrame, switch: str = "1994-01-01",
                        dem_per_eur: float = DEM_PER_EUR) -> pd.DataFrame:
    """Long table (date, currency) -> eur_per_unit, dem_per_unit, on the calendar-day
    grid of comp.g_exrt_dly, with the numeraire rule of the module docstring.
    `eur_source` / `dem_source` say whether a value is a quote or the fixed rate."""
    wide = fx.pivot_table(index="datadate", columns="tocurd", values="exratd").sort_index()
    if "GBP" not in wide:
        wide["GBP"] = 1.0
    sw = pd.Timestamp(switch)
    eur_q, dem_q = wide.get("EUR"), wide.get("DEM")
    if eur_q is None or dem_q is None:
        raise ValueError("fx table needs EUR and DEM quotes")
    # numeraire per GBP: EUR quote from the switch, DEM/1.95583 before it
    eur_num = eur_q.where(wide.index >= sw, dem_q / dem_per_eur)
    eur_source = np.where(wide.index >= sw, "eur_quote", "dem_fixed")
    dem_num = dem_q.where(dem_q.notna(), eur_num * dem_per_eur)
    dem_source = np.where(dem_q.notna(), "dem_quote", "eur_fixed")
    out = []
    for ccy in wide.columns:
        rate = wide[ccy]
        t = pd.DataFrame({"date": wide.index, "currency": ccy,
                          "eur_per_unit": (eur_num / rate).values, "dem_per_unit": (dem_num / rate).values,
                          "eur_source": eur_source, "dem_source": dem_source})
        out.append(t[t["eur_per_unit"].notna()])
    # DEM after its last quote: continue it as a currency at the fixed rate (for
    # completeness; no price in the sample is quoted in DEM after 1998)
    tab = pd.concat(out, ignore_index=True)
    dem_rows = tab[tab["currency"] == "DEM"]
    if len(dem_rows) and dem_rows["date"].max() < wide.index.max():
        ext = tab[(tab["currency"] == "EUR") & (tab["date"] > dem_rows["date"].max())].copy()
        ext["currency"], ext["eur_per_unit"], ext["dem_per_unit"] = "DEM", ext["eur_per_unit"] / dem_per_eur, 1.0
        ext["dem_source"] = "eur_fixed"
        tab = pd.concat([tab, ext], ignore_index=True)
    return tab.sort_values(["currency", "date"], ignore_index=True)


def eur_factor(dates: pd.Series, currencies: pd.Series, fxtab: pd.DataFrame, target: str = "eur_per_unit",
               max_stale_days: int = 7) -> pd.Series:
    """Per-row conversion factor at the row's date (latest rate within `max_stale_days`)."""
    key = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]").values,
                        "currency": np.asarray(currencies.astype(str)), "_i": np.arange(len(dates))}).sort_values("date")
    fx = fxtab[["date", "currency", target]].assign(date=lambda t: t["date"].astype("datetime64[ns]")).sort_values("date")
    m = pd.merge_asof(key, fx, on="date", by="currency", tolerance=pd.Timedelta(days=max_stale_days),
                      direction="backward")
    return pd.Series(m.sort_values("_i")[target].values, index=dates.index)


# ------------------------------------------------------------------ returns


def daily_returns(daily: pd.DataFrame, fxtab: pd.DataFrame) -> pd.DataFrame:
    """Per (gvkey, iid, date): EUR price, local and EUR total returns.

    Rows without a price, an adjustment factor or a usable FX rate carry no return and
    break the chain (the next row's return is NaN too)."""
    d = daily.sort_values(["gvkey", "iid", "datadate"]).copy()
    d["qunit"] = d["qunit"].fillna(1.0)
    ok = (d["prccd"] > 0) & (d["ajexdi"] > 0) & d["trfd"].notna() & d["curcdd"].notna()
    d["fx_eur"] = eur_factor(d["datadate"], d["curcdd"], fxtab)
    d["prc_eur"] = (d["prccd"] / d["qunit"] * d["fx_eur"]).where(ok)
    d["_al"] = (d["prccd"] / d["qunit"] / d["ajexdi"] * d["trfd"]).where(ok)   # adjusted local price
    d["_ae"] = d["prc_eur"] / d["ajexdi"] * d["trfd"]                             # adjusted EUR price
    g = d.groupby(["gvkey", "iid"], sort=False, observed=True)
    d["ret_local"] = d["_al"] / g["_al"].shift(1) - 1
    d.loc[d["curcdd"].astype(str) != g["curcdd"].shift(1).astype(str), "ret_local"] = np.nan  # quotation currency changed
    d["ret"] = d["_ae"] / g["_ae"].shift(1) - 1
    d["traded"] = d["prcstd"] == 10
    cols = ["gvkey", "iid", "datadate", "curcdd", "prccd", "prc_eur", "fx_eur", "ajexdi", "trfd", "cshoc",
            "cshtrd", "prchd", "prcld", "prcstd", "traded", "ret_local", "ret"]
    return d[cols].rename(columns={"datadate": "date"}).reset_index(drop=True)


def trading_calendar(daily: pd.DataFrame, min_lines: int = 100) -> pd.DatetimeIndex:
    """Days on which at least `min_lines` listings have a TRADED close (prcstd 10): the
    pooled calendar. Compustat carries prices (prcstd 5) on exchange holidays for a few
    lines, which would otherwise make Good Friday or Christmas a trading day."""
    n = daily[daily["prcstd"] == 10].groupby("datadate").size()
    return pd.DatetimeIndex(n[n >= min_lines].index).sort_values()


# ------------------------------------------------------------------ universe


def asof_ranking(mktcap: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n eligible companies per month end (AS-OF), from the cap table."""
    e = mktcap[mktcap["eligible"] & mktcap["mktcap"].notna()]
    e = e.sort_values(["datadate", "mktcap", "gvkey"], ascending=[True, False, True])
    e = e.assign(cap_rank=(e.groupby("datadate").cumcount() + 1).astype("int16"))
    return e[e["cap_rank"] <= n].reset_index(drop=True)


def universe_lagged(asof: pd.DataFrame, calendar: pd.DatetimeIndex, start: str, end: str, n: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(universe per docs/schemas.md, membership detail) for months start..end.
    Month M uses the AS-OF ranking at the end of M-1."""
    first_day = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).min()
    a = asof.assign(month=asof["datadate"].dt.to_period("M") + 1)
    lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
    a = a[(a["month"] >= lo) & (a["month"] <= hi)]
    counts = a.groupby("month").size()
    missing = [str(m) for m in pd.period_range(lo, hi, freq="M") if counts.get(m, 0) != n]
    if missing:
        raise ValueError(f"months without exactly {n} members: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if not a["month"].isin(first_day.index).all():
        raise ValueError("a universe month has no trading day on the calendar")
    detail = a.assign(month_start=a["month"].map(first_day))
    uni = pd.DataFrame({"month": detail["month_start"].astype("datetime64[ns]"), "sec_id": detail["gvkey"].astype(str),
                        "cap_rank": detail["cap_rank"].astype("int16")}).sort_values(["month", "sec_id"], ignore_index=True)
    schemas.validate_universe(uni, size=n)
    return uni, detail


def returns_table(dret: pd.DataFrame, mktcap: pd.DataFrame, start: str, end: str,
                  delisting_return: float | None = None, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """docs/schemas.md returns table for every company in `dret`: the return of the
    line that priced the company's cap at the end of the previous month, on every
    calendar day of the month, with that cap as mktcap_lag (EUR mn), country ISO-2,
    currency. A line's carried price on its own exchange's holiday gives a zero local
    return on a pooled trading day; the detail table's `traded` flag says which."""
    d = dret[["gvkey", "iid", "date", "ret", "curcdd"]].copy()
    if calendar is not None:
        d = d[d["date"].isin(calendar)]
    d["month_prev"] = d["date"].dt.to_period("M") - 1
    lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
    d = d[(d["month_prev"] + 1 >= lo) & (d["month_prev"] + 1 <= hi)]
    prev = mktcap[["gvkey", "iid", "datadate", "mktcap", "country"]].copy()
    prev["month_prev"] = prev["datadate"].dt.to_period("M")
    d = d.merge(prev.drop(columns="datadate"), on=["gvkey", "iid", "month_prev"], how="inner")
    if delisting_return is not None:
        last = d.groupby("gvkey")["date"].transform("max")
        d.loc[(d["date"] == last), "ret"] = (1 + d.loc[d["date"] == last, "ret"]) * (1 + delisting_return) - 1
    d = d[d["ret"].notna() & d["mktcap"].notna()]
    out = pd.DataFrame({"date": d["date"].astype("datetime64[ns]"), "sec_id": d["gvkey"].astype(str),
                        "ret": d["ret"].astype("float32"), "mktcap_lag": d["mktcap"].astype("float32"),
                        "country": d["country"].map(ISO2).fillna(d["country"]).astype(object),
                        "currency": d["curcdd"].astype(object)})
    out = out.sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(out, schemas.RETURNS)
    return out


def market_returns(ret: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Daily value- and equal-weighted EUR return of the month's universe, weights =
    mktcap_lag (the cap that set membership). `n` = members with a return that day."""
    r = ret.assign(month=ret["date"].dt.to_period("M"))
    u = uni.assign(month=uni["month"].dt.to_period("M"))[["month", "sec_id"]]
    m = r.merge(u, on=["month", "sec_id"])
    m["w"] = m["mktcap_lag"].astype("float64")
    m["wr"] = m["w"] * m["ret"].astype("float64")
    g = m.groupby("date")
    out = pd.DataFrame({"mkt_vw": g["wr"].sum() / g["w"].sum(), "mkt_ew": g["ret"].mean(), "n": g.size()})
    return out.reset_index()
