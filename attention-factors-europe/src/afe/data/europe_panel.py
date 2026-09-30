"""Stage 2 building blocks of the European builds (europe17, europe12): raw tables -> the
inputs of characteristics.py, every amount and return in the numeraire (ECU to 1998-12-31,
EUR from 1999-01-01; europe_fx.py).

The formulas are the US ones (characteristics.py, docs/us_characteristics.md); this module
only decides WHO and WHEN, as us_panel does for the US, and substitutes what Compustat
Global lacks:

  company_lines      the listing that carries a company's series in trading month M: the
                     line that priced its cap at the end of M-1 in the cap table (as for the
                     returns table of eu_panel); before its first cap row, the first line.
  company_daily      that line's rows: numeraire close / high / return, local high / low /
                     close for the spread, volume, shares, adjustment factor.
  Spread             Global has no bid/ask. Corwin & Schultz (2012) high-low estimator from
                     two consecutive traded days of the same line, local prices, the
                     overnight adjustment of their Section IV.A, negative estimates set to
                     zero, dated on the second day; monthly = the mean over the month (>= 10
                     days, the US rule). JKP's bidaskhl_21d is the same estimator.
  market / factors   Ken French's Europe three factors (USD) in the numeraire (ff_europe),
                     and the top N's own: the value-weighted numeraire return of the month's
                     top-N (the universe from 1990; the as-of top-N of the ranking before),
                     weights = cap at the end of M-1, and SMB / HML from a 2x3 sort of the
                     same companies (cap median, BEME 30/70), value-weighted, formed with the
                     BEME usable at the end of M-1. Which feeds Beta and Resid_Var: config
                     `factors.source` (build_europe).
Fundamentals and the JKP table are in europe_accounts.py.

Index names: the company id level is called `permno` inside the characteristic frames so
characteristics.py and build_us's assembly run unchanged; its values are gvkeys.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from afe.data import characteristics as ch
from afe.data import us_panel as up

ID = "permno"   # company id level name expected by characteristics.py / build_us (values: gvkey)


# ------------------------------------------------------------------ loading


@dataclass
class RawEurope:
    mktcap: pd.DataFrame        # version cap table, all companies with a cap, `eligible` flag
    ranking: pd.DataFrame       # as-of top-N per month end (ranking_asof.parquet)
    lines: pd.DataFrame         # (gvkey, iid) home lines of the version's ever-members
    funda: pd.DataFrame         # union g_funda
    fx: pd.DataFrame            # comp.g_exrt_dly rows (extract + raw pull), units per GBP
    daily_dir: Path             # union secd_daily/<year>.parquet
    jkp_dir: Path               # union jkp/<year>.parquet

    def truncate(self, cutoff: pd.Timestamp) -> "RawEurope":
        c = pd.Timestamp(cutoff)
        return RawEurope(self.mktcap[self.mktcap["datadate"] <= c], self.ranking[self.ranking["datadate"] <= c],
                         self.lines, self.funda[self.funda["datadate"] <= c], self.fx[self.fx["datadate"] <= c],
                         self.daily_dir, self.jkp_dir)


def load_raw(root: Path, cfg: dict) -> RawEurope:
    ins, raw = root / cfg["output"]["inspect_dir"], root / cfg["raw"]["dir"]
    mk = pd.read_parquet(ins / "company_month_mktcap.parquet")
    rk = pd.read_parquet(ins / "ranking_asof.parquet")
    lines = pd.read_csv(ins / "listings.csv", dtype=str)
    funda = pd.read_parquet(raw / "g_funda.parquet")
    fx = pd.concat([pd.read_parquet(root / cfg["extract"]["dir"] / "fx_daily.parquet"),
                    pd.read_parquet(raw / "fx_daily.parquet")], ignore_index=True)
    for df, col in [(mk, "datadate"), (rk, "datadate"), (funda, "datadate"), (fx, "datadate")]:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    # one id dtype everywhere: WRDS hands back pandas' string dtype, CSVs and joins give object,
    # and merge_asof refuses to match the two
    for df in (mk, rk, lines, funda):
        for col in ("gvkey", "iid"):
            if col in df:
                df[col] = df[col].astype(str).astype(object)
    for df, col in [(mk, "country"), (rk, "country"), (funda, "curcd"), (funda, "indfmt")]:
        df[col] = df[col].astype(object)
    fx["tocurd"] = fx["tocurd"].astype(object)
    fx["exratd"] = fx["exratd"].astype("float64")
    fx = fx.drop_duplicates(["datadate", "tocurd"]).sort_values(["tocurd", "datadate"], ignore_index=True)
    return RawEurope(mk, rk, lines, funda, fx, raw / "secd_daily", raw / "jkp")


# ------------------------------------------------------------------ risk-free


def risk_free(external_dir: Path, calendar: pd.DatetimeIndex, day_count: int = 360) -> pd.DataFrame:
    """Daily risk-free return on the calendar from the Bundesbank 1-month series: Frankfurt
    banks' 1-month funds before 1990-07-02 (the first FIBOR day), FIBOR to 1998-12-31,
    EURIBOR from 1999-01-01; each forward-filled over non-quoted days. Trading day t earns
    the rate quoted on the previous trading day over the calendar days in between, act/360
    (Friday to Monday: three days), so a year of it adds up to the quoted rate x 365/360;
    one day's rate/360 per trading day would give only ~72% of the rate."""
    def read(name):
        df = pd.read_csv(external_dir / name, skiprows=1, header=None, usecols=[0, 1], names=["date", "v"])
        df = df[df["date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")]
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        return df.dropna().assign(date=lambda t: pd.to_datetime(t["date"])).set_index("date")["v"].sort_index()
    ffm, fibor, euribor = (read(f) for f in ("bbk_ST0104_frankfurt_1m_daily.csv", "bbk_ST0262_fibor_1m_daily.csv",
                                             "bbk_ST0310_euribor_1m_daily.csv"))
    f0, e0 = fibor.index.min(), pd.Timestamp("1999-01-01")
    parts = [ffm[ffm.index < f0].to_frame("pct").assign(series="FRANKFURT_1M_DEM"),
             fibor[fibor.index < e0].to_frame("pct").assign(series="FIBOR_1M_DEM"),
             euribor[euribor.index >= e0].to_frame("pct").assign(series="EURIBOR_1M_EUR")]
    q = pd.concat(parts).sort_index()
    q = q[~q.index.duplicated(keep="last")]
    out = q.reindex(q.index.union(calendar)).ffill().reindex(calendar)
    prev = out.shift(1)
    prev.iloc[0] = out.iloc[0]
    days = pd.Series(calendar, index=calendar).diff().dt.days.fillna(1.0).to_numpy()
    return pd.DataFrame({"date": calendar, "rf": (prev["pct"] / 100.0 * days / day_count).to_numpy(),
                         "rate_pct_pa": prev["pct"].to_numpy(), "series": prev["series"].to_numpy(),
                         "accrual_days": days.astype("int16")})


# ------------------------------------------------------------------ Ken French's Europe factors


def read_ff_daily(path: Path) -> pd.DataFrame:
    """A daily factor file of Ken French's data library (the .zip as downloaded, or its .csv):
    the dated rows of its table, in decimals (the file is in percent; -99.99 = missing)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
    else:
        text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    head = next(i for i, s in enumerate(lines) if s.replace(" ", "").lower().startswith(",mkt-rf"))
    cols = [c.strip().lower().replace("-", "") for c in lines[head].split(",")[1:]]     # mktrf, smb, hml, rf
    rows = []
    for s in lines[head + 1:]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows and s.strip():
            break                                                                         # the next table
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype("float64").replace(-99.99, np.nan) / 100.0


def ff_europe(path: Path, fxtab: pd.DataFrame, calendar: pd.DatetimeIndex,
              tbill_monthly: Path | None = None) -> pd.DataFrame:
    """Ken French's Europe three factors (value-weighted, all sizes, 16 countries: europe17's
    less Luxembourg; every column of the file is in USD) as daily numeraire returns on the
    calendar, NaN outside the file: `mkt` (total return), `smb`, `hml`.

    mkt: USD return = Mkt-RF + RF, RF being FF's monthly 1-month T-bill spread over the
    month's days of the file when `tbill_monthly` is given (FF's daily RF is built that way,
    and the daily file rounds it to 0.01%: up to 1 pp a year off), else the file's RF; the
    USD index times the numeraire per USD (`fxtab`, europe_fx.conversion_table) is read on
    the calendar. smb, hml: long-short spreads of USD returns; both legs convert with the same
    factor g = x_t / x_t-1 (numeraire per USD), so the numeraire spread is the USD one times g.
    A file day off the calendar compounds into the next calendar day (spreads as 1 + s)."""
    ff = read_ff_daily(path)
    rf = ff["rf"]
    if tbill_monthly is not None:
        m = pd.read_parquet(tbill_monthly, columns=["date", "rf"])
        rfm = pd.Series(m["rf"].to_numpy(dtype="float64"), index=pd.to_datetime(m["date"]).dt.to_period("M"))
        month = ff.index.to_period("M")
        n = pd.Series(1.0, index=ff.index).groupby(month).transform("size").to_numpy()
        rf = pd.Series((1.0 + month.map(rfm).to_numpy(dtype="float64")) ** (1.0 / n) - 1.0,
                       index=ff.index).fillna(ff["rf"])
    idx = (1.0 + pd.DataFrame({"mkt": ff["mktrf"] + rf, "smb": ff["smb"], "hml": ff["hml"]})).cumprod()
    before = calendar[calendar < ff.index.min()]
    if len(before):
        idx = pd.concat([pd.DataFrame(1.0, index=before[-1:], columns=idx.columns), idx])   # base: the day before
    usd = fxtab[fxtab["currency"] == "USD"].set_index("date")["eur_per_unit"].astype("float64").sort_index()
    days = idx.index.union(calendar)
    x = usd.reindex(usd.index.union(days)).ffill().reindex(days).reindex(calendar)
    on_cal = idx.reindex(days).ffill().reindex(calendar)
    g = x / x.shift(1)
    out = pd.DataFrame({"mkt": (1.0 + on_cal["mkt"].pct_change()) * g - 1.0,
                        "smb": on_cal["smb"].pct_change() * g, "hml": on_cal["hml"].pct_change() * g})
    out.loc[(calendar <= idx.index.min()) | (calendar > ff.index.max())] = np.nan
    return out


# ------------------------------------------------------------------ spread


def corwin_schultz(d: pd.DataFrame) -> pd.Series:
    """Two-day Corwin-Schultz (2012) spread per row of `d` (one line's consecutive rows,
    sorted), from local high `prchd`, low `prcld`, close `prccd`, using this row and the
    previous one; NaN unless both days are traded closes with 0 < low <= high. Overnight
    adjustment: if today's low is above yesterday's close, today's high and low are moved
    down by the gap; if today's high is below it, up by the gap."""
    g = d.groupby(["gvkey", "iid"], sort=False, observed=True)
    h1, l1, c0 = d["prchd"].astype("float64"), d["prcld"].astype("float64"), g["prccd"].shift(1).astype("float64")
    h0, l0 = g["prchd"].shift(1).astype("float64"), g["prcld"].shift(1).astype("float64")
    ok = ((d["prcstd"] == 10) & (g["prcstd"].shift(1) == 10) & (l0 > 0) & (l1 > 0) & (h0 >= l0) & (h1 >= l1))
    up_gap = (l1 - c0).clip(lower=0).fillna(0.0)          # today's range entirely above yesterday's close
    dn_gap = (c0 - h1).clip(lower=0).fillna(0.0)          # entirely below
    h1a, l1a = h1 - up_gap + dn_gap, l1 - up_gap + dn_gap
    beta = np.log(h0 / l0) ** 2 + np.log(h1a / l1a) ** 2
    gamma = np.log(np.maximum(h0, h1a) / np.minimum(l0, l1a)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0.0).where(ok)


# ------------------------------------------------------------------ company series


def _month_number(p: pd.Series) -> pd.Series:
    """Period[M] -> int (year * 12 + month), for merge_asof, which takes no Period keys."""
    return (p.dt.year * 12 + p.dt.month).astype("int64")


def company_lines(mktcap: pd.DataFrame) -> pd.DataFrame:
    """(gvkey, m, iid): the line that priced the company's cap at the end of month number m
    (year * 12 + month); its series is the company's in month m + 1."""
    m = pd.DataFrame({"gvkey": mktcap["gvkey"].astype(str).to_numpy(), "iid": mktcap["iid"].astype(str).to_numpy(),
                      "m": _month_number(mktcap["datadate"].dt.to_period("M")).to_numpy()})
    return m.drop_duplicates(["gvkey", "m"]).sort_values(["gvkey", "m"], ignore_index=True)


def company_daily(dret: pd.DataFrame, mktcap: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (company, pooled trading day): the carrying line's values. The line for
    month M is the cap-table line at the end of M-1; if the company has no cap row then,
    the latest earlier one; before its first cap row, the first one (the only choice
    that uses no later information than the first row itself)."""
    d = dret[dret["date"].isin(calendar)].copy()
    d["gvkey"], d["iid"] = d["gvkey"].astype(str), d["iid"].astype(str)
    d["m"] = _month_number(d["date"].dt.to_period("M")) - 1
    lines = company_lines(mktcap)
    key = d[["gvkey", "m"]].drop_duplicates().sort_values("m", ignore_index=True)
    lk = pd.merge_asof(key, lines.sort_values("m", ignore_index=True), on="m", by="gvkey", direction="backward")
    first = lines.groupby("gvkey")["iid"].first()
    lk["iid"] = lk["iid"].fillna(lk["gvkey"].map(first))
    d = d.merge(lk.rename(columns={"iid": "line"}), on=["gvkey", "m"], how="left")
    d = d[d["iid"] == d["line"]].drop(columns=["line", "m"])
    # plain float64: the parquet metadata restores nullable Float64, whose to_numpy() is an
    # object array, and SUV's 0/0 then raises ZeroDivisionError (the us_panel.load_daily trap)
    num = [c for c in d.columns if c not in ("gvkey", "iid", "date", "curcdd", "traded", "prcstd")
           and pd.api.types.is_numeric_dtype(d[c])]
    d[num] = d[num].astype("float64")
    return d.sort_values(["gvkey", "date"], ignore_index=True)


def monthly_frame(cd: pd.DataFrame, mktcap: pd.DataFrame) -> pd.DataFrame:
    """Per (company, month): compounded numeraire return over the month's days with a
    return, last close (numeraire), volume summed (NaN when no day has volume), shares at
    the last row (thousands, the US shrout unit), company cap from the cap table."""
    c = cd.assign(month=cd["date"].dt.to_period("M"))
    c["lr"] = np.log1p(c["ret"].astype("float64"))
    c["cshtrd"] = c["cshtrd"].astype("float64")
    g = c.groupby(["gvkey", "month"], sort=False)
    out = pd.DataFrame({"lr": g["lr"].sum(min_count=1), "prc": g["prc_eur"].last(),
                        "vol": g["cshtrd"].sum(min_count=1), "shrout": g["cshoc"].last()}).reset_index()
    out["ret"] = np.expm1(out.pop("lr"))
    out["shrout"] = out["shrout"].astype("float64") / 1000.0
    cap = mktcap[["gvkey", "datadate", "mktcap"]].assign(month=lambda t: t["datadate"].dt.to_period("M"))
    out = out.merge(cap[["gvkey", "month", "mktcap"]].rename(columns={"mktcap": "cap_co"}), on=["gvkey", "month"], how="left")
    return out


# ------------------------------------------------------------------ market and factors


def members_by_month(ranking: pd.DataFrame, n: int) -> pd.DataFrame:
    """(month M, gvkey, cap): the top-n of the as-of ranking at the end of M-1, i.e. the
    universe of M (from the sample start) and the market's constituents before it."""
    r = ranking[ranking["cap_rank"] <= n][["datadate", "gvkey", "mktcap", "cap_rank"]].copy()
    r["month"] = r["datadate"].dt.to_period("M") + 1
    r["mktcap"] = r["mktcap"].astype("float64")          # nullable Float64 from the cap table
    return r.drop(columns="datadate").rename(columns={"mktcap": "cap"})


def value_weighted(cd: pd.DataFrame, members: pd.DataFrame, weight: str = "cap") -> pd.DataFrame:
    """Daily value-weighted and equal-weighted return of each month's members."""
    r = cd[["gvkey", "date", "ret"]].assign(month=cd["date"].dt.to_period("M"))
    m = r.merge(members[["month", "gvkey", weight]], on=["month", "gvkey"])
    m = m[m["ret"].notna()]
    m["w"] = m[weight].astype("float64")
    m["wr"] = m["w"] * m["ret"].astype("float64")
    g = m.groupby("date")
    return pd.DataFrame({"mkt_vw": g["wr"].sum() / g["w"].sum(), "mkt_ew": g["ret"].mean(), "n": g.size()})


def smb_hml(cd: pd.DataFrame, members: pd.DataFrame, beme: pd.DataFrame, min_per_portfolio: int) -> pd.DataFrame:
    """Daily SMB and HML from a 2x3 sort of each month's members: size at the median cap,
    BEME at the 30th/70th percentiles, positive BEME only, value-weighted by cap at the end
    of M-1. `beme`: (month, gvkey, BEME) with the value usable at the end of `month`,
    applied to month + 1. A month with a portfolio below min_per_portfolio gets NaN."""
    b = beme.assign(month=beme["month"] + 1)[["month", "gvkey", "BEME"]]
    m = members.merge(b, on=["month", "gvkey"], how="inner")
    m = m[m["BEME"] > 0].copy()
    if m.empty:
        return pd.DataFrame(columns=["smb", "hml"])
    g = m.groupby("month")
    m["big"] = m["cap"] > g["cap"].transform("median")
    lo, hi = g["BEME"].transform(lambda s: s.quantile(0.3)), g["BEME"].transform(lambda s: s.quantile(0.7))
    m["bm"] = np.where(m["BEME"] <= lo, "L", np.where(m["BEME"] > hi, "H", "M"))
    m["port"] = np.where(m["big"], "B", "S") + m["bm"]
    size_ok = m.groupby(["month", "port"]).size().unstack("port").reindex(columns=["SL", "SM", "SH", "BL", "BM", "BH"])
    good = size_ok.index[(size_ok.fillna(0) >= min_per_portfolio).all(axis=1)]
    port = m[m["month"].isin(good)][["month", "gvkey", "port", "cap"]]
    r = cd[["gvkey", "date", "ret"]].assign(month=cd["date"].dt.to_period("M"))
    x = r.merge(port, on=["month", "gvkey"])
    x = x[x["ret"].notna()]
    x["wr"] = x["cap"] * x["ret"].astype("float64")
    pr = x.groupby(["date", "port"]).agg(wr=("wr", "sum"), w=("cap", "sum"))
    pr = (pr["wr"] / pr["w"]).unstack("port")
    out = pd.DataFrame(index=pr.index)
    out["smb"] = pr[["SL", "SM", "SH"]].mean(axis=1, skipna=False) - pr[["BL", "BM", "BH"]].mean(axis=1, skipna=False)
    out["hml"] = pr[["SH", "BH"]].mean(axis=1, skipna=False) - pr[["SL", "BL"]].mean(axis=1, skipna=False)
    return out


# ------------------------------------------------------------------ characteristic inputs


def monthly_daily_stats(cd: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """us_panel.monthly_daily_stats on the company series, with the Corwin-Schultz daily
    estimate (column `cs`) in place of the closing-quote spread."""
    d = cd[["gvkey", "date", "ret", "cshtrd", "cs"]].rename(columns={"gvkey": ID, "cshtrd": "vol"})
    d["bid"], d["ask"] = np.nan, np.nan
    st = up.monthly_daily_stats(d, ff)
    cs = d.assign(month=d["date"].dt.to_period("M")).groupby(["month", ID])["cs"]
    st["n_spread"] = cs.count().astype("float32").reindex(st.index)
    st["sum_spread"] = cs.sum(min_count=1).reindex(st.index)
    return st


def daily_inputs(cd: pd.DataFrame, ff: pd.DataFrame, windows: dict, calendar: pd.DatetimeIndex) -> ch.DailyInputs:
    wide = lambda col: (cd.pivot(index="date", columns="gvkey", values=col).reindex(calendar)  # noqa: E731
                        .rename_axis(columns=ID).astype("float32"))
    prc = wide("prc_eur")
    high = (wide("high_eur")).fillna(prc)
    return ch.DailyInputs(ret=wide("ret"), prc=prc, high=high, vol=wide("cshtrd"), cumfacpr=wide("ajexdi"),
                          ff=ff.reindex(calendar), cfg=windows)


def monthly_inputs(mf: pd.DataFrame, ids, stats: pd.DataFrame, D: ch.DailyInputs, windows: dict) -> ch.MonthlyInputs:
    wide = lambda col: mf.pivot(index="month", columns="gvkey", values=col).astype("float64")  # noqa: E731
    ret = wide("ret")
    full = pd.period_range(ret.index.min(), ret.index.max(), freq="M")
    grid = lambda w: w.reindex(index=full, columns=sorted(ids)).rename_axis(columns=ID)  # noqa: E731
    return ch.MonthlyInputs(ret=grid(ret), prc=grid(wide("prc")), vol=grid(wide("vol")), shrout=grid(wide("shrout")),
                            cap_co=grid(wide("cap_co")), stats=stats, daily=D, cfg=windows)
