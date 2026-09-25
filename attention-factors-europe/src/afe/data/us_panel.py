"""Stage 2 building blocks for the US panel: raw tables -> aligned inputs.

Everything in here is about WHO is in the data and WHEN a number is known. The
formulas live in characteristics.py. Reading order:

  load_raw            frozen parquet from stage 1 (wrds_us.py), incl. the extra security
                      types of wrds_us.pull_extra; Compustat amounts converted to USD
  eligible_monthly    CIZ flags -> the pool a stock can be ranked in (config: eligibility)
  receipt_caps        depositary receipts sized at company level, not receipts outstanding
  company_month       permco-level market cap (sum of classes) and the primary permno
  line_month          the same per line, for a universe that ranks every line (config: universe)
  build_universe      top-N by cap at the end of month M-1, applied to month M
  load_daily          daily rows for a set of permnos (plus their delisting-day rows, which
                      the stage-1 filter cannot keep), and per-month sufficient statistics
  daily_inputs        long daily rows -> wide date x permno frames
  monthly_inputs      wide month x permno frames from the monthly file
  permno_gvkey        CRSP permno -> Compustat gvkey by month, via the CCM link table
  annual_frame        fundamentals with lags, book equity, December ME, and the month
                      from which each fiscal year is usable (config: fundamentals)
  align_annual        the usable fiscal year for each (permno, month)

Units: CRSP cap and shrout arrive in thousands; cap is converted to USD millions here
(cap_co, me_dec) so that it matches Compustat's millions. Volume stays in shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from afe import schemas
from afe.data import characteristics as ch
from afe.data import compustat_global, wrds_us


@dataclass
class RawUS:
    monthly: pd.DataFrame
    secinfo: pd.DataFrame
    ccm: pd.DataFrame
    funda: pd.DataFrame
    ff_daily: pd.DataFrame
    ff_monthly: pd.DataFrame
    daily_dir: Path
    delisting: pd.DataFrame = None   # crsp_delisting.parquet; empty frame if not pulled
    daily_extra_dir: Path | None = None   # crsp_daily_extra/: receipts, units, trusts
    secm: pd.DataFrame | None = None      # comp_secm_receipts.parquet: receipts' share base

    def truncate(self, cutoff: pd.Timestamp) -> "RawUS":
        """Everything dated after `cutoff` removed: the input of a prefix-invariance test.
        The daily files are filtered when loaded (see load_daily)."""
        c = pd.Timestamp(cutoff)
        return RawUS(
            monthly=self.monthly[self.monthly["date"] <= c],
            secinfo=self.secinfo, ccm=self.ccm,
            funda=self.funda[self.funda["datadate"] <= c],
            ff_daily=self.ff_daily[self.ff_daily.index <= c],
            ff_monthly=self.ff_monthly[self.ff_monthly["date"] <= c],
            daily_dir=self.daily_dir,
            delisting=self.delisting[self.delisting["date"] <= c],
            daily_extra_dir=self.daily_extra_dir,
            secm=None if self.secm is None else self.secm[self.secm["datadate"] <= c],
        )


def load_raw(raw_dir: Path) -> RawUS:
    """The stage-1 tables. With wrds_us.pull_extra's files present, the monthly file also
    holds the receipts, units and trusts, and comp_funda is converted to USD."""
    rd = lambda n: pd.read_parquet(raw_dir / f"{n}.parquet")  # noqa: E731
    has = lambda n: (raw_dir / n).exists()  # noqa: E731
    monthly = rd("crsp_monthly")
    if "sharetype" not in monthly:
        monthly["sharetype"] = "NS"                      # the base pull keeps plain shares only
    if has("crsp_monthly_extra.parquet"):              # one share type per permno-month: no overlap
        monthly = pd.concat([monthly, rd("crsp_monthly_extra")], ignore_index=True)
        monthly = monthly.drop_duplicates(["permno", "date"], keep="first")
    r = RawUS(monthly, rd("crsp_secinfo"), rd("ccm_link"), rd("comp_funda"),
              rd("ff_daily"), rd("ff_monthly"), raw_dir / "crsp_daily")
    if has("crsp_daily_extra"):
        r.daily_extra_dir = raw_dir / "crsp_daily_extra"
    if has("comp_secm_receipts.parquet"):
        r.secm = rd("comp_secm_receipts")
        r.secm["datadate"] = pd.to_datetime(r.secm["datadate"]).astype("datetime64[ns]")
    for df, col in [(r.monthly, "date"), (r.funda, "datadate"), (r.ff_daily, "date"), (r.ff_monthly, "date"),
                    (r.ccm, "linkdt"), (r.ccm, "linkenddt")]:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    if has("comp_funda_currency.parquet"):
        r.funda = funda_to_usd(r.funda, rd("comp_funda_currency"), rd("fx_daily"))
    for col in ("permno", "permco"):  # plain int64: nullable Int32 does not survive pivots and merges well
        r.monthly[col] = r.monthly[col].astype("int64")
        r.ccm[col] = r.ccm[col].astype("int64")
    r.ff_daily = r.ff_daily.set_index("date").sort_index()
    dl = raw_dir / "crsp_delisting.parquet"
    r.delisting = pd.read_parquet(dl) if dl.exists() else pd.DataFrame(columns=DAILY_COLS)
    r.delisting["date"] = pd.to_datetime(r.delisting["date"]).astype("datetime64[ns]")
    for col in ("permno", "permco"):
        r.delisting[col] = pd.to_numeric(r.delisting[col]).astype("int64")
    return r


# comp_funda items that are amounts in the reporting currency. csho (shares), ajex (a
# factor) and prcc_f (not used downstream) stay as reported.
FUNDA_AMOUNTS = [c for c in wrds_us.FUNDA_ITEMS if c not in ("csho", "ajex", "prcc_f")]


def funda_to_usd(funda: pd.DataFrame, currency: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Compustat North America reports Canadian filers in CAD (`curcd`); every other row,
    depositary-receipt issuers included, is in USD. Amounts are converted at the rate of
    the fiscal year end (datadate), the latest daily rate at most a week old."""
    cur = currency[["gvkey", "datadate", "curcd"]].copy()
    cur["datadate"] = pd.to_datetime(cur["datadate"]).astype("datetime64[ns]")
    f = funda.merge(cur, on=["gvkey", "datadate"], how="left")
    f["curcd"] = f["curcd"].fillna("USD")
    fx = fx.assign(datadate=pd.to_datetime(fx["datadate"]).astype("datetime64[ns]"))
    f["fx_usd"] = compustat_global.convert_to_currency(f.rename(columns={"curcd": "curcdd"}), fx, "USD")
    f[FUNDA_AMOUNTS] = f[FUNDA_AMOUNTS].mul(f["fx_usd"], axis=0)
    return f


# ------------------------------------------------------------------ eligibility, companies


def eligible_monthly(monthly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Monthly rows that may enter the universe: the share types, incorporation, issuer
    types and exchanges of config eligibility, with a positive market cap."""
    e = cfg["eligibility"]
    m = monthly[monthly["usincflg"].isin(e["usincflg"]) & monthly["issuertype"].isin(e["issuertype"])
                & monthly["primaryexch"].isin(e["primaryexch"])
                & monthly["sharetype"].isin(e.get("sharetype", ["NS"])) & (monthly["cap"] > 0)]
    if e.get("min_price", 0) > 0:
        m = m[m["prc"].abs() >= e["min_price"]]
    m = m.copy()
    m["month"] = m["date"].dt.to_period("M")
    return m


def company_month(elig: pd.DataFrame) -> pd.DataFrame:
    """One row per (permco, month): company cap in USD millions summed over eligible
    share classes, and the class with the largest cap as `primary_permno`."""
    co = (elig.groupby(["permco", "month"], sort=False)
          .agg(cap_co=("cap", "sum"), n_classes=("permno", "size")).reset_index())
    co["cap_co"] = co["cap_co"] / 1000.0
    prim = (elig.sort_values(["permco", "month", "cap"], ascending=[True, True, False])
            .drop_duplicates(["permco", "month"])[["permco", "month", "permno"]]
            .rename(columns={"permno": "primary_permno"}))
    return co.merge(prim, on=["permco", "month"])


def line_month(elig: pd.DataFrame, co: pd.DataFrame) -> pd.DataFrame:
    """One row per (permno, month): every eligible line ranked on its own cap, the paper's
    "500 largest stocks" (its Figure 3 has Alphabet twice). Same columns as company_month
    so build_universe ranks either: primary_permno is the line, cap_co its cap (USD mn),
    n_classes the number of eligible lines of its company."""
    ln = elig[["permno", "permco", "month", "cap"]].rename(columns={"permno": "primary_permno"})
    ln = ln.assign(cap_co=ln.pop("cap") / 1000.0)
    return ln.merge(co[["permco", "month", "n_classes"]], on=["permco", "month"], how="left")


def receipt_caps(elig: pd.DataFrame, secm: pd.DataFrame | None, ccm: pd.DataFrame, funda: pd.DataFrame,
                 lag_months: int = 6, max_age_months: int = 30,
                 max_fallback_ratio: float = 20_000.0) -> pd.DataFrame:
    """`elig` with the cap of each depositary receipt (sharetype AD) at company level:
    shares in receipt equivalents x the receipt's close, in CRSP units (USD thousands).
    CRSP's own cap counts only receipts outstanding (Mizuho 2020-12: $0.2bn, company $32bn).

      1. comp.secm of the linked Compustat issue: underlying shares of the class over the
         receipt ratio, cshom / adrrm (monthly; cshom exists from 1998-04);
      2. else comp.funda csho, which Compustat states in receipt equivalents, from the
         latest fiscal year that ended at least `lag_months` before the month (a 20-F is
         due six months after the year end) and at most `max_age_months`, carried to the
         month with CRSP's cumulative price factor so that a split in between does not bite.
    Some funda csho are ordinary shares, not receipt equivalents, which inflates the cap by
    the receipt ratio (Smith & Nephew 1999, Centaur Mining 2000). Only data known at the
    time may judge it, so a fallback above `max_fallback_ratio` x CRSP's cap is dropped
    (those two: 111,000x and 34,000x; secm caps reach 24,000x legitimately, Sanofi 2002,
    so the rule is for the fallback only; smaller errors, Fiat's ~4x, stay).
    Never below CRSP's cap (a class-A-only cshom understates, e.g. Baidu). CRSP's cap is
    kept as cap_crsp; cap_source says which of secm / funda / crsp was used."""
    out = elig.copy()
    out["cap_crsp"], out["cap_source"] = out["cap"], "crsp"
    ad = out.loc[out["sharetype"] == "AD", ["permno", "permco", "month", "prc", "cap", "cumfacpr"]]
    ad = ad.rename_axis("_i").reset_index()
    if ad.empty:
        return out
    ad["key"] = ad["month"].dt.to_timestamp(how="end").dt.normalize().astype("datetime64[ns]")
    links = ccm[ccm["permno"].isin(ad["permno"].unique())].copy()        # the issue linked at month end
    links["linkenddt"] = links["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    links["prio"] = links["linkprim"].map(LINK_PRIORITY)
    lk = ad[["_i", "permno", "key"]].merge(links[["permno", "gvkey", "liid", "linkdt", "linkenddt", "prio"]],
                                           on="permno")
    lk = lk[(lk["linkdt"] <= lk["key"]) & (lk["key"] <= lk["linkenddt"])]
    ad = ad.merge(lk.sort_values(["_i", "prio"]).drop_duplicates("_i")[["_i", "gvkey", "liid"]], on="_i", how="left")

    ad["sh"] = np.nan                                                   # 1. secm
    if secm is not None and len(secm):
        s = secm[(secm["cshom"] > 0) & (secm["adrrm"] > 0)].copy()
        s["month"], s["sh"] = s["datadate"].dt.to_period("M"), s["cshom"] / s["adrrm"]
        s = s.rename(columns={"iid": "liid"}).drop_duplicates(["gvkey", "liid", "month"])
        # A receipt whose own link lapsed (Shell's B line, 2005-2022) takes the gvkey of a linked
        # sibling line of its permco and the other issue of that gvkey whose close matches its own.
        sib = ad.loc[ad["gvkey"].notna(), ["permco", "month", "gvkey", "liid"]]
        c = ad.loc[ad["gvkey"].isna(), ["_i", "permco", "month", "prc"]].merge(
            sib[["permco", "month", "gvkey"]].drop_duplicates(["permco", "month"]), on=["permco", "month"])
        c = c.merge(s[["gvkey", "liid", "month", "prccm"]], on=["gvkey", "month"])
        c = c.merge(sib.assign(used=True), on=["permco", "month", "gvkey", "liid"], how="left")
        c = c[c["used"].isna() & ((c["prccm"] / c["prc"].abs() - 1).abs() < 0.02)].drop_duplicates("_i")
        ad = ad.set_index("_i")
        ad.loc[c["_i"], ["gvkey", "liid"]] = c[["gvkey", "liid"]].to_numpy()
        ad = ad.reset_index()
        ad = ad.drop(columns="sh").merge(s[["gvkey", "liid", "month", "sh"]], on=["gvkey", "liid", "month"], how="left")
    ad["src"] = np.where(ad["sh"].notna(), "secm", "crsp")

    fa = funda.loc[funda["csho"] > 0, ["gvkey", "datadate", "csho"]].copy()   # 2. funda
    fa["key"] = ((fa["datadate"] + pd.DateOffset(months=lag_months)).dt.to_period("M")
                 .dt.to_timestamp(how="end").dt.normalize().astype("datetime64[ns]"))
    fb = pd.merge_asof(ad.loc[ad["gvkey"].notna(), ["_i", "gvkey", "key"]].sort_values("key"),
                       fa.sort_values("key"), on="key", by="gvkey", direction="backward")
    fb["month"] = fb["datadate"].dt.to_period("M")
    fac = out.loc[out["sharetype"] == "AD", ["permno", "month", "cumfacpr"]].drop_duplicates(["permno", "month"])
    fb = fb.merge(ad[["_i", "permno", "cumfacpr"]], on="_i")
    fb = fb.merge(fac.rename(columns={"cumfacpr": "fac_fy"}), on=["permno", "month"], how="left")
    carry = (fb["fac_fy"] / fb["cumfacpr"]).where(lambda v: v > 0).fillna(1.0)
    age = (fb["key"].dt.year - fb["datadate"].dt.year) * 12 + (fb["key"].dt.month - fb["datadate"].dt.month)
    fb["sh_f"] = (fb["csho"] * 1e6 * carry).where(age <= max_age_months)
    ad = ad.merge(fb[["_i", "sh_f"]], on="_i", how="left")
    ad.loc[ad["prc"].abs() * ad["sh_f"] / 1000.0 > max_fallback_ratio * ad["cap"], "sh_f"] = np.nan
    use_f = ad["sh"].isna() & ad["sh_f"].notna()
    ad.loc[use_f, "sh"], ad.loc[use_f, "src"] = ad.loc[use_f, "sh_f"], "funda"

    cap = ad["prc"].abs() * ad["sh"] / 1000.0
    better = cap.notna() & (cap > ad["cap"])
    out.loc[ad.loc[better, "_i"], "cap"] = cap[better].to_numpy()
    out.loc[ad.loc[better, "_i"], "cap_source"] = ad.loc[better, "src"].to_numpy()
    return out


def build_universe(co: pd.DataFrame, cfg: dict, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membership for month M = the `universe_size` largest rows of `co` (companies from
    company_month, or lines from line_month) by cap_co at the end of month M-1. Returns
    (universe per docs/schemas.md, as-of ranking table).

    `month` is the first trading day of M on `calendar`. sec_id is the row's (primary)
    permno AT THE RANKING DATE, as a string.
    """
    n = int(cfg["sample"]["universe_size"])
    start, end = pd.Period(cfg["sample"]["start"], "M"), pd.Period(cfg["sample"]["end"], "M")
    first_day = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).min()
    by_month = {m: g for m, g in co.groupby("month")}
    asof, rows = [], []
    for M in pd.period_range(start, end, freq="M"):
        snap = by_month.get(M - 1, co.iloc[:0]).nlargest(n, "cap_co")
        if len(snap) < n:
            raise ValueError(f"{M}: only {len(snap)} eligible companies at the end of {M - 1}")
        if M not in first_day.index:
            raise ValueError(f"{M}: no trading day on the calendar")
        snap = snap.assign(cap_rank=np.arange(1, n + 1, dtype="int16"), month=M)
        asof.append(snap)
        rows.append(pd.DataFrame({"month": first_day[M], "sec_id": snap["primary_permno"].astype(str).values,
                                  "cap_rank": snap["cap_rank"].values}))
    uni = pd.concat(rows, ignore_index=True)
    uni["month"] = uni["month"].astype("datetime64[ns]")
    uni = uni.sort_values(["month", "sec_id"], ignore_index=True)
    schemas.validate_universe(uni, size=n)
    return uni, pd.concat(asof, ignore_index=True)


# ------------------------------------------------------------------ daily data


DAILY_COLS = ["permno", "permco", "date", "ret", "prc", "vol", "bid", "ask", "high", "cap", "shrout", "cumfacpr"]


def load_daily(daily_dir: Path | list[Path], permnos, years, ff_daily: pd.DataFrame,
               cutoff: pd.Timestamp | None = None, delisting: pd.DataFrame | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily rows for `permnos` over `years`, and the per-(permno, month) statistics
    that the monthly-from-daily characteristics are built from. One year at a time so
    50M rows never sit in memory together. `daily_dir` may be a list (crsp_daily plus
    crsp_daily_extra): a permno has one share type on a date, so the files do not overlap.

    `delisting` (crsp_delisting.parquet) supplies each permno's delisting-day row, whose
    dlyret is the delisting return; the yearly files cannot contain it (module docstring
    of wrds_us). It is appended to the year's rows; the daily-file row wins if both exist."""
    permnos = [int(p) for p in permnos]
    dirs = [daily_dir] if isinstance(daily_dir, Path) else list(daily_dir)
    dl = pd.DataFrame(columns=DAILY_COLS) if delisting is None else delisting[delisting["permno"].isin(permnos)]
    frames, stats = [], []
    for y in years:
        parts = [pq.read_table(d / f"{y}.parquet", columns=DAILY_COLS, filters=[("permno", "in", permnos)]).to_pandas()
                 for d in dirs if (d / f"{y}.parquet").exists()]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        # plain float64: the parquet metadata restores nullable Float64/Int32, whose to_numpy() is
        # an object array, and a receipt-month with returns but no volume then divides a Python
        # 0.0 in SUV (ZeroDivisionError, 2026-09-23). NA becomes NaN, values are unchanged.
        num = [c for c in DAILY_COLS if c not in ("permno", "permco", "date")]
        df[num] = df[num].astype("float64")
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df["permno"], df["permco"] = df["permno"].astype("int64"), df["permco"].astype("int64")
        extra = dl[dl["date"].dt.year == y][DAILY_COLS] if len(dl) else dl
        if len(extra):
            df = pd.concat([df, extra.astype(df.dtypes.to_dict())], ignore_index=True)
            df = df.drop_duplicates(["permno", "date"], keep="first")
        if cutoff is not None:
            df = df[df["date"] <= cutoff]
        if len(df) == 0:
            continue
        df["prc"] = df["prc"].abs()
        frames.append(df)
        stats.append(monthly_daily_stats(df, ff_daily))
    daily = pd.concat(frames, ignore_index=True).sort_values(["permno", "date"], ignore_index=True)
    return daily, pd.concat(stats).sort_index()


def monthly_daily_stats(df: pd.DataFrame, ff_daily: pd.DataFrame) -> pd.DataFrame:
    """Per-(month, permno) sums of daily quantities. Each monthly characteristic that
    looks at daily data (Variance, Resid_Var, Spread, SUV) is an exact function of these
    sums pooled over its window, so nothing daily has to be re-read per month.

      n_ret, sum_ret, sum_ret2      returns                      -> Variance
      x_<a>_<b>, xy_<a>, sum_exret2 FF3 cross products           -> Resid_Var
      n_spread, sum_spread          relative closing spread      -> Spread
      n_vol, sum_vol, sum_vol2, sum_rp, sum_rn, sum_rp2, sum_rn2, sum_rp_vol, sum_rn_vol
                                    volume on |r+| and |r-|      -> SUV
    """
    d = df[["permno", "date", "ret", "vol", "bid", "ask"]].merge(
        ff_daily[["mktrf", "smb", "hml", "rf"]], left_on="date", right_index=True, how="left")
    d["month"] = d["date"].dt.to_period("M")
    r = d["ret"]
    out = pd.DataFrame({"permno": d["permno"], "month": d["month"]})
    out["n_ret"] = r.notna().astype("float32")
    out["sum_ret"] = r
    out["sum_ret2"] = r ** 2

    exret = r - d["rf"]
    x = {"one": exret.notna().astype("float64"), "mktrf": d["mktrf"], "smb": d["smb"], "hml": d["hml"]}
    for i, a in enumerate(ch.FF3_X):
        out[f"xy_{a}"] = x[a] * exret
        for b in ch.FF3_X[i:]:
            out[ch.cross_key(a, b)] = (x[a] * x[b]).where(exret.notna())
    out["sum_exret2"] = exret ** 2

    mid = (d["ask"] + d["bid"]) / 2
    rel = ((d["ask"] - d["bid"]) / mid).where((d["bid"] > 0) & (d["ask"] > d["bid"]))
    out["n_spread"] = rel.notna().astype("float32")
    out["sum_spread"] = rel

    v = d["vol"].where(r.notna())
    rp, rn = r.clip(lower=0).where(v.notna()), (-r).clip(lower=0).where(v.notna())
    out["n_vol"] = v.notna().astype("float32")
    out["sum_vol"], out["sum_vol2"] = v, v ** 2
    out["sum_rp"], out["sum_rn"], out["sum_rp2"], out["sum_rn2"] = rp, rn, rp ** 2, rn ** 2
    out["sum_rp_vol"], out["sum_rn_vol"] = rp * v, rn * v
    return out.groupby(["month", "permno"]).sum(min_count=1)


def daily_inputs(daily: pd.DataFrame, ff_daily: pd.DataFrame, windows: dict) -> ch.DailyInputs:
    wide = lambda col: daily.pivot(index="date", columns="permno", values=col).astype("float32")  # noqa: E731
    ret = wide("ret")
    prc = wide("prc")
    high = wide("high").fillna(prc)
    ff = ff_daily.reindex(ret.index)
    return ch.DailyInputs(ret=ret, prc=prc, high=high, vol=wide("vol"), cumfacpr=wide("cumfacpr"), ff=ff, cfg=windows)


def monthly_inputs(monthly: pd.DataFrame, co: pd.DataFrame, permnos, stats: pd.DataFrame,
                   D: ch.DailyInputs, windows: dict) -> ch.MonthlyInputs:
    """Wide monthly frames for `permnos`. The return/price/volume series are the
    permno's own (unfiltered, so a spell outside the eligible pool does not punch a hole
    in its momentum); cap_co is the company cap from the eligible classes."""
    m = monthly[monthly["permno"].isin(permnos)].copy()
    m["month"] = m["date"].dt.to_period("M")
    m["prc"] = m["prc"].abs()
    m = m.merge(co[["permco", "month", "cap_co"]], on=["permco", "month"], how="left")
    wide = lambda col: m.pivot(index="month", columns="permno", values=col).astype("float64")  # noqa: E731
    ret = wide("ret")
    full = pd.period_range(ret.index.min(), ret.index.max(), freq="M")
    grid = lambda w: w.reindex(index=full, columns=sorted(permnos))  # noqa: E731
    return ch.MonthlyInputs(ret=grid(ret), prc=grid(wide("prc")), vol=grid(wide("vol")),
                            shrout=grid(wide("shrout")), cap_co=grid(wide("cap_co")),
                            stats=stats, daily=D, cfg=windows)


# ------------------------------------------------------------------ Compustat side


LINK_PRIORITY = {"P": 0, "C": 1, "J": 2}


def permno_gvkey(ccm: pd.DataFrame, permnos, months: pd.PeriodIndex) -> pd.DataFrame:
    """(permno, month) -> gvkey. A link applies at month end m if linkdt <= m <= linkenddt;
    a primary link (P, then C) wins over a secondary-class link (J)."""
    links = ccm[ccm["permno"].isin(list(permnos))].copy()
    links["linkenddt"] = links["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    links["prio"] = links["linkprim"].map(LINK_PRIORITY)
    grid = pd.MultiIndex.from_product([sorted(permnos), months], names=["permno", "month"]).to_frame(index=False)
    grid["m_end"] = grid["month"].dt.to_timestamp(how="end").dt.normalize()
    x = grid.merge(links[["permno", "gvkey", "linkdt", "linkenddt", "prio"]], on="permno", how="inner")
    x = x[(x["linkdt"] <= x["m_end"]) & (x["m_end"] <= x["linkenddt"])]
    x = x.sort_values(["permno", "month", "prio"]).drop_duplicates(["permno", "month"])
    return x[["permno", "month", "gvkey"]].reset_index(drop=True)


def gvkey_permco(ccm: pd.DataFrame) -> pd.DataFrame:
    """Company links with a permco, for looking up December market equity."""
    links = ccm[ccm["permco"].notna()].copy()
    links["linkenddt"] = links["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    links["prio"] = links["linkprim"].map(LINK_PRIORITY)
    return links[["gvkey", "permco", "linkdt", "linkenddt", "prio"]]


LAGGED = ["at", "ppegt", "invt", "wcap", "noa_level", "nwc"]


def annual_frame(funda: pd.DataFrame, ccm: pd.DataFrame, co: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (gvkey, datadate) with the items, their one-year lags, book equity,
    net operating assets, non-cash working capital, December market equity and the
    month from which the record may be used (`avail_month`)."""
    fc = cfg["fundamentals"]
    f = funda.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last").copy()
    for c in fc["fill_zero"]:
        f[c] = f[c].fillna(0.0)
    f["be"] = ch.book_equity(f)
    f["noa_level"] = ch.net_operating_assets(f)
    f["nwc"] = ch.noncash_working_capital(f)

    # lag = the previous fiscal year of the same company (fyear exactly one less)
    prev = f[["gvkey", "fyear"] + LAGGED].copy()
    prev["fyear"] = prev["fyear"] + 1
    f = f.merge(prev.rename(columns={c: f"{c}_lag" for c in LAGGED}), on=["gvkey", "fyear"], how="left")

    # December market equity of the fiscal year's calendar year, company level (USD mn)
    f["dec_month"] = pd.PeriodIndex.from_fields(year=f["datadate"].dt.year, month=12, freq="M")
    f["dec_end"] = f["dec_month"].dt.to_timestamp(how="end").dt.normalize()
    lk = f[["gvkey", "dec_end"]].drop_duplicates().merge(gvkey_permco(ccm), on="gvkey", how="inner")
    lk = lk[(lk["linkdt"] <= lk["dec_end"]) & (lk["dec_end"] <= lk["linkenddt"])]
    lk = lk.sort_values(["gvkey", "dec_end", "prio"]).drop_duplicates(["gvkey", "dec_end"])
    lk["dec_month"] = lk["dec_end"].dt.to_period("M")
    lk = lk.merge(co[["permco", "month", "cap_co"]].rename(columns={"month": "dec_month", "cap_co": "me_dec"}),
                  on=["permco", "dec_month"], how="left")
    f = f.merge(lk[["gvkey", "dec_end", "me_dec"]], on=["gvkey", "dec_end"], how="left")

    if fc["availability"] == "june":
        f["avail_month"] = pd.PeriodIndex.from_fields(year=f["datadate"].dt.year + 1, month=6, freq="M")
    elif fc["availability"] == "lag_months":
        f["avail_month"] = (f["datadate"] + pd.DateOffset(months=int(fc["lag_months"]))).dt.to_period("M")
    else:
        raise ValueError(f"fundamentals.availability: {fc['availability']}")
    return f


def align_annual(annual: pd.DataFrame, mapping: pd.DataFrame, cols: list[str], max_age_months: int,
                 last_observed: bool = False) -> pd.DataFrame:
    """For each (permno, month, gvkey) in `mapping`, the latest fiscal record whose
    avail_month <= month and whose datadate is at most max_age_months old. Returns
    (permno, month) + `cols`. With last_observed, each column comes instead from the latest
    such record in which it is not missing: the stock's last observed value."""
    left = mapping.copy()
    left["key"] = left["month"].dt.to_timestamp(how="end").dt.normalize()
    left = left.sort_values("key", ignore_index=True)
    right = annual[["gvkey", "datadate", "avail_month"] + cols].copy()
    right["key"] = right["avail_month"].dt.to_timestamp(how="end").dt.normalize()
    # A fiscal-year-end change puts two records in one calendar year (AMD: 1987-03 and 1987-12),
    # both usable from the same June; merge_asof takes the last tie, so order ties by datadate.
    # sort_values("key") alone is unstable and picked either, depending on the frame's length,
    # which broke prefix invariance (found 2026-09-23 with the 2010-09-15 cutoff build).
    right = right.sort_values(["key", "datadate"], kind="mergesort")
    out = left[["permno", "month"]].copy()
    for group in ([[c] for c in cols] if last_observed else [cols]):
        r = right.dropna(subset=group) if last_observed else right
        m = pd.merge_asof(left, r[["gvkey", "key", "datadate"] + group], on="key", by="gvkey",
                          direction="backward")
        age = (m["key"].dt.year - m["datadate"].dt.year) * 12 + (m["key"].dt.month - m["datadate"].dt.month)
        m.loc[~(age <= max_age_months), group] = np.nan
        out[group] = m[group].to_numpy()
    return out
