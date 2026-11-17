"""Stage 2 building blocks for the US panel: raw tables -> aligned inputs.

Everything in here is about WHO is in the data and WHEN a number is known. The
formulas live in characteristics.py. Reading order:

  load_raw            frozen parquet from stage 1 (wrds_us.py), incl. the extra security
                      types of wrds_us.pull_extra; Compustat amounts converted to USD
  eligible_monthly    CIZ flags -> the pool a stock can be ranked in (config: eligibility)
  receipt_caps        depositary receipts sized at company level, not receipts outstanding
  home_caps           the receipt issuers' caps in their home market (Compustat Global), the
                      check receipt_caps holds its share counts against
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
from afe.data import compustat_global, hl_spread, wrds_us


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
    global_home: dict | None = None       # global_home/: issuers' home-market month-end rows
    daily_low_dir: Path | None = None     # crsp_daily_low/: daily low, for the Corwin-Schultz spread

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
            daily_low_dir=self.daily_low_dir,
            global_home=None if self.global_home is None else
            {**self.global_home, "secd": self.global_home["secd"][self.global_home["secd"]["datadate"] <= c]},
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
    if has("crsp_daily_low"):
        r.daily_low_dir = raw_dir / "crsp_daily_low"
    if has("global_home/secd_monthend.parquet"):
        gh = {k: pd.read_parquet(raw_dir / "global_home" / f"{f}.parquet")
              for k, f in [("secd", "secd_monthend"), ("security", "security"), ("company", "company"), ("fx", "fx_daily")]}
        for k in ("secd", "fx"):
            gh[k]["datadate"] = pd.to_datetime(gh[k]["datadate"]).astype("datetime64[ns]")
        r.global_home = gh
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
    types and exchanges of config eligibility, with a positive market cap, less the lines
    of `universe.exclude_permnos` (receipts whose share count makes a micro cap a top-500
    company and that no home-market cap can check; audit A8, 2026-11-17)."""
    e = cfg["eligibility"]
    m = monthly[monthly["usincflg"].isin(e["usincflg"]) & monthly["issuertype"].isin(e["issuertype"])
                & monthly["primaryexch"].isin(e["primaryexch"])
                & monthly["sharetype"].isin(e.get("sharetype", ["NS"])) & (monthly["cap"] > 0)]
    if e.get("min_price", 0) > 0:
        m = m[m["prc"].abs() >= e["min_price"]]
    excl = [int(x["permno"]) for x in cfg.get("universe", {}).get("exclude_permnos", []) or []]
    if excl:
        m = m[~m["permno"].isin(excl)]
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


def home_caps(gh: dict | None) -> pd.DataFrame | None:
    """Per (gvkey, month): the receipt issuer's cap in its home market from Compustat Global,
    in USD thousands like CRSP's cap. Each exchange country is sized on its own
    (compustat_global.company_month_mktcap: listings collapsed to classes, classes added up,
    preferred shares included, the country left out when its most active listing is dormant)
    and the largest country is the home: one gvkey can hold both halves of a dual listing
    (Rio Tinto plc + Ltd, BHP Ltd + plc), whose more active half need not be the receipt's.
    `n_classes_home` counts that country's classes."""
    if gh is None:
        return None
    parts = []
    for country, rows in gh["secd"].groupby("excntry"):
        cap = compustat_global.company_month_mktcap(rows, gh["company"], gh["security"], gh["fx"], [country], "USD",
                                                    issue_types=("0", "1", "8"), genussschein_countries=("CHE",))
        parts.append(cap.loc[cap["active"], ["gvkey", "datadate", "mktcap", "n_classes"]])
    caps = pd.concat(parts, ignore_index=True)
    caps["mktcap"] = caps["mktcap"].astype("float64") * 1000.0
    top = caps.sort_values(["gvkey", "datadate", "mktcap"], ascending=[True, True, False]).drop_duplicates(["gvkey", "datadate"])
    return pd.DataFrame({"gvkey": top["gvkey"].astype(str), "month": top["datadate"].dt.to_period("M"),
                         "cap_home": top["mktcap"], "n_classes_home": top["n_classes"].astype("int64")})


def receipt_caps(elig: pd.DataFrame, secm: pd.DataFrame | None, ccm: pd.DataFrame, funda: pd.DataFrame,
                 lag_months: int = 6, max_age_months: int = 30,
                 max_fallback_ratio: float = 20_000.0, home: pd.DataFrame | None = None,
                 home_tol: float = 1.5, home_max_ratio: float = 100.0, carry_links: bool = False) -> pd.DataFrame:
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
    so the rule is for the fallback only; smaller errors, Fiat's ~4x, stay) where the
    issuer has no home-market cap that month; where it has one, rule 3 is the check.
    Never below CRSP's cap (a class-A-only cshom understates, e.g. Baidu), unless rule 3 says so.
      3. Where the issuer's home-market cap is known (`home`, from home_caps: same month,
         Compustat Global) it checks the result: a receipt cap above `home_tol` x the whole
         company counts its shares in the wrong unit and takes the home cap, whatever its
         source (funda csho in ordinary shares: Fiat 1990-2000 ~5x, Amersham 1999-2003 ~5x;
         a stale secm receipt ratio: CNOOC 2004 20 for 100, Turkcell 2003 250 for 2,500;
         CRSP counting underlying shares as receipts: TIM Participacoes 2011-20 ~5x).
         Same-month market data: point in time. Beyond `home_max_ratio` it is the home cap
         that is broken (De Beers 1995-2001, 3,000x below its receipt cap) and nothing is
         replaced; unit errors are 2-25x. A receipt cap BELOW its home company is left
         alone: a receipt of one class of several is rightly below, and a raise would import
         Global's own errors (Hong Kong Telecom 1994 to rank 1) or another entity's history
         behind a re-used gvkey (MUFG 1990-96); an understated receipt is also what the
         paper's CRSP caps are.
    CRSP's cap is kept as cap_crsp; cap_source says which of secm / funda / crsp / home was used."""
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
    if carry_links:
        # A receipt still without a link keeps the last link its own line had (one that has ended
        # by then: point in time), so the issuer's share base and the home-market check of rule 3
        # still apply. CCM has no link for AmBev's receipt from 1998-04 to 2013-11, which left it
        # at CRSP's receipts-only cap for 15 years (Kevin's data check, 2026-11-17).
        past = links[links["linkenddt"] < pd.Timestamp("2099-12-31")][["permno", "gvkey", "liid", "linkenddt"]]
        need = ad.loc[ad["gvkey"].isna(), ["_i", "permno", "key"]].sort_values("key")
        if len(need) and len(past):
            cf = pd.merge_asof(need, past.sort_values("linkenddt"), left_on="key", right_on="linkenddt", by="permno",
                               direction="backward")
            cf = cf[cf["gvkey"].notna()]
            ad = ad.set_index("_i")
            ad.loc[cf["_i"], ["gvkey", "liid"]] = cf[["gvkey", "liid"]].to_numpy()
            ad = ad.reset_index()
    if secm is not None and len(secm):
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
    # The guard is for counts nothing else can check: where the home market sizes the issuer,
    # rule 3 judges the count instead (a tiny receipt programme made a right count look
    # 26,000x CRSP's: Mitsubishi Bank 1990-12, $43bn against $1.6m; audit B2, 2026-11-17).
    homed = (ad[["gvkey", "month"]].merge(home[["gvkey", "month"]].drop_duplicates().assign(_h=True),
                                          on=["gvkey", "month"], how="left")["_h"].notna().to_numpy()
             if home is not None and len(home) else np.zeros(len(ad), dtype=bool))
    ad.loc[(ad["prc"].abs() * ad["sh_f"] / 1000.0 > max_fallback_ratio * ad["cap"]) & ~homed, "sh_f"] = np.nan
    use_f = ad["sh"].isna() & ad["sh_f"].notna()
    ad.loc[use_f, "sh"], ad.loc[use_f, "src"] = ad.loc[use_f, "sh_f"], "funda"

    cap = ad["prc"].abs() * ad["sh"] / 1000.0
    better = cap.notna() & (cap > ad["cap"])
    out.loc[ad.loc[better, "_i"], "cap"] = cap[better].to_numpy()
    out.loc[ad.loc[better, "_i"], "cap_source"] = ad.loc[better, "src"].to_numpy()

    if home is not None and len(home):                                  # 3. the home market
        h = ad.loc[ad["gvkey"].notna(), ["_i", "gvkey", "month"]].merge(home, on=["gvkey", "month"])
        cur = out.loc[h["_i"], "cap"].to_numpy(dtype=float)
        hc = h["cap_home"].to_numpy()
        use = (cur > home_tol * hc) & (cur <= home_max_ratio * hc)
        out.loc[h.loc[use, "_i"], "cap"] = hc[use]
        out.loc[h.loc[use, "_i"], "cap_source"] = "home"
    return out


def build_universe(co: pd.DataFrame, cfg: dict, calendar: pd.DatetimeIndex,
                   dropped: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membership for month M = the `universe_size` largest rows of `co` (companies from
    company_month, or lines from line_month) by cap_co at the end of month M-1. Returns
    (universe per docs/schemas.md, as-of ranking table).

    `sample.universe_first_rank` (default 1) starts the window further down: 301 with size
    500 keeps the rows ranked 301-800. cap_rank is then 1..N within the window (the schema's
    dense ranks); the as-of table's cap_rank_all keeps the rank among all rows.

    `dropped` (month -> permnos) are ranked with the rest at the end of M-1 and then left out,
    not replaced: M keeps the plain window's members without them, so it has fewer names,
    cap_rank 1..m among those left and cap_rank_all still counting the ones taken out.

    `month` is the first trading day of M on `calendar`. sec_id is the row's (primary)
    permno AT THE RANKING DATE, as a string.
    """
    n = int(cfg["sample"]["universe_size"])
    first = int(cfg["sample"].get("universe_first_rank", 1))
    if first < 1:
        raise ValueError(f"sample.universe_first_rank must be >= 1, not {first}")
    start, end = pd.Period(cfg["sample"]["start"], "M"), pd.Period(cfg["sample"]["end"], "M")
    first_day = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).min()
    by_month = {m: g for m, g in co.groupby("month")}
    asof, rows = [], []
    for M in pd.period_range(start, end, freq="M"):
        eligible = by_month.get(M - 1, co.iloc[:0])
        snap = eligible.nlargest(first - 1 + n, "cap_co").iloc[first - 1:]
        if len(snap) < n:
            raise ValueError(f"{M}: only {len(eligible)} eligible companies at the end of {M - 1}")
        if M not in first_day.index:
            raise ValueError(f"{M}: no trading day on the calendar")
        snap = snap.assign(cap_rank=np.arange(1, n + 1, dtype="int16"),
                           cap_rank_all=np.arange(first, first + n, dtype="int16"), month=M)
        if dropped is not None:
            snap = snap[~snap["primary_permno"].isin(dropped.get(M - 1, ()))]
            snap = snap.assign(cap_rank=np.arange(1, len(snap) + 1, dtype="int16"))
        asof.append(snap)
        rows.append(pd.DataFrame({"month": first_day[M], "sec_id": snap["primary_permno"].astype(str).values,
                                  "cap_rank": snap["cap_rank"].values}))
    uni = pd.concat(rows, ignore_index=True)
    uni["month"] = uni["month"].astype("datetime64[ns]")
    uni = uni.sort_values(["month", "sec_id"], ignore_index=True)
    schemas.validate_universe(uni, size=None if dropped is not None else n)
    return uni, pd.concat(asof, ignore_index=True)


def receipt_activity(extra_dir: Path | None, permnos, years, window: int,
                     cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """(permno, month, traded_share): the share of the trading days of months m-window+1..m
    on which the receipt line traded (volume > 0), from its daily rows (crsp_daily_extra).
    On a day without a trade CRSP prices a line at its quote midpoint, so a receipt that
    trades on few days has a return series of stale and quote-driven prices (Daiei 87% of
    its member-days without volume, Kubota 42%, Mitsui 41%; audit B1, 2026-11-17)."""
    permnos = [int(p) for p in permnos]
    if extra_dir is None or not permnos:
        return pd.DataFrame(columns=["permno", "month", "traded_share"])
    parts = []
    for y in years:
        f = extra_dir / f"{y}.parquet"
        if f.exists():
            t = pq.read_table(f, columns=["permno", "date", "vol"], filters=[("permno", "in", permnos)]).to_pandas()
            t["date"] = pd.to_datetime(t["date"])
            if cutoff is not None:
                t = t[t["date"] <= cutoff]
            t = t.assign(month=t["date"].dt.to_period("M"), traded=(t["vol"].astype("float64") > 0),
                         permno=t["permno"].astype("int64"))
            parts.append(t.groupby(["permno", "month"]).agg(days=("traded", "size"), traded=("traded", "sum")))
    if not parts:
        return pd.DataFrame(columns=["permno", "month", "traded_share"])
    m = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index().sort_values(["permno", "month"])
    out = []
    for p, g in m.groupby("permno", sort=False):
        g = g.set_index("month").reindex(pd.period_range(g["month"].min(), g["month"].max(), freq="M"), fill_value=0)
        roll = g[["days", "traded"]].rolling(window, min_periods=1).sum()
        out.append(pd.DataFrame({"permno": p, "month": roll.index,
                                 "traded_share": (roll["traded"] / roll["days"].where(roll["days"] > 0)).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def listed_at_month_end(daily_dirs: list[Path], years, delisting: pd.DataFrame | None, within_days: int,
                        cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """(permno, month, listed): whether a line is still listed at the end of the month: its last
    daily return in the month is at most `within_days` calendar days before the month end, and
    it has no delisting row (crsp_delisting.parquet) on or before the month end. A line that
    stopped trading in the middle of month m was still ranked on its m row and held a slot for
    all of m+1 without a return (Kevin's data check, 2026-11-17: 339 US member-months). Every
    permno of the daily files, one year at a time; known at the month end: point in time."""
    parts = []
    for y in years:
        for d in daily_dirs:
            f = d / f"{y}.parquet"
            if not f.exists():
                continue
            t = pq.read_table(f, columns=["permno", "date", "ret"]).to_pandas()
            t = t[t["ret"].notna()]
            t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
            if cutoff is not None:
                t = t[t["date"] <= cutoff]
            t = t.assign(permno=t["permno"].astype("int64"), month=t["date"].dt.to_period("M"))
            parts.append(t.groupby(["permno", "month"])["date"].max())
    last = pd.concat(parts).groupby(level=[0, 1]).max().rename("last").reset_index()
    m_end = last["month"].dt.to_timestamp(how="end").dt.normalize()
    last["listed"] = (m_end - last["last"]).dt.days <= int(within_days)
    if delisting is not None and len(delisting):
        dl = delisting.assign(permno=delisting["permno"].astype("int64"), date=pd.to_datetime(delisting["date"]))
        first = dl.groupby("permno")["date"].min()
        out = last["permno"].map(first)
        last.loc[out.notna() & (out <= m_end), "listed"] = False
    return last[["permno", "month", "listed"]]


def rankable(frame: pd.DataFrame, elig: pd.DataFrame, monthly: pd.DataFrame, cfg: dict,
             activity: pd.DataFrame | None = None, listed: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rows of `frame` (company_month or line_month) that may be RANKED for the universe;
    the rest keep their caps (LME, December ME) but cannot be members. `listed`
    (listed_at_month_end, config `universe.listed_within_days`): a line must still be listed
    at the end of the ranking month. Two more rules of config `universe` (audit of 2026-11-17):
      receipt_min_traded_share  a depositary receipt (sharetype AD) whose line traded on
                                fewer than this share of the days of the last
                                receipt_activity_months months (receipt_activity) is not
                                ranked; no daily rows to judge: ranked (B1).
      min_history_months        a line needs a CRSP monthly row this many months before the
                                ranking month: a listing is not ranked on its first month's
                                price (Addentax 2022-08: $21bn on its IPO day, -95% the day
                                after; B7). Spin-offs and new merger permnos wait a month too."""
    u = cfg.get("universe", {})
    keep = pd.Series(True, index=frame.index)
    k = int(u.get("min_history_months", 0) or 0)
    if k:
        have = pd.DataFrame({"primary_permno": monthly["permno"].astype("int64"),
                             "_m": monthly["date"].dt.to_period("M"), "_have": True}).drop_duplicates(["primary_permno", "_m"])
        x = pd.DataFrame({"primary_permno": frame["primary_permno"].astype("int64").to_numpy(),
                          "_m": (frame["month"] - k).to_numpy()}).merge(have, on=["primary_permno", "_m"], how="left")
        keep &= x["_have"].notna().to_numpy()
    if listed is not None and len(listed):
        x = frame[["primary_permno", "month"]].merge(listed.rename(columns={"permno": "primary_permno"}),
                                                     on=["primary_permno", "month"], how="left")
        keep &= x["listed"].fillna(False).astype(bool).to_numpy()          # no return at all in the month: not listed
    thr = u.get("receipt_min_traded_share")
    if thr and activity is not None and len(activity):
        st = elig[["permno", "month", "sharetype"]].rename(columns={"permno": "primary_permno"})
        x = frame[["primary_permno", "month"]].merge(st, on=["primary_permno", "month"], how="left").merge(
            activity.rename(columns={"permno": "primary_permno"}), on=["primary_permno", "month"], how="left")
        keep &= ~((x["sharetype"] == "AD") & (x["traded_share"] < float(thr))).to_numpy()
    return frame[keep.to_numpy()]


# ------------------------------------------------------------------ daily data


DAILY_COLS =["permno", "permco", "date", "ret", "prc", "vol", "bid", "ask", "high", "cap", "shrout", "cumfacpr",
              "primaryexch"]

# Gao and Ritter (2010): Nasdaq's dealer market reported both legs of a trade, so its share
# volume is divided by these factors before it is compared with exchange volume (Jensen-
# Kelly-Pedersen and Chen-Zimmermann do the same). Each divisor applies before its date.
NASDAQ_DIVISORS = [(pd.Timestamp("2001-02-01"), 2.0), (pd.Timestamp("2002-01-01"), 1.8),
                   (pd.Timestamp("2004-01-01"), 1.6)]


def nasdaq_divisor(dates: pd.Series, exch: pd.Series) -> np.ndarray:
    """Per row: the Gao-Ritter divisor of a Nasdaq (primaryexch 'Q') volume, else 1."""
    d = pd.to_datetime(dates).to_numpy()
    out = np.ones(len(d))
    lo = np.datetime64("1900-01-01")
    for end, k in NASDAQ_DIVISORS:
        out[(d >= lo) & (d < end.to_datetime64())] = k
        lo = end.to_datetime64()
    return np.where(np.asarray(exch.astype(str)) == "Q", out, 1.0)


def fold_broken_quotes(df: pd.DataFrame, max_rel_spread: float) -> pd.DataFrame:
    """`df` (rows sorted by permno and date) with no-trade days priced off a broken quote
    neutralised: volume zero or missing and a closing quote wider than `max_rel_spread` of
    its midpoint. CRSP prices such a day at the midpoint and its return follows it (ABN
    AMRO's receipt 2007-10-04: bid 53, ask 141, close 97, +82.6% then -45.4%, at rank 52;
    audit A6, 2026-11-17). The day keeps the previous close and a zero return, and its
    return is compounded into the permno's next row, so the return over the two days is
    CRSP's. Whether a day is flagged depends on that day's data only: point in time.
    Adds `broken_quote` (bool)."""
    bid, ask = df["bid"], df["ask"]
    rel = ((ask - bid) / ((ask + bid) / 2)).where((bid > 0) & (ask > bid))
    flag = (~(df["vol"] > 0) & (rel > max_rel_spread)).to_numpy()
    df = df.copy()
    df["broken_quote"] = flag
    if not flag.any():
        return df
    key = df["permno"].to_numpy()
    ret = df["ret"].to_numpy(dtype="float64").copy()
    # each flagged row belongs to the next row of its permno that has a return: a row without
    # one in between (ABN AMRO 2007-10-05) is skipped, as CRSP's next return skips it too
    absorb = ~flag & ~np.isnan(ret)
    nf = pd.Series(absorb.astype("int64")[::-1])
    grp = nf.groupby(key[::-1]).cumsum().to_numpy()[::-1]
    factor = np.where(flag, 1.0 + np.nan_to_num(ret), 1.0)
    prod = pd.Series(factor).groupby([key, grp]).transform("prod").to_numpy()
    target = absorb & (prod != 1.0)
    ret[target] = (1.0 + ret[target]) * prod[target] - 1.0
    ret[flag] = 0.0
    df["ret"] = ret
    df.loc[flag, ["prc", "high"]] = np.nan
    carried = df.groupby("permno")["prc"].ffill()          # only the flagged rows take it
    df.loc[flag, "prc"] = carried[flag]
    return df


def load_daily(daily_dir: Path | list[Path], permnos, years, ff_daily: pd.DataFrame,
               cutoff: pd.Timestamp | None = None, delisting: pd.DataFrame | None = None,
               low_dir: Path | None = None, fixes: dict | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily rows for `permnos` over `years`, and the per-(permno, month) statistics
    that the monthly-from-daily characteristics are built from. `daily_dir` may be a list
    (crsp_daily plus crsp_daily_extra): a permno has one share type on a date, so the files
    do not overlap.

    `delisting` (crsp_delisting.parquet) supplies each permno's delisting-day row, whose
    dlyret is the delisting return; the yearly files cannot contain it (module docstring
    of wrds_us). It is appended to the year's rows; the daily-file row wins if both exist.

    `fixes` (config, all optional; audit of 2026-11-17): `broken_quote_spread` neutralises
    no-trade days priced off a broken quote (fold_broken_quotes); `nasdaq_volume:
    gao_ritter` divides Nasdaq volume (nasdaq_divisor); `spread: corwin_schultz` adds the
    daily high-low estimate `cs` (hl_spread, from the daily low in `low_dir`, written by
    fetch_us_raw.py --low), which the Spread sums then use instead of the closing quotes."""
    fixes = fixes or {}
    permnos = [int(p) for p in permnos]
    dirs = [daily_dir] if isinstance(daily_dir, Path) else list(daily_dir)
    dl = pd.DataFrame(columns=DAILY_COLS) if delisting is None else delisting[delisting["permno"].isin(permnos)]
    dl = dl.reindex(columns=DAILY_COLS)
    use_cs = fixes.get("spread", "quotes") == "corwin_schultz"
    if use_cs and low_dir is None:
        raise ValueError("spread: corwin_schultz needs the daily low (raw crsp_daily_low/, fetch_us_raw.py --low)")
    frames = []
    for y in years:
        parts = [pq.read_table(d / f"{y}.parquet", columns=DAILY_COLS, filters=[("permno", "in", permnos)]).to_pandas()
                 for d in dirs if (d / f"{y}.parquet").exists()]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        # plain float64: the parquet metadata restores nullable Float64/Int32, whose to_numpy() is
        # an object array, and a receipt-month with returns but no volume then divides a Python
        # 0.0 in SUV (ZeroDivisionError, 2026-09-23). NA becomes NaN, values are unchanged.
        num = [c for c in DAILY_COLS if c not in ("permno", "permco", "date", "primaryexch")]
        df[num] = df[num].astype("float64")
        df["primaryexch"] = df["primaryexch"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df["permno"], df["permco"] = df["permno"].astype("int64"), df["permco"].astype("int64")
        extra = dl[dl["date"].dt.year == y] if len(dl) else dl
        if len(extra):
            df = pd.concat([df, extra.astype(df.dtypes.to_dict())], ignore_index=True)
            df = df.drop_duplicates(["permno", "date"], keep="first")
        if cutoff is not None:
            df = df[df["date"] <= cutoff]
        if len(df) == 0:
            continue
        if use_cs:
            lp = low_dir / f"{y}.parquet"
            low = (pq.read_table(lp, columns=["permno", "date", "low"], filters=[("permno", "in", permnos)]).to_pandas()
                   if lp.exists() else pd.DataFrame({"permno": [], "date": [], "low": []}))
            low = low.astype({"permno": "int64", "low": "float64"})
            low["date"] = pd.to_datetime(low["date"]).astype("datetime64[ns]")
            df = df.merge(low.drop_duplicates(["permno", "date"]), on=["permno", "date"], how="left")
        df["prc"] = df["prc"].abs()
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True).sort_values(["permno", "date"], ignore_index=True)
    if fixes.get("broken_quote_spread"):
        daily = fold_broken_quotes(daily, float(fixes["broken_quote_spread"]))
    if fixes.get("nasdaq_volume") == "gao_ritter":
        daily["vol"] = daily["vol"] / nasdaq_divisor(daily["date"], daily["primaryexch"])
    if use_cs:
        daily["cs"] = hl_spread.by_line(daily.assign(_traded=daily["vol"] > 0), ["permno"], "high", "low", "prc",
                                        "_traded", factor="cumfacpr").to_numpy()
    year = daily["date"].dt.year.to_numpy()
    stats = [monthly_daily_stats(daily[year == y], ff_daily) for y in np.unique(year)]
    return daily, pd.concat(stats).sort_index()


def monthly_daily_stats(df: pd.DataFrame, ff_daily: pd.DataFrame) -> pd.DataFrame:
    """Per-(month, permno) sums of daily quantities. Each monthly characteristic that
    looks at daily data (Variance, Resid_Var, Spread, SUV) is an exact function of these
    sums pooled over its window, so nothing daily has to be re-read per month.

      n_ret, sum_ret, sum_ret2      returns                      -> Variance
      x_<a>_<b>, xy_<a>, sum_exret2 FF3 cross products           -> Resid_Var
      n_spread, sum_spread          daily spread: the Corwin-Schultz estimate `cs` where the
                                    frame has it, else the relative closing quote -> Spread
      n_vol, sum_vol, sum_vol2, sum_rp, sum_rn, sum_rp2, sum_rn2, sum_rp_vol, sum_rn_vol
                                    volume on |r+| and |r-|      -> SUV
    """
    cols = ["permno", "date", "ret", "vol", "bid", "ask"] + (["cs"] if "cs" in df else [])
    d = df[cols].merge(
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

    if "cs" in d:
        rel = d["cs"].astype("float64")
    else:
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
                   D: ch.DailyInputs, windows: dict, nasdaq_volume: str | None = None) -> ch.MonthlyInputs:
    """Wide monthly frames for `permnos`. The return/price/volume series are the
    permno's own (unfiltered, so a spell outside the eligible pool does not punch a hole
    in its momentum); cap_co is the company cap from the eligible classes. `nasdaq_volume:
    gao_ritter` divides Nasdaq's monthly volume as load_daily divides the daily one."""
    m = monthly[monthly["permno"].isin(permnos)].copy()
    m["month"] = m["date"].dt.to_period("M")
    m["prc"] = m["prc"].abs()
    if nasdaq_volume == "gao_ritter":
        m["vol"] = m["vol"].astype("float64") / nasdaq_divisor(m["date"], m["primaryexch"])
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
                 last_observed: bool = False, carry_for: list[str] | None = None) -> pd.DataFrame:
    """For each (permno, month, gvkey) in `mapping`, the latest fiscal record whose
    avail_month <= month and whose datadate is at most max_age_months old. Returns
    (permno, month) + `cols`. With last_observed, each column comes instead from the latest
    such record in which it is not missing: the stock's last observed value. For each column
    of `carry_for`, also `_dd_<col>` and `_me_<col>`: the datadate and the December market
    equity of the record it came from (validate_dec_me reads them)."""
    carry_for = [c for c in (carry_for or []) if c in cols]
    left = mapping.copy()
    left["key"] = left["month"].dt.to_timestamp(how="end").dt.normalize()
    left = left.sort_values("key", ignore_index=True)
    extra = ["me_dec"] if carry_for and "me_dec" in annual else []
    right = annual[["gvkey", "datadate", "avail_month"] + cols + extra].copy()
    right["key"] = right["avail_month"].dt.to_timestamp(how="end").dt.normalize()
    # A fiscal-year-end change puts two records in one calendar year (AMD: 1987-03 and 1987-12),
    # both usable from the same June; merge_asof takes the last tie, so order ties by datadate.
    # sort_values("key") alone is unstable and picked either, depending on the frame's length,
    # which broke prefix invariance (found 2026-09-23 with the 2010-09-15 cutoff build).
    right = right.sort_values(["key", "datadate"], kind="mergesort")
    out = left[["permno", "month"]].copy()
    for group in ([[c] for c in cols] if last_observed else [cols]):
        r = right.dropna(subset=group) if last_observed else right
        m = pd.merge_asof(left, r[["gvkey", "key", "datadate"] + group + extra], on="key", by="gvkey",
                          direction="backward")
        age = (m["key"].dt.year - m["datadate"].dt.year) * 12 + (m["key"].dt.month - m["datadate"].dt.month)
        old = ~(age <= max_age_months)
        m.loc[old, group] = np.nan
        out[group] = m[group].to_numpy()
        for c in (c for c in group if c in carry_for):
            out[f"_dd_{c}"] = m["datadate"].where(~old).to_numpy()
            out[f"_me_{c}"] = m["me_dec"].where(~old).to_numpy() if extra else np.nan
    return out


def validate_dec_me(aligned: pd.DataFrame, chars: list[str], cap: pd.DataFrame, ret: pd.DataFrame,
                    max_ratio: float = 8.0) -> pd.DataFrame:
    """`aligned` (align_annual output with carry_for = `chars`) with each value of `chars`
    set missing where its record's December market equity does not belong to the company
    priced in that month: the company cap of the month, `cap` (month x id), over December ME
    grown by the company's return since that December (`ret`, month x id), outside
    1/max_ratio..max_ratio. That is a share count that moved more than max_ratio-fold
    without a return to explain it. Causes found by the 2026-11-17 audit (B2): a receipt
    whose December cap fell back to CRSP's receipts outstanding (Mitsubishi Bank Dec 1990:
    $1.6m against a $43bn company, BEME 6,387 at rank 3), a CRSP share count in thousands
    of the wrong unit (EnCana Dec 2001), a pre-IPO or certificate line under the same id
    (GDF 2004, Credit Agricole 2000, Areva's certificates), and real recapitalisations whose
    old December ME no longer prices the company (AIB 2011). Uses only months up to the row's
    own: point in time. Rows it cannot check (no cap that month, December before the
    return grid) keep their value. Drops the `_dd_` / `_me_` columns."""
    out = aligned.copy()
    grid = cap.index.union(ret.index)
    lr = np.log1p(ret.reindex(index=grid).astype("float64").fillna(0.0)).cumsum()
    lr_s = lr.stack(future_stack=True)
    cap_s = cap.reindex(index=grid).astype("float64").stack(future_stack=True)
    ids = out["permno"].to_numpy()
    month = pd.PeriodIndex(out["month"], freq="M")
    for c in chars:
        if f"_dd_{c}" not in out:
            continue
        dd = pd.to_datetime(out[f"_dd_{c}"])
        dec = pd.PeriodIndex.from_fields(year=dd.dt.year.fillna(1800).astype(int), month=12, freq="M")
        l_m = lr_s.reindex(pd.MultiIndex.from_arrays([month, ids])).to_numpy()
        l_d = lr_s.reindex(pd.MultiIndex.from_arrays([dec, ids])).to_numpy()
        cap_m = cap_s.reindex(pd.MultiIndex.from_arrays([month, ids])).to_numpy()
        me = pd.to_numeric(out[f"_me_{c}"], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.log(cap_m / (me * np.exp(l_m - l_d)))
        bad = np.isfinite(q) & (np.abs(q) > np.log(max_ratio)) & dd.notna().to_numpy()
        out.loc[bad, c] = np.nan
    return out.drop(columns=[x for x in out.columns if x.startswith(("_dd_", "_me_"))])
