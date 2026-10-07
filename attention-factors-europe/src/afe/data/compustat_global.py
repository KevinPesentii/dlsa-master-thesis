"""Monthly company market cap for European markets from Compustat Global (comp.g_secd).

Compustat Global has no monthly share count (`comp.g_secm` carries prices only), so the
month-end rows of the daily security file (`monthend = 1`) are used: prccd / qunit is the
close in the quotation currency `curcdd`, `cshoc` the shares outstanding of that issue.

Three things differ from the North American build in `compustat_us`:

1. Cross-listings. One share class is often listed on several exchanges (Fiat on Milan,
   Paris and Xetra; every DAX name on Xetra, Frankfurt, Stuttgart, ...), each listing its
   own `iid` with the SAME `cshoc`. Summing over iid as the US code does would double
   count, so listings are collapsed to share classes: rows of one company-month that
   share an ISIN, or that share an identical `cshoc`, are one class and the best-traded
   listing in the home country prices it. Distinct classes (Shell A + B in Amsterdam)
   still add up, as in the US permco definition. The most active class prices the
   COMPANY (`iid`: the line whose returns and characteristics stage 2 follows) unless it
   carries less than `min_class_share` of the company's cap; then the largest class does.
   Turnover is volume over shares, so a small class scores high on modest volume (SEB C,
   Land Securities' 2002 B shares, short-lived new-share lines such as Sanofi's "RFD").
   Lines that are not shares although g_secd files them as common (`non_equity`: Belgian
   VVPR strips, subscription / bonus / offer rights, stock-dividend rights, nil-paid
   lines) are dropped first: they neither price nor add to a cap. Before this rule
   Electrabel 2006-07 was priced by its EUR 0.01 strip (+100% / -50% days).
   Lines of a class that are not a class of their own are dropped too, because their share
   count is not the class's: Chi-X / Cboe Europe quotes (`MTF_EXCHANGES`, from 2012-05,
   no ISIN; RELX's differed from the London line by a few thousand shares in half the
   months and doubled its cap, Rotork's carried ten times the shares) and Swiss buyback,
   second-trading and tendered ("ASD") lines (`non_equity`; Adecco's buyback line, a
   stale price on 750m shares, made its cap five times too large). A class whose share
   count AND price are within `dup_tol` of a class already counted is the same class
   under another ISIN and is not added again.
   Participation certificates count as classes (`issue_types`, tpci '8': Swiss
   Partizipationsscheine, French certificats d'investissement; non-voting equity), and
   so do Genussscheine on the exchanges of `genussschein_countries` (tpci 'Q': Roche's,
   about 80% of its equity; German Genussscheine are profit-participation debt and stay
   out).

2. Country and eligibility. `g_security.excntry` is the country of the exchange a
   listing trades on. Frankfurt alone carries lines for ~11,000 gvkeys, most of them
   dormant secondary listings of US and Asian companies, and the header primary-issue
   tags (`g_company.prirow`, `priusa`) cannot tell those apart from real ones: Accenture's
   `prirow` IS its Frankfurt line, while Linde plc, Fiat and Alcatel carry a NYSE
   `priusa` next to their real Frankfurt/Milan/Paris listing. So a company-month is
   ELIGIBLE when (a) the header links the company to the set (`fic`, `loc` or the
   country of `prirow`) AND (b) its most active listing in the set is alive: trailing
   median of month-end-day volume over shares >= `min_turnover`. Real home listings
   turn over 0.1-1% of shares a day (Siemens, Total, CRH), dormant lines 0.0001% or less;
   the 3e-5 default sits below the largest German names in 1999-2000, when g_secd still
   reports Frankfurt floor volume. (b) also drops sub-3%-float subsidiaries (Elf 2000-07,
   Audi, Hoechst 1999-2004, EnBW) and Frankfurt shells with fantasy caps. The HOME
   country is that most active listing's exchange country; incorporation is NOT the
   country (Airbus, ArcelorMittal, Stellantis, Ferrari trade in Paris/Amsterdam/Milan).

3. Currency. Prices are in the exchange's quotation currency. Everything is converted to
   the target currency with Compustat's GBP-based cross rates (`g_exrt_dly`, units of
   currency per GBP), at the row's own date, falling back to the latest earlier rate.

`prirow`, `excntry`, `exchg`, `fic`, `loc`, `isin`, `conm` are header fields (current
value on every historical row). Companies that moved their primary listing are assigned
by where it is today; the build report lists the largest fic/home mismatches.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# g_security.dsci of lines that are not shares (tpci is still '0'): "ORD NPV VVPR STRIP",
# "ORD EUR2.29 (SUB RIGHT)", "EUR0.2 BN RTS 17/11/21", "ORD EUR1(STOCK DIV 5/7/2012)",
# "ORD NPV (NIL PAID 05/08/10)"; and second lines of a class: "ORD CHF1(REGD)(2ND BUY BACK)",
# "ORD CHF.10 (REGD) (2ND BUY B" (descriptions are cut at 28 characters),
# "CHF0.1 (SEPARATE 2 TRADING L", "ORD EUR2 (SEC LINE)", "ORD NPV (ASD 06/07/18 EON CS",
# "ORD NPV (TENDERED SHARES)"; Swedish redemption shares, issued one per share and so
# carrying the class's own share count: "ORD NPV(REDEMPTION SHARES)", "ORD SER'A' NPV (RED
# SHS11/06", and "EUR0.75 (STK DIV 25/01/22)". Shares trading ex or cum rights, and lines
# without a dividend right, are shares.
NON_EQUITY = re.compile(r"STRIP|STO?C?K DIV|NIL PAID|RIGHT|\bRTS\b|REDEMPTION|\bRED SH"
                        r"|BUY[- ]?B|\bLINE\b|TRADING L|TENDERED|ASSENTED|\bASD\b", re.I)
SHARE_DESPITE_RIGHTS = re.compile(r"EX[- ]R(?:IGH)?TS|CUM RTS|DIVIDEND RIGHT", re.I)

# g_security.exchg of multilateral trading facilities: a quote of a share listed elsewhere,
# never a listing (349: Chi-X / Cboe Europe, London, "ORD GBP0.005 (CHI-X)").
MTF_EXCHANGES = (349,)

SECD_COLS = [
    "gvkey", "iid", "datadate", "prccd", "cshoc", "ajexdi", "curcdd", "prcstd", "qunit",
    "tpci", "exchg", "isin", "cshtrd", "conm", "fic", "loc",
]
NUMERIC = ["prccd", "cshoc", "ajexdi", "prcstd", "qunit", "exchg", "cshtrd"]


# ----------------------------------------------------------------------------- WRDS


def _sql_list(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def fetch_security_header(db, countries: list[str]) -> pd.DataFrame:
    """Every listing (gvkey, iid), on ANY exchange, of companies with at least one listing
    on an exchange in `countries`, from comp.g_security. The foreign lines are needed to
    resolve where a company's primary issue trades."""
    sec = db.raw_sql(
        "select gvkey, iid, excntry, exchg, isin, sedol, tpci, secstat, dsci, dldtei, dlrsni "
        "from comp.g_security where gvkey in "
        f"(select distinct gvkey from comp.g_security where excntry in {_sql_list(countries)})",
        date_cols=["dldtei"],
    )
    sec["exchg"] = pd.to_numeric(sec["exchg"], errors="coerce")
    print(f"  g_security: {len(sec):,} listings, {sec['gvkey'].nunique():,} gvkeys", flush=True)
    return sec


def fetch_company_header(db) -> pd.DataFrame:
    """comp.g_company: primary issue (prirow), incorporation/HQ country, SIC, delisting."""
    return db.raw_sql(
        "select gvkey, conm, prirow, priusa, prican, fic, loc, sic, gind, dldte, dlrsn, ipodate "
        "from comp.g_company",
        date_cols=["dldte", "ipodate"],
    )


def fetch_secd_monthend(db, start_year: int, end_year: int, countries: list[str],
                        issue_types: tuple[str, ...] = ("0", "1")) -> pd.DataFrame:
    """Month-end rows of comp.g_secd for listings on exchanges in `countries`.

    Common ('0') and preferred ('1') issues are both pulled; the class filter is applied
    in pandas so the extract on disk can be re-cut without going back to WRDS. One query
    per year; g_secd is large and the year bound keeps each one short.
    """
    parts = []
    for year in range(start_year, end_year + 1):
        sql = f"""
            select {", ".join("s." + c for c in SECD_COLS)}, h.excntry
            from comp.g_secd s
            join comp.g_security h on s.gvkey = h.gvkey and s.iid = h.iid
            where s.monthend = 1 and s.prccd is not null
              and s.tpci in {_sql_list(issue_types)}
              and h.excntry in {_sql_list(countries)}
              and s.datadate between '{year}-01-01' and '{year}-12-31'
        """
        part = db.raw_sql(sql, date_cols=["datadate"])
        for c in NUMERIC:
            part[c] = pd.to_numeric(part[c], errors="coerce")
        print(f"  g_secd {year}: {len(part):>7,} rows, {part['gvkey'].nunique():>5,} gvkeys", flush=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fetch_secd_monthend_lines(db, pairs: list[tuple[str, str]], start_year: int, end_year: int) -> pd.DataFrame:
    """Month-end rows of comp.g_secd for the listed (gvkey, iid) lines, every year at once,
    with the same columns as fetch_secd_monthend: adds issue types to an existing extract."""
    sql = f"""
        select {", ".join("s." + c for c in SECD_COLS)}, h.excntry
        from comp.g_secd s
        join comp.g_security h on s.gvkey = h.gvkey and s.iid = h.iid
        where s.monthend = 1 and s.prccd is not null
          and (s.gvkey, s.iid) in ({", ".join(f"('{g}', '{i}')" for g, i in pairs)})
          and s.datadate between '{start_year}-01-01' and '{end_year}-12-31'
    """
    out = db.raw_sql(sql, date_cols=["datadate"])
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_fx(db, currencies: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    """Daily GBP-based cross rates (units of `tocurd` per GBP) for the currencies seen."""
    fx = db.raw_sql(
        "select datadate, tocurd, exratd from comp.g_exrt_dly "
        f"where fromcurd = 'GBP' and tocurd in {_sql_list(currencies)} "
        f"and datadate between '{start_year}-01-01' and '{end_year}-12-31'",
        date_cols=["datadate"],
    )
    fx["exratd"] = pd.to_numeric(fx["exratd"], errors="coerce")
    print(f"  g_exrt_dly: {len(fx):,} rows, {fx['tocurd'].nunique()} currencies", flush=True)
    return fx.dropna().sort_values(["tocurd", "datadate"]).reset_index(drop=True)


# ------------------------------------------------------------------- pure transforms


def non_equity(dsci: pd.Series) -> pd.Series:
    """True where a g_security description names a line that is not a share (NON_EQUITY)."""
    d = dsci.fillna("").astype(str)
    return d.str.contains(NON_EQUITY) & ~d.str.contains(SHARE_DESPITE_RIGHTS)


def convert_to_currency(rows: pd.DataFrame, fx: pd.DataFrame, target: str,
                        max_stale_days: int = 7, date_col: str = "datadate") -> pd.Series:
    """Factor that turns `curcdd` amounts into `target`, per row, at the row's `date_col`.

    rate(target) / rate(local) where rate(c) = units of c per GBP. The latest rate at or
    before the date is used, at most `max_stale_days` old; NaN otherwise.
    """
    fx = fx.rename(columns={"tocurd": "ccy"}).assign(datadate=lambda t: t["datadate"].astype("datetime64[ns]"))
    fx = fx.sort_values("datadate")
    out = pd.Series(np.nan, index=rows.index)
    key = rows[[date_col, "curcdd"]].rename(columns={date_col: "datadate"})
    key["datadate"] = key["datadate"].astype("datetime64[ns]")
    key["_i"] = rows.index
    key = key.sort_values("datadate")
    loc = pd.merge_asof(key, fx.rename(columns={"exratd": "r_local"}), left_on="datadate",
                        right_on="datadate", left_by="curcdd", right_by="ccy",
                        tolerance=pd.Timedelta(days=max_stale_days), direction="backward")
    tgt_fx = fx[fx["ccy"] == target].drop(columns="ccy").rename(columns={"exratd": "r_target"})
    tgt = pd.merge_asof(key[["datadate", "_i"]], tgt_fx, on="datadate",
                        tolerance=pd.Timedelta(days=max_stale_days), direction="backward")
    factor = (tgt.set_index("_i")["r_target"] / loc.set_index("_i")["r_local"])
    out.loc[factor.index] = factor.values
    out[rows["curcdd"] == target] = 1.0
    return out


def apply_share_corrections(df: pd.DataFrame, corrections: list[dict] | None,
                            date_col: str = "datadate") -> pd.DataFrame:
    """Divide `cshoc` by `divide_by` on the rows of each correction's (gvkey, iid) dated
    `start`..`end`: share counts g_secd states in another unit than the price (configs,
    `corrections.share_count`). A copy; rows outside every correction are unchanged."""
    if not corrections:
        return df
    df = df.copy()
    df["cshoc"] = df["cshoc"].astype("float64")
    for c in corrections:
        hit = ((df["gvkey"] == str(c["gvkey"])) & (df["iid"] == str(c["iid"]))
               & df[date_col].between(pd.Timestamp(c["start"]), pd.Timestamp(c["end"])))
        df.loc[hit, "cshoc"] = df.loc[hit, "cshoc"] / float(c["divide_by"])
    return df


def _in_spans(df: pd.DataFrame, spans: list[dict] | None, date_col: str, by_iid: bool = True) -> pd.Series:
    """True on rows of `df` inside any span: same gvkey (and iid if `by_iid`), date in start..end."""
    hit = pd.Series(False, index=df.index)
    for c in spans or []:
        m = (df["gvkey"] == str(c["gvkey"])) & df[date_col].between(pd.Timestamp(c["start"]), pd.Timestamp(c["end"]))
        if by_iid:
            m &= df["iid"] == str(c["iid"])
        hit |= m
    return hit


def _drop_near_duplicate_classes(home: pd.DataFrame, key: list[str], tol: float) -> pd.DataFrame:
    """Drop a class whose share count and price are both within `tol` (log) of a class
    earlier in its company-month: the same class under another ISIN or iid."""
    h = home.reset_index(drop=True)
    h["_pos"] = h.groupby(key).cumcount()
    multi = h[h.groupby(key)["_pos"].transform("size") > 1][key + ["_pos", "cshoc", "price"]]
    if multi.empty:
        return h.drop(columns="_pos")
    p = multi.merge(multi, on=key, suffixes=("", "_o"))
    p = p[(p["_pos_o"] < p["_pos"])
          & (np.log(p["cshoc"] / p["cshoc_o"]).abs() < tol) & (np.log(p["price"] / p["price_o"]).abs() < tol)]
    dup = h.merge(p[key + ["_pos"]].drop_duplicates().assign(_dup=True), on=key + ["_pos"], how="left")["_dup"]
    return h[dup.isna().to_numpy()].drop(columns="_pos")


def company_month_mktcap(secd: pd.DataFrame, company: pd.DataFrame, security: pd.DataFrame,
                         fx: pd.DataFrame, countries: list[str], target_ccy: str = "EUR",
                         issue_types: tuple[str, ...] = ("0",), min_turnover: float = 3e-5,
                         turnover_window: int = 6, min_class_share: float = 0.05,
                         genussschein_countries: tuple[str, ...] = (), share_corrections: list[dict] | None = None,
                         dup_tol: float = 0.02, exclude_lines: list[dict] | None = None,
                         dual_listed: list[dict] | None = None) -> pd.DataFrame:
    """One row per (gvkey, datadate) with market cap in `target_ccy` millions.

    Steps: keep priced issues of the requested classes with a share count (`issue_types`,
    plus tpci 'Q' on the exchanges of `genussschein_countries`), less the non-equity lines
    of `security.dsci`, the MTF quotes and the dated `exclude_lines`; correct the share
    counts of `share_corrections`; convert the price; score every listing by trailing
    turnover; the best-traded listing sets the home country; collapse the home country's
    listings to share classes (for a `dual_listed` gvkey, the listings of all its
    countries: both halves of a dual-listed company filed under one gvkey); sum the
    classes; the most active class prices the company unless it is below `min_class_share`
    of the cap (then the largest). `eligible` = header link to the country set AND an
    active home listing (see module docstring). Nothing else is dropped: the caller
    filters on `eligible`. The three correction lists are the config's `corrections`.
    """
    keep = secd["tpci"].isin(issue_types)
    if genussschein_countries:
        keep |= (secd["tpci"] == "Q") & secd["excntry"].isin(genussschein_countries)
    df = secd[keep & ~secd["exchg"].isin(MTF_EXCHANGES)].copy()
    df = df[~_in_spans(df, exclude_lines, "datadate")]
    if "dsci" in security:
        bad = security.loc[non_equity(security["dsci"]), ["gvkey", "iid"]].drop_duplicates()
        df = df.merge(bad.assign(_bad=True), on=["gvkey", "iid"], how="left")
        df = df[df["_bad"].isna()].drop(columns="_bad")
    df = apply_share_corrections(df, share_corrections)
    if df.duplicated(["gvkey", "iid", "datadate"]).any():
        raise ValueError("g_secd extract has duplicate (gvkey, iid, datadate) rows")
    # monthend = 1 marks each listing's own last row of the month (its exchange's last
    # trading day, or the delisting day), so dates differ inside a month. Key on the
    # calendar month end; keep the actual date as price_date and the latest row per month.
    df["price_date"] = df["datadate"]
    df["datadate"] = df["datadate"] + pd.offsets.MonthEnd(0)
    df = (df.sort_values(["gvkey", "iid", "price_date"])
          .drop_duplicates(["gvkey", "iid", "datadate"], keep="last"))
    df = df[(df["prccd"] > 0) & (df["cshoc"] > 0) & df["curcdd"].notna()].copy()
    df["qunit"] = df["qunit"].fillna(1.0)
    df["fx"] = convert_to_currency(df, fx, target_ccy, date_col="price_date")
    df = df[df["fx"].notna()].copy()
    df["price"] = df["prccd"] / df["qunit"] * df["fx"]
    df["cap_listing"] = df["price"] * df["cshoc"] / 1e6

    # Activity of a listing: month-end-day volume over shares, trailing median over the
    # listing's last `turnover_window` month ends. Volume is never zero in g_secd, only
    # missing. A missing day counts as NO trading when the country's DOMESTIC listings
    # (fic == exchange country) mostly carry volume that month, which is when a blank
    # means a dormant line (Frankfurt / Borsa Italiana lines of US companies); it is
    # skipped when the market as a whole reports none (Ireland before 2000-06, Luxembourg).
    # Domestic listings, because foreign lines are the majority on German venues and
    # would vote the coverage down. A listing with only skipped days gets NaN: indeterminate.
    df = df.sort_values(["gvkey", "iid", "datadate"])
    df["turnover_day"] = df["cshtrd"] / df["cshoc"]
    dom = df[df["fic"] == df["excntry"]]
    cov = dom.groupby(["excntry", "datadate"])["cshtrd"].apply(lambda v: v.notna().mean()).rename("_cov")
    df = df.join(cov, on=["excntry", "datadate"])
    df.loc[df["cshtrd"].isna() & (df["_cov"] >= 0.5), "turnover_day"] = 0.0
    df = df.drop(columns="_cov")
    df["turnover"] = (df.groupby(["gvkey", "iid"])["turnover_day"]
                      .transform(lambda v: v.rolling(turnover_window, min_periods=1).median()))

    # Header link: incorporation, headquarters or primary issue in the country set.
    prim = company[["gvkey", "prirow"]].merge(
        security[["gvkey", "iid", "excntry"]].rename(columns={"iid": "prirow", "excntry": "prirow_country"}),
        on=["gvkey", "prirow"], how="left")
    hdr = company[["gvkey", "prirow", "priusa", "fic", "loc", "sic", "conm"]].merge(
        prim[["gvkey", "prirow_country"]], on="gvkey", how="left")
    hdr["header_link"] = (hdr["fic"].isin(countries) | hdr["loc"].isin(countries)
                          | hdr["prirow_country"].isin(countries))
    df = df.drop(columns=["conm", "fic", "loc"]).merge(hdr, on="gvkey", how="left")
    df["header_link"] = df["header_link"].fillna(False).astype(bool)
    df["is_prirow"] = df["iid"] == df["prirow"]

    # Home country of the company-month: the most active listing's; ties to the primary
    # issue, then to the largest cap. Listings with no turnover information sort last.
    key = ["gvkey", "datadate"]
    order = key + ["turnover", "is_prirow", "cap_listing"]
    df = df.sort_values(order, ascending=[True, True, False, False, False], na_position="last")
    best = df.drop_duplicates(key)[key + ["excntry", "turnover", "iid"]].rename(
        columns={"excntry": "country", "turnover": "turnover_home", "iid": "iid_home"})
    df = df.merge(best, on=key, how="left")
    at_home = df["excntry"] == df["country"]
    for c in dual_listed or []:
        span = _in_spans(df, [c], "datadate", by_iid=False)
        at_home |= span & df["excntry"].isin(c["countries"]) & df["country"].isin(c["countries"])
    home = df[at_home].copy()

    # Collapse listings to share classes: same ISIN, or identical share count, = one class.
    home = home.sort_values(order, ascending=[True, True, False, False, False], na_position="last")
    home["class_id"] = home["isin"].fillna("iid:" + home["iid"])
    home = home.drop_duplicates(key + ["class_id"]).drop_duplicates(key + ["cshoc"])
    home = _drop_near_duplicate_classes(home, key, dup_tol)
    # The first class per company-month prices it: the most active, unless that one holds
    # less than min_class_share of the cap, in which case the largest class goes first.
    small = (home.groupby(key)["cap_listing"].transform("first")
             < min_class_share * home.groupby(key)["cap_listing"].transform("sum"))
    home["_pos"] = np.where(small, -home["cap_listing"], np.arange(len(home)))
    home = home.sort_values(key + ["_pos"], kind="mergesort").drop(columns="_pos")

    agg = home.groupby(key, sort=False).agg(
        mktcap=("cap_listing", "sum"), mktcap_main_class=("cap_listing", "first"),
        n_classes=("cap_listing", "size"),
        country=("country", "first"), iid=("iid", "first"), exchg=("exchg", "first"),
        curcdd=("curcdd", "first"), prccd=("prccd", "first"), cshoc=("cshoc", "first"),
        prcstd=("prcstd", "first"), fx=("fx", "first"), isin=("isin", "first"),
        price_date=("price_date", "first"), turnover=("turnover_home", "first"),
    ).reset_index()
    n_list = df.groupby(key, sort=False).size().rename("n_listings").reset_index()
    out = agg.merge(n_list, on=key).merge(hdr, on="gvkey", how="left")
    out["active"] = out["turnover"].isna() | (out["turnover"] >= min_turnover)
    out["eligible"] = out["header_link"] & out["active"]
    out = out.sort_values(key, ignore_index=True)
    cols = key + ["mktcap", "mktcap_main_class", "country", "n_classes", "n_listings", "eligible",
                  "header_link", "active", "turnover", "prirow", "prirow_country", "priusa", "fic",
                  "loc", "sic", "conm", "iid", "exchg", "isin", "curcdd", "prccd", "cshoc", "prcstd",
                  "fx", "price_date"]
    return out[cols]


def rank_within(mcap: pd.DataFrame, by: list[str], value: str = "mktcap") -> pd.DataFrame:
    """Add `cap_rank` (1 = largest) of `value` within each group of `by`."""
    m = mcap[mcap[value].notna()].sort_values(by + [value, "gvkey"],
                                              ascending=[True] * len(by) + [False, True])
    rank = m.groupby(by).cumcount() if by else pd.Series(range(len(m)), index=m.index)
    return m.assign(cap_rank=(rank + 1).astype("int32"))


def cutoff_table(mcap: pd.DataFrame, ranks: list[int], by: list[str],
                 value: str = "mktcap") -> pd.DataFrame:
    """Cap of the k-th largest company per group, for each k in `ranks`, plus the count.
    With `by` empty the whole frame is one group (index label 'POOLED')."""
    r = rank_within(mcap, by, value)
    if not by:
        r = r.assign(_g="POOLED")
        by = ["_g"]
    counts = r.groupby(by).size().rename("n_companies")
    wide = r[r["cap_rank"].isin(ranks)].pivot_table(index=by, columns="cap_rank", values=value)
    wide.columns = [f"rank_{k}" for k in wide.columns]
    wide = wide.reindex(index=counts.index)  # keep groups with fewer companies than the smallest rank
    wide.insert(0, "n_companies", counts)
    wide = wide.reindex(columns=["n_companies"] + [f"rank_{k}" for k in ranks])
    return wide.rename_axis(index=[None if b == "_g" else b for b in by])
