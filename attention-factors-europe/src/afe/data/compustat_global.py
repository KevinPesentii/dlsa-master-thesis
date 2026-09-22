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
   still add up, as in the US permco definition.

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

import numpy as np
import pandas as pd

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


def company_month_mktcap(secd: pd.DataFrame, company: pd.DataFrame, security: pd.DataFrame,
                         fx: pd.DataFrame, countries: list[str], target_ccy: str = "EUR",
                         issue_types: tuple[str, ...] = ("0",), min_turnover: float = 3e-5,
                         turnover_window: int = 6) -> pd.DataFrame:
    """One row per (gvkey, datadate) with market cap in `target_ccy` millions.

    Steps: keep priced issues of the requested classes with a share count; convert the
    price; score every listing by trailing turnover; the best-traded listing sets the
    home country; collapse the home country's listings to share classes; sum the classes.
    `eligible` = header link to the country set AND an active home listing (see module
    docstring). Nothing is dropped: the caller filters on `eligible`.
    """
    df = secd[secd["tpci"].isin(issue_types)].copy()
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
    home = df[df["excntry"] == df["country"]].copy()

    # Collapse listings to share classes: same ISIN, or identical share count, = one class.
    home = home.sort_values(order, ascending=[True, True, False, False, False], na_position="last")
    home["class_id"] = home["isin"].fillna("iid:" + home["iid"])
    home = home.drop_duplicates(key + ["class_id"]).drop_duplicates(key + ["cshoc"])

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
