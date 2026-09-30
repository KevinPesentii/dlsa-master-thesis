"""Numeraire of the European builds: the ECU to 1998-12-31, the euro from 1999-01-01.

The euro replaced the ECU one for one on 1 January 1999 (Council Regulation 1103/97,
art. 2(1)), so one unit runs through the whole sample: ECU before, EUR after. In the FX
table it is the pseudo currency `XEU` (the ISO 4217 code of the ECU): units of XEU per
GBP, the same form as Compustat's `g_exrt_dly` quotes (units of currency per GBP), so
every existing conversion (`compustat_global.convert_to_currency`,
`eu_panel.eur_factor`) works unchanged with `XEU` as the target.

The ECU was a basket of fixed amounts of the member currencies. Its value in any currency
is the sum of the amounts converted at that day's market rates; from Compustat's GBP
cross rates:

    GBP per ECU = sum_i a_i / exratd_i       (exratd_i = units of currency i per GBP)
    XEU per GBP = 1 / (GBP per ECU)          before 1999-01-01
    XEU per GBP = Compustat's EUR quote      from 1999-01-01

The basket changed twice while it existed; each revision was set so the ECU's value did
not jump on the day (the build report checks the ratio of old to new basket on the
revision day). The irrevocable conversion rates of 31 December 1998 are that day's
official ECU rates, so the basket value on 1998-12-31 must reproduce them (1.95583 DEM,
6.55957 FRF, ...): the second check. Compustat also quotes the ECU itself (`XEU`,
business days 1985-09-30 .. 1998-12-31): the third check, not the source, because the
basket covers every calendar day from the first quote of its currencies and uses the same
cross rates as every price conversion. Compustat's EUR series before 1999 is a synthetic
euro, not the ECU (0.8% off at the 1998-12-31 seam, see docs/eu_raw_data.md), so it is
used from 1999 only.

The Luxembourg franc was at par with the Belgian franc (monetary union); where Compustat
has no LUF quote the BEF quote stands in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERAIRE = "XEU"
EURO_START = pd.Timestamp("1999-01-01")

# Official composition, units of each currency in one ECU, with the date each basket took
# effect (European Commission; Council Regulations 3180/78, 2626/84, 1971/89). The third
# basket was frozen by the Maastricht Treaty (1 November 1993) and ran to 1998-12-31.
BASKETS: list[tuple[pd.Timestamp, dict[str, float]]] = [
    (pd.Timestamp("1979-03-13"), {"DEM": 0.828, "FRF": 1.15, "GBP": 0.0885, "ITL": 109.0, "NLG": 0.286,
                                  "BEF": 3.66, "LUF": 0.14, "DKK": 0.217, "IEP": 0.00759}),
    (pd.Timestamp("1984-09-17"), {"DEM": 0.719, "FRF": 1.31, "GBP": 0.0878, "ITL": 140.0, "NLG": 0.256,
                                  "BEF": 3.71, "LUF": 0.14, "DKK": 0.219, "IEP": 0.00871, "GRD": 1.15}),
    (pd.Timestamp("1989-09-21"), {"DEM": 0.6242, "FRF": 1.332, "GBP": 0.08784, "ITL": 151.8, "NLG": 0.2198,
                                  "BEF": 3.301, "LUF": 0.130, "DKK": 0.1976, "IEP": 0.008552, "GRD": 1.440,
                                  "ESP": 6.885, "PTE": 1.393}),
]

# Irrevocable conversion rates, units of legacy currency per euro (1999-01-01; GRD 2001-01-01)
EURO_FIXED_RATES = {"DEM": 1.95583, "FRF": 6.55957, "ITL": 1936.27, "NLG": 2.20371, "BEF": 40.3399,
                    "LUF": 40.3399, "ESP": 166.386, "ATS": 13.7603, "FIM": 5.94573, "IEP": 0.787564,
                    "PTE": 200.482, "GRD": 340.750}


def _quotes(fx: pd.DataFrame, max_stale_days: int = 7, keep_xeu: bool = False) -> pd.DataFrame:
    """Wide calendar-day table of units per GBP, each quote carried at most
    `max_stale_days` forward (Compustat quotes every calendar day inside a currency's range).
    Compustat's own ECU quote (XEU) is dropped unless `keep_xeu` (it is only a check)."""
    if not keep_xeu:
        fx = fx[fx["tocurd"] != NUMERAIRE]
    q = fx.pivot_table(index="datadate", columns="tocurd", values="exratd").sort_index()
    q.index = pd.DatetimeIndex(q.index).astype("datetime64[ns]")
    q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D")).ffill(limit=max_stale_days)
    q["GBP"] = 1.0
    if "LUF" not in q or q["LUF"].isna().all():
        q["LUF"] = q.get("BEF")
    else:
        q["LUF"] = q["LUF"].fillna(q.get("BEF"))
    return q


def basket_value_gbp(q: pd.DataFrame, basket: dict[str, float]) -> pd.Series:
    """GBP value of one ECU under `basket` on every date of `q`; NaN where a component
    currency has no quote."""
    missing = [c for c in basket if c not in q]
    if missing:
        raise ValueError(f"fx table lacks basket currencies {missing}")
    parts = pd.DataFrame({c: a / q[c] for c, a in basket.items()})
    return parts.sum(axis=1, min_count=len(basket))


def ecu_per_gbp(fx: pd.DataFrame, max_stale_days: int = 7) -> pd.Series:
    """ECU per GBP on the calendar-day grid, the basket in force on each date. Dates
    before the first basket or from 1999-01-01 on are NaN."""
    q = _quotes(fx, max_stale_days)
    out = pd.Series(np.nan, index=q.index)
    starts = [b[0] for b in BASKETS] + [EURO_START]
    for (start, basket), end in zip(BASKETS, starts[1:]):
        sel = (q.index >= start) & (q.index < end)
        if sel.any():
            out[sel] = 1.0 / basket_value_gbp(q.loc[sel], basket)
    return out.rename("ecu_per_gbp")


def numeraire_per_gbp(fx: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """Date -> XEU per GBP and its source: the ECU basket before 1999-01-01, Compustat's
    EUR quote from then on."""
    q = _quotes(fx, max_stale_days)
    ecu = ecu_per_gbp(fx, max_stale_days)
    eur = q["EUR"] if "EUR" in q else pd.Series(np.nan, index=q.index)
    after = q.index >= EURO_START
    val = pd.Series(np.where(after, eur.reindex(q.index), ecu.reindex(q.index)), index=q.index)
    src = np.where(after, "eur_quote", "ecu_basket")
    out = pd.DataFrame({"date": q.index, "xeu_per_gbp": val.to_numpy(), "source": src})
    return out[out["xeu_per_gbp"].notna()].reset_index(drop=True)


def with_numeraire(fx: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """Compustat's fx rows plus the pseudo currency XEU (rows datadate, tocurd, exratd), so
    that `convert_to_currency(..., target='XEU')` converts into ECU/EUR."""
    num = numeraire_per_gbp(fx, max_stale_days)
    x = pd.DataFrame({"datadate": num["date"], "tocurd": NUMERAIRE, "exratd": num["xeu_per_gbp"]})
    base = fx[fx["tocurd"] != NUMERAIRE][["datadate", "tocurd", "exratd"]]
    out = pd.concat([base, x], ignore_index=True)
    out["tocurd"] = out["tocurd"].astype(fx["tocurd"].dtype)       # merge_asof `by` keys must match the prices' dtype
    out["datadate"] = out["datadate"].astype("datetime64[ns]")
    out["exratd"] = out["exratd"].astype("float64")                 # WRDS hands back nullable Float64
    return out.sort_values(["tocurd", "datadate"], ignore_index=True)


def conversion_table(fx: pd.DataFrame, max_stale_days: int = 7) -> pd.DataFrame:
    """Long table (date, currency) -> eur_per_unit = units of the numeraire (ECU before
    1999, EUR after) per unit of the currency, plus `numeraire` (ECU | EUR). Same columns
    that eu_panel.daily_returns reads, so the returns are in the numeraire."""
    q = _quotes(fx, max_stale_days)
    num = numeraire_per_gbp(fx, max_stale_days).set_index("date")["xeu_per_gbp"].reindex(q.index)
    label = np.where(q.index >= EURO_START, "EUR", "ECU")
    parts = []
    for ccy in q.columns:
        v = num / q[ccy]
        t = pd.DataFrame({"date": q.index, "currency": ccy, "eur_per_unit": v.to_numpy(), "numeraire": label})
        parts.append(t[t["eur_per_unit"].notna()])
    tab = pd.concat(parts, ignore_index=True)
    # A legacy euro currency after its last Compustat quote continues at the fixed rate
    # (for fundamentals reported in it late; no price is quoted in one after 2001).
    ext = []
    eur_dates = q.index[q.index >= EURO_START]
    for ccy, rate in EURO_FIXED_RATES.items():
        have = tab.loc[tab["currency"] == ccy, "date"]
        last = pd.Timestamp(have.max()) if len(have) else pd.Timestamp("1998-12-31")
        days = eur_dates[eur_dates > last]
        if len(days):
            ext.append(pd.DataFrame({"date": days, "currency": ccy, "eur_per_unit": 1.0 / rate, "numeraire": "EUR"}))
    if ext:
        tab = pd.concat([tab] + ext, ignore_index=True)
    return tab.sort_values(["currency", "date"], ignore_index=True)


def checks(fx: pd.DataFrame, official: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Numbers for the build report: (1) the basket on 1998-12-31 against the irrevocable
    conversion rates (the eleven of 1999; GRD's was set in 2000), (2) old over new basket
    on each revision day (1 = no jump), (3) against the official daily ECU rates of
    Eurostat (`official`: date index, columns = currencies, units per ECU; dataset
    ert_bil_eur_d), (4) against Compustat's own ECU quote (XEU) and (5) against Compustat's
    synthetic EUR before 1999."""
    q = _quotes(fx, keep_xeu=True)
    out = {}
    d = pd.Timestamp("1998-12-31")
    ecu = ecu_per_gbp(fx)
    if d in ecu.index and pd.notna(ecu[d]):
        rows = []
        for ccy, fixed in EURO_FIXED_RATES.items():
            if ccy != "GRD" and ccy in q and pd.notna(q.loc[d, ccy]):
                basket_rate = q.loc[d, ccy] / ecu[d]          # units of ccy per ECU
                rows.append((ccy, fixed, basket_rate, basket_rate / fixed - 1))
        out["seam_1998"] = pd.DataFrame(rows, columns=["currency", "fixed_rate", "basket_1998_12_31", "rel_diff"])
    rows = []
    for (s_old, b_old), (s_new, b_new) in zip(BASKETS[:-1], BASKETS[1:]):
        if s_new in q.index:
            day = q.loc[[s_new]]
            try:
                rows.append((s_new.date(), float(basket_value_gbp(day, b_old).iloc[0] / basket_value_gbp(day, b_new).iloc[0])))
            except ValueError:
                pass
    out["revisions"] = pd.DataFrame(rows, columns=["revision_day", "old_over_new"])
    if official is not None and len(official):
        parts = {}
        for c in official.columns:
            if c == "GBP" or c in q:
                basket = (1.0 if c == "GBP" else q[c]) / ecu          # units of c per ECU
                rel = (basket.reindex(official.index) / official[c] - 1).dropna()
                parts[f"{c}_mean"] = rel.groupby(rel.index.year).mean()
                parts[f"{c}_med_abs"] = rel.abs().groupby(rel.index.year).median()
        out["vs_official"] = pd.DataFrame(parts).rename_axis("year")
    raw_xeu = fx[fx["tocurd"] == NUMERAIRE].set_index("datadate")["exratd"]   # quote days only, not carried
    raw_xeu.index = pd.DatetimeIndex(raw_xeu.index).astype("datetime64[ns]")
    for name, other in [("vs_compustat_ecu", raw_xeu), ("vs_compustat_eur", q.get("EUR"))]:
        if other is None:
            continue
        pre = pd.DataFrame({"ecu": ecu, "other": other}).dropna()
        pre = pre[pre.index < EURO_START]
        if len(pre):
            ratio = pre["other"] / pre["ecu"] - 1          # Compustat's units per GBP over the basket's
            out[name] = ratio.groupby(ratio.index.year).agg(["size", "mean", "min", "max"]).rename_axis("year")
    return out
