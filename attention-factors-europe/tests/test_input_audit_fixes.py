"""The data rules added after the input audit of 2026-11-17
(Thesis/diagnostics/2026-11-17_input_data_audit): accounts zeros, ECU accounts, bad FX
quotes, company merges, broken-quote days, Nasdaq volume, ranking screens, December ME,
value traded. Numbers built by hand."""

import numpy as np
import pandas as pd
import pytest

from afe.data import characteristics as ch
from afe.data import compustat_global as cg
from afe.data import europe_accounts as xa
from afe.data import europe_fx as efx
from afe.data import us_panel as up


# ------------------------------------------------------------------ Europe accounts


def _fx_eur(dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "currency": "EUR", "eur_per_unit": 1.0})


def _funda(**items):
    base = dict(gvkey="1", datadate=pd.Timestamp("2000-12-31"), fyear=2000, fyr=12, curcd="EUR", indfmt="INDL",
                at=100.0, lt=60.0, ceq=40.0, sale=80.0, revt=80.0, cogs=50.0, xsga=20.0, xopr=70.0,
                ib=5.0, xido=0.0, nicon=np.nan, mib=np.nan, mibt=np.nan)
    return pd.DataFrame([{**base, **items}])


def test_zero_cogs_and_xsga_next_to_operating_costs_are_missing():
    f = xa.funda_numeraire(_funda(cogs=0.0, xsga=0.0), _fx_eur(["2000-12-31"]))
    assert np.isnan(f["cogs"].iloc[0]) and np.isnan(f["xsga"].iloc[0])
    assert np.isnan(f["gp"].iloc[0])                 # gp = revt - cogs is not revt


def test_a_zero_is_kept_when_the_other_item_is_all_of_operating_costs():
    f = xa.funda_numeraire(_funda(cogs=0.0, xsga=70.0), _fx_eur(["2000-12-31"]))
    assert f["cogs"].iloc[0] == 0.0 and f["xsga"].iloc[0] == 70.0
    g = xa.funda_numeraire(_funda(cogs=69.0, xsga=0.0), _fx_eur(["2000-12-31"]))
    assert np.isnan(g["xsga"].iloc[0])               # 69 is not within 1% of 70: the split is incomplete


def test_a_record_with_zero_total_assets_has_no_balance_sheet():
    f = xa.funda_numeraire(_funda(at=0.0, lt=0.0, ceq=0.0), _fx_eur(["2000-12-31"]))
    assert f[["at", "lt", "ceq"]].isna().all(axis=None)
    assert f["sale"].iloc[0] == 80.0                 # the income statement stays


def test_accounts_from_replaces_the_records_in_the_span():
    f = pd.concat([_funda(gvkey="A", at=60.0), _funda(gvkey="B", at=100.0),
                   _funda(gvkey="A", at=70.0, datadate=pd.Timestamp("2006-12-31"))], ignore_index=True)
    out = xa.accounts_from(f, [{"gvkey": "A", "source": "B", "start": "1984-01-01", "end": "2005-07-31"}])
    a = out[out["gvkey"] == "A"].set_index("datadate")["at"]
    assert a[pd.Timestamp("2000-12-31")] == 100.0 and a[pd.Timestamp("2006-12-31")] == 70.0
    assert (out["gvkey"] == "B").sum() == 1          # the source keeps its own record


def test_ecu_accounts_convert_at_one():
    fx = pd.DataFrame({"datadate": pd.to_datetime(["2000-01-03"] * 2), "tocurd": ["EUR", "USD"],
                       "exratd": [1.6, 1.6]})
    tab = efx.conversion_table(fx)
    x = tab[(tab["currency"] == efx.NUMERAIRE) & (tab["date"] == pd.Timestamp("2000-01-03"))]
    assert x["eur_per_unit"].iloc[0] == 1.0


def test_a_bad_fx_quote_and_its_carried_copies_are_dropped():
    days = pd.date_range("1993-07-22", "1993-07-26")
    fx = pd.DataFrame({"datadate": days, "tocurd": "NOK", "exratd": [11.035, 11.985, 11.985, 11.985, 11.035]})
    out = efx.drop_bad_quotes(fx, [{"currency": "NOK", "date": "1993-07-23"}])
    assert list(out["datadate"]) == [pd.Timestamp("1993-07-22"), pd.Timestamp("1993-07-26")]


# ------------------------------------------------------------------ Europe cap table


def test_merge_companies_adds_the_second_half_and_retires_it():
    m = pd.DataFrame({"gvkey": ["P", "N", "P", "N"], "datadate": pd.to_datetime(["2000-01-31"] * 2 + ["2019-01-31"] * 2),
                      "mktcap": [60.0, 40.0, 70.0, 45.0], "n_classes": [1, 1, 1, 1], "eligible": True})
    out = cg.merge_companies(m, [{"into": "P", "from": "N", "start": "1984-01-01", "end": "2018-09-30"}])
    jan2000 = out[out["datadate"] == "2000-01-31"].set_index("gvkey")
    assert jan2000.loc["P", "mktcap"] == 100.0 and jan2000.loc["P", "n_classes"] == 2
    assert not jan2000.loc["N", "eligible"]
    after = out[out["datadate"] == "2019-01-31"].set_index("gvkey")
    assert after.loc["P", "mktcap"] == 70.0 and after.loc["N", "eligible"]   # outside the span: untouched


# ------------------------------------------------------------------ US daily data


def _days(rows):
    cols = ["permno", "date", "ret", "prc", "vol", "bid", "ask", "high"]
    return pd.DataFrame(rows, columns=cols).assign(date=lambda t: pd.to_datetime(t["date"]))


def test_a_broken_quote_day_is_folded_into_the_next_return():
    # the ABN AMRO rows as CRSP has them: no volume, bid 53 / ask 141, close 97 (+82.6%); an empty
    # row on 10-05; then 53.01 (-45.35%, against 97)
    d = _days([(1, "2007-10-03", 0.0011, 53.13, 1e6, 53.06, 53.10, 53.44),
               (1, "2007-10-04", 97 / 53.13 - 1, 97.0, 0.0, 53.0, 141.0, np.nan),
               (1, "2007-10-05", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
               (1, "2007-10-08", 53.01 / 97 - 1, 53.01, 3e5, 53.06, 53.15, 53.49)])
    out = up.fold_broken_quotes(d, 0.10)
    assert out["ret"].iloc[1] == 0.0 and out["prc"].iloc[1] == 53.13
    assert np.isnan(out["ret"].iloc[2])                                        # the empty row stays empty
    assert out["ret"].iloc[3] == pytest.approx(53.01 / 53.13 - 1)            # the return since 10-03 is CRSP's
    assert out["ret"].iloc[0] == d["ret"].iloc[0]


def test_a_traded_day_with_a_wide_quote_is_left_alone():
    d = _days([(1, "2007-10-03", 0.0, 50.0, 1e6, 50.0, 50.1, 50.5),
               (1, "2007-10-04", 0.2, 60.0, 1e3, 40.0, 80.0, 61.0)])
    out = up.fold_broken_quotes(d, 0.10)
    assert out["ret"].iloc[1] == 0.2 and not out["broken_quote"].any()


def test_nasdaq_volume_divisors():
    dates = pd.Series(pd.to_datetime(["2000-06-30", "2001-06-29", "2003-06-30", "2004-06-30", "2000-06-30"]))
    exch = pd.Series(["Q", "Q", "Q", "Q", "N"])
    assert list(up.nasdaq_divisor(dates, exch)) == [2.0, 1.8, 1.6, 1.0, 1.0]


def test_vol_is_value_traded():
    idx = pd.bdate_range("2000-01-03", periods=5)
    vol = pd.DataFrame({1: [100.0] * 5, 2: [10.0] * 5}, index=idx)
    prc = pd.DataFrame({1: [1.0] * 5, 2: [20.0] * 5}, index=idx)
    D = ch.DailyInputs(ret=vol * 0, prc=prc, high=prc, vol=vol, cumfacpr=prc * 0 + 1, ff=pd.DataFrame(index=idx),
                       cfg={"weekly_days": 5})
    v = ch.weekly_volume(D).iloc[-1]
    assert v[1] == 500.0 and v[2] == 1000.0          # fewer shares, more value: ranks higher


# ------------------------------------------------------------------ US universe screens


def test_rankable_needs_a_prior_month_and_an_active_receipt():
    M = pd.Period("2000-06", "M")
    frame = pd.DataFrame({"primary_permno": [1, 2, 3], "month": [M, M, M], "cap_co": [1.0, 2.0, 3.0]})
    monthly = pd.DataFrame({"permno": [1, 1, 2, 3, 3], "date": pd.to_datetime(
        ["2000-05-31", "2000-06-30", "2000-06-30", "2000-05-31", "2000-06-30"])})
    elig = pd.DataFrame({"permno": [1, 2, 3], "month": [M, M, M], "sharetype": ["NS", "NS", "AD"]})
    activity = pd.DataFrame({"permno": [3], "month": [M], "traded_share": [0.5]})
    cfg = {"universe": {"min_history_months": 1, "receipt_min_traded_share": 0.9}}
    out = up.rankable(frame, elig, monthly, cfg, activity)
    assert list(out["primary_permno"]) == [1]        # 2: first month listed; 3: a receipt trading half the days


# ------------------------------------------------------------------ December ME


def test_dec_me_far_from_the_company_cap_is_dropped():
    months = pd.period_range("1990-12", "1991-09", freq="M")
    cap = pd.DataFrame({1: 43_000.0, 2: 1_000.0}, index=months)                 # USD mn
    ret = pd.DataFrame({1: 0.0, 2: 0.0}, index=months)
    ret.loc[pd.Period("1991-03", "M"), 2] = 9.0                                  # company 2 rose tenfold
    cap.loc[pd.Period("1991-03", "M"):, 2] = 10_000.0
    aligned = pd.DataFrame({"permno": [1, 2], "month": [pd.Period("1991-09", "M")] * 2, "BEME": [6387.0, 0.5],
                            "_dd_BEME": pd.to_datetime(["1990-03-31", "1990-03-31"]), "_me_BEME": [1.6, 1_000.0]})
    out = up.validate_dec_me(aligned, ["BEME"], cap, ret, 8.0)
    assert np.isnan(out["BEME"].iloc[0])             # $1.6m against a $43bn company: a receipt count, not the company
    assert out["BEME"].iloc[1] == 0.5                # a tenfold cap explained by a tenfold return stays
    assert not any(c.startswith(("_dd_", "_me_")) for c in out.columns)


# ------------------------------------------------------------------ Kevin's data check (2026-11-17)


def test_rankable_drops_a_line_no_longer_listed_at_the_month_end():
    M = pd.Period("2000-06", "M")
    frame = pd.DataFrame({"primary_permno": [1, 2], "month": [M, M], "cap_co": [1.0, 2.0]})
    monthly = pd.DataFrame({"permno": [1, 1, 2, 2], "date": pd.to_datetime(["2000-05-31", "2000-06-30"] * 2)})
    elig = pd.DataFrame({"permno": [1, 2], "month": [M, M], "sharetype": ["NS", "NS"]})
    listed = pd.DataFrame({"permno": [1, 2], "month": [M, M], "listed": [True, False]})
    out = up.rankable(frame, elig, monthly, {"universe": {}}, None, listed)
    assert list(out["primary_permno"]) == [1]


def test_a_company_whose_line_is_delisted_mid_month_moves_to_its_other_line():
    from afe.data import europe_panel as xp
    days = pd.bdate_range("2007-11-12", "2007-11-16")
    rows = []
    for d in days:
        if d < pd.Timestamp("2007-11-13"):
            rows.append(dict(gvkey="1", iid="03W", date=d, ret=0.01, cshtrd=100.0))
        rows.append(dict(gvkey="1", iid="01W", date=d, ret=0.02, cshtrd=50.0))
    dret = pd.DataFrame(rows)
    mk = pd.DataFrame({"gvkey": ["1"], "datadate": pd.to_datetime(["2007-10-31"]), "iid": ["03W"]})
    dl = pd.DataFrame({"gvkey": ["1"], "iid": ["03W"], "dldtei": pd.to_datetime(["2007-11-13"])})
    cd = xp.company_daily(dret, mk, pd.DatetimeIndex(days), dl).set_index("date")
    assert cd.loc["2007-11-12", "iid"] == "03W"
    assert (cd.loc["2007-11-13":, "iid"] == "01W").all() and len(cd) == 5
    without = xp.company_daily(dret, mk, pd.DatetimeIndex(days))
    assert len(without) == 1                          # the old rule: one day of returns, then nothing


def test_an_untraded_row_keeps_the_last_traded_close():
    from afe.data import eu_panel as ep
    d = pd.DataFrame({"gvkey": "1", "iid": "02W", "datadate": pd.to_datetime(["1997-05-01", "1997-05-02", "1997-05-05", "1997-05-06"]),
                      "curcdd": "GBP", "prccd": [5.325, 5.125, 5.325, 5.215], "ajexdi": 1.0, "trfd": 1.0, "qunit": 1.0,
                      "cshoc": 1.0, "cshtrd": 1.0, "prchd": np.nan, "prcld": np.nan, "prcstd": [10, 10, 5, 10]})
    fx = pd.DataFrame({"date": pd.date_range("1997-04-25", "1997-05-10"), "currency": "GBP", "eur_per_unit": 1.0})
    r = ep.daily_returns(d, fx, carry_untraded=True)["ret_local"].to_numpy()
    assert r[2] == 0.0 and r[3] == pytest.approx(5.215 / 5.125 - 1)


def test_daily_rf_from_the_monthly_tbill():
    from afe.data import build_us as bu
    idx = pd.bdate_range("2019-01-01", "2019-12-31")
    d = pd.DataFrame({"mktrf": 0.001, "rf": 0.0001, "smb": 0.0, "hml": 0.0}, index=idx)
    m = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=12, freq="MS"), "rf": 0.002})
    n = bu.rf_from_monthly(d, m)
    jan = n.loc["2019-01"]
    # the first row accrues one day, Jan 2 .. Jan 31 the 30 days since: all 31 days of January's 0.2%
    assert (1 + jan["rf"]).prod() - 1 == pytest.approx(0.002, rel=1e-9)
    feb = n.loc["2019-02"]                            # Feb 1 accrues Jan 31 -> Feb 1, ..., Feb 28: 28 of 28 days
    assert (1 + feb["rf"]).prod() - 1 == pytest.approx(0.002, rel=1e-9)
    assert np.allclose(n["mktrf"] + n["rf"], d["mktrf"] + d["rf"])
