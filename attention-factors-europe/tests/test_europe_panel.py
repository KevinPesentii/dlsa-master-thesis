"""European builds: the Corwin-Schultz spread, the line that carries a company's series, the
2x3 factor sort, the risk-free accrual and the FF Europe factors in the numeraire
(europe_panel). Numbers built by hand."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from afe.data import europe_panel as xp


def _line(rows):
    base = dict(gvkey="1", iid="01W", prcstd=10)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_corwin_schultz_recovers_a_pure_bid_ask_bounce():
    # no volatility: high = ask, low = bid around an unchanged midpoint of 100 on two days;
    # the estimator's alpha is then exactly ln(H/L), so S = 2(H-L)/(H+L) = (ask-bid)/mid
    d = _line([dict(datadate=pd.Timestamp("2000-01-03"), prchd=100.5, prcld=99.5, prccd=100.0),
               dict(datadate=pd.Timestamp("2000-01-04"), prchd=100.5, prcld=99.5, prccd=100.0)])
    s = xp.corwin_schultz(d)
    assert np.isnan(s.iloc[0])                       # needs the previous day
    assert s.iloc[1] == pytest.approx(0.01)


def test_corwin_schultz_overnight_gap_is_removed_and_carried_days_skipped():
    base = [dict(datadate=pd.Timestamp("2000-01-03"), prchd=100.5, prcld=99.5, prccd=100.0),
            dict(datadate=pd.Timestamp("2000-01-04"), prchd=100.5, prcld=99.5, prccd=100.0)]
    gap = [base[0], dict(datadate=pd.Timestamp("2000-01-04"), prchd=103.5, prcld=102.5, prccd=103.0)]
    # a range wholly 3 above yesterday's close is moved down by the gap (2.5 = low - close),
    # to 101.0 / 100.0: a different, but finite and non-negative, estimate
    s = xp.corwin_schultz(_line(gap)).iloc[1]
    h1, l1 = 101.0, 100.0
    beta = np.log(100.5 / 99.5) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(max(100.5, h1) / min(99.5, l1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    assert s == pytest.approx(max(0.0, 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))))
    carried = [base[0], {**base[1], "prcstd": 5}]
    assert np.isnan(xp.corwin_schultz(_line(carried)).iloc[1])


def test_company_series_uses_the_line_of_the_previous_month_end():
    days = pd.bdate_range("2000-01-03", "2000-04-28")
    dret = pd.concat([pd.DataFrame({"gvkey": "1", "iid": iid, "date": days, "ret": r})
                      for iid, r in [("01W", 0.01), ("02W", 0.02)]], ignore_index=True)
    # the cap table has the company from 2000-01 on: line 01W at the end of Jan, 02W at the end of Feb, none in March
    mk = pd.DataFrame({"gvkey": "1", "datadate": pd.to_datetime(["2000-01-31", "2000-02-29"]), "iid": ["01W", "02W"]})
    cd = xp.company_daily(dret, mk, pd.DatetimeIndex(days)).set_index("date")["iid"]
    assert (cd["2000-01"] == "01W").all()            # before the first cap row: the first line
    assert (cd["2000-02"] == "01W").all()            # set at the end of January
    assert (cd["2000-03"] == "02W").all()            # set at the end of February
    assert (cd["2000-04"] == "02W").all()            # no row for March: the latest earlier one


def test_smb_hml_from_a_two_by_three_sort():
    # six companies at the end of 2000-01, one per portfolio; March returns are the company numbers / 100
    caps = {"a": 1, "b": 2, "c": 3, "d": 10, "e": 20, "f": 30}
    beme = {"a": 0.2, "b": 0.5, "c": 2.0, "d": 0.3, "e": 0.6, "f": 3.0}
    members = pd.DataFrame({"month": pd.Period("2000-02", "M"), "gvkey": list(caps), "cap": list(caps.values())})
    b = pd.DataFrame({"month": pd.Period("2000-01", "M"), "gvkey": list(beme), "BEME": list(beme.values())})
    day = pd.Timestamp("2000-02-15")
    rets = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05, "f": 0.06}
    cd = pd.DataFrame({"gvkey": list(rets), "date": day, "ret": list(rets.values())})
    out = xp.smb_hml(cd, members, b, min_per_portfolio=1).loc[day]
    # small = a, b, c (caps below the median 6.5); BEME 30/70 cutoffs over all six (0.45, 1.3):
    # a (0.2) and d (0.3) low; b (0.5) and e (0.6) middle; c (2.0) and f (3.0) high
    assert out["smb"] == pytest.approx((0.01 + 0.02 + 0.03) / 3 - (0.04 + 0.05 + 0.06) / 3)
    assert out["hml"] == pytest.approx((0.03 + 0.06) / 2 - (0.01 + 0.04) / 2)


def _bbk(path, rows):
    path.write_text("header\n" + "".join(f"{d},{v}\n" for d, v in rows))


def test_risk_free_earns_the_previous_quote_over_calendar_days(tmp_path):
    # FIBOR 8% on Thursday 1990-07-05, 9% on Friday: Friday's return is Thursday's 8% for one
    # day, Monday's is Friday's 9% for the three days since Friday (act/360)
    _bbk(tmp_path / "bbk_ST0104_frankfurt_1m_daily.csv", [("1990-07-02", 7.0)])
    _bbk(tmp_path / "bbk_ST0262_fibor_1m_daily.csv", [("1990-07-05", 8.0), ("1990-07-06", 9.0)])
    _bbk(tmp_path / "bbk_ST0310_euribor_1m_daily.csv", [("1999-01-04", 3.0)])
    cal = pd.DatetimeIndex(["1990-07-05", "1990-07-06", "1990-07-09"])
    rf = xp.risk_free(tmp_path, cal).set_index("date")["rf"]
    assert rf.loc["1990-07-06"] == pytest.approx(0.08 / 360)
    assert rf.loc["1990-07-09"] == pytest.approx(0.09 * 3 / 360)


def test_ff_europe_factors_in_the_numeraire(tmp_path):
    # all in USD: market +1% Mon, 0% Tue, +2% Wed (not a calendar day), 0% Thu; SMB +1% Tue;
    # HML +0.5% Wed. The dollar gains 10% against the numeraire on Tuesday. Numeraire market:
    # Mon 1%, Tue 10%, Thu Wednesday's 2%. SMB Tue: both legs gain the 10%, 1% x 1.1 = 1.1%.
    # HML Thu: Wednesday's 0.5%, the dollar flat since Tuesday
    csv = ("This file was created using the 202608 Bloomberg database.\n\n,Mkt-RF,SMB,HML,RF\n"
           "19900702    ,1.00    ,0.00   ,0.00    ,0.00\n19900703    ,0.00    ,1.00   ,0.00    ,0.00\n"
           "19900704    ,2.00    ,0.00   ,0.50    ,0.00\n19900705    ,0.00    ,0.00   ,0.00    ,0.00\n\n"
           "Copyright 2026 Eugene F. Fama and Kenneth R. French\n")
    path = tmp_path / "Europe_3_Factors_Daily_CSV.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Europe_3_Factors_Daily.csv", csv)
    fx = pd.DataFrame({"date": pd.to_datetime(["1990-06-29", "1990-07-02", "1990-07-03"]), "currency": "USD",
                       "eur_per_unit": [0.80, 0.80, 0.88], "numeraire": "XEU"})
    cal = pd.DatetimeIndex(["1990-06-29", "1990-07-02", "1990-07-03", "1990-07-05"])
    f = xp.ff_europe(path, fx, cal)
    assert f.loc["1990-06-29"].isna().all()
    assert f.loc["1990-07-02", "mkt"] == pytest.approx(0.01)
    assert f.loc["1990-07-03", "mkt"] == pytest.approx(0.10)
    assert f.loc["1990-07-05", "mkt"] == pytest.approx(0.02)
    assert f.loc["1990-07-03", "smb"] == pytest.approx(0.011)
    assert f.loc["1990-07-05", "smb"] == pytest.approx(0.0)
    assert f.loc["1990-07-05", "hml"] == pytest.approx(0.005)
    # with the monthly T-bill, RF = the July rate spread over the month's 4 file days: 0.1% a day
    pd.DataFrame({"date": pd.to_datetime(["1990-07-01"]), "rf": [1.001 ** 4 - 1]}).to_parquet(tmp_path / "m.parquet")
    assert xp.ff_europe(path, fx, cal, tmp_path / "m.parquet").loc["1990-07-02", "mkt"] == pytest.approx(0.011)
