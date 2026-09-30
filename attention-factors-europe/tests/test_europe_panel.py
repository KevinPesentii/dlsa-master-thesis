"""European builds: the Corwin-Schultz spread, the line that carries a company's series, and
the 2x3 factor sort (europe_panel). Numbers built by hand."""

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
