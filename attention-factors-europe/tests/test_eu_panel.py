"""EUR market data layer: the numeraire rule, EUR returns of foreign-quoted lines, the
one-month universe lag and the value-weighted market. Numbers hand-built."""

import numpy as np
import pandas as pd
import pytest

from afe.data import eu_panel as ep

DEM = ep.DEM_PER_EUR


def _fx():
    """GBP-based quotes: EUR and USD every day, DEM only until 1995-01-02 (like Compustat
    stopping DEM in 2018), on a window straddling the 1994-01-01 switch."""
    days = pd.date_range("1993-12-30", "1995-01-05", freq="D")
    rows = []
    for i, d in enumerate(days):
        rows.append(dict(datadate=d, tocurd="EUR", exratd=1.30 + 0.001 * i))
        rows.append(dict(datadate=d, tocurd="USD", exratd=1.50 + 0.002 * i))
        rows.append(dict(datadate=d, tocurd="GBP", exratd=1.0))
        if d <= pd.Timestamp("1995-01-02"):
            rows.append(dict(datadate=d, tocurd="DEM", exratd=2.50 - 0.001 * i))
    return pd.DataFrame(rows)


def test_numeraire_is_eur_from_switch_and_mark_before():
    fx = _fx()
    tab = ep.fx_conversion_table(fx, switch="1994-01-01").set_index(["currency", "date"])
    q = fx.pivot(index="datadate", columns="tocurd", values="exratd")
    after, before = pd.Timestamp("1994-06-01"), pd.Timestamp("1993-12-31")
    # after the switch: EUR quote over the local quote
    assert tab.loc[("USD", after), "eur_per_unit"] == pytest.approx(q.loc[after, "EUR"] / q.loc[after, "USD"])
    assert tab.loc[("USD", after), "eur_source"] == "eur_quote"
    # before the switch: Deutsche Mark at the irrevocable rate
    assert tab.loc[("USD", before), "eur_per_unit"] == pytest.approx(q.loc[before, "DEM"] / q.loc[before, "USD"] / DEM)
    assert tab.loc[("USD", before), "eur_source"] == "dem_fixed"
    # DEM leg: quoted while Compustat quotes it, EUR x 1.95583 afterwards
    assert tab.loc[("USD", after), "dem_per_unit"] == pytest.approx(q.loc[after, "DEM"] / q.loc[after, "USD"])
    late = pd.Timestamp("1995-01-05")
    assert tab.loc[("USD", late), "dem_per_unit"] == pytest.approx(q.loc[late, "EUR"] * DEM / q.loc[late, "USD"])
    assert tab.loc[("USD", late), "dem_source"] == "eur_fixed"
    # DEM itself continues as a currency at the fixed rate
    assert tab.loc[("DEM", late), "eur_per_unit"] == pytest.approx(1 / DEM)
    assert tab.loc[("GBP", after), "eur_per_unit"] == pytest.approx(q.loc[after, "EUR"])


def _daily(rows):
    base = dict(ajexdi=1.0, trfd=1.0, qunit=1.0, cshoc=100.0, cshtrd=10.0, prchd=np.nan, prcld=np.nan, prcstd=10)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_eur_return_of_a_gbp_line_carries_the_currency_move():
    fx = _fx()
    tab = ep.fx_conversion_table(fx)
    days = pd.bdate_range("1994-03-01", "1994-03-04")
    # constant GBP price: the EUR return is exactly the EUR/GBP move
    d = _daily([dict(gvkey="1", iid="01W", datadate=t, prccd=10.0, curcdd="GBP") for t in days])
    r = ep.daily_returns(d, tab).set_index("date")
    q = fx.pivot(index="datadate", columns="tocurd", values="exratd")["EUR"]
    for t0, t1 in zip(days[:-1], days[1:]):
        assert r.loc[t1, "ret"] == pytest.approx(q.loc[t1] / q.loc[t0] - 1)
        assert r.loc[t1, "ret_local"] == pytest.approx(0.0)


def test_quotation_switch_split_and_dividend_leave_no_break():
    fx = _fx()
    tab = ep.fx_conversion_table(fx)
    q = fx.pivot(index="datadate", columns="tocurd", values="exratd")
    t = pd.bdate_range("1994-03-01", "1994-03-04")
    dem_eur = lambda d: q.loc[d, "DEM"] / q.loc[d, "EUR"]  # noqa: E731
    # day 0-1 quoted in DEM, day 2 on in EUR at the same EUR value; day 2 also a 2:1
    # split: Compustat's cumulative ajexdi puts the PRE-split rows on today's basis
    # (2.0 before, 1.0 after); day 3 a dividend folded into trfd
    d = _daily([
        dict(gvkey="1", iid="01W", datadate=t[0], prccd=20.0 * dem_eur(t[0]), curcdd="DEM", ajexdi=2.0),
        dict(gvkey="1", iid="01W", datadate=t[1], prccd=22.0 * dem_eur(t[1]), curcdd="DEM", ajexdi=2.0),
        dict(gvkey="1", iid="01W", datadate=t[2], prccd=11.0, curcdd="EUR"),
        dict(gvkey="1", iid="01W", datadate=t[3], prccd=11.0, curcdd="EUR", trfd=1.05),
    ])
    r = ep.daily_returns(d, tab).set_index("date")["ret"]
    assert r.loc[t[1]] == pytest.approx(0.10)     # 20 -> 22 EUR-equivalent, currency neutral
    assert r.loc[t[2]] == pytest.approx(0.0)      # 22 EUR -> 11 EUR post-split = flat
    assert r.loc[t[3]] == pytest.approx(0.05)     # price flat, 5% distribution


def _mktcap():
    """Three companies, four month ends; caps chosen so the ranking flips in the last."""
    months = pd.date_range("1999-01-31", "1999-04-30", freq="ME")
    rows = []
    for i, m in enumerate(months):
        caps = {"A": 300.0, "B": 200.0, "C": 100.0} if i < 3 else {"A": 100.0, "B": 200.0, "C": 300.0}
        for g, c in caps.items():
            rows.append(dict(datadate=m, gvkey=g, iid="01W", mktcap=c, country="DEU", eligible=True))
    return pd.DataFrame(rows)


def test_universe_month_m_uses_ranking_at_end_of_m_minus_1():
    cal = pd.bdate_range("1999-01-01", "1999-06-30")
    asof = ep.asof_ranking(_mktcap(), n=2)
    uni, detail = ep.universe_lagged(asof, cal, "1999-02", "1999-05", n=2)
    may = uni[uni["month"] == cal[cal.to_period("M") == pd.Period("1999-05", "M")].min()]
    assert set(may["sec_id"]) == {"C", "B"}                # ranking of 1999-04-30
    apr = uni[uni["month"] == cal[cal.to_period("M") == pd.Period("1999-04", "M")].min()]
    assert set(apr["sec_id"]) == {"A", "B"}                # ranking of 1999-03-31
    assert uni.groupby("month").size().eq(2).all()
    with pytest.raises(ValueError, match="exactly"):
        ep.universe_lagged(asof, cal, "1999-01", "1999-05", n=2)   # no ranking at 1998-12-31


def test_returns_table_lags_cap_and_market_is_value_weighted():
    mk = _mktcap()
    days = pd.bdate_range("1999-04-01", "1999-05-05")
    rows = []
    for g, r in {"A": 0.01, "B": 0.02, "C": -0.01}.items():
        for t in days:
            rows.append(dict(gvkey=g, iid="01W", date=t, ret=r, curcdd="EUR"))
    dret = pd.DataFrame(rows)
    ret = ep.returns_table(dret, mk, "1999-04", "1999-05")
    a_apr = ret[(ret["sec_id"] == "A") & (ret["date"] == days[0])]
    a_may = ret[(ret["sec_id"] == "A") & (ret["date"] == pd.Timestamp("1999-05-03"))]
    assert a_apr["mktcap_lag"].item() == 300.0 and a_may["mktcap_lag"].item() == 100.0
    assert set(ret["country"]) == {"DE"} and set(ret["currency"]) == {"EUR"}
    cal = pd.bdate_range("1999-01-01", "1999-06-30")
    uni, _ = ep.universe_lagged(ep.asof_ranking(mk, 2), cal, "1999-04", "1999-05", 2)
    mkt = ep.market_returns(ret, uni).set_index("date")
    # April: A (300) and B (200) -> vw = (300*0.01 + 200*0.02)/500
    assert mkt.loc[days[0], "mkt_vw"] == pytest.approx((300 * 0.01 + 200 * 0.02) / 500)
    assert mkt.loc[days[0], "mkt_ew"] == pytest.approx(0.015) and mkt.loc[days[0], "n"] == 2
    # May: C (300) and B (200) -> vw = (300*-0.01 + 200*0.02)/500
    assert mkt.loc[pd.Timestamp("1999-05-03"), "mkt_vw"] == pytest.approx((300 * -0.01 + 200 * 0.02) / 500)
