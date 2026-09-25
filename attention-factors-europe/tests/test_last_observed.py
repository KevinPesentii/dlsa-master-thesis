"""Missing characteristics take the stock's last observed value, else the cross-sectional
median: Epstein et al. (2025), Section 4.1. The median half is unchanged (build_us); these
cases pin the carry.

1. Annual: an item missing from the latest fiscal year comes from the year before, but
   only while that year is within max_age_months; items the latest year has are its own.
2. Monthly: a value is carried at most `carry` months; nothing before the first
   observation is filled; truncating the input leaves the earlier values unchanged.
3. Daily: the same with trading days.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from afe.data import build_us
from afe.data import characteristics as ch
from afe.data import us_panel as up

P = lambda s: pd.Period(s, "M")  # noqa: E731


def test_annual_item_missing_from_the_latest_year_comes_from_the_year_before_within_max_age():
    rec = pd.DataFrame([dict(gvkey="012345", datadate=pd.Timestamp("2000-12-31"), Lev=0.42, C=0.10,
                             avail_month=P("2001-06")),
                        dict(gvkey="012345", datadate=pd.Timestamp("2001-12-31"), Lev=np.nan, C=0.12,
                             avail_month=P("2002-06"))])
    months = [P("2002-06"), P("2003-06"), P("2003-07")]      # FY2000 is 18, 30, 31 months old
    mapping = pd.DataFrame(dict(permno=10001, month=months, gvkey="012345"))

    latest = up.align_annual(rec, mapping, ["Lev", "C"], max_age_months=30)
    assert latest["Lev"].isna().all()
    assert latest["C"].tolist() == pytest.approx([0.12] * 3)

    carried = up.align_annual(rec, mapping, ["Lev", "C"], max_age_months=30, last_observed=True)
    assert carried["Lev"].tolist()[:2] == pytest.approx([0.42, 0.42])
    assert np.isnan(carried["Lev"].iloc[2])
    assert carried["C"].tolist() == pytest.approx([0.12] * 3)


def _stub(name, frequency):
    return ch.Characteristic(name=name, theme="test", frequency=frequency, reference="", doc="",
                             fn=lambda X: X.value)


def test_monthly_value_is_carried_at_most_carry_months_and_never_backwards():
    months = pd.period_range("2000-01", "2001-04", freq="M")
    value = pd.DataFrame(np.nan, index=months, columns=[1, 2])
    value.loc[P("2000-01"), 1] = 0.002                    # then missing for 15 months
    value.loc[P("2000-06"), 2] = 0.003                    # missing before its first observation
    M = SimpleNamespace(ret=value, value=value)

    out = build_us.monthly_characteristics(M, [_stub("Spread", "monthly")], carry=12)["Spread"]
    a = out.xs(1, level="permno")
    assert a.loc[P("2000-01"):P("2001-01")].tolist() == pytest.approx([0.002] * 13)
    assert a.loc[P("2001-02"):].isna().all()
    b = out.xs(2, level="permno")
    assert b.loc[:P("2000-05")].isna().all() and b.loc[P("2000-06"):].notna().all()

    plain = build_us.monthly_characteristics(M, [_stub("Spread", "monthly")], carry=0)["Spread"]
    assert int(plain.notna().sum()) == 2

    cut = months[months <= P("2000-09")]
    Mc = SimpleNamespace(ret=value.loc[cut], value=value.loc[cut])
    early = build_us.monthly_characteristics(Mc, [_stub("Spread", "monthly")], carry=12)["Spread"]
    pd.testing.assert_series_equal(early, out.loc[early.index])


def test_daily_value_is_carried_at_most_carry_trading_days():
    dates = pd.bdate_range("2020-03-02", periods=10)
    value = pd.DataFrame(np.nan, index=dates, columns=[1])
    value.iloc[0, 0] = -0.031
    D = SimpleNamespace(value=value)

    out = build_us.daily_characteristics(D, [_stub("Ret_D1", "daily")], carry=5)["Ret_D1"]
    assert out.notna().tolist() == [True] * 6 + [False] * 4
