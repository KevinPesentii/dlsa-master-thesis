"""European builds: the ECU/EUR numeraire (europe_fx). Numbers built by hand."""

import pandas as pd
import pytest

from afe.data import europe_fx as efx

B89 = dict(efx.BASKETS[2][1])


def _fx(days, eur=1.40):
    """Every basket currency quoted at 100 x its basket amount per GBP, so each non-GBP
    component is worth exactly 0.01 GBP; EUR quoted from 1999 only."""
    rows = []
    for d in days:
        for c, a in B89.items():
            if c != "GBP":
                rows.append(dict(datadate=d, tocurd=c, exratd=100.0 * a))
        rows.append(dict(datadate=d, tocurd="USD", exratd=1.60))
        if d >= efx.EURO_START:
            rows.append(dict(datadate=d, tocurd="EUR", exratd=eur))
    return pd.DataFrame(rows)


def test_ecu_is_the_1989_basket_before_1999_and_the_euro_after():
    days = pd.date_range("1998-12-29", "1999-01-05", freq="D")
    fx = _fx(days)
    num = efx.numeraire_per_gbp(fx).set_index("date")
    gbp_per_ecu = 11 * 0.01 + B89["GBP"]                       # eleven components at 0.01 GBP, plus GBP itself
    assert num.loc["1998-12-30", "xeu_per_gbp"] == pytest.approx(1 / gbp_per_ecu)
    assert num.loc["1998-12-30", "source"] == "ecu_basket"
    assert num.loc["1999-01-04", "xeu_per_gbp"] == pytest.approx(1.40)
    assert num.loc["1999-01-04", "source"] == "eur_quote"
    tab = efx.conversion_table(fx).set_index(["currency", "date"])
    # DEM before 1999: numeraire per GBP over DEM per GBP
    assert tab.loc[("DEM", pd.Timestamp("1998-12-30")), "eur_per_unit"] == pytest.approx((1 / gbp_per_ecu) / (100 * B89["DEM"]))
    assert tab.loc[("DEM", pd.Timestamp("1998-12-30")), "numeraire"] == "ECU"
    assert tab.loc[("USD", pd.Timestamp("1999-01-04")), "eur_per_unit"] == pytest.approx(1.40 / 1.60)


def test_basket_in_force_depends_on_the_date():
    b84 = efx.BASKETS[1][1]
    days = pd.date_range("1989-09-19", "1989-09-22", freq="D")
    rev = pd.Timestamp("1989-09-21")
    # quotes at 100 x the amounts of the basket in force that day (ESP, PTE: the 1989 ones throughout)
    rows = [dict(datadate=d, tocurd=c, exratd=100.0 * ((b84 if d < rev else B89).get(c) or B89[c]))
            for d in days for c in set(b84) | set(B89) if c != "GBP"]
    num = efx.numeraire_per_gbp(pd.DataFrame(rows)).set_index("date")["xeu_per_gbp"]
    # 1984 basket: nine components worth 0.01 GBP + GBP 0.0878; ESP and PTE are not in it
    assert num["1989-09-20"] == pytest.approx(1 / (9 * 0.01 + b84["GBP"]))
    # 1989 basket from 1989-09-21: eleven components worth 0.01 GBP + GBP 0.08784
    assert num["1989-09-21"] == pytest.approx(1 / (11 * 0.01 + B89["GBP"]))
