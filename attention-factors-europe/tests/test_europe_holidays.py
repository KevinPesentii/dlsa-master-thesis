"""Holiday fill of the European returns table, numbers worked out by hand.

Pooled days d0..d5. A (GBP) has no row on d2, a London holiday; its d3 return of +10% is
the move since d1. Sterling gains 2% against the numeraire on d2 (1.50 -> 1.53 EUR).
Filled: d2 = +2% (zero local return, the currency's move), traded False; d3 = 1.10 / 1.02 - 1
= +7.8431%, so the two days still compound to +10%. B (EUR) misses four days (longer than
max_gap 3) and C (GBP) has no return after its gap: neither is filled.
"""

import numpy as np
import pandas as pd
import pytest

from afe.data.europe_holidays import fill_holiday_gaps

D = pd.bdate_range("2000-01-03", periods=6)  # d0 .. d5


def _row(sec, i, ret, cur):
    return {"date": D[i], "sec_id": sec, "ret": np.float32(ret), "mktcap_lag": np.float32(1e9),
            "country": "GBR" if cur == "GBP" else "DEU", "currency": cur}


def _case():
    rows = [_row("A", i, r, "GBP") for i, r in [(0, 0.01), (1, -0.01), (3, 0.10), (4, 0.0), (5, 0.02)]]
    rows += [_row("B", i, 0.005, "EUR") for i in (0, 5)]
    rows += [_row("C", i, 0.003, "GBP") for i in (0, 1, 2, 3)]
    ret = pd.DataFrame(rows)
    universe = pd.DataFrame({"month": D[0], "sec_id": ["A", "B", "C"], "cap_rank": [1, 2, 3]})
    gbp = [1.49, 1.50, 1.53, 1.53, 1.53, 1.54]
    fx = pd.DataFrame({"currency": ["GBP"] * 6 + ["EUR"] * 6, "date": list(D) * 2, "eur_per_unit": gbp + [1.0] * 6})
    traded = pd.DataFrame({"sec_id": ["A"], "date": [D[4]], "traded": [False]})  # a carried close
    return ret, universe, fx, traded


def test_one_day_holiday_is_filled_with_the_currency_move():
    out, st = fill_holiday_gaps(*_case(), max_gap=3)
    a = out[out["sec_id"] == "A"].set_index("date")
    assert len(a) == 6
    assert a.loc[D[2], "ret"] == pytest.approx(0.02, abs=1e-6)
    assert not a.loc[D[2], "traded"]
    assert a.loc[D[3], "ret"] == pytest.approx(1.10 / 1.02 - 1, abs=1e-6)
    assert (1 + a.loc[D[2], "ret"]) * (1 + a.loc[D[3], "ret"]) == pytest.approx(1.10, abs=1e-6)
    assert a.loc[D[2], "currency"] == "GBP" and a.loc[D[2], "country"] == "GBR"
    assert st["filled_rows"] == 1 and st["gaps"] == 1 and st["next_day_adjusted"] == 1


def test_traded_flag_comes_from_company_daily():
    out, _ = fill_holiday_gaps(*_case(), max_gap=3)
    a = out[out["sec_id"] == "A"].set_index("date")["traded"]
    assert list(a) == [True, True, False, True, False, True]  # d2 filled, d4 a carried close


def test_long_gaps_and_open_ended_gaps_stay_missing():
    out, _ = fill_holiday_gaps(*_case(), max_gap=3)
    assert sorted(out.loc[out["sec_id"] == "B", "date"]) == [D[0], D[5]]
    assert sorted(out.loc[out["sec_id"] == "C", "date"]) == list(D[:4])
    out4, _ = fill_holiday_gaps(*_case(), max_gap=4)  # B's four days are filled at max_gap 4
    assert len(out4[out4["sec_id"] == "B"]) == 6
    assert out4.loc[out4["sec_id"] == "B", "ret"].iloc[1:5].eq(0).all()  # EUR: no currency move
