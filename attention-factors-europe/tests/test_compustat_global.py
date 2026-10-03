"""Company cap from Compustat Global: cross-listings must not double count, dormant
secondary listings must not enter, and share classes must add up. Hand-built rows modelled
on what comp.g_secd showed for 2005-12-30 (Fiat, Shell, Unilever NV, Accenture, CRH)."""

import pandas as pd
import pytest

from afe.data import compustat_global as cg

D = pd.Timestamp("2005-12-30")
COUNTRIES = ["AUT", "DEU", "FRA", "IRL", "ITA", "NLD"]


def _row(gvkey, iid, prccd, cshoc, excntry, exchg, cshtrd, isin=None, curcdd="EUR", tpci="0", fic=None):
    """fic is the g_secd header copy of the incorporation country; domestic by default."""
    return dict(gvkey=gvkey, iid=iid, datadate=D, prccd=prccd, cshoc=cshoc, ajexdi=1.0,
                curcdd=curcdd, prcstd=10, qunit=1.0, tpci=tpci, exchg=exchg, isin=isin,
                cshtrd=cshtrd, conm="x", fic=fic or excntry, loc="x", excntry=excntry)


@pytest.fixture
def fixture():
    rows = [
        # one class listed in Milan (primary), Paris and Xetra: same ISIN and cshoc
        _row("1", "01W", 7.36, 1000.0, "ITA", 209, 4e6, "IT1"),
        _row("1", "04W", 7.44, 1000.0, "FRA", 286, 6e3, "IT1"),
        _row("1", "05W", 7.34, 1000.0, "DEU", 154, 5e3, "IT1"),
        # two classes in Amsterdam, primary issue in London: cap adds up, not eligible
        _row("2", "03W", 27.08, 2759.0, "NLD", 104, 7e3),
        _row("2", "04W", 25.78, 3935.0, "NLD", 104, 4e6),
        # two classes at home plus a Xetra line of the first class under another ISIN
        _row("3", "01W", 10.0, 500.0, "NLD", 104, 1e5, "NL1"),
        _row("3", "02W", 20.0, 100.0, "NLD", 104, 1e4, "NL2"),
        _row("3", "09W", 10.1, 500.0, "DEU", 154, 1e3, "NL3"),
        # Irish-incorporated, NYSE-listed company whose only line in the set is a dormant
        # Frankfurt one (100 shares a day on 1bn): header link, but not active
        _row("4", "02W", 50.0, 1e9, "DEU", 171, 1e2, "US1", fic="IRL"),
        # Irish company quoted in USD, no prirow on file: eligible through fic
        _row("5", "01W", 17.167, 100.0, "IRL", 172, 1e2, "IE1", curcdd="USD"),
        # Irish company whose primary issue moved to London: active Dublin line keeps it
        _row("6", "01W", 20.0, 1000.0, "IRL", 172, 2e3, "IE6"),
        # Swiss company with a dormant Frankfurt line only and no header link
        _row("7", "05W", 80.0, 1e9, "DEU", 171, 5e2, "CH1", fic="CHE"),
        # preferred share of company 3: pulled, not counted
        _row("3", "03W", 99.0, 100.0, "NLD", 104, 1e4, "NL9", tpci="1"),
    ]
    secd = pd.DataFrame(rows)
    company = pd.DataFrame(dict(
        gvkey=list("1234567"), conm=list("ABCDEFG"),
        prirow=["01W", "01W", "01W", "02W", None, "02W", "01W"],
        priusa=[None, None, None, "01", None, "01", None],
        fic=["ITA", "GBR", "NLD", "IRL", "IRL", "IRL", "CHE"], loc=["ITA", "GBR", "NLD", "IRL", "IRL", "IRL", "CHE"],
        sic=["1"] * 7,
    ))
    security = pd.DataFrame([dict(gvkey=r["gvkey"], iid=r["iid"], excntry=r["excntry"]) for r in rows]
                            + [dict(gvkey="2", iid="01W", excntry="GBR"), dict(gvkey="6", iid="02W", excntry="GBR"),
                               dict(gvkey="7", iid="01W", excntry="CHE")])
    fx = pd.DataFrame([
        dict(datadate=D - pd.Timedelta(days=1), tocurd="EUR", exratd=1.4551),  # EUR per GBP, day before
        dict(datadate=D - pd.Timedelta(days=1), tocurd="USD", exratd=1.7167),
        dict(datadate=D, tocurd="GBP", exratd=1.0),
    ])
    return secd, company, security, fx


def test_company_month_mktcap(fixture):
    out = cg.company_month_mktcap(*fixture, COUNTRIES, "EUR").set_index("gvkey")
    expected_mn = {
        "1": 7.36 * 1000 / 1e6,                              # primary listing only, once
        "2": (27.08 * 2759 + 25.78 * 3935) / 1e6,            # A + B
        "3": (10.0 * 500 + 20.0 * 100) / 1e6,                # two classes, Xetra line collapsed, pref excluded
        "4": 50.0 * 1e9 / 1e6,
        "5": 17.167 * 100 * 1.4551 / 1.7167 / 1e6,          # USD -> EUR through GBP
        "6": 20.0 * 1000 / 1e6,
    }
    for g, e in expected_mn.items():
        assert out.loc[g, "mktcap"] == pytest.approx(e)
    assert out["eligible"].to_dict() == {"1": True, "2": False, "3": True, "4": False, "5": True,
                                         "6": True, "7": False}
    assert out["header_link"].to_dict() == {"1": True, "2": False, "3": True, "4": True, "5": True,
                                            "6": True, "7": False}
    assert out["active"].to_dict() == {"1": True, "2": True, "3": True, "4": False, "5": True,
                                       "6": True, "7": False}
    assert out["country"].to_dict() == {"1": "ITA", "2": "NLD", "3": "NLD", "4": "DEU", "5": "IRL",
                                        "6": "IRL", "7": "DEU"}
    assert out.loc["1", "n_listings"] == 3 and out.loc["1", "n_classes"] == 1
    assert out.loc["3", "n_classes"] == 2


def test_fx_stale_rate_is_rejected(fixture):
    secd, company, security, fx = fixture
    old = fx.assign(datadate=fx["datadate"] - pd.Timedelta(days=30))
    out = cg.company_month_mktcap(secd, company, security, old, COUNTRIES, "EUR")
    assert "5" not in set(out["gvkey"])          # USD row has no usable rate
    assert set(out["gvkey"]) == {"1", "2", "3", "4", "6", "7"}  # EUR rows need no rate


def test_cutoff_table_ranks_within_group(fixture):
    out = cg.company_month_mktcap(*fixture, COUNTRIES, "EUR")
    tab = cg.cutoff_table(out[out["eligible"]], [1, 2], ["country"])
    assert tab.loc["NLD", "n_companies"] == 1 and tab.loc["IRL", "n_companies"] == 2
    assert tab.loc["ITA", "rank_1"] == pytest.approx(7.36 * 1000 / 1e6)
    assert pd.isna(tab.loc["ITA", "rank_2"])


def test_turnover_is_a_trailing_median_that_skips_missing_volume(fixture):
    """A liquid line with one missing volume day stays active; a line that went dormant
    for the whole window does not; a line with no volume on an exchange that reports
    volume for its other listings is dormant; no volume on an exchange that reports none
    (Luxembourg) is indeterminate and kept."""
    secd, company, security, fx = fixture
    months = pd.date_range("2005-07-31", "2005-12-31", freq="ME")
    rows = []
    for i, d in enumerate(months):
        rows.append(dict(_row("1", "01W", 7.0, 1000.0, "ITA", 209, 4e6 if i != 2 else None, "IT1"), datadate=d))
        rows.append(dict(_row("8", "01W", 5.0, 1000.0, "ITA", 209, 4e6 if i < 1 else 1e-3, "IT8"), datadate=d))
        rows.append(dict(_row("9", "01W", 5.0, 1000.0, "LUX", 198, None, "LU9"), datadate=d))
        rows.append(dict(_row("10", "03W", 5.0, 1e9, "ITA", 209, None, "IE10", fic="IRL"), datadate=d))
    secd = pd.DataFrame(rows)
    company = pd.concat([company, pd.DataFrame(dict(gvkey=["8", "9", "10"], conm=["H", "I", "J"],
                                                    prirow=["01W", "01W", "03W"], priusa=[None, None, "01"],
                                                    fic=["ITA", "LUX", "IRL"], loc=["ITA", "LUX", "IRL"],
                                                    sic=["1", "1", "1"]))])
    security = pd.concat([security, pd.DataFrame(dict(gvkey=["8", "9", "10"], iid=["01W", "01W", "03W"],
                                                      excntry=["ITA", "LUX", "ITA"]))])
    fx = pd.DataFrame(dict(datadate=months, tocurd="GBP", exratd=1.0))
    out = cg.company_month_mktcap(secd, company, security, fx, ["ITA", "LUX", "IRL"], "EUR")
    last = out[out["datadate"] == months[-1]].set_index("gvkey")
    assert last.loc["1", "active"] and last.loc["1", "turnover"] == pytest.approx(4e6 / 1000)
    assert not last.loc["8", "active"]
    assert last.loc["9", "active"] and pd.isna(last.loc["9", "turnover"])
    assert last.loc["10", "header_link"] and not last.loc["10", "active"] and last.loc["10", "turnover"] == 0


@pytest.mark.parametrize("dsci, expected", [
    ("ORD NPV VVPR STRIP", True), ("ORD NPV (VVPR STRIP)", True), ("ORD EUR2.29 (SUB RIGHT)", True),
    ("EUR0.2 BN RTS 17/11/21", True), ("ORD EUR1(STOCK DIV 5/7/2012)", True), ("ORD NPV (NIL PAID 05/08/10)", True),
    ("ORD NPV", False), ("CL B ORD GBP1.02", False), ("ORD EUR2 (RFD 1/1/11)", False),
    ("DFD NPV 09/16/08 (EX-RIGHTS)", False), ("CLS A ORD NPV CUM RTS 1/2 WT", False), ("ABB U NO DIVIDEND RIGHT", False),
])
def test_non_equity_descriptions(dsci, expected):
    assert bool(cg.non_equity(pd.Series([dsci]))[0]) is expected


def _one_company(rows, dsci):
    """Brussels / London company `g` with the given lines; dsci per iid in g_security."""
    secd = pd.DataFrame(rows)
    company = pd.DataFrame(dict(gvkey=["g"], conm=["G"], prirow=["01W"], priusa=[None], fic=["BEL"],
                                loc=["BEL"], sic=["1"]))
    security = pd.DataFrame([dict(gvkey="g", iid=r["iid"], excntry=r["excntry"], dsci=dsci[r["iid"]]) for r in rows])
    fx = pd.DataFrame([dict(datadate=D, tocurd="EUR", exratd=1.0), dict(datadate=D, tocurd="GBP", exratd=1.0)])
    return cg.company_month_mktcap(secd, company, security, fx, ["BEL", "GBR"], "EUR").set_index("gvkey").loc["g"]


def test_vvpr_strip_neither_prices_nor_counts():
    """Electrabel, 2006: the ordinary line (EUR 390, Suez held ~98%) trades less, per share,
    than its VVPR strip (EUR 0.01). The strip must not price the company or add to its cap."""
    rows = [_row("g", "04W", 390.0, 54.9e6, "BEL", 132, 1.1e3, "BE04"),
            _row("g", "11W", 0.01, 10.2e6, "BEL", 132, 5e4, "BE11")]
    out = _one_company(rows, {"04W": "ORD NPV", "11W": "ORD NPV VVPR STRIP"})
    assert out["iid"] == "04W" and out["n_classes"] == 1
    assert out["mktcap"] == pytest.approx(390.0 * 54.9e6 / 1e6)
    assert out["turnover"] == pytest.approx(1.1e3 / 54.9e6)     # the share's own, below 3e-5
    assert not out["active"]


def test_subscription_right_neither_prices_nor_counts():
    """AXA, June 2006: the rights line out-trades the share for the month."""
    rows = [_row("g", "01W", 20.0, 2e9, "BEL", 132, 1e7, "FR01"),
            _row("g", "03W", 0.66, 2e9, "BEL", 132, 5e7, "FR03")]
    out = _one_company(rows, {"01W": "ORD EUR2.29", "03W": "ORD EUR2.29 (SUB RIGHT)"})
    assert out["iid"] == "01W" and out["mktcap"] == pytest.approx(20.0 * 2e9 / 1e6)


def test_class_below_min_share_does_not_price_the_company():
    """Land Securities, 2002-03: the B shares of a return of capital (GBP 1.01, under 1% of
    the cap) turn over ten times the ordinary line per share. They still count in the cap
    (a class), but the ordinary class prices the company."""
    rows = [_row("g", "01W", 7.85, 465e6, "GBR", 194, 2e6, "GB01", curcdd="GBP"),
            _row("g", "02W", 1.01, 29.7e6, "GBR", 194, 1.4e6, "GB02", curcdd="GBP")]
    out = _one_company(rows, {"01W": "ORD GBP0.10", "02W": "CL B ORD GBP1.02"})
    assert out["iid"] == "01W" and out["n_classes"] == 2
    assert out["mktcap"] == pytest.approx((7.85 * 465e6 + 1.01 * 29.7e6) / 1e6)


def test_material_second_class_that_trades_more_still_prices():
    """A Swedish A/B pair: the B class holds 30% of the cap and out-trades A; B prices."""
    rows = [_row("g", "01W", 100.0, 7e6, "BEL", 132, 1e3, "SE01"),
            _row("g", "02W", 100.0, 3e6, "BEL", 132, 3e4, "SE02")]
    out = _one_company(rows, {"01W": "CL A ORD", "02W": "CL B ORD"})
    assert out["iid"] == "02W" and out["mktcap"] == pytest.approx(1000.0)
