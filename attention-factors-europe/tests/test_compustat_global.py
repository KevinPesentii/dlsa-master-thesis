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


@pytest.mark.parametrize("dsci, expected", [
    ("ORD CHF1(REGD)(2ND BUY BACK)", True), ("CHF0.1 (SEPARATE 2 TRADING L", True), ("ORD EUR2 (SEC LINE)", True),
    ("ORD NPV (ASD 06/07/18 EON CS", True), ("ORD NPV (TENDERED SHARES)", True), ("ORD CHF.1 2ND LINE", True),
    ("ORD CHF.10 (REGD) (2ND BUY B", True), ("ORD NPV(REDEMPTION SHARES)", True), ("ORD SER'A' NPV (RED SHS11/06", True), ("EUR0.75 (STK DIV 25/01/22)", True),
    ("PTG CERTS CHF.1 (POST SUBD)", False), ("ORD GBP0.005", False), ("CL C ORD NPV (POST 2ND CON)", False),
    ("PFD GBP1 RED(N) 3.15%", False), ("SER B ORD NPV", False),
])
def test_second_lines_of_a_class_are_not_equity(dsci, expected):
    assert bool(cg.non_equity(pd.Series([dsci]))[0]) is expected


def _cap(rows, dsci, countries=("BEL", "GBR"), fic="BEL", **kw):
    secd = pd.DataFrame(rows)
    company = pd.DataFrame(dict(gvkey=["g"], conm=["G"], prirow=["01W"], priusa=[None], fic=[fic],
                                loc=[fic], sic=["1"]))
    security = pd.DataFrame([dict(gvkey="g", iid=r["iid"], excntry=r["excntry"], dsci=dsci[r["iid"]]) for r in rows])
    fx = pd.DataFrame([dict(datadate=D, tocurd=c, exratd=1.0) for c in ("EUR", "GBP", "CHF")])
    return cg.company_month_mktcap(secd, company, security, fx, list(countries), "EUR", **kw).set_index("gvkey").loc["g"]


def test_chix_quote_neither_prices_nor_counts():
    """Rotork, 2013: the London line (86.8m shares) and its Chi-X quote (exchg 349, no ISIN)
    carrying 871.6m shares. Summed, the cap was eleven times the company's."""
    rows = [_row("g", "01W", 26.47, 86.8e6, "GBR", 194, 1.4e5, "GB01", curcdd="GBP"),
            _row("g", "03W", 26.47, 871.6e6, "GBR", 349, 2.9e4, None, curcdd="GBP")]
    out = _cap(rows, {"01W": "ORD GBP.005", "03W": "ORD GBP0.005 (CHI-X)"}, fic="GBR")
    assert out["iid"] == "01W" and out["n_classes"] == 1 and out["n_listings"] == 1
    assert out["mktcap"] == pytest.approx(26.47 * 86.8e6 / 1e6)


def test_buyback_line_is_not_a_class():
    """Adecco, 2016: the registered share and a second buyback line with a stale price on
    750m 'shares'."""
    rows = [_row("g", "04W", 62.5, 174.5e6, "BEL", 132, 1.3e6, "CH04", curcdd="CHF"),
            _row("g", "06W", 59.07, 750.1e6, "BEL", 132, None, "CH06", curcdd="CHF")]
    out = _cap(rows, {"04W": "ORD CHF.1 (REGD)", "06W": "ORD CHF1(REGD)(2ND BUY BACK)"})
    assert out["n_classes"] == 1 and out["mktcap"] == pytest.approx(62.5 * 174.5e6 / 1e6)


def test_near_identical_classes_are_one_class():
    """RELX, 2015-03: two London lines without a common ISIN whose share counts differ by
    86k on 1.13bn. Same class: counted once. A real second class (other count) still adds."""
    rows = [_row("g", "01W", 11.59, 1126693676.0, "GBR", 194, 5.1e6, "GB01", curcdd="GBP"),
            _row("g", "07W", 11.61, 1126779387.0, "GBR", 194, 9.3e5, None, curcdd="GBP"),
            _row("g", "02W", 11.60, 300e6, "GBR", 194, 1e5, "GB02", curcdd="GBP")]
    out = _cap(rows, {"01W": "ORD", "07W": "ORD", "02W": "CL B ORD"}, fic="GBR")
    assert out["n_classes"] == 2
    assert out["mktcap"] == pytest.approx((11.59 * 1126693676.0 + 11.60 * 300e6) / 1e6)


def test_swiss_genussschein_counts_and_prices_german_one_does_not():
    """Roche, 2020-01: 160m bearer shares and 702.6m Genussscheine (tpci Q) on SIX; the
    Genussschein trades far more and prices the company. A German Genussschein (debt) of the
    same company on Xetra never counts."""
    rows = [_row("g", "04W", 330.0, 160e6, "CHE", 151, 2.0e4, "CH04", curcdd="CHF"),
            _row("g", "03W", 324.3, 702.5627e6, "CHE", 151, 2.0e6, "CH03", curcdd="CHF", tpci="Q"),
            _row("g", "09W", 100.0, 5e6, "DEU", 154, 1e6, "DE09", tpci="Q")]
    dsci = {"04W": "ORD CHF1 (BRR)", "03W": "CHF0.001", "09W": "GENUSSCHEINE DEM100 7%"}
    with_gs = _cap(rows, dsci, ("CHE", "DEU"), fic="CHE", genussschein_countries=("CHE",))
    assert with_gs["iid"] == "03W" and with_gs["country"] == "CHE" and with_gs["n_classes"] == 2
    assert with_gs["mktcap"] == pytest.approx((330.0 * 160e6 + 324.3 * 702.5627e6) / 1e6)
    without = _cap(rows, dsci, ("CHE", "DEU"), fic="CHE")
    assert without["iid"] == "04W" and without["mktcap"] == pytest.approx(330.0 * 160e6 / 1e6)


def test_participation_certificate_is_a_class():
    """Schindler: registered shares and participation certificates (tpci 8) add up."""
    rows = [_row("g", "01W", 250.0, 47e6, "BEL", 132, 1e4, "CH01", curcdd="CHF"),
            _row("g", "03W", 255.0, 60e6, "BEL", 132, 1e5, "CH03", curcdd="CHF", tpci="8")]
    out = _cap(rows, {"01W": "ORD CHF.10 (REGD)", "03W": "PTG CERTS CHF.1 (POST SUBD)"}, issue_types=("0", "8"))
    assert out["iid"] == "03W" and out["n_classes"] == 2
    assert out["mktcap"] == pytest.approx((250.0 * 47e6 + 255.0 * 60e6) / 1e6)


def test_share_correction_divides_inside_its_dates_only():
    """Eurocommercial: price per depositary receipt (10 shares), cshoc in shares until the
    2005-04-27 switch to receipts."""
    d = pd.DataFrame(dict(gvkey=["208226"] * 3 + ["1"], iid=["01W"] * 4, cshoc=[313.6e6, 34.46e6, 35.2e6, 1e6],
                          datadate=pd.to_datetime(["2005-04-26", "2005-04-27", "2006-01-02", "2005-01-03"])))
    fix = [dict(gvkey="208226", iid="01W", start="1984-01-01", end="2005-04-26", divide_by=10)]
    out = cg.apply_share_corrections(d, fix)
    assert out["cshoc"].tolist() == pytest.approx([31.36e6, 34.46e6, 35.2e6, 1e6])
    assert d["cshoc"].iloc[0] == 313.6e6                     # input untouched
    rows = [dict(_row("208226", "01W", 20.85, 302.3e6, "NLD", 104, 1e5, "NL01"), datadate=pd.Timestamp("2003-12-31"))]
    secd = pd.DataFrame(rows)
    company = pd.DataFrame(dict(gvkey=["208226"], conm=["E"], prirow=["01W"], priusa=[None], fic=["NLD"],
                                loc=["NLD"], sic=["1"]))
    security = pd.DataFrame(dict(gvkey=["208226"], iid=["01W"], excntry=["NLD"]))
    fx = pd.DataFrame([dict(datadate=pd.Timestamp("2003-12-31"), tocurd="EUR", exratd=1.0)])
    cap = cg.company_month_mktcap(secd, company, security, fx, ["NLD"], "EUR", share_corrections=fix)
    assert cap["mktcap"].iloc[0] == pytest.approx(20.85 * 30.23e6 / 1e6)


def test_dual_listed_halves_add_up_and_excluded_line_does_not_count():
    """Unilever, 1998-07: NV's CVA (Amsterdam, traded) and its ordinary line (stale price,
    the same shares) and PLC in London, all under one gvkey. NV + PLC count once each."""
    d = pd.Timestamp("1998-07-31")
    rows = [dict(_row("u", "02W", 143.5, 640.164e6, "NLD", 104, 4.0e6, "NL02"), datadate=d),
            dict(_row("u", "01W", 147.3, 640.165e6 * 1.05, "NLD", 104, 7.9e3, "NL01"), datadate=d),
            dict(_row("u", "16W", 6.02, 3261e6, "GBR", 194, 1.1e7, "GB16", curcdd="GBP"), datadate=d)]
    secd = pd.DataFrame(rows)
    company = pd.DataFrame(dict(gvkey=["u"], conm=["U"], prirow=["16W"], priusa=[None], fic=["GBR"], loc=["GBR"], sic=["1"]))
    security = pd.DataFrame(dict(gvkey=["u"] * 3, iid=["02W", "01W", "16W"], excntry=["NLD", "NLD", "GBR"]))
    fx = pd.DataFrame([dict(datadate=d, tocurd=c, exratd=r) for c, r in (("GBP", 1.0), ("EUR", 1.5))])
    excl = [dict(gvkey="u", iid="01W", start="1984-01-01", end="2019-06-30")]
    dual = [dict(gvkey="u", countries=["NLD", "GBR"], start="1984-01-01", end="2020-10-31")]
    out = cg.company_month_mktcap(secd, company, security, fx, ["NLD", "GBR"], "EUR",
                                  exclude_lines=excl, dual_listed=dual).set_index("gvkey").loc["u"]
    nv, plc = 143.5 * 640.164e6, 6.02 * 3261e6 * 1.5                  # NV quoted in EUR here
    assert out["n_classes"] == 2 and out["mktcap"] == pytest.approx((nv + plc) / 1e6)
    one_country = cg.company_month_mktcap(secd, company, security, fx, ["NLD", "GBR"], "EUR",
                                          exclude_lines=excl).set_index("gvkey").loc["u"]
    assert one_country["n_classes"] == 1
