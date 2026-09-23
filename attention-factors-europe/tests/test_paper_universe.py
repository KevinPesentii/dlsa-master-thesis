"""The paper's universe: Figures 3-4 of Epstein et al. (2025) name depositary receipts
(Mizuho, NatWest, Ecopetrol), Canadian listings and Alphabet twice. Each rule of
us_panel is pinned by a hand-built case with real magnitudes.

1. A receipt is sized at company level: underlying shares of its class over the receipt
   ratio, times the receipt's close. Mizuho 2020: 25.3925bn shares at ratio 2.0 before the
   October consolidation, 2.53925bn at 0.2 after; either way 12.69625bn receipt
   equivalents. CRSP's cap counts receipts outstanding only: $0.18bn.
2. Before comp.secm has shares (1998-04): funda csho, in receipt equivalents, of a fiscal
   year that ended at least six months earlier, carried through a later 2:1 split with
   CRSP's cumulative price factor.
3. Never below CRSP's cap (Baidu's secm counts class A only).
4. Ranking lines gives Alphabet's two classes two slots; ranking companies gives one.
5. Canadian filers' amounts are converted to USD at the fiscal year end; shares are not.
6. (Found while checking prefix invariance of this build.) When a fiscal-year-end change
   makes two records usable from the same month, the later one is used, in any row order.
"""

import numpy as np
import pandas as pd
import pytest

from afe.data import us_panel as up
from afe.data import wrds_us

P = lambda s: pd.Period(s, "M")  # noqa: E731
NO_FUNDA = pd.DataFrame({"gvkey": pd.Series(dtype=str), "datadate": pd.Series(dtype="datetime64[ns]"),
                         "csho": pd.Series(dtype=float)})


def _elig(rows):
    df = pd.DataFrame(rows)
    df["sharetype"] = df.get("sharetype", "AD")
    return df


def _link(permno, gvkey, liid, start="1980-01-01"):
    return pd.DataFrame([dict(gvkey=gvkey, liid=liid, linkprim="P", permno=permno, permco=permno,
                              linkdt=pd.Timestamp(start), linkenddt=pd.NaT)])


def test_receipt_takes_underlying_shares_over_ratio_across_a_ratio_change():
    elig = _elig([dict(permno=91582, permco=55555, month=P("2020-09"), prc=2.50, cap=180547.65, cumfacpr=1.0),
                  dict(permno=91582, permco=55555, month=P("2020-11"), prc=2.55, cap=180547.65, cumfacpr=1.0)])
    secm = pd.DataFrame([dict(gvkey="248136", iid="90", datadate=pd.Timestamp("2020-09-30"), cshom=25_392_500_000.0, adrrm=2.0, prccm=2.50),
                         dict(gvkey="248136", iid="90", datadate=pd.Timestamp("2020-11-30"), cshom=2_539_250_000.0, adrrm=0.2, prccm=2.55)])
    out = up.receipt_caps(elig, secm, _link(91582, "248136", "90"), NO_FUNDA)
    assert out["cap"].tolist() == pytest.approx([2.50 * 12_696_250_000 / 1000, 2.55 * 12_696_250_000 / 1000])
    assert out["cap"].iloc[1] / 1e6 == pytest.approx(32.375, abs=1e-3)          # USD bn
    assert out["cap_source"].tolist() == ["secm", "secm"]
    assert out["cap_crsp"].tolist() == [180547.65, 180547.65]


def test_fallback_uses_a_fiscal_year_six_months_old_and_carries_it_through_a_split():
    # 2:1 split in 1996-07: CRSP's cumulative price factor is 2 before, 1 after
    elig = _elig([dict(permno=70000, permco=1, month=P(m), prc=p, cap=1000.0, cumfacpr=f)
                  for m, p, f in [("1994-12", 18.0, 2.0), ("1995-12", 19.0, 2.0), ("1996-03", 20.0, 2.0), ("1996-09", 10.0, 1.0)]])
    funda = pd.DataFrame([dict(gvkey="000001", datadate=pd.Timestamp("1994-12-31"), csho=50.0),
                          dict(gvkey="000001", datadate=pd.Timestamp("1995-12-31"), csho=100.0)])
    out = up.receipt_caps(elig, None, _link(70000, "000001", "90"), funda).set_index("month")
    # 1996-03: FY1995 is not six months old yet, so FY1994's 50m receipts, no split in between
    assert out.loc[P("1996-03"), "cap"] == pytest.approx(20.0 * 50e6 / 1000)
    # 1996-09: FY1995's 100m receipts became 200m in the split
    assert out.loc[P("1996-09"), "cap"] == pytest.approx(10.0 * 200e6 / 1000)
    assert out.loc[P("1996-09"), "cap_source"] == "funda"
    assert out.loc[P("1994-12"), "cap_source"] == "crsp"                          # nothing usable yet


def test_fallback_counted_in_ordinary_shares_is_dropped_by_a_point_in_time_rule():
    """Centaur Mining, 2000-09: funda csho 440m are ordinary shares, not receipts; at the
    receipt's $30 close that is a $13.2bn 'company' against CRSP's $0.43m, 30,000x. Only
    data of the time may judge it: above 20,000x the fallback is dropped, CRSP's cap kept."""
    elig = _elig([dict(permno=87519, permco=3, month=P("1999-06"), prc=3.0, cap=500.0, cumfacpr=1.0),
                  dict(permno=87519, permco=3, month=P("2000-09"), prc=30.0, cap=434.0, cumfacpr=1.0)])
    funda = pd.DataFrame([dict(gvkey="220235", datadate=pd.Timestamp("1999-06-30"), csho=440.1)])
    out = up.receipt_caps(elig, None, _link(87519, "220235", "90"), funda).set_index("month")
    assert out.loc[P("2000-09"), "cap"] == 434.0 and out.loc[P("2000-09"), "cap_source"] == "crsp"
    # the same count at a ratio a real company can have (Ecopetrol-like 1,000x) is kept
    kept = up.receipt_caps(elig.assign(cap=elig["cap"] * 50), None, _link(87519, "220235", "90"), funda)
    assert kept.set_index("month").loc[P("2000-09"), "cap_source"] == "funda"


def test_receipt_cap_never_falls_below_crsp():
    elig = _elig([dict(permno=90857, permco=2, month=P("2020-12"), prc=216.24, cap=58_680_400.08, cumfacpr=1.0)])
    secm = pd.DataFrame([dict(gvkey="164532", iid="90", datadate=pd.Timestamp("2020-12-31"), cshom=26_956_000.0, adrrm=0.1, prccm=216.24)])
    out = up.receipt_caps(elig, secm, _link(90857, "164532", "90"), NO_FUNDA)
    assert out["cap"].iloc[0] == 58_680_400.08 and out["cap_source"].iloc[0] == "crsp"


def test_line_ranking_gives_two_classes_two_slots():
    elig = _elig([dict(permno=1, permco=10, month=P("2020-12"), cap=5.0e8, sharetype="NS"),     # GOOG
                  dict(permno=2, permco=10, month=P("2020-12"), cap=4.9e8, sharetype="NS"),     # GOOGL
                  dict(permno=3, permco=20, month=P("2020-12"), cap=6.0e8, sharetype="NS"),
                  dict(permno=4, permco=30, month=P("2020-12"), cap=1.0e5, sharetype="NS")])
    cfg = {"sample": {"start": "2021-01", "end": "2021-01", "universe_size": 3}}
    cal = pd.bdate_range("2021-01-04", "2021-01-29")
    co = up.company_month(elig)
    by_line, _ = up.build_universe(up.line_month(elig, co), cfg, cal)
    by_company, _ = up.build_universe(co, cfg, cal)
    assert set(by_line["sec_id"]) == {"1", "2", "3"}
    assert set(by_company["sec_id"]) == {"1", "3", "4"}


def test_fiscal_year_end_change_takes_the_later_record_whatever_the_row_order():
    """AMD moved its year end from March to December in 1987: two records dated 1987, both
    usable from June 1988 under the june rule. The later one must win, in any row order."""
    rec = pd.DataFrame([dict(gvkey="001161", datadate=pd.Timestamp("1987-03-31"), C=28.903 / 843.543),
                        dict(gvkey="001161", datadate=pd.Timestamp("1987-12-31"), C=235.155 / 1113.716)])
    rec["avail_month"] = pd.Period("1988-06", "M")
    mapping = pd.DataFrame(dict(permno=61241, month=[P("1988-06"), P("1988-12")], gvkey="001161"))
    for order in (rec, rec.iloc[::-1]):
        out = up.align_annual(order, mapping, ["C"], max_age_months=30)
        assert out["C"].tolist() == pytest.approx([235.155 / 1113.716] * 2)


def test_cad_amounts_are_converted_at_the_fiscal_year_end_and_shares_are_not():
    base = {c: np.nan for c in wrds_us.FUNDA_ITEMS}
    funda = pd.DataFrame([{**base, "gvkey": "015633", "datadate": pd.Timestamp("2017-10-31"), "at": 1000.0, "sale": 50.0, "csho": 7.0},
                          {**base, "gvkey": "000002", "datadate": pd.Timestamp("2017-10-31"), "at": 1000.0, "csho": 7.0}])
    currency = pd.DataFrame([dict(gvkey="015633", datadate=pd.Timestamp("2017-10-31"), curcd="CAD"),
                             dict(gvkey="000002", datadate=pd.Timestamp("2017-10-31"), curcd="USD")])
    fx = pd.DataFrame([dict(datadate=pd.Timestamp("2017-10-30"), tocurd="CAD", exratd=1.70),   # units per GBP, a day stale
                       dict(datadate=pd.Timestamp("2017-10-31"), tocurd="USD", exratd=1.36)])
    out = up.funda_to_usd(funda, currency, fx).set_index("gvkey")
    assert out.loc["015633", ["at", "sale"]].tolist() == pytest.approx([800.0, 40.0])    # x 1.36 / 1.70
    assert out.loc["015633", "csho"] == 7.0
    assert out.loc["000002", "at"] == 1000.0 and out.loc["000002", "fx_usd"] == 1.0
