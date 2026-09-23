"""Two defects of the first US build, each pinned by a hand-built case.

1. CRSP's delisting-day row (dlydelflg = Y) carries the delisting return but fails the
   common-share filter of the daily pull; stage 2 must merge it back from
   crsp_delisting.parquet (Lehman: -60% on 2008-09-18 after -56.7% on 09-17).
2. pandas ranks the garbage under the NA mask of a nullable Float64 column; the
   characteristics must be plain float64 before the cross-sectional rank.
"""

import numpy as np
import pandas as pd
import pytest

from afe.data import characteristics as ch
from afe.data import us_panel as up


def _daily_year(tmp_path, year, rows):
    d = tmp_path / "daily"
    d.mkdir(exist_ok=True)
    base = {c: np.nan for c in up.DAILY_COLS}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(d / f"{year}.parquet", index=False)
    return d


def test_delisting_row_is_merged_once_and_wins_nothing_over_a_traded_row(tmp_path):
    days = pd.bdate_range("2008-09-12", "2008-09-17")
    rows = [dict(permno=80599, permco=20000, date=t, ret=-0.1, prc=1.0, vol=1e6) for t in days]
    rows.append(dict(permno=11111, permco=20001, date=days[0], ret=0.01, prc=10.0, vol=1e5))
    daily_dir = _daily_year(tmp_path, 2008, rows)
    delisting = pd.DataFrame([
        dict(permno=80599, permco=20000, date=pd.Timestamp("2008-09-18"), ret=-0.6, prc=0.052),
        dict(permno=80599, permco=20000, date=days[-1], ret=-0.99, prc=0.0),   # duplicate of a traded day
        dict(permno=22222, permco=20002, date=pd.Timestamp("2008-03-03"), ret=-0.3),  # not in the pool
    ])
    for c in up.DAILY_COLS:
        delisting[c] = delisting.get(c, np.nan)
    ff = pd.DataFrame({"mktrf": 0.0, "smb": 0.0, "hml": 0.0, "rf": 0.0},
                      index=pd.bdate_range("2008-01-01", "2008-12-31"))
    daily, stats = up.load_daily(daily_dir, [80599, 11111], [2008], ff, delisting=delisting)
    leh = daily[daily["permno"] == 80599].set_index("date")["ret"]
    assert leh.loc["2008-09-18"] == pytest.approx(-0.6)          # merged
    assert leh.loc[days[-1]] == pytest.approx(-0.1)              # the daily-file row wins
    assert not daily.duplicated(["permno", "date"]).any()
    assert 22222 not in set(daily["permno"])                     # pool filter applies
    assert stats.loc[(pd.Period("2008-09", "M"), 80599), "n_ret"] == len(days) + 1
    # a cutoff before the delisting date drops the merged row like any other
    trunc, _ = up.load_daily(daily_dir, [80599], [2008], ff, cutoff=pd.Timestamp("2008-09-17"), delisting=delisting)
    assert trunc["date"].max() == days[-1]


def test_nullable_float_ranks_garbage_under_the_mask_unless_cast():
    """The masked array's rank sees the raw buffer, not the mask: construct NA cells with
    a huge number underneath and check that only the float64 cast ranks them as NaN."""
    arr = pd.array([1.0, 2.0, 3.0, 4.0], dtype="Float64")
    arr._data[1] = 1e9            # garbage under the NA
    arr._mask[1] = True
    raw = pd.DataFrame({"x": arr}, index=pd.MultiIndex.from_product([[pd.Timestamp("2010-01-04")], list("abcd")],
                                                                     names=["date", "permno"]))
    key = raw.index.get_level_values("date")
    buggy = raw.groupby(key).transform(ch.rank_quantile)["x"]
    fixed = raw.astype("float64").groupby(key).transform(ch.rank_quantile)["x"]
    assert pd.isna(fixed.iloc[1]) and list(fixed.dropna().round(4)) == [-0.1667, 0.1667, 0.5]
    # the buggy path either ranks the NA cell or shifts the others; it must not equal the fixed one
    assert not buggy.astype("float64").equals(fixed) or not pd.isna(buggy.iloc[1])
