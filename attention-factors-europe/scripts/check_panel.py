"""Check what the three parquet files actually contain, before trusting a run on them.

Runs locally on the data (it never leaves the machine) and answers three questions:

1. TIMING. Is char_Ret_D1 on row t the return of day t (the convention afe.data.slots
   assumes, and then lags by one day), or is it already lagged? Measured as the average
   cross-sectional rank correlation with the same-day and the previous-day return.
   Same-day ~1 is correct. Previous-day ~1 would mean the loader lags twice.

2. MISSING FILL. After rank normalisation the build fills missing values with the
   cross-sectional median, which is one single value. Its share per characteristic and
   year should match 1 - coverage in build_report.txt. A share near zero would mean the
   missing values were ranked as if they were data: the Float64 ranking bug fixed in
   commit 4dad4ff, i.e. files built before the fix.

3. LAST OBSERVED. The paper fills with the last observed value when there is one.
   Counts rows where a name had a real value the day before and gets the fill value
   today: under the paper's rule those rows should carry yesterday's value instead.

    python3 scripts/check_panel.py --data-dir data/us
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOW_COVERAGE = ["CF", "Spread", "FC2Y", "OP", "OA", "OL"]
YEARS = [1991, 2005, 2020]


def cs_rank_corr(df: pd.DataFrame, a: str, b: str, n_dates: int, seed: int = 0) -> float:
    d = df.dropna(subset=[a, b])
    dates = d["date"].drop_duplicates()
    pick = dates.sample(min(n_dates, len(dates)), random_state=seed)
    d = d[d["date"].isin(pick)]
    ra = d.groupby("date")[a].rank()
    rb = d.groupby("date")[b].rank()
    tmp = pd.DataFrame({"date": d["date"], "ra": ra, "rb": rb})
    return float(tmp.groupby("date").apply(lambda g: g["ra"].corr(g["rb"])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/us")
    ap.add_argument("--dates", type=int, default=300, help="dates sampled for the timing test")
    args = ap.parse_args()
    root = Path(args.data_dir)

    names = pq.read_schema(root / "features.parquet").names
    chars = [c for c in names if c.startswith("char_")]
    print(f"features.parquet: {len(chars)} char_ columns, "
          f"{sum(c.startswith('med_') for c in names)} med_ columns, rf: {'rf' in names}")

    # ---------------------------------------------------------------- 1. timing
    print("\n1. TIMING of char_Ret_D1 and char_Ret_W1")
    f = pd.read_parquet(root / "features.parquet",
                        columns=["date", "sec_id", "char_Ret_D1", "char_Ret_W1"])
    r = pd.read_parquet(root / "returns.parquet", columns=["date", "sec_id", "ret"])
    r = r.sort_values(["sec_id", "date"])
    g = r.groupby("sec_id")["ret"]
    r["ret_prev"] = g.shift(1)
    r["w1_now"] = g.transform(lambda s: s.rolling(5).sum())
    r["w1_prev"] = r.groupby("sec_id")["w1_now"].shift(1)
    m = f.merge(r, on=["date", "sec_id"], how="inner")
    for ch, now, prev in [("char_Ret_D1", "ret", "ret_prev"), ("char_Ret_W1", "w1_now", "w1_prev")]:
        c_now = cs_rank_corr(m, ch, now, args.dates)
        c_prev = cs_rank_corr(m, ch, prev, args.dates)
        verdict = ("as of day t: CORRECT for the loader" if c_now > 0.9 > c_prev else
                   "ALREADY LAGGED: the loader would lag twice" if c_prev > 0.9 > c_now else
                   "UNCLEAR: look at it")
        print(f"   {ch:12s}  corr with day t {c_now:+.3f} | with day t-1 {c_prev:+.3f}  ->  {verdict}")
    del f, r, m

    # ------------------------------------------------------------ 2. missing fill
    print("\n2. MISSING FILL: share of the single most frequent value (the median fill)")
    cols = ["date", "sec_id"] + [f"char_{c}" for c in LOW_COVERAGE if f"char_{c}" in chars]
    f = pd.read_parquet(root / "features.parquet", columns=cols)
    f["year"] = f["date"].dt.year
    rows = []
    for c in cols[2:]:
        for y in YEARS:
            v = f.loc[f["year"] == y, c].dropna()
            if v.empty:
                continue
            counts = v.value_counts()
            rows.append((c.removeprefix("char_"), y, counts.index[0], counts.iloc[0] / len(v)))
    out = pd.DataFrame(rows, columns=["char", "year", "fill_value", "share"])
    print(out.pivot(index="char", columns="year", values="share").map(lambda x: f"{100*x:5.1f}%"))
    print("   fill values found:", sorted(set(np.round(out["fill_value"], 6))))

    # ---------------------------------------------------------- 3. last observed
    print("\n3. LAST OBSERVED: rows that switch from a real value to the fill value")
    f = f.sort_values(["sec_id", "date"])
    same = f["sec_id"].values[1:] == f["sec_id"].values[:-1]
    for c in cols[2:]:
        fill = out.loc[out["char"] == c.removeprefix("char_"), "fill_value"].mode().iloc[0]
        x = f[c].values
        is_fill = np.isclose(x, fill)
        switch = same & ~is_fill[:-1] & is_fill[1:]
        n_fill = int(is_fill.sum())
        print(f"   {c.removeprefix('char_'):7s}  fill rows {n_fill:>9,}  of which real->fill switches "
              f"{int(switch.sum()):>7,}  ({100*switch.sum()/max(n_fill,1):4.1f}%)")


if __name__ == "__main__":
    main()
