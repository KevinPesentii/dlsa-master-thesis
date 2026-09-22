"""Euro 1-month risk-free series from the Bundesbank money-market files.

    python scripts/build_eu_rf.py

Reads data/eu/private/raw/external/ (downloaded from api.statistiken.bundesbank.de, dataflow
BBIG1, daily 1-month quotations, act/360, percent p.a.):
  bbk_ST0262_fibor_1m_daily.csv      FIBOR 1-month, DEM, 1990-07-02 .. 1998-12-30
  bbk_ST0310_euribor_1m_daily.csv    EURIBOR 1-month, EUR, 1998-12-30 ..
  bbk_ST0104_frankfurt_1m_daily.csv  Frankfurt banks' 1-month funds, .. 2012-05 (cross-check only)
and writes data/eu/private/raw/rf_euro.csv: date, rate_pct_pa, series, with EURIBOR from
1999-01-01 and FIBOR before, the Mark-then-euro rule of the numeraire. Non-quoted days
(weekends, holidays) are dropped here and forward-filled onto the trading calendar by
build_eu_returns.py. EURIBOR daily values are fee-liable at EMMI for commercial use;
this is academic use (see the Bundesbank note in the file header).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "eu" / "private" / "raw"   # raw.dir of configs/eu_data.yaml
EXT = RAW / "external"


def read_bbk(name: str) -> pd.Series:
    df = pd.read_csv(EXT / name, skiprows=1, header=None, usecols=[0, 1], names=["date", "value"])
    df = df[df["date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().assign(date=lambda t: pd.to_datetime(t["date"])).set_index("date")["value"].sort_index()


def main():
    fibor, euribor, ffm = read_bbk("bbk_ST0262_fibor_1m_daily.csv"), read_bbk("bbk_ST0310_euribor_1m_daily.csv"), read_bbk("bbk_ST0104_frankfurt_1m_daily.csv")
    switch = pd.Timestamp("1999-01-01")
    rf = pd.concat([fibor[fibor.index < switch].rename("rate_pct_pa").to_frame().assign(series="FIBOR_1M_DEM"),
                    euribor[euribor.index >= switch].rename("rate_pct_pa").to_frame().assign(series="EURIBOR_1M_EUR")])
    rf = rf.rename_axis("date").reset_index()
    rf.to_csv(RAW / "rf_euro.csv", index=False)
    print(f"rf_euro.csv: {len(rf):,} quoted days {rf['date'].min().date()} .. {rf['date'].max().date()}; "
          f"{rf['series'].value_counts().to_dict()}")
    # cross-checks: FIBOR vs Frankfurt banks' rate on the overlap, EURIBOR vs the same to 2012
    for name, s in [("FIBOR", fibor), ("EURIBOR", euribor)]:
        j = pd.concat([s.rename("a"), ffm.rename("b")], axis=1).dropna()
        print(f"{name} vs Frankfurt banks 1M ({j.index.min().date()}..{j.index.max().date()}, {len(j):,} days): "
              f"mean diff {(j.a - j.b).mean():+.3f} pp, corr {j.a.corr(j.b):.4f}")
    print("at the switch:", fibor.loc["1998-12-28":].round(3).to_dict(), euribor.loc[:"1999-01-05"].round(3).to_dict())


if __name__ == "__main__":
    main()
