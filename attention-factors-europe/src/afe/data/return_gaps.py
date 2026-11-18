"""Member-days without a return inside a name's trading life: a halt, not a delisting.

Kevin's data check (2026-11-17, item 2). The loaders masked a member on every day without a row in
returns.parquet. The mask of day t took the name out of the factor softmax and set its weight for t to
0, although that weight is decided at the close of t-1, before anyone could know the stock would not
trade on t. Such a day has no `traded = False` row either, so trading.execute did not hold the position:
the book sold at the close before the gap and missed the reopening return. Right for a delisting, which
is announced and so known at t-1; look-ahead for a halt.

`data.return_gaps: closed` treats a halt day like an exchange holiday in the holiday fill (a zero-return,
traded = False row): return 0, market closed, so the position decided at t-1 is held through the halt and
earns the reopening return, which spans it (CRSP and the Europe build chain prices over missing days).
The name stays in the universe on those days. `masked` (the default) is the behaviour before 2026-11-18,
so every earlier run reproduces.

THE RULE. Company i trades on a market day when it has a row in returns.parquet or returns_history.parquet
(the market calendar = the dates of returns.parquet). A run of days on which i does not trade is a HALT when
i traded both before and after it and it lasts at most MAX_HALT_DAYS market days. From the first day of a
halt on which i is a member (in the month's universe) to its end, every day is closed with return 0: a halt
that runs past the end of i's last member month stays closed until i trades again, so a position held into
it is not sold at a close at which i did not trade. A longer run is treated as an exit, as a delisting is:
masked. Separately, a member day on which only returns_history has a row (another line of the company
traded, Europe) is closed with return 0 too.

MAX_HALT_DAYS = 63 (a quarter). On the 2026-11-17 builds every halt that starts on a member day lasts at most
20 market days in the US and 41 in Europe; the next are 128, 991, 1,284 and 1,916 days (names gone for months
or years and listed again: US permno 79996 2011-2019, Europe gvkey 246217 1996-2001), so any cap from 42 to 127
gives the same split. That a name trades again within the cap is what the exchange's notice says (a halt,
not a delisting); no return of the halt or after it enters a decision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODES = ("masked", "closed")
MAX_HALT_DAYS = 63


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown data.return_gaps: {mode} (one of {', '.join(MODES)})")
    return mode


def _rows(data_dir: Path, name: str, cal: pd.DatetimeIndex, col: pd.Index) -> np.ndarray:
    """(T, N) bool on the calendar `cal`: `name` has a row for (date, sec_id)."""
    out = np.zeros((len(cal), len(col)), dtype=bool)
    f = data_dir / name
    if f.exists():
        t = pq.read_table(f, columns=["date", "sec_id"]).to_pandas()
        r = cal.get_indexer(pd.to_datetime(t["date"]).astype("datetime64[ns]"))
        c = col.get_indexer(t["sec_id"].astype(str))
        ok = (r >= 0) & (c >= 0)
        out[r[ok], c[ok]] = True
    return out


def _members(data_dir: Path, cal: pd.DatetimeIndex, col: pd.Index, max_rank: int | None) -> np.ndarray:
    """(T, N) bool: sec_id in universe.parquet (cap_rank <= max_rank) in the month of the date."""
    uni = pd.read_parquet(data_dir / "universe.parquet", columns=["month", "sec_id", "cap_rank"])
    if max_rank is not None:
        uni = uni[uni["cap_rank"] <= max_rank]
    w = pd.crosstab(pd.to_datetime(uni["month"]).dt.to_period("M"), uni["sec_id"].astype(str))
    w = w.reindex(columns=col, fill_value=0) > 0
    return w.reindex(cal.to_period("M"), fill_value=False).to_numpy(dtype=bool)


def halted(data_dir: Path, dates, cols, max_rank: int | None = None) -> np.ndarray:
    """(len(dates), len(cols)) bool: the cells `data.return_gaps: closed` turns into closed, zero-return
    days. Worked out on the whole market calendar of the tables (a halt is measured from end to end,
    whatever span is loaded), then read off at `dates`; a date outside the calendar is never closed.
    `max_rank`: membership as in a run with a cap_rank limit."""
    data_dir = Path(data_dir)
    dates = pd.DatetimeIndex(dates).astype("datetime64[ns]")
    col = pd.Index(np.asarray(cols).astype(str))
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(
        pq.read_table(data_dir / "returns.parquet", columns=["date"]).column("date").unique().to_pandas()))).astype("datetime64[ns]")
    has = _rows(data_dir, "returns.parquet", cal, col)
    other = _rows(data_dir, "returns_history.parquet", cal, col) & ~has
    member = _members(data_dir, cal, col, max_rank)
    trade = has | other
    T = len(cal)
    t = np.arange(T, dtype=np.int32)[:, None]
    prev = np.maximum.accumulate(np.where(trade, t, -1), axis=0)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(trade, t, T), axis=0), axis=0), axis=0)
    halt = ~trade & (prev >= 0) & (nxt < T) & (nxt - prev - 1 <= MAX_HALT_DAYS)
    member_halt = np.maximum.accumulate(np.where(member & halt, t, -1), axis=0)    # latest member day of a halt
    closed = (halt & (member_halt > prev)) | (member & other)                       # ... in this halt, at or before t
    out = np.zeros((len(dates), len(col)), dtype=bool)
    r = cal.get_indexer(dates)
    out[r >= 0] = closed[r[r >= 0]]
    return out
