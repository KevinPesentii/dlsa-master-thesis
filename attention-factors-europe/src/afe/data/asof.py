"""Stage 2c: the tables that let a name be traded from the day it enters the universe.

    features_asof.parquet    what is known at the close of t-1, in the row dated t
    returns_history.parquet  returns of pool names that returns.parquet has no row for

docs/schemas.md is the contract. scripts/build_asof_tables.py runs this from the outputs of
stage 2 (the inspection tables characteristics_*.parquet, the schema tables), so nothing has
to be pulled or rebuilt.

Why the first table exists. features.parquet dates a row by the day its daily characteristics
are measured, and the loader shifts the whole row one trading day. Monthly and annual
characteristics are already lagged inside that row (a row in month M carries the end of M-1),
so on the first trading day of M the shift shows the end of M-2: they are lagged twice, one
day longer than the information needs. The same shift ranks that day's daily characteristics
in last month's universe and gives an entering name zeros, because it has no row on its last
day outside. Here only the daily block moves, by re-dating it (lag_daily), and assemble_year
does the rest as for features.parquet: every row is the universe of its own date.

It is safe to use the end of M-1 on the first day of M: those values use the close of the
last trading day of M-1, the same close as the daily characteristics of t-1 and as the cap
that sets M's membership.

Point in time: a row dated t uses nothing dated t, tests/test_asof_features.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from afe import schemas
from afe.data import build_us as bu


def lag_daily(dc: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """dc (date, permno) re-dated to the next trading day: the row dated t holds the value of
    t-1. The value of the last day of `calendar` has no next day and leaves."""
    next_day = pd.Series(calendar[1:], index=calendar[:-1])
    dates = pd.DatetimeIndex(next_day.reindex(dc.index.get_level_values("date")).to_numpy())
    keep = np.asarray(dates.notna())
    out = dc[keep].copy()
    out.index = pd.MultiIndex.from_arrays([dates[keep], dc.index.get_level_values("permno")[keep]],
                                          names=["date", "permno"])
    return out


def members_by_month(uni: pd.DataFrame, as_int: bool) -> dict:
    """universe.parquet -> {Period[M]: sorted members}, in the id type of the inspection tables."""
    month = pd.to_datetime(uni["month"]).dt.to_period("M")
    return {m: sorted(g.astype("int64").tolist() if as_int else g.tolist())
            for m, g in uni["sec_id"].groupby(month)}


def write_features(dc: pd.DataFrame, mc: pd.DataFrame, uni: pd.DataFrame, rf: pd.Series, names: list[str],
                   path: Path, log=print) -> None:
    """features_asof.parquet from the inspection tables (dc, mc) and the universe. `rf` by date
    gives the rows' dates, those of features.parquet."""
    days = pd.DatetimeIndex(dc.index.get_level_values("date").unique()).sort_values()
    calendar = days.union(rf.index)
    lagged = lag_daily(dc, calendar)
    members = members_by_month(uni, pd.api.types.is_integer_dtype(dc.index.get_level_values("permno")))
    months = pd.PeriodIndex(sorted(members), freq="M")
    start, end = months.min(), months.max()
    writer = None
    for year in range(start.year, end.year + 1):
        feat, _ = bu.assemble_year(year, members, rf.index, mc, lagged, names, rf, start, end, levels=False)
        table = pa.Table.from_pandas(feat, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        log(f"  {year}: {len(feat):,} rows")
    writer.close()


def only_new(rows: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """rows (date, sec_id, ret) without a row in returns.parquet, on its calendar, one per
    (date, sec_id), in the table's dtypes and order."""
    rows = rows[rows["ret"].notna()].copy()
    rows["date"] = pd.to_datetime(rows["date"]).astype("datetime64[ns]")
    rows["sec_id"] = rows["sec_id"].astype(str).astype(object)
    rows = rows[rows["date"].isin(ret["date"].unique())].drop_duplicates(["date", "sec_id"])
    have = pd.MultiIndex.from_frame(ret[["date", "sec_id"]])
    rows = rows[~pd.MultiIndex.from_frame(rows[["date", "sec_id"]]).isin(have)]
    out = pd.DataFrame({"date": rows["date"], "sec_id": rows["sec_id"], "ret": rows["ret"].astype("float32")})
    out = out.sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(out, schemas.RETURNS_HISTORY)
    return out


def us_history(raw_dirs: list[Path], ret: pd.DataFrame) -> pd.DataFrame:
    """US: the raw daily files of the pool. CRSP's return of a line's first listing month has
    no prior-month cap, so the schema table leaves it out."""
    pool = sorted(ret["sec_id"].astype("int64").unique().tolist())
    years = sorted({y for y in pd.to_datetime(ret["date"]).dt.year.unique()})
    parts = []
    for d in raw_dirs:
        for y in years:
            if (d / f"{y}.parquet").exists():
                t = pq.read_table(d / f"{y}.parquet", columns=["permno", "date", "ret"],
                                  filters=[("permno", "in", pool)]).to_pandas()
                t["ret"] = t["ret"].astype("float64")      # the parquet metadata restores nullable Float64
                parts.append(t.rename(columns={"permno": "sec_id"}).assign(sec_id=lambda x: x["sec_id"].astype("int64")))
    return only_new(pd.concat(parts, ignore_index=True), ret)


def europe_history(detail: pd.DataFrame, asof: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """Europe: a company's listing rows (returns_detail_daily, in the numeraire), the line that
    carried it when it first entered the universe (universe_asof), as the company's series
    before its first table row is that line's."""
    first = asof.sort_values("month").drop_duplicates("gvkey")[["gvkey", "iid"]].astype(str)
    d = detail.assign(gvkey=detail["gvkey"].astype(str), iid=detail["iid"].astype(str))
    d = d.merge(first, on=["gvkey", "iid"], how="inner")
    return only_new(d[["date", "gvkey", "ret"]].rename(columns={"gvkey": "sec_id"}), ret)
