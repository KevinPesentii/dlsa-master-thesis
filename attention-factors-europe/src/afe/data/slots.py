"""The three schema tables -> the slot arrays of model/attention_pipeline.py.

Reads returns.parquet, universe.parquet and features.parquet (docs/schemas.md) and
returns tensors in the slot layout of policy/trading.py: on each date the members of
the month fill slots 0..S-1 in cap_rank order, and `idx` says which pool column (one
per company, stable through time) a slot holds. Market-agnostic: it reads the contract,
not CRSP.

Three decisions live here and nowhere else.

LAG. forward_span wants X row t to hold what was known at the close of t-1, and a
features.parquet row (t, i) holds characteristics as of t itself (char_Ret_D1 IS the
return of day t). So X[t, s] = features[t-1, company(t, s)]: the previous trading
day's row of the SAME company, looked up by sec_id, never by slot. Shifting a
slot-space array would hand a company its neighbour's characteristics after a
rebalance. The whole row is shifted, monthly characteristics included, which is one
day more conservative than strictly needed on the first trading day of a month.

ENTRY DAY. A name entering the universe has no features row on its last day outside
it. Its char_* are set to 0 (the cross-sectional median in rank units, the build's own
convention for a missing characteristic); its med_* and rf take the date's
cross-sectional values, which do not depend on the name.

MEMBERSHIP. in_universe[t, s] is membership for the month AND a return on the day. A
member without a return row (delisted mid-month, halted) keeps its slot and its idx so
the pool column stays stable, but is masked: R is 0 there, and the factor model, the
residual windows and the policy all ignore it.

`rf` is the rate of the traded day t itself, unshifted, for the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from afe import schemas


@dataclass
class SlotPanel:
    dates: pd.DatetimeIndex        # trading days, ascending
    sec_ids: np.ndarray            # pool, str: pool column j is company sec_ids[j]
    features: list[str]            # the M feature columns, in schemas.feature_columns order
    X: torch.Tensor                # (T, S, M) float32, row t known at the close of t-1
    R: torch.Tensor                # (T, S) float32, 0 where no return
    in_universe: torch.Tensor      # (T, S) bool
    idx: torch.Tensor              # (T, S) int64, pool column, n_pool for padding
    rf: torch.Tensor               # (T,) float32, rate of day t
    R_pool: np.ndarray             # (T, n_pool) float32, 0 where no return; for evaluation
    mkt_ew: np.ndarray             # (T,) equal-weighted return of the members with a return

    @property
    def n_pool(self) -> int:
        return len(self.sec_ids)


def load_slots(data_dir: Path, start: str, end: str) -> SlotPanel:
    """Slot arrays for the trading days in [start, end]. Features of the trading day
    before `start` are read too, so that the first row is lagged like every other."""
    data_dir = Path(data_dir)
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)
    lo, hi = (t_start - pd.DateOffset(days=10)).to_pydatetime(), t_end.to_pydatetime()

    uni = pd.read_parquet(data_dir / "universe.parquet")
    uni["ym"] = uni["month"].dt.to_period("M")
    uni = uni[(uni["ym"] >= t_start.to_period("M")) & (uni["ym"] <= t_end.to_period("M"))]
    uni = uni.sort_values(["ym", "cap_rank"], ignore_index=True)
    sec_ids = np.sort(uni["sec_id"].unique())
    n_pool = len(sec_ids)
    col = pd.Series(np.arange(n_pool), index=sec_ids)          # sec_id -> pool column

    ret = pq.read_table(data_dir / "returns.parquet", columns=["date", "sec_id", "ret"],
                        filters=[("date", ">=", t_start.to_pydatetime()), ("date", "<=", hi)]).to_pandas()
    ret["date"] = pd.to_datetime(ret["date"]).astype("datetime64[ns]")
    ret = ret[ret["sec_id"].isin(col.index)]
    dates = pd.DatetimeIndex(np.sort(ret["date"].unique()))
    T = len(dates)

    # idx: the month's members, cap_rank order, one row per trading day of the month
    months = pd.PeriodIndex(uni["ym"].unique(), freq="M")
    S = int(uni.groupby("ym").size().max())
    idx_month = np.full((len(months), S), n_pool, dtype=np.int64)
    for m, (_, g) in enumerate(uni.groupby("ym", sort=True)):
        idx_month[m, :len(g)] = col[g["sec_id"].to_numpy()].to_numpy()
    m_of_t = months.get_indexer(dates.to_period("M"))
    idx = np.where(m_of_t[:, None] >= 0, idx_month[np.clip(m_of_t, 0, None)], n_pool)

    # returns in pool space, then gathered into slots; missing return = masked slot
    R_pool = np.full((T, n_pool + 1), np.nan, dtype=np.float32)
    R_pool[dates.get_indexer(ret["date"]), col[ret["sec_id"].to_numpy()].to_numpy()] = ret["ret"].to_numpy()
    R_slot = np.take_along_axis(R_pool, idx, axis=1)
    in_universe = ~np.isnan(R_slot)
    R_slot = np.nan_to_num(R_slot)
    R_pool = np.nan_to_num(R_pool[:, :n_pool])
    n_mem = in_universe.sum(axis=1)
    mkt_ew = np.where(n_mem > 0, (R_slot * in_universe).sum(axis=1) / np.maximum(n_mem, 1), 0.0)

    # features: rows of the previous trading day, looked up by company
    head = pq.read_schema(data_dir / "features.parquet").names
    feats = schemas.feature_columns(pd.DataFrame(columns=head))
    tbl = pq.read_table(data_dir / "features.parquet", columns=["date", "sec_id"] + feats,
                        filters=[("date", ">=", lo), ("date", "<=", hi)])
    f_date = pd.DatetimeIndex(tbl.column("date").to_pandas()).astype("datetime64[ns]")
    f_sec = tbl.column("sec_id").to_pandas().to_numpy()
    F = np.column_stack([tbl.column(c).to_numpy() for c in feats]).astype(np.float32, copy=False)
    del tbl
    f_t = f_date.unique().sort_values()
    f_ti = f_t.get_indexer(f_date)
    known = np.isin(f_sec, col.index)
    f_j = np.full(len(f_sec), n_pool, dtype=np.int64)
    f_j[known] = col[f_sec[known]].to_numpy()
    fpos = np.full((len(f_t), n_pool + 1), -1, dtype=np.int64)
    fpos[f_ti, f_j] = np.arange(len(f_sec))
    fpos[:, n_pool] = -1                                        # padding never has a row

    is_char = np.array([c.startswith("char_") for c in feats])
    date_vals = F[fpos.max(axis=1).clip(0)]                     # any row of the date: med_*, rf
    date_vals[fpos.max(axis=1) < 0] = 0.0
    date_vals[:, is_char] = 0.0

    prev = np.full(T, -1, dtype=np.int64)                       # feature date of t-1, exact
    prev[1:] = f_t.get_indexer(dates[:-1])
    prev[0] = f_t.searchsorted(dates[0]) - 1                    # the day before `start`, if read
    rows = np.where(prev[:, None] >= 0, fpos[np.clip(prev, 0, None)][np.arange(T)[:, None], idx], -1)
    X = F[np.clip(rows, 0, None)]
    miss = rows < 0
    X[miss] = np.broadcast_to(date_vals[np.clip(prev, 0, None)][:, None, :], X.shape)[miss]
    X[prev < 0] = 0.0
    del F

    own = f_t.get_indexer(dates)                                # rf of day t itself
    rf_col = feats.index("rf")
    rf = np.where(own >= 0, date_vals[np.clip(own, 0, None), rf_col], 0.0).astype(np.float32)

    return SlotPanel(dates, sec_ids, feats, torch.from_numpy(X), torch.from_numpy(R_slot),
                     torch.from_numpy(in_universe), torch.from_numpy(idx), torch.from_numpy(rf),
                     R_pool, mkt_ew)
