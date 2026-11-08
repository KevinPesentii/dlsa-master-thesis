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
timing "asof" reads features_asof.parquet instead, whose row t already holds what was known
at the close of t-1, so nothing is shifted and that extra day is gone (data/asof.py). It
fixes the entry day too: a member has its row on its first day.

ENTRY DAY. A name entering the universe has no features row on its last day outside
it. Its char_* are set to 0 (the cross-sectional median in rank units, the build's own
convention for a missing characteristic); its med_* and rf take the date's
cross-sectional values, which do not depend on the name.

MEMBERSHIP. in_universe[t, s] is membership for the month AND a return on the day. A
member without a return row (delisted mid-month, halted) keeps its slot and its idx so
the pool column stays stable, but is masked: R is 0 there, and the factor model, the
residual windows and the policy all ignore it.

`rf` is the rate of the traded day t itself, unshifted, for the objective. `closed`
(optional returns.traded) marks closes at which a name could not be traded.

Two options for robustness runs: `max_rank` keeps the members up to that cap_rank (a
top-200 US run; the features stay ranked over the full universe of the build), and
`drop_groups` removes a characteristic theme's char_* and med_* columns (Table 3).

`medians` and `rf_input` say whether the med_* columns and rf enter X. Neither can move the
factor weights: a column that is the same for every name on a date adds a constant to a
row of scores, which the softmax over names drops (model/attention_factors.py). rf stays in
the panel as the rate of the objective either way.

`history` adds returns_history.parquet to R_win and has_ret, the two arrays only the fixed
residual window reads. R_pool, which the book is valued with, never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from afe import schemas
from afe.data.characteristics import REGISTRY


@dataclass
class SlotPanel:
    dates: pd.DatetimeIndex        # trading days, ascending
    sec_ids: np.ndarray            # pool, str: pool column j is company sec_ids[j]
    features: list[str]            # the M feature columns of X, in schemas.feature_columns order
    X: torch.Tensor                # (T, S, M) float32, row t known at the close of t-1
    R: torch.Tensor                # (T, S) float32, 0 where no return
    in_universe: torch.Tensor      # (T, S) bool
    idx: torch.Tensor              # (T, S) int64, pool column, n_pool for padding
    rf: torch.Tensor               # (T,) float32, rate of day t
    R_pool: np.ndarray             # (T, n_pool) float32, 0 where no return; for evaluation
    mkt_ew: np.ndarray             # (T,) equal-weighted return of the members with a return
    closed: np.ndarray | None = None  # (T, n_pool) bool, the name's market did not trade at the
                                      # close of t (returns.traded False); None = all traded
    has_ret: np.ndarray | None = None  # (T, n_pool) bool, R_win holds a return there (member or not);
                                       # policy.residual_window fixed needs it
    R_win: np.ndarray | None = None    # (T, n_pool) float32, R_pool, plus returns_history with `history`;
                                       # the returns of the fixed residual window
    dev: dict | None = None            # tensor copies of R_pool, R_win, has_ret and closed on the device
                                       # of X (to_device); None = on the CPU, read through torch.from_numpy

    @property
    def n_pool(self) -> int:
        return len(self.sec_ids)


ON_DEVICE = ("R_pool", "R_win", "has_ret", "closed")


def to_device(p: SlotPanel, device) -> SlotPanel:
    """The panel with its tensors on `device`, and tensor copies of the numpy arrays the training step
    slices (ON_DEVICE), so a GPU step copies nothing from the host. The numpy arrays stay as they are,
    for the evaluation on the CPU."""
    dev = {k: torch.from_numpy(getattr(p, k)).to(device) for k in ON_DEVICE if getattr(p, k) is not None}
    return replace(p, X=p.X.to(device), R=p.R.to(device), in_universe=p.in_universe.to(device),
                   idx=p.idx.to(device), rf=p.rf.to(device), dev=dev)


def rows(p: SlotPanel, name: str, a: int, b: int) -> torch.Tensor:
    """Rows a..b-1 of the numpy array `name` (one of ON_DEVICE) as a tensor on the device of p.X."""
    if p.dev is not None and name in p.dev:
        return p.dev[name][a:b]
    return torch.from_numpy(getattr(p, name)[a:b])


def load_config(cfg: dict, data_dir: Path, start: str, end: str, max_rank: int | None = None) -> SlotPanel:
    """load_slots with the choices of a run config (its `features` and `data` blocks)."""
    f, d = cfg["features"], cfg.get("data") or {}
    return load_slots(data_dir, start, end, max_rank, timing=f.get("timing", "lagged"),
                      medians=f.get("include_cross_sectional_median", True),
                      rf_input=f.get("include_risk_free", True), history=bool(d.get("returns_history")))


def load_slots(data_dir: Path, start: str, end: str, max_rank: int | None = None, timing: str = "lagged",
               medians: bool = True, rf_input: bool = True, history: bool = False) -> SlotPanel:
    """Slot arrays for the trading days in [start, end]. Features of the trading day
    before `start` are read too, so that the first row is lagged like every other.
    `max_rank`: only members with cap_rank <= max_rank. timing "lagged": features.parquet,
    shifted one trading day; "asof": features_asof.parquet as it is."""
    if timing not in ("lagged", "asof"):
        raise ValueError(f"unknown features.timing: {timing}")
    if timing == "asof" and medians:
        raise ValueError("features_asof.parquet has no med_* columns: set include_cross_sectional_median false")
    data_dir = Path(data_dir)
    table = data_dir / ("features.parquet" if timing == "lagged" else "features_asof.parquet")
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)
    lo, hi = (t_start - pd.DateOffset(days=10)).to_pydatetime(), t_end.to_pydatetime()

    uni = pd.read_parquet(data_dir / "universe.parquet")
    uni["ym"] = uni["month"].dt.to_period("M")
    uni = uni[(uni["ym"] >= t_start.to_period("M")) & (uni["ym"] <= t_end.to_period("M"))]
    if max_rank is not None:
        uni = uni[uni["cap_rank"] <= max_rank]
    uni = uni.sort_values(["ym", "cap_rank"], ignore_index=True)
    sec_ids = np.sort(uni["sec_id"].unique())
    n_pool = len(sec_ids)
    col = pd.Series(np.arange(n_pool), index=sec_ids)          # sec_id -> pool column

    has_traded = "traded" in pq.read_schema(data_dir / "returns.parquet").names
    ret = pq.read_table(data_dir / "returns.parquet", columns=["date", "sec_id", "ret"] + (["traded"] if has_traded else []),
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
    has_ret = ~np.isnan(R_pool[:, :n_pool])
    R_win = None
    if history:
        h = pq.read_table(data_dir / "returns_history.parquet", columns=["date", "sec_id", "ret"],
                          filters=[("date", ">=", t_start.to_pydatetime()), ("date", "<=", hi)]).to_pandas()
        h["date"] = pd.to_datetime(h["date"]).astype("datetime64[ns]")
        rows = dates.get_indexer(h["date"])
        on = (rows >= 0) & h["sec_id"].isin(col.index).to_numpy()
        R_win = R_pool[:, :n_pool].copy()
        R_win[rows[on], col[h["sec_id"].to_numpy()[on]].to_numpy()] = h["ret"].to_numpy()[on]
        has_ret = ~np.isnan(R_win)
        R_win = np.nan_to_num(R_win)
    R_pool = np.nan_to_num(R_pool[:, :n_pool])
    R_win = R_pool if R_win is None else R_win
    closed = None
    if has_traded:
        closed = np.zeros((T, n_pool + 1), dtype=bool)
        nt = ~ret["traded"].to_numpy(dtype=bool)
        closed[dates.get_indexer(ret["date"][nt]), col[ret["sec_id"].to_numpy()[nt]].to_numpy()] = True
        closed = closed[:, :n_pool]
    n_mem = in_universe.sum(axis=1)
    mkt_ew = np.where(n_mem > 0, (R_slot * in_universe).sum(axis=1) / np.maximum(n_mem, 1), 0.0)

    # features: rows of the previous trading day, looked up by company
    head = pq.read_schema(table).names
    feats = schemas.feature_columns(pd.DataFrame(columns=head))
    tbl = pq.read_table(table, columns=["date", "sec_id"] + feats,
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

    if timing == "asof":
        prev = f_t.get_indexer(dates)                           # row t is already what t-1 knew
    else:
        prev = np.full(T, -1, dtype=np.int64)                   # feature date of t-1, exact
        prev[1:] = f_t.get_indexer(dates[:-1])
        prev[0] = f_t.searchsorted(dates[0]) - 1                # the day before `start`, if read
    rows = np.where(prev[:, None] >= 0, fpos[np.clip(prev, 0, None)][np.arange(T)[:, None], idx], -1)
    X = F[np.clip(rows, 0, None)]
    miss = rows < 0
    X[miss] = np.broadcast_to(date_vals[np.clip(prev, 0, None)][:, None, :], X.shape)[miss]
    X[prev < 0] = 0.0
    del F

    own = f_t.get_indexer(dates)                                # rf of day t itself
    rf_col = feats.index("rf")
    rf = np.where(own >= 0, date_vals[np.clip(own, 0, None), rf_col], 0.0).astype(np.float32)

    keep = [i for i, f in enumerate(feats) if f.startswith("char_") or (medians and f.startswith("med_"))
            or (rf_input and f == "rf")]
    if len(keep) < len(feats):
        X, feats = X[:, :, keep], [feats[i] for i in keep]

    return SlotPanel(dates, sec_ids, feats, torch.from_numpy(X), torch.from_numpy(R_slot),
                     torch.from_numpy(in_universe), torch.from_numpy(idx), torch.from_numpy(rf),
                     R_pool, mkt_ew, closed, has_ret, R_win)


def drop_groups(p: SlotPanel, groups: list[str]) -> SlotPanel:
    """The panel without the char_* and med_* columns of the characteristics whose theme
    (characteristics.REGISTRY, e.g. "Past Returns" or past_returns) is in `groups`."""
    key = lambda s: s.lower().replace(" ", "_")  # noqa: E731
    want = {key(g) for g in groups}
    themes = {key(c.theme) for c in REGISTRY.values()}
    if want - themes:
        raise ValueError(f"unknown characteristic groups {sorted(want - themes)}; themes are {sorted(themes)}")
    names = {n for n, c in REGISTRY.items() if key(c.theme) in want}
    keep = [i for i, f in enumerate(p.features) if f.split("_", 1)[-1] not in names or f == "rf"]
    return replace(p, X=p.X[:, :, keep].contiguous(), features=[p.features[i] for i in keep])
