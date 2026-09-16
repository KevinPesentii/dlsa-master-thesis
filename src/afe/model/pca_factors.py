"""Rolling PCA residuals: step one of the two-step benchmark (Table 2, "PCA Factors").

Construction is Guijarro-Ordonez, Pelger & Zanotti's (`factor_models/pca.py` in
dlsa-public, the non-vectorised branch that also builds the composition matrices):

  at each date t, on the names in the universe with a full `cov_window` return history,
    Z      = standardised returns over R[t-252 : t]        (the window ends the day BEFORE t)
    V      = top-K eigenvectors of Z'Z                      (N x K)
    F      = (R[t-60 : t] / vol) V                          (60 x K factor returns)
    B      = argmin ||R[t-60 : t] - F B'||                  (N x K loadings, no intercept)
    eps_t  = R_t - B (R_t / vol) V  =  (I - B V' D^-1) R_t   with D = diag(vol)

so the residual composition matrix of the paper is  w_eps_{t-1} = I - B V' D^-1, which
only uses information up to t-1.  Nothing here stores it as N x N: `V`, `B` and `1/vol`
are enough to apply it and its transpose (see policy/trading.py).

Residuals are computed for every pool name with a full history, not only universe
members, so that a name entering the universe already has the residual lookback the
policy needs.  Only members have factor weights (V) and are ever traded.

K >= loading_window makes the 60-day regression under-determined; numpy's lstsq then
returns the minimum-norm solution, which is what sklearn's LinearRegression in the GPZ
code does as well.  Table 2 has K = 100, so this case is real, and it is kept as is.

`loading_window` is a config choice because Epstein et al. only say "PCA using the past
252 trading days": 60 is GPZ's regression window, 252 regresses on the whole PCA window,
and 0 skips the regression and uses the PCA projection itself, B = D V (the loadings of
the standardised returns on their own principal components), so w_eps = I - D V V' D^-1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass
class Panel:
    dates: pd.DatetimeIndex        # trading days, ascending
    sec_ids: np.ndarray            # pool, str
    R: np.ndarray                  # (T, N) float64, NaN where no return
    member: np.ndarray             # (T, N) bool, in the universe on that day
    rf: np.ndarray                 # (T,) float64 daily risk-free rate


def load_panel(data_dir: Path, end: str, history_start: str | None = None,
               raw_daily_dir: Path | None = None) -> Panel:
    """Wide return panel of the pool plus the universe mask.  Rows before the first
    universe month are taken from the raw daily files (needed for the first PCA window)."""
    ret = pd.read_parquet(data_dir / "returns.parquet", columns=["date", "sec_id", "ret"])
    ret = ret[ret["date"] <= pd.Timestamp(end)]
    pool = np.sort(ret["sec_id"].unique())
    first = ret["date"].min()
    if history_start is not None and raw_daily_dir is not None:
        pre = []
        for year in range(pd.Timestamp(history_start).year, first.year + 1):
            f = raw_daily_dir / f"{year}.parquet"
            if f.exists():
                d = pd.read_parquet(f, columns=["permno", "date", "ret"])
                d["sec_id"] = d["permno"].astype(str)
                d = d[d["sec_id"].isin(pool) & (d["date"] < first) & (d["date"] >= pd.Timestamp(history_start))]
                pre.append(d[["date", "sec_id", "ret"]])
        if pre:
            ret = pd.concat(pre + [ret], ignore_index=True)
    R = ret.pivot(index="date", columns="sec_id", values="ret").reindex(columns=pool).astype("float64")
    dates = R.index

    uni = pd.read_parquet(data_dir / "universe.parquet")
    uni["ym"] = uni["month"].dt.to_period("M")
    uni_wide = uni.assign(one=1.0).pivot(index="ym", columns="sec_id", values="one").reindex(columns=pool).notna()
    member = uni_wide.reindex(dates.to_period("M"), fill_value=False).to_numpy()

    ff = pd.read_parquet(data_dir / "raw" / "ff_daily.parquet", columns=["date", "rf"])
    rf = ff.set_index("date")["rf"].astype("float64").reindex(dates).ffill().fillna(0.0).to_numpy()
    return Panel(dates, pool, R.to_numpy(), member, rf)


def full_history(R: np.ndarray, window: int) -> np.ndarray:
    """ok[t, i] is True when R[t-window : t, i] has no NaN (the window ends before t)."""
    T = R.shape[0]
    c = np.zeros((T + 1, R.shape[1]), dtype=np.int32)
    np.cumsum(np.isnan(R), axis=0, out=c[1:])
    ok = np.zeros(R.shape, dtype=bool)
    ok[window:] = (c[window:T] - c[:T - window]) == 0
    return ok


def _pca_one_date(R, t, P, Q, Ks, cov_window, loading_window, k_max):
    """Eigenvectors, loadings and residuals at one date. P: PCA/tradable set, Q: pool
    names with full history (superset of P). Returns dict K -> (V, B_P, resid_Q), 1/vol."""
    W = R[t - cov_window:t][:, P]
    mu = W.mean(axis=0)
    vol = W.std(axis=0)
    Z = (W - mu) / vol
    n = Z.shape[1]
    evals, evecs = eigh(Z.T @ Z, subset_by_index=[n - k_max, n - 1])
    evals, evecs = evals[::-1], evecs[:, ::-1]               # descending eigenvalue order
    r_t = np.nan_to_num(R[t, P])                             # NaN today -> 0 in the factor
    P_in_Q = P[Q]                                            # rows of P inside Q, same order
    if loading_window == 0:                                  # projection: standardised Q on the PCs
        WQ = R[t - cov_window:t][:, Q]
        volQ = WQ.std(axis=0)
        ZQ = (WQ - WQ.mean(axis=0)) / volQ
        G = ZQ.T @ (Z @ evecs)                               # (nQ, k_max) = Z_Q' F_z; members give V * evals
    else:
        RQ = R[t - loading_window:t][:, Q]                   # (loading_window, nQ)
    out = {}
    for K in Ks:
        V = evecs[:, :K]
        F_t = (r_t / vol) @ V                                # (K,)
        if loading_window == 0:
            B_Q = volQ[:, None] * G[:, :K] / evals[:K]       # equals vol * V on the PCA set
        else:
            F = (W[-loading_window:] / vol) @ V              # (loading_window, K)
            B_Q = np.linalg.lstsq(F, RQ, rcond=None)[0].T    # (nQ, K), min-norm if K >= window
        resid_Q = R[t, Q] - B_Q @ F_t                        # NaN where R[t] is NaN
        out[K] = (V, B_Q[P_in_Q], resid_Q)
    return out, 1.0 / vol


def build(panel: Panel, Ks: list[int], cov_window: int, loading_window: int, out_dir: Path,
          max_members: int = 500, log=print) -> dict:
    """Write, per K, `out_dir/K<K>/` with
        resid.npy   (T, N_pool) float32   residual return, NaN where undefined
        V.npy       (T, max_members, K)   factor weights of the PCA set, zero-padded
        B.npy       (T, max_members, K)   loadings of the PCA set, zero-padded
        invvol.npy  (T, max_members)      1 / vol of the PCA set, zero-padded
        idx.npy     (T, max_members) int32  pool column of each slot, N_pool for padding
        n.npy       (T,) int16            size of the PCA set (0 before the first date)
    plus dates.npy, sec_ids.npy and meta.json at `out_dir`."""
    R, T, N = panel.R, *panel.R.shape
    Ks = sorted(Ks)
    k_max = max(Ks)
    ok = full_history(R, cov_window)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dates.npy", panel.dates.to_numpy())
    np.save(out_dir / "sec_ids.npy", panel.sec_ids)

    mm = {}
    for K in Ks:
        d = out_dir / f"K{K}"
        d.mkdir(exist_ok=True)
        mm[K] = {
            "resid": np.lib.format.open_memmap(d / "resid.npy", "w+", np.float32, (T, N)),
            "V": np.lib.format.open_memmap(d / "V.npy", "w+", np.float32, (T, max_members, K)),
            "B": np.lib.format.open_memmap(d / "B.npy", "w+", np.float32, (T, max_members, K)),
        }
        mm[K]["resid"][:] = np.nan
    invvol = np.zeros((T, max_members), dtype=np.float32)
    idx = np.full((T, max_members), N, dtype=np.int32)
    n_set = np.zeros(T, dtype=np.int16)

    for t in range(cov_window, T):
        Q = ok[t].copy()
        Q[Q] = R[t - cov_window:t][:, Q].std(axis=0) > 0            # constant window: no correlation
        P = panel.member[t] & Q
        n = int(P.sum())
        if n <= k_max:
            continue
        if n > max_members:  # cannot happen with a 500-name universe; guard the layout
            raise ValueError(f"{n} names in the PCA set at {panel.dates[t]:%Y-%m-%d} > {max_members}")
        res, iv = _pca_one_date(R, t, P, Q, Ks, cov_window, loading_window, k_max)
        cols = np.flatnonzero(P)
        idx[t, :n] = cols
        invvol[t, :n] = iv
        n_set[t] = n
        for K, (V, B, resid_Q) in res.items():
            mm[K]["V"][t, :n] = V
            mm[K]["B"][t, :n] = B
            mm[K]["resid"][t, Q] = resid_Q
        if (t - cov_window) % 250 == 0:
            log(f"  {panel.dates[t]:%Y-%m-%d}: PCA set {n}, pool with history {int(Q.sum())}")

    for K in Ks:
        for a in mm[K].values():
            a.flush()
        np.save(out_dir / f"K{K}" / "invvol.npy", invvol)
        np.save(out_dir / f"K{K}" / "idx.npy", idx)
        np.save(out_dir / f"K{K}" / "n.npy", n_set)
    meta = {"cov_window": cov_window, "loading_window": loading_window, "Ks": Ks,
            "n_pool": int(N), "n_dates": int(T), "max_members": max_members,
            "first_date": str(panel.dates[cov_window].date()), "last_date": str(panel.dates[-1].date())}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def composition_check(R_t: np.ndarray, V: np.ndarray, B: np.ndarray, invvol: np.ndarray,
                      resid_t: np.ndarray) -> float:
    """max |eps_t - (I - B V' D^-1) R_t| on the PCA set: the identity the policy relies on."""
    eps = R_t - B @ (V.T @ (invvol * R_t))
    return float(np.nanmax(np.abs(eps - resid_t)))
