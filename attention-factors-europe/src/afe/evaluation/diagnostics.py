"""What a run keeps besides its weights, small enough to keep for every run.

Written by scripts/run_attention_us.py with `evaluation.diagnostics: {top_n: 20}`. A K = 30, 24-year
run adds about 6 MB, measured on one year. Full factor weights are not stored: models/<year>.pt and the panel rebuild
them for any date (scripts/factor_replay.py), so what is here is what is worth having
without the data.

    diag_daily.parquet      one row per out-of-sample day, the columns below
    factor_returns.parquet  date x K daily factor returns, w_F,t R_t
    factor_top.parquet      first trading day of each month: the top_n names of each factor
    factor_chars.parquet    same dates: each factor's weighted mean rank of every characteristic,
                            its effective number of names and the share of its ten largest
    name_year.parquet       year x name: days held, gross and net exposure, P&L, turnover, short
    train_log.csv           year x epoch: loss, its Sharpe part, explained variance

diag_daily, by question:

  Who can be traded. n_member (slots filled), n_return (members with a return), n_tradable.
  n_new, l1_new, pnl_new, turn_new: the book in names with fewer than L days in the universe, the
  ones policy.residual_window rolling gives no policy position yet (its hedge can still hold them).

  Where the book comes from. w = (w_port - hedge) / l1. l1_port, pnl_port, turn_port are the
  policy's own positions, normalised like w; the rest of gross and turnover is the hedge.

  How good the factors are. r2_cs = 1 - sum eps^2 / sum R^2 over the day's members. eps_abs = mean
  |eps|.

  What the factors look like. eff_n = mean over factors of 1 / sum w_F^2, names per factor. top10 =
  mean share of a factor's ten largest. eff_factors = tr(G)^2 / ||G||_F^2 with G = w_F w_F', the
  effective number of different factors: K if orthogonal and equal, 1 if all the same.

  How fast they move. dw_1d, dw_21d = mean over factors of half the L1 change of w_F over 1 and 21
  days, by company, so a monthly reshuffle of slots does not count. dbeta_1d, dbeta_21d = sum
  |change in loadings| / sum |loadings|. Within a test year: blank on its first 1 and 21 days,
  as a new model has no comparable factor k.

Positions (held) are the executed book, the one that earns gross; the policy columns are targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from afe.model.attention_pipeline import to_pool

FILES = ("diag_daily", "factor_returns", "factor_top", "factor_chars", "name_year")


def not_yet_rolling(p, lookback: int) -> np.ndarray:
    """(T, n_pool) bool: in the universe at t with a return, but on fewer than `lookback` of the
    days before it, so the rolling residual window could not trade it at t."""
    T, P = len(p.dates), p.n_pool
    iu = np.zeros((T, P + 1), dtype=bool)
    np.put_along_axis(iu, p.idx.numpy(), p.in_universe.numpy(), axis=1)
    iu = iu[:, :P]
    c = np.zeros((T + 1, P), dtype=np.int32)
    c[1:] = iu.cumsum(axis=0, dtype=np.int32)
    return iu & (c[np.arange(T)] - c[np.clip(np.arange(T) - lookback, 0, None)] < lookback)


def effective_factors(w_F: torch.Tensor) -> torch.Tensor:
    """tr(G)^2 / ||G||_F^2 per date, G = w_F w_F' (K x K): the participation ratio of its
    eigenvalues, without computing them."""
    G = w_F @ w_F.transpose(1, 2)
    tr = G.diagonal(dim1=1, dim2=2).sum(-1)
    return tr ** 2 / (G ** 2).sum((1, 2)).clamp_min(1e-30)


def by_company(x: torch.Tensor, idx: torch.Tensor, dim: int) -> torch.Tensor:
    """x with slot dimension `dim` (1 of (T, S, K) or 2 of (T, K, S)) re-indexed by the companies
    in idx, so a date can be compared with an earlier one across a rebalance."""
    cols, inv = torch.unique(idx, return_inverse=True)
    shape = list(x.shape)
    shape[dim] = len(cols)
    index = inv.unsqueeze(3 - dim).expand(x.shape)
    return x.new_zeros(shape).scatter_(dim, index, x)


def change(x: torch.Tensor, h: int) -> torch.Tensor:
    """(T, ...) -> the absolute change over h dates (T, ...) with NaN on the first h."""
    out = torch.full_like(x, float("nan"))
    if x.shape[0] > h:
        out[h:] = (x[h:] - x[:-h]).abs()
    return out


class Recorder:
    def __init__(self, p, lookback: int, top_n: int = 20):
        self.p, self.top_n = p, int(top_n)
        self.new = not_yet_rolling(p, lookback)
        self.sec = np.append(p.sec_ids.astype(object), "")          # padding reads ""
        self.chars = [i for i, f in enumerate(p.features) if f.startswith("char_")]
        self.days, self.fret, self.top, self.fchars, self.port, self.train = [], [], [], [], [], []
        self.year_stats: dict = {}

    def add_year(self, out: dict, t0: int, t1: int, year: int, train: list[dict], test_ev: float, tau: float):
        """out: forward_span's output for the trade dates t0..t1-1, eval mode, no grad."""
        p, w_F, bT, tradable = self.p, out["w_F"], out["betaT"], out["tradable"]
        idx, R, iu = p.idx[t0:t1], p.R[t0:t1], p.in_universe[t0:t1]
        dates = p.dates[t0:t1]
        T, K, S = w_F.shape
        eps = out["eps"][-T:] * iu
        Rz = R * iu
        fret = torch.einsum("tks,ts->tk", w_F, Rz)
        self.fret.append(pd.DataFrame(fret.numpy(), index=dates, columns=[f"f{k:03d}" for k in range(K)]))

        eff_n = 1.0 / (w_F ** 2).sum(-1).clamp_min(1e-30)                    # (T, K)
        top10 = w_F.topk(min(10, S), dim=-1).values.sum(-1)                  # (T, K)
        wp, bp = by_company(w_F, idx, 2), by_company(bT, idx, 1)
        beta_size = bp.abs().sum((1, 2)).clamp_min(1e-30)
        self.days.append(pd.DataFrame({
            "date": dates,
            "n_member": (idx < p.n_pool).sum(1).numpy(), "n_return": iu.sum(1).numpy(),
            "n_tradable": tradable.sum(1).numpy(),
            "r2_cs": (1.0 - (eps ** 2).sum(1) / (Rz ** 2).sum(1).clamp_min(1e-30)).numpy(),
            "eps_abs": (eps.abs().sum(1) / iu.sum(1).clamp_min(1)).numpy(),
            "eff_n": eff_n.mean(1).numpy(), "top10": top10.mean(1).numpy(),
            "eff_factors": effective_factors(w_F).numpy(),
            "dw_1d": (0.5 * change(wp, 1).sum(2)).mean(1).numpy(),
            "dw_21d": (0.5 * change(wp, 21).sum(2)).mean(1).numpy(),
            "dbeta_1d": (change(bp, 1).sum((1, 2)) / beta_size).numpy(),
            "dbeta_21d": (change(bp, 21).sum((1, 2)) / beta_size).numpy()}))

        months = dates.to_period("M")
        for t in np.flatnonzero(np.r_[True, months[1:] != months[:-1]]):
            vals, slots = w_F[t].topk(min(self.top_n, S), dim=-1)            # (K, n), largest first
            n = vals.shape[1]
            top = pd.DataFrame({"date": dates[t], "factor": np.repeat(np.arange(K, dtype=np.int16), n),
                                "rank": np.tile(np.arange(1, n + 1, dtype=np.int16), K),
                                "sec_id": self.sec[idx[t].numpy()[slots.numpy().ravel()]],
                                "w": vals.numpy().ravel()})
            self.top.append(top[top["w"] > 0])
            exposure = (w_F[t] @ p.X[t0 + t][:, self.chars]).numpy()         # (K, chars)
            row = pd.DataFrame(exposure, columns=[p.features[i] for i in self.chars])
            row.insert(0, "top10", top10[t].numpy())
            row.insert(0, "eff_n", eff_n[t].numpy())
            row.insert(0, "factor", np.arange(K, dtype=np.int16))
            row.insert(0, "date", dates[t])
            self.fchars.append(row)

        l1 = out["l1"]
        scaled = out["w_port"] / torch.where(l1 > 0, l1, torch.ones_like(l1)).unsqueeze(1)
        self.port.append(to_pool(scaled, idx, p.n_pool).numpy())
        self.train += [{"year": year, **e} for e in train]
        self.year_stats[str(year)] = {"test_ev": test_ev, "tau": tau}

    def write(self, run_dir: Path, held: np.ndarray, t_idx: np.ndarray) -> None:
        """held: (T, n_pool) executed book of the days t_idx, all test years together."""
        p, R, new = self.p, self.p.R_pool[t_idx], self.new[t_idx]
        port = np.concatenate(self.port)
        zero = np.zeros((1, held.shape[1]), dtype=held.dtype)
        dw, dp = np.abs(np.diff(held, axis=0, prepend=zero)), np.abs(np.diff(port, axis=0, prepend=zero))
        daily = pd.concat(self.days, ignore_index=True)
        daily["n_new"] = ((held != 0) & new).sum(1)
        daily["l1_new"] = (np.abs(held) * new).sum(1)
        daily["pnl_new"] = (held * R * new).sum(1)
        daily["turn_new"] = (dw * new).sum(1)
        daily["l1_port"] = np.abs(port).sum(1)
        daily["pnl_port"] = (port * R).sum(1)
        daily["turn_port"] = dp.sum(1)
        for c in daily.columns.difference(["date"]):
            daily[c] = daily[c].astype("int16" if c.startswith("n_") else "float32")

        years, rows = p.dates[t_idx].year.to_numpy(), []
        for y in np.unique(years):
            m = years == y
            w = held[m]
            on = np.flatnonzero((w != 0).any(axis=0))
            rows.append(pd.DataFrame({
                "year": y.astype(np.int16), "sec_id": p.sec_ids[on], "days": (w[:, on] != 0).sum(0).astype(np.int16),
                "l1": np.abs(w[:, on]).sum(0), "net": w[:, on].sum(0), "pnl": (w[:, on] * R[m][:, on]).sum(0),
                "turnover": dw[m][:, on].sum(0), "short": np.clip(-w[:, on], 0, None).sum(0)}))

        fret = pd.concat(self.fret).rename_axis("date").reset_index()
        tables = {"diag_daily": daily, "factor_returns": fret, "factor_top": pd.concat(self.top, ignore_index=True),
                  "factor_chars": pd.concat(self.fchars, ignore_index=True),
                  "name_year": pd.concat(rows, ignore_index=True)}
        for name in FILES:
            tables[name].to_parquet(run_dir / f"{name}.parquet", index=False, compression="zstd")
        pd.DataFrame(self.train).to_csv(run_dir / "train_log.csv", index=False, float_format="%.5g")
