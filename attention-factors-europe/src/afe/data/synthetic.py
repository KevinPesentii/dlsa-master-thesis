"""A fake panel that honours docs/schemas.md.

Purpose: the model, policy and evaluation code is written and tested against this while
the real WRDS build is still being written, so the two halves of the project can proceed
in parallel. It is a shape and a set of invariants, not a simulation of equity returns.
Nothing in the thesis is ever computed from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afe import schemas


def make_returns(
    n_days: int = 500, n_names: int = 60, seed: int = 0, start: str = "2000-01-03"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    ids = [f"S{i:04d}" for i in range(n_names)]

    # One market factor plus idiosyncratic noise, so residualisation has something to do.
    mkt = rng.normal(0.0003, 0.010, size=n_days)
    beta = rng.uniform(0.5, 1.5, size=n_names)
    idio = rng.normal(0.0, 0.015, size=(n_days, n_names))
    ret = mkt[:, None] * beta[None, :] + idio

    cap0 = np.exp(rng.normal(9.0, 1.0, size=n_names))
    cap = cap0[None, :] * np.cumprod(1.0 + ret, axis=0)

    df = pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_names),
            "sec_id": np.tile(ids, n_days),
            "ret": ret.reshape(-1).astype("float32"),
            "mktcap_lag": np.vstack([cap0[None, :], cap[:-1]]).reshape(-1).astype("float32"),
        }
    )
    df = df.sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(df, schemas.RETURNS)
    return df


def make_universe(returns: pd.DataFrame, size: int = 40) -> pd.DataFrame:
    """Top `size` names by market cap as of the END OF THE PRIOR MONTH.

    The lag is the point of this function. Membership for month m may not look at
    anything dated inside month m.
    """
    r = returns.copy()
    r["month"] = r["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    month_end_cap = r.groupby(["month", "sec_id"], as_index=False)["mktcap_lag"].last()

    months = sorted(month_end_cap["month"].unique())
    rows = []
    for prev, cur in zip(months[:-1], months[1:]):
        snap = month_end_cap[month_end_cap["month"] == prev]
        top = snap.nlargest(size, "mktcap_lag").reset_index(drop=True)
        rows.append(
            pd.DataFrame(
                {
                    "month": cur,
                    "sec_id": top["sec_id"].values,
                    "cap_rank": np.arange(1, len(top) + 1, dtype="int16"),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True).sort_values(["month", "sec_id"], ignore_index=True)
    schemas.validate_universe(out, size=size)
    return out


def _rank_quantile(s: pd.Series) -> pd.Series:
    """Cross-sectional rank mapped to [-0.5, 0.5], per Chen, Pelger & Zhu.

    CONFIRM 1 in docs/schemas.md: the attention paper does not state the range. Verify
    against the authors' code before the real feature build uses this.
    """
    return s.rank(method="average", pct=True) - 0.5


def _build_features(
    returns: pd.DataFrame, universe: pd.DataFrame, n_chars: int, leak: bool
) -> pd.DataFrame:
    r = returns.sort_values(["sec_id", "date"]).copy()
    g = r.groupby("sec_id", sort=False)["ret"]

    raw = pd.DataFrame(index=r.index)
    windows = [5, 21, 63, 126, 252, 21]
    for j in range(n_chars):
        w = windows[j % len(windows)]
        if j % 2 == 0:
            raw[f"char_{j:02d}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        else:
            raw[f"char_{j:02d}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    raw["date"] = r["date"].values
    raw["sec_id"] = r["sec_id"].values

    char_cols = [c for c in raw.columns if c.startswith("char_")]

    if leak:
        # THE BUG: scale each name by its own FULL-SAMPLE volatility. Uses the future.
        vol = g.transform("std")
        for c in char_cols:
            raw[c] = raw[c] / vol.values

    raw = raw.dropna(subset=char_cols)

    # Restrict to the universe of the month the date falls in, before normalising.
    raw["month"] = raw["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    raw = raw.merge(universe[["month", "sec_id"]], on=["month", "sec_id"], how="inner")

    # Cross-sectional median of the RAW characteristic, broadcast to every name.
    # CONFIRM 3 in docs/schemas.md: universe median or all-listed median.
    meds = raw.groupby("date")[char_cols].transform("median")
    meds.columns = [c.replace("char_", "med_") for c in char_cols]

    norm = raw.groupby("date")[char_cols].transform(_rank_quantile)

    out = pd.concat([raw[["date", "sec_id"]], norm, meds], axis=1)
    out["rf"] = 0.0001

    value_cols = [c for c in out.columns if c not in ("date", "sec_id")]
    out[value_cols] = out[value_cols].astype("float32")
    out = out.sort_values(["date", "sec_id"], ignore_index=True)
    schemas.validate(out, schemas.FEATURES)
    return out


def make_features(
    returns: pd.DataFrame, universe: pd.DataFrame, n_chars: int = 6
) -> pd.DataFrame:
    """Causal characteristics: every value at date t uses only data up to and including t.

    tests/test_no_lookahead.py holds this to prefix invariance. The real builder must
    pass the same test.
    """
    return _build_features(returns, universe, n_chars=n_chars, leak=False)


def leaky_features(
    returns: pd.DataFrame, universe: pd.DataFrame, n_chars: int = 6
) -> pd.DataFrame:
    """Deliberately broken: scales each name by its full-sample volatility.

    A realistic bug. It never crashes, the output passes every schema check, and it
    quietly improves the Sharpe. Exists only so tests can prove the no-lookahead
    harness actually catches leakage. Never import this outside tests.
    """
    return _build_features(returns, universe, n_chars=n_chars, leak=True)
