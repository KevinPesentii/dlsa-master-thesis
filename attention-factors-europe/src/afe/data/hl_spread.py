"""Corwin and Schultz (2012) high-low spread estimator, shared by both markets.

The characteristic Spread is the monthly mean of this two-day estimate in Europe (Compustat
Global has no bid/ask) and, since 2026-11-17, in the US too: CRSP's NYSE closing quotes are
several times too wide until a break in 2003, so a quote-based Spread ranked NYSE against
Nasdaq rather than wide against narrow (Thesis/diagnostics/2026-11-17_input_data_audit, B3).

Inputs are the high, low and close of two consecutive traded days of one line, on a common
price basis: a split between the two days would otherwise read as a two-day range of the
split ratio. Both builds divide by their cumulative adjustment factor (CRSP cumfacpr,
Compustat ajexdi); only ratios of prices enter, so the basis itself does not matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(h0, l0, h1, l1, c0) -> np.ndarray:
    """Two-day spread estimate from day 0 (high h0, low l0, close c0) and day 1 (h1, l1),
    dated on day 1. Overnight adjustment of their Section IV.A: if day 1's low is above day
    0's close, day 1's high and low move down by the gap; if its high is below it, up by the
    gap. Negative estimates are set to zero. NaN where a price is missing or not positive,
    or a low exceeds its high."""
    h0, l0, h1, l1, c0 = (np.asarray(x, dtype="float64") for x in (h0, l0, h1, l1, c0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (l0 > 0) & (l1 > 0) & (h0 >= l0) & (h1 >= l1) & np.isfinite(c0) & (c0 > 0)
        up_gap = np.nan_to_num(np.clip(l1 - c0, 0.0, None))
        dn_gap = np.nan_to_num(np.clip(c0 - h1, 0.0, None))
        h1a, l1a = h1 - up_gap + dn_gap, l1 - up_gap + dn_gap
        beta = np.log(h0 / l0) ** 2 + np.log(h1a / l1a) ** 2
        gamma = np.log(np.maximum(h0, h1a) / np.minimum(l0, l1a)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(ok, np.clip(s, 0.0, None), np.nan)


def by_line(d: pd.DataFrame, keys: list[str], high: str, low: str, close: str, traded: str,
            factor: str | None = None) -> pd.Series:
    """Corwin-Schultz per row of `d` (sorted by keys and date): this row and the line's
    previous row, both traded (`traded` True), prices divided by `factor` if given."""
    adj = (lambda c: d[c].astype("float64") / d[factor].astype("float64")) if factor else (lambda c: d[c].astype("float64"))
    h, l, c = adj(high), adj(low), adj(close)
    t = d[traded].astype("boolean").fillna(False).astype(bool).astype("float64")   # NA (no status): not traded
    g = pd.DataFrame({"h": h, "l": l, "c": c, "t": t}, index=d.index).groupby(
        [d[k] for k in keys], sort=False, observed=True)
    h0, l0, c0, t0 = g["h"].shift(1), g["l"].shift(1), g["c"].shift(1), g["t"].shift(1)
    s = corwin_schultz(h0, l0, h, l, c0)
    both = (t.to_numpy() == 1.0) & (t0.to_numpy() == 1.0)
    return pd.Series(np.where(both, s, np.nan), index=d.index)
