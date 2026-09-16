"""The no-lookahead harness.

Point-in-time correctness is the first-order risk in this thesis, and it is invisible
in a code review: leaked code runs, produces plausible numbers, and inflates the Sharpe.
The only reliable check is behavioural. If a builder is causal, then rebuilding it from
data truncated at t must leave the values at t untouched.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def assert_prefix_invariant(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    cut_dates,
    key: tuple[str, ...] = ("date", "sec_id"),
    date_col: str = "date",
    rtol: float = 1e-6,
) -> None:
    """Rebuild from truncated inputs and compare the rows dated at the cut.

    Raises AssertionError naming the first column that moved, which is usually enough
    to find the leak.
    """
    full = build(raw)
    for t in pd.to_datetime(list(cut_dates)):
        truncated = build(raw[raw[date_col] <= t])
        a = full[full[date_col] == t].set_index(list(key)).sort_index()
        b = truncated[truncated[date_col] == t].set_index(list(key)).sort_index()

        if len(b) == 0:
            raise AssertionError(f"no rows at cut {t.date()} after truncation; widen the sample")
        if not a.index.equals(b.index):
            missing = a.index.difference(b.index)
            extra = b.index.difference(a.index)
            raise AssertionError(
                f"cut {t.date()}: row set changed under truncation "
                f"(missing {len(missing)}, extra {len(extra)})"
            )

        for col in a.columns:
            if not np.issubdtype(a[col].dtype, np.number):
                continue
            x, y = a[col].to_numpy(), b[col].to_numpy()
            if not np.allclose(x, y, rtol=rtol, atol=0, equal_nan=True):
                worst = int(np.nanargmax(np.abs(x - y)))
                raise AssertionError(
                    f"LOOKAHEAD: column {col!r} at {t.date()} changes when data after "
                    f"{t.date()} is removed (worst row {a.index[worst]}: "
                    f"{x[worst]:.8g} -> {y[worst]:.8g})"
                )
