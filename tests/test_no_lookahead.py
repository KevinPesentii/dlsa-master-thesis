"""Prefix invariance: rebuilding from truncated data must not move past values.

Every feature builder that reaches main is held to this, including the real WRDS one.
The second test checks the harness itself, because a check that cannot fail is worse
than no check at all.
"""

import pandas as pd

import pytest

from afe.data import synthetic
from afe.testing import assert_prefix_invariant

from conftest import N_CHARS, UNIVERSE_SIZE


def _causal_build(raw: pd.DataFrame) -> pd.DataFrame:
    uni = synthetic.make_universe(raw, size=UNIVERSE_SIZE)
    return synthetic.make_features(raw, uni, n_chars=N_CHARS)


def _leaky_build(raw: pd.DataFrame) -> pd.DataFrame:
    uni = synthetic.make_universe(raw, size=UNIVERSE_SIZE)
    return synthetic.leaky_features(raw, uni, n_chars=N_CHARS)


def _cut_dates(features: pd.DataFrame, n: int = 3) -> list:
    dates = sorted(features["date"].unique())
    span = len(dates)
    return [dates[span // 2], dates[2 * span // 3], dates[-1]][:n]


def test_feature_build_is_point_in_time(returns, features):
    assert_prefix_invariant(_causal_build, returns, _cut_dates(features))


def test_harness_catches_a_known_leak(returns, features):
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        assert_prefix_invariant(_leaky_build, returns, _cut_dates(features, n=1))
