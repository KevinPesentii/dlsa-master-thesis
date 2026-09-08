import pandas as pd
import pytest

from afe.data import synthetic

UNIVERSE_SIZE = 40
N_CHARS = 4


@pytest.fixture(scope="session")
def returns() -> pd.DataFrame:
    return synthetic.make_returns(n_days=500, n_names=60, seed=0)


@pytest.fixture(scope="session")
def universe(returns) -> pd.DataFrame:
    return synthetic.make_universe(returns, size=UNIVERSE_SIZE)


@pytest.fixture(scope="session")
def features(returns, universe) -> pd.DataFrame:
    return synthetic.make_features(returns, universe, n_chars=N_CHARS)
