"""evaluation.stats against values derived by hand.

iid normal daily returns with daily Sharpe s have Var(SR_d_hat) = (1 + s^2 / 2) / T (Lo 2002),
so the annualised standard error is sqrt(252 (1 + s^2 / 2) / T): with s = 0.1 and T = 200,000,
sqrt(252 * 1.005 / 200000) = 0.03559. The difference test is antisymmetric, and OLS recovers
y = 0.001 + 2 x (alpha 0.001 a day = 25.2% a year).
"""

import numpy as np
import pytest

from afe.evaluation import stats


def test_hac_without_lags_is_the_sample_covariance():
    u = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 0.0]])
    assert np.allclose(stats.hac(u, 0), np.cov(u.T, ddof=0))


def test_sharpe_se_matches_lo_for_iid_normal_returns():
    rng = np.random.default_rng(0)
    r = 0.01 * (0.1 + rng.standard_normal(200_000))
    assert stats.sharpe_se(r, lags=0) == pytest.approx(np.sqrt(252 * 1.005 / 200_000), rel=0.02)


def test_sharpe_diff_is_antisymmetric_and_zero_for_equal_series():
    rng = np.random.default_rng(1)
    a, b = 0.01 * (0.1 + rng.standard_normal(5000)), 0.01 * (0.05 + rng.standard_normal(5000))
    ab, ba = stats.sharpe_diff(a, b), stats.sharpe_diff(b, a)
    assert ab["diff"] == pytest.approx(-ba["diff"]) and ab["se"] == pytest.approx(ba["se"])
    assert ab["p"] == pytest.approx(ba["p"])
    assert stats.sharpe_diff(a, a + 1e-9 * rng.standard_normal(5000))["diff"] == pytest.approx(0, abs=1e-5)


def test_ols_recovers_alpha_and_slope():
    rng = np.random.default_rng(2)
    x = 0.01 * rng.standard_normal(20_000)
    y = 0.001 + 2 * x + 1e-5 * rng.standard_normal(20_000)
    out = stats.ols_nw(y, x[:, None], ["mkt"])
    assert out["alpha_ann_pct"] == pytest.approx(25.2, abs=0.01)
    assert out["b_mkt"] == pytest.approx(2.0, abs=1e-3) and out["r2"] > 0.999
