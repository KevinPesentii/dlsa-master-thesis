"""The parametric benchmark (Table 2, "PCA + OU Thresh"): signal against GPZ's own code, OU
parameters against a simulated AR(1), the allocation against the cases of GPZ Section III."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from afe.policy.ou_threshold import ou_signal, threshold_weights
from afe.policy.trading import make_windows

GPZ_PREPROCESS = Path(__file__).resolve().parents[2] / "preprocess.py"   # dlsa-public clone root


def ar1_path(a=0.2, b=0.8, sd=0.1, n=100_000, seed=1):
    """x_{l+1} = a + b x_l + e: an OU process sampled daily, mu = a / (1 - b) = 1."""
    e = np.random.default_rng(seed).normal(0.0, sd, n)
    x = np.empty(n)
    x[0] = a / (1 - b)
    for i in range(1, n):
        x[i] = a + b * x[i - 1] + e[i]
    return x, a / (1 - b), sd / np.sqrt(1 - b * b)


def test_signal_is_gpz_preprocess_ou():
    if not GPZ_PREPROCESS.exists():
        pytest.skip("GPZ clone not present")
    spec = importlib.util.spec_from_file_location("gpz_preprocess", GPZ_PREPROCESS)
    gpz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpz)

    rng = np.random.default_rng(0)
    T, N, L = 120, 50, 30
    resid = rng.normal(0.0, 0.01, (T, N))
    resid[:, :5] += 0.1 * (-1.0) ** np.arange(T)[:, None]      # zig-zag paths: b < 0, not traded
    out, selected = gpz.preprocess_ou(resid, L)                # (T-L, N, 4): x_L, mu, sigma, R2
    x_L, mu, sigma, R2 = np.moveaxis(out.astype(np.float64), -1, 0)

    # the runner's windows (float32) are GPZ's cumulative paths x_1..x_L ending the day before
    X, tradable = make_windows(resid, np.tile(np.arange(N), (T, 1)), np.full(T, N), L, T, L, 1.0, True)
    X64 = np.stack([np.cumsum(resid[t - L:t], axis=0).T for t in range(L, T)])
    assert tradable.all()
    np.testing.assert_allclose(X, X64, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(X64[..., -1], x_L, rtol=1e-6)

    s, r2, ok = ou_signal(X64)
    np.testing.assert_array_equal(ok, selected.numpy())
    assert ok.any() and not ok[:, :5].any()
    np.testing.assert_allclose(r2[ok], R2[ok], rtol=1e-5, atol=1e-7)
    s_gpz = (x_L - mu) / sigma                                 # GPZ store mu and sigma as float32
    np.testing.assert_allclose(s[ok], s_gpz[ok], rtol=1e-5, atol=1e-5)
    rule = np.where(selected.numpy() & (R2 > 0.25), -np.sign(s_gpz) * (np.abs(s_gpz) > 1.25), 0.0)
    np.testing.assert_array_equal(threshold_weights(X, tradable), rule)
    # s and R2 are ratios: the units of the residuals do not matter
    np.testing.assert_array_equal(threshold_weights(100 * X, tradable), threshold_weights(X, tradable))


def test_s_score_uses_the_ou_mean_and_stationary_sd_of_appendix_b():
    x, mu, sd_eq = ar1_path()
    for level in (-2.0, -1.0, 0.0, 1.0, 2.0):
        x[-1] = mu + level * sd_eq
        s, r2, ok = ou_signal(x[None])
        assert ok[0] and abs(s[0] - level) < 0.05
    assert abs(r2[0] - 0.8 ** 2) < 0.01          # corr(x_l, x_{l+1}) = b for a stationary AR(1)


def test_threshold_rule_of_section_iii():
    x, mu, sd_eq = ar1_path()

    def weight(level, c_crit=0.25, tradable=True):
        y = x.copy()
        y[-1] = mu + level * sd_eq
        return threshold_weights(y[None], np.array([tradable]), 1.25, c_crit)[0]

    assert weight(2.0) == -1                     # far above the mean: short the residual
    assert weight(-2.0) == 1                     # far below: long
    assert weight(1.0) == 0 and weight(-1.0) == 0          # inside the band: flat, no memory
    assert weight(2.0, c_crit=0.7) == 0          # R2 = b^2 = 0.64: the fit is not trusted
    assert weight(2.0, tradable=False) == 0
    zigzag = np.cumsum(0.02 * (-1.0) ** np.arange(30)) + np.random.default_rng(2).normal(0, 1e-3, 30)
    assert not ou_signal(zigzag[None])[2][0]     # b < 0: not an OU process, never traded
    assert threshold_weights(zigzag[None], np.array([True]))[0] == 0
