"""The PCA runner's thesis-matrix options, numbers worked out by hand.

window_vol: residuals 0.01 and 0.03 have mean 0.02 and (population) sd 0.01, so the window
becomes [1, 3] and its cumulative path [1, 4], times input_scale 0.5 -> [0.5, 2].
Execution: a name whose market is closed at the close before day 2 keeps day 1's weight.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afe.model.pca_factors import Panel
from afe.policy import trading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pca_longconv as rp  # noqa: E402


def test_window_vol_divides_by_the_window_sd_before_the_cumulative_sum():
    resid = np.array([[0.01], [0.03], [0.0]])
    X, ok = trading.make_windows(resid, np.array([[0]] * 3), np.array([1, 1, 1]), 2, 3, 2, 0.5, True, "window_vol")
    assert ok[0, 0]
    assert X[0, 0] == pytest.approx([0.5, 2.0])
    X0, _ = trading.make_windows(resid, np.array([[0]] * 3), np.array([1, 1, 1]), 2, 3, 2, 1.0, True)
    assert X0[0, 0] == pytest.approx([0.01, 0.04])          # "none" is unchanged


def test_closed_name_keeps_its_position():
    dates = pd.bdate_range("2000-01-03", periods=4)
    closed = np.zeros((4, 2), dtype=bool)
    closed[1, 0] = True                                     # name 0 closed at the close of day 1
    panel = Panel(dates, np.array(["a", "b"]), np.zeros((4, 2)), np.ones((4, 2), bool), np.zeros(4), closed)
    wp = np.array([[0.5, -0.5], [0.2, -0.2], [0.1, -0.1]], dtype=np.float32)   # targets for days 1..3
    cfg = {"execution": {"lag": 0, "stale_when_closed": True}}
    held = rp.execute(wp, panel, np.arange(1, 4), cfg)
    assert held[:, 0] == pytest.approx([0.5, 0.5, 0.1])     # day 2 traded at the closed close of day 1
    assert held[:, 1] == pytest.approx([-0.5, -0.2, -0.1])
    assert rp.execute(wp, panel, np.arange(1, 4), {"execution": {"stale_when_closed": False}}) is wp
