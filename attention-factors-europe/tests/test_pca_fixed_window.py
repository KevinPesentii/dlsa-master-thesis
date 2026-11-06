"""policy.residual_window "fixed" on the PCA rows: the window of trade date t re-hedged with
the composition of date t. Numbers worked out by hand from the definition, the spec:

    eps~_{s,tau} = R_{s,tau} - B_{s,t}' V_t' D_t^-1 R_tau,   tau = t-L..t-1

Two names, one factor, L = 2. With V = (1, 0), 1/vol = (2, 1), B = (0.5, 1), the factor return
is twice name 0's return: 0.02 on the first day, 0.04 on the second. Name 0's residuals are
0.01 - 0.5 * 0.02 = 0 and 0.02 - 0.5 * 0.04 = 0. Name 1's are 0.03 - 0.02 = 0.01 and
-0.01 - 0.04 = -0.05, whose cumulative path is 0.01, -0.04.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from afe.model import pca_factors
from afe.policy import trading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pca_longconv as rp  # noqa: E402

R = np.array([[0.01, 0.03], [0.02, -0.01], [9.0, 9.0]], dtype=np.float32)
V = np.array([[[1.0], [0.0]]] * 3, dtype=np.float32)
B = np.array([[[0.5], [1.0]]] * 3, dtype=np.float32)
INV = np.array([[2.0, 1.0]] * 3, dtype=np.float32)
IDX = np.array([[0, 1]] * 3)
N = np.array([2, 2, 2])
HAS = np.ones((3, 2), bool)


def fixed(R=R, has=HAS, V=V, B=B, idx=IDX, scale=1.0, cum=False, norm="none"):
    return trading.make_fixed_windows(R, has, idx, N, V, B, INV, 2, 3, 2, scale, cum, norm)


def test_every_entry_by_hand():
    X, ok = fixed()
    assert ok[0].all()
    assert X[0, 0] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert X[0, 1] == pytest.approx([0.01, -0.05])


def test_cumulative_and_scale_as_the_rolling_window():
    X, _ = fixed(scale=100.0, cum=True)
    assert X[0, 1] == pytest.approx([1.0, -4.0])


def test_the_slot_follows_the_name():
    X, _ = fixed(V=V[:, ::-1], B=B[:, ::-1], idx=IDX[:, ::-1])
    inv = trading.make_fixed_windows(R, HAS, IDX[:, ::-1], N, V[:, ::-1], B[:, ::-1], INV[:, ::-1], 2, 3, 2, 1.0, False)
    assert inv[0][0, 1] == pytest.approx([0.0, 0.0], abs=1e-7)   # name 0 now in slot 1
    assert inv[0][0, 0] == pytest.approx([0.01, -0.05])


def test_nothing_of_the_trade_date_or_later_enters():
    a, _ = fixed()
    R2 = R.copy()
    R2[2] = -5.0                                                 # the return being earned at t
    V2, B2 = V.copy(), B.copy()
    V2[:2] = 7.0                                                 # the compositions of the window days
    B2[:2] = 7.0
    b, _ = fixed(R=R2, V=V2, B=B2)
    assert a == pytest.approx(b)


def test_a_name_without_a_return_in_the_window_is_not_traded():
    has = HAS.copy()
    has[0, 1] = False
    Rm = R.copy()
    Rm[0, 1] = 0.0                                               # as Stage1.R_pool
    X, ok = fixed(R=Rm, has=has)
    assert ok[0].tolist() == [True, False]
    assert X[0, 1] == pytest.approx([0.0, 0.0])
    assert X[0, 0] == pytest.approx([0.0, 0.0], abs=1e-7)        # it counts as 0 in the factor


def test_a_constant_composition_gives_the_rolling_window():
    """With V, B and vol the same on every date, each day's own hedge is date t's hedge."""
    resid = R - (R * INV[0]) @ V[0] @ B[0].T
    a, _ = fixed()
    b, _ = trading.make_windows(resid, IDX, N, 2, 3, 2, 1.0, False)
    assert a == pytest.approx(b)


def test_stage1_on_a_built_panel():
    """The batch applies V, B and 1/vol of the date being traded, as built by stage one."""
    rng = np.random.default_rng(0)
    T, P, L = 60, 8, 5
    dates = pd.bdate_range("2000-01-03", periods=T)
    panel = pca_factors.Panel(dates, np.array([f"n{i}" for i in range(P)]), rng.normal(0, 0.01, (T, P)),
                              np.ones((T, P), bool), np.zeros(T))
    return_dir = Path(pytest.importorskip("tempfile").mkdtemp())
    pca_factors.build(panel, [2], 30, 30, return_dir, max_members=P, log=lambda s: None)
    s1 = rp.Stage1(return_dir, 2, panel)
    t = 50
    b = s1.batch(t, t + 1, L, 1.0, False, "none", "fixed")
    v, c, iv = np.array(s1.V[t]), np.array(s1.B[t]), s1.invvol[t]
    cols = s1.idx[t, :P]
    win = panel.R[t - L:t][:, cols]
    want = win - ((win * iv) @ v) @ c.T
    assert b.windows[0].numpy() == pytest.approx(want.T.astype(np.float32), abs=1e-6)
    assert b.tradable.all()
    roll = s1.batch(t, t + 1, L, 1.0, False, "none", "rolling")
    assert roll.windows[0, 0, -1].item() != pytest.approx(b.windows[0, 0, -1].item(), abs=1e-9)
    with pytest.raises(ValueError):
        s1.batch(t, t + 1, L, 1.0, False, "none", "both")


def test_every_v2_search_holds_the_fixed_window_and_cumulative_input():
    names = ["us_search_pca_v2", "us_search_pca_v2_lag1", "europe_search_pca_v2", "europe_search_pca_v2_lag1"]
    for n in names:
        s = yaml.safe_load((ROOT / "configs" / f"{n}.yaml").read_text())
        assert s["fixed"]["policy.residual_window"] == "fixed" and s["fixed"]["policy.input"] == "cumulative", n
        assert "policy.input" not in s["search"]["space"], n
        assert s["search"]["n_points"] == 64 and s["K"] == 30 and s["validation_years"] == 2, n
        assert ("execution.lag" in s["fixed"]) == n.endswith("lag1"), n
    for n in ["us_search_ou_v2", "europe_search_ou_v2"]:
        s = yaml.safe_load((ROOT / "configs" / f"{n}.yaml").read_text())
        assert s["fixed"]["policy.residual_window"] == "fixed", n
