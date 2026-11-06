"""evaluation/diagnostics.py on panels small enough to work out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from afe.data.slots import SlotPanel
from afe.evaluation import diagnostics as dg
from afe.evaluation.diagnostics import Recorder, by_company, change, effective_factors, not_yet_rolling
from afe.model.attention_pipeline import AttentionArb


def panel(idx, in_universe, R_pool, X=None, features=("char_a", "char_b")):
    """R_pool (T, P); idx (T, S); slot arrays follow from them."""
    idx = torch.tensor(idx)
    in_universe = torch.tensor(in_universe)
    T, S = idx.shape
    R_pool = np.asarray(R_pool, dtype=np.float32)
    P = R_pool.shape[1]
    ext = np.c_[R_pool, np.zeros(T, dtype=np.float32)]
    R = torch.from_numpy(np.take_along_axis(ext, idx.numpy(), axis=1))
    X = torch.zeros(T, S, len(features)) if X is None else X
    dates = pd.date_range("2000-01-03", periods=T, freq="B")
    return SlotPanel(dates, np.array([f"n{j}" for j in range(P)]), list(features), X, R, in_universe, idx,
                     torch.zeros(T), R_pool, np.zeros(T), None, np.ones((T, P), dtype=bool), R_pool)


def test_not_yet_rolling_marks_the_first_L_days_in_the_universe():
    # name 1 enters on day 2. Lookback 2: it needs days 2 and 3 behind it, so day 4 is its first
    idx = [[0, 2], [0, 2], [0, 1], [0, 1], [0, 1]]
    iu = [[True, False]] * 2 + [[True, True]] * 3
    p = panel(idx, iu, np.zeros((5, 2)))
    new = not_yet_rolling(p, 2)
    assert new[:, 1].tolist() == [False, False, True, True, False]
    assert new[2:, 0].tolist() == [False, False, False]


def test_a_gap_in_membership_counts_again():
    # name 0 leaves on day 2 and returns on day 3: the rolling window needs two days again
    idx = [[0], [0], [1], [0], [0], [0]]
    iu = [[True], [True], [False], [True], [True], [True]]
    new = not_yet_rolling(panel(idx, iu, np.zeros((6, 1))), 2)
    assert new[:, 0].tolist() == [True, True, False, True, True, False]


def test_effective_factors_by_hand():
    same = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    assert effective_factors(same).item() == pytest.approx(1.0)
    apart = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert effective_factors(apart).item() == pytest.approx(2.0)
    # G = [[.25, .25], [.25, 1]]: tr 1.25, ||G||_F^2 = 1.1875
    mixed = torch.tensor([[[0.25] * 4, [1.0, 0.0, 0.0, 0.0]]])
    assert effective_factors(mixed).item() == pytest.approx(1.25 ** 2 / 1.1875)


def test_by_company_undoes_a_reshuffle_of_slots():
    w = torch.tensor([[[0.75, 0.25, 0.0]], [[0.25, 0.75, 0.0]]])          # (T=2, K=1, S=3)
    idx = torch.tensor([[0, 1, 3], [1, 0, 3]])                            # the slots swap companies
    assert change(by_company(w, idx, 2), 1)[1].sum().item() == 0.0


def test_change_is_blank_before_it_can_exist():
    x = torch.tensor([1.0, 3.0, 6.0])
    out = change(x, 1)
    assert torch.isnan(out[0]) and out[1:].tolist() == [2.0, 3.0]


def recorder_with_one_year():
    """Two names, then a third entering on day 2; one factor, K = 1, lookback 2."""
    idx = [[0, 1, 3]] * 2 + [[0, 1, 2]] * 2
    iu = [[True, True, False]] * 2 + [[True, True, True]] * 2
    R_pool = np.array([[0.01, 0.02, 0.0], [0.03, -0.01, 0.0], [0.02, 0.0, 0.04], [0.0, 0.01, -0.02]])
    X = torch.tensor([[[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]]] * 4)
    p = panel(idx, iu, R_pool, X)
    w_F = torch.tensor([[[0.75, 0.25, 0.0]], [[0.75, 0.25, 0.0]], [[0.5, 0.25, 0.25]], [[0.5, 0.25, 0.25]]])
    bT = w_F.transpose(1, 2) * 2.0
    w_port = torch.tensor([[0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, -0.25, -0.25], [0.5, -0.5, 0.0]])
    out = {"w_F": w_F, "betaT": bT, "eps": p.R * 0.5, "tradable": p.in_universe, "w_port": w_port * 2.0,
           "l1": torch.full((4,), 2.0)}
    rec = Recorder(p, 2, top_n=2)
    rec.add_year(out, 0, 4, 2000, [{"epoch": 1, "loss": -1.0, "sharpe_part": -0.5, "ev": 0.2}], 0.3, 1.5)
    return rec, p


def test_factor_return_is_the_weights_times_the_returns():
    rec, _ = recorder_with_one_year()
    f = rec.fret[0]
    assert f.shape == (4, 1)
    assert f.iloc[0, 0] == pytest.approx(0.75 * 0.01 + 0.25 * 0.02)
    assert f.iloc[2, 0] == pytest.approx(0.5 * 0.02 + 0.0 + 0.25 * 0.04)


def test_daily_factor_stats_by_hand():
    rec, _ = recorder_with_one_year()
    d = rec.days[0]
    assert d["n_member"].tolist() == [2, 2, 3, 3]
    assert d["eff_n"].iloc[0] == pytest.approx(1 / (0.75 ** 2 + 0.25 ** 2))
    assert d["dw_1d"].iloc[1] == pytest.approx(0.0)
    assert d["dw_1d"].iloc[2] == pytest.approx(0.25)                      # 0.25 moved from name 0 to name 2
    assert np.isnan(d["dw_1d"].iloc[0]) and d["dw_21d"].isna().all()
    assert d["r2_cs"].iloc[0] == pytest.approx(0.75)                      # eps = R / 2


def test_factor_top_keeps_only_real_weights_largest_first():
    rec, _ = recorder_with_one_year()
    t = rec.top[0]
    assert t["sec_id"].tolist() == ["n0", "n1"] and t["rank"].tolist() == [1, 2]
    assert t["w"].tolist() == [0.75, 0.25]


def test_factor_chars_is_the_weighted_mean_rank():
    rec, _ = recorder_with_one_year()
    row = rec.fchars[0].iloc[0]
    assert row["char_a"] == pytest.approx(0.75 * 0.5 - 0.25 * 0.5)
    assert row["char_b"] == 0.0


def test_write_adds_up(tmp_path):
    rec, p = recorder_with_one_year()
    held = np.array([[0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, -0.25, -0.25], [0.5, -0.5, 0.0]], dtype=np.float32)
    t_idx = np.arange(4)
    rec.write(tmp_path, held, t_idx)
    for f in dg.FILES:
        assert (tmp_path / f"{f}.parquet").exists()
    assert pd.read_csv(tmp_path / "train_log.csv")[["year", "epoch"]].values.tolist() == [[2000, 1]]
    d = pd.read_parquet(tmp_path / "diag_daily.parquet")
    n = pd.read_parquet(tmp_path / "name_year.parquet")
    gross = (held * p.R_pool).sum()
    assert n["pnl"].sum() == pytest.approx(gross, abs=1e-6)
    assert n["turnover"].sum() == pytest.approx(np.abs(np.diff(held, axis=0, prepend=0)).sum())
    assert n["short"].sum() == pytest.approx(np.clip(-held, 0, None).sum())
    # name 2 is new on days 2-3 (lookback 2): held only on day 2, at -0.25, return 0.04
    assert d["n_new"].tolist() == [2, 2, 1, 0]                            # days 0-1: the panel's own start
    assert d["pnl_new"].iloc[2] == pytest.approx((-0.25 * 0.04), abs=1e-7)
    assert d["l1_port"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert d["pnl_port"].iloc[0] == pytest.approx(0.5 * 0.01 - 0.5 * 0.02)


def test_a_rolling_run_has_no_new_names():
    """Whatever the rolling window calls tradable has L days of membership behind it."""
    idx = [[0, 2], [0, 2], [0, 1], [0, 1], [0, 1], [0, 1]]
    iu = [[True, False]] * 2 + [[True, True]] * 4
    R_pool = np.random.default_rng(0).normal(0, 0.01, (6, 2))
    p = panel(idx, iu, R_pool, torch.rand(6, 2, 2))
    m = AttentionArb(n_features=2, n_factors=1, embedding_dim=2, hidden=4, lookback=2)
    with torch.no_grad():
        out = m.forward_span(p.X, p.R, p.in_universe, p.idx, p.n_pool)
    new = np.c_[not_yet_rolling(p, 2), np.zeros(6, dtype=bool)]
    assert out["tradable"].any()
    assert not (out["tradable"].numpy() & np.take_along_axis(new, p.idx.numpy(), axis=1)[2:]).any()
