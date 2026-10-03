"""Execution timing: trading.execute turns the target weights of each day into the positions
actually held, for an execution lag and for markets that did not trade at a close. Worked
by hand: target[k] is decided at the close of day k-1 and earns day k when traded at once."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from afe.data import slots
from afe.policy import trading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_attention_us as base  # noqa: E402
import run_attention_us_val as val  # noqa: E402

# 4 days x 2 names; row k = the target for day k, values chosen so every cell is distinct
TARGET = torch.tensor([[0.5, -0.5], [0.4, -0.6], [0.3, -0.7], [0.2, -0.8]])


def test_no_lag_and_open_markets_hold_the_target():
    assert torch.equal(trading.execute(TARGET, None, 0), TARGET)


def test_lag_one_holds_yesterdays_target():
    """Decided at the close of k-1, traded at the close of k, so it earns day k+1; day 0 is flat."""
    held = trading.execute(TARGET, None, 1)
    assert torch.equal(held, torch.tensor([[0.0, 0.0], [0.5, -0.5], [0.4, -0.6], [0.3, -0.7]]))


def test_a_closed_market_keeps_the_old_position_for_one_day():
    """Name 0's exchange is shut at the close before day 2 (closes index t0-1, t0, t0+1, ...):
    the day-2 target cannot be traded, so name 0 holds its day-1 position over day 2 and
    catches up at the next close. Name 1 trades as usual."""
    closed = torch.zeros(4, 2, dtype=torch.bool)
    closed[2, 0] = True
    held = trading.execute(TARGET, closed, 0)
    assert torch.equal(held, torch.tensor([[0.5, -0.5], [0.4, -0.6], [0.4, -0.7], [0.2, -0.8]]))


def test_lag_and_two_closed_closes_combine():
    """Lag 1 and name 0 shut at the closes before days 1 and 2: at the close before day 3
    it trades what was decided one close earlier, target[2]."""
    closed = torch.zeros(4, 2, dtype=torch.bool)
    closed[1, 0] = closed[2, 0] = True
    held = trading.execute(TARGET, closed, 1)
    assert torch.equal(held[:, 0], torch.stack([torch.tensor(0.0)] * 3 + [TARGET[2, 0]]))
    assert torch.equal(held[1:, 1], TARGET[:3, 1]) and held[0, 1] == 0


def test_gradient_reaches_the_targets_that_are_held():
    t = TARGET.clone().requires_grad_(True)
    trading.execute(t, None, 1).sum().backward()
    assert t.grad.tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]


def test_lagged_span_uses_no_return_after_it(tmp_path, returns, universe, features):
    """Training with lag 1 on days t0..t1-1 must not see the return of t1 (the first test
    day when t1 is the window end): changing every return from t1 on leaves the loss alone."""
    for name, df in [("returns", returns), ("universe", universe), ("features", features)]:
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    p = slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))
    cfg = yaml.safe_load((ROOT / "configs" / "us_replication.yaml").read_text())
    cfg["policy"].update(residual_lookback=10, hidden=8)
    cfg["model"].update(embedding_dim=8)
    cfg["execution"]["lag"] = 1
    model = val.fresh_model(cfg, 3, len(p.features), 3)
    model.eval()
    t0, t1 = 20, 60
    with torch.no_grad():
        loss, parts, _ = base.span(model, p, t0, t1, cfg)
        p.R_pool[t1:] += 0.05
        loss2, parts2, _ = base.span(model, p, t0, t1, cfg)
    assert torch.equal(parts["net"], parts2["net"]) and float(loss) == float(loss2)
    assert float(parts["net"][0]) == 0.0                      # day t0 holds nothing yet


def test_daily_book_is_the_executed_book(tmp_path, returns, universe, features):
    """The out-of-sample book applies the same execution across the whole test period."""
    for name, df in [("returns", returns), ("universe", universe), ("features", features)]:
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    p = slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))
    t_idx = np.arange(5, 15)
    wp = np.random.default_rng(0).normal(size=(len(t_idx), p.n_pool)).astype(np.float32)
    cfg = {"execution": {"lag": 1}}
    d = base.daily_book(p, wp, t_idx, cfg)
    expect = np.r_[0.0, (wp[:-1] * p.R_pool[t_idx[1:]]).sum(axis=1)]
    assert np.allclose(d["gross"], expect, atol=1e-6)
    assert isinstance(d, pd.DataFrame) and len(d) == len(t_idx)
