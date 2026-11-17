"""Re-scoring saved weights (evaluation/robustness.py).

The toy numbers are worked out by hand. The reproduction test books a real run's weights.parquet again
and asks for its own metrics.json; it needs the run and the data, so it is skipped elsewhere
(AFE_ROBUSTNESS_RUN=<run dir> and AFE_ROBUSTNESS_DATA=<data dir> point it at one).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afe.evaluation import robustness as rb


def toy_market(closed=None) -> rb.Market:
    dates = pd.bdate_range("2020-01-01", periods=4)
    R = np.array([[0.00, 0.00], [0.01, -0.02], [0.02, 0.00], [-0.01, 0.03]])
    return rb.Market(dates, pd.Index(["a", "b"]), R, closed, pd.DataFrame())


def test_book_by_hand():
    m = toy_market()
    tgt = np.array([[0, 0], [0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    t = np.array([1, 2, 3])
    H = rb.held(tgt, m, t, lag=0)
    d = rb.book(H, m.R[t], tc=5e-4, sc=1e-4)
    # day 1: buy a 0.5, short b 0.5; day 2 nothing; day 3 flip both legs: |dw| = 1 + 1
    assert np.allclose(d["turnover"], [1.0, 0.0, 2.0])
    assert np.allclose(d["short"], [0.5, 0.5, 0.5])
    assert np.allclose(d["gross"], [0.5 * 0.01 + 0.5 * 0.02, 0.5 * 0.02, 0.5 * 0.01 + 0.5 * 0.03])
    assert np.allclose(d["net"], d["gross"] - 5e-4 * d["turnover"] - 1e-4 * 0.5)
    # 20bp purchase tax on b only: day 1 shorts b (a sale, untaxed), day 3 covers and buys b: +1.0
    tax = (np.array([[0, 20e-4]] * 3), np.zeros((3, 2)))
    assert np.allclose(rb.book(H, m.R[t], 5e-4, 1e-4, tax_rate=tax)["tax"], [0.0, 0.0, 1.0 * 20e-4])
    # per-name cost: 10bp on a, 30bp on b
    c = np.array([[10e-4, 30e-4]] * 3)
    assert np.allclose(rb.book(H, m.R[t], 5e-4, 1e-4, cost_per_unit=c)["trade_cost"],
                       [0.5 * 10e-4 + 0.5 * 30e-4, 0.0, 1.0 * 10e-4 + 1.0 * 30e-4])


def test_delay_and_closed_market():
    tgt = np.array([[0, 0], [0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    t = np.array([1, 2, 3])
    H1 = rb.held(tgt, toy_market(), t, lag=1)
    # lag 1: the day-1 target is held from day 2 on; nothing is held on day 1
    assert np.allclose(H1, [[0, 0], [0.5, -0.5], [0.5, -0.5]])
    closed = np.zeros((4, 2), bool)
    closed[2, 0] = True          # a's market shut at the close before day 3: a keeps its day-2 position
    H = rb.held(tgt, toy_market(closed), t, lag=0)
    assert np.allclose(H, [[0.5, -0.5], [0.5, -0.5], [0.5, 0.5]])


def test_tax_threshold_and_dates():
    m = toy_market()
    m.rows = pd.DataFrame({"date": list(m.dates) * 2, "sec_id": ["a"] * 4 + ["b"] * 4,
                           "mktcap_lag": [900.0] * 4 + [2000.0] * 4})
    home = pd.Series({"a": "FR", "b": "FR"})
    cal = [{"country": "FR", "start": str(m.dates[2].date()), "end": None, "rate_bp": 30, "min_mktcap_eur_m": 1000}]
    buy, sell = rb.tax_rates(m, np.array([1, 2, 3]), home, cal)
    # row t is the trade at close t-1: only the trade at m.dates[2] (row 3) is on or after the start; a is too small
    assert np.allclose(buy, [[0, 0], [0, 0], [0, 30e-4]]) and not sell.any()


RUN, DATA = os.environ.get("AFE_ROBUSTNESS_RUN"), os.environ.get("AFE_ROBUSTNESS_DATA")


@pytest.mark.skipif(not (RUN and DATA and Path(RUN).exists()), reason="needs a saved run and its data")
def test_baseline_reproduces_stored_metrics():
    run = Path(RUN)
    st = json.loads((run / "metrics.json").read_text())
    m = rb.load_market(Path(DATA), pd.read_parquet(run / "weights.parquet", columns=["sec_id"])["sec_id"])
    t = rb.oos_index(m, st["first"], st["last"])
    s = rb.summarise(rb.book(rb.held(rb.targets(run, m), m, t, 0), m.R[t], 5e-4, 1e-4), 1e-4)
    assert s["SR"] == pytest.approx(st["gross"]["SR"], abs=1e-4)
    assert s["SR_net"] == pytest.approx(st["net"]["SR"], abs=1e-4)
    assert s["turnover"] == pytest.approx(st["turnover_daily"], abs=1e-5)
    assert s["short"] == pytest.approx(st["short_exposure"], abs=1e-5)
    assert s["break_even_bps"] == pytest.approx(st["break_even_turnover_cost_bps"], abs=1e-3)
