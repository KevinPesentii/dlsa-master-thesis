"""The slot loader is where a lookahead would enter the one-step model.

test_x_is_yesterdays_row_of_the_same_company is the one that matters: X[t] must be
built from features dated t-1, looked up by company, and must contain nothing dated t.
The tiny panel is written out by hand so every expected number is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from afe.data import slots

JAN = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"])
FEB = pd.to_datetime(["2000-02-01", "2000-02-02"])
DAYS = JAN.append(FEB)
CODE = {"A": 1.0, "B": 2.0, "C": 3.0}


def write_panel(d):
    """Universe of 2: {A, B} in January, {B, C} in February. B has no return on 01-04.
    Features: char_a = 10*day + code(name), med_a = day, rf = day/1e4 (day = 1..5)."""
    days = {t: i + 1 for i, t in enumerate(DAYS)}
    members = {t: (["A", "B"] if t in JAN else ["B", "C"]) for t in DAYS}
    ret = pd.DataFrame([(t, s, 0.01 * days[t] * CODE[s], 100.0) for t in DAYS for s in "ABC"
                        if not (s == "B" and t == pd.Timestamp("2000-01-04"))],
                       columns=["date", "sec_id", "ret", "mktcap_lag"])
    for c in ("ret", "mktcap_lag"):
        ret[c] = ret[c].astype("float32")
    uni = pd.DataFrame([(pd.Timestamp("2000-01-03"), "A", 1), (pd.Timestamp("2000-01-03"), "B", 2),
                        (pd.Timestamp("2000-02-01"), "B", 1), (pd.Timestamp("2000-02-01"), "C", 2)],
                       columns=["month", "sec_id", "cap_rank"])
    uni["cap_rank"] = uni["cap_rank"].astype("int16")
    feat = pd.DataFrame([(t, s, 10 * days[t] + CODE[s], float(days[t]), days[t] / 1e4)
                         for t in DAYS for s in members[t]],
                        columns=["date", "sec_id", "char_a", "med_a", "rf"])
    for c in ("char_a", "med_a", "rf"):
        feat[c] = feat[c].astype("float32")
    for name, df in (("returns", ret), ("universe", uni), ("features", feat)):
        key = "month" if name == "universe" else "date"
        df.sort_values([key, "sec_id"]).to_parquet(d / f"{name}.parquet", index=False)


@pytest.fixture
def panel(tmp_path):
    write_panel(tmp_path)
    return slots.load_slots(tmp_path, "2000-01-01", "2000-02-28")


def slot_of(p, t, sec):
    j = int(np.flatnonzero(p.sec_ids == sec)[0])
    s = torch.nonzero(p.idx[t] == j)
    assert len(s) == 1, f"{sec} not in a slot on {p.dates[t].date()}"
    return int(s[0])


def test_x_is_yesterdays_row_of_the_same_company(panel):
    p = panel
    assert p.features == ["char_a", "med_a", "rf"]
    # 01-04, A: features of A on 01-03 (day 1): char 11, med 1, rf 1e-4
    assert torch.allclose(p.X[1, slot_of(p, 1, "A")], torch.tensor([11.0, 1.0, 1e-4]))
    # 02-02, C: features of C on 02-01 (day 4)
    assert torch.allclose(p.X[4, slot_of(p, 4, "C")], torch.tensor([43.0, 4.0, 4e-4]))
    # 02-01, B: B was a member on 01-05 (day 3), so its own row carries across the month end
    assert torch.allclose(p.X[3, slot_of(p, 3, "B")], torch.tensor([32.0, 3.0, 3e-4]))
    # nothing dated t appears in X[t]
    for t in range(len(p.dates)):
        assert not (p.X[t, :, 0] == 10 * (t + 1) + torch.tensor([CODE[s] for s in "ABC"])[:, None]).any()


def test_entry_day_gets_median_characteristics_and_the_dates_medians(panel):
    p = panel
    # 02-01, C: not a member on 01-05, so char -> 0 and med/rf -> the 01-05 cross-section (day 3)
    assert torch.allclose(p.X[3, slot_of(p, 3, "C")], torch.tensor([0.0, 3.0, 3e-4]))
    assert torch.equal(p.X[0], torch.zeros_like(p.X[0]))       # first day has no yesterday


def test_member_without_a_return_is_masked_but_keeps_its_slot(panel):
    p = panel
    s = slot_of(p, 1, "B")                                       # idx still points at B
    assert not p.in_universe[1, s] and p.R[1, s] == 0.0
    assert p.in_universe.sum() == 2 * len(DAYS) - 1
    assert p.R[2, slot_of(p, 2, "B")] == pytest.approx(0.01 * 3 * 2)
    assert p.mkt_ew[1] == pytest.approx(0.01 * 2 * 1)             # only A on 01-04


def test_rf_is_the_traded_days_rate_and_pool_columns_are_stable(panel):
    p = panel
    assert torch.allclose(p.rf, torch.tensor([1e-4, 2e-4, 3e-4, 4e-4, 5e-4]))
    assert slot_of(p, 0, "B") == 1 and slot_of(p, 3, "B") == 0     # B moves slot at the month end...
    jB = int(np.flatnonzero(p.sec_ids == "B")[0])
    assert p.idx[0, 1] == jB == p.idx[3, 0]                         # ...its pool column does not
    assert p.R_pool.shape == (len(DAYS), 3) and p.R_pool[1, jB] == 0.0


def test_synthetic_fixture_round_trips_by_company(tmp_path, returns, universe, features):
    """On the schema fixture: X[t, s] equals the features row dated t-1 of the company in
    slot s, for every slot that has one."""
    returns.to_parquet(tmp_path / "returns.parquet", index=False)
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    p = slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))
    f = features.set_index(["date", "sec_id"])[p.features]
    rng = np.random.default_rng(0)
    checked = 0
    for t in rng.integers(1, len(p.dates), 40):
        for s in rng.integers(0, p.idx.shape[1], 5):
            j = int(p.idx[t, s])
            key = (p.dates[t - 1], p.sec_ids[j])
            if j < p.n_pool and key in f.index:
                assert np.allclose(p.X[t, s].numpy(), f.loc[key].to_numpy(dtype=np.float32))
                checked += 1
    assert checked > 100
    assert p.in_universe.sum(dim=1).max() == universe.groupby("month").size().max()
