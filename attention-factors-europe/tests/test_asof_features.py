"""Stage 2c and the loader's timing "asof": the first trading day of a month.

A small panel written out by hand, so every expected number is visible. Three names, five
trading days, universe {A, B} in January and {B, C} in February. C enters on 02-01.

    daily characteristic   10 * day + code(name), day 1 = 1999-12-31 ... day 6 = 2000-02-02
    monthly characteristic end 1999-12: A 1, B 2, C 3     end 2000-01: A 6, B 4, C 5

So on 02-01 a member must show the daily value of 01-05 (day 4) and the monthly value of the
end of January, ranked among {B, C}. Ranks of two names are 0.0 and 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from afe import schemas
from afe.data import asof, slots
from afe.data.build_us import assemble_year
from afe.model.attention_factors import AttentionFactors
from afe.model.attention_pipeline import AttentionArb

DAYS = pd.to_datetime(["1999-12-31", "2000-01-03", "2000-01-04", "2000-01-05", "2000-02-01", "2000-02-02"])
TRADED = DAYS[1:]
CODE = {"A": 1.0, "B": 2.0, "C": 3.0}
JAN, FEB = pd.Period("2000-01", "M"), pd.Period("2000-02", "M")
MEMBERS = {JAN: ["A", "B"], FEB: ["B", "C"]}
RF = pd.Series([(i + 1) / 1e4 for i in range(len(TRADED))], index=TRADED)


def dc_table():
    rows = [(t, s, 10.0 * (i + 1) + CODE[s]) for i, t in enumerate(DAYS) for s in "ABC"]
    return pd.DataFrame(rows, columns=["date", "permno", "Ret_D1"]).set_index(["date", "permno"]).sort_index()


def mc_table():
    vals = {("1999-12", "A"): 1.0, ("1999-12", "B"): 2.0, ("1999-12", "C"): 3.0,
            ("2000-01", "A"): 6.0, ("2000-01", "B"): 4.0, ("2000-01", "C"): 5.0}
    idx = pd.MultiIndex.from_tuples([(pd.Period(m, "M"), s) for m, s in vals], names=["month", "permno"])
    return pd.DataFrame({"BEME": list(vals.values())}, index=idx)


def build(dc, mc):
    lagged = asof.lag_daily(dc, DAYS)
    feat, _ = assemble_year(2000, MEMBERS, TRADED, mc, lagged, ["Ret_D1", "BEME"], RF, JAN, FEB, levels=False)
    return feat


def row(feat, day, sec):
    r = feat[(feat["date"] == pd.Timestamp(day)) & (feat["sec_id"] == sec)]
    assert len(r) == 1
    return r.iloc[0]


def test_lag_daily_moves_a_value_to_the_next_trading_day():
    out = asof.lag_daily(dc_table(), DAYS)
    assert out.loc[(pd.Timestamp("2000-02-01"), "B"), "Ret_D1"] == 42.0     # day 4, 01-05
    assert out.loc[(pd.Timestamp("2000-01-03"), "A"), "Ret_D1"] == 11.0     # day 1, 1999-12-31
    assert pd.Timestamp("1999-12-31") not in out.index.get_level_values("date")


def test_first_day_of_the_month_uses_last_months_end_and_yesterdays_close():
    feat = build(dc_table(), mc_table())
    assert sorted(c for c in feat.columns if c.startswith("char_")) == ["char_BEME", "char_Ret_D1"]
    assert not [c for c in feat.columns if c.startswith("med_")]
    # 02-01, {B, C}: end of January B 4 < C 5, 01-05 B 42 < C 43
    assert (row(feat, "2000-02-01", "B")["char_BEME"], row(feat, "2000-02-01", "B")["char_Ret_D1"]) == (0.0, 0.0)
    assert (row(feat, "2000-02-01", "C")["char_BEME"], row(feat, "2000-02-01", "C")["char_Ret_D1"]) == (0.5, 0.5)
    # 01-03, {A, B}: end of December A 1 < B 2
    assert row(feat, "2000-01-03", "A")["char_BEME"] == 0.0
    assert row(feat, "2000-01-03", "B")["char_BEME"] == 0.5


def test_a_rank_is_over_the_universe_of_the_row_not_of_the_day_before():
    """A is put far below B on 01-05. Among last month's {A, B}, B would rank 0.5 on 02-01;
    among that day's {B, C} it ranks 0.0, and A, gone, must not change it."""
    dc = dc_table()
    dc.loc[(pd.Timestamp("2000-01-05"), "A"), "Ret_D1"] = -1000.0
    assert row(build(dc, mc_table()), "2000-02-01", "B")["char_Ret_D1"] == 0.0


def test_a_row_uses_nothing_dated_on_its_own_day():
    """Everything dated from the row's date on removed, and the row does not move."""
    day = pd.Timestamp("2000-02-01")
    dc, mc = dc_table(), mc_table()
    full = build(dc, mc)
    cut_dc = dc[dc.index.get_level_values("date") < day]
    cut_mc = mc[mc.index.get_level_values("month") < FEB]
    cut = asof.lag_daily(cut_dc, DAYS)
    feat, _ = assemble_year(2000, {FEB: MEMBERS[FEB]}, TRADED[TRADED <= day], cut_mc, cut, ["Ret_D1", "BEME"],
                            RF, FEB, FEB, levels=False)
    for sec in MEMBERS[FEB]:
        assert (row(feat, day, sec)[["char_Ret_D1", "char_BEME", "rf"]]
                == row(full, day, sec)[["char_Ret_D1", "char_BEME", "rf"]]).all()


def test_changing_the_days_own_values_leaves_the_row_alone():
    dc = dc_table()
    dc.loc[(pd.Timestamp("2000-02-01"), "B"), "Ret_D1"] = 1e6
    assert row(build(dc, mc_table()), "2000-02-01", "B")["char_Ret_D1"] == 0.0


def write_panel(d, feat):
    days = {t: i + 1 for i, t in enumerate(TRADED)}
    ret = pd.DataFrame([(t, s, 0.01 * days[t] * CODE[s], 100.0) for t in TRADED for s in "ABC"
                        if not (s == "C" and t < pd.Timestamp("2000-01-05"))],
                       columns=["date", "sec_id", "ret", "mktcap_lag"])
    ret[["ret", "mktcap_lag"]] = ret[["ret", "mktcap_lag"]].astype("float32")
    uni = pd.DataFrame([(TRADED[0], "A", 1), (TRADED[0], "B", 2), (TRADED[3], "B", 1), (TRADED[3], "C", 2)],
                       columns=["month", "sec_id", "cap_rank"])
    uni["cap_rank"] = uni["cap_rank"].astype("int16")
    ret.to_parquet(d / "returns.parquet", index=False)
    uni.to_parquet(d / "universe.parquet", index=False)
    feat.to_parquet(d / "features_asof.parquet", index=False)
    # the old table: rows dated by the daily block, monthly lagged inside the row, with medians
    old = feat.copy()
    old["date"] = old["date"].map(dict(zip(TRADED[:-1], TRADED[1:]))).where(old["date"] != TRADED[0], pd.NaT)
    old = old.dropna(subset=["date"]).assign(med_BEME=1.0, med_Ret_D1=1.0)
    old[["med_BEME", "med_Ret_D1"]] = old[["med_BEME", "med_Ret_D1"]].astype("float32")
    old.to_parquet(d / "features.parquet", index=False)


def history_row(d, day="2000-01-04"):
    h = pd.DataFrame({"date": [pd.Timestamp(day)], "sec_id": ["C"], "ret": [0.5]})
    h["ret"] = h["ret"].astype("float32")
    h.to_parquet(d / "returns_history.parquet", index=False)


@pytest.fixture
def panel_dir(tmp_path):
    write_panel(tmp_path, build(dc_table(), mc_table()))
    return tmp_path


def load(d, **kw):
    return slots.load_slots(d, "2000-01-01", "2000-02-28", timing="asof", medians=False, rf_input=False, **kw)


def slot_of(p, t, sec):
    j = int(np.flatnonzero(p.sec_ids == sec)[0])
    return int(torch.nonzero(p.idx[t] == j)[0])


def test_asof_loader_reads_the_row_of_the_date_for_the_same_company(panel_dir):
    p = load(panel_dir)
    assert p.features == ["char_BEME", "char_Ret_D1"]              # 39 in the real table: rf is not an input
    t = int(p.dates.get_loc(pd.Timestamp("2000-02-01")))
    assert p.X[t, slot_of(p, t, "C")].tolist() == [0.5, 0.5]       # the entrant has real values
    assert p.X[t, slot_of(p, t, "B")].tolist() == [0.0, 0.0]


def test_rf_stays_the_rate_of_the_day_without_being_an_input(panel_dir):
    assert torch.allclose(load(panel_dir).rf, torch.tensor(RF.to_numpy(), dtype=torch.float32))


def test_asof_needs_the_medians_switched_off(panel_dir):
    with pytest.raises(ValueError, match="med_"):
        slots.load_slots(panel_dir, "2000-01-01", "2000-02-28", timing="asof")


def test_medians_and_rf_can_be_left_out_of_the_lagged_table(panel_dir):
    a = slots.load_slots(panel_dir, "2000-01-01", "2000-02-28")
    b = slots.load_slots(panel_dir, "2000-01-01", "2000-02-28", medians=False, rf_input=False)
    assert a.features == ["char_BEME", "char_Ret_D1", "med_BEME", "med_Ret_D1", "rf"]
    assert b.features == ["char_BEME", "char_Ret_D1"]
    assert torch.equal(a.X[:, :, :2], b.X) and torch.equal(a.rf, b.rf)


def test_config_chooses_the_table(panel_dir):
    cfg = {"features": {"timing": "asof", "include_cross_sectional_median": False, "include_risk_free": False},
           "data": {}}
    assert slots.load_config(cfg, panel_dir, "2000-01-01", "2000-02-28").features == ["char_BEME", "char_Ret_D1"]
    old = {"features": {}, "data": {}}                              # every config before the change
    assert len(slots.load_config(old, panel_dir, "2000-01-01", "2000-02-28").features) == 5


def test_history_reaches_the_window_arrays_and_nothing_else(panel_dir):
    history_row(panel_dir)
    a, b = load(panel_dir), load(panel_dir, history=True)
    t, j = int(a.dates.get_loc(pd.Timestamp("2000-01-04"))), int(np.flatnonzero(a.sec_ids == "C")[0])
    assert not a.has_ret[t, j] and b.has_ret[t, j]
    assert a.R_win[t, j] == 0.0 and b.R_win[t, j] == 0.5
    assert b.R_pool[t, j] == 0.0                                    # the book is valued without it
    assert np.array_equal(a.R_pool, b.R_pool)
    assert torch.equal(a.in_universe, b.in_universe) and torch.equal(a.R, b.R)


def test_history_makes_an_entrant_tradable_on_its_first_day(panel_dir):
    """C lacks a return on 01-04 only. Lookback 3: on 02-01 the window is 01-03..01-05, and
    C misses 01-03 too, so give it that day as well to open the window."""
    h = pd.DataFrame({"date": pd.to_datetime(["2000-01-03", "2000-01-04"]), "sec_id": ["C", "C"], "ret": [0.1, 0.2]})
    h["ret"] = h["ret"].astype("float32")
    h.to_parquet(panel_dir / "returns_history.parquet", index=False)
    p = load(panel_dir, history=True)
    m = AttentionArb(n_features=2, n_factors=1, embedding_dim=2, hidden=4, lookback=3)
    with torch.no_grad():
        out = m.forward_span(p.X, p.R, p.in_universe, p.idx, p.n_pool, residual_window="fixed",
                             R_pool=torch.from_numpy(p.R_win), has_ret=torch.from_numpy(p.has_ret))
    t = int(p.dates.get_loc(pd.Timestamp("2000-02-01")))
    assert bool(out["tradable"][t - 3, slot_of(p, t, "C")])
    q = load(panel_dir)                                             # without the history it cannot trade yet
    with torch.no_grad():
        out = m.forward_span(q.X, q.R, q.in_universe, q.idx, q.n_pool, residual_window="fixed",
                             R_pool=torch.from_numpy(q.R_win), has_ret=torch.from_numpy(q.has_ret))
    assert not bool(out["tradable"][t - 3, slot_of(q, t, "C")])


def test_only_new_keeps_what_the_returns_table_lacks():
    ret = pd.DataFrame({"date": pd.to_datetime(["2000-01-03", "2000-01-04"]), "sec_id": ["1", "1"]})
    rows = pd.DataFrame({"date": pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-04", "2000-01-04", "1999-01-04"]),
                         "sec_id": [1, 2, 2, 3, 2], "ret": [0.1, 0.2, 0.2, np.nan, 0.3]})
    out = asof.only_new(rows, ret)
    assert out[["sec_id", "ret"]].values.tolist() == [["2", pytest.approx(0.2)]]   # in the table, NaN, off-calendar: out
    schemas.validate(out, schemas.RETURNS_HISTORY)


def test_europe_history_takes_the_line_that_entered():
    ret = pd.DataFrame({"date": pd.to_datetime(["2000-01-03", "2000-01-04"]), "sec_id": ["X", "X"]})
    detail = pd.DataFrame({"gvkey": ["Y", "Y"], "iid": ["01", "02"], "date": pd.to_datetime(["2000-01-04"] * 2),
                           "ret": [0.1, 0.9]})
    uni = pd.DataFrame({"month": pd.PeriodIndex(["2000-02", "2000-03"], freq="M"), "gvkey": ["Y", "Y"], "iid": ["01", "02"]})
    assert asof.europe_history(detail, uni, ret)["ret"].tolist() == [pytest.approx(0.1)]


def test_a_column_constant_across_names_cannot_move_the_factor_weights():
    """Equation (1): a column that is the same for every name adds a constant to a row of scores,
    which the softmax over names drops. So 39 characteristics give the same w_F as 79 features."""
    torch.manual_seed(0)
    T, S, K, d = 4, 6, 3, 8
    chars = torch.rand(T, S, 2) - 0.5
    levels = torch.tensor([3.4e7, 0.02, 7.0])[None, None, :] * torch.arange(1.0, T + 1)[:, None, None]
    X = torch.cat([chars, levels.expand(T, S, 3)], dim=-1)
    full, short = AttentionFactors(5, K, d), AttentionFactors(2, K, d)
    with torch.no_grad():
        short.Q.copy_(full.Q)
        short.W_K.copy_(full.W_K[:2])
    m = torch.ones(T, S, dtype=torch.bool)
    assert torch.allclose(full.factor_weights(X, m), short.factor_weights(chars, m), atol=1e-6)
