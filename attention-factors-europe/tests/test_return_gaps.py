"""data.return_gaps: a member-day without a return inside a name's trading life (a halt).

Kevin's data check, item 2 (2026-11-17): masking such a day takes the name out of the day-t
weights, which are decided at the close of t-1, so the backtest knew in advance that the stock
would not trade. test_weights_up_to_a_halt_do_not_depend_on_it is that point as a test. The
tiny panel below is written out by hand so every expected cell is visible:

    days   Jan 3 4 5 6 7, Feb 1 2 3 (2000)
    A      member Jan + Feb, a return every day
    B      member Jan + Feb, no rows Jan 5-6 (halt), Jan 7 = +10% (the reopening return)
    C      member Jan, last row Jan 4 (delisted)
    E      member Jan only, no rows Jan 6 - Feb 1 (a halt across its last member month end),
           trades again Feb 2 (+20%)
    F      member Feb, first row Feb 2 (has not traded yet on Feb 1)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from afe.data import slots
from afe.data.return_gaps import halted
from afe.evaluation import robustness
from afe.model import pca_factors
from afe.model.attention_pipeline import to_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_attention_us as base  # noqa: E402
import run_attention_us_val as val  # noqa: E402
import run_pca_longconv as rp  # noqa: E402

JAN = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07"])
FEB = pd.to_datetime(["2000-02-01", "2000-02-02", "2000-02-03"])
DAYS = JAN.append(FEB)
D = {t.strftime("%m-%d"): i for i, t in enumerate(DAYS)}
MEMBERS = {"A": JAN.append(FEB), "B": JAN.append(FEB), "C": JAN, "E": JAN, "F": FEB}
NO_ROW = {"B": ["01-05", "01-06"], "C": ["01-05", "01-06", "01-07"] + list(FEB.strftime("%m-%d")),
          "E": ["01-06", "01-07", "02-01"], "F": ["01-03", "01-04", "01-05", "01-06", "01-07", "02-01"]}
SPECIAL = {("B", "01-07"): 0.10, ("E", "02-02"): 0.20}


def write_panel(d: Path, history: list[tuple[str, str]] | None = None) -> None:
    ret = [(t, s, SPECIAL.get((s, t.strftime("%m-%d")), 0.01), 100.0) for s in "ABCEF" for t in DAYS
           if t.strftime("%m-%d") not in NO_ROW.get(s, [])]
    ret = pd.DataFrame(ret, columns=["date", "sec_id", "ret", "mktcap_lag"]).astype({"ret": "float32", "mktcap_lag": "float32"})
    uni = pd.DataFrame([(pd.Timestamp("2000-01-03"), s, r) for r, s in enumerate("ABCE", 1)]
                       + [(pd.Timestamp("2000-02-01"), s, r) for r, s in enumerate("ABF", 1)],
                       columns=["month", "sec_id", "cap_rank"]).astype({"cap_rank": "int16"})
    feat = pd.DataFrame([(t, s, float(i), 0.0, 1e-4) for i, t in enumerate(DAYS) for s in "ABCEF" if t in MEMBERS[s]],
                        columns=["date", "sec_id", "char_a", "med_a", "rf"]).astype({c: "float32" for c in ("char_a", "med_a", "rf")})
    for name, df in (("returns", ret), ("universe", uni), ("features", feat)):
        df.sort_values(["month" if name == "universe" else "date", "sec_id"]).to_parquet(d / f"{name}.parquet", index=False)
    if history:
        h = pd.DataFrame([(pd.Timestamp(f"2000-{md}"), s, 0.0) for s, md in history], columns=["date", "sec_id", "ret"])
        h.astype({"ret": "float32"}).to_parquet(d / "returns_history.parquet", index=False)
    (d / "raw").mkdir(exist_ok=True)
    pd.DataFrame({"date": DAYS, "rf": 0.0}).to_parquet(d / "raw" / "ff_daily.parquet", index=False)


def cells(p, sec: str, days: list[str], what: str) -> list:
    """p.closed / p.R_pool / p.has_ret of `sec` (pool space) on `days`."""
    j = int(np.flatnonzero(p.sec_ids == sec)[0])
    return [getattr(p, what)[D[d], j].item() for d in days]


def member_cell(p, sec: str, day: str, what: str):
    """p.in_universe / p.R of `sec` in its slot on `day`."""
    j = int(np.flatnonzero(p.sec_ids == sec)[0])
    s = torch.nonzero(p.idx[D[day]] == j)
    assert len(s) == 1, f"{sec} not in a slot on {day}"
    return getattr(p, what)[D[day], int(s[0])].item()


@pytest.fixture
def closed(tmp_path):
    write_panel(tmp_path)
    return slots.load_slots(tmp_path, "2000-01-01", "2000-02-28", return_gaps="closed")


def test_masked_is_the_old_loader(tmp_path):
    write_panel(tmp_path)
    p = slots.load_slots(tmp_path, "2000-01-01", "2000-02-28")
    assert not member_cell(p, "B", "01-05", "in_universe") and not member_cell(p, "B", "01-06", "in_universe")
    assert p.closed is None                                      # no traded column, nothing closed


def test_a_halt_is_a_closed_zero_return_day_of_a_member(closed):
    p = closed
    for day in ("01-05", "01-06"):
        assert member_cell(p, "B", day, "in_universe") and member_cell(p, "B", day, "R") == 0.0
    assert cells(p, "B", ["01-04", "01-05", "01-06", "01-07"], "closed") == [False, True, True, False]
    assert cells(p, "B", ["01-05", "01-06"], "has_ret") == [True, True]     # zero returns, as a filled holiday
    assert cells(p, "B", ["01-07"], "R_pool") == [pytest.approx(0.10)]       # the reopening return stays
    assert not p.closed[:, int(np.flatnonzero(p.sec_ids == "A")[0])].any()


def test_after_the_last_return_and_before_the_first_stay_masked(closed):
    p = closed
    for day in ("01-05", "01-06", "01-07"):                                   # C delisted after 01-04
        assert not member_cell(p, "C", day, "in_universe")
    assert cells(p, "C", ["01-05", "01-06", "01-07"], "closed") == [False] * 3
    assert not member_cell(p, "F", "02-01", "in_universe")                    # F's first row is 02-02
    assert cells(p, "F", ["02-01"], "closed") == [False]


def test_a_halt_across_the_month_end_stays_closed_until_the_name_trades(closed):
    """E leaves the universe at the January month end while halted: its market stays closed on
    02-01 (not a member then), so a position held into the halt is not sold at a close at which
    E did not trade; it trades again, and can be sold, at the close of 02-02."""
    assert cells(closed, "E", ["01-05", "01-06", "01-07", "02-01", "02-02"], "closed") == [False, True, True, True, False]
    assert member_cell(closed, "E", "01-07", "in_universe")


def test_a_returns_history_row_ends_the_halt(tmp_path):
    write_panel(tmp_path, history=[("E", "02-01")])                       # E's other line trades on 02-01
    p = slots.load_slots(tmp_path, "2000-01-01", "2000-02-28", return_gaps="closed")
    assert cells(p, "E", ["01-07", "02-01"], "closed") == [True, False]


def test_the_book_holds_through_the_halt_and_earns_the_reopening_return(closed):
    """Targets for B: 0.5 up to 01-05, then 0 (the policy would exit). Decided at the close of 01-05
    and 01-06, the exit cannot be traded (B's market is closed there), so B holds 0.5 over 01-06 and
    01-07 and earns 0.5 x 10% on 01-07; it is sold at the close of 01-07."""
    p = closed
    jB = int(np.flatnonzero(p.sec_ids == "B")[0])
    t_idx = np.arange(D["01-04"], len(DAYS))
    wp = np.zeros((len(t_idx), p.n_pool), dtype=np.float32)
    wp[: D["01-05"] - D["01-04"] + 1, jB] = 0.5                               # targets for 01-04, 01-05
    d = base.daily_book(p, wp, t_idx, {"execution": {"lag": 0}}).set_index("date")
    held = base.held_book(p, wp, t_idx, {"execution": {"lag": 0}})[:, jB]
    assert held.tolist() == [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]               # 01-04 .. 02-03
    assert d.loc[pd.Timestamp("2000-01-07"), "gross"] == pytest.approx(0.05)
    assert d.loc[pd.Timestamp("2000-02-01"), "turnover"] == pytest.approx(0.5)              # sold at the close of 01-07


def test_weights_up_to_a_halt_do_not_depend_on_it(tmp_path, returns, universe, features):
    """Whether a member trades on day t is not known at the close of t-1. With return_gaps closed,
    every weight up to and including day t is the same whether the name trades on t or halts from
    t (two days, then resumes). Masked, the day-t weight differs: the look-ahead of item 2."""
    cfg = yaml.safe_load((ROOT / "configs" / "us_headline_v2.yaml").read_text())
    cfg["policy"].update(residual_lookback=10, hidden=8)
    cfg["model"].update(embedding_dim=8)
    dates = np.sort(returns["date"].unique())
    t = 300
    month = pd.Timestamp(dates[t]).to_period("M")
    name = universe[pd.to_datetime(universe["month"]).dt.to_period("M") == month]["sec_id"].iloc[0]
    halt = returns["date"].isin(dates[t:t + 2]) & (returns["sec_id"] == name)
    assert halt.sum() == 2
    da, db = tmp_path / "halt", tmp_path / "trades"
    for d, ret in ((da, returns[~halt]), (db, returns)):
        d.mkdir()
        for nm, df in (("returns", ret), ("universe", universe), ("features", features)):
            df.to_parquet(d / f"{nm}.parquet", index=False)
    lo, hi = str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())

    def weights(d, mode):
        p = slots.load_slots(d, lo, hi, return_gaps=mode)
        model = val.fresh_model(cfg, 3, len(p.features), 3)
        model.eval()
        with torch.no_grad():
            _, _, out = base.span(model, p, t - 5, t + 1, cfg)                  # trade dates t-5 .. t
        return to_pool(out["w"], p.idx[t - 5:t + 1], p.n_pool), p

    wa, pa = weights(da, "closed")
    wb, pb = weights(db, "closed")
    assert torch.equal(wa, wb)
    j = int(np.flatnonzero(pa.sec_ids == name)[0])
    assert wa[-1, j] != 0                                                     # the name is in the day-t book
    wm, _ = weights(da, "masked")
    assert wm[-1, j] == 0 and not torch.equal(wm[-1], wb[-1])


def test_pca_panel_and_robustness_market_see_the_same_cells(tmp_path):
    write_panel(tmp_path)
    p = slots.load_slots(tmp_path, "2000-01-01", "2000-02-28", return_gaps="closed")
    pp = pca_factors.load_panel(tmp_path, "2000-02-28", return_gaps="closed")
    m = robustness.load_market(tmp_path, ["A", "B", "C", "E", "F"], start="2000-01-01", return_gaps="closed")
    for sec in "ABCEF":
        j, jp, jm = (int(np.flatnonzero(np.asarray(x) == sec)[0]) for x in (p.sec_ids, pp.sec_ids, m.cols))
        assert (p.closed[:, j] == pp.closed[:, jp]).all() and (p.closed[:, j] == m.closed[:, jm]).all()
    jB = int(np.flatnonzero(pp.sec_ids == "B")[0])
    assert pp.R[D["01-05"], jB] == 0.0 and pp.R[D["01-07"], jB] == pytest.approx(0.10)
    assert np.isnan(pca_factors.load_panel(tmp_path, "2000-02-28").R[D["01-05"], jB])   # masked: NaN as before


def test_pca_stage_one_built_another_way_is_refused(tmp_path):
    write_panel(tmp_path)
    pp = pca_factors.load_panel(tmp_path, "2000-02-28", return_gaps="closed")
    pca_dir = tmp_path / "pca"
    pca_dir.mkdir()
    np.save(pca_dir / "dates.npy", pp.dates.to_numpy())
    (pca_dir / "meta.json").write_text(json.dumps({"cov_window": 252}))     # built before return_gaps existed
    with pytest.raises(ValueError, match="return_gaps"):
        rp.Stage1(pca_dir, 1, pp)


def test_halted_needs_a_member_day_to_start(tmp_path):
    """A non-member's missing days never become closed on their own: only a halt with a member day
    in it is. In a top-3 run E (rank 4 in January) is never a member, so its halt is not closed."""
    write_panel(tmp_path)
    assert halted(tmp_path, DAYS, ["E"]).any()
    assert not halted(tmp_path, DAYS, ["E"], max_rank=3).any()


def test_a_halt_longer_than_the_cap_is_an_exit(tmp_path):
    """G stops trading for MAX_HALT_DAYS + 1 market days, H for MAX_HALT_DAYS: G's run is treated
    as an exit (masked, as a delisting), H's as a halt (closed)."""
    from afe.data.return_gaps import MAX_HALT_DAYS as M
    days = pd.bdate_range("2000-01-03", periods=M + 20)
    off = {"G": range(5, 5 + M + 1), "H": range(5, 5 + M)}
    ret = pd.DataFrame([(t, s, 0.01, 100.0) for s in "AGH" for i, t in enumerate(days) if i not in off.get(s, ())],
                       columns=["date", "sec_id", "ret", "mktcap_lag"])
    ret.to_parquet(tmp_path / "returns.parquet", index=False)
    months = days.to_period("M").unique().to_timestamp()
    pd.DataFrame([(m, s, r) for m in months for r, s in enumerate("AGH", 1)],
                 columns=["month", "sec_id", "cap_rank"]).to_parquet(tmp_path / "universe.parquet", index=False)
    h = halted(tmp_path, days, ["A", "G", "H"])
    assert not h[:, 0].any() and not h[:, 1].any()
    assert h[:, 2].sum() == M and h[5:5 + M, 2].all()


def test_every_thesis_v3_config_closes_halts():
    m = yaml.safe_load((ROOT / "configs" / "thesis_v3_matrix.yaml").read_text(encoding="utf-8"))
    paths = set()

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("config"), str):
                paths.add(x["config"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(m)
    assert len(paths) >= 10
    for f in sorted(paths):
        c = yaml.safe_load((ROOT / f).read_text(encoding="utf-8"))
        assert "data.return_gaps" not in str(c), f
        root = c if "data" in c else yaml.safe_load((ROOT / c["base_config"]).read_text(encoding="utf-8"))
        assert root["data"].get("return_gaps") == "closed", f
