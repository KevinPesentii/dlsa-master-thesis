"""configs/thesis_v2_matrix.yaml claims: model v2 is the old headline with its four changes and nothing
else; the four searches hold those changes and differ from the old searches in nothing else;
every row is a row of the thesis matrices or a ladder step that changes one thing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_bundle as mb  # noqa: E402
import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from test_gap2_matrix import changed, flat, load  # noqa: E402

V2 = {"features.timing": "asof", "features.include_cross_sectional_median": False,
      "features.include_risk_free": False, "policy.residual_window": "fixed", "data.returns_history": True}
SAVE = {"evaluation.save_models": True, "evaluation.diagnostics.top_n": 20}
NO_LEVELS = {"features.include_cross_sectional_median": False, "features.include_risk_free": False}
SOBOL110 = {"objective.lambda_var": 1.09, "policy.input_scale": 42.3, "model.score_std_target": None,
            "model.lambda_ridge": 0.00255, "training.batch_days": 250, "policy.input": "raw",
            "policy.input_normalise": "none"}
HELD = {"policy.residual_window": "fixed", "policy.input": "cumulative", "model.level_hidden": 0,
        "features.timing": "asof", "features.include_cross_sectional_median": False,
        "features.include_risk_free": False, "data.returns_history": True}


def rows():
    matrix = load("thesis_v2_matrix")
    return matrix, {s["name"]: s for s in matrix["specs"]}


def test_v2_configs_are_the_headlines_with_the_four_changes():
    assert changed(load("us_headline_v2"), load("us_headline")) == {**V2, **SAVE}
    assert changed(load("europe_headline_v2"), load("europe_headline")) == {**V2, **SAVE}


def test_searches_hold_the_changes_and_keep_the_rest():
    for new, old, extra in (("us_search_v2", "us_search", {}), ("us_search_v2_lag1", "us_search", {"execution.lag": 1}),
                            ("europe_search_v2", "europe_search_headline", {"data.dir": "data/europe17/holfill"}),
                            ("europe_search_v2_lag1", "europe_search_headline",
                             {"data.dir": "data/europe17/holfill", "execution.lag": 1})):
        a, b = load(new), load(old)
        assert a["fixed"] == {**HELD, **extra}, new
        for block in ("base_config", "K", "first_oos_year", "validation_years", "eval_epochs", "robustness"):
            assert a[block] == b[block], (new, block)
        assert set(b["search"]["space"]) - set(a["search"]["space"]) == {"policy.input", "model.level_hidden"}
        assert not set(a["search"]["space"]) & set(a["fixed"]), new             # a held key must not be drawn
        assert {k: v for k, v in a["search"].items() if k != "space"} == \
               {k: v for k, v in b["search"].items() if k != "space"}, new
        assert a["finalists"]["top"] == 5 and a["finalists"]["seeds"] == [0, 1, 2]


def test_held_keys_exist_in_the_bases():
    for base in ("us_replication", "europe_replication"):
        sv.with_overrides(load(base), HELD)


def test_us_and_europe_searches_draw_the_same_points():
    us, eu = load("us_search_v2")["search"], load("europe_search_v2")["search"]
    assert us["space"] == eu["space"] and us["n_points"] == eu["n_points"] and us["sobol_seed"] == eu["sobol_seed"]


def test_headline_rows_trade_with_their_market_winner():
    _, r = rows()
    assert r["us_attention"]["hyper"] == "us_v2" and r["eu_attention"]["hyper"] == "eu_v2"
    assert r["eu_attention_usvalues"]["hyper"] == "us_v2"
    assert r["us_lag1"]["hyper"] == "us_v2_lag1" and r["eu_lag1"]["hyper"] == "eu_v2_lag1"
    for name, s in r.items():
        if s["group"] != "ladder":
            assert s.get("hyper"), name                                         # every reported return is validated


def test_every_old_attention_row_is_here_except_the_raw_input():
    _, r = rows()
    old = {s["name"] for f in ("thesis_matrix", "thesis_matrix_europe") for s in load(f)["specs"] if s["runner"] == "attention"}
    assert old - set(r) == {"us_raw_input", "eu_raw_input"}
    assert {"us_no_receipts"} <= set(r)


def test_ladder_steps_change_one_thing_each():
    matrix, r = rows()
    get = lambda n: rm.resolve(r[n], matrix)  # noqa: E731
    us0, eu0 = get("us_ladder0_old"), get("eu_ladder0_old")
    assert changed(us0, load("us_headline")) == SAVE
    assert changed(eu0, load("europe_headline")) == {**SAVE, **SOBOL110}
    for m, base in (("us", us0), ("eu", eu0)):
        strip = {k: v for k, v in base.items() if k != "matrix"}
        assert changed(get(f"{m}_ladder1_no_levels"), strip) == NO_LEVELS
        assert changed(get(f"{m}_ladder2_asof"), strip) == {**NO_LEVELS, "features.timing": "asof"}
        assert changed(get(f"{m}_ladder3_fixed"), strip) == {
            **NO_LEVELS, "features.timing": "asof", "policy.residual_window": "fixed", "data.returns_history": True}
    # the US ladder ends on model v2 at the old values
    assert changed(get("us_ladder3_fixed"), load("us_headline_v2")) == {}
    last = {k: v for k, v in get("eu_ladder3_fixed").items() if k != "matrix"}
    assert changed(get("eu_ladder4_cumulative"), last) == {"policy.input": "cumulative"}


def test_job_list(tmp_path):
    matrix, r = rows()
    jobs = {j.id: j for j in rm.expand(matrix, None, tmp_path, threads=2, smoke=False)}
    assert sum(1 for j in jobs if j.startswith("search_")) == 4
    runs = [j for j in jobs.values() if j.spec]
    assert len(runs) == 3 * 7 * 5 + 12 * 3 + 6 * 3 + 2 * 5 + 9 * 5
    assert jobs["us_attention_K30_s0"].after == ["search_us_v2"]
    assert jobs["eu_attention_K30_s0"].after == ["search_eu_v2"]
    assert jobs["search_eu_v2"].after == ["search_us_v2"] and jobs["search_eu_v2_lag1"].after == ["search_us_v2_lag1"]
    assert jobs["us_ladder3_fixed_K30_s0"].after == []
    smoke = rm.expand(matrix, None, tmp_path, threads=2, smoke=True)
    assert len([j for j in smoke if j.spec]) == len(r)


def test_keep_from_passes_the_winner_by_name(tmp_path):
    d = tmp_path / "searches" / "us_v2"
    d.mkdir(parents=True)
    base = {"model": {"lambda_ridge": 1e-4}}
    (d / "manifest.json").write_text(json.dumps({"config": {"search": {"search": {"space": {"model.lambda_ridge": {}}}}, "base": base}}))
    (d / "selected.yaml").write_text("x: 1\n")
    (d / "metrics.json").write_text(json.dumps({"winner": {"name": "sobol-007", "overrides": {}, "val_net_SR": 7.0}}))
    old = rm.ROOT
    rm.ROOT = tmp_path
    try:
        job = rm.search_job("eu_v2", {"config": "c.yaml", "keep_from": "us_v2"}, tmp_path, 2, smoke=False)
        cmd = job.make_cmd()
        assert cmd[cmd.index("--keep") + 1] == "sobol-007"
        assert "--keep" not in rm.search_job("eu_v2", {"config": "c.yaml", "keep_from": "us_v2"}, tmp_path, 2, smoke=True).make_cmd()
    finally:
        rm.ROOT = old


def test_bundle_ships_what_the_matrix_reads():
    matrix, _ = rows()
    files = {f for m in ("us", "europe", "us_no_receipts", "us_no_receipts_asof") for f in mb.FILES[m]}
    for spec in matrix["specs"]:
        cfg = rm.resolve(spec, matrix)
        table = "features_asof" if cfg["features"].get("timing") == "asof" else "features"
        needed = ["returns", "universe", table] + (["returns_history"] if cfg["data"].get("returns_history") else [])
        for t in needed:
            assert f"{cfg['data']['dir']}/{t}.parquet" in files, (spec["name"], t)


def test_flat_helper_sees_nested_diagnostics():
    assert flat({"evaluation": {"diagnostics": {"top_n": 20}}}) == {"evaluation.diagnostics.top_n": 20}
