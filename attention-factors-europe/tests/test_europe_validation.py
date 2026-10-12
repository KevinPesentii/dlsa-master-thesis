"""The Europe-validated matrix (configs/thesis_matrix_europe.yaml) claims that its searches are
the US searches with only the data changed, and that it reruns every Europe row of the thesis
matrix on Europe's own winners. These tests hold the configs to that claim."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402

PAIRS = {"us_search": "europe_search_headline", "us_search_raw": "europe_search_raw",
         "us_search_lag1": "europe_search_lag1", "us_search_pca": "europe_search_pca",
         "us_search_pca_lag1": "europe_search_pca_lag1", "us_search_ou": "europe_search_ou"}
DATA_KEYS = {"market", "data", "sample", "seed"}


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def without_data(cfg: dict) -> dict:
    out = {k: v for k, v in cfg.items() if k not in DATA_KEYS}
    if "factors" in out:
        out["factors"] = {k: v for k, v in out["factors"].items() if k != "out_dir"}
    return out


@pytest.mark.parametrize("us,eu", PAIRS.items())
def test_europe_search_is_the_us_search_on_european_data(us, eu):
    U, E = load(us), load(eu)
    for key in ("runner", "K", "first_oos_year", "validation_years", "eval_epochs", "search", "robustness"):
        assert U.get(key) == E.get(key), key
    assert {k: v for k, v in E["finalists"].items() if k != "keep"} == U["finalists"]
    fixed_u, fixed_e = U.get("fixed") or {}, E.get("fixed") or {}
    assert {k: v for k, v in fixed_e.items() if k.split(".")[0] not in DATA_KEYS} == fixed_u
    base_u, base_e = (yaml.safe_load((ROOT / S["base_config"]).read_text()) for S in (U, E))
    base_u, base_e = sv.with_overrides(base_u, fixed_u), sv.with_overrides(base_e, fixed_e)
    assert base_e["market"] == "europe17" and base_e["data"]["dir"] == "data/europe17/holfill"
    assert without_data(base_u) == without_data(base_e), "the base candidates differ beyond the data"
    if "training" in base_e:          # fit 1993-1995, validate 1996-1997
        assert sv.base.window_start(base_e, E["first_oos_year"]) == pd.Timestamp("1993-01-01")


@pytest.mark.parametrize("eu", PAIRS.values())
def test_carried_us_winner_is_a_point_of_the_space(eu):
    S = load(eu)["search"]
    names = ([f"grid-{i:03d}" for i in range(len(sv.grid_points(S["grid"])))] if "grid" in S
             else [f"sobol-{i:03d}" for i in range(S["n_points"])])
    assert load(eu)["finalists"]["keep"] and set(load(eu)["finalists"]["keep"]) <= set(names)


def test_every_europe_row_is_rerun_on_a_europe_search():
    us_matrix, eu_matrix = load("thesis_matrix"), load("thesis_matrix_europe")
    eu_rows = {s["name"]: s for s in us_matrix["specs"] if s["name"].startswith("eu_")}
    rows = {s["name"]: s for s in eu_matrix["specs"]}
    assert set(rows) == set(eu_rows)
    for name, s in rows.items():
        old = eu_rows[name]
        for key in ("group", "runner", "config", "K", "seeds", "use", "set"):
            assert s.get(key) == old.get(key), (name, key)
        assert s["hyper"] in eu_matrix["searches"], name
    for name, s in eu_matrix["searches"].items():
        S = yaml.safe_load((ROOT / s["config"]).read_text())
        assert yaml.safe_load((ROOT / S["base_config"]).read_text())["market"] == "europe17", name


def test_hyper_carries_the_europe_winner_but_never_its_data_dir(tmp_path, monkeypatch):
    """A spec trades with the winner's searched and fixed values; the search's data.dir (the
    holiday-filled tables) is not among them, so a row's own data choice stands."""
    matrix = load("thesis_matrix_europe")
    S = load("europe_search_headline")
    base = sv.with_overrides(yaml.safe_load((ROOT / S["base_config"]).read_text()), S["fixed"])
    win = {"objective.lambda_var": 3.3, "policy.input_scale": 2.2, "model.score_std_target": None,
           "model.lambda_ridge": 1e-4, "training.batch_days": 250, "policy.input": "raw",
           "policy.input_normalise": "none", "model.level_hidden": 8}
    d = tmp_path / "runs" / "_matrix" / "thesis_eu" / "searches" / "eu"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"config": {"search": S, "base": base}}, default=str))
    (d / "metrics.json").write_text(json.dumps({"winner": {"name": "sobol-007", "overrides": win,
                                                           "val_net_SR": 1.0}}))
    _, values = rm.selected_values(d)
    assert values == win and "data.dir" not in values
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "europe_headline.yaml", tmp_path / "configs")
    monkeypatch.setattr(rm, "ROOT", tmp_path)
    for name, data_dir in (("eu_attention", "data/europe17/holfill"), ("eu_europe12", "data/europe12/holfill")):
        cfg = rm.resolve(next(s for s in matrix["specs"] if s["name"] == name), matrix, d.parents[1])
        assert all(cfg[k.split(".")[0]][k.split(".")[1]] == v for k, v in win.items()), name
        assert cfg["data"]["dir"] == data_dir and cfg["matrix"]["hyper"]["candidate"] == "sobol-007"
