"""The factor-competition experiment (configs/exp_competition_matrix.yaml) claims that each row is
its market's headline with only the changes it names, that its Europe values are the Europe
headline's (the Europe search's winner, copied because the matrix runs no search), and that its
baselines run Equation (1) unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
from test_gap2_matrix import changed, load  # noqa: E402

EU = {"objective.lambda_var": 1.09, "policy.input_scale": 42.3, "model.score_std_target": None,
      "model.lambda_ridge": 0.00255, "training.batch_days": 250, "policy.input": "raw",
      "policy.input_normalise": "none"}          # level_hidden 0 is the base config's value already
COMPETE = {"model.factor_weights": "compete"}
SHARP = {"model.score_std_target": 3.0}
EXPECTED = {
    "us_softmax": {},
    "us_compete": COMPETE,
    "us_softmax_sharp": {"model.score_std_target": 2.0},
    "us_compete_sharp": {**COMPETE, **SHARP},
    "us_compete_balanced": {**COMPETE, **SHARP, "model.sinkhorn_iters": 5},
    "eu_softmax": EU,
    "eu_compete": {**EU, **COMPETE},
    "eu_softmax_sharp": {**EU, "model.score_std_target": 2.0},
    "eu_compete_sharp": {**EU, **COMPETE, **SHARP},
    "eu_compete_balanced": {**EU, **COMPETE, **SHARP, "model.sinkhorn_iters": 5},
}


def test_every_row_is_its_headline_with_its_named_changes():
    matrix = load("exp_competition_matrix")
    assert {s["name"] for s in matrix["specs"]} == set(EXPECTED)
    for spec in matrix["specs"]:
        base = load("us_headline" if spec["name"].startswith("us_") else "europe_headline")
        assert spec["config"] == f"configs/{'us' if spec['name'].startswith('us_') else 'europe'}_headline.yaml"
        assert changed(rm.resolve(spec, matrix), base) == {**EXPECTED[spec["name"]], "evaluation.save_models": True}, \
            spec["name"]


def test_sixty_runs_and_nothing_selected(tmp_path):
    matrix = load("exp_competition_matrix")
    assert not matrix.get("searches") and not matrix["prep"]
    assert all("hyper" not in s for s in matrix["specs"])
    jobs = rm.expand(matrix, None, tmp_path, 2, False)
    assert len(jobs) == 60 and {j.K for j in jobs} == {8, 30} and {j.seed for j in jobs} == {0, 1, 2}


@pytest.mark.skipif(not (ROOT / "runs/_matrix/thesis_eu/searches/eu/metrics.json").exists(),
                    reason="the Europe-validated search run is only on the laptop")
def test_europe_values_are_the_europe_search_winner():
    _, values = rm.selected_values(ROOT / "runs/_matrix/thesis_eu/searches/eu")
    block = load("exp_competition_matrix")["blocks"]["eu_values"]
    assert block == values
