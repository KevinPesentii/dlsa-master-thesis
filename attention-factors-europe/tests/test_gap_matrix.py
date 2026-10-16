"""The paper-gap matrix (configs/gap_matrix.yaml) claims that every spec is the US headline with
one change, and that its 800-name data build (configs/us_data_top800.yaml) differs from the
500-name build only in its size and its output directories, so the default build is never
touched. These tests hold the configs to that claim."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_bundle as mb  # noqa: E402
import run_matrix as rm  # noqa: E402

EXPECTED = {                      # dotted key -> value, beyond the save_models block every spec uses
    "us_headline": {},
    "us_top800": {"data.dir": "data/us/top800", "sample.universe_size": 800},
    "us_ridge_1e3": {"model.lambda_ridge": 0.001},
    "us_no_temperature": {"model.score_std_target": None},
    "us_paper_literal": {"objective.lambda_var": 6.3, "model.score_std_target": None, "policy.input": "raw",
                         "policy.input_normalise": "none", "policy.input_scale": 1.0,
                         "training.grad_clip": None},
}


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def changed(cfg: dict, base: dict) -> dict:
    a, b = flat({k: v for k, v in cfg.items() if k != "matrix"}), flat(base)
    return {k: a.get(k) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def test_every_spec_is_the_headline_with_one_change():
    matrix, base = load("gap_matrix"), load("us_headline")
    assert {s["name"] for s in matrix["specs"]} == set(EXPECTED)
    for spec in matrix["specs"]:
        assert spec["config"] == "configs/us_headline.yaml", spec["name"]
        assert changed(rm.resolve(spec, matrix), base) == {**EXPECTED[spec["name"]], "evaluation.save_models": True}, \
            spec["name"]


def test_top800_build_differs_only_in_size_and_outputs():
    a, b = flat(load("us_data_top800")), flat(load("us_data"))
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == \
        {"sample.universe_size", "output.dir", "output.inspect_dir"}
    assert a["sample.universe_size"] == 800 and a["output.dir"] == "data/us/top800"
    assert a["output.inspect_dir"] not in (b["output.dir"], b["output.inspect_dir"])


def test_jobs_carry_the_larger_universe_memory_and_cost(tmp_path):
    matrix = load("gap_matrix")
    jobs = {j.id: j for j in rm.expand(matrix, None, tmp_path, threads=2, smoke=False)}
    assert len(jobs) == len(EXPECTED) * 3 * 3
    spec = next(s for s in matrix["specs"] if s["name"] == "us_top800")
    big, ref = jobs["us_top800_K100_s0"], jobs["us_headline_K100_s0"]
    assert big.mem_gb == float(spec["mem_gb"]) > ref.mem_gb == rm.MEM_GB[("attention", "us")]
    assert big.cost == pytest.approx(float(spec["cost_scale"]) * ref.cost)
    smoke = rm.expand(matrix, None, tmp_path, threads=2, smoke=True)
    assert len(smoke) == len(EXPECTED) and all(j.K == 8 and j.seed == 0 for j in smoke)


def test_bundle_default_markets_are_unchanged():
    assert mb.DEFAULT_MARKETS == ["us", "europe"]
    assert mb.FILES["us_top800"] == [f"data/us/top800/{t}.parquet" for t in mb.TABLES]
