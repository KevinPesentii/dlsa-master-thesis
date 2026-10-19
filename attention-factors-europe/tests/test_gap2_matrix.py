"""The second paper-gap matrix (configs/gap2_matrix.yaml) claims that every spec is the US headline with
only the changes it names, that its two baselines are the first gap matrix's rows unchanged, and that
its data build (configs/us_data_rank301_800.yaml) differs from the 500-name build only in where the
rank window starts and in its output directories, so no other build is touched."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_bundle as mb  # noqa: E402
import run_matrix as rm  # noqa: E402

TOP800 = {"data.dir": "data/us/top800", "sample.universe_size": 800}
PAPER_LITERAL = {"objective.lambda_var": 6.3, "model.score_std_target": None, "policy.input": "raw",
                 "policy.input_normalise": "none", "policy.input_scale": 1.0, "training.grad_clip": None}
EXPECTED = {                      # dotted key -> value, beyond the save_models block every spec uses
    "us_headline": {},
    "us_top800": TOP800,
    "us_rank301_800": {"data.dir": "data/us/rank301_800"},
    "us_top800_paper_literal": {**TOP800, **PAPER_LITERAL},
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


def test_every_spec_is_the_headline_with_its_named_changes():
    matrix, base = load("gap2_matrix"), load("us_headline")
    assert {s["name"] for s in matrix["specs"]} == set(EXPECTED)
    for spec in matrix["specs"]:
        assert spec["config"] == "configs/us_headline.yaml", spec["name"]
        assert changed(rm.resolve(spec, matrix), base) == {**EXPECTED[spec["name"]], "evaluation.save_models": True}, \
            spec["name"]


def test_baselines_and_paper_literal_are_the_first_gap_runs():
    new, old = load("gap2_matrix"), load("gap_matrix")
    spec = lambda m, name: next(s for s in m["specs"] if s["name"] == name)  # noqa: E731
    strip = lambda cfg: {k: v for k, v in cfg.items() if k != "matrix"}  # noqa: E731
    for name in ("us_headline", "us_top800"):
        assert strip(rm.resolve(spec(new, name), new)) == strip(rm.resolve(spec(old, name), old)), name
        assert spec(new, name).get("mem_gb") == spec(old, name).get("mem_gb"), name
    assert new["blocks"]["paper_literal"] == old["blocks"]["paper_literal"]


def test_rank_window_build_differs_only_in_first_rank_and_outputs():
    a, b = flat(load("us_data_rank301_800")), flat(load("us_data"))
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == \
        {"sample.universe_first_rank", "output.dir", "output.inspect_dir"}
    assert a["sample.universe_size"] == 500 and a["sample.universe_first_rank"] == 301
    assert a["output.dir"] == "data/us/rank301_800"
    others = [flat(load(n)) for n in ("us_data", "us_data_top800")]
    assert {a["output.dir"], a["output.inspect_dir"]}.isdisjoint(
        {o[k] for o in others for k in ("output.dir", "output.inspect_dir")})


def test_jobs_carry_the_universe_memory_and_cost(tmp_path):
    matrix = load("gap2_matrix")
    jobs = {j.id: j for j in rm.expand(matrix, None, tmp_path, threads=2, smoke=False)}
    assert len(jobs) == len(EXPECTED) * 3 * 3
    ref = jobs["us_headline_K100_s0"]
    assert jobs["us_rank301_800_K100_s0"].mem_gb == ref.mem_gb == rm.MEM_GB[("attention", "us")]
    assert jobs["us_rank301_800_K100_s0"].cost == ref.cost
    for name in ("us_top800", "us_top800_paper_literal"):
        assert jobs[f"{name}_K100_s0"].mem_gb == 10.0 and jobs[f"{name}_K100_s0"].cost == 1.6 * ref.cost
    smoke = rm.expand(matrix, None, tmp_path, threads=2, smoke=True)
    assert len(smoke) == len(EXPECTED) and all(j.K == 8 and j.seed == 0 for j in smoke)


def test_bundle_ships_the_rank_window_tables_outside_the_default():
    assert mb.DEFAULT_MARKETS == ["us", "europe"]
    assert mb.FILES["us_rank301_800"] == [f"data/us/rank301_800/{t}.parquet" for t in mb.TABLES]
