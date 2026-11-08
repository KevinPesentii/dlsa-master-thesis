"""configs/*_search_t4.yaml and configs/search_t4_*_matrix.yaml claim: every tuning parameter of the
paper's Table 4 is searched, with the paper's value inside its range; the rest is the v2 search
unchanged; the base candidate is Table 4 itself; Europe differs from the US only in its data; the
all-K matrix is the K = 30 one plus the other K of Table 2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from test_gap2_matrix import load  # noqa: E402

# Table 4 of Epstein et al. (2025), "Selected tuning parameters", copied from the paper
TABLE4 = {"policy.hidden": 32, "policy.dropout": 0.1, "model.embedding_dim": 32, "training.epochs": 30,
          "policy.layers": 1, "objective.lambda_var": 100, "training.lr": 0.003, "training.weight_decay": 0.05,
          "policy.kernel_init": "geom_decay", "policy.lambda_squash": 0.001}


def get(cfg: dict, key: str):
    section, name = key.split(".")
    return cfg[section][name]


def test_every_table4_parameter_is_searched_around_the_papers_value():
    for name in ("us_search_t4", "europe_search_t4"):
        space = load(name)["search"]["space"]
        for key, paper in TABLE4.items():
            spec = space[key]
            if "choice" in spec:
                assert paper in spec["choice"], (name, key)
            else:
                lo, hi = spec["log"]
                assert lo < paper < hi and spec.get("off", 0) == 0, (name, key)


def test_the_base_candidate_is_table4():
    for name in ("us_search_t4", "europe_search_t4"):
        S = load(name)
        base = sv.with_overrides(load(Path(S["base_config"]).stem), S["fixed"])
        assert {k: get(base, k) for k in TABLE4} == TABLE4


def test_the_rest_is_the_v2_search():
    for new, old in (("us_search_t4", "us_search_v2"), ("europe_search_t4", "europe_search_v2")):
        N, O = load(new), load(old)
        new, old_space = N["search"]["space"], O["search"]["space"]
        assert set(new) == set(old_space) | set(TABLE4)                   # lambda_var was searched already
        assert {k: new[k] for k in old_space} == old_space
        assert N["fixed"] == {**O["fixed"], "policy.residual_lookback": 30}
        for k in ("base_config", "K", "first_oos_year", "validation_years", "eval_epochs"):
            assert N[k] == O[k]
        assert N["finalists"]["seeds"] == O["finalists"]["seeds"]


def test_europe_draws_the_us_points_on_its_own_data():
    us, eu = load("us_search_t4"), load("europe_search_t4")
    assert us["search"] == eu["search"]
    assert {k: v for k, v in eu["fixed"].items() if k != "data.dir"} == us["fixed"]
    assert eu["finalists"]["extra"]["us_v2_headline"] == us["finalists"]["extra"]["us_v2_headline"]


def test_every_key_exists_in_its_base_config():
    for name in ("us_search_t4", "europe_search_t4"):
        S = load(name)
        base = load(Path(S["base_config"]).stem)
        for k in [*S["search"]["space"], *S["fixed"]]:
            sv.with_overrides(base, {k: None})
        for o in S["finalists"]["extra"].values():
            sv.with_overrides(base, o)


def test_sobol_points_cover_every_choice():
    space = load("us_search_t4")["search"]["space"]
    pts = sv.sobol_points(space, 512, 0)
    for key, spec in space.items():
        vals = [p[key] for p in pts]
        if "choice" in spec:
            counts = [vals.count(c) for c in spec["choice"]]
            assert min(counts) >= 512 // len(spec["choice"]) - 1, key     # balanced: Sobol at a power of 2
        else:
            x = np.array([v for v in vals if v is not None])
            assert spec["log"][0] <= x.min() and x.max() <= spec["log"][1], key


def test_matrices(tmp_path):
    k30, allk = load("search_t4_k30_matrix"), load("search_t4_allk_matrix")
    assert k30["name"] == allk["name"] and not k30["specs"] and not allk["specs"]
    S = allk["searches"]
    assert sorted(s["K"] for s in S.values()) == sorted(2 * [1, 3, 5, 8, 15, 30, 100])
    for name, s in S.items():
        if name.startswith("eu_"):
            assert s["keep_from"] == "us_" + name[3:] and S[s["keep_from"]]["K"] == s["K"]
    for name, s in k30["searches"].items():
        assert {k: v for k, v in s.items() if k != "workers"} == {k: v for k, v in S[name].items() if k != "workers"}
    old = rm.ROOT
    rm.ROOT = tmp_path
    try:
        jobs = {j.id: j for j in rm.expand(allk, None, tmp_path, threads=2, smoke=False)}
        assert len(jobs) == 14 and jobs["search_eu_t4_K8"].after == ["search_us_t4_K8"]
        cmd = jobs["search_us_t4_K100"].make_cmd()
        assert cmd[cmd.index("--K") + 1] == "100" and cmd[cmd.index("--jobs") + 1] == "23"
    finally:
        rm.ROOT = old
