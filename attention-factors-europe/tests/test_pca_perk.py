"""configs/pca_perk_matrix.yaml and its four LongConv searches claim: each PCA search is the attention quick search
(configs/us_search_t4_quick.yaml) for every knob the PCA policy has, on the same draw, so PCA point sobol-i carries
the LongConv and training values of attention point sobol-i; the attention-only knobs are drawn and dropped and never
reach a candidate or a selected config; Europe and the rolling twin differ only where they say; every K, market and
window has its search and its rows, and Europe keeps that K's US winner of the same window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from test_gap2_matrix import load  # noqa: E402

KS = [1, 3, 5, 8, 15, 30, 100]
SEARCHES = ["us_search_pca_perk", "us_search_pca_perk_rolling", "europe_search_pca_perk",
            "europe_search_pca_perk_rolling"]


def test_pca_points_are_the_attention_points():
    att, pca = load("us_search_t4_quick")["search"], load("us_search_pca_perk")["search"]
    assert list(pca["space"]) == list(att["space"])            # same dimensions, same order
    assert {k for k, v in pca["space"].items() if v == {"unused": True}} == \
           {"objective.lambda_var", "model.score_std_target", "model.lambda_ridge", "model.embedding_dim"}
    for k, v in pca["space"].items():
        assert v == {"unused": True} or v == att["space"][k]
    for k in ("n_points", "sobol_seed", "seed"):
        assert pca[k] == att[k]
    a = sv.sobol_points(att["space"], att["n_points"], att["sobol_seed"])
    p = sv.sobol_points(pca["space"], pca["n_points"], pca["sobol_seed"])
    used = sv.used(pca["space"])
    assert len(p) == 256 and all(list(q) == used for q in p)   # placeholders never reach a candidate
    assert p == [{k: q[k] for k in used} for q in a]


def test_every_used_key_is_a_pca_config_key():
    for name in SEARCHES:
        S = load(name)
        base = load(Path(S["base_config"]).stem)
        for key in [*sv.used(S["search"]["space"]), *S["fixed"], *(k for o in S["finalists"]["extra"].values() for k in o)]:
            section, k = key.split(".")
            assert k in base[section], (name, key)
        assert base["policy"]["kind"] == "longconv" and S["runner"] == "pca"


def test_twins_differ_only_where_they_say():
    us, us_r, eu, eu_r = (load(n) for n in SEARCHES)
    assert us["fixed"] == {"policy.residual_window": "fixed", "policy.input": "cumulative"}
    assert us_r["fixed"] == {"policy.residual_window": "rolling", "policy.input": "cumulative"}
    assert eu["fixed"] == us["fixed"] and eu_r["fixed"] == us_r["fixed"]
    for a, b in ((us, us_r), (us, eu), (us_r, eu_r)):
        assert a["search"] == b["search"]
        for k in ("K", "first_oos_year", "validation_years", "runner"):
            assert a[k] == b[k]
        assert {k: v for k, v in a["finalists"].items() if k != "extra"} == \
               {k: v for k, v in b["finalists"].items() if k != "extra"}
    assert us["finalists"] == {"top": 10, "seeds": [0, 1, 2], "extra": {"us_pca_oct_k30": {
        "policy.input_scale": 181.0, "training.batch_days": 125, "policy.input_normalise": "none"}}}
    assert eu["finalists"]["extra"]["us_pca_oct_k30"] == us["finalists"]["extra"]["us_pca_oct_k30"]
    assert eu["finalists"]["extra"]["eu_pca_oct_k30"] == {
        "policy.input_scale": 35.1, "training.batch_days": 63, "policy.input_normalise": "none"}
    assert us_r["finalists"] == us["finalists"] and eu_r["finalists"] == eu["finalists"]
    assert us["base_config"] == "configs/us_pca_longconv.yaml" and eu["base_config"] == "configs/europe_pca_longconv.yaml"
    assert load("us_search_t4_quick")["first_oos_year"] == us["first_oos_year"]


def test_matrix_has_every_k_market_and_window():
    m = load("pca_perk_matrix")
    s = m["searches"]
    for mk, cfg in (("us", "us"), ("eu", "europe")):
        for sfx, roll, ou in (("", "", "_v2"), ("_roll", "_rolling", "")):
            for K in KS:
                lc, oq = s[f"{mk}_pcaq{sfx}_K{K}"], s[f"{mk}_ouq{sfx}_K{K}"]
                assert lc["config"] == f"configs/{cfg}_search_pca_perk{roll}.yaml" and lc["K"] == K
                assert oq["config"] == f"configs/{cfg}_search_ou{ou}.yaml" and oq["K"] == K
                assert lc.get("keep_from") == (f"us_pcaq{sfx}_K{K}" if mk == "eu" else None)
    assert len(s) == 2 * 2 * 2 * len(KS)
    assert load("us_search_ou_v2")["fixed"] == {"policy.residual_window": "fixed"} and "fixed" not in load("us_search_ou")
    assert load("us_search_ou_v2")["search"] == load("us_search_ou")["search"] == load("europe_search_ou_v2")["search"]
    specs = {sp["name"]: sp for sp in m["specs"]}
    for mk, cfg in (("us", "us"), ("eu", "europe")):
        for sfx, roll in (("", ""), ("_roll", "_rolling")):
            for K in KS:
                lc, ou = specs[f"{mk}_pca_longconv_perk{roll}_K{K}"], specs[f"{mk}_pca_ou_perk{roll}_K{K}"]
                assert lc["hyper"] == f"{mk}_pcaq{sfx}_K{K}" and ou["hyper"] == f"{mk}_ouq{sfx}_K{K}"
                assert m["sets"][lc["K"]] == m["sets"][ou["K"]] == [K]
                assert lc["config"] == f"configs/{cfg}_pca_longconv.yaml" and ou["config"] == f"configs/{cfg}_pca_ou.yaml"
                assert m["sets"][lc["seeds"]] == [0, 1, 2, 3, 4] and m["sets"][ou["seeds"]] == [0]
                assert lc["group"] == ou["group"] == "candidate" and "use" not in lc and "set" not in lc
    v2 = {sp["name"]: sp for sp in load("thesis_v2_pca_matrix")["specs"] if sp["group"] == "ladder"}
    v2_blocks = load("thesis_v2_pca_matrix")["blocks"]
    for name, sp in v2.items():                                    # the ladder of the v2 PCA matrix, unchanged
        assert {**specs[name], "seeds": "seeds5"} == sp
        for b in sp.get("use", []):
            assert m["blocks"][b] == v2_blocks[b]
    assert len(specs) == 4 * 2 * len(KS) + len(v2)
    assert m["name"] not in ("thesis", "thesis_eu", "thesis_v2_pca", "search_t4q", "search_t4q_eu")


def test_selected_values_carry_no_placeholder(tmp_path):
    """A finished search's values (what a row trades with) are the fixed block and the used keys only."""
    S = load("us_search_pca_perk")
    base = load("us_pca_longconv")
    for k, v in S["fixed"].items():
        base["policy"][k.split(".")[1]] = v
    d = tmp_path / "s"
    d.mkdir()
    pt = sv.sobol_points(S["search"]["space"], 256, 0)[7]
    (d / "manifest.json").write_text(json.dumps({"config": {"search": S, "base": base}}, default=str))
    (d / "metrics.json").write_text(json.dumps({"winner": {"name": "sobol-007", "overrides": pt, "val_net_SR": 1.0}}))
    w, values = rm.selected_values(d)
    assert set(values) == set(S["fixed"]) | set(sv.used(S["search"]["space"]))
    assert all(values[k] == v for k, v in pt.items())


def test_unused_takes_no_other_key():
    with pytest.raises(SystemExit, match="unknown keys"):
        sv.sobol_points({"a.b": {"unused": True, "log": [1, 2]}}, 4, 0)
