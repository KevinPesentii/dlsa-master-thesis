"""The review follow-ups of 2026-11-26 (configs/review_t1_matrix.yaml, review_t2, review_t3): the design in their
headers, read off the matrices against thesis_v3's, and the opt-in universe limit of a search. Config checks, no data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402

V3 = yaml.safe_load((ROOT / "configs" / "thesis_v3_matrix.yaml").read_text())
T = {t: yaml.safe_load((ROOT / "configs" / f"review_t{t}_matrix.yaml").read_text()) for t in (1, 2, 3)}
MKT = {"us": ("us_headline_v2", "us_search_t4_quick"), "eu": ("europe_headline_v2", "europe_search_t4_quick")}


def spec(matrix: dict, name: str) -> dict:
    return next(s for s in matrix["specs"] if s["name"] == name)


def jobs(t: int, tmp_path) -> list:
    return rm.expand(T[t], None, tmp_path, 2, False, max_jobs=46)


def test_names_and_job_counts(tmp_path):
    assert [T[t]["name"] for t in (1, 2, 3)] == ["review_t1", "review_t2", "review_t3"]
    assert len(T[1]["specs"]) == 14 and len(T[2]["specs"]) == 4 and len(T[3]["specs"]) == 1
    assert [len(jobs(t, tmp_path / str(t))) for t in (1, 2)] == [64, 20]
    # tier 3 waits for thesis_v3's finished search (keep_from), so expand() needs it unpacked: count by hand
    assert len(T[3]["searches"]) + sum(len(T[3]["sets"][s["seeds"]]) for s in T[3]["specs"]) == 6


def test_the_headline_blocks_are_the_same_in_both_tiers():
    for b in ("v3_us_K30", "v3_eu_K30"):
        assert T[1]["blocks"][b] == T[2]["blocks"][b]
    assert T[1]["blocks"]["v3_us_K30"]["objective.lambda_var"] == 10.5
    assert T[1]["blocks"]["v3_eu_K30"]["objective.lambda_var"] == 0.842


def test_the_sweep_changes_lambda_var_only_and_spans_the_decades():
    grid = {}
    for s in T[1]["specs"]:
        if s["group"] == "lambda_sweep":
            m = s["name"][:2]
            assert (s["config"], s["use"], set(s["set"])) == (f"configs/{MKT[m][0]}.yaml", [f"v3_{m}_K30"], {"objective.lambda_var"})
            assert T[1]["sets"][s["seeds"]] == [0, 1, 2, 3, 4] and T[1]["sets"][s["K"]] == [30]
            grid.setdefault(m, []).append(s["set"]["objective.lambda_var"])
    assert grid == {"us": [0.1, 1.0, 10.0, 100.0, 1000.0], "eu": [0.1, 1.0, 10.0, 100.0, 1000.0]}
    assert not any(s.get("set", {}).get("training.two_step") for s in T[1]["specs"])


@pytest.mark.parametrize("m,cfg,mkt", [("us", "us", "us"), ("eu", "europe", "europe17")])
def test_the_pca_holdout_runs_on_its_own_stage_one(m, cfg, mkt):
    prep = T[1]["prep"][f"pca_{mkt}_2025"]
    stage = yaml.safe_load((ROOT / prep["config"]).read_text())
    base = yaml.safe_load((ROOT / f"configs/{cfg}_pca_longconv.yaml").read_text())
    assert stage["data"] == base["data"] and str(stage["sample"]["end"]) == "2025-12-31"
    assert {k: v for k, v in stage["factors"].items() if k != "out_dir"} == \
           {k: v for k, v in base["factors"].items() if k != "out_dir"}
    block = T[1]["blocks"][f"holdout_pca_{mkt}"]
    assert block["factors.out_dir"] == stage["factors"]["out_dir"] != base["factors"]["out_dir"]
    assert prep["done"] == stage["factors"]["out_dir"] + "/meta.json"
    for kind in ("longconv", "ou"):
        s = spec(T[1], f"{m}_pca_{kind}_perk_K30_2022_2025")
        twin = spec(V3, f"{m}_pca_{kind}_perk_K30")
        assert (s["runner"], s["config"], s["seeds"]) == (twin["runner"], twin["config"], twin["seeds"])
        assert s["use"][0] == f"holdout_pca_{mkt}" and s["after"] == f"pca_{mkt}_2025"
        assert str(block["sample.start"]) == str(V3["blocks"]["holdout"]["sample.start"])


@pytest.mark.parametrize("m", ["us", "eu"])
def test_the_paper_rows_hold_the_search_base_candidate(m):
    """`base` of a quick search = its base config with the fixed block; run_matrix carries every fixed and
    searched key but data / sample ones (selected_values)."""
    S = yaml.safe_load((ROOT / "configs" / f"{MKT[m][1]}.yaml").read_text())
    base = sv.with_overrides(yaml.safe_load((ROOT / S["base_config"]).read_text()), S["fixed"])
    keys = [k for k in [*S["fixed"], *S["search"]["space"]] if k.split(".")[0] not in ("data", "sample")]
    want = {k: base[k.split(".")[0]][k.split(".")[1]] for k in keys}
    assert T[2]["blocks"][f"v3_{m}_K30_base"] == want
    s = spec(T[2], f"{m}_attention_K30_paper")
    assert s["use"] == [f"v3_{m}_K30_base"] and "set" not in s
    rf = spec(T[2], f"{m}_attention_t4q_K30_rf")
    assert rf["use"] == [f"v3_{m}_K30"] and rf["set"] == {"objective.subtract_rf": True}


def test_the_top200_search_is_the_us_quick_search_on_200_names(tmp_path):
    s = T[3]["searches"]["us200_t4q_K30"]
    v3 = V3["searches"]["us_t4q_K30"]
    assert (s["config"], s["K"]) == (v3["config"], v3["K"])
    assert s["fixed"] == {"sample.universe_size": 200} and s["keep_from"] == "thesis_v3/us_t4q_K30"
    row, twin = T[3]["specs"][0], spec(V3, "us_top200_t4q_K30")
    assert (row["config"], row["set"], row["hyper"]) == (twin["config"], twin["set"], "us200_t4q_K30")
    S = yaml.safe_load((ROOT / s["config"]).read_text())
    sv.with_overrides(yaml.safe_load((ROOT / S["base_config"]).read_text()), {**S["fixed"], **s["fixed"]})


def test_no_committed_search_config_holds_the_universe_size():
    """The opt-in of search_validation (max_rank from a fixed sample.universe_size) changes no earlier search."""
    for f in sorted((ROOT / "configs").glob("*.yaml")):
        S = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if "search" in S and "base_config" in S:
            assert "sample.universe_size" not in (S.get("fixed") or {}), f.name
    for f in sorted((ROOT / "configs").glob("*_matrix.yaml")):
        if f.name.startswith("review_t3"):
            continue
        M = yaml.safe_load(f.read_text(encoding="utf-8"))
        for s in (M.get("searches") or {}).values():
            assert "sample.universe_size" not in (s.get("fixed") or {}), f.name


def test_a_search_panel_takes_the_universe_limit_only_when_given(monkeypatch):
    seen = []

    class Panel:
        dates = pd.bdate_range("1990-01-01", "1997-12-31")

    def load_config(cfg, data_dir, start, end, max_rank=None):
        seen.append(max_rank)
        return Panel()
    monkeypatch.setattr(sv.slots, "load_config", load_config)
    cfg = yaml.safe_load((ROOT / "configs" / "us_replication.yaml").read_text())
    sv.init_worker(cfg, "data", 1998, 2, 30, [], 1)
    sv.init_worker(cfg, "data", 1998, 2, 30, [], 1, max_rank=200)
    assert seen == [None, 200]
