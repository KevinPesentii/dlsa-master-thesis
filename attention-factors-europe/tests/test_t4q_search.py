"""configs/europe_search_t4_quick.yaml and configs/search_t4q_eu_matrix.yaml claim: the Europe quick search is
the US quick search on Europe's data (same space, points, finalists and rule; the fixed block of the Europe
Table 4 search), each Europe K carries that K's US quick winner from the finished US matrix without rerunning
anything US, and the Europe rows are the US rows with Europe's headline config and searches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
from test_gap2_matrix import load  # noqa: E402

KS = [1, 3, 5, 8, 15, 30, 100]


def test_europe_quick_is_the_us_quick_search_on_europes_data():
    us, eu, eu_t4 = load("us_search_t4_quick"), load("europe_search_t4_quick"), load("europe_search_t4")
    assert eu["search"] == us["search"]                      # same space, layers at 1, 256 points, same draw
    assert eu["fixed"] == eu_t4["fixed"]                     # only the data differ from the US fixed block
    assert {k: v for k, v in eu["fixed"].items() if k != "data.dir"} == us["fixed"]
    for k in ("K", "first_oos_year", "validation_years", "eval_epochs"):
        assert eu[k] == us[k] == eu_t4[k]
    assert eu["base_config"] == eu_t4["base_config"] == "configs/europe_replication.yaml"
    assert {k: v for k, v in eu["finalists"].items() if k != "extra"} == \
           {k: v for k, v in us["finalists"].items() if k != "extra"}
    assert eu["finalists"]["extra"] == eu_t4["finalists"]["extra"]
    assert eu["finalists"]["extra"]["us_v2_headline"] == us["finalists"]["extra"]["us_v2_headline"]


def test_europe_rows_mirror_the_us_rows():
    us, eu = load("search_t4q_us_matrix"), load("search_t4q_eu_matrix")
    assert eu["name"] != us["name"] and eu["sets"] == us["sets"]
    assert sorted(s["K"] for s in eu["searches"].values()) == KS
    for name, s in eu["searches"].items():
        assert name == f"eu_t4q_K{s['K']}" and s["config"] == "configs/europe_search_t4_quick.yaml"
        assert s["keep_from"] == f"search_t4q/us_t4q_K{s['K']}" and us["searches"][f"us_t4q_K{s['K']}"]["K"] == s["K"]
    assert len(eu["specs"]) == len(us["specs"]) == len(KS)
    for u, e in zip(us["specs"], eu["specs"]):
        assert e == {**u, "name": u["name"].replace("us_", "eu_", 1), "config": "configs/europe_headline_v2.yaml",
                     "hyper": u["hyper"].replace("us_", "eu_", 1)}


def finished_search(root: Path, matrix: str, search: str, candidate: str) -> None:
    d = root / "runs" / "_matrix" / matrix / "searches" / search
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"config": {"search": {"search": {"space": {}}, "fixed": {}},
                                                            "base": {}}}))
    (d / "metrics.json").write_text(json.dumps({"winner": {"name": candidate, "overrides": {}, "val_net_SR": 1.0}}))
    (d / "selected.yaml").write_text("{}\n")


def test_keep_from_another_matrix(tmp_path, monkeypatch):
    eu = {**load("search_t4q_eu_matrix"), "specs": []}
    out = tmp_path / "runs" / "_matrix" / eu["name"]
    monkeypatch.setattr(rm, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="unpack it first"):     # never a silent search without its US winner
        rm.expand(eu, None, out, threads=2, smoke=False, max_jobs=23)
    rm.expand(eu, None, out, threads=2, smoke=True, max_jobs=8)  # a smoke run keeps nothing, needs nothing
    for K in KS:
        finished_search(tmp_path, "search_t4q", f"us_t4q_K{K}", "base" if K == 1 else f"sobol-{K:03d}")
    jobs = {j.id: j for j in rm.expand(eu, None, out, threads=2, smoke=False, max_jobs=23)}
    assert sorted(jobs) == sorted(f"search_eu_t4q_K{K}" for K in KS)
    for K in KS:
        j = jobs[f"search_eu_t4q_K{K}"]
        cmd = j.make_cmd()
        assert j.after == [] and cmd[cmd.index("--K") + 1] == str(K)
        assert (cmd[cmd.index("--keep") + 1] == f"sobol-{K:03d}") if K != 1 else "--keep" not in cmd
        assert cmd[cmd.index("--jobs") + 1] == ("23" if K == 100 else "9")
        assert cmd[cmd.index("--run-dir") + 1] == f"runs/_matrix/search_t4q_eu/searches/eu_t4q_K{K}"
