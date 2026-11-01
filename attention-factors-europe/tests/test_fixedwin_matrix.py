"""The fixed-window matrix (configs/exp_fixedwin_matrix.yaml) claims two rows: the US headline unchanged,
and the headline with policy.residual_window fixed and nothing else."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
from test_gap2_matrix import changed, load  # noqa: E402


def test_rows_are_the_headline_and_the_headline_with_a_fixed_window():
    matrix, base = load("exp_fixedwin_matrix"), load("us_headline")
    rows = {s["name"]: s for s in matrix["specs"]}
    assert set(rows) == {"us_headline", "us_fixedwin"}
    assert changed(rm.resolve(rows["us_headline"], matrix), base) == {}
    assert changed(rm.resolve(rows["us_fixedwin"], matrix), base) == {"policy.residual_window": "fixed"}
    assert all(s["K"] == "K_few" and s["seeds"] == "seeds2" for s in rows.values())


RUNS = sorted(f"{r}_K{K}_s{s}" for r in ("us_headline", "us_fixedwin") for K in (8, 30, 100) for s in (0, 1))


def test_matrix_expands_to_twelve_jobs(tmp_path):
    matrix = yaml.safe_load((ROOT / "configs" / "exp_fixedwin_matrix.yaml").read_text())
    jobs = rm.expand(matrix, None, tmp_path, threads=2, smoke=False)
    assert sorted(j.id for j in jobs) == RUNS


def test_split_years_is_one_job_per_year_and_a_merge_under_the_run_id(tmp_path):
    """24 one-year jobs per run (1998-2021), each with --years <its year>, and a merge job that
    keeps the run's id, waits for exactly those 24, and is the only one the summary reads."""
    matrix = yaml.safe_load((ROOT / "configs" / "exp_fixedwin_matrix.yaml").read_text())
    jobs = rm.expand(matrix, None, tmp_path, threads=2, smoke=False, split_years=True)
    merges = {j.id: j for j in jobs if j.year is None}
    assert sorted(merges) == RUNS
    years = list(range(1998, 2022))
    for run_id, m in merges.items():
        assert m.after == [f"{run_id}_y{y}" for y in years]
    by_id = {j.id: j for j in jobs}
    assert len(jobs) == 12 * 25
    for j in jobs:
        if j.year is not None:
            assert j.cmd[j.cmd.index("--years") + 1] == str(j.year)
            assert j.cmd.count("--years") == 1 and "--epochs" not in j.cmd
            assert by_id[j.id.rsplit("_y", 1)[0]].K == j.K
    smoke = rm.expand(matrix, None, tmp_path, threads=2, smoke=True, split_years=True)
    assert sorted(j.id for j in smoke) == sorted(f"{r}_K8_s0{y}" for r in ("us_headline", "us_fixedwin")
                                                 for y in ("", "_y1998", "_y1999"))


def test_merge_job_reads_its_year_runs_from_the_ledger_and_names_the_merged_dir(tmp_path):
    matrix = yaml.safe_load((ROOT / "configs" / "exp_fixedwin_matrix.yaml").read_text())
    jobs = rm.expand(matrix, None, tmp_path, threads=2, smoke=True, split_years=True)
    merge = next(j for j in jobs if j.id == "us_fixedwin_K8_s0")
    ledger = [{"job": "us_fixedwin_K8_s0_y1999", "status": "ok", "run_dir": "runs/b"},
              {"job": "us_headline_K8_s0_y1998", "status": "ok", "run_dir": "runs/x"},
              {"job": "us_fixedwin_K8_s0_y1998", "status": "ok", "run_dir": "runs/a"}]
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(e) for e in ledger))
    assert merge.make_cmd()[-3:] == ["scripts/merge_year_runs.py", "runs/a", "runs/b"]
    log = tmp_path / "merge.log"
    log.write_text("  seam 1998-12-31 -> 1999-01-04: turnover 1.000 -> 0.412\n"
                   "  K=  8  ...\n  -> runs/20261101T170000Z_attention_K8_s0_merged\n")
    assert rm.run_dir_of(log) == "runs/20261101T170000Z_attention_K8_s0_merged"
    log.write_text("K=8: run dir /home/ubuntu/dlsa/runs/20261101T170000Z_attention_K8_s0\n  1998: train\n")
    assert rm.run_dir_of(log) == "/home/ubuntu/dlsa/runs/20261101T170000Z_attention_K8_s0"


def test_full_matrix_is_the_thesis_headline_row_twice_once_with_the_fixed_window(tmp_path):
    """Sequential rows: the baseline resolves exactly like us_attention of the thesis matrix (same
    config, K grid, seeds, save_models), the fixed row like it plus residual_window fixed."""
    full, thesis = load("exp_fixedwin_full_matrix"), load("thesis_matrix")
    rows = {s["name"]: s for s in full["specs"]}
    assert set(rows) == {"us_attention", "us_attention_fixedwin"}
    ref = next(s for s in thesis["specs"] if s["name"] == "us_attention")
    ref_cfg = {k: v for k, v in rm.resolve(ref, thesis).items() if k != "matrix"}
    for name, extra in (("us_attention", {}), ("us_attention_fixedwin", {"policy.residual_window": "fixed"})):
        s = rows[name]
        assert full["sets"][s["K"]] == thesis["sets"][ref["K"]] and full["sets"][s["seeds"]] == thesis["sets"][ref["seeds"]]
        assert changed(rm.resolve(s, full), ref_cfg) == extra, name
    jobs = rm.expand(full, None, tmp_path, threads=2, smoke=False)
    runs_ = [j for j in jobs if j.spec is not None]
    assert len(runs_) == 2 * 7 * 5 and all(j.year is None for j in runs_)
    assert [j.id for j in jobs if j.spec is None] == ["search_us_fixedwin"]
    assert all(j.after == (["search_us_fixedwin"] if j.spec["name"] == "us_attention_fixedwin" else [])
               for j in runs_)


def test_fixed_row_takes_the_winner_of_us_search_with_only_the_window_held_fixed():
    """The pre-registered search of the fixed row is the headline's search (configs/us_search.yaml:
    space, window, budget, rule) with residual_window held at fixed, and nothing else."""
    full, search, ref = load("exp_fixedwin_full_matrix"), load("us_search_fixedwin"), load("us_search")
    rows = {s["name"]: s for s in full["specs"]}
    assert rows["us_attention_fixedwin"]["hyper"] == "us_fixedwin" and "hyper" not in rows["us_attention"]
    assert full["searches"]["us_fixedwin"]["config"] == "configs/us_search_fixedwin.yaml"
    assert search.pop("fixed") == {"policy.residual_window": "fixed"} and "fixed" not in ref
    assert search == ref
    import search_validation as sv
    base = load("us_replication")
    assert base["policy"]["residual_window"] == "rolling"     # the default, so the fixed key is valid
    assert sv.with_overrides(base, {"policy.residual_window": "fixed"})["policy"]["residual_window"] == "fixed"
