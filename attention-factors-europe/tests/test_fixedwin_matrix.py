"""The fixed-window matrix (configs/exp_fixedwin_matrix.yaml) claims two rows: the US headline unchanged,
and the headline with policy.residual_window fixed and nothing else."""

from __future__ import annotations

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


def test_matrix_expands_to_twelve_jobs(tmp_path):
    matrix = yaml.safe_load((ROOT / "configs" / "exp_fixedwin_matrix.yaml").read_text())
    jobs = rm.expand(matrix, None, tmp_path, threads=2, smoke=False)
    assert sorted(j.id for j in jobs) == sorted(f"{r}_K{K}_s{s}" for r in ("us_headline", "us_fixedwin")
                                               for K in (8, 30, 100) for s in (0, 1))
