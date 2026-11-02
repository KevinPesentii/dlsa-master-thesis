"""Seed ensemble against config ensemble (scripts/ensemble_validation.py, configs/us_ensemble_val*.yaml,
configs/exp_ensemble_k30_matrix.yaml): which models each book averages, what averaging does to a book,
and that the candidates are the searches' finalists as the headline and the matrix use them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ensemble_validation as ev  # noqa: E402
import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from test_gap2_matrix import changed, flat, load  # noqa: E402

CANDS, SEEDS = ["a", "b", "c", "d", "e"], [0, 1, 2, 3, 4]


def test_seed_books_hold_one_candidate_and_config_books_one_seed():
    books = ev.ensembles(CANDS, SEEDS)
    for c in CANDS:
        assert sorted(books[f"seeds:{c}"]) == [(c, s) for s in SEEDS]
    for s in SEEDS:
        assert sorted(books[f"configs:s{s}"]) == [(c, s) for c in CANDS]
    assert sorted(books["all"]) == sorted((c, s) for c in CANDS for s in SEEDS)


def test_latin_books_use_every_candidate_once_and_never_a_shared_seed():
    books = ev.ensembles(CANDS, SEEDS)
    latin = [books[f"configs:latin{r}"] for r in range(5)]
    for b in latin:
        assert sorted(c for c, _ in b) == CANDS and sorted(s for _, s in b) == SEEDS
    assert len({m for b in latin for m in b}) == 25           # together: every model exactly once


def test_average_of_copies_is_the_book_and_rescaling_restores_unit_exposure():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 9))
    w /= np.abs(w).sum(axis=1, keepdims=True)
    assert np.allclose(ev.average([w] * 5, rescale=False), w)
    half = ev.average([w, np.zeros_like(w)], rescale=False)
    assert np.allclose(np.abs(half).sum(axis=1), 0.5)
    assert np.allclose(ev.average([w, np.zeros_like(w)], rescale=True), w)
    flat = ev.average([w, -w], rescale=True)                  # seeds that cancel hold nothing, not NaN
    assert np.isfinite(flat).all() and not flat.any()


def test_the_first_candidate_of_each_config_is_the_headline():
    head = load("us_headline")
    for name, fixed in (("us_ensemble_val", {}), ("us_ensemble_val_fixedwin", {"policy.residual_window": "fixed"})):
        E = load(name)
        assert E["base_config"] == "configs/us_replication.yaml" and E["fixed"] == fixed and E["K"] == 30
        assert E["seeds"] == SEEDS and len(E["candidates"]) == 5
        winner, values = next(iter(E["candidates"].items()))
        assert winner == "sobol-036"
        cfg = sv.with_overrides(sv.with_overrides(load("us_replication"), fixed), values)
        assert changed(cfg, head) == {"policy.residual_window": fixed.get("policy.residual_window", "rolling")}


def test_matrix_runs_every_other_candidate_with_its_values(tmp_path):
    matrix, base = load("exp_ensemble_k30_matrix"), load("us_replication")
    specs = {s["name"]: s for s in matrix["specs"]}
    for name, group, extra in (("us_ensemble_val", "rolling", {}),
                               ("us_ensemble_val_fixedwin", "fixed", {"policy.residual_window": "fixed"})):
        others = {c: v for c, v in load(name)["candidates"].items() if c != "sobol-036"}
        rows = {s["use"][-1]: s for s in specs.values() if s["group"] == group}
        assert set(rows) == set(others)
        for c, values in others.items():
            assert matrix["blocks"][c] == values
            cfg, want = rm.resolve(rows[c], matrix), {**values, **extra}
            assert {k: flat(cfg)[k] for k in want} == want, c
            assert set(changed(cfg, base)) <= set(want), c
            assert rows[c]["K"] == [30] and rows[c]["seeds"] == "seeds5"
    jobs = rm.expand(yaml.safe_load((ROOT / "configs" / "exp_ensemble_k30_matrix.yaml").read_text()), None,
                     tmp_path, threads=2, smoke=False)
    assert len(jobs) == 40 and all(j.K == 30 for j in jobs)
