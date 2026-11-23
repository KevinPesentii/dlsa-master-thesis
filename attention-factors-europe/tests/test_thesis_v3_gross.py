"""The gross-objective cells of the mechanism table (configs/thesis_v3_gross_matrix.yaml, 2026-11-23):
the design in its header, read off the matrix against thesis_v3's. Config checks, no data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from afe.policy import trading  # noqa: E402

V3 = yaml.safe_load((ROOT / "configs" / "thesis_v3_matrix.yaml").read_text())
GROSS = yaml.safe_load((ROOT / "configs" / "thesis_v3_gross_matrix.yaml").read_text())
NO_COSTS = {"objective.turnover_cost": 0.0, "objective.short_cost": 0.0}
# the thesis_v3 search each new one is (config, K, carried winner) but for the keys held on top
SAME_AS = {"us_t4q_twostep_gross_K30": "us_t4q_twostep_K30", "eu_t4q_twostep_gross_K30": "eu_t4q_twostep_K30",
           "us_pcaq_gross_K30": "us_pcaq_K30", "eu_pcaq_gross_K30": "eu_pcaq_K30"}


def spec(matrix: dict, name: str) -> dict:
    return next(s for s in matrix["specs"] if s["name"] == name)


def search_config(matrix: dict, name: str) -> dict:
    s = matrix["searches"][name]
    S = yaml.safe_load((ROOT / s["config"]).read_text())
    S["fixed"] = {**(S.get("fixed") or {}), **(s.get("fixed") or {})}
    return S


def test_four_searches_and_twenty_runs():
    assert set(GROSS["searches"]) == set(SAME_AS)
    assert len(GROSS["specs"]) == 4
    assert all(GROSS["sets"][s["K"]] == [30] and GROSS["sets"][s["seeds"]] == [0, 1, 2, 3, 4] for s in GROSS["specs"])
    assert GROSS["name"] != V3["name"]


@pytest.mark.parametrize("new,old", list(SAME_AS.items()))
def test_a_gross_search_is_the_v3_search_with_the_costs_at_zero(new, old):
    a, b = search_config(GROSS, new), search_config(V3, old)
    assert GROSS["searches"][new]["config"] == V3["searches"][old]["config"]
    assert GROSS["searches"][new]["K"] == V3["searches"][old]["K"] == 30
    assert a["search"] == b["search"] and a["finalists"] == b["finalists"]      # same points, same names
    assert a["fixed"] == {**b["fixed"], **NO_COSTS}                             # two_step kept, costs added
    assert ("training.two_step" in a["fixed"]) == ("twostep" in new)


def test_europe_carries_the_us_winner_of_its_own_kind():
    for eu in (n for n in SAME_AS if n.startswith("eu_")):
        assert GROSS["searches"][eu]["keep_from"] == eu.replace("eu_", "us_", 1)


@pytest.mark.parametrize("new,old", [("us_attention_K30_two_step_gross_trained", "us_attention_K30_two_step"),
                                     ("eu_attention_K30_two_step_gross_trained", "eu_attention_K30_two_step"),
                                     ("us_pca_longconv_K30_gross_trained", "us_pca_longconv_perk_K30"),
                                     ("eu_pca_longconv_K30_gross_trained", "eu_pca_longconv_perk_K30")])
def test_a_row_is_its_v3_twin_with_its_own_search_and_the_reported_costs(new, old):
    a, b = spec(GROSS, new), spec(V3, old)
    assert (a["runner"], a["config"], a["K"], a["seeds"]) == (b["runner"], b["config"], b["K"], b["seeds"])
    assert a["hyper"] in GROSS["searches"] and SAME_AS[a["hyper"]] == b["hyper"]
    assert a.get("after") == b.get("after")                    # the PCA rows wait for stage one
    assert a["set"] == {"evaluation.turnover_cost": 0.0005, "evaluation.short_cost": 0.0001}   # 5bp / 1bp
    # the same reported costs as thesis_v3's gross-trained attention rows
    assert a["set"] == spec(V3, new.split("_")[0] + "_attention_K30_gross_trained")["set"]


@pytest.mark.parametrize("name", list(SAME_AS))
def test_every_held_key_exists_in_the_base_config(name):
    S = search_config(GROSS, name)
    base = yaml.safe_load((ROOT / S["base_config"]).read_text())
    assert base["objective"]["turnover_cost"] == 0.0005 and base["objective"]["short_cost"] == 0.0001   # held at 0 here
    c = sv.with_overrides(base, S["fixed"])                    # raises on a key the base config lacks
    assert c["objective"]["turnover_cost"] == c["objective"]["short_cost"] == 0.0


def test_a_job_of_the_new_search_writes_costs_of_zero_into_the_search_config(tmp_path):
    s = GROSS["searches"]["us_pcaq_gross_K30"]
    job = rm.search_job("us_pcaq_gross_K30", s, tmp_path, 2, False, jobs=23)
    assert job.after == ["prep_pca_us"] and job.slots == 8
    S = yaml.safe_load((tmp_path / "configs" / "search_us_pcaq_gross_K30.yaml").read_text())
    assert S["fixed"]["objective.turnover_cost"] == 0.0 and S["fixed"]["objective.short_cost"] == 0.0


def slot_batch(R: torch.Tensor) -> trading.SlotBatch:
    T, S = R.shape
    ones = torch.ones(T, S)
    return trading.SlotBatch(windows=ones[..., None], tradable=ones.bool(), V=ones[..., None], B=0 * ones[..., None],
                             invvol=ones, idx=torch.arange(S).repeat(T, 1), R=R, rf=torch.zeros(T), n_pool=S)


def test_zero_costs_make_the_training_return_the_gross_return():
    """With both costs 0, trading.net_returns returns the gross return whatever the book trades, so the
    Sharpe loss train_one_window reads is the gross Sharpe ratio; at 5bp / 1bp it is not."""
    rng = torch.Generator().manual_seed(0)
    R = 0.01 * torch.randn(60, 8, generator=rng)
    b = slot_batch(R)
    w = trading.compose(torch.randn(60, 8, generator=rng), b)
    gross, cost, net, turnover, _, _ = trading.net_returns(w, b, 0.0, 0.0)
    assert (turnover[1:] > 0).all() and torch.equal(net, gross) and torch.equal(cost, torch.zeros(60))
    assert not torch.equal(trading.net_returns(w, b, 5e-4, 1e-4)[2], gross)
    valid = torch.ones(60, dtype=torch.bool)
    loss = trading.sharpe_loss(net, b.rf, valid, False)
    assert torch.isclose(loss, -(gross.mean() / (gross.std() + 1e-12)))
