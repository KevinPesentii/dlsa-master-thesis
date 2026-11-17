"""The pieces the thesis_v3 rerun added (2026-11-17): evaluation costs apart from the objective's,
two-step attention training, a search's n-th finalist, search variants, the ensemble validation
job, and the matrix itself."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_attention_us as base  # noqa: E402
import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402

MATRIX = yaml.safe_load((ROOT / "configs" / "thesis_v3_matrix.yaml").read_text())


# ------------------------------------------------------------------ evaluation costs

def test_evaluation_costs_default_to_the_objective_and_can_differ():
    cfg = {"objective": {"turnover_cost": 0.0, "short_cost": 0.0}, "evaluation": {}}
    assert metrics.eval_costs(cfg) == (0.0, 0.0)
    cfg["evaluation"].update(turnover_cost=0.0005, short_cost=0.0001)
    assert metrics.eval_costs(cfg) == (0.0005, 0.0001)
    assert metrics.eval_costs({"objective": {"turnover_cost": 5e-4, "short_cost": 1e-4}}) == (5e-4, 1e-4)


# ------------------------------------------------------------------ training

def small_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "us_replication.yaml").read_text())
    cfg["training"].update(epochs=2, batch_days=40)
    cfg["policy"].update(residual_lookback=10, hidden=8)
    cfg["model"].update(embedding_dim=8)
    return cfg


@pytest.fixture
def panel(tmp_path, returns, universe, features):
    returns.to_parquet(tmp_path / "returns.parquet", index=False)
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    return slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))


def params(module) -> list[torch.Tensor]:
    return [q.detach().clone() for q in module.parameters()]


def same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def reference_joint(model, p, t0, t1, cfg, rng):
    """The training loop as it was before two_step existed (commit 87fe9bf), without the logging."""
    tr = cfg["training"]
    opt = torch.optim.AdamW([{"params": model.policy.parameters(), "weight_decay": tr["weight_decay"]},
                             {"params": model.factors.parameters(), "weight_decay": 0.0}], lr=tr["lr"])
    starts = list(range(t0 + model.lookback, t1, tr["batch_days"]))
    blocks = [(b0, min(b0 + tr["batch_days"], t1)) for b0 in starts]
    blocks = [(b0, b1) for b0, b1 in blocks if b1 - b0 >= 20]
    for _ in range(tr["epochs"]):
        model.train()
        rng.shuffle(blocks)
        for b0, b1 in blocks:
            opt.zero_grad()
            loss, _, _ = base.span(model, p, b0, b1, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()


def test_joint_training_is_the_loop_of_before(panel):
    cfg, T = small_cfg(), len(panel.dates)
    cfg["model"]["score_std_target"] = None
    a = sv.val.fresh_model(cfg, 3, len(panel.features), 7)          # seeds torch's generator (dropout)
    base.train_window(a, panel, 0, T - 60, cfg, np.random.default_rng(7), lambda s: None)
    b = sv.val.fresh_model(cfg, 3, len(panel.features), 7)
    reference_joint(b, panel, 0, T - 60, cfg, np.random.default_rng(7))
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(x, y), name


def test_two_step_step_one_moves_only_the_factors(panel):
    cfg, T = small_cfg(), len(panel.dates)
    model = sv.val.fresh_model(cfg, 3, len(panel.features), 7)
    f0, p0 = params(model.factors), params(model.policy)
    starts = list(range(model.lookback, T - 60, 40))
    blocks = [(b0, b1) for b0, b1 in ((b0, min(b0 + 40, T - 60)) for b0 in starts) if b1 - b0 >= 20]
    opt = torch.optim.AdamW(model.factors.parameters(), lr=cfg["training"]["lr"], weight_decay=0.0)
    base._epochs(model, panel, cfg, blocks, np.random.default_rng(7), lambda s: None, None, opt,
                 lambda loss, parts: -parts["ev"], None, "factors")
    assert same(params(model.policy), p0)
    assert not same(params(model.factors), f0)


def test_two_step_step_two_keeps_the_factors_frozen(panel):
    cfg, T = small_cfg(), len(panel.dates)
    cfg["training"]["two_step"] = True
    model = sv.val.fresh_model(cfg, 3, len(panel.features), 7)
    seen = []
    base.train_window(model, panel, 0, T - 60, cfg, np.random.default_rng(7), lambda s: None,
                      lambda epoch, m: seen.append((epoch, params(m.factors), params(m.policy))))
    assert [e for e, _, _ in seen] == [1, 2]                 # on_epoch runs in step two only
    assert same(seen[0][1], seen[1][1]) and same(seen[1][1], params(model.factors))
    assert not same(seen[0][2], seen[1][2])                  # the policy trains
    assert all(not q.requires_grad for q in model.factors.parameters())


# ------------------------------------------------------------------ run_matrix

def fake_search(d: Path, finalists: list[dict], fixed: dict | None = None) -> Path:
    d.mkdir(parents=True)
    S = {"base_config": "configs/us_replication.yaml", "fixed": fixed or {"policy.input": "cumulative"}, "K": 30,
         "first_oos_year": 1998, "validation_years": 2,
         "search": {"space": {"objective.lambda_var": {"log": [0.1, 300]}}}}
    base_cfg = yaml.safe_load((ROOT / S["base_config"]).read_text())
    (d / "manifest.json").write_text(json.dumps({"config": {"search": S, "base": base_cfg}}, default=str))
    (d / "metrics.json").write_text(json.dumps({"winner": finalists[0], "finalists": finalists}))
    (d / "selected.yaml").write_text("{}")
    return d


FINALISTS = [{"name": f"sobol-00{i}", "overrides": {"objective.lambda_var": float(i)}, "val_net_SR": 5.0 - i}
             for i in (1, 2, 3, 4)]


def test_the_nth_finalist_is_the_nth_row_of_the_ranking(tmp_path):
    d = fake_search(tmp_path / "m" / "searches" / "s", FINALISTS)
    assert rm.ranked(d, 1)["name"] == "sobol-001" and rm.ranked(d, 3)["name"] == "sobol-003"
    w, values = rm.selected_values(d, 2)
    assert w["name"] == "sobol-002" and values["objective.lambda_var"] == 2.0
    assert values["policy.input"] == "cumulative"            # fixed keys travel with every rank
    with pytest.raises(SystemExit):
        rm.ranked(d, 5)
    smoke = fake_search(tmp_path / "m_smoke" / "searches" / "s", FINALISTS[:2])
    assert rm.ranked(smoke, 3)["name"] == "sobol-002"         # a smoke search stands in its last


def test_a_search_variant_adds_its_fixed_keys(tmp_path):
    job = rm.search_job("us_lag1", {"config": "configs/us_search_t4_quick.yaml", "K": 30, "workers": 10,
                                    "fixed": {"execution.lag": 1}},
                        tmp_path, 2, False)
    assert job.after == [] and job.slots == 10
    S = yaml.safe_load((tmp_path / "configs" / "search_us_lag1.yaml").read_text())
    orig = yaml.safe_load((ROOT / "configs" / "us_search_t4_quick.yaml").read_text())
    assert S["fixed"] == {**orig["fixed"], "execution.lag": 1} and S["search"] == orig["search"]


def test_the_ensemble_job_takes_the_top_finalists_in_order(tmp_path):
    out = tmp_path / "runs" / "_matrix" / "m"
    fake_search(out / "searches" / "us_t4q_K30", FINALISTS, fixed={"data.returns_history": True})
    job = rm.ensemble_job("us_ens", {"search": "us_t4q_K30", "top": 3, "seeds": [0, 1, 2, 3, 4],
                                     "compare": ["seeds", "all"]}, out, 2, False)
    cmd = job.make_cmd()
    path = Path(cmd[cmd.index("--config") + 1])
    E = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text())
    assert list(E["candidates"]) == ["sobol-001", "sobol-002", "sobol-003"]
    assert E["fixed"] == {"data.returns_history": True} and E["compare"] == ["seeds", "all"]
    assert job.after == ["search_us_t4q_K30"] and job.slots == 15


# ------------------------------------------------------------------ the matrix

def search_config(name: str) -> dict:
    s = MATRIX["searches"][name]
    S = yaml.safe_load((ROOT / s["config"]).read_text())
    S["fixed"] = {**(S.get("fixed") or {}), **(s.get("fixed") or {})}
    return S


def test_every_row_trades_with_a_search_of_the_matrix():
    for spec in MATRIX["specs"]:
        assert spec["hyper"] in MATRIX["searches"], spec["name"]
        assert int(spec.get("hyper_rank", 1)) <= search_config(spec["hyper"])["finalists"]["top"], spec["name"]
    for e in MATRIX["ensembles"].values():
        assert e["search"] in MATRIX["searches"] and set(e["compare"]) <= {"seeds", "configs", "all"}


@pytest.mark.parametrize("name", list(MATRIX["searches"]))
def test_every_search_names_only_keys_of_its_base_config(name):
    S = search_config(name)
    sv.with_overrides(yaml.safe_load((ROOT / S["base_config"]).read_text()), S["fixed"])


def test_every_row_saves_what_the_analyses_read():
    for spec in MATRIX["specs"]:
        ev = yaml.safe_load((ROOT / spec["config"]).read_text())["evaluation"]
        assert ev.get("save_weights"), spec["name"]
        if spec["runner"] == "attention":
            assert ev.get("save_models") and ev.get("diagnostics"), spec["name"]


def test_the_gross_rows_train_without_costs_and_report_after_them():
    for name in ("us_t4q_gross_K30", "eu_t4q_gross_K30"):
        assert search_config(name)["fixed"]["objective.turnover_cost"] == 0.0
        assert search_config(name)["fixed"]["objective.short_cost"] == 0.0
    rows = [s for s in MATRIX["specs"] if "gross" in s["name"]]
    assert len(rows) == 2
    for s in rows:
        assert s["set"] == {"evaluation.turnover_cost": 0.0005, "evaluation.short_cost": 0.0001}
