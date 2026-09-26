"""scripts/search_validation.py: scoring the validation years between epochs must leave the
training path untouched (the learning curve is only free if it is), and the Sobol draws must
stay inside the pre-registered space."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_validation as sv  # noqa: E402
from afe.data import slots  # noqa: E402


def small_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "us_replication.yaml").read_text())
    cfg["training"].update(epochs=3, batch_days=40)
    cfg["policy"].update(residual_lookback=10, hidden=8)
    cfg["model"].update(embedding_dim=8)
    return cfg


def test_scoring_between_epochs_leaves_training_unchanged(tmp_path, returns, universe, features):
    returns.to_parquet(tmp_path / "returns.parquet", index=False)
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    p = slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))
    cfg, T = small_cfg(), len(p.dates)
    sv.WORKER.update(p=p, split=(0, T - 60, T))

    def train(hook):
        model = sv.val.fresh_model(cfg, 3, len(p.features), 7)
        sv.base.train_window(model, p, 0, T - 60, cfg, np.random.default_rng(7), lambda s: None, hook)
        return model

    scores = []
    plain = train(None)
    hooked = train(lambda epoch, m: scores.append(sv.score(m, cfg)))
    assert len(scores) == 3 and all(np.isfinite(s["net_SR"]) for s in scores)
    for (name, a), b in zip(plain.state_dict().items(), hooked.state_dict().values()):
        assert torch.equal(a, b), name


def test_sobol_draws_stay_in_the_space():
    space = {"objective.lambda_var": {"log": [0.1, 300]},
             "model.score_std_target": {"log": [0.3, 3], "off": 0.25},
             "policy.input": {"choice": ["cumulative", "raw"]}}
    pts = sv.sobol_points(space, 64, 0)
    lam = np.array([q["objective.lambda_var"] for q in pts])
    assert 0.1 <= lam.min() and lam.max() <= 300 and len(set(lam)) == 64
    tau = [q["model.score_std_target"] for q in pts]
    assert sum(t is None for t in tau) == 16            # a quarter, exactly: 64 points are balanced
    assert all(t is None or 0.3 <= t <= 3 for t in tau)
    assert sum(q["policy.input"] == "raw" for q in pts) == 32
    assert pts == sv.sobol_points(space, 64, 0)
    with pytest.raises(SystemExit):
        sv.sobol_points(space, 60, 0)


def test_overrides_reject_unknown_keys():
    cfg = small_cfg()
    assert sv.with_overrides(cfg, {"objective.lambda_var": 1.0})["objective"]["lambda_var"] == 1.0
    assert cfg["objective"]["lambda_var"] == 100
    with pytest.raises(KeyError):
        sv.with_overrides(cfg, {"objective.lambda_vra": 1.0})
