"""The Sharpe half of the training objective is mean(net) / sd(net), without the risk-free
rate (decided 2026-11-17): the book is roughly self-financing, and the validation score and
the reported Sharpe ratios (evaluation.metrics) subtract no rf either."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from afe.evaluation import metrics
from afe.policy.trading import sharpe_loss

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _net_rf_valid():
    g = torch.Generator().manual_seed(0)
    net = 0.001 + 0.01 * torch.randn(300, generator=g, dtype=torch.float64)
    rf = torch.full((300,), 2e-4, dtype=torch.float64)
    valid = torch.ones(300, dtype=torch.bool)
    valid[:20] = False
    return net, rf, valid


def test_without_rf_the_loss_is_minus_mean_over_sd_and_ignores_rf():
    net, rf, valid = _net_rf_valid()
    r = net[valid]
    loss = sharpe_loss(net, rf, valid, subtract_rf=False)
    assert torch.isclose(loss, -(r.mean() / r.std()), rtol=1e-9)
    assert torch.equal(loss, sharpe_loss(net, 10 * rf, valid, subtract_rf=False))


def test_without_rf_the_loss_matches_the_evaluation_sharpe():
    net, rf, valid = _net_rf_valid()
    r = net[valid].numpy()
    loss = sharpe_loss(net, rf, valid, subtract_rf=False).item()
    assert np.isclose(-loss * np.sqrt(metrics.ANN), metrics.annualised(r)["SR"], rtol=1e-9)


def test_the_paper_variant_still_subtracts_rf():
    net, rf, valid = _net_rf_valid()
    r = net[valid]
    loss = sharpe_loss(net, rf, valid, subtract_rf=True)
    assert torch.isclose(loss, -((r - 2e-4).mean() / r.std()), rtol=1e-9)


def _subtract_rf_values(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "subtract_rf":
                yield v
            else:
                yield from _subtract_rf_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from _subtract_rf_values(v)


def test_no_config_subtracts_rf_in_the_objective():
    seen = {}
    for f in sorted(CONFIGS.glob("*.yaml")):
        vals = list(_subtract_rf_values(yaml.safe_load(f.read_text(encoding="utf-8"))))
        if vals:
            seen[f.name] = vals
    assert len(seen) >= 10, "the ten base configs set objective.subtract_rf"
    assert all(v is False for vals in seen.values() for v in vals), seen
