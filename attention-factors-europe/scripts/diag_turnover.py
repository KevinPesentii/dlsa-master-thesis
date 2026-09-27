"""Where does the turnover come from: the policy changing its mind, or the residuals moving?

The traded weight is w_t = (omega^eps_t)' w^port_t, and BOTH factors move every day: the
policy reacts to new residuals, and the composition omega^eps_t = I - beta_t' omega^F_t
is rebuilt from that date's characteristics. Total turnover is the L1 change of w, and it
splits exactly:

    w_t - w_{t-1} = (omega^eps_t)'(w^port_t - w^port_{t-1})      <- the policy
                  + ((omega^eps_t)' - (omega^eps_{t-1})') w^port_{t-1}   <- the composition

The script trains one window with the given config, then reports on its test year the
average L1 of each part, plus how concentrated the factor portfolios are (the effective
number of names behind each factor, 1 / sum of squared weights: 500 means equal weight
over the whole universe, a handful means the factor is a few names).

    python scripts/diag_turnover.py --config configs/us_replication.yaml --year 2015

Reads the same config as the runners, so it measures the model you actually ran.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_attention_us as base  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.model.attention_pipeline import AttentionArb, compose, residual_windows, to_pool  # noqa: E402


def l1_normalise(w: torch.Tensor) -> torch.Tensor:
    n = w.abs().sum(dim=1, keepdim=True)
    return w / torch.where(n > 0, n, torch.ones_like(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_replication.yaml")
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--K", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--data-dir")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    K = args.K or cfg["model"]["n_factors"]
    mc, pc, tr = cfg["model"], cfg["policy"], cfg["training"]
    W, L = tr["window_years"], pc["residual_lookback"]

    data_dir = Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    p = slots.load_slots(data_dir, f"{args.year - W}-01-01", f"{args.year}-12-31")
    dates = p.dates
    t_tr0 = dates.searchsorted(pd.Timestamp(args.year - W, 1, 1))
    t_te0 = dates.searchsorted(pd.Timestamp(args.year, 1, 1))
    t_te1 = len(dates)
    print(f"panel {tuple(p.X.shape)}  train {dates[t_tr0]:%Y-%m-%d}..{dates[t_te0 - 1]:%Y-%m-%d}  "
          f"test {dates[t_te0]:%Y-%m-%d}..{dates[t_te1 - 1]:%Y-%m-%d}")

    torch.manual_seed(cfg["seed"])
    model = AttentionArb(n_features=len(p.features), n_factors=K, embedding_dim=mc["embedding_dim"],
                         lambda_ridge=mc["lambda_ridge"], hidden=pc["hidden"], lookback=L,
                         dropout=pc["dropout"], lambda_squash=pc["lambda_squash"])
    t0 = time.time()
    base.train_window(model, p, t_tr0, t_te0, cfg, np.random.default_rng(cfg["seed"]),
                      lambda s: None)
    model.eval()
    print(f"trained in {time.time() - t0:.0f}s, lambda_var {cfg['objective']['lambda_var']}, "
          f"input {pc['input']}, batch_days {tr['batch_days']}")

    s = t_te0 - L
    with torch.no_grad():
        w_F, betaT, eps = model.factors(p.X[s:t_te1], p.in_universe[s:t_te1], p.R[s:t_te1])
        eps_pool = to_pool(eps, p.idx[s:t_te1], p.n_pool)
        valid_pool = to_pool(p.in_universe[s:t_te1].float(), p.idx[s:t_te1], p.n_pool) > 0
        windows, tradable = residual_windows(eps_pool, valid_pool, p.idx[s:t_te1], L,
                                             pc["input"] == "cumulative", pc["input_scale"])
        w_port = model.policy(windows) * tradable                      # (T, S), test dates
        wF, bT = w_F[L:], betaT[L:]
        w = l1_normalise(compose(w_port, wF, bT))                      # what we traded
        # counterfactual: yesterday's policy signal, today's composition. Shifted by company,
        # not by slot: slots are re-sorted by cap_rank every month, so the previous slot row
        # holds other companies on each month's first day. Names not tradable today get 0.
        idx_test = p.idx[t_te0:t_te1]
        pp = to_pool(w_port, idx_test, p.n_pool)
        pp_prev = torch.cat([pp[:1], pp[:-1]])
        pp_prev = torch.cat([pp_prev, pp_prev.new_zeros(pp_prev.shape[0], 1)], dim=1)   # padding column
        w_port_prev = torch.gather(pp_prev, 1, idx_test) * tradable
        w_mixed = l1_normalise(compose(w_port_prev, wF, bT))

        wp = to_pool(w, idx_test, p.n_pool)
        wm = to_pool(w_mixed, idx_test, p.n_pool)
        total = (wp[1:] - wp[:-1]).abs().sum(1)
        policy = (wp[1:] - wm[1:]).abs().sum(1)
        composition = (wm[1:] - wp[:-1]).abs().sum(1)

        eff = 1.0 / (wF ** 2).sum(-1)                                  # effective names per factor
        top = wF.max(-1).values
        top10 = wF.topk(10, dim=-1).values.sum(-1)                     # the paper: 10%-23%

    print()
    print(f"turnover medio giornaliero            {total.mean():.3f}")
    print(f"  di cui la policy cambia idea        {policy.mean():.3f}   ({100*policy.mean()/total.mean():.0f}% del totale)")
    print(f"  di cui la composizione si muove     {composition.mean():.3f}   ({100*composition.mean()/total.mean():.0f}% del totale)")
    print(f"  (le due parti sommano a piu' del totale quando si compensano)")
    print()
    print(f"fattori: numero efficace di titoli    mediana {eff.median():.0f} su {wF.shape[-1]} slot"
          f"   (min {eff.min():.0f}, max {eff.max():.0f})")
    print(f"         peso massimo su un titolo    mediana {top.median():.4f}   (max {top.max():.4f})")
    print(f"         peso dei primi 10 titoli     mediana {100*top10.median():.1f}%   "
          f"(il paper riporta 10%-23%; equipeso darebbe {1000.0/wF.shape[-1]:.1f}%)")


if __name__ == "__main__":
    main()
