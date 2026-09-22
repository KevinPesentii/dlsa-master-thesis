"""One-step attention arbitrage with lambda_var chosen on validation data, as in the paper.

Section 3.3: "The tradeoff between these two objectives is selected optimally on the
validation data." run_attention_us.py fixes lambda_var from the config; this script picks
it per refit, without ever looking at the year being tested.

For each out-of-sample year Y, with the usual window [Y-W, Y):

    1. split it into a training part [Y-W, Y-V) and a validation part [Y-V, Y);
    2. for every lambda in the grid, train a fresh model on the training part and
       measure its net Sharpe on the validation part;
    3. keep the lambda with the best validation net Sharpe;
    4. retrain a fresh model with that lambda on the whole window [Y-W, Y) and trade Y.

Every candidate of a year, and its final model, start from the same initial weights and
see the training blocks in the same order, so the comparison is about lambda and not
about the draw. Training, spans and the objective are the functions of
run_attention_us.py, imported, so the two runs differ only in how lambda is set.

Cost: about (len(grid) * (W-V)/W + 1) times a fixed-lambda run. With the default grid of
four values, a one-year validation part and an 8-year window: 4.5 times, roughly 50
minutes for 1998-2021 on an M5 MacBook Pro.

    python scripts/run_attention_us_val.py --config configs/us_replication.yaml --K 30
                                           [--grid 0.1 1 10 100] [--val-years 1]
                                           [--years 2005 2006] [--epochs 30]

Caveat worth keeping in view: one validation year gives a noisy Sharpe (standard error
near 1), so the chosen lambda can jump between neighbouring values from year to year.
metrics.json records the chosen lambda and every validation score per year.
"""

from __future__ import annotations

import argparse
import copy
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

import run_attention_us as base  # noqa: E402  (span, train_window)
from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.model.attention_pipeline import AttentionArb, to_pool  # noqa: E402

DEFAULT_GRID = [0.1, 1.0, 10.0, 100.0]


def with_lambda(cfg: dict, lam: float) -> dict:
    c = copy.deepcopy(cfg)
    c["objective"]["lambda_var"] = float(lam)
    return c


def fresh_model(cfg: dict, K: int, n_features: int, init_seed: int) -> AttentionArb:
    mc, pc = cfg["model"], cfg["policy"]
    torch.manual_seed(init_seed)
    return AttentionArb(n_features=n_features, n_factors=K, embedding_dim=mc["embedding_dim"],
                        lambda_ridge=mc["lambda_ridge"], hidden=pc["hidden"],
                        lookback=pc["residual_lookback"], dropout=pc["dropout"],
                        lambda_squash=pc["lambda_squash"])


def run_K_val(K: int, cfg: dict, seed: int, p: slots.SlotPanel, years: list[int],
              grid: list[float], val_years: int, log) -> dict:
    pc, ob, ev = cfg["policy"], cfg["objective"], cfg["evaluation"]
    assert pc["layers"] == 1
    W = cfg["training"]["window_years"]
    if not 0 < val_years < W:
        raise SystemExit(f"--val-years must be between 1 and {W - 1}")
    run_dir = runs.create_run(f"attention_val_K{K}_s{seed}",
                              {**cfg, "K": K, "years": years, "lambda_grid": grid,
                               "validation_years": val_years}, seed, root=ROOT / "runs")
    log(f"K={K}: run dir {run_dir}")
    quiet = lambda s: None  # noqa: E731
    dates = p.dates
    w_slot_all, w_pool_all, t_all = [], [], []
    chosen, val_scores = {}, {}

    for year in years:
        t_tr0 = dates.searchsorted(pd.Timestamp(year - W, 1, 1))
        t_va0 = dates.searchsorted(pd.Timestamp(year - val_years, 1, 1))
        t_te0 = dates.searchsorted(pd.Timestamp(year, 1, 1))
        t_te1 = dates.searchsorted(pd.Timestamp(year + 1, 1, 1))
        if dates[t_tr0].year > year - W:
            raise SystemExit(f"{year}: the panel starts {dates[0]:%Y-%m-%d}, too late for a {W}-year window")
        init_seed = seed * 10007 + year
        t0 = time.time()

        scores = {}
        for lam in grid:
            c = with_lambda(cfg, lam)
            model = fresh_model(c, K, len(p.features), init_seed)
            base.train_window(model, p, t_tr0, t_va0, c, np.random.default_rng(init_seed), quiet)
            model.eval()
            with torch.no_grad():
                _, parts, _ = base.span(model, p, t_va0, t_te0, c)
            scores[lam] = float(metrics.annualised(parts["net"].numpy())["SR"])
        lam_star = max(grid, key=lambda g: scores[g])
        chosen[str(year)], val_scores[str(year)] = lam_star, {str(k): v for k, v in scores.items()}

        c = with_lambda(cfg, lam_star)
        model = fresh_model(c, K, len(p.features), init_seed)
        base.train_window(model, p, t_tr0, t_te0, c, np.random.default_rng(init_seed), quiet)
        model.eval()
        with torch.no_grad():
            _, parts, out = base.span(model, p, t_te0, t_te1, c)
        w = out["w"]
        w_slot_all.append(w.numpy())
        w_pool_all.append(to_pool(w, p.idx[t_te0:t_te1], p.n_pool).numpy())
        t_all.append(np.arange(t_te0, t_te1))
        score_txt = "  ".join(f"{g:g}:{scores[g]:+.2f}" for g in grid)
        log(f"  {year}: validation {dates[t_va0]:%Y}..{dates[t_te0 - 1]:%Y} net SR  {score_txt}"
            f"  ->  lambda {lam_star:g}   | test net SR "
            f"{metrics.annualised(parts['net'].numpy())['SR']:+.2f}  [{time.time() - t0:.0f}s]")

    t_idx = np.concatenate(t_all)
    wp = np.concatenate(w_pool_all)
    gross = (wp * p.R_pool[t_idx]).sum(axis=1)
    turnover = np.abs(np.diff(wp, axis=0, prepend=np.zeros((1, wp.shape[1]), dtype=wp.dtype))).sum(axis=1)
    short = np.clip(-wp, 0, None).sum(axis=1)
    daily = pd.DataFrame({"date": dates[t_idx], "gross": gross, "turnover": turnover, "short": short,
                          "n_traded": (wp != 0).sum(axis=1), "mkt_ew": p.mkt_ew[t_idx],
                          "rf": p.rf.numpy()[t_idx]})
    m = metrics.performance(daily, ob["turnover_cost"], ob["short_cost"], ev["cost_grid_bps"])
    m.update({"K": K, "seed": seed, "factor_model": "attention", "policy_input": pc["input"],
              "lambda_selection": "validation", "lambda_grid": grid, "validation_years": val_years,
              "lambda_by_year": chosen, "validation_net_SR_by_year": val_scores})
    runs.write_metrics(run_dir, m)
    daily.to_csv(run_dir / "oos_daily.csv", index=False)
    if ev.get("save_weights"):
        ws = np.concatenate(w_slot_all)
        tt, ss = np.nonzero(ws != 0)
        pd.DataFrame({"date": dates[t_idx[tt]], "sec_id": p.sec_ids[p.idx.numpy()[t_idx[tt], ss]],
                      "w": ws[tt, ss].astype(np.float32)}).to_parquet(run_dir / "weights.parquet", index=False)
    counts = pd.Series(list(chosen.values())).value_counts().sort_index()
    log(f"lambda chosen: " + ", ".join(f"{k:g} x{v}" for k, v in counts.items()))
    log(f"K={K:3d}  {metrics.table_row(m)}   turnover {m['turnover_daily']:.3f}  "
        f"short {m['short_exposure']:.3f}  break-even {m['break_even_turnover_cost_bps']:.1f}bp")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_replication.yaml")
    ap.add_argument("--K", type=int, nargs="*")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--data-dir")
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--val-years", type=int, default=1)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    if args.threads:
        torch.set_num_threads(args.threads)
    seed = cfg["seed"] if args.seed is None else args.seed
    Ks = args.K or [cfg["model"]["n_factors"]]
    y0, y1 = pd.Timestamp(cfg["sample"]["start"]).year, pd.Timestamp(cfg["sample"]["end"]).year
    years = args.years or list(range(y0, y1 + 1))

    log = lambda s: print(s, flush=True)  # noqa: E731
    data_dir = Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    t0 = time.time()
    p = slots.load_slots(data_dir, f"{min(years) - cfg['training']['window_years']}-01-01", f"{max(years)}-12-31")
    log(f"panel {tuple(p.X.shape)} from {data_dir} in {time.time() - t0:.0f}s, "
        f"{p.dates[0]:%Y-%m-%d}..{p.dates[-1]:%Y-%m-%d}, torch threads {torch.get_num_threads()}, "
        f"lambda grid {args.grid}, validation {args.val_years}y")
    log("      K    SR     mu   sigma   SRnet  munet signet   beta")
    for K in Ks:
        run_K_val(K, cfg, seed, p, years, args.grid, args.val_years, log)


if __name__ == "__main__":
    main()
