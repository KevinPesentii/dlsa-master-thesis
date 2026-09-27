"""One-step attention arbitrage on the real panel: the headline specification of Table 2.

    python scripts/run_attention_us.py [--config configs/us_replication.yaml] [--K 30]
                                       [--seed 0] [--years 1998 1999] [--epochs 2]
                                       [--data-dir ../../attention-factors-europe/data/us]

Rolling `window_years` training windows refit every January, out-of-sample from
sample.start to sample.end, one run directory runs/<stamp>_attention_K<K>_s<seed>/ with
the manifest, the daily out-of-sample series, the asset-space weights and metrics.json
in the units of Table 2. The training loop is that of run_attention_synthetic.py; the
panel comes from data/slots.py instead of the generator, and the schedule and the
evaluation are those of run_pca_longconv.py, so the two Table 2 rows are comparable.

Runs against this clone's src/ whatever `afe` is installed in the environment, and
against `data.dir` of the config relative to this clone unless --data-dir says otherwise.
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

from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.model.attention_pipeline import AttentionArb, objective, slot_batch, to_pool  # noqa: E402


def span(model, p: slots.SlotPanel, t0: int, t1: int, cfg: dict, w_prev=None):
    """Trade dates t0..t1-1; the span carries `lookback` extra dates in front, which feed
    the residual windows and are not traded. Returns the loss, its parts and the weights."""
    L, pc, ob = model.lookback, cfg["policy"], cfg["objective"]
    s = t0 - L
    out = model.forward_span(p.X[s:t1], p.R[s:t1], p.in_universe[s:t1], p.idx[s:t1], p.n_pool,
                             cumulative=pc["input"] == "cumulative", scale=pc["input_scale"])
    b = slot_batch(p.R[t0:t1], p.idx[t0:t1], p.rf[t0:t1], out["tradable"], p.n_pool)
    loss, parts = objective(out, b, ob["turnover_cost"], ob["short_cost"], ob["lambda_var"],
                            ob["subtract_rf"], w_prev)
    return loss, parts, out


def train_window(model, p: slots.SlotPanel, t_tr0: int, t_te0: int, cfg: dict,
                 rng: np.random.Generator, log, on_epoch=None) -> None:
    """`on_epoch(epoch, model)`, if given, runs after every epoch (1-based), e.g. to score a
    validation span in eval mode; tests/test_search_validation.py checks that this leaves
    the training path unchanged."""
    tr = cfg["training"]
    # Table 4 of the paper: weight decay 0.05 is "Adam weight decay in LongConv model", so it
    # applies to the sequence model only. The attention factor parameters (Q, W_K) are not
    # decayed; decaying them would shrink the scores and flatten the softmax towards equal
    # weights, i.e. towards less discriminating factors.
    opt = torch.optim.AdamW([
        {"params": model.policy.parameters(), "weight_decay": tr["weight_decay"]},
        {"params": model.factors.parameters(), "weight_decay": 0.0},
    ], lr=tr["lr"])
    target = cfg["model"].get("score_std_target")
    if target:
        # Calibrated on the training dates of this window only; see
        # AttentionFactors.calibrate_temperature for why the scores need a scale at all.
        tau = model.factors.calibrate_temperature(p.X[t_tr0:t_te0], p.in_universe[t_tr0:t_te0],
                                                  float(target))
        log(f"      score temperature calibrated to {tau:.1f} (target spread {target})")
    starts = list(range(t_tr0 + model.lookback, t_te0, tr["batch_days"]))
    blocks = [(b0, min(b0 + tr["batch_days"], t_te0)) for b0 in starts]
    blocks = [(b0, b1) for b0, b1 in blocks if b1 - b0 >= 20]
    for epoch in range(tr["epochs"]):
        model.train()
        rng.shuffle(blocks)
        tot, ev = 0.0, 0.0
        for b0, b1 in blocks:
            opt.zero_grad()
            loss, parts, _ = span(model, p, b0, b1, cfg)
            loss.backward()
            if tr.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            tot += loss.item()
            ev += parts["ev"].item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            log(f"      epoch {epoch + 1:2d}: loss {tot / len(blocks):+.3f}  "
                f"(sharpe part {parts['sharpe_loss'].item():+.3f}, EV {ev / len(blocks):.3f})")
        if on_epoch is not None:
            on_epoch(epoch + 1, model)


def run_K(K: int, cfg: dict, seed: int, p: slots.SlotPanel, years: list[int], log) -> dict:
    mc, pc, ob, ev = cfg["model"], cfg["policy"], cfg["objective"], cfg["evaluation"]
    assert pc["layers"] == 1, "AttentionArb reads the LongConv layer at the last position; one layer only"
    run_dir = runs.create_run(f"attention_K{K}_s{seed}", {**cfg, "K": K, "years": years}, seed,
                              root=ROOT / "runs")
    log(f"K={K}: run dir {run_dir}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L, W = pc["residual_lookback"], cfg["training"]["window_years"]
    dates = p.dates
    w_slot_all, w_pool_all, t_all = [], [], []
    for year in years:
        t_tr0 = dates.searchsorted(pd.Timestamp(year - W, 1, 1))
        t_te0 = dates.searchsorted(pd.Timestamp(year, 1, 1))
        t_te1 = dates.searchsorted(pd.Timestamp(year + 1, 1, 1))
        if dates[t_tr0].year > year - W:
            raise SystemExit(f"{year}: the panel starts {dates[0]:%Y-%m-%d}, too late for a "
                             f"{W}-year training window")
        t0 = time.time()
        log(f"  {year}: train {dates[t_tr0]:%Y-%m-%d}..{dates[t_te0 - 1]:%Y-%m-%d} "
            f"({t_te0 - t_tr0} days), test {t_te1 - t_te0} days")
        model = AttentionArb(n_features=len(p.features), n_factors=K, embedding_dim=mc["embedding_dim"],
                             lambda_ridge=mc["lambda_ridge"], hidden=pc["hidden"], lookback=L,
                             dropout=pc["dropout"], lambda_squash=pc["lambda_squash"])
        train_window(model, p, t_tr0, t_te0, cfg, rng, log)
        model.eval()
        with torch.no_grad():
            _, parts, out = span(model, p, t_te0, t_te1, cfg)
        w = out["w"]
        w_slot_all.append(w.numpy())
        w_pool_all.append(to_pool(w, p.idx[t_te0:t_te1], p.n_pool).numpy())
        t_all.append(np.arange(t_te0, t_te1))
        log(f"      test: net SR {metrics.annualised(parts['net'].numpy())['SR']:.2f} within the year, "
            f"EV {parts['ev'].item():.3f}, {int(out['tradable'].sum(1).float().mean())} tradable/day, "
            f"{time.time() - t0:.0f}s")

    t_idx = np.concatenate(t_all)
    wp = np.concatenate(w_pool_all)
    gross = (wp * p.R_pool[t_idx]).sum(axis=1)
    turnover = np.abs(np.diff(wp, axis=0, prepend=np.zeros((1, wp.shape[1]), dtype=wp.dtype))).sum(axis=1)
    short = np.clip(-wp, 0, None).sum(axis=1)
    daily = pd.DataFrame({"date": dates[t_idx], "gross": gross, "turnover": turnover, "short": short,
                          "n_traded": (wp != 0).sum(axis=1), "mkt_ew": p.mkt_ew[t_idx],
                          "rf": p.rf.numpy()[t_idx]})
    m = metrics.performance(daily, ob["turnover_cost"], ob["short_cost"], ev["cost_grid_bps"])
    m.update({"K": K, "seed": seed, "factor_model": "attention", "policy_input": pc["input"]})
    runs.write_metrics(run_dir, m)
    daily.to_csv(run_dir / "oos_daily.csv", index=False)
    if ev.get("save_weights"):
        ws = np.concatenate(w_slot_all)
        tt, ss = np.nonzero(ws != 0)
        pd.DataFrame({"date": dates[t_idx[tt]], "sec_id": p.sec_ids[p.idx.numpy()[t_idx[tt], ss]],
                      "w": ws[tt, ss].astype(np.float32)}).to_parquet(run_dir / "weights.parquet", index=False)
    log(f"K={K:3d}  {metrics.table_row(m)}   turnover {m['turnover_daily']:.3f}  "
        f"short {m['short_exposure']:.3f}  break-even {m['break_even_turnover_cost_bps']:.1f}bp")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_replication.yaml")
    ap.add_argument("--K", type=int, nargs="*")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--years", type=int, nargs="*", help="out-of-sample years to run (default: all)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--data-dir", help="directory with the three schema tables (default: config data.dir)")
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
        f"{p.dates[0]:%Y-%m-%d}..{p.dates[-1]:%Y-%m-%d}, torch threads {torch.get_num_threads()}")
    log("      K    SR     mu   sigma   SRnet  munet signet   beta")
    for K in Ks:
        run_K(K, cfg, seed, p, years, log)


if __name__ == "__main__":
    main()
