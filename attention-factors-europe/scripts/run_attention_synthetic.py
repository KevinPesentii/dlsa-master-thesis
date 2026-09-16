"""One-step attention arbitrage on synthetic data, end to end.

Exists to exercise the wiring while the real panel is being built: shapes, gradient flow
to W_K and Q, the rolling refit, the cost accounting and the evaluation all run here.
Swapping the generator for the parquet loader is then the only change needed.

The generator plants a mean-reverting component in the idiosyncratic part, so a policy
that works should beat zero; it is not calibrated to anything and the numbers here mean
nothing beyond "the machine turns".

    python3 scripts/run_attention_synthetic.py --days 1500 --pool 120 --slots 80
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

from afe.model.attention_pipeline import AttentionArb, objective, slot_batch

log = logging.getLogger("attention")


def synthetic_panel(n_days, n_pool, n_slots, n_chars, seed=0, rho=-0.15):
    """Slot-layout arrays. Universe membership rotates, so slots really do move.

    Characteristics are already lagged: row t holds what was known at the close of t-1.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0003, 0.010, (n_days, 1))
    beta = rng.uniform(0.5, 1.5, (1, n_pool))

    idio = np.zeros((n_days, n_pool))
    shock = rng.normal(0, 0.015, (n_days, n_pool))
    for t in range(1, n_days):                      # AR(1) with negative rho: reversal
        idio[t] = rho * idio[t - 1] + shock[t]
    R_pool = mkt * beta + idio

    size = rng.normal(0, 1, n_pool)                 # a stable ranking to rotate membership
    idx = np.zeros((n_days, n_slots), dtype=np.int64)
    for t in range(n_days):
        drift = size + rng.normal(0, 0.3, n_pool) * (t / n_days)
        idx[t] = np.argsort(-drift)[:n_slots]

    R_slot = np.take_along_axis(R_pool, idx, axis=1).astype(np.float32)
    in_universe = np.ones((n_days, n_slots), dtype=bool)

    # Characteristics: lagged returns over several horizons plus noise, rank-normalised.
    X = np.zeros((n_days, n_slots, n_chars), dtype=np.float32)
    for j, h in enumerate([1, 5, 21, 63][:n_chars]):
        cum = np.zeros_like(R_pool)
        cum[h:] = np.cumsum(R_pool, 0)[h:] - np.cumsum(R_pool, 0)[:-h]
        X[:, :, j] = np.take_along_axis(cum, idx, axis=1)
    for j in range(4, n_chars):
        X[:, :, j] = rng.normal(0, 1, (n_days, n_slots))
    ranks = X.argsort(axis=1).argsort(axis=1) / max(n_slots - 1, 1) - 0.5
    X = ranks.astype(np.float32)
    X = np.concatenate([X[:1], X[:-1]], axis=0)     # the lag, made explicit

    rf = np.zeros(n_days, dtype=np.float32)
    return (torch.from_numpy(X), torch.from_numpy(R_slot), torch.from_numpy(in_universe),
            torch.from_numpy(idx), torch.from_numpy(rf), n_pool)


def run_block(model, X, R, univ, idx, rf, n_pool, t0, t1, cfg, train: bool):
    """One contiguous block of dates. The block carries `lookback` extra dates in front,
    which feed the windows but are not traded."""
    L = model.lookback
    s, e = t0 - L, t1
    out = model.forward_span(X[s:e], R[s:e], univ[s:e], idx[s:e], n_pool,
                             cumulative=cfg["cumulative"], scale=cfg["scale"])
    b = slot_batch(R[t0:e], idx[t0:e], rf[t0:e], out["tradable"], n_pool)
    loss, parts = objective(out, b, cfg["tc"], cfg["sc"], cfg["lambda_var"])
    return loss, parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=1500)
    p.add_argument("--pool", type=int, default=120)
    p.add_argument("--slots", type=int, default=80)
    p.add_argument("--chars", type=int, default=6)
    p.add_argument("--factors", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--block", type=int, default=125)
    p.add_argument("--train-days", type=int, default=750)
    p.add_argument("--test-days", type=int, default=250)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)

    X, R, univ, idx, rf, n_pool = synthetic_panel(args.days, args.pool, args.slots, args.chars,
                                                  seed=args.seed)
    cfg = {"tc": 0.0005, "sc": 0.0001, "lambda_var": args.lambda_var, "cumulative": True,
           "scale": 100.0}

    L = 30
    oos = []
    start = L + args.train_days
    for w0 in range(start, args.days - args.test_days + 1, args.test_days):
        model = AttentionArb(n_features=args.chars, n_factors=args.factors, embedding_dim=16,
                             hidden=32, lookback=L)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        tr0, tr1 = w0 - args.train_days, w0
        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            tot = 0.0
            for b0 in range(tr0, tr1, args.block):
                b1 = min(b0 + args.block, tr1)
                if b1 - b0 < 20:
                    continue
                opt.zero_grad()
                loss, parts = run_block(model, X, R, univ, idx, rf, n_pool, b0, b1, cfg, True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            if epoch % 4 == 0:
                log.info("  epoch %2d  loss %+.4f  (sharpe part %+.3f, EV %.3f)",
                         epoch, tot, parts["sharpe_loss"].item(), parts["ev"].item())

        model.eval()
        with torch.no_grad():
            te1 = min(w0 + args.test_days, args.days)
            _, parts = run_block(model, X, R, univ, idx, rf, n_pool, w0, te1, cfg, False)
        net = parts["net"]
        sr = float(net.mean() / (net.std() + 1e-12) * np.sqrt(252))
        log.info("window ending %d: OOS Sharpe %+.2f, turnover %.3f, short %.2f  [%.0fs]",
                 w0, sr, float(parts["turnover"].mean()), float(parts["short"].mean()),
                 time.time() - t0)
        oos.append(net)

    allnet = torch.cat(oos)
    print(f"\nOOS aggregato su {len(allnet)} giorni: "
          f"Sharpe {float(allnet.mean()/(allnet.std()+1e-12)*np.sqrt(252)):+.2f}")


if __name__ == "__main__":
    main()
