"""How long will the real replication take on THIS machine? Measures it, in about a minute.

Times one training block (forward + backward + optimiser step) of the one-step pipeline
at the paper's dimensions, then scales by the schedule of Table 4 and Section 4.2:
500 names, X of dimension 79, embedding 32, LongConv hidden 32, lookback 30,
8-year window in blocks of 125 days, 30 epochs, 24 annual refits (1998-2021).

    python3 scripts/bench_timing.py                 # CPU, all cores
    python3 scripts/bench_timing.py --device mps    # Apple GPU, if every op is supported

Only compute is measured. Loading the panel and the final evaluation are extra, and small
next to training; building the panel from WRDS is a one-off and not counted.
"""

from __future__ import annotations

import argparse
import time

import torch

from afe.model.attention_pipeline import AttentionArb, objective, slot_batch

N_NAMES, N_POOL, M, LOOKBACK, BLOCK = 500, 800, 79, 30, 125
BLOCKS_PER_WINDOW = (8 * 252) // BLOCK        # 8-year training window
EPOCHS, REFITS = 30, 24                       # Table 4; annual refit 1998-2021
TABLE2_KS = (1, 3, 5, 8, 10, 15, 30, 100)


def time_block(K: int, device: str, reps: int = 3) -> float:
    torch.manual_seed(0)
    span = BLOCK + LOOKBACK
    X = torch.randn(span, N_NAMES, M, device=device)
    R = torch.randn(span, N_NAMES, device=device) * 0.02
    univ = torch.ones(span, N_NAMES, dtype=torch.bool, device=device)
    idx = torch.randperm(N_POOL)[:N_NAMES].repeat(span, 1).to(device)
    rf = torch.zeros(span, device=device)
    model = AttentionArb(n_features=M, n_factors=K, embedding_dim=32, hidden=32,
                         lookback=LOOKBACK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)

    def step():
        opt.zero_grad()
        out = model.forward_span(X, R, univ, idx, N_POOL, scale=100.0)
        b = slot_batch(R[LOOKBACK:], idx[LOOKBACK:], rf[LOOKBACK:], out["tradable"], N_POOL)
        loss, _ = objective(out, b, tc=5e-4, sc=1e-4, lambda_var=100.0)
        loss.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()

    step()                                     # warm-up, not timed
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    return (time.perf_counter() - t0) / reps


def hours(block_seconds: float) -> float:
    return block_seconds * BLOCKS_PER_WINDOW * EPOCHS * REFITS / 3600


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--quick", action="store_true", help="only K=30, the headline")
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available on this machine")

    print(f"device={args.device}  torch threads={torch.get_num_threads()}  torch {torch.__version__}")
    ks = (30,) if args.quick else TABLE2_KS
    per_k = {}
    for K in ks:
        try:
            s = time_block(K, args.device)
        except (NotImplementedError, RuntimeError) as e:
            raise SystemExit(f"K={K} failed on {args.device}: {e}\nUse --device cpu.")
        per_k[K] = hours(s)
        print(f"  K={K:3d}  block {s:.3f}s  ->  full 1998-2021 run {per_k[K]:.2f} h")

    h30 = per_k[30]
    print("\nScenarios (compute only):")
    print(f"  headline K=30, 1 seed            {h30:6.1f} h")
    print(f"  headline K=30, 5 seeds           {5 * h30:6.1f} h")
    if len(per_k) == len(TABLE2_KS):
        t2 = sum(per_k.values())
        print(f"  all of Table 2, 1 seed           {t2:6.1f} h")
        print(f"  all of Table 2, 3 seeds          {3 * t2:6.1f} h")


if __name__ == "__main__":
    main()
