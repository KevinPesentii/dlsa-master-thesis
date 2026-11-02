"""Seed ensemble against config ensemble, on the first training window (2026-11-02).

Outside the thesis rows. One book holding the mean of the five seeds' weights earned far more out
of sample than any seed (runs/_matrix/exp_fixedwin_full/seed_ensemble.py). Which average would the
pre-registered protocol pick? Every (candidate, seed) of the config is a search trial
(search_validation.evaluate: the same panel, initial draw and training path, fitted on 1990-1995)
whose weights on 1996-1997 are kept, and the books are averaged two ways:

- seeds:<c>: one candidate, every seed. Rule "seeds" = seeds:<the search winner> (first candidate).
- configs:s<s>: every candidate at seed s. Rule "configs" = the mean of these books over the seeds.
  configs:latin<r> gives candidate i seed (i + r) mod n instead, so no two members share a draw.
- all: every model.

Each book is booked in pool space with run_attention_us.daily_book, the out-of-sample accounting:
`avg` as averaged, `avg_l1` rescaled to ||w||_1 = 1 on every day, the single runs' exposure. The
config's `book` names the one the decision reads: the rule with the higher validation net Sharpe.

    python scripts/ensemble_validation.py --config configs/us_ensemble_val.yaml [--jobs 25] [--threads 2]
        [--run-dir runs/...] [--smoke]

--run-dir resumes a run: models already in trials.jsonl are not refitted. --smoke: 2 candidates,
2 seeds, 1 epoch.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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

import run_attention_us as base  # noqa: E402  (span, train_window, daily_book)
import run_attention_us_val as val  # noqa: E402  (fresh_model)
import search_validation as sv  # noqa: E402  (init_worker, score, with_overrides)
from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.model.attention_pipeline import to_pool  # noqa: E402


def fit(job: dict) -> tuple[dict, np.ndarray]:
    """One search trial, scored as the search scores it, and its validation weights in pool space."""
    W = sv.WORKER
    c = sv.with_overrides(W["cfg"], job["overrides"])
    p, (t_tr0, t_va0, t_va1) = W["p"], W["split"]
    init_seed = job["seed"] * 10007 + W["first_oos"]
    model = val.fresh_model(c, W["K"], len(p.features), init_seed)
    t0 = time.time()
    base.train_window(model, p, t_tr0, t_va0, c, np.random.default_rng(init_seed), lambda s: None)
    trial = sv.score(model, c)                       # leaves the model in eval mode
    with torch.no_grad():
        _, _, out = base.span(model, p, t_va0, t_va1, c)
    wp = to_pool(out["w"], p.idx[t_va0:t_va1], p.n_pool).numpy()
    return {**job, **trial, "seconds": round(time.time() - t0)}, wp


def ensembles(cands: list[str], seeds: list[int]) -> dict[str, list[tuple[str, int]]]:
    """Book name -> the (candidate, seed) models it averages."""
    books = {f"seeds:{c}": [(c, s) for s in seeds] for c in cands}
    books |= {f"configs:s{s}": [(c, s) for c in cands] for s in seeds}
    if len(cands) == len(seeds):
        books |= {f"configs:latin{r}": [(c, seeds[(i + r) % len(seeds)]) for i, c in enumerate(cands)]
                  for r in range(len(seeds))}
    books["all"] = [(c, s) for c in cands for s in seeds]
    return books


def average(ws: list[np.ndarray], rescale: bool) -> np.ndarray:
    """Mean of the members' weights; `rescale`: divided by its ||w||_1 on every day it holds anything."""
    w = np.mean(np.stack(ws), axis=0, dtype=np.float64)
    if rescale:
        l1 = np.abs(w).sum(axis=1, keepdims=True)
        w = np.divide(w, l1, out=np.zeros_like(w), where=l1 > 0)
    return w


def book(p: slots.SlotPanel, wp: np.ndarray, t_idx: np.ndarray, cfg: dict) -> dict:
    d = base.daily_book(p, wp, t_idx, cfg)
    ob = cfg["objective"]
    net = (d["gross"] - ob["turnover_cost"] * d["turnover"] - ob["short_cost"] * d["short"]).to_numpy(float)
    return {"net_SR": float(metrics.annualised(net)["SR"]),
            "gross_SR": float(metrics.annualised(d["gross"].to_numpy(float))["SR"]),
            "turnover": float(d["turnover"].mean()), "short": float(d["short"].mean()),
            "l1": float(np.abs(wp).sum(axis=1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_ensemble_val.yaml")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--run-dir", type=Path, help="write to (or resume) this directory instead of runs/<stamp>_...")
    ap.add_argument("--smoke", action="store_true", help="2 candidates, 2 seeds, 1 epoch: does the config run")
    args = ap.parse_args()
    E = yaml.safe_load(Path(args.config).read_text())
    cfg = sv.with_overrides(yaml.safe_load((ROOT / E["base_config"]).read_text()), E.get("fixed") or {})
    cands, seeds = dict(E["candidates"]), list(E["seeds"])
    for o in cands.values():
        sv.with_overrides(cfg, o)                    # a key the base config lacks is an error
    if args.smoke:
        cands, seeds = dict(list(cands.items())[:2]), seeds[:2]
        cfg["training"]["epochs"] = 1
    K, first_oos, V = E["K"], E["first_oos_year"], E["validation_years"]
    data_dir = ROOT / cfg["data"]["dir"]
    log = lambda s: print(s, flush=True)  # noqa: E731
    resume = args.run_dir is not None and (args.run_dir / "manifest.json").exists()
    run_dir = args.run_dir if resume else runs.create_run(
        f"ensval_K{K}_val{first_oos - V}-{first_oos - 1}", {"ensemble": E, "base": cfg, "smoke": args.smoke},
        seeds[0], root=ROOT / "runs", run_dir=args.run_dir)
    (run_dir / "weights").mkdir(exist_ok=True)
    path = run_dir / "trials.jsonl"
    trials = {(r["name"], r["seed"]): r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    jobs = [{"name": n, "overrides": o, "seed": s} for n, o in cands.items() for s in seeds if (n, s) not in trials]
    log(f"ensemble validation {run_dir}: K={K}, {len(cands)} candidates x seeds {seeds}, {len(jobs)} to fit")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.threads)
    if jobs:
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs)), initializer=sv.init_worker, initargs=(
                cfg, str(data_dir), first_oos, V, K, [], args.threads)) as pool:
            for i, (r, wp) in enumerate(pool.imap_unordered(fit, jobs), 1):
                np.save(run_dir / "weights" / f"{r['name']}_s{r['seed']}.npy", wp)
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                trials[(r["name"], r["seed"])] = r
                log(f"  [{i}/{len(jobs)}] {r['name']} seed {r['seed']}: val net SR {r['net_SR']:+.2f}  {r['seconds']}s")

    w0 = base.window_start(cfg, first_oos)              # the panel of the search's workers
    p = slots.load_slots(data_dir, f"{w0:%Y-%m-%d}", f"{first_oos - 1}-12-31")
    t_idx = np.arange(p.dates.searchsorted(pd.Timestamp(first_oos - V, 1, 1)), len(p.dates))
    W = {(n, s): np.load(run_dir / "weights" / f"{n}_s{s}.npy") for n in cands for s in seeds}
    rows = [{"book": f"single:{n}:s{s}", "kind": "single", "models": 1, "search_net_SR": trials[(n, s)]["net_SR"],
             **book(p, w.astype(np.float64), t_idx, cfg)} for (n, s), w in W.items()]
    for name, members in ensembles(list(cands), seeds).items():
        for kind in ("avg", "avg_l1"):
            rows.append({"book": name, "kind": kind, "models": len(members),
                         **book(p, average([W[m] for m in members], kind == "avg_l1"), t_idx, cfg)})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "books.csv", index=False)

    kind, win = E.get("book", "avg_l1"), next(iter(cands))
    sr = df[df.kind == kind].set_index("book")["net_SR"]
    single = df[df.kind == "single"].set_index("book")["net_SR"]
    by_seed = sr[[f"configs:s{s}" for s in seeds]]
    seeds_rule, configs_rule = float(sr[f"seeds:{win}"]), float(by_seed.mean())
    summary = {"book": kind, "winner": win,
               "single_winner_mean": float(single[[f"single:{win}:s{s}" for s in seeds]].mean()),
               "rule_seeds": seeds_rule, "rule_configs": configs_rule, "rule_configs_by_seed": by_seed.to_dict(),
               "seeds_books": {c: float(sr[f"seeds:{c}"]) for c in cands},
               "latin": {b: float(v) for b, v in sr.items() if b.startswith("configs:latin")},
               "all": float(sr["all"]), "choice": "seeds" if seeds_rule >= configs_rule else "configs"}
    runs.write_metrics(run_dir, summary)
    pd.set_option("display.width", 200)
    log(df.round(3).to_string(index=False))
    log(f"validation net SR ({kind}): seeds of {win} {seeds_rule:+.2f}, configs (mean of {len(by_seed)} seeds) "
        f"{configs_rule:+.2f}, single {win} {summary['single_winner_mean']:+.2f} -> {summary['choice']}")


if __name__ == "__main__":
    main()
