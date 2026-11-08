"""Before a search runs on GPUs (search_validation.py --device cuda): is a GPU trial the CPU trial, how long does
it take and how much GPU memory does it need. Reads a finished search (its manifest: base config, K, window).

    python scripts/gpu_probe.py --search runs/_matrix/search_t4/searches/us_t4_K30 [--stage all|step|time|replay]
                                [--n 16] [--jobs 4] [--data-dir ...]

1. step: one training step and one validation span from the same initial weights on the CPU and on GPU 0, with
   dropout 0, so no random number is drawn: loss, gradients and the validation net returns must agree to float32
   rounding (relative differences around 1e-5; above 1e-3 something on the GPU computes another thing).
   Checks the Table 4 base and a deep stack (2 layers, hidden 64), each on the first training block.
2. time: one epoch of four settings on GPU 0, seconds and peak memory: the heaviest of the search (2 layers,
   hidden 64, d_x 64, 250-day batches) with the deeper pieces kept and recomputed, the base, and the base on
   63-day batches. Peak memory per process decides how many workers share a GPU (docs/cloud_runs.md section 19).
3. replay: `n` trials of the search's first stage (seed 0), evenly spaced in its ranking, plus its winner, run
   again on the GPUs with `jobs` workers, against the net Sharpe the CPU recorded. Not the same numbers: the
   dropout masks come from the GPU's generator and the sums run in another order, and a CPU trial already moves
   by about 0.25 between two machines (2026-11-08). What must hold is the ranking (Spearman rho near 1).
Writes runs/<stamp>_gpu_probe_<search>/ (manifest, metrics.json, replay.csv).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_validation as sv  # noqa: E402
from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.policy import longconv  # noqa: E402

DEEP = {"policy.layers": 2, "policy.hidden": 64}
HEAVIEST = {**DEEP, "model.embedding_dim": 64, "training.batch_days": 250}
CHECKS = {"base": {}, "deep": DEEP}
TIMED = {"heaviest": HEAVIEST, "heaviest_recompute": HEAVIEST, "base": {}, "base_b63": {"training.batch_days": 63}}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def step_check(cfg: dict, K: int, p: slots.SlotPanel, pg: slots.SlotPanel, split: tuple, log) -> dict:
    t_tr0, t_va0, t_va1 = split
    out = {}
    for name, ov in CHECKS.items():
        c = sv.with_overrides(cfg, {**ov, "policy.dropout": 0.0})
        m_cpu = sv.val.fresh_model(c, K, len(p.features), 1998)
        m_gpu = copy.deepcopy(m_cpu).to(pg.X.device)
        b0 = t_tr0 + m_cpu.lookback
        b1 = b0 + c["training"]["batch_days"]
        res = []
        for m, panel in ((m_cpu, p), (m_gpu, pg)):
            m.train()
            loss, parts, _ = sv.base.span(m, panel, b0, b1, c)
            loss.backward()
            m.eval()
            with torch.no_grad():
                _, val, _ = sv.base.span(m, panel, t_va0, t_va1, c)
            res.append((loss, parts["net"], val["net"], dict(m.named_parameters())))
        (lc, nc, vc, gc), (lg, ng, vg, gg) = res
        grads = {k: rel(gc[k].grad, gg[k].grad) for k in gc if gc[k].grad is not None}
        out[name] = {"loss": rel(lc, lg), "train_net": rel(nc, ng), "val_net": rel(vc, vg),
                     "grad_max": max(grads.values()), "grad_worst": max(grads, key=grads.get)}
        r = out[name]
        flag = "ok" if max(r["loss"], r["val_net"], r["grad_max"]) < 1e-3 else "CHECK: above 1e-3"
        log(f"  step {name:5s}: relative difference loss {r['loss']:.1e}, train net {r['train_net']:.1e}, "
            f"val net {r['val_net']:.1e}, gradients up to {r['grad_max']:.1e} ({r['grad_worst']})  {flag}")
    return out


def time_epochs(cfg: dict, K: int, pg: slots.SlotPanel, split: tuple, log) -> dict:
    dev = pg.X.device
    sv.WORKER.update(p=pg, split=split, device=dev)
    out = {}
    for name, ov in [("warm-up", {}), *TIMED.items()]:
        c = sv.with_overrides(cfg, ov)
        c["training"]["epochs"] = 1
        keep = longconv.GPU_KEEP
        longconv.GPU_KEEP = 0 if name.endswith("_recompute") else keep
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            model = sv.val.fresh_model(c, K, len(pg.features), 1998).to(dev)
            sv.base.train_window(model, pg, split[0], split[1], c, np.random.default_rng(0), lambda s: None)
            torch.cuda.synchronize(dev)
            train_s = time.time() - t0
            t0 = time.time()
            sc = sv.score(model, c)
            score_s = time.time() - t0
        finally:
            longconv.GPU_KEEP = keep
        if name == "warm-up":
            continue
        out[name] = {"seconds_per_epoch": round(train_s, 1), "seconds_per_score": round(score_s, 1),
                     "peak_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
                     "peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2),
                     "net_SR_after_1_epoch": round(sc["net_SR"], 2)}
        r = out[name]
        log(f"  time {name:18s}: {r['seconds_per_epoch']:6.1f} s an epoch, {r['seconds_per_score']:5.1f} s a "
            f"score, peak {r['peak_allocated_gb']:.2f} GB allocated / {r['peak_reserved_gb']:.2f} GB reserved")
    return out


def replay(search: Path, S: dict, cfg: dict, data_dir: Path, n: int, jobs: int, threads: int, log,
           device: str = "cuda") -> pd.DataFrame:
    rs = [r for r in runs.read_jsonl(search / "trials.jsonl") if r["seed"] == S["search"]["seed"]]
    rs = sorted({r["cid"]: r for r in rs}.values(), key=lambda r: -sv.finite(r["net_SR"]))
    pick = {int(round(i)) for i in np.linspace(0, len(rs) - 1, n)}
    winner = json.loads((search / "metrics.json").read_text())["winner"]["cid"]
    todo = [r for i, r in enumerate(rs) if i in pick or r["cid"] == winner]
    todo_jobs = [{k: r[k] for k in ("name", "cid", "overrides", "seed")} for r in todo]
    cost = lambda j: sv.rough_cost(sv.with_overrides(cfg, j["overrides"]))  # noqa: E731
    todo_jobs.sort(key=lambda j: -cost(j))
    log(f"  replay: {len(todo_jobs)} trials of {len(rs)} on {jobs} GPU workers")
    got = {}
    with mp.get_context("spawn").Pool(jobs, initializer=sv.init_worker, initargs=(
            cfg, str(data_dir), S["first_oos_year"], S["validation_years"], S["K"], S.get("eval_epochs", []),
            threads, "attention", device, jobs)) as pool:
        for r in pool.imap_unordered(sv.evaluate, todo_jobs):
            got[r["cid"]] = r
            log(f"    {r['name']:12s} GPU net SR {r['net_SR']:+.2f}  {r['seconds']} s")
    rows = [{"name": r["name"], "layers": sv.with_overrides(cfg, r["overrides"])["policy"]["layers"],
             "dropout": sv.with_overrides(cfg, r["overrides"])["policy"]["dropout"],
             "cpu_net_SR": r["net_SR"], "gpu_net_SR": got[r["cid"]]["net_SR"],
             "diff": got[r["cid"]]["net_SR"] - r["net_SR"], "cpu_seconds": r["seconds"],
             "gpu_seconds": got[r["cid"]]["seconds"], "oom_retry": got[r["cid"]].get("oom_retry", False)}
            for r in todo]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=Path, required=True, help="a finished search's run directory")
    ap.add_argument("--stage", choices=["all", "step", "time", "replay"], default="all")
    ap.add_argument("--n", type=int, default=16, help="replayed trials, evenly spaced in the ranking")
    ap.add_argument("--jobs", type=int, default=4, help="GPU workers of the replay")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data-dir", help="default: the base config's data.dir")
    args = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # as search_validation.py
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible to torch")
    search = args.search if args.search.is_absolute() else ROOT / args.search
    man = json.loads((search / "manifest.json").read_text())["config"]
    S, cfg = man["search"], man["base"]
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    data_dir = Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    K, first_oos, V = S["K"], S["first_oos_year"], S["validation_years"]
    log = lambda s: print(s, flush=True)  # noqa: E731
    run_dir = runs.create_run(f"gpu_probe_{search.name}", {**vars(args), "search": str(search),
                                                            "data_dir": str(data_dir)}, 0)
    log(f"probe of {search.name} (K={K}) on {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)} -> {run_dir}")
    metrics: dict = {"gpu": torch.cuda.get_device_name(0), "n_gpus": torch.cuda.device_count()}
    if args.stage in ("all", "step", "time"):
        dev = sv.gpu_device()
        w0 = sv.base.window_start(cfg, first_oos)
        p = slots.load_config(cfg, data_dir, f"{w0:%Y-%m-%d}", f"{first_oos - 1}-12-31")
        split = (int(p.dates.searchsorted(w0)), int(p.dates.searchsorted(pd.Timestamp(first_oos - V, 1, 1))),
                 len(p.dates))
        pg = slots.to_device(p, dev)
        if args.stage in ("all", "step"):
            metrics["step"] = step_check(cfg, K, p, pg, split, log)
        if args.stage in ("all", "time"):
            metrics["time"] = time_epochs(cfg, K, pg, split, log)
        del p, pg
        sv.WORKER.clear()               # holds the GPU panel after `time`; the replay's workers need the memory
        torch.cuda.empty_cache()
    if args.stage in ("all", "replay"):
        df = replay(search, S, cfg, data_dir, args.n, args.jobs, args.threads, log)
        df.to_csv(run_dir / "replay.csv", index=False)
        rho = float(spearmanr(df["cpu_net_SR"], df["gpu_net_SR"]).statistic)
        metrics["replay"] = {"n": len(df), "spearman": rho, "mean_abs_diff": float(df["diff"].abs().mean()),
                             "mean_diff": float(df["diff"].mean()), "cpu_seconds": int(df["cpu_seconds"].sum()),
                             "gpu_seconds": int(df["gpu_seconds"].sum()), "oom_retries": int(df["oom_retry"].sum())}
        with pd.option_context("display.width", 160):
            log(df.round(2).to_string(index=False))
        m = metrics["replay"]
        log(f"  replay: Spearman rho {rho:.3f}, mean |GPU - CPU| {m['mean_abs_diff']:.2f} (mean {m['mean_diff']:+.2f}); "
            f"trial seconds CPU (AWS m7i, 46 workers) {m['cpu_seconds']}, GPU ({args.jobs} workers) {m['gpu_seconds']}")
    runs.write_metrics(run_dir, metrics)
    log(f"-> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
