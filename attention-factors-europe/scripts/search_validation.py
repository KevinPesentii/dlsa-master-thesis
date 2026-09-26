"""Hyperparameter search on the first training window, the paper's protocol.

Appendix B: "We use the last two years of the first training window to select tuning
parameters"; Table 4 lists what that gave. For a 1998 start the first window is 1990-1997:
every candidate is fitted on 1990-1995 and scored by its net Sharpe on 1996-1997. The panel
is loaded only up to 1997-12-31, so no out-of-sample year can move the choice.
configs/us_search.yaml fixes the space, the budget and the rule; commit it first.

    python scripts/search_validation.py [--config configs/us_search.yaml] [--jobs 3]
        [--threads 2] [--data-dir ...] [--stage all|select|robustness] [--resume runs/<dir>]

1. search: Sobol points over the knobs the paper leaves open, plus the base config, one seed.
2. finalists: the best `top` points and the base config on more seeds; the winner has the
   best MEAN validation net Sharpe. Writes selected.yaml (base config + the winner's values)
   for the out-of-sample run:
       scripts/run_years_parallel.sh --seeds "0 1 2 3 4" -- scripts/run_attention_us.py \
           --config runs/<this run>/selected.yaml
3. robustness: the winner with one Table 4 value moved at a time, on the same seeds.
   Written to robustness.csv, never adopted.

A trial with seed s starts from the draw (s, first out-of-sample year), as in
run_attention_us_val.py, so candidates are compared on common random numbers. The scores
along the learning curve come from the same training run (train_window's on_epoch hook).
trials.jsonl keeps every trial; --resume <run dir> skips the ones already there.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_attention_us as base  # noqa: E402  (span, train_window)
import run_attention_us_val as val  # noqa: E402  (fresh_model)
from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402

WORKER: dict = {}  # per process: the panel, the base config and the split


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """cfg with "section.key" values replaced. A key the base config does not have is an
    error, so a typo in the search space cannot silently search nothing."""
    c = copy.deepcopy(cfg)
    for key, v in overrides.items():
        section, name = key.split(".")
        if name not in c.get(section, {}):
            raise KeyError(f"{key} is not in the base config")
        c[section][name] = v
    return c


def draw(spec: dict, u: float):
    """u in [0, 1) -> a value: one of `choice`, or log-uniform on `log` after a mass `off` on null."""
    if "choice" in spec:
        return spec["choice"][min(int(u * len(spec["choice"])), len(spec["choice"]) - 1)]
    off = spec.get("off", 0.0)
    if u < off:
        return None
    lo, hi = np.log10(spec["log"])
    return float(f"{10 ** (lo + (u - off) / (1 - off) * (hi - lo)):.3g}")


def sobol_points(space: dict, n: int, seed: int) -> list[dict]:
    if n < 1 or n & (n - 1):
        raise SystemExit(f"n_points {n}: use a power of 2, Sobol points are balanced only then")
    for key, spec in space.items():
        if not set(spec) <= {"log", "off", "choice"}:
            raise SystemExit(f"{key}: unknown keys {set(spec) - {'log', 'off', 'choice'}} "
                             "(in YAML, quote \"off\": bare off is the boolean false)")
    u = qmc.Sobol(d=len(space), scramble=True, seed=seed).random_base2(int(math.log2(n)))
    return [{k: draw(spec, float(x)) for x, (k, spec) in zip(row, space.items())} for row in u]


def candidate(name: str, overrides: dict) -> dict:
    cid = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:10]
    return {"name": name, "cid": cid, "overrides": overrides}


def init_worker(cfg: dict, data_dir: str, first_oos: int, val_years: int, K: int,
                eval_epochs: list[int], threads: int) -> None:
    torch.set_num_threads(threads)
    W = cfg["training"]["window_years"]
    p = slots.load_slots(Path(data_dir), f"{first_oos - W}-01-01", f"{first_oos - 1}-12-31")
    t = [p.dates.searchsorted(pd.Timestamp(y, 1, 1)) for y in (first_oos - W, first_oos - val_years)]
    if p.dates[0].year > first_oos - W:
        raise SystemExit(f"the panel starts {p.dates[0]:%Y-%m-%d}, too late for a {W}-year window")
    WORKER.update(p=p, cfg=cfg, K=K, first_oos=first_oos, eval_epochs=eval_epochs,
                  split=(t[0], t[1], len(p.dates)))


def score(model, c: dict) -> dict:
    """Validation metrics of the model as it stands, in eval mode (no dropout, no RNG draws)."""
    p, (_, t_va0, t_va1) = WORKER["p"], WORKER["split"]
    model.eval()
    with torch.no_grad():
        _, parts, _ = base.span(model, p, t_va0, t_va1, c)
    ob = c["objective"]
    net, to, sh = (parts[k].numpy().astype(float) for k in ("net", "turnover", "short"))
    g = metrics.annualised(net + ob["turnover_cost"] * to + ob["short_cost"] * sh)
    return {"net_SR": float(metrics.annualised(net)["SR"]), "gross_SR": float(g["SR"]),
            "sigma_pct": float(g["sigma_pct"]), "turnover": float(to.mean()), "ev": float(parts["ev"])}


def evaluate(job: dict) -> dict:
    c = with_overrides(WORKER["cfg"], job["overrides"])
    t_tr0, t_va0, _ = WORKER["split"]
    init_seed = job["seed"] * 10007 + WORKER["first_oos"]
    model = val.fresh_model(c, WORKER["K"], len(WORKER["p"].features), init_seed)
    E = c["training"]["epochs"]
    at = {e for e in WORKER["eval_epochs"] if e < E} | {E}
    curve = {}

    def on_epoch(epoch, m):
        if epoch in at:
            curve[epoch] = score(m, c)

    t0 = time.time()
    base.train_window(model, WORKER["p"], t_tr0, t_va0, c, np.random.default_rng(init_seed),
                      lambda s: None, on_epoch)
    return {**job, **curve[E], "curve": curve, "seconds": round(time.time() - t0)}


def run_jobs(pool, jobs: list[dict], trials: dict, path: Path, log, label: str) -> list[dict]:
    todo = [j for j in jobs if (j["cid"], j["seed"]) not in trials]
    log(f"{label}: {len(jobs)} trials, {len(jobs) - len(todo)} already done")
    for i, r in enumerate(pool.imap_unordered(evaluate, todo), 1):
        trials[(r["cid"], r["seed"])] = r
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        log(f"  [{label} {i}/{len(todo)}] {r['name']} seed {r['seed']}: val net SR {r['net_SR']:+.2f} "
            f"(gross {r['gross_SR']:+.2f}, turnover {r['turnover']:.2f})  {r['seconds']}s")
    return [trials[(j["cid"], j["seed"])] for j in jobs]


def finite(x: float) -> float:
    return x if np.isfinite(x) else -np.inf


def sem(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def table(cands: list[dict], res: list[dict], ref: str) -> list[dict]:
    """Mean over seeds per candidate, and the paired (same seed) difference to candidate `ref`."""
    ref_sr = {r["seed"]: r["net_SR"] for r in res if r["cid"] == ref}
    rows = []
    for c in cands:
        rs = [r for r in res if r["cid"] == c["cid"]]
        sr = np.array([r["net_SR"] for r in rs])
        d = np.array([r["net_SR"] - ref_sr[r["seed"]] for r in rs])
        rows.append({**c, "val_net_SR": float(sr.mean()), "se": sem(sr), "diff": float(d.mean()),
                     "diff_se": sem(d), "gross_SR": float(np.mean([r["gross_SR"] for r in rs])),
                     "turnover": float(np.mean([r["turnover"] for r in rs])), "n_seeds": len(rs)})
    return sorted(rows, key=lambda r: -finite(r["val_net_SR"]))


def show(rows: list[dict], ref_name: str, log) -> None:
    log(f"  {'name':34s} {'val SR':>7s} {'se':>5s}  {'vs ' + ref_name + ' (se)':>15s}  {'gross':>6s} {'turn':>5s}")
    for r in rows:
        flag = "  !" if abs(r["diff"]) > 2 * r["diff_se"] else ""
        log(f"  {r['name']:34s} {r['val_net_SR']:+7.2f} {r['se']:5.2f}  {r['diff']:+7.2f} ({r['diff_se']:4.2f}){flag:3s}"
            f"  {r['gross_SR']:+6.2f} {r['turnover']:5.2f}  {json.dumps(r['overrides'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "us_search.yaml")
    ap.add_argument("--data-dir", help="directory with the three schema tables (default: base config data.dir)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stage", choices=["all", "select", "robustness"], default="all")
    ap.add_argument("--resume", help="run directory of an earlier search: its trials are reused")
    args = ap.parse_args()
    S = yaml.safe_load(Path(args.config).read_text())
    base_path = Path(S["base_config"])
    cfg = yaml.safe_load((base_path if base_path.is_absolute() else ROOT / base_path).read_text())
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    data_dir = Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    K, first_oos, sr, fin, rob = S["K"], S["first_oos_year"], S["search"], S["finalists"], S["robustness"]
    for key in [*sr["space"], *rob["knobs"]]:
        with_overrides(cfg, {key: None})

    log = lambda s: print(s, flush=True)  # noqa: E731
    run_dir = Path(args.resume) if args.resume else runs.create_run(
        f"search_K{K}", {"search": S, "base": cfg}, sr["seed"], root=ROOT / "runs")
    path, trials = run_dir / "trials.jsonl", {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            trials[(r["cid"], r["seed"])] = r
    W = cfg["training"]["window_years"]
    log(f"search run dir {run_dir}")
    log(f"fit {first_oos - W}-{first_oos - S['validation_years'] - 1}, validate "
        f"{first_oos - S['validation_years']}-{first_oos - 1}, K={K}, {args.jobs} jobs x {args.threads} threads")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.threads)
    summary_path = run_dir / "metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    with mp.get_context("spawn").Pool(args.jobs, initializer=init_worker, initargs=(
            cfg, str(data_dir), first_oos, S["validation_years"], K, S["eval_epochs"], args.threads)) as pool:
        run = lambda jobs, label: run_jobs(pool, jobs, trials, path, log, label)  # noqa: E731
        if args.stage in ("all", "select"):
            base_c = candidate("base", {})
            pts = [candidate(f"sobol-{i:03d}", o)
                   for i, o in enumerate(sobol_points(sr["space"], sr["n_points"], sr["sobol_seed"]))]
            res = run([{**c, "seed": sr["seed"]} for c in [base_c, *pts]], "search")
            best = {r["cid"] for r in sorted(res, key=lambda r: -finite(r["net_SR"]))[:fin["top"]]}
            finalists = [c for c in [base_c, *pts] if c["cid"] in best or c is base_c]
            rows = table(finalists, run([{**c, "seed": s} for c in finalists for s in fin["seeds"]],
                                        "finalists"), base_c["cid"])
            log(f"finalists, mean over seeds {fin['seeds']} of the validation net Sharpe:")
            show(rows, "base", log)
            win = rows[0]
            selected = with_overrides(cfg, win["overrides"])
            selected["selection"] = {"search_run": run_dir.name, "candidate": win["name"],
                                     "validation_net_SR_mean": win["val_net_SR"], "seeds": fin["seeds"]}
            (run_dir / "selected.yaml").write_text(
                f"# Selected on the first training window by scripts/search_validation.py, run "
                f"{run_dir.name}:\n# the base config with the winner's values.\n"
                + yaml.safe_dump(selected, sort_keys=False))
            summary.update(winner=win, finalists=rows, n_search_points=len(pts))
            log(f"winner {win['name']} {json.dumps(win['overrides'])} -> {run_dir / 'selected.yaml'}")
        if args.stage in ("all", "robustness"):
            win = candidate("winner", summary["winner"]["overrides"])
            alts = [candidate(f"{k}={v}", {**win["overrides"], k: v})
                    for k, vals in rob["knobs"].items() for v in vals]
            rows = table([win, *alts], run([{**c, "seed": s} for c in [win, *alts] for s in rob["seeds"]],
                                           "robustness"), win["cid"])
            log("robustness around the paper's Table 4 values (reported, not adopted; ! = beyond 2 se):")
            show(rows, "winner", log)
            pd.DataFrame(rows).to_csv(run_dir / "robustness.csv", index=False)
            summary["robustness"] = rows
    runs.write_metrics(run_dir, {**summary, "n_trials": len(trials)})


if __name__ == "__main__":
    main()
