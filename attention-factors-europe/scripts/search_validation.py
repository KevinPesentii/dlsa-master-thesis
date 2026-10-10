"""Hyperparameter search on the first training window, the paper's protocol.

Appendix B: "We use the last two years of the first training window to select tuning
parameters"; Table 4 lists what that gave. For a 1998 start the first window is 1990-1997:
every candidate is fitted on 1990-1995 and scored by its net Sharpe on 1996-1997. The panel
is loaded only up to 1997-12-31, so no out-of-sample year can move the choice.
configs/us_search.yaml fixes the space, the budget and the rule; commit it first.

    python scripts/search_validation.py [--config configs/us_search.yaml] [--jobs 3]
        [--threads 2] [--data-dir ...] [--stage all|select|robustness] [--resume runs/<dir>]
        [--first-oos-year 2012] [--keep sobol-035]

--first-oos-year moves the window: 2012 fits 2004-2009 and validates 2010-2011. Inside the
out-of-sample period that is a check of how stable the choice is, not a replacement for it.
--keep carries named candidates into the finalists (e.g. an earlier window's winner).

1. search: Sobol points over the knobs the paper leaves open, plus the base config, one seed.
2. finalists: the best `top` points and the base config on more seeds; the winner has the
   best MEAN validation net Sharpe. Writes selected.yaml (base config + the winner's values)
   for the out-of-sample run:
       scripts/run_years_parallel.sh --seeds "0 1 2 3 4" -- scripts/run_attention_us.py \
           --config runs/<this run>/selected.yaml
3. robustness: the winner with one Table 4 value moved at a time, on the same seeds.
   Written to robustness.csv, never adopted.

`runner: pca` in the search config searches the two-step benchmarks instead
(run_pca_longconv.py configs): the LongConv policy on fixed PCA residuals, or the OU
threshold rule (nothing trained). The residuals come from the stage one, built to the
sample end; each one uses only the returns before its date (pca_factors), and the search
reads dates before the first out-of-sample year only. `search.grid` ({key: [values]})
evaluates the full grid instead of Sobol points (the OU thresholds). `fixed` holds keys
constant for every candidate (e.g. execution.lag 1 or policy.input raw).
--run-dir writes to that directory instead of a timestamped one (scripts/run_matrix.py).
--smoke: two points, one finalist, one seed, one epoch, to check that a config runs.

A trial with seed s starts from the draw (s, first out-of-sample year), as in
run_attention_us_val.py, so candidates are compared on common random numbers. The scores
along the learning curve come from the same training run (train_window's on_epoch hook).
trials.jsonl keeps every trial; --resume <run dir> skips the ones already there.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
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
import run_pca_longconv as rp  # noqa: E402  (Stage1, train_one_window, execute)
from afe import runs  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.evaluation import metrics  # noqa: E402
from afe.policy import ou_threshold, trading  # noqa: E402
from afe.policy.longconv import LongConvPolicy  # noqa: E402

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


def grid_points(grid: dict) -> list[dict]:
    return [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]


def candidate(name: str, overrides: dict) -> dict:
    cid = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:10]
    return {"name": name, "cid": cid, "overrides": overrides}


def init_worker(cfg: dict, data_dir: str, first_oos: int, val_years: int, K: int,
                eval_epochs: list[int], threads: int, runner: str = "attention") -> None:
    torch.set_num_threads(threads)
    if runner == "pca":
        panel = rp.load_panel(cfg, str(cfg["sample"]["end"]))
        s1 = rp.Stage1(ROOT / cfg["factors"]["out_dir"], K, panel)
        t_tr0 = s1.dates.searchsorted(rp.window_start(cfg, first_oos)) if "training" in cfg else 0
        t = [s1.dates.searchsorted(pd.Timestamp(y, 1, 1)) for y in (first_oos - val_years, first_oos)]
        WORKER.update(runner="pca", panel=panel, s1=s1, cfg=cfg, K=K, first_oos=first_oos, split=(t_tr0, *t))
        return
    w0 = base.window_start(cfg, first_oos)
    if first_oos - val_years <= w0.year:
        raise SystemExit(f"a window from {w0:%Y-%m-%d} leaves no training years before {val_years} validation years")
    p = slots.load_slots(Path(data_dir), f"{w0:%Y-%m-%d}", f"{first_oos - 1}-12-31")
    t = [p.dates.searchsorted(d) for d in (w0, pd.Timestamp(first_oos - val_years, 1, 1))]
    if p.dates[0].year > w0.year:
        raise SystemExit(f"the panel starts {p.dates[0]:%Y-%m-%d}, too late for a window from {w0:%Y-%m-%d}")
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
            "sigma_pct": float(g["sigma_pct"]), "turnover": float(to.mean()), "ev": float(parts["ev"]),
            "beta": float(metrics.beta(net, p.mkt_ew[t_va0:t_va1].astype(float)))}


def evaluate_pca(job: dict) -> dict:
    """One candidate of a two-step benchmark: train on the fit years (LongConv) or apply the
    rule (OU), then score the executed book on the validation years."""
    c = with_overrides(WORKER["cfg"], job["overrides"])
    s1, panel, (t_tr0, t_va0, t_va1) = WORKER["s1"], WORKER["panel"], WORKER["split"]
    pc, ob = c["policy"], c["objective"]
    L, scale, cum = pc["residual_lookback"], pc["input_scale"], pc["input"] == "cumulative"
    norm = pc.get("input_normalise", "none")
    t0 = time.time()
    val_b = s1.batch(t_va0, t_va1, L, scale, cum, norm)
    if pc["kind"] == "longconv":
        init_seed = job["seed"] * 10007 + WORKER["first_oos"]
        torch.manual_seed(init_seed)
        model = LongConvPolicy(pc["hidden"], L, pc["dropout"], pc["lambda_squash"], pc["layers"])
        rp.train_one_window(model, s1.batch(t_tr0, t_va0, L, scale, cum, norm), c,
                            np.random.default_rng(init_seed), lambda s: None)
        w = rp.evaluate_window(model, val_b)
    else:
        w_port = ou_threshold.threshold_weights(val_b.windows.numpy(), val_b.tradable.numpy(),
                                                pc["c_thresh"], pc["c_crit"])
        w = trading.compose(torch.from_numpy(w_port), val_b)
    t_idx = np.arange(t_va0, t_va1)
    wp = rp.execute(trading.to_pool(w, val_b).numpy(), panel, t_idx, c).astype(float)
    gross = (wp * s1.R_pool[t_idx]).sum(axis=1)
    to = np.abs(np.diff(wp, axis=0, prepend=np.zeros((1, wp.shape[1])))).sum(axis=1)
    sh = np.clip(-wp, 0, None).sum(axis=1)
    net = gross - ob["turnover_cost"] * to - ob["short_cost"] * sh
    g = metrics.annualised(gross)
    r = {"net_SR": float(metrics.annualised(net)["SR"]), "gross_SR": float(g["SR"]),
         "sigma_pct": float(g["sigma_pct"]), "turnover": float(to.mean()), "ev": float("nan"),
         "beta": float(metrics.beta(net, s1.mkt_ew[t_idx].astype(float)))}
    return {**job, **r, "curve": {}, "seconds": round(time.time() - t0)}


def evaluate(job: dict) -> dict:
    if WORKER.get("runner") == "pca":
        return evaluate_pca(job)
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
    ap.add_argument("--resume", help="run directory of an earlier search: its configuration and trials are reused")
    ap.add_argument("--first-oos-year", type=int,
                    help="validate on the last years of the window before this year instead of the config's")
    ap.add_argument("--keep", nargs="*", default=[], help="candidate names always carried into the finalists")
    ap.add_argument("--run-dir", type=Path, help="write to this directory (must not exist) instead of runs/<stamp>_...")
    ap.add_argument("--smoke", action="store_true", help="2 points, 1 finalist, 1 seed, 1 epoch: does the config run")
    args = ap.parse_args()
    if args.resume:                                  # a resumed run keeps its own configuration
        man = json.loads((Path(args.resume) / "manifest.json").read_text())["config"]
        S, cfg = man["search"], man["base"]          # "base" already holds the fixed block
    else:
        S = yaml.safe_load(Path(args.config).read_text())
        base_path = Path(S["base_config"])
        cfg = yaml.safe_load((base_path if base_path.is_absolute() else ROOT / base_path).read_text())
        cfg = with_overrides(cfg, S.get("fixed") or {})   # held fixed for every candidate, part of the base
        if args.first_oos_year:
            S["first_oos_year"] = args.first_oos_year
        S["finalists"]["keep"] = args.keep
        if args.smoke:
            S["search"]["n_points"] = 2
            S["finalists"].update(top=1, seeds=S["finalists"]["seeds"][:1])
            S["robustness"] = {"seeds": [], "knobs": {}}
            if "training" in cfg:
                cfg["training"]["epochs"] = 1
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    data_dir = Path(cfg["data"]["dir"])
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    K, first_oos, sr, fin = S["K"], S["first_oos_year"], S["search"], S["finalists"]
    rob, runner = S.get("robustness") or {"seeds": [], "knobs": {}}, S.get("runner", "attention")
    for key in [*sr.get("space", {}), *sr.get("grid", {}), *rob["knobs"]]:
        with_overrides(cfg, {key: None})

    log = lambda s: print(s, flush=True)  # noqa: E731
    V = S["validation_years"]
    run_dir = Path(args.resume) if args.resume else runs.create_run(
        f"search_K{K}_val{first_oos - V}-{first_oos - 1}", {"search": S, "base": cfg}, sr["seed"], root=ROOT / "runs",
        run_dir=args.run_dir)
    path, trials = run_dir / "trials.jsonl", {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            trials[(r["cid"], r["seed"])] = r
    log(f"search run dir {run_dir}")
    fit = f"fit {base.window_start(cfg, first_oos).year}-{first_oos - V - 1}, " if "training" in cfg else ""
    log(f"{runner}: {fit}validate {first_oos - V}-{first_oos - 1}, K={K}, {args.jobs} jobs x {args.threads} threads")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.threads)
    summary_path = run_dir / "metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    with mp.get_context("spawn").Pool(args.jobs, initializer=init_worker, initargs=(
            cfg, str(data_dir), first_oos, V, K, S.get("eval_epochs", []), args.threads, runner)) as pool:
        run = lambda jobs, label: run_jobs(pool, jobs, trials, path, log, label)  # noqa: E731
        if args.stage in ("all", "select"):
            base_c = candidate("base", {})
            if "grid" in sr:
                pts = [candidate(f"grid-{i:03d}", o) for i, o in enumerate(grid_points(sr["grid"]))]
                pts = pts[:2] if args.smoke else pts
            else:
                pts = [candidate(f"sobol-{i:03d}", o)
                       for i, o in enumerate(sobol_points(sr["space"], sr["n_points"], sr["sobol_seed"]))]
            res = run([{**c, "seed": sr["seed"]} for c in [base_c, *pts]], "search")
            best = {r["cid"] for r in sorted(res, key=lambda r: -finite(r["net_SR"]))[:fin["top"]]}
            keep = set(fin.get("keep", []))
            if keep - {c["name"] for c in pts}:
                raise SystemExit(f"--keep: no candidates named {sorted(keep - {c['name'] for c in pts})}")
            finalists = [c for c in [base_c, *pts] if c["cid"] in best or c is base_c or c["name"] in keep]
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
        if args.stage in ("all", "robustness") and rob["knobs"]:
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
