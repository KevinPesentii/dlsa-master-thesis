"""Run the thesis matrix (configs/thesis_matrix.yaml) on one many-core machine.

    python scripts/run_matrix.py [--matrix configs/thesis_matrix.yaml] [--groups headline ablation ...]
                                 [--jobs 46] [--threads 2] [--mem-gb 340] [--dry-run] [--smoke]
                                 [--retry-failed] [--summary]

One job = one runner process for one (spec, K, seed): scripts/run_attention_us.py or
scripts/run_pca_longconv.py with --config <resolved config> --K <K> --seed <seed>. The
resolved config (base + `use` blocks + `set`, plus a `matrix` key naming the spec) is
written to runs/_matrix/<name>/configs/<spec>.yaml, so every run manifest holds exactly
what ran. Prep jobs (PCA stage one) run first and the specs that name them in `after`
wait; a prep job whose `done` file exists is skipped.

Searches (`searches:`, scripts/search_validation.py --stage select) run on the machine
too, into runs/_matrix/<name>/searches/<search>/. A spec with `hyper: <search>` waits for
it and runs with that search's winning values (its `fixed` block and the winner's
overrides, applied after `use` and before `set`); its config is resolved when its first job
starts and records the search, the candidate and the validation Sharpe under `matrix.hyper`.
A search counts as `workers` jobs against --jobs. A restarted search resumes its trials.

Scheduling: longest jobs first (a rough cost per job), at most --jobs at a time and at
most --mem-gb of estimated peak memory in flight. Each job's log is
runs/_matrix/<name>/logs/<job>.log; runs/_matrix/<name>/ledger.jsonl records every start
and finish, so a restart skips the jobs that finished (their run directory has a
metrics.json) and reruns the rest; --retry-failed also reruns failures. --summary (also
written at the end of a run) collects every finished job's metrics into summary.csv and
summary_by_spec.csv (mean and sd over seeds) next to the ledger.

--smoke runs each spec once (smallest K, seed 0, 1 epoch, the first out-of-sample year)
and each search on 2 points at 1 epoch, with its own ledger (<name>_smoke), and moves the
run directories to runs/_smoke/, to check that everything starts and writes its outputs. A real run refuses a dirty git tree: the
manifests must name a commit that exists.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe import runs  # noqa: E402

RUNNERS = {"attention": "scripts/run_attention_us.py", "pca": "scripts/run_pca_longconv.py"}
MEM_GB = {("attention", "us"): 5.5, ("attention", "europe"): 3.0, ("pca", "us"): 3.0, ("pca", "europe"): 2.0}


@dataclass
class Job:
    id: str
    cmd: list[str]
    mem_gb: float
    cost: float                      # rough minutes on one laptop core pair, for ordering only
    after: list[str] = field(default_factory=list)
    spec: dict | None = None
    K: int | None = None
    seed: int | None = None
    done_file: Path | None = None    # prep jobs and searches
    slots: int = 1                   # job slots it occupies (a search: its workers)
    make_cmd: Callable[[], list[str]] | None = None   # built at launch (resolved after a search)


def set_dotted(cfg: dict, key: str, value) -> None:
    *path, last = key.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def selected_values(search_dir: Path) -> tuple[dict, dict]:
    """(winner, values) of a finished search: the value of EVERY searched or fixed key in the
    selected config (base config + winner's overrides), so a winning "base" candidate carries
    the base config's values too; data / sample keys of a fixed block are never carried."""
    man = json.loads((search_dir / "manifest.json").read_text())["config"]
    S, base = man["search"], man["base"]
    w = json.loads((search_dir / "metrics.json").read_text())["winner"]
    keys = [*(S.get("fixed") or {}), *S["search"].get("space", {}), *S["search"].get("grid", {})]
    values = {}
    for k in keys:
        section, name = k.split(".")
        if section not in ("data", "sample"):
            values[k] = w["overrides"].get(k, base[section][name])
    return w, values


def winner(out: Path, search: str) -> dict:
    d = out / "searches" / search
    w, values = selected_values(d)
    return {"search": search, "run_dir": d.relative_to(ROOT).as_posix(), "candidate": w["name"],
            "validation_net_SR": w["val_net_SR"], "values": values}


def resolve(spec: dict, matrix: dict, out: Path | None = None) -> dict:
    """Base config + `use` blocks + the `hyper` search's values (needs `out`) + `set`."""
    cfg = yaml.safe_load((ROOT / spec["config"]).read_text())
    for b in spec.get("use", []):
        for k, v in matrix["blocks"][b].items():
            set_dotted(cfg, k, copy.deepcopy(v))
    hyper = winner(out, spec["hyper"]) if spec.get("hyper") and out is not None else None
    for k, v in (hyper or {}).get("values", {}).items():
        set_dotted(cfg, k, copy.deepcopy(v))
    for k, v in (spec.get("set") or {}).items():
        set_dotted(cfg, k, copy.deepcopy(v))
    cfg["matrix"] = {"name": matrix["name"], "spec": spec["name"], "group": spec["group"],
                     "base_config": spec["config"], "use": spec.get("use", []), "set": spec.get("set") or {},
                     "hyper": hyper}
    return cfg


def years_of(cfg: dict) -> list[int]:
    return list(range(pd.Timestamp(cfg["sample"]["start"]).year, pd.Timestamp(cfg["sample"]["end"]).year + 1))


def search_job(name: str, s: dict, out: Path, threads: int, smoke: bool) -> Job:
    d = out / "searches" / name
    workers = 2 if smoke else int(s.get("workers", 12))

    def make_cmd() -> list[str]:
        where = ["--resume", d.relative_to(ROOT).as_posix()] if (d / "manifest.json").exists() else \
                ["--run-dir", d.relative_to(ROOT).as_posix()]
        return [sys.executable, "-u", "scripts/search_validation.py", "--config", s["config"], "--stage", "select",
                "--jobs", str(workers), "--threads", str(threads)] + where + (["--smoke"] if smoke else [])
    after = [f"prep_{a}" for a in ([s["after"]] if isinstance(s.get("after"), str) else s.get("after", []))]
    return Job(f"search_{name}", [], workers * float(s.get("mem_gb_per_worker", 2.5)), 1e6 + 100, after,
               done_file=d / "selected.yaml", slots=workers, make_cmd=make_cmd)


def expand(matrix: dict, groups: list[str] | None, out: Path, threads: int, smoke: bool) -> list[Job]:
    sets = matrix["sets"]
    jobs = [Job(f"prep_{name}", [sys.executable, "-u", p["script"], "--config", p["config"]], p["mem_gb"],
                p["cost"] + 1e6, done_file=ROOT / p["done"]) for name, p in matrix["prep"].items()]
    jobs += [search_job(name, s, out, threads, smoke) for name, s in (matrix.get("searches") or {}).items()]
    (out / "configs").mkdir(parents=True, exist_ok=True)
    for spec in matrix["specs"]:
        if groups and spec["group"] not in groups:
            continue
        cfg = resolve(spec, matrix)          # without the search's values: sample, market, runner only
        path = out / "configs" / f"{spec['name']}.yaml"

        def write_config(spec=spec, path=path) -> str:
            path.write_text(yaml.safe_dump(resolve(spec, matrix, out), sort_keys=False))
            return path.relative_to(ROOT).as_posix()
        Ks = sets[spec["K"]] if isinstance(spec["K"], str) else spec["K"]
        seeds = sets[spec["seeds"]] if isinstance(spec["seeds"], str) else spec["seeds"]
        if smoke:
            Ks, seeds = [min(Ks)], [seeds[0]]
        years = years_of(cfg)
        market = "us" if cfg["market"] == "us" else "europe"
        trained = spec["runner"] == "attention" or cfg["policy"]["kind"] == "longconv"
        after = [f"prep_{a}" for a in ([spec["after"]] if isinstance(spec.get("after"), str) else spec.get("after", []))]
        after += [f"search_{spec['hyper']}"] if spec.get("hyper") else []
        for K in Ks:
            for seed in seeds:
                extra = ["--K", str(K), "--seed", str(seed), "--threads", str(threads)]
                if smoke:
                    extra += ["--years", str(years[0])] + (["--epochs", "1"] if trained else [])
                make_cmd = (lambda extra=extra, write_config=write_config, r=spec["runner"]:
                            [sys.executable, "-u", RUNNERS[r], "--config", write_config()] + extra)
                per_year = (2.0 if market == "us" else 1.0) * (1 + K / 60) if spec["runner"] == "attention" else \
                           ((1.0 if market == "us" else 0.5) if trained else 0.05)
                jobs.append(Job(f"{spec['name']}_K{K}_s{seed}", [RUNNERS[spec["runner"]], path.name] + extra,
                                MEM_GB[(spec["runner"], market)], per_year * len(years), after, spec, K, seed,
                                make_cmd=make_cmd))
    return jobs


def ledger_state(path: Path) -> dict[str, dict]:
    state: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            e = json.loads(line)
            state[e["job"]] = e
    return state


def finished(e: dict | None) -> bool:
    return bool(e and e.get("status") == "ok" and e.get("run_dir") and (ROOT / e["run_dir"] / "metrics.json").exists())


def run_dir_of(log: Path) -> str | None:
    m = re.findall(r"^K=\d+: run dir (.+)$", log.read_text(errors="replace"), flags=re.M)
    return m[-1].strip() if m else None


def summary(jobs: list[Job], state: dict, out: Path) -> None:
    rows = []
    for j in jobs:
        e = state.get(j.id)
        if j.spec is None or not finished(e):
            continue
        m = json.loads((ROOT / e["run_dir"] / "metrics.json").read_text())
        rows.append({"spec": j.spec["name"], "group": j.spec["group"], "K": j.K, "seed": j.seed,
                     "SR": m["gross"]["SR"], "mu_pct": m["gross"]["mu_pct"], "sigma_pct": m["gross"]["sigma_pct"],
                     "SR_net": m["net"]["SR"], "mu_net_pct": m["net"]["mu_pct"], "beta": m["beta_ew_market"],
                     "turnover": m["turnover_daily"], "short": m["short_exposure"],
                     "break_even_bps": m["break_even_turnover_cost_bps"], "first": m["first"], "last": m["last"],
                     "minutes": e.get("minutes"), "run_dir": e["run_dir"]})
    if not rows:
        print("no finished jobs yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    agg = df.groupby(["group", "spec", "K"], sort=False).agg(
        seeds=("seed", "size"), SR=("SR", "mean"), SR_sd=("SR", "std"), SR_net=("SR_net", "mean"),
        SR_net_sd=("SR_net", "std"), sigma_pct=("sigma_pct", "mean"), beta=("beta", "mean"),
        turnover=("turnover", "mean"), short=("short", "mean"), break_even_bps=("break_even_bps", "mean"))
    agg.round(3).to_csv(out / "summary_by_spec.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(agg.round(2).to_string())
    print(f"{len(df)} finished runs -> {out / 'summary.csv'}, {out / 'summary_by_spec.csv'}")
    picks = {d.name: winner(out, d.name) for d in sorted((out / "searches").glob("*")) if (d / "selected.yaml").exists()}
    if picks:
        (out / "searches.json").write_text(json.dumps(picks, indent=2))
        print(f"selected values of {len(picks)} searches -> {out / 'searches.json'}")


def total_mem_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (AttributeError, ValueError, OSError):
        return 16.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=ROOT / "configs" / "thesis_matrix.yaml")
    ap.add_argument("--groups", nargs="*", help="only these groups (default: all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mem-gb", type=float, help="estimated peak memory allowed in flight (default 85%% of RAM)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--summary", action="store_true", help="only write the summary of finished jobs")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    out = ROOT / "runs" / "_matrix" / (matrix["name"] + ("_smoke" if args.smoke else ""))
    jobs = expand(matrix, args.groups, out, args.threads, args.smoke)
    ledger = out / "ledger.jsonl"
    state = ledger_state(ledger)
    if args.summary:
        return summary(jobs, state, out)

    todo = [j for j in jobs if not (j.done_file and j.done_file.exists()) and not finished(state.get(j.id))
            and (args.retry_failed or state.get(j.id, {}).get("status") != "failed")]
    skipped = sum(1 for j in jobs if state.get(j.id, {}).get("status") == "failed") if not args.retry_failed else 0
    budget = args.mem_gb or 0.85 * total_mem_gb()
    print(f"{len(jobs)} jobs in the matrix, {len(todo)} to run ({skipped} failed, not retried); "
          f"{args.jobs} at a time x {args.threads} threads, memory budget {budget:.0f} GB; "
          f"total cost ~{sum(j.cost for j in todo if j.spec) / 60:.0f} laptop process-hours")
    if args.dry_run:
        for j in sorted(todo, key=lambda j: -j.cost):
            wait = f"  after {','.join(j.after)}" if j.after else ""
            print(f"  {j.id:42s} cost {min(j.cost, 9999):6.0f}  mem {j.mem_gb:5.1f}  slots {j.slots:2d}{wait}")
        return
    commit = runs.git_commit(ROOT)
    if commit.endswith("-dirty") and not (args.smoke or args.allow_dirty):
        raise SystemExit(f"git tree is dirty ({commit}); commit first so the manifests name a real commit")

    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
           "OPENBLAS_NUM_THREADS": str(args.threads), "PYTHONUNBUFFERED": "1"}
    pending = sorted(todo, key=lambda j: -j.cost)
    running: dict[str, tuple[Job, subprocess.Popen, float]] = {}
    done_ids = {j.id for j in jobs if (j.done_file and j.done_file.exists()) or finished(state.get(j.id))}
    failed_ids = {i for i, e in state.items() if e.get("status") == "failed" and not args.retry_failed}

    def record(e: dict) -> None:
        with ledger.open("a") as f:
            f.write(json.dumps({**e, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        state[e["job"]] = e

    t_start = time.time()
    while pending or running:
        for jid, (j, proc, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[jid]
            log = out / "logs" / f"{jid}.log"
            rd = run_dir_of(log) if j.spec else None
            ok = proc.returncode == 0 and (rd and (ROOT / rd / "metrics.json").exists() if j.spec else
                                           (j.done_file is None or j.done_file.exists()))
            if ok and args.smoke and rd:
                dst = ROOT / "runs" / "_smoke" / Path(rd).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ROOT / rd), dst)
                rd = str(dst.relative_to(ROOT))
            minutes = round((time.time() - t0) / 60, 1)
            record({"job": jid, "status": "ok" if ok else "failed", "returncode": proc.returncode,
                    "run_dir": rd, "minutes": minutes, "commit": commit})
            (done_ids if ok else failed_ids).add(jid)
            print(f"{time.strftime('%H:%M:%S')} {'done  ' if ok else 'FAILED'} {jid} ({minutes} min)"
                  f"  [{len(done_ids)} done, {len(failed_ids)} failed, {len(running)} running, {len(pending)} queued]",
                  flush=True)
        blocked = [j for j in pending if any(a in failed_ids for a in j.after)]
        for j in blocked:
            pending.remove(j)
            failed_ids.add(j.id)
            print(f"SKIP {j.id}: its prep job failed", flush=True)
        mem = sum(j.mem_gb for j, _, _ in running.values())
        used = sum(min(j.slots, args.jobs) for j, _, _ in running.values())
        for j in list(pending):
            if used >= args.jobs:
                break
            if any(a not in done_ids for a in j.after) or (running and (mem + j.mem_gb > budget or
                                                                       used + min(j.slots, args.jobs) > args.jobs)):
                continue
            cmd = j.make_cmd() if j.make_cmd else j.cmd
            log = (out / "logs" / f"{j.id}.log").open("w")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[j.id] = (j, proc, time.time())
            pending.remove(j)
            mem += j.mem_gb
            used += min(j.slots, args.jobs)
            record({"job": j.id, "status": "started", "cmd": cmd[1:], "commit": commit})
            print(f"{time.strftime('%H:%M:%S')} start  {j.id}", flush=True)
            time.sleep(1)                        # spread the panel loads
        time.sleep(5)
    print(f"matrix done in {(time.time() - t_start) / 3600:.1f} h: {len(done_ids)} ok, {len(failed_ids)} failed")
    summary(jobs, state, out)


if __name__ == "__main__":
    main()
